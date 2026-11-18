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

Video diffusion models, which generate videos by removing noise, have greatly improved text-to-video generation. Alignment tuning has helped drive these gains. This post-training process uses reward models trained to predict human preferences. Although these methods improve visual quality, they can also encode and amplify social biases.

We introduce VideoBiasEval, a diagnostic framework that tracks social representation throughout the alignment pipeline. The framework draws on established social bias taxonomies, or systems for grouping bias-related attributes. It uses event-based prompts to separate semantic content—verbs and contexts—from actor attributes—gender and ethnicity. Its multi-granular metrics measure bias at several levels: (1) overall ethnicity bias, (2) gender bias within each ethnicity, (3) changes in social attribute distributions across model variants, and (4) the persistence of bias over time within videos.

We use this framework for the first end-to-end analysis linking biases in human preference datasets to their amplification in reward models and their spread through alignment-tuned video diffusion models. Our results show that alignment tuning strengthens representational biases and makes them stable over time. The resulting portrayals are smoother but more stereotyped. These findings show the need to evaluate and mitigate bias throughout alignment to support fair and socially responsible video generation.

Reviewed on OpenReview: https://openreview.net/forum?id=C0yxuS6jty

**Figure 1:** Overview of our work. (1) We introduce VideoBiasEval, a framework that evaluates social bias in video generation. It uses event-based prompts and metrics at several levels to assess ethnicity and gender bias (bottom left, Section 3). The framework represents videos through social attribute annotations (top). Vision–language models (VLMs), which process images and text, infer actor attributes such as gender and ethnicity across frames. We aggregate these annotations to measure bias (Section 3.3). (2) We conduct the first comprehensive analysis of how image-based reward models trained on human-labeled preferences affect social attribute distributions in diffusion-generated videos. We separate human preference biases, reward preference biases, and their downstream effects on video diffusion model generation biases (bottom right, Section 5.1, Section 5.2, and Section 6).

## 1 Introduction

Video diffusion models have greatly improved the generation of high-quality videos from natural language prompts (Chen et al., 2024a; Wang et al., 2023a; Yuan et al., 2024; Li et al., 2024). These advances support applications in creative media, education, and professional simulation (Cho et al., 2024; Miller et al., 2024). Many leading video generation systems now use alignment tuning to improve visual fidelity, coherence, and control. The most common approach learns from human preferences (Wu et al., 2023; Xu et al., 2024; Li et al., 2024; Yuan et al., 2024; Liu et al., 2024a; Prabhudesai et al., 2024; Black et al., 2023; Ma et al., 2025). These methods usually rely on reward functions trained on human-labeled comparisons (Wu et al., 2023; Kirstain et al., 2023a; Xu et al., 2024). They often use judgments of individual frames or images to guide post-training optimization.

We use *aligned models* to mean video diffusion models that undergo post-training optimization to better match human preferences beyond likelihood-based training. Likelihood-based training fits a model to the probabilities of its training data. In practice, alignment commonly uses reward-weighted fine-tuning or procedures based on reinforcement learning from human feedback (RLHF). These methods use a learned preference or aesthetic reward model, trained on human comparisons, to guide further optimization of a base video diffusion model. The general pipeline is: base video diffusion model, then preference or reward model, then post-training optimization. We compare aligned models with *unaligned models*, which train only with likelihood-based objectives. These unaligned models provide controlled baselines for isolating alignment effects.

Alignment tuning improves perceived quality and temporal coherence, or consistency over time. Yet its reliance on subjective “preference” creates a major, understudied risk. Human preference judgments often do not explicitly account for cultural, social, or demographic diversity. They may therefore encode systematic biases that reward models and downstream generative systems inherit and reinforce (Qiu et al., 2023). Alignment can shape not only what a model generates well, but also who appears, how often, and in what roles. We investigate how alignment tuning changes social attribute distributions in video diffusion models.

This question requires a framework with three parts: a probing method that elicits social attributes, metrics that measure bias distributions, and an analysis protocol that tracks distributions before and after alignment. Existing frameworks (Huang et al., 2024; Liu et al., 2024b; Sun et al., 2024) have three limits:

1. Their prompts do not adequately represent diverse social identities. This restricts analysis of how models portray or misrepresent those attributes.
2. They lack broad identity coverage and specific metrics. This makes it hard to track how alignment affects social attribute distributions.
3. They lack a dedicated method for tracking social attribute distributions before and after alignment.

We address these limits with VideoBiasEval (Section 3), a comprehensive framework for analyzing social bias in video diffusion models. It uses event-based prompting and established social bias taxonomies (Zhao et al., 2017; Hendricks & Nematzadeh, 2021; Cho et al., 2023; Qiu et al., 2023). This design lets us control verb types and actor identity attributes separately. Separating social identity from semantic content supports robust, interpretable assessments across varied contexts. Following work on alignment and fairness in generative systems (Luccioni et al., 2023; Shen et al., 2023), we focus on gender and ethnicity. These two social dimensions have comparatively well-defined boundaries for evaluation. The framework also introduces metrics at several levels to assess (1) ethnicity bias, (2) gender bias within each ethnicity, (3) changes in social attribute distributions across models, and (4) the persistence of bias over time within videos. We use this framework to trace social attribute distributions through the alignment pipeline.

We first examine demographic preferences in two human preference datasets: HPDv2 (Wu et al., 2023) and Pick-a-Pic (Kirstain et al., 2023a) (Section 5.1). We then study how image reward models inherit these patterns. These models include HPSv2.0, HPSv2.1 (Wu et al., 2023), and PickScore (Kirstain et al., 2023a) (Section 5.2). Finally, we use different image reward models to fine-tune a video consistency model distilled from VideoCrafter-2 (Chen et al., 2023) (Section 6). Distillation trains a model to reproduce the behavior of another model. This experiment lets us compare video outputs before and after alignment in detail.

Both human preference datasets show non-neutral gender preferences and a strong imbalance favoring White representations. Reward models trained on these datasets inherit and amplify these biases. Alignment-tuned video diffusion models then inherit them. Reward models that prefer men or women systematically shift gender portrayals during alignment, while ethnic representation remains uneven. Our Temporal Attribute Stability (TAS) analysis also shows that alignment improves consistency in social attribute portrayals over time. However, it can make biased representations more persistent and visually stable. Thus, alignment improves video quality and coherence while reinforcing or stabilizing existing social biases. These findings show the need to evaluate bias and design bias-aware alignment strategies throughout the generation pipeline to support equitable and socially responsible video generation.

We also test whether we can deliberately construct controllable image reward datasets by changing their social attribute distributions (Section 7). We assess whether reward models trained on these curated datasets let video diffusion models produce outputs with controllable bias representations. This approach offers a possible path toward more equitable generative systems. We also analyze changes in reward model preferences across 42 events and the resulting changes in video model bias before and after alignment. Based on these findings, we offer guidance on using targeted data composition and counter-biased reward models to reduce representational disparities in future video generation systems.

We make three main contributions:

1. We introduce VideoBiasEval, a comprehensive framework for evaluating social bias in video generation. It uses event-based prompts and metrics at several levels to assess ethnicity bias, gender bias within each ethnicity, and the temporal stability of social attributes across videos.
2. We present the first end-to-end analysis linking human preference datasets, reward models, and alignment-tuned video diffusion models. It shows how the alignment pipeline inherits, amplifies, and stabilizes social biases.
3. Through systematic experiments, we show that preference alignment improves perceived quality and temporal coherence but also changes—and sometimes reinforces—the social composition of generated content. We offer guidance on controllable preference modeling. Targeted data composition and counter-biased reward design can steer video diffusion models toward more equitable generation.

## 2 Related Work

**T2V Evaluation.** Recent text-to-video (T2V) benchmarks include VBench (Huang et al., 2024), EvalCrafter (Liu et al., 2024b), and T2V-CompBench (Sun et al., 2024). They use measures such as Fréchet Video Distance, which compares video distributions (Unterthiner et al., 2019), CLIP-Score, which measures image–text agreement (Hessel et al., 2021), and object consistency. Yet they overlook who appears and how models portray identities. GRiT-based metrics, which use image descriptions (Wu et al., 2025), may confirm that a “doctor” appears but fail to flag that all doctors are white men. CLIP-based alignment rewards agreement with text but ignores demographic balance. Fair and trustworthy T2V evaluation must go beyond surface-level metrics and explicitly audit social attribute distributions. Our event-centered framework meets this need by measuring gender- and ethnicity-aware biases throughout the T2V generation pipeline.

**Bias Evaluation in Generative Models.** Most studies of social bias in text-to-image or language generation focus on static, single-frame outputs, such as portraits or isolated object scenes. StableBias (Luccioni et al., 2023), DALL-Eval (Cho et al., 2023), and SocialCounterfactuals (Howard et al., 2024) mainly count identities. They rarely examine what those identities do. Even recent demographic representation benchmarks often assess each image independently. This can hide recurring patterns, such as portraying men in authoritative roles and women in supportive roles. Without jointly analyzing actors, verbs, and context, these methods miss role-specific stereotypes. They also cannot reveal bias in narratives or over time. We address this limit by auditing events. We separate actor attributes from verbs and environments to reveal how social representation changes across scenarios.

## 3 VideoBiasEval

We introduce VideoBiasEval, a comprehensive framework for evaluating social bias in video generation models. It uses event-based prompts that systematically vary characters’ gender and ethnicity across diverse real-world events (Section 3.1). We use these structured prompts to generate videos with leading diffusion models (Section 3.2). We then extract social attribute representations and evaluate them at several levels across event categories to measure consistency and fairness in identity portrayals (Section 3.3). Figure 2 shows sample prompts and how we capture and analyze social attributes in generated outputs.

**Figure 2:** Videos generated by different diffusion models with prompt templates that specify actor attributes, as detailed in Section 3.2. Our proposed VLM-based method extracts the main character’s social attributes, including gender and ethnicity, as described in Section 3.3.

### 3.1 Event Definition

We study social bias in how video generation models portray events. We focus on how models visually represent different actors as they perform verbs. Bias often appears as unequal portrayals across gender or ethnic groups. These patterns can reinforce stereotypes and undermine fairness, as prior work has documented (Bolukbasi et al., 2016; Sun & Peng, 2021; Zajko, 2021).

We represent each event as a tuple, or ordered group, $\langle p,a,c\rangle$. Here, actor $p$ performs verb $a$ in context $c$. Following prior studies, we focus on socially associated verbs: verbs historically linked to identity-related stereotypes (Zhao et al., 2017; Garg et al., 2018; Cho et al., 2023; Qiu et al., 2023). This representation lets us assess how demographic attributes affect visual depictions and measure systematic biases in generated videos.

**Controlled Attribute Space.** We deliberately balance coverage with experimental control. We combine four gender categories and seven ethnic groups with 42 verbs. These choices follow established social bias taxonomies and prior benchmark conventions. This bounded setup supports the first end-to-end tracing of bias propagation in video generation. It allows clear attribution and interpretation while preserving reproducibility. Open-ended or intersectional attributes—combinations of identity dimensions—offer a promising next step. However, a clearly defined, theoretically grounded scope is crucial for isolating representational disparities before moving to unconstrained settings.

**Actors.** We describe each actor, $p$, with gender and ethnicity attributes to support structured bias analysis. For gender, we use the four categories from Luccioni et al. (2023): man, woman, the neutral term “person,” and non-binary person. Although inclusive, these categories cannot capture all gender identities. They provide clear boundaries for controlled evaluation.

For ethnicity, we use seven groups: White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, and Latino. These follow Karkkainen & Joo (2021) and U.S. Census Bureau conventions. The categories aim to be inclusive, but they are socially constructed. They are neither exhaustive nor universally representative.

**Verbs.** We examine 42 verbs, denoted by $a$: bake, bike, call, clean, climb, cook, cough, cry, drink, drive, eat, exercise, fish, hit, jump, kick, kneel, laugh, lift, paint, pick, pitch, pray, read, ride, row, run, shop, shout, sit, skate, sleep, smile, stand, stare, stretch, study, sweep, throw, walk, wash, work. Prior research has identified statistical links between these verbs and particular genders or ethnic groups (Zhao et al., 2017; Garg et al., 2018; Cho et al., 2023; Qiu et al., 2023).

We selected these verbs for their documented social associations, clear visual differences, and ease of depiction in short clips. These properties support reliable annotation and interpretable outputs. This curated, socially meaningful set provides a reproducible basis for future studies of more complex, culturally specific, or temporally changing events.

### 3.2 Event-Based Prompting Protocol

| Settings | # of Prompts | Examples |
|---|---:|---|
| Person Only | 42 | A person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |
| Ethnicity + Person | 294 | An East Asian person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |

**Table 1:** Statistics of social bias evaluation prompts for video generation. Each prompt explicitly states the actor’s ethnicity when specified, the verb, and the surrounding context. This structure supports analysis of social representations in generated videos.

We use a fully deterministic event template to create varied but systematically comparable prompts: “A/An [actor] is [verb]-ing [context].” A deterministic template produces prompts from fixed rules rather than random generation. Neither the actor nor the context comes from a large language model (LLM) at runtime. We instead build every prompt from a fixed, predefined set to support reproducibility and controlled comparisons. The [verb] slot covers the 42 curated activities. The [context] slot adds situational variety without changing the verb’s meaning.

We define two conditions to separate demographic effects:

1. **Person-only:** The [actor] is always “person.” This gives one prompt per verb, or 42 prompts.
2. **Ethnicity+Person:** We deterministically place a predefined ethnicity descriptor before “person,” as in “An East Asian person.” Combining the same 42 verbs with seven ethnic groups gives 294 prompts.

Table 1 summarizes these prompts and gives examples. Because ethnicity+person already contains ethnic information, a separate ethnicity-only setting is unnecessary for isolating ethnicity effects.

We query each target video model directly with each prompt. We generate a fixed number of videos per prompt under identical decoding settings, which control how the model produces outputs. We aim to assess how models represent actors when a prompt leaves demographic attributes unspecified under a fixed verb. In the person-only condition, we examine which gender and ethnicity attributes models assign without explicit demographic cues. In the ethnicity+person condition, we examine how explicit ethnic descriptors interact with inferred attributes such as gender.

Importantly, we do not introduce, constrain, or infer demographic attributes during generation. We conduct all demographic analysis afterward. An ensemble, or combined group, of VLMs infers actor attributes from generated frames (Section 3.3). This controlled prompting framework lets us reliably attribute observed disparities to demographic conditioning rather than uncontrolled changes in context. That property is essential for reproducible bias auditing in generative video models.

### 3.3 Multi-Granular Event-Centric Bias Evaluation

Our evaluation captures detailed frame-level changes and aggregated video-level fairness. It assesses how demographic portrayals emerge and persist over time. By examining several temporal and representational levels, we identify who appears and how consistently videos maintain those identities.

**Social Attribute Representations.** We use three open-source VLMs as automated judges: Qwen2-VL-7B (Wang et al., 2024a), Qwen2.5-VL-7B (Yang et al., 2024), and InternVL2.5-8B (Chen et al., 2024b). Each classifies social attributes frame by frame. We sample 16 frames at evenly spaced points in each video. This count matches the standard inference defaults of the evaluated models, such as VideoCrafter. Our preliminary experiments showed that fewer frames substantially reduced attribute detection stability.

We ask each VLM to infer the depicted gender and ethnicity independently in each frame. Each model outputs a gender label from the set $G$ and an ethnicity label from the set $E$:

$$
g\in G=\{\text{man, woman}\},
$$

$$
e\in E=\{\text{White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, Latino}\}.
$$

Within each model, we first use majority voting across frames to assign video-level labels. We then combine labels across models through an ensemble strategy. We use this “VLMs-as-judges” ensemble because initial manual checks found that individual models often gave unstable predictions or hallucinated attributes—that is, reported attributes not supported by the image. Combining distinct architectures, rather than repeatedly sampling one model, lets us cross-check results. This improves robustness and reduces model-specific biases (Qiu et al., 2024b).

Figure 2 shows sample classification prompts and video outputs. The frame- and video-level representations let us assess whether videos match the demographic attributes in their prompts. The top of Figure 1 illustrates this process. We also analyze correlations between human and model judgments to assess reliability and fairness (Section 4).

**Temporal Attribute Stability.** We introduce Temporal Attribute Stability (TAS) to measure how stable social attribute representations remain within a video. TAS is the percentage of frames whose classified attributes match the video’s final majority-voted label. A high TAS indicates a stable, temporally coherent demographic portrayal. A low TAS indicates attribute “flickering,” or inconsistent identity depiction across frames. This flickering is an important temporal artifact in biased or unstable generation.

We also assess social balance across videos, beyond frame-level and temporal consistency. The following video-level metrics measure gender and ethnicity fairness across events and demographic groups.

**Ethnicity-Aware Gender Bias.** We use the Proportion Bias Score for Gender (PBSG) to assess gender portrayals within ethnic groups.[^1] For each verb and ethnicity, we define:

$$
\text{PBS}_{G}
=\frac{N_{\text{man}}-N_{\text{woman}}}{N_{\text{total}}}
\in[-1,1].
$$

Here, $N_{\text{man}}$ and $N_{\text{woman}}$ count representations of men and women. $N_{\text{total}}$ is their sum. A PBSG of $0.5$ means male representations appear three times as often as female representations: $N_{\text{man}}=3N_{\text{woman}}$. A value of $-0.5$ means the reverse. Positive values indicate male bias, negative values indicate female bias, and values near zero indicate balanced representation.

[^1]: We exclude gender bias from this analysis because, without explicit ethnicity specifications, generative models mainly produce White individuals (Figure 14). As a result, analyzing gender alone effectively measures gender bias within the White demographic (Figure 30).

We do not treat $\text{PBS}_{G}\approx 0$ as a claim about real-world gender distributions. Those distributions may legitimately differ across verbs, ethnicities, and sociocultural contexts. We use zero as a diagnostic benchmark for controlled bias auditing, not as factual or normative ground truth. Our neutral “person”-based prompts explicitly separate verbs from identity attributes. Deviations from $\text{PBS}_{G}\approx 0$ therefore measure how strongly models internalize and reproduce gendered stereotypes linked to specific verbs.

This approach follows established fairness and representational harm analysis. Such work uses uniform or parity-based baselines to expose systematic skew in generative models, not to claim that those baselines are empirically correct. Computing PBSG under the ethnicity+gender condition reveals how gender portrayals vary within each ethnic group. It shows whether gender balance holds across groups or whether disparities grow more strongly for some ethnicities. Thus, PBSG probes stereotype amplification across ethnicity-conditioned verbs. It does not assert that identical gender proportions are always appropriate.

**Ethnicity Bias.** We use two complementary metrics: the Representation Deviation Score for Ethnicity, $\text{RDS}_{e}$ (Feldman et al., 2015; Mehrabi et al., 2021), and Simpson’s Diversity Index (SDI) (Simpson, 1949). For each ethnic group $e$ in the set $E$, its representation proportion is:

$$
P_e=\frac{N_e}{N_{\text{total}}}.
$$

Here, $N_e$ counts outputs identified as ethnicity $e$. $N_{\text{total}}$ counts all outputs with an identifiable ethnicity.

The first metric measures each group’s deviation from an ideal uniform distribution:

$$
\text{RDS}_{e}=P_e-\frac{1}{|E|}.
$$

The notation $|E|$ means the number of ethnic groups. The reference $1/|E|$ gives equal coverage to every group. With seven groups, that probability is $1/7\approx 14.3\%$. An $\text{RDS}_{e}$ of 0.769 means a group appears in $0.769+0.143\approx 91.2\%$ of generated videos—more than six times the expected frequency. Positive values indicate overrepresentation; negative values indicate underrepresentation.

This uniform reference does not represent real-world demographic proportions. Those proportions may legitimately differ by region, cultural context, or application. We instead use it as a diagnostic benchmark under the neutral person-only condition. These prompts deliberately remove contextual and identity cues. Deviations from uniformity therefore reveal how much models internally favor or suppress particular ethnic identities, rather than reflect external demographic statistics. This use of parity-based baselines follows prior fairness and representational harm analysis. It exposes systematic skew in learned representations rather than asserting empirical correctness.

SDI measures overall representational diversity:

$$
\text{SDI}=1-\sum_{e\in E}P_e^2.
$$

It captures the probability that two randomly selected outputs belong to different ethnic groups. Higher values indicate more diverse and balanced distributions. Lower values indicate concentration in a few dominant groups. Together, these metrics show who is over- or underrepresented and how balanced the full distribution is. They reveal whether models fairly portray global ethnic diversity or repeat skewed, homogeneous visual patterns.

**Bias Shift.** We compare unaligned and aligned models to assess how alignment affects video-level social fairness. For each metric, we compute its change, denoted by $\Delta$: $\Delta$ PBSG, $\Delta$ RDSe, and $\Delta$ SDI. Changes toward balanced gender ratios, less ethnic skew, or greater diversity indicate improvement. This evaluation suite combines frame dynamics, temporal stability, and video-level demographic fairness to assess bias formation and mitigation in diffusion-based video generation.

## 4 Social Biases in Video Generative Models

We apply VideoBiasEval to four leading video diffusion models with different alignment strategies. As defined above, aligned models undergo post-training optimization to better match human preferences. They typically use reward-weighted fine-tuning or RLHF-style procedures with learned preference or aesthetic reward models. The general pipeline is: base video diffusion model, then preference/reward model, then post-training optimization.

The aligned models include InstructVideo (Yuan et al., 2024), built on ModelScope (Wang et al., 2023a) and aligned with HPSv2.0. They also include T2V-Turbo-V1 (Li et al., 2024), built on VideoCrafter-2 (Chen et al., 2024a) and aligned with HPSv2.1, InternVid2-S2 (Wang et al., 2024b), and ViCLIP (Wang et al., 2023b). Their unaligned counterparts, ModelScope and VideoCrafter-2, provide controlled baselines. These comparisons let us separate alignment effects from base architecture effects.

Following Section 3, we generate videos 10 times for each prompt and model, using different random seeds. A random seed sets the initial random state of generation. We average the results to reduce sampling variance. Table 2 reports ethnicity-aware gender bias through PBSG and ethnic representation through RDSe and SDI. Appendix A provides further analysis across 42 verbs, analysis of the training data WebVid-10M (Bain et al., 2021), and statistical analysis (Table 9).

| Models | Average PBSG | White PBSG | White RDS | Black PBSG | Black RDS | Latino PBSG | Latino RDS | East Asian PBSG | East Asian RDS | Southeast Asian PBSG | Southeast Asian RDS | India PBSG | India RDS | Middle Eastern PBSG | Middle Eastern RDS | Overall SDI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ModelScope (u) | 0.4815 | 0.5683 | 0.7690 | 0.3912 | -0.1952 | 0.6308 | -0.1952 | 0.4406 | -0.1810 | 0.4611 | — | 0.3938 | — | 0.4833 | -0.1976 | 0.0538 |
| InstructVideo (a) | 0.5295 | 0.5584 | 0.7833 | 0.5114 | -0.1976 | 0.6729 | -0.1929 | 0.4282 | -0.1976 | 0.5020 | — | 0.4878 | — | 0.5393 | -0.1952 | 0.0267 |
| $\Delta$ | +0.0480 | -0.0099 | +0.0143 | +0.1202 | -0.0024 | +0.0421 | +0.0023 | -0.0124 | -0.0166 | +0.0409 | — | +0.0940 | — | +0.0560 | +0.0024 | -0.0271 |
| Video-Crafter-V2 (u) | 0.7581 | 0.7485 | 0.6905 | 0.6167 | -0.1905 | 0.8599 | -0.1952 | 0.6976 | -0.1500 | 0.8272 | — | 0.8032 | — | 0.7560 | -0.1548 | 0.1252 |
| T2V-Turbo-V1 (a) | 0.8306 | 0.8713 | 0.6381 | 0.8095 | — | 0.8599 | -0.2476 | 0.7762 | -0.2426 | 0.8929 | — | 0.7762 | — | 0.7664 | -0.1476 | 0.1119 |
| $\Delta$ | +0.0725 | +0.1228 | -0.0524 | +0.1928 | — | 0.0000 | -0.0524 | +0.0786 | -0.0926 | +0.0657 | — | -0.0270 | — | +0.0104 | +0.0072 | -0.0133 |

**Table 2:** Social attribute distributions in two pairs of unaligned (u) and aligned (a) video diffusion models. Each entry averages scores across 42 verbs. For PBSG, the sign gives the model’s absolute direction of gender bias: positive values indicate man-preference, negative values indicate woman-preference, and values closer to zero indicate less gender skew. Non-zero PBSG values in the unaligned baselines, ModelScope and VideoCrafter-V2, therefore show inherent pre-alignment bias, not bias-free behavior. For RDSe, positive values indicate overrepresentation and negative values indicate underrepresentation of an ethnicity. These signs also describe absolute scores, not post-alignment changes. SDI measures overall ethnic balance; higher values indicate greater diversity and more uniform representation. A dash (“—”) marks cases with zero recognizable instances of an ethnicity under person-only prompts, where ethnicity is unspecified. This severe lack of diversity prevents metric computation for those subgroups. The $\Delta$ rows show alignment-induced changes from the corresponding unaligned model. For $\Delta$ PBSG, (+) marks a shift toward man-preference and (–) a shift toward woman-preference. For $\Delta$ RDSe, (+) and (–) mark shifts toward overrepresentation and underrepresentation, respectively.

**Ethnicity-Aware Gender Bias.** We use PBSG to evaluate gender portrayals under ethnicity+person prompts. All ethnic groups show consistent male bias, with average PBSG values above zero. Alignment appears to amplify this imbalance. InstructVideo and T2V-Turbo-V1 increase PBSG by 0.04 and 0.0725, respectively. These results suggest that preference-based fine-tuning may worsen gender disparity rather than reduce it. Figures 7, 8, 12, 10, 9, 11 and 13 report detailed PBSG scores across 42 verbs for each ethnicity.

**Ethnicity Bias.** Under person-only prompts, we assess representation balance with RDSe and SDI. Every model strongly overrepresents White individuals, although the degree varies. ModelScope shows the strongest imbalance: RDSWhite = 0.769 and SDI = 0.0538. Alignment further amplifies this imbalance in InstructVideo, with RDSWhite = 0.783 and SDI = 0.0267.

VideoCrafter-2 achieves moderately better balance, with RDSWhite = 0.6905 and SDI = 0.1252. T2V-Turbo-V1 further reduces White dominance to RDSWhite = 0.6381, but lowers diversity to SDI = 0.1119. Thus, alignment may reduce some ethnic skews while suppressing demographic diversity. This suggests a trade-off between bias reduction and representational variety. Figure 14 shows ethnicity bias across 42 verbs.

**Human Evaluation.** We check the reliability of our VLM evaluators through human verification of 400 generated videos. Three independent annotators label gender and ethnicity. All three are researchers with prior experience in social bias analysis. They follow standardized written guidelines and receive task-specific training based on the gender and ethnicity categories used in prior bias evaluation work. Their task only asks whether the generated content correctly reflects the prompted demographic attribute. It does not ask for subjective or normative judgments.

Our three-model VLM ensemble uses majority voting and strongly agrees with human judgments. Its Pearson correlations, which measure association between judgments, reach 0.89 for gender and 0.73 for ethnicity. The ethnicity result shows some remaining disagreement. Further analysis finds that many differences arise from human holistic judgment versus the evaluator’s frame-level precision. Humans sometimes miss or infer fleeting but visually valid attributes, such as faces visible for only a few frames in fast-motion clips. The VLM pipeline explicitly captures these attributes. This indicates higher recall, or detection of more valid instances, rather than systematic false positives.

Agreement measures also show high consistency. Across all 400 videos, Cohen’s Kappa reaches 0.91 for gender and 0.78 for ethnicity. Fleiss’ Kappa reaches 0.92 and 0.82, respectively. These Kappa measures assess agreement beyond chance and indicate strong to near-perfect inter-annotator agreement.

We also test whether evaluator disagreement changes our conclusions. We restrict analysis to a high-agreement subset where human majority votes exactly match VLM predictions. Consensus rates are high: 99.0% for gender and 83.3% for ethnicity. All core bias metrics remain statistically stable on this subset. For example, $\text{PBS}_{G}$ changes from 0.846 to 0.863, with $p=0.68$, while SDI changes from 0.745 to 0.761. The $p$ value expresses the statistical test’s evidence against its null hypothesis. This non-significant difference confirms that remaining evaluator uncertainty does not distort the reported bias measurements.

Human verification grounds and checks the automated pipeline; it does not replace large-scale evaluation. Larger human annotation efforts would be ideal, but they require substantial labor. Our approach follows prior work that validates model-based judges through small-scale human review with high consensus. Together, these results show that the VLM ensemble offers reliable, scalable evaluation aligned with human judgments for large-scale social attribute analysis.

These findings lead to our main question: how does alignment tuning shape social attribute distributions in video generative models? We address it by (1) analyzing demographic distributions in image reward datasets (Section 5.1), (2) examining biases in trained reward models (Section 5.2), and (3) assessing how these reward models affect gender and ethnicity in videos when used for alignment (Section 6).

## 5 Social Biases in Image Reward Datasets and Reward Models

Image-based reward models have become the standard, state-of-the-art approach to aligning video diffusion models (Wu et al., 2023; Xu et al., 2024; Li et al., 2024; Yuan et al., 2024; Liu et al., 2024a; Prabhudesai et al., 2024; Black et al., 2023; Ma et al., 2025). These reward models train on static images but guide learning for temporally coherent videos. Their social biases can therefore spread and even grow across frames. Tracing bias from reward datasets to trained reward models is crucial. It can reveal both demographic disparities in human-labeled image preferences and their effects on social attributes over time in generated videos.

### 5.1 Preference in Image Reward Datasets

We study human preference biases in two widely used image reward datasets: HPDv2 (Wu et al., 2023) and Pick-a-Pic (Kirstain et al., 2023b). For each dataset, GPT-4o-mini extracts gender, ethnicity, and verb attributes from captions. Three VLMs—Qwen2-VL-7B, Qwen2.5-VL-7B, and InternVL2.5-8B—classify attributes from images. We combine attributes from the caption and image modalities, or types of data, and keep only instances containing one of our predefined verbs.

After processing, HPDv2 contains 28,783 validated (images, caption, preference) tuples covering 29 verbs. Pick-a-Pic contains 14,958 tuples across 19 verbs. Each tuple presents two images, and a human annotator chooses the image that best matches the caption. We measure how often annotators prefer particular gender or ethnicity representations for each verb.

**Figure 3:** Gender preference distributions in image reward datasets across 42 verbs.

**Figure 4:** Ethnicity preferences in image reward datasets across 42 verbs.

We directly analyze pairs in which the preferred image depicts one gender, such as a man, and the dispreferred image depicts the other, such as a woman. These pairs reveal explicit gender preference patterns within comparisons. Figure 3 shows gender preference bias across 42 verbs in both datasets.

After filtering for valid man–woman pairs, 26 out of 42 verbs in HPDv2 meet our criteria. Of these, 69.23% (18/26) favor men, showing a consistent man-preferred tendency. In Pick-a-Pic, 18 out of 42 verbs qualify. Of these, 61.11% (11/18) favor women, showing a relatively woman-preferred trend.

Table 10 and Figure 4 report ethnicity preference distributions. Both datasets strongly prefer the White group: 43.34% in HPDv2 and 40.08% in Pick-a-Pic. East Asian and Indian representations follow. Some verbs show distinct preferences, such as “bake” favoring Black individuals and “fish” favoring East Asians. Yet the overall distributions suggest that collected human preferences may implicitly favor Western-centric aesthetics and representation. These imbalances might spread representational bias during reward model training and reinforce existing social inequities in downstream video generation. The findings show an urgent need for more inclusive, demographically representative preference datasets that capture global diversity.

### 5.2 Preference in Image Reward Models

We next examine how reward models trained on human preference datasets inherit the gender and ethnicity biases identified above.

**Evaluation Benchmark Construction.** We build a controlled benchmark using text-to-image (T2I) generation to evaluate reward model bias. HPDv2 (Wu et al., 2023) and ImageRewardDB (Xu et al., 2024) inspire this design. We use the event prompt templates from Section 3.2 with FLUX (Labs, 2023), a leading T2I model. This produces diverse image sets that vary systematically by gender, ethnicity, and verb.

The benchmark has two generation settings: (1) Ethnicity+Person, which specifies only the actor’s ethnicity, and (2) Ethnicity+Gender, which explicitly specifies both gender and ethnicity. Table 3 summarizes prompt coverage and gives examples. We generate 100 images per prompt to support statistical robustness and create a large, diverse evaluation set. Figure 5 shows sample outputs.

| Settings | # of Prompts | Examples |
|---|---:|---|
| Ethnicity + Person | 294 | An East Asian person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |
| Ethnicity + Gender | 1176 | An East Asian woman is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |

**Table 3:** Statistics of Benchmark Construction Prompts for Image Reward Models. Each prompt explicitly states the actor’s ethnicity when specified, the verbs, and the surrounding context. This structure supports analysis of social representations in image reward models.

We do not assume FLUX is bias-free. We use it to create a controlled, reproducible setting for probing reward model preferences. By systematically varying demographic attributes in prompts, the benchmark isolates reward model preferences independently of generator-specific artifacts.

To check benchmark integrity, three annotators independently review 100 randomly sampled images. They unanimously agree on 77% of samples. This figure is deliberately conservative: closer inspection shows majority agreement on 97% of samples, with at least 2/3 annotators confirming correct attribute representation. Only three images lack consensus. These rare disagreements involve visually ambiguous prompts, not systematic failures or demographic-specific distortions. This high signal-to-noise ratio indicates that reward models can reliably process the images’ gender and ethnicity cues. Remaining label noise is unlikely to materially affect the observed preferences.

**Figure 5:** Example images from our benchmark for evaluating image reward model preferences. Generation prompts follow “A/An [ethnicity][gender] is baking [context].” We show only images where [gender] belongs to {man, woman}.

**Preference Bias Evaluation.** We evaluate four image reward models:

1. HPSv2.0 (Wu et al., 2023), trained on HPDv2.
2. HPSv2.1 (Wu et al., 2023), trained on the unreleased HPDv2.1 dataset.
3. PickScore (Kirstain et al., 2023b), trained on Pick-a-Pic.
4. CLIP (Radford et al., 2021), the shared image–text base model for HPSv2.0, HPSv2.1, and PickScore before preference fine-tuning.

Each reward model gives each generated sample a continuous scalar preference score—a single numerical value, such as an HPSv2 score—not a discrete class label. We standardize these scores to measure relative preferences for gender and ethnicity attributes under controlled comparisons. Table 4 reports ethnicity-aware gender bias through PBSG and ethnic representation through RDSe and SDI. Appendix C provides implementation details, analysis across 42 verbs, and statistical analysis (Table 11).

| Models | Average PBSG | White PBSG | White RDS | Black PBSG | Black RDS | Latino PBSG | Latino RDS | East Asian PBSG | East Asian RDS | Southeast Asian PBSG | Southeast Asian RDS | India PBSG | India RDS | Middle Eastern PBSG | Middle Eastern RDS | Overall SDI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP | -0.0726 | 0.0343 | 0.0182 | -0.1198 | 0.0002 | -0.0934 | -0.0013 | -0.1315 | 0.0141 | -0.0865 | 0.0094 | -0.0508 | -0.0299 | -0.0607 | -0.0108 | 0.8495 |
| HPSv2.0 | 0.6039 | 0.6090 | -0.0423 | 0.7341 | -0.0069 | 0.6512 | 0.0237 | 0.4752 | -0.0031 | 0.5192 | -0.0100 | 0.5922 | 0.0070 | 0.6464 | 0.0315 | 0.8492 |
| $\Delta$ | +0.6765 | +0.5747 | -0.0605 | +0.8539 | -0.0071 | +0.7446 | +0.0250 | +0.6067 | -0.0172 | +0.6057 | -0.0194 | +0.6430 | +0.0369 | +0.7071 | +0.0423 | -0.0003 |
| HPSv2.1 | -0.0984 | -0.0833 | -0.0189 | 0.0257 | -0.0321 | -0.0031 | 0.0382 | -0.3044 | 0.0091 | -0.2181 | -0.0099 | -0.0006 | -0.0077 | -0.1053 | 0.0214 | 0.8470 |
| $\Delta$ | -0.0258 | -0.1176 | -0.0371 | +0.1455 | -0.0323 | +0.0903 | +0.0395 | -0.1729 | -0.0050 | -0.1316 | -0.0193 | +0.0502 | +0.0222 | -0.0446 | +0.0322 | -0.0025 |
| PickScore | -0.1157 | 0.0321 | 0.0069 | -0.0777 | 0.0279 | -0.3479 | -0.0118 | -0.2257 | 0.0316 | -0.2163 | 0.0115 | 0.1531 | -0.0391 | -0.1277 | -0.0271 | 0.8483 |
| $\Delta$ | -0.0431 | -0.0022 | -0.0113 | +0.0421 | +0.0277 | -0.2545 | -0.0105 | -0.0942 | +0.0175 | -0.1298 | +0.0021 | +0.2039 | -0.0092 | -0.0670 | -0.0163 | -0.0012 |

**Table 4:** Preference bias of different reward models. All values average scores across 42 verbs.

**Ethnicity-Aware Gender Bias.** Evaluation prompts follow “A/An [ethnicity] person is [verb]-ing [context]” and cover all ethnicity–verb combinations. This is the ethnicity+person evaluation condition. For each evaluation prompt, we generate images with “A/An [ethnicity] [gender] is [verb]-ing [context].” This ethnicity+gender generation condition explicitly specifies gender, ethnicity, and verb. Evaluation prompts omit gender to measure inherent reward model preference; generation prompts specify it.

We standardize reward scores using their mean and standard deviation, which measures score spread. For each generation prompt, we then average standardized scores across its 100 images. To obtain PBSG, we hold ethnicity and verb fixed and subtract the women’s average standardized score from the men’s. This adapted PBSG can exceed one. Positive values indicate a preference for men, and negative values indicate a preference for women.

CLIP shows a slight woman-preference bias of –0.0726. Fine-tuning on HPDv2 shifts HPSv2.0 toward a strong man-preference of +0.6039 across ethnic groups. PickScore, at –0.1157, and HPSv2.1, at –0.0984, instead show woman-preference biases. HPSv2.1’s training data remain undisclosed. These shifts align with each model’s training data and show consistent gender preferences across ethnicities. Figures 19, 20, 21, 22, 23, 24 and 25 report PBSG across 42 verbs for each ethnic group.

**Ethnicity Bias.** Evaluation prompts follow “A person is [verb]-ing [context],” the person-only condition. For each evaluation prompt, we generate images with the more specific “A/An [ethnicity] person is [verb]-ing [context],” the ethnicity+person condition. Generation prompts specify ethnicity and verb. Evaluation prompts omit ethnicity to measure inherent reward model preference.

We standardize reward scores using their mean and standard deviation, then average the 100 images for each generation prompt. For each fixed verb, we apply the softmax function to scores across ethnicities. Softmax converts scores into normalized proportions (Bridle, 1990; Bishop, 2006). These proportions represent ethnicity preferences within each verb context and let us calculate RDSe and SDI. Positive RDSe indicates overrepresentation, and negative RDSe indicates underrepresentation. Higher SDI indicates more balanced, diverse outputs.

CLIP slightly favors White individuals, with RDS = 0.0182. It also has the highest SDI, 0.8495, indicating relatively balanced ethnic representation. After fine-tuning, HPSv2.0 shifts toward Middle Eastern individuals (RDS = 0.0315), HPSv2.1 toward Latino individuals (RDS = 0.0382), and PickScore toward East Asian individuals (RDS = 0.0352). All models show lower SDI after alignment, indicating reduced ethnic diversity. Figures 26, 27, 28 and 29 report ethnicity bias across 42 verbs.

## 6 Social Biases in Preference Alignment

We next examine how preference alignment affects bias in video generation. We fine-tune a Video Consistency Model distilled from VideoCrafter-V2, called VCM-VC2 (Li et al., 2024), with three image reward models: HPSv2.0, HPSv2.1, and PickScore. A consistency model supports generation through a distilled process. We compare social bias distributions before and after tuning to assess how each reward model shapes identity representation.

Following the T2V-Turbo-V1 training protocol (Li et al., 2024), we add reward feedback to Latent Consistency Distillation (Luo et al., 2023). This state-of-the-art process distills a model in its latent space, a compressed representation of the data. We use single-step video generation. During distillation, a student model learns from a pretrained teacher text-to-video model. We directly optimize decoded video frames to maximize image–text alignment reward scores. This guides each frame toward representations that better match human preferences. Appendix G provides video model post-training and inference details.

We evaluate the aligned models with our framework (Section 4). Table 5 reports PBSG for gender imbalance within ethnic groups, plus RDSe and SDI for ethnic representation disparity and overall diversity. Appendix D provides further analysis across 42 verbs and statistical analysis (Table 12).

| Models | Average PBSG | White PBSG | White RDS | Black PBSG | Black RDS | Latino PBSG | Latino RDS | East Asian PBSG | East Asian RDS | Southeast Asian PBSG | Southeast Asian RDS | India PBSG | India RDS | Middle Eastern PBSG | Middle Eastern RDS | Overall SDI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VCM-VC2 | 0.8034 | 0.7925 | 0.6405 | 0.7758 | -0.2381 | 0.8090 | — | 0.7115 | -0.2333 | 0.7945 | — | 0.8634 | — | 0.8071 | -0.1690 | 0.1433 |
| + HPSv2.0 | 0.9116 | 0.9667 | 0.3667 | 0.9214 | — | 0.9667 | — | 0.8500 | — | 0.9214 | — | 0.9000 | — | 0.8548 | -0.3667 | 0.1257 |
| $\Delta$ | +0.1082 | +0.1742 | -0.2738 | +0.1456 | — | +0.1577 | — | +0.1385 | — | +0.1269 | — | +0.0366 | — | +0.0477 | -0.1977 | -0.0176 |
| + HPSv2.1 | 0.2267 | 0.1321 | 0.4286 | 0.2381 | — | 0.3738 | — | 0.1571 | — | 0.3619 | — | 0.1452 | — | 0.1786 | -0.4286 | 0.0976 |
| $\Delta$ | -0.5767 | -0.6604 | -0.2119 | -0.5377 | — | -0.4352 | — | -0.5544 | — | -0.4326 | — | -0.7182 | — | -0.6285 | -0.2596 | -0.0457 |
| + PickScore | 0.3714 | 0.3429 | 0.6833 | 0.3357 | -0.1810 | 0.7190 | -0.1929 | 0.1450 | -0.1952 | 0.4548 | — | 0.2500 | — | 0.3515 | -0.1143 | 0.1467 |
| $\Delta$ | -0.4320 | -0.4496 | +0.0428 | -0.4401 | +0.0571 | -0.0900 | -0.1929 | -0.5665 | +0.0381 | -0.3397 | — | -0.6134 | — | -0.4556 | +0.0547 | +0.0034 |

**Table 5:** Social biases of aligned models. All values average scores across 42 verbs. A dash (“—”) marks cases with zero recognizable instances of the corresponding ethnicity under person-only prompts, where ethnicity is unspecified. This severe lack of diversity prevents metric computation for those subgroups.

**Ethnicity-Aware Gender Bias.** We evaluate gender portrayals under ethnicity+person prompts using PBSG. Positive scores indicate more frequent portrayals of men; negative scores indicate a preference for women. VCM-VC2 strongly favors men across ethnicities. HPSv2.0 alignment strengthens this bias.

In contrast, alignment with HPSv2.1 and PickScore substantially lowers PBSG, shifting outputs toward more balanced or woman-preferred portrayals. This reflects the woman bias in both reward models, which steer generation away from the base model’s male dominance. Figures 30, 31, 32, 33, 35, 34, 36 and 37(b) report PBSG across 42 verbs for each ethnic group.

**Ethnicity Bias.** Under the ethnicity-only condition, we evaluate representation balance with RDSe and SDI. Positive RDSe indicates overrepresentation; negative RDSe indicates underrepresentation. Higher SDI indicates more equitable overall demographic representation.

VCM-VC2 strongly favors White individuals (RDS = 0.6405), while underrepresenting Black, East Asian, and Middle Eastern groups. HPSv2.1 alignment reduces some disparities by improving balance for White and Black groups. However, it substantially reduces Latino representation (RDS = –0.4352) and lowers SDI, indicating less diversity. PickScore instead achieves the highest SDI and more balanced representation across most ethnic groups. It therefore produces the most demographically equitable outputs. Figure 38 reports ethnicity bias across 42 verbs.

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

**Table 6:** Temporal Attribute Stability (TAS) across models and attributes. A high TAS indicates that the actor’s identity representation stays stable throughout the video. A low TAS indicates that the representation “flickers” or changes, an important temporal artifact. “Std” means standard deviation. Subscripts in red and blue indicate improvements and degradations relative to the base model VCM-VC2.

**Temporal Attribute Stability.** Table 6 summarizes temporal consistency in identity portrayals after alignment. Overall, alignment substantially improves technical coherence. For ethnicity, mean TAS rises from 0.8904 in VCM-VC2 to 0.9632 after HPSv2.1 alignment. The share of videos with perfect stability rises by 0.2660, from 0.5670 to 0.8330. Gender shows similar gains, reaching near-saturation stability of 0.9895 and perfect stability of 0.9440. TAS standard deviation also consistently falls—for example, by –0.0654 for ethnicity. This indicates more uniform frame-level consistency across videos.

These stability gains carry an important caveat. When aligned models inherit biased preferences, greater stability can entrench rather than reduce bias. HPSv2.0 alignment raises overall man-preference PBSG from 0.8034 to 0.9116. It also locks this bias in over time: gender stability reaches 0.9967, and nearly all videos, 0.9830, achieve perfect stability. In other words, the model becomes better at being biased. Its portrayals become smoother and more coherent, yet more stereotyped.

Our temporal framework uniquely reveals this insight: although alignment improves perceived quality and temporal consistency, it can make social bias more persistent and deeply embedded in generation. VideoBiasEval therefore detects not only whether a model is biased, but also whether alignment makes that bias resilient over time. Alignment can turn representational artifacts into stable, systematic distortions.

**Summary and Implications.** Preference alignment can reduce or amplify existing social biases, depending on the reward model. HPSv2.0 reinforces the base model’s male- and White-preferred tendencies. HPSv2.1 and PickScore both show woman-preferred reward patterns and counter the base model’s male dominance, producing more balanced portrayals. However, neither achieves complete demographic parity. Ethnic disparities remain. These results suggest that aligned video diffusion models largely inherit the direction and strength of social bias from their reward models.

## 7 Controllable Preference Modeling for Video Diffusion Models

We next ask whether we can control these biases. Specifically, can changes to social attribute distributions in image preference datasets systematically steer reward models—and then video diffusion models—toward more equitable or deliberately calibrated portrayals (Sheng et al., 2020)? We use gender as a case study. We examine how gender composition in preference data affects the direction of bias and the alignment behavior of reward and video models.

### 7.1 Image Reward Dataset Construction

We construct two reward datasets using images from Section 5.1: a man-preferred version and a woman-preferred version. These datasets guide diffusion models toward gender-specific representations. Each includes 2.94 million preference pairs from the Ethnicity+Gender set. Each pair shows the same action and ethnicity but differs in gender, such as M-1 versus W-1 in Figure 5. Prompts follow “A/An [ethnicity] person is [action]-ing [context].”

In the man-preferred dataset, male images receive label 1 and female images receive label 0. The woman-preferred dataset reverses these labels. We also add 537,660 image pairs from HPDv2 to increase face-free diversity. When the base model already favors men, the woman-preferred dataset helps correct that imbalance and promotes more equitable gender representation.

### 7.2 Image Reward Model Development & Alignment Tuning

We fine-tune two reward models on a pretrained CLIP vision encoder, the part of CLIP that represents images. The Man-Preferred Reward Model (RMM) uses the man-preferred dataset. The Woman-Preferred Reward Model (RMW) uses the woman-preferred dataset. Each learns the gender-specific aesthetic and representational preferences in its data.

Table 7 shows that RMM consistently assigns higher PBSG scores across demographic groups, favoring man-preferred portrayals. RMW shows the opposite pattern and systematically favors woman-preferred content. This clear difference shows that reward tuning captures and amplifies gendered preferences. Appendix F provides reward model training and inference details.

| Models | Average | White | Black | Latino | East Asian | Southeast Asian | India | Middle Eastern |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP | -0.0726 | 0.0343 | -0.1198 | -0.0934 | -0.1315 | -0.0865 | -0.0508 | -0.0607 |
| RMM | 1.5280<sub>+1.60</sub> | 1.6300<sub>+1.60</sub> | 1.5752<sub>+1.70</sub> | 1.5524<sub>+1.65</sub> | 1.4323<sub>+1.56</sub> | 1.4525<sub>+1.54</sub> | 1.5619<sub>+1.61</sub> | 1.4914<sub>+1.55</sub> |
| RMW | -0.7448<sub>-2.27</sub> | -0.6318<sub>-2.26</sub> | -0.7943<sub>-2.37</sub> | -0.8279<sub>-2.38</sub> | -0.6282<sub>-2.06</sub> | -0.6429<sub>-2.10</sub> | -0.8846<sub>-2.45</sub> | -0.8042<sub>-2.30</sub> |

**Table 7:** Preference bias of reward models. All values average scores across 42 verbs.

We deliberately do not train a “fair” or demographically neutral reward model. As Section 6 shows, the base video generator, such as VCM-VC2, already strongly favors men. With this unequal starting point, a neutral reward signal would merely reinforce existing imbalances rather than correct them. We instead use a deliberately counter-biased reward model: the woman-preferred RMW, analogous to HPSv2.1. It actively steers generation toward gender equilibrium. This targeted strategy produces much more balanced portrayals, showing that directional reward tuning can effectively correct bias.

We then use RMM and RMW to align VCM-VC2. The same preference-optimization framework produces two variants: one tuned toward man-preferred content and one toward woman-preferred content. Table 8 shows that RMM alignment raises PBSG across demographic groups and reinforces man-preference bias. RMW alignment substantially lowers PBSG, producing a strong shift toward woman-preference bias.

These results confirm that controllable preference modeling can finely adjust gender bias in video generation. It offers a principled, flexible way to amplify or reduce social tendencies in diffusion-based models. We can also frame the search for the most balanced aligned video generator as a data composition problem. By systematically changing the share of man- and woman-preferred samples in the reward dataset, one can identify a mixture that minimizes overall bias while preserving aesthetic fidelity. This offers a promising direction for dataset-driven bias control in alignment. Figures 39, 40, 41, 42 and 43 report PBSG across 42 actions for each ethnic group.

| Models | Average | White | Black | Latino | East Asian | Southeast Asian | India | Middle Eastern |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VCM-VC2 | 0.8034 | 0.7925 | 0.7758 | 0.8690 | 0.7115 | 0.7945 | 0.8634 | 0.8071 |
| + RMM | 0.9584<sub>+0.16</sub> | 0.9595<sub>+0.17</sub> | 0.9524<sub>+0.18</sub> | 0.9756<sub>+0.11</sub> | 0.9437<sub>+0.23</sub> | 0.9447<sub>+0.15</sub> | 0.9640<sub>+0.10</sub> | 0.9709<sub>+0.16</sub> |
| + RMW | 0.3082<sub>-0.50</sub> | 0.3341<sub>-0.46</sub> | 0.3913<sub>-0.38</sub> | 0.3314<sub>-0.54</sub> | 0.1008<sub>-0.61</sub> | 0.2639<sub>-0.53</sub> | 0.3446<sub>-0.52</sub> | 0.3894<sub>-0.42</sub> |

**Table 8:** Social biases of aligned models. All values average scores across 42 verbs.

**Summary and Implications.** Our framework shows that constructing and tuning social-attribute-aware reward datasets can systematically direct bias in video diffusion models. Changes in gender composition teach reward models directional biases. These models then transmit those biases, reinforcing or countering the base generator’s tendencies. Counter-biased reward models, such as woman-preferred RMW, can actively correct skewed portrayals and produce more balanced, socially representative outputs.

Beyond binary control, these results suggest a continuous range of alignment. Carefully chosen mixtures of man- and woman-preferred data could offer adjustable trade-offs between bias mitigation and aesthetic coherence. Sensitivity analysis across 42 actions also shows that even semantically neutral activities can undergo large gender shifts during alignment. This reinforces the need for detailed event-level auditing. Overall, dataset composition—not architectural changes alone—offers a powerful, interpretable way to steer generative models’ social behavior.

(a) Change in PBSG for the video generation model before and after alignment with RMM and RMW, broken down by action. Figure 6(b) and Figure 6(c) build on this figure.

(b) Sensitive actions in man-preferred alignment tuning.

(c) Sensitive actions in woman-preferred alignment tuning.

**Figure 6:** Action-level effects of alignment tuning guided by RMM and RMW.

**Which Actions Are Most Sensitive During Alignment Tuning?** We measure PBSG changes before and after alignment to study how specific actions respond to gender-oriented reward tuning. In Figure 6(a), actions form distinct groups according to the reward model. RMM alignment, shown in teal, strengthens male-preferred portrayals for exercise, row, and cook. RMW alignment, shown in orange, shifts representations toward women-preferred actions such as bake, sleep, and sweep.

Figure 6(b) and Figure 6(c) rank actions by normalized sensitivity. We define this as $\Delta$ PBSG divided by each reward model’s baseline PBSG. Sleep, stretch, and read rank highest under both man- and woman-preferred tuning. Thus, even socially neutral or domestic activities can show strong gender bias after alignment. These results show systematic, action-specific bias amplification during alignment. They also demonstrate that our event-centered framework can reveal detailed behavioral sensitivities across gendered dimensions.

**Responsible Use of Controllable Preference Modeling.** Controllable preference modeling can steer social attribute distributions, but technical “debiasing” is not a complete solution. It also carries substantial risks when applied unilaterally. Practitioners should use it thoughtfully to support diverse stakeholder input, not replace that input.

“Equitable representation” is a context-dependent normative judgment—a decision about what representation should be. It requires active community consultation. We encourage practitioners to work with affected communities to set appropriate representation goals. Automated optimization alone cannot resolve complex social biases.

## 8 Conclusion

Our work identifies and studies major blind spots in evaluating social bias in text-to-video generation. VideoBiasEval separates identity attributes from content meaning and systematically tracks how alignment changes social representations. Our analyses show that reward-model-based alignment inherits and often amplifies biases in human preference data. These findings stress the need to audit and mitigate bias at every stage of video generation. Such work can advance more equitable and socially aware generative systems.

## Limitations

Our evaluation of social biases introduced through alignment tuning has several limitations.

First, we study only gender and ethnicity, using predefined categories from U.S. Census conventions and prior research. These categories support controlled evaluation, but they are socially constructed. They cannot fully capture identity’s fluidity, intersectionality, or cultural nuances (Yin et al., 2024; Qiu et al., 2024a; Qiu et al., 2025). Future work should explore richer identity representations, including intersectional groups.

Second, our VLM evaluators rely on image-level classification. Although we validate them against human judgments, they may have their own biases or make errors, especially with stylized or ambiguous frames. Combining multiple models helps reduce these problems. However, ground-truth annotations for a larger, more diverse video set would further strengthen measurement reliability.

Third, we mainly assess one training strategy: single-step latent consistency distillation. We also study a limited set of reward models. Other protocols, such as reinforcement-learning-based tuning or multi-turn video instruction alignment, may produce different bias dynamics that our study does not capture.

Fourth, our controllable preference experiments show that targeted bias adjustment is feasible, but they only manipulate synthetic gender preferences. They do not address broader questions of value alignment, normative appropriateness, or long-term societal effects. These questions remain crucial for responsible deployment.

Lastly, we currently benchmark VideoBiasEval on a fixed set of 42 socially associated actions. This supports detailed control but may limit generalization to open-ended generation or actions outside our taxonomy. We hope these limitations encourage research on comprehensive, culturally grounded, and ethically aligned evaluation pipelines for video generative models.

## Broader Impact Statement

Video diffusion models are moving from research prototypes to widely used content-creation tools. Their influence on visual culture and social representation is therefore growing. Our work highlights an important but often overlooked consequence: alignment tuning can stabilize and entrench social biases.

**Stabilization of Bias.** Alignment tuning, such as RLHF, is essential for improving visual fidelity and temporal consistency. Yet our results show that it can also act as a “bias stabilizer.” By removing visual artifacts and smoothing temporal flickers, alignment makes stereotyped representations appear more natural, coherent, and authoritative. Unlike obvious errors in unaligned models, these “polished” biases are harder to recognize as failures. They risk subtly normalizing skewed demographic portrayals in synthetic media. Without checks, this stabilization could reinforce harmful stereotypes under the appearance of high-quality, preferred content.

**Dual-Use of Controllable Preference Modeling.** Section 7 also shows that controllable preference modeling has dual uses: the same technique can help or harm. Curated preference datasets let reward models systematically amplify or suppress social attributes. This capability offers a powerful way to correct historical underrepresentation and reduce bias. However, it also makes alignment a double-edged sword. Tools designed to promote fairness could instead deliberately exclude groups, enforce uniformity, or generate propaganda aligned with exclusionary worldviews.

**Mitigation and Responsibility.** These risks require a change in how we evaluate generative video systems. Aesthetic metrics alone are not enough. Developers must use multi-granular, event-centered auditing frameworks such as VideoBiasEval to continually monitor generated content’s social composition. We urge researchers and practitioners to treat preference dataset curation as rigorously as model architecture design. The “human preferences” guiding these systems should reflect a diverse, equitable society rather than reinforce existing prejudices.

## Acknowledgment

We thank the anonymous reviewers for their thoughtful feedback and suggestions. The AFOSR MURI grant #FA9550-22-1-0380, the Office of Naval Research (ONR) under Award #N00014-23-1-2780, and the National Science Foundation under Award #IIS-2449768 supported this research in part. The NVIDIA Academic Grant Program provided cloud credits. The Technical AI Safety Research Program at Coefficient Giving provided additional funding. The views and conclusions in this work belong to the authors. They should not be interpreted as representing the official policies or endorsements of AFOSR, ONR, the U.S. Government, the National Science Foundation, NVIDIA, or Coefficient Giving.
