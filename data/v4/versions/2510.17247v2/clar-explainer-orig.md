## Summary

This paper asks how alignment tuning—post-training optimization toward human preferences—changes social representation in video diffusion models. It introduces VideoBiasEval, an event-based evaluation framework that separates actor attributes from verbs and contexts. The framework measures ethnicity bias, gender bias within ethnic groups, changes after alignment, and the temporal stability of demographic portrayals. The authors examine human preference datasets, image reward models trained on those datasets, and video models aligned with those rewards. They report that preference biases can be inherited, amplified, and made more temporally stable. They also construct man-preferred and woman-preferred reward datasets. These experiments show that reward-data composition can reinforce or counteract a video model’s existing gender bias. The paper cautions that demographic categories, automated evaluators, training choices, and the fixed action set limit the analysis.

## Glossary

*This glossary defines the paper’s technical terms, labels, symbols, and codes.*

### Methods and evaluation terms

- **Text-to-video (T2V) generation** — Generation of videos from natural-language prompts.
- **Text-to-image (T2I) generation** — Generation of images from text prompts.
- **Video diffusion model** — A video generation model evaluated or optimized in this study.
- **Alignment tuning / preference alignment / post-training optimization** — Further optimization of a base model to better match human preferences beyond likelihood-based training.
- **Aligned model (a)** — A video diffusion model that undergoes preference-guided post-training optimization.
- **Unaligned model (u)** — A model trained solely with likelihood-based objectives and used as a baseline.
- **Likelihood-based objective** — The training objective used by the paper’s unaligned baseline models.
- **Reward model / preference model** — A model that assigns preference scores and guides further optimization of a generator.
- **Reward function** — A function trained from human-labeled comparisons to guide optimization.
- **Aesthetic reward** — A reward used to guide generation toward preferred visual representations.
- **Reward-weighted fine-tuning** — A commonly used form of alignment that uses learned reward feedback.
- **RLHF-style procedure / RL-based tuning** — Reward-guided post-training procedures discussed as alignment methods or alternative protocols.
- **Human preference pair** — Two images compared by a human annotator, who selects the image that best matches a caption.
- **Preferred / dispreferred image** — The selected / unselected image in a preference comparison.
- **VideoBiasEval** — The paper’s event-based framework for evaluating social representation in video generation.
- **Event / event-centric evaluation** — An actor performing a verb within a context, used as the unit of bias analysis.
- **Event-based prompting** — Prompt construction that controls actor attributes separately from verbs and contexts.
- **Social attributes / demographic attributes** — The gender and ethnicity attributes analyzed in this paper.
- **Representational bias** — Skew in who appears, how often they appear, or the roles in which they appear.
- **Ethnicity-aware gender bias** — Gender imbalance measured separately within ethnic groups.
- **Ethnicity bias** — Overrepresentation or underrepresentation of ethnic groups.
- **Bias shift** — A change in a bias metric between unaligned and aligned models.
- **Multi-granular evaluation** — Evaluation at frame, temporal, video, event, and demographic-group levels.
- **Frame-level classification** — Classification of gender and ethnicity in individual video frames.
- **Video-level label** — An attribute label obtained by aggregating frame-level predictions.
- **Majority voting** — Aggregation of predictions by selecting the label receiving the most votes.
- **VLMs-as-judges ensemble** — Multiple vision–language models used together to classify attributes.
- **Vision–language model (VLM)** — An automated judge that infers gender and ethnicity from generated frames or images.
- **LLM** — A model that is not used at runtime to generate actors or contexts in the deterministic prompting protocol.
- **Attribute flickering** — Inconsistent identity classification across frames.
- **Temporal coherence / temporal consistency** — Consistency of generated portrayals over time.
- **Perfect stability** — The perfect-stability outcome reported alongside TAS in Table 6.
- **Uniform or parity-based reference** — A diagnostic balance benchmark, not a claim about real-world demographic proportions.
- **Counter-biased reward model** — A deliberately directional reward model used to counter an existing generator bias.
- **Controllable preference modeling** — Construction and tuning of reward data to steer social-attribute distributions.
- **Data composition** — The proportions of preference examples used to construct a reward dataset.
- **Normalized sensitivity** — An action’s change in PBSG divided by the reward model’s baseline PBSG.
- **Latent Consistency Distillation** — The distillation process into which the study incorporates image-reward feedback.
- **Teacher / student model** — The pretrained text-to-video model and the model distilled from it.
- **Single-step video generation** — The generation approach used during the study’s reward-guided distillation.
- **Face-free diversity** — The diversity for which Section 7.1 adds HPDv2 image pairs.
- **Human verification** — Human review used to ground and sanity-check automated evaluation.
- **Pearson correlation** — The reported measure of agreement between VLM and human judgments.
- **Cohen’s Kappa / Fleiss’ Kappa** — Agreement metrics reported for human verification.
- **Sensitivity analysis** — Re-evaluation on a high-agreement subset, or action-level analysis of alignment responses.
- **Softmax** — The function used to normalize reward scores across ethnicities.
- **Standardized score** — A reward score standardized using its mean and standard deviation.
- **Mean / median / standard deviation (Std)** — The summary statistics reported for TAS.
- **Fréchet Video Distance, CLIP-Score, object consistency** — Existing T2V evaluation metrics discussed in Section 2.
- **GRiT-based metrics** — Existing metrics that can check whether an object or role appears without flagging demographic imbalance.
- **CLIP-based alignment** — Textual-fidelity evaluation that the paper says ignores demographic balance.
- **Intersectional attributes / groups** — Richer identity representations identified as a future direction.
- **Multi-turn video instruction alignment** — An alternative training protocol not evaluated in this study.
- **Dual use** — The possibility of using controllable preference modeling either to mitigate bias or to exclude groups.

### Symbols, metrics, and prompt codes

- **\(\langle p,a,c\rangle\)** — An event tuple containing an actor, a verb, and a context.
- **\(p\)** — The actor in an event; in statistical reporting, \(p=0.68\) is the reported significance value.
- **\(a\)** — The verb in an event tuple; in model tables, “a” marks an aligned model.
- **\(c\)** — The context in an event.
- **\(g\)** — A classified gender label.
- **\(G=\{\text{man, woman}\}\)** — The set of gender labels output by the VLM judges.
- **\(e\)** — An ethnicity label or ethnic group.
- **\(E\)** — The set of seven ethnicity labels used in evaluation.
- **\(|E|\)** — The number of ethnicity groups.
- **\(N_{\text{man}}\), \(N_{\text{woman}}\)** — Counts of representations depicting men and women.
- **\(N_e\)** — The number of outputs identified as ethnicity \(e\).
- **\(N_{\text{total}}\)** — The relevant total: men plus women for PBSG, or outputs with identifiable ethnicity for ethnicity metrics.
- **\(P_e=N_e/N_{\text{total}}\)** — The representation proportion of ethnic group \(e\).
- **PBSG / \(\mathrm{PBS}_G\)** — Proportion Bias Score for Gender; video-count difference divided by total, or a standardized reward-score difference in reward-model evaluation.
- **RDSe / \(\mathrm{RDS}_e\) / RDS** — Representation Deviation Score for Ethnicity, \(P_e-1/|E|\).
- **RDSWhite** — The RDS value for White representation.
- **SDI** — Simpson’s Diversity Index, \(1-\sum_{e\in E}P_e^2\).
- **TAS** — Temporal Attribute Stability: the percentage of frames matching the video’s final majority-voted attribute label.
- **\(\Delta\)** — A change relative to an unaligned or baseline model.
- **\(\Delta\)PBSG, \(\Delta\)RDSe, \(\Delta\)SDI** — Alignment-induced changes in the corresponding metrics.
- **\(\sum\)** — Summation over the ethnicity groups in the SDI formula.
- **\(\in\), \(\{\ \}\), \([-1,1]\)** — Set membership, a listed set, and the stated range of the video-count PBSG metric.
- **\(\approx\), \(\rightarrow\), \(+\), \(−\)** — Approximation, a pipeline or reported change, and positive or negative values or shifts.
- **“—”** — A missing ethnicity-metric value where no recognizable instances of that ethnicity were generated under person-only prompting.
- **[actor], [verb], [context]** — Fixed-template slots for an actor, activity, and surrounding situation.
- **[ethnicity], [gender], [action]** — Prompt slots for an ethnic descriptor, gender descriptor, and activity.
- **A/An; [verb]-ing; [action]-ing** — The article and activity forms used in the prompt templates.
- **Person Only / person-only** — Prompting with “person” and no explicit demographic descriptor.
- **Ethnicity+Person / ethnicity+person** — Prompting with a predefined ethnic descriptor followed by “person.”
- **Ethnicity+Gender / ethnicity+gender** — Prompting with both ethnicity and gender explicitly specified.
- **Ethnicity-only** — The condition named in Section 6’s ethnicity analysis; Section 3.2 says an additional ethnicity-only setting is unnecessary.
- **M-1 / W-1** — Example man/woman image labels used to illustrate same-action, same-ethnicity preference pairs.
- **RM\(_{\text{M}}\) / RMM** — Man-Preferred Reward Model.
- **RM\(_{\text{W}}\) / RMW** — Woman-Preferred Reward Model.
- **# of Prompts** — The prompt-count column in Tables 1 and 3.
- **Overall / Average** — Table labels for overall diversity and scores averaged across the evaluated verbs.

### Identity and action labels

- **Man** — A gender-prompt category and one of the two VLM output labels.
- **Woman** — A gender-prompt category and one of the two VLM output labels.
- **Person** — The neutral actor term used in prompts.
- **Non-binary person** — One of the four gender-prompt categories.
- **White** — One of the seven evaluated ethnic groups.
- **Black** — One of the seven evaluated ethnic groups.
- **Indian** — One of the seven evaluated ethnic groups.
- **India** — The column label used for this group in the result tables.
- **East Asian** — One of the seven evaluated ethnic groups.
- **Southeast Asian** — One of the seven evaluated ethnic groups.
- **Middle Eastern** — One of the seven evaluated ethnic groups.
- **Latino** — One of the seven evaluated ethnic groups.
- **Bake, bike, call, clean, climb, cook, cough, cry, drink, drive, eat, exercise, fish, hit, jump, kick, kneel, laugh, lift, paint, pick, pitch, pray, read, ride, row, run, shop, shout, sit, skate, sleep, smile, stand, stare, stretch, study, sweep, throw, walk, wash, work** — Each is one of the 42 evaluated verbs, selected for prior social associations and visual distinctiveness.

### Models, datasets, benchmarks, and identifiers

- **ModelScope** — An unaligned baseline and the base model for InstructVideo.
- **InstructVideo** — A ModelScope-based model aligned using HPSv2.0.
- **VideoCrafter / VideoCrafter-2 / VideoCrafter-V2 / Video-Crafter-V2** — Video-model names used in the paper; VideoCrafter-2 is the unaligned base for T2V-Turbo-V1 and the source of the distilled consistency model.
- **T2V-Turbo-V1** — A VideoCrafter-2-based model aligned with HPSv2.1, InternVid2-S2, and ViCLIP.
- **Video Consistency Model / VCM-VC2** — The consistency model distilled from VideoCrafter-V2 and aligned in Sections 6–7.
- **InternVid2-S2 / ViCLIP** — Models used in T2V-Turbo-V1 alignment.
- **HPDv2** — An analyzed human image-preference dataset used to train HPSv2.0.
- **HPDv2.1** — The unreleased dataset used to train HPSv2.1.
- **Pick-a-Pic** — An analyzed human image-preference dataset used to train PickScore.
- **HPSv2.0** — An image reward model trained on HPDv2.
- **HPSv2.1** — An image reward model trained on unreleased HPDv2.1.
- **HPSv2** — The reward-model label used in Table 6.
- **PickScore** — An image reward model trained on Pick-a-Pic.
- **CLIP** — The shared base model for HPSv2.0, HPSv2.1, and PickScore; a pretrained vision encoder also used for RMM and RMW.
- **Qwen2-VL-7B / Qwen2.5-VL-7B / InternVL2.5-8B** — The three open-source VLM judges.
- **GPT-4o-mini** — The model used to extract gender, ethnicity, and verb attributes from captions.
- **FLUX** — The T2I model used to generate the controlled reward-model evaluation benchmark.
- **ImageRewardDB** — A dataset that, with HPDv2, inspires the reward-model benchmark construction.
- **WebVid-10M** — The training dataset whose effects are analyzed in Appendix A.
- **VBench / EvalCrafter / T2V-CompBench** — Existing T2V evaluation benchmarks discussed in Section 2.
- **StableBias / DALL-Eval / SocialCounterfactuals** — Existing generative-model bias evaluations discussed in Section 2.
- **Appendices A, C, D, F, G** — Referenced locations for additional model analysis, reward evaluation, alignment analysis, reward training, and video post-training details.
- **Tables 9, 10, 11, 12** — Referenced statistical or preference-distribution tables not included in the supplied text.
- **OpenReview / C0yxuS6jty** — The review platform and identifier in the paper’s review link.
- **VideoBiasEval repository** — The GitHub address included in the author block.
- **AFOSR MURI / #FA9550-22-1-0380** — A supporting grant program and award identifier.
- **ONR / #N00014-23-1-2780** — The Office of Naval Research and its supporting award identifier.
- **National Science Foundation / #IIS-2449768** — A supporting agency and award identifier.
- **NVIDIA Academic Grant Program** — The source of supporting cloud credits.
- **Technical AI Safety Research Program at Coefficient Giving** — A source of research funding.

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

*This section introduces the evaluation framework and summarizes how preference biases pass through alignment.*

Video diffusion models have substantially improved text-to-video generation: generating videos from text prompts. Alignment tuning uses reward models trained on human preferences to improve visual quality. It can also encode and amplify social biases.

We introduce VideoBiasEval, a diagnostic framework for evaluating social representation in generated videos. It uses established social bias taxonomies and event-based prompts. These prompts separate semantic content—verbs and contexts—from actor attributes—gender and ethnicity. Its multi-granular metrics measure: (1) overall ethnicity bias; (2) gender bias conditioned on ethnicity; (3) social-attribute distribution shifts across model variants; and (4) the temporal persistence of bias within videos.

We conduct the first end-to-end analysis linking biases in human preference datasets, their amplification in reward models, and their propagation into aligned video diffusion models. Alignment strengthens representational biases and makes them temporally stable. The resulting portrayals are smoother but more stereotyped. These findings support bias-aware evaluation and mitigation throughout alignment.

Reviewed on OpenReview: https://openreview.net/forum?id=C0yxuS6jty

Figure 1: Overview of our work: (1) We introduce VideoBiasEval, a social bias evaluation framework for video generation that leverages event-based prompts and multi-granular metrics to assess ethnicity and gender bias (bottom left, Section 3). The framework represents videos through social attribute annotations (top), where visual-language models (VLMs) infer actor attributes such as gender and ethnicity across frames and aggregate them for bias quantification Section 3.3). (2) We conduct the first comprehensive analysis of how image-based reward models, shaped by human-labeled preferences, influence the distribution of social attributes in diffusion-generated videos, disentangling human preference biases, reward preference biases, and their downstream impact on video diffusion model generation biases (bottom right, Section 5.1, Section 5.2, and Section 6).

**Claims and evidence**

- **Preference biases pass through the pipeline.** Section 5.1 reports man preference for 69.23% (18/26) of qualifying HPDv2 verbs and woman preference for 61.11% (11/18) of qualifying Pick-a-Pic verbs. Table 4, Average PBSG column: HPSv2.0, 0.6039; PickScore, -0.1157.
- **Alignment can amplify and stabilize bias.** Table 5, Average PBSG: VCM-VC2, 0.8034; + HPSv2.0, 0.9116. Table 6, Gender rows: mean TAS rises from 0.9763 to 0.9967.

## 1 Introduction

*This section defines alignment, states the evaluation gaps, and outlines the paper’s analysis and contributions.*

Video diffusion models now generate high-quality videos from natural-language prompts (Chen et al., 2024a; Wang et al., 2023a; Yuan et al., 2024; Li et al., 2024). Applications include creative media, education, and professional simulation (Cho et al., 2024; Miller et al., 2024). Many leading systems use alignment tuning to improve visual fidelity, coherence, and controllability. A prominent approach learns from human preferences (Wu et al., 2023; Xu et al., 2024; Li et al., 2024; Yuan et al., 2024; Liu et al., 2024a; Prabhudesai et al., 2024; Black et al., 2023; Ma et al., 2025). Reward functions trained on human-labeled comparisons often judge individual images or frames and guide post-training optimization (Wu et al., 2023; Kirstain et al., 2023a; Xu et al., 2024).

Here, **aligned models** are video diffusion models further optimized to match human preferences beyond likelihood-based training. Common methods include reward-weighted fine-tuning and RLHF-style procedures: post-training guided by learned preference or aesthetic rewards. The general pipeline is:

\[
\text{base video diffusion model}
\rightarrow
\text{preference or reward model}
\rightarrow
\text{post-training optimization}.
\]

**Unaligned models** use only likelihood-based training. They provide controlled baselines for isolating alignment effects.

Alignment improves perceptual quality and temporal coherence, but subjective preferences introduce risks. Human judgments may be collected without explicit attention to cultural, social, or demographic diversity. Reward models and generators may inherit and reinforce these judgments’ biases (Qiu et al., 2023). Alignment can therefore change who appears, how often, and in which roles. We ask: **How does alignment tuning influence the distribution of social attributes in video diffusion models?**

Answering this question requires prompts that elicit social attributes, metrics that measure their distributions, and a protocol that tracks changes before and after alignment. Existing frameworks (Huang et al., 2024; Liu et al., 2024b; Sun et al., 2024) have three limitations:

1. Their prompts do not adequately represent diverse social identities.
2. Their identity coverage and metrics do not comprehensively track alignment’s effects.
3. They lack a dedicated method for measuring before-and-after social-attribute shifts.

VideoBiasEval addresses these gaps (Section 3). Its event-based prompts build on established taxonomies (Zhao et al., 2017; Hendricks & Nematzadeh, 2021; Cho et al., 2023; Qiu et al., 2023). They control verb types separately from actor identities. This separation supports interpretable comparisons across contexts. Following work on alignment and fairness (Luccioni et al., 2023; Shen et al., 2023), we focus on gender and ethnicity because these dimensions have comparatively well-defined evaluative boundaries. Our metrics assess (1) ethnicity bias, (2) ethnicity-conditioned gender bias, (3) distribution shifts across models, and (4) temporal persistence.

We trace the alignment pipeline in three stages. First, we examine demographic preferences in **HPDv2** and **Pick-a-Pic**, two human image-preference datasets (Wu et al., 2023; Kirstain et al., 2023a; Section 5.1). Second, we examine **HPSv2.0**, **HPSv2.1**, and **PickScore**, image reward models trained on preference data (Wu et al., 2023; Kirstain et al., 2023a; Section 5.2). Third, we align a video consistency model distilled from **VideoCrafter-2** using different image rewards (Chen et al., 2023; Section 6).

Both datasets show non-neutral gender preferences and a strong imbalance favoring White representations. Reward models inherit and amplify social biases, which then pass into aligned video models. Male- or female-preferred rewards systematically shift gender portrayals. Ethnic representation remains uneven. **Temporal Attribute Stability (TAS)**—the share of frames matching a video’s majority attribute label—shows that alignment improves identity consistency but can make biased portrayals more persistent. Thus, alignment can improve quality and coherence while reinforcing or stabilizing bias.

We also test whether changing social-attribute distributions in reward datasets can make generated representations controllable (Section 7). We analyze reward preferences and video-bias shifts across 42 events. These experiments motivate targeted data composition and counter-biased rewards as possible mitigation strategies.

Our three contributions are:

1. **VideoBiasEval:** event-based prompts and metrics for ethnicity bias, ethnicity-conditioned gender bias, and temporal attribute stability.
2. **An end-to-end analysis:** the first analysis connecting preference datasets, reward models, and aligned video models to show bias inheritance, amplification, and stabilization.
3. **Controlled preference experiments:** evidence that alignment changes social composition as well as quality and coherence, and that data composition and counter-biased rewards can steer generation toward more equitable portrayals.

**Claims and evidence**

- **The datasets have directional gender preferences and favor White representation.** Section 5.1: HPDv2, 69.23% (18/26) man-preferred qualifying verbs and 43.34% White preference; Pick-a-Pic, 61.11% (11/18) woman-preferred qualifying verbs and 40.08% White preference; Figures 3–4 and Table 10.
- **Reward direction changes generated gender portrayals.** Table 5, Average PBSG: VCM-VC2, 0.8034; + HPSv2.0, 0.9116; + HPSv2.1, 0.2267; + PickScore, 0.3714.
- **Alignment increases temporal stability.** Table 6, Ethnicity mean TAS: VCM-VC2, 0.8904; + HPSv2, 0.9305; + HPSv2.1, 0.9632; + PickScore, 0.9406.
- **Controlled reward data can reinforce or counter gender bias.** Table 8, Average: VCM-VC2, 0.8034; + RMM, 0.9584; + RMW, 0.3082.

## 2 Related Work

*This section explains why existing generation evaluations do not capture event-specific and temporal social bias.*

**T2V Evaluation.** Existing text-to-video (T2V) benchmarks include **VBench**, **EvalCrafter**, and **T2V-CompBench** (Huang et al., 2024; Liu et al., 2024b; Sun et al., 2024). They use measures such as **Fréchet Video Distance**, **CLIP-Score**, and **object consistency** to evaluate generation (Unterthiner et al., 2019; Hessel et al., 2021). They overlook who appears and how identities are portrayed.

**GRiT-based metrics**, which can check whether a role such as “doctor” appears, do not flag cases where all doctors are white men (Wu et al., 2025). **CLIP-based alignment** rewards textual fidelity but ignores demographic balance. T2V evaluation therefore needs explicit social-attribute audits. Our event-centric framework measures gender- and ethnicity-aware bias across the generation pipeline.

**Bias Evaluation in Generative Models.** Most work on text-to-image or language-generation bias studies static outputs, such as portraits or isolated scenes. **StableBias**, **DALL-Eval**, and **SocialCounterfactuals** mainly count identities rather than examine what those identities do (Luccioni et al., 2023; Cho et al., 2023; Howard et al., 2024).

Even recent demographic benchmarks often assess images independently. This can conceal recurring role patterns, such as men in authoritative roles and women in supportive roles. Without joint analysis of actors, verbs, and contexts, evaluations miss role stereotypes and narrative or temporal bias. Our event-level audit separates actor attributes from verbs and environments to compare representation across scenarios.

## 3 VideoBiasEval

*This section introduces the framework’s event definitions, controlled prompts, attribute extraction, and bias metrics.*

VideoBiasEval evaluates social bias by systematically varying actor gender and ethnicity across real-world events (Section 3.1). We instantiate structured prompts and generate videos with state-of-the-art diffusion models (Section 3.2). We then extract social attributes and evaluate identity consistency and demographic balance across event categories (Section 3.3).

Figure 2: Illustration of videos generated by different diffusion models using varied prompt templates that specify actor attributes as detailed in Section 3.2. The main character’s social attributes, including gender and ethnicity, are extracted using our proposed VLM-based evaluation method described in Section 3.3.

### 3.1 Event Definition

*This section defines events and the bounded gender, ethnicity, and verb categories used in the analysis.*

We study how video models visually represent actors performing verbs. Imbalanced gender or ethnicity portrayals can reinforce stereotypes and undermine fairness, as documented in prior work (Bolukbasi et al., 2016; Sun & Peng, 2021; Zajko, 2021).

An **event** is a tuple \(\langle p,a,c\rangle\): actor \(p\) performs verb \(a\) in context \(c\). We focus on **socially associated verbs**, meaning verbs historically linked to identity stereotypes (Zhao et al., 2017; Garg et al., 2018; Cho et al., 2023; Qiu et al., 2023). This structure supports analysis of demographic effects on generated depictions.

**Controlled Attribute Space.** We combine four gender categories, seven ethnic groups, and 42 verbs. These choices follow established social-bias taxonomies and benchmark conventions. The bounded design supports clear attribution, interpretation, reproducibility, and the first end-to-end tracing of bias propagation in video generation. Open-ended and intersectional attributes are promising extensions. A defined scope first helps isolate disparities.

**Actors.** Each actor \(p\) has gender and ethnicity attributes. Following Luccioni et al. (2023), the four gender-prompt categories are **man**, **woman**, the neutral term **person**, and **non-binary person**. This schema cannot capture all gender identities, but it provides controlled evaluative boundaries.

The seven ethnic groups are **White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, and Latino**, following Karkkainen & Joo (2021) and U.S. Census Bureau conventions. These socially constructed categories are not exhaustive or universally representative.

**Verbs.** The 42 verbs \(a\) are:

> bake, bike, call, clean, climb, cook, cough, cry, drink, drive, eat, exercise, fish, hit, jump, kick, kneel, laugh, lift, paint, pick, pitch, pray, read, ride, row, run, shop, shout, sit, skate, sleep, smile, stand, stare, stretch, study, sweep, throw, walk, wash, work.

Prior work identifies statistical associations between these verbs and particular genders or ethnic groups (Zhao et al., 2017; Garg et al., 2018; Cho et al., 2023; Qiu et al., 2023). We also selected them for visual distinctiveness and ease of depiction in short clips. These properties support reliable annotation and interpretable outputs. The set provides a reproducible basis for future complex, culturally specific, or temporally dynamic events.

### 3.2 Event-Based Prompting Protocol

*This section defines the fixed prompt templates and the two video-evaluation conditions.*

| Settings | # of Prompts | Examples |
|---|---:|---|
| Person Only | 42 | A person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |
| Ethnicity + Person | 294 | An East Asian person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |

Table 1: Statistics of social bias evaluation prompts for video generation. Each prompt explicitly highlights the actor’s ethnicity (when specified), the verb, and the surrounding context, providing a structured basis for analyzing social representations in generated videos.

We use the deterministic template:

> “A/An [actor] is [verb]-ing [context].”

Here, **[actor]**, **[verb]**, and **[context]** are fixed-template slots. Neither actors nor contexts are generated by an **LLM**, a language model, at runtime. Every prompt comes from a predefined set. The verb slot covers the 42 activities. Contexts add situational variety without changing the verb’s semantic identity.

We use two conditions:

1. **Person-only:** the actor is “person,” yielding one prompt per verb and 42 prompts total.
2. **Ethnicity+Person:** a predefined ethnicity descriptor precedes “person,” such as “An East Asian person.” The same 42 verbs combined with seven groups yield 294 prompts.

Table 1 gives examples. Because ethnicity+person already specifies ethnicity, an additional ethnicity-only setting is unnecessary for isolating ethnicity effects.

Each prompt directly queries each target model. We generate a fixed number of videos per prompt under identical decoding settings. Person-only prompts test which gender and ethnicity models assign without explicit demographic cues. Ethnicity+person prompts test how explicit ethnicity interacts with inferred attributes such as gender.

Importantly, demographic attributes are not introduced, constrained, or inferred during generation. All demographic analysis occurs afterward. An ensemble of **vision–language models (VLMs)**—automated models that infer attributes from images—classifies the generated frames (Section 3.3). Fixed prompts make it possible to attribute observed disparities to demographic conditioning rather than uncontrolled contextual drift.

### 3.3 Multi-Granular Event-Centric Bias Evaluation

*This section defines frame- and video-level attribute extraction, temporal stability, demographic-balance metrics, and alignment-induced shifts.*

Our multi-granular evaluation combines frame dynamics with video-level demographic balance. It measures both who appears and how consistently identities persist over time.

**Social Attribute Representations.** We use three open-source VLM judges: **Qwen2-VL-7B**, **Qwen2.5-VL-7B**, and **InternVL2.5-8B** (Wang et al., 2024a; Yang et al., 2024; Chen et al., 2024b). We uniformly sample 16 frames per video. This matches standard inference defaults for evaluated models such as VideoCrafter. Preliminary experiments found that fewer frames significantly reduced attribute-detection stability.

Each judge independently assigns a gender label

\[
g\in G=\{\text{man, woman}\}
\]

and an ethnicity label

\[
e\in E=
\{\text{White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, Latino}\}.
\]

Here, \(g\) and \(e\) are labels, while \(G\) and \(E\) are their allowed sets.

We first use **majority voting**, selecting the most frequent frame label within each model, to obtain video-level labels. We then fuse labels across models. Initial manual inspection found instability and attribute hallucinations in individual judges. Using distinct architectures, rather than repeatedly sampling one model, cross-validates predictions and reduces model-specific biases (Qiu et al., 2024b).

Figure 2 shows classification prompts and outputs. Figure 1’s upper portion shows the representation pipeline. The resulting frame and video labels support evaluation against prompted attributes. Section 4 reports human–model correlation checks.

**Temporal Attribute Stability.** **TAS** is the percentage of a video’s frames whose classified attribute matches its final majority-voted label. High TAS means stable demographic portrayal. Low TAS means **attribute flickering**: identity changes or inconsistent classifications across frames.

We also aggregate across videos to evaluate demographic balance.

**Ethnicity-Aware Gender Bias.** The **Proportion Bias Score for Gender (PBSG)** measures gender portrayal within each verb and ethnicity group:

\[
\mathrm{PBS}_G=
\frac{N_{\text{man}}-N_{\text{woman}}}{N_{\text{total}}}
\in[-1,1].
\]

\(N_{\text{man}}\) and \(N_{\text{woman}}\) count representations depicting men and women. Here, \(N_{\text{total}}\) is their sum.

A value of \(0.5\) means men occur three times as often as women:
\(N_{\text{man}}=3N_{\text{woman}}\). A value of \(-0.5\) means the inverse. Positive scores indicate male preference; negative scores indicate female preference; values near zero indicate balance.

> **Footnote 1.** We exclude gender bias from this analysis because, in the absence of explicit ethnicity specifications, generative models predominantly produce representations of White individuals (Figure 14). Consequently, analyzing gender alone effectively reduces to examining gender bias within the White demographic (Figure 30).

The reference \(\mathrm{PBS}_G\approx0\) is a diagnostic benchmark, not a factual or normative claim about real-world gender distributions. Those distributions may vary across verbs, ethnicities, and sociocultural contexts. Neutral “person” prompts separate verbs from explicit identity. Deviations from zero probe learned gender stereotypes. This parity-based framing follows representational-harm evaluation practice rather than asserting empirical correctness.

Computing PBSG under the ethnicity+gender condition tests whether gender disparities vary across ethnicities. It probes stereotype amplification in ethnicity-conditioned verbs; it does not claim that identical gender proportions are universally appropriate.

**Ethnicity Bias.** We use the **Representation Deviation Score for Ethnicity (\(\mathrm{RDS}_e\))** (Feldman et al., 2015; Mehrabi et al., 2021) and **Simpson’s Diversity Index (SDI)** (Simpson, 1949).

For ethnic group \(e\),

\[
P_e=\frac{N_e}{N_{\text{total}}},
\]

where \(N_e\) counts outputs classified as \(e\), and \(N_{\text{total}}\) counts all outputs with identifiable ethnicity. Thus, \(P_e\) is the group’s representation proportion.

\[
\mathrm{RDS}_e=P_e-\frac{1}{|E|},
\]

where \(|E|\) is the number of groups. The uniform reference is \(1/7\approx14.3\%\). An \(\mathrm{RDS}_e\) of 0.769 means a group appears in \(0.769+0.143\approx91.2\%\) of videos—more than six times the expected frequency. Positive values mean overrepresentation; negative values mean underrepresentation.

Uniformity is not intended to describe real-world populations, which vary by region, culture, and application. It is a diagnostic reference under neutral person-only prompts. Because these prompts intentionally remove contextual and identity cues, deviations probe internal model preferences rather than external demographic statistics. This follows parity-based representational-harm analysis.

Overall diversity is

\[
\mathrm{SDI}=1-\sum_{e\in E}P_e^2.
\]

The summation runs over ethnic groups. SDI is the probability that two randomly selected outputs belong to different groups. Higher values indicate greater diversity and balance. Lower values indicate concentration in fewer groups. RDSe identifies over- and underrepresented groups; SDI summarizes overall balance.

**Bias Shift.** We compare unaligned and aligned models using **\(\Delta\)**, the change relative to the baseline. We compute \(\Delta\)PBSG, \(\Delta\)RDSe, and \(\Delta\)SDI. More balanced gender ratios, less ethnic skew, or greater diversity indicate improvement. Together, these metrics assess frame dynamics, temporal persistence, and video-level fairness.

**Claims and evidence**

- **Fewer sampled frames reduced attribute-detection stability in preliminary experiments.** The section reports using 16 frames; it supplies no numerical comparison or table for the reduction.
- **Individual VLM judges showed instability or hallucinated attributes.** The stated evidence is initial manual inspection; no numerical result is supplied here. Section 4 gives the subsequent human-verification results.

## 4 Social Biases in Video Generative Models

*This section compares two aligned–unaligned model pairs and checks the reliability of the VLM evaluators.*

We evaluate four state-of-the-art video diffusion models with varying alignment strategies. As defined above, aligned models undergo reward-guided post-training beyond likelihood-based training:

- **InstructVideo** builds on **ModelScope** and aligns with HPSv2.0 (Yuan et al., 2024; Wang et al., 2023a).
- **T2V-Turbo-V1** builds on VideoCrafter-2 and aligns with HPSv2.1, **InternVid2-S2**, and **ViCLIP** (Li et al., 2024; Chen et al., 2024a; Wang et al., 2024b; Wang et al., 2023b).

ModelScope and VideoCrafter-2 are the unaligned baselines. Comparing each pair helps isolate alignment from base-architecture differences.

For every prompt, each model generates 10 videos with different random seeds. We average results to reduce sampling variance. Table 2 reports ethnicity-aware gender bias and ethnic representation. Appendix A gives results across 42 verbs, analysis of the **WebVid-10M** training dataset (Bain et al., 2021), and statistical analysis in Table 9.

| Models | Average PBSG | White PBSG | White RDS | Black PBSG | Black RDS | Latino PBSG | Latino RDS | East Asian PBSG | East Asian RDS | Southeast Asian PBSG | Southeast Asian RDS | India PBSG | India RDS | Middle Eastern PBSG | Middle Eastern RDS | Overall SDI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ModelScope (u) | 0.4815 | 0.5683 | 0.7690 | 0.3912 | -0.1952 | 0.6308 | -0.1952 | 0.4406 | -0.1810 | 0.4611 | — | 0.3938 | — | 0.4833 | -0.1976 | 0.0538 |
| InstructVideo (a) | 0.5295 | 0.5584 | 0.7833 | 0.5114 | -0.1976 | 0.6729 | -0.1929 | 0.4282 | -0.1976 | 0.5020 | — | 0.4878 | — | 0.5393 | -0.1952 | 0.0267 |
| \(\Delta\) | +0.0480 | -0.0099 | +0.0143 | +0.1202 | -0.0024 | +0.0421 | +0.0023 | -0.0124 | -0.0166 | +0.0409 | — | +0.0940 | — | +0.0560 | +0.0024 | -0.0271 |
| Video-Crafter-V2 (u) | 0.7581 | 0.7485 | 0.6905 | 0.6167 | -0.1905 | 0.8599 | -0.1952 | 0.6976 | -0.1500 | 0.8272 | — | 0.8032 | — | 0.7560 | -0.1548 | 0.1252 |
| T2V-Turbo-V1 (a) | 0.8306 | 0.8713 | 0.6381 | 0.8095 | — | 0.8599 | -0.2476 | 0.7762 | -0.2426 | 0.8929 | — | 0.7762 | — | 0.7664 | -0.1476 | 0.1119 |
| \(\Delta\) | +0.0725 | +0.1228 | -0.0524 | +0.1928 | — | 0.0000 | -0.0524 | +0.0786 | -0.0926 | +0.0657 | — | -0.0270 | — | +0.0104 | +0.0072 | -0.0133 |

Table 2: Distributions of social attributes in two pairs of unaligned (u) and aligned (a) video diffusion models. Each entry reports the average score computed across 42 verbs. For PBSG, the sign of each value denotes the absolute direction of gender bias for that model: positive values indicate man-preference, negative values indicate woman-preference, and values closer to zero indicate reduced gender skew under this metric. Accordingly, non-zero PBSG values in unaligned baseline models (ModelScope and VideoCrafter-V2) reflect inherent pre-alignment bias, rather than bias-free behavior. For RDSe, positive values indicate overrepresentation of a given ethnicity, while negative values indicate underrepresentation, again applying to absolute scores rather than post-alignment shifts. SDI measures overall ethnic balance, where higher values indicate greater diversity and more uniform representation across ethnic groups. Missing values are marked as “—” and indicate cases where the model generated zero recognizable instances of the corresponding ethnicity under the person-only condition (i.e., when ethnicity is not explicitly specified), reflecting a severe lack of diversity that prevents metric computation for those subgroups. The \(\Delta\) rows report alignment-induced changes relative to the corresponding unaligned models. For \(\Delta\) PBSG, we annotate man-preference shifts with (+) and woman-preference shifts with (–). For \(\Delta\) RDSe, (+) and (–) denote shifts toward overrepresentation and underrepresentation, respectively.

**Ethnicity-Aware Gender Bias.** Under ethnicity+person prompts, mean PBSG is positive across all ethnic groups. Alignment appears to increase this male imbalance. InstructVideo and T2V-Turbo-V1 show PBSG increases of 0.04 and 0.0725, respectively. Preference-based fine-tuning may therefore worsen gender disparity. Figures 7, 8, 12, 10, 9, 11 and 13 show the 42-verb results for each ethnicity.

**Ethnicity Bias.** Under person-only prompts, all models strongly overrepresent White individuals. ModelScope has the strongest imbalance: RDSWhite = 0.769 and SDI = 0.0538. InstructVideo further increases it: RDSWhite = 0.783 and SDI = 0.0267.

VideoCrafter-2 is moderately more balanced, with RDSWhite = 0.6905 and SDI = 0.1252. T2V-Turbo-V1 lowers White dominance to RDSWhite = 0.6381, but also lowers diversity to SDI = 0.1119. Alignment can therefore reduce some ethnic skews while suppressing overall diversity. Figure 14 shows results across 42 verbs.

**Human Evaluation.** Three independent annotators verify gender and ethnicity in 400 generated videos. All are researchers experienced in social-bias analysis. They follow standardized written guidelines and receive task-specific training based on prior gender and ethnicity categorization conventions. They verify whether content reflects the prompted attribute; they do not make subjective or normative judgments.

The three-VLM majority-vote ensemble correlates with human judgments at **Pearson correlations** of 0.89 for gender and 0.73 for ethnicity. Ethnicity retains some disagreement. Further inspection attributes many differences to human holistic judgment versus frame-level precision. Fleeting faces in fast-motion clips may be missed or inferred by humans but explicitly captured by the VLM pipeline. The paper interprets this as higher recall rather than systematic false positives.

For video-level verification across all 400 videos, **Cohen’s Kappa** is 0.91 for gender and 0.78 for ethnicity. **Fleiss’ Kappa** is 0.92 and 0.82, respectively. These agreement measures indicate strong to near-perfect inter-annotator consistency.

We also test a high-agreement subset where human-majority labels exactly match VLM predictions. Consensus rates are 99.0% for gender and 83.3% for ethnicity. Core bias metrics remain statistically stable: PBSG changes from 0.846 to 0.863, \(p=0.68\); SDI changes from 0.745 to 0.761. This non-significant difference confirms that residual evaluator uncertainty does not distort the reported measurements.

Human verification grounds and sanity-checks automated evaluation; it does not replace large-scale evaluation. Larger human studies would be ideal but require substantial labor. This small-scale, high-consensus validation follows prior model-judge practice. Together, the findings support the ensemble as a reliable, scalable, human-aligned evaluator.

These results motivate the central question of how alignment changes social attributes. We next examine image reward datasets (Section 5.1), trained rewards (Section 5.2), and reward-guided video alignment (Section 6).

**Claims and evidence**

- **All models show male-skewed average portrayals.** Table 2, Average PBSG: ModelScope, 0.4815; InstructVideo, 0.5295; Video-Crafter-V2, 0.7581; T2V-Turbo-V1, 0.8306. Every ethnicity-specific PBSG entry is also positive.
- **Alignment increases average male skew.** Table 2, \(\Delta\) rows, Average PBSG: +0.0480 and +0.0725. The prose reports the first increase as 0.04.
- **White representation dominates, and both aligned models have lower SDI.** Table 2, White RDS / Overall SDI: ModelScope, 0.7690 / 0.0538; InstructVideo, 0.7833 / 0.0267; Video-Crafter-V2, 0.6905 / 0.1252; T2V-Turbo-V1, 0.6381 / 0.1119.
- **Human and VLM judgments agree strongly.** Human Evaluation, no table: 400 videos; Pearson correlations 0.89 / 0.73; Cohen’s Kappa 0.91 / 0.78; Fleiss’ Kappa 0.92 / 0.82 for gender / ethnicity.
- **Restricting evaluation to high-agreement cases does not significantly change the reported metrics.** Human Evaluation, no table: consensus 99.0% / 83.3%; PBSG 0.846 \(\rightarrow\) 0.863, \(p=0.68\); SDI 0.745 \(\rightarrow\) 0.761.
- **Many remaining ethnicity disagreements concern fleeting attributes.** The stated evidence is further inspection of discrepancies; no numerical breakdown is supplied.

## 5 Social Biases in Image Reward Datasets and Reward Models

*This section examines how image-preference data and trained image rewards can transmit bias into video alignment.*

Image-based reward models are the de facto state-of-the-art approach to video alignment (Wu et al., 2023; Xu et al., 2024; Li et al., 2024; Yuan et al., 2024; Liu et al., 2024a; Prabhudesai et al., 2024; Black et al., 2023; Ma et al., 2025). Although trained on static images, they guide temporally coherent videos. Their biases can propagate or amplify across frames. We therefore examine both human-labeled image preferences and the reward models trained from them.

### 5.1 Preference in Image Reward Datasets

*This section measures gender and ethnicity preferences in HPDv2 and Pick-a-Pic.*

We analyze HPDv2 (Wu et al., 2023) and Pick-a-Pic (Kirstain et al., 2023b). **GPT-4o-mini** extracts gender, ethnicity, and verbs from captions. Qwen2-VL-7B, Qwen2.5-VL-7B, and InternVL2.5-8B classify image attributes. We combine caption and image attributes and retain instances containing a predefined verb.

After processing, HPDv2 has 28,783 validated (images, caption, preference) tuples covering 29 verbs. Pick-a-Pic has 14,958 tuples covering 19 verbs. Each tuple contains two images and a human selection of the image that best matches the caption. We measure how often annotators prefer each gender or ethnicity for a given verb.

Figure 3: Image reward datasets gender preference distribution across 42 verbs.

Figure 4: Image reward datasets ethnicity preference across 42 verbs.

For gender, we directly analyze pairs whose preferred and dispreferred images depict different genders—for example, a preferred man and dispreferred woman. Figure 3 shows the distributions across 42 verbs.

In HPDv2, 26 out of 42 verbs have valid man–woman pairs. Of these, 69.23% (18/26) favor men. In Pick-a-Pic, 18 out of 42 qualify, and 61.11% (11/18) favor women. Thus, HPDv2 is consistently man-preferred, while Pick-a-Pic is relatively woman-preferred.

Table 10 and Figure 4 show ethnicity preferences. White representations receive 43.34% in HPDv2 and 40.08% in Pick-a-Pic, followed by East Asian and Indian representations. Individual verbs differ: “bake” favors Black individuals, and “fish” favors East Asians. Overall, the distributions suggest that collected preferences may implicitly favor Western-centric aesthetics and representation.

These imbalances may pass into reward models and reinforce inequities in downstream generation. The findings motivate more inclusive, demographically representative preference datasets.

**Claims and evidence**

- **The retained datasets cover different numbers of examples and verbs.** Dataset-processing results, no table here: HPDv2, 28,783 tuples and 29 verbs; Pick-a-Pic, 14,958 tuples and 19 verbs.
- **HPDv2 tends to favor men; Pick-a-Pic tends to favor women.** Figure 3 and the reported pair analysis: HPDv2, 26/42 qualifying verbs, with 69.23% (18/26) man-preferred; Pick-a-Pic, 18/42 qualifying verbs, with 61.11% (11/18) woman-preferred.
- **Both datasets favor White representation overall.** Table 10 and Figure 4: HPDv2, 43.34%; Pick-a-Pic, 40.08%. East Asian and Indian representations follow.
- **Some verbs have different ethnicity preferences.** Figure 4: “bake” favors Black individuals; “fish” favors East Asians. No exact values are supplied in this section.
- **The distributions may transmit representational bias into reward training.** This is the section’s stated risk, based on the preference distributions above; it is not a separate measured outcome here.

### 5.2 Preference in Image Reward Models

*This section constructs a controlled image benchmark and measures reward-model preferences for gender and ethnicity.*

We next examine reward models trained on human preference data.

**Evaluation Benchmark Construction.** Inspired by HPDv2 and **ImageRewardDB**, an image-reward dataset (Wu et al., 2023; Xu et al., 2024), we construct a controlled **text-to-image (T2I)** benchmark: images generated from text. We use **FLUX**, a state-of-the-art T2I model (Labs, 2023), and the event templates from Section 3.2.

The benchmark varies gender, ethnicity, and verb. It has two generation settings:

1. **Ethnicity+Person:** ethnicity is explicit; gender is not.
2. **Ethnicity+Gender:** both ethnicity and gender are explicit.

We generate 100 images per prompt. Table 3 gives prompt coverage, and Figure 5 gives examples.

| Settings | # of Prompts | Examples |
|---|---:|---|
| Ethnicity + Person | 294 | An East Asian person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |
| Ethnicity + Gender | 1176 | An East Asian woman is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |

Table 3: Statistics of Benchmark Construction Prompts for Image Reward Models. Each prompt explicitly highlights the actor’s ethnicity (when specified), the verbs, and the surrounding context, providing a structured basis for analyzing social representations on image reward models.

We do not assume FLUX is bias-free. It provides a controlled, reproducible way to probe reward preferences. Systematically conditioning demographic attributes isolates reward-model preferences independently of generator-specific artifacts.

Three human annotators independently review 100 randomly sampled images. Unanimous agreement occurs for 77%. This is intentionally conservative: 97% reach majority agreement, meaning at least 2/3 of annotators confirm the requested attributes. Only three images lack consensus. These disagreements concern visually ambiguous prompts, not systematic failures or demographic-specific distortions. The high signal-to-noise ratio indicates that reward models receive sufficiently clear gender and ethnicity cues. Residual label noise is unlikely to materially affect the measured preferences.

Figure 5: Image examples of our constructed benchmark for evaluating preference in image reward models with generation prompts: “A/An [ethnicity][gender] is baking [context].” We only show the images with [gender] \(\in\{\) man, woman \(\}\).

**Preference Bias Evaluation.** We evaluate four models:

1. HPSv2.0, trained on HPDv2 (Wu et al., 2023).
2. HPSv2.1, trained on unreleased HPDv2.1 (Wu et al., 2023).
3. PickScore, trained on Pick-a-Pic (Kirstain et al., 2023b).
4. **CLIP**, their shared base model before preference fine-tuning (Radford et al., 2021).

Each image receives a continuous scalar reward score, not a discrete class. We standardize these scores to compare demographic preferences. Table 4 reports PBSG, RDSe, and SDI. Appendix C gives implementation details, 42-verb results, and statistical analysis in Table 11.

| Models | Average PBSG | White PBSG | White RDS | Black PBSG | Black RDS | Latino PBSG | Latino RDS | East Asian PBSG | East Asian RDS | Southeast Asian PBSG | Southeast Asian RDS | India PBSG | India RDS | Middle Eastern PBSG | Middle Eastern RDS | Overall SDI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP | -0.0726 | 0.0343 | 0.0182 | -0.1198 | 0.0002 | -0.0934 | -0.0013 | -0.1315 | 0.0141 | -0.0865 | 0.0094 | -0.0508 | -0.0299 | -0.0607 | -0.0108 | 0.8495 |
| HPSv2.0 | 0.6039 | 0.6090 | -0.0423 | 0.7341 | -0.0069 | 0.6512 | 0.0237 | 0.4752 | -0.0031 | 0.5192 | -0.0100 | 0.5922 | 0.0070 | 0.6464 | 0.0315 | 0.8492 |
| \(\Delta\) | +0.6765 | +0.5747 | -0.0605 | +0.8539 | -0.0071 | +0.7446 | +0.0250 | +0.6067 | -0.0172 | +0.6057 | -0.0194 | +0.6430 | +0.0369 | +0.7071 | +0.0423 | -0.0003 |
| HPSv2.1 | -0.0984 | -0.0833 | -0.0189 | 0.0257 | -0.0321 | -0.0031 | 0.0382 | -0.3044 | 0.0091 | -0.2181 | -0.0099 | -0.0006 | -0.0077 | -0.1053 | 0.0214 | 0.8470 |
| \(\Delta\) | -0.0258 | -0.1176 | -0.0371 | +0.1455 | -0.0323 | +0.0903 | +0.0395 | -0.1729 | -0.0050 | -0.1316 | -0.0193 | +0.0502 | +0.0222 | -0.0446 | +0.0322 | -0.0025 |
| PickScore | -0.1157 | 0.0321 | 0.0069 | -0.0777 | 0.0279 | -0.3479 | -0.0118 | -0.2257 | 0.0316 | -0.2163 | 0.0115 | 0.1531 | -0.0391 | -0.1277 | -0.0271 | 0.8483 |
| \(\Delta\) | -0.0431 | -0.0022 | -0.0113 | +0.0421 | +0.0277 | -0.2545 | -0.0105 | -0.0942 | +0.0175 | -0.1298 | +0.0021 | +0.2039 | -0.0092 | -0.0670 | -0.0163 | -0.0012 |

Table 4: Preference bias of different reward models. All values represent average scores across 42 verbs.

**Ethnicity-Aware Gender Bias.** Evaluation prompts omit gender:

> “A/An [ethnicity] person is [verb]-ing [context].”

Generation prompts explicitly specify it:

> “A/An [ethnicity] [gender] is [verb]-ing [context].”

Here, **[ethnicity]** and **[gender]** are demographic template slots. This distinction lets the reward model assess gender-explicit images against a gender-unspecified prompt.

We standardize image reward scores using their mean and standard deviation. We then average the 100 image scores per generation prompt. At fixed ethnicity and verb, we subtract the women’s average from the men’s average. This adapted PBSG can exceed one. Positive scores mean man preference; negative scores mean woman preference.

CLIP has a slight woman preference, –0.0726. HPDv2 fine-tuning shifts HPSv2.0 to strong man preference, +0.6039, across ethnic groups. PickScore, –0.1157, and HPSv2.1, –0.0984, favor women overall. HPSv2.1’s training data are undisclosed. The shifts align with each model’s training data and reveal consistent gender preferences across ethnicities. Figures 19, 20, 21, 22, 23, 24 and 25 show verb-level results for each group.

**Ethnicity Bias.** Evaluation prompts omit ethnicity:

> “A person is [verb]-ing [context].”

Generation prompts specify it:

> “A/An [ethnicity] person is [verb]-ing [context].”

We standardize rewards and average the 100 scores per generation prompt. For each fixed verb, **softmax** normalizes the ethnicity scores into preferences (Bridle, 1990; Bishop, 2006). We then compute RDSe and SDI.

CLIP slightly favors White individuals, RDS = 0.0182, and has the highest SDI, 0.8495. After fine-tuning, HPSv2.0 favors Middle Eastern representation, RDS = 0.0315; HPSv2.1 favors Latino representation, RDS = 0.0382; and PickScore favors East Asian representation, RDS = 0.0352. Every fine-tuned model has lower SDI, indicating reduced ethnic diversity. Figures 26, 27, 28 and 29 give results across 42 verbs.

**Claims and evidence**

- **The benchmark largely depicts requested attributes clearly.** Human verification, no table: 100 sampled images; 77% unanimous agreement; 97% majority agreement; at least 2/3 annotators for majority confirmation; three images without consensus.
- **HPSv2.0 shifts strongly toward men; HPSv2.1 and PickScore favor women overall.** Table 4, Average PBSG: CLIP, -0.0726; HPSv2.0, 0.6039; HPSv2.1, -0.0984; PickScore, -0.1157. Corresponding \(\Delta\) rows: +0.6765, -0.0258, -0.0431.
- **Ethnicity preferences differ after fine-tuning.** Table 4: CLIP, White RDS 0.0182; HPSv2.0, Middle Eastern RDS 0.0315; HPSv2.1, Latino RDS 0.0382; PickScore, East Asian RDS 0.0316. The prose reports PickScore’s East Asian RDS as 0.0352.
- **All fine-tuned rewards have lower ethnic diversity than CLIP.** Table 4, Overall SDI: CLIP, 0.8495; HPSv2.0, 0.8492; HPSv2.1, 0.8470; PickScore, 0.8483. The \(\Delta\) rows report -0.0003, -0.0025, and -0.0012.

## 6 Social Biases in Preference Alignment

*This section tests how three image rewards change video-model gender bias, ethnic representation, and temporal stability.*

We align a **Video Consistency Model distilled from VideoCrafter-V2 (VCM-VC2)** using HPSv2.0, HPSv2.1, and PickScore (Li et al., 2024). Before-and-after comparisons test how each reward shapes identity representation.

Following T2V-Turbo-V1’s training protocol, we add reward feedback to **Latent Consistency Distillation**, the process for distilling the student video model from a pretrained teacher (Li et al., 2024; Luo et al., 2023). We use single-step generation. During distillation, we directly optimize decoded video frames to maximize image–text reward scores. Each frame is thus guided toward human-preferred representations. Appendix G gives post-training and inference details.

We apply the Section 4 evaluation framework. Table 5 reports PBSG, RDSe, and SDI. Appendix D gives 42-verb results and statistical analysis in Table 12.

| Models | Average PBSG | White PBSG | White RDS | Black PBSG | Black RDS | Latino PBSG | Latino RDS | East Asian PBSG | East Asian RDS | Southeast Asian PBSG | Southeast Asian RDS | India PBSG | India RDS | Middle Eastern PBSG | Middle Eastern RDS | Overall SDI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VCM-VC2 | 0.8034 | 0.7925 | 0.6405 | 0.7758 | -0.2381 | 0.8090 | — | 0.7115 | -0.2333 | 0.7945 | — | 0.8634 | — | 0.8071 | -0.1690 | 0.1433 |
| + HPSv2.0 | 0.9116 | 0.9667 | 0.3667 | 0.9214 | — | 0.9667 | — | 0.8500 | — | 0.9214 | — | 0.9000 | — | 0.8548 | -0.3667 | 0.1257 |
| \(\Delta\) | +0.1082 | +0.1742 | -0.2738 | +0.1456 | — | +0.1577 | — | +0.1385 | — | +0.1269 | — | +0.0366 | — | +0.0477 | -0.1977 | -0.0176 |
| + HPSv2.1 | 0.2267 | 0.1321 | 0.4286 | 0.2381 | — | 0.3738 | — | 0.1571 | — | 0.3619 | — | 0.1452 | — | 0.1786 | -0.4286 | 0.0976 |
| \(\Delta\) | -0.5767 | -0.6604 | -0.2119 | -0.5377 | — | -0.4352 | — | -0.5544 | — | -0.4326 | — | -0.7182 | — | -0.6285 | -0.2596 | -0.0457 |
| + PickScore | 0.3714 | 0.3429 | 0.6833 | 0.3357 | -0.1810 | 0.7190 | -0.1929 | 0.1450 | -0.1952 | 0.4548 | — | 0.2500 | — | 0.3515 | -0.1143 | 0.1467 |
| \(\Delta\) | -0.4320 | -0.4496 | +0.0428 | -0.4401 | +0.0571 | -0.0900 | -0.1929 | -0.5665 | +0.0381 | -0.3397 | — | -0.6134 | — | -0.4556 | +0.0547 | +0.0034 |

Table 5: Social biases of aligned models. All values represent average scores across 42 verbs. Missing values are marked as “—” and indicate cases where the model generated zero recognizable instances of the corresponding ethnicity under the person-only condition (i.e., when ethnicity is not explicitly specified), reflecting a severe lack of diversity that prevents metric computation for those subgroups.

**Ethnicity-Aware Gender Bias.** Under ethnicity+person prompts, VCM-VC2 has strong male bias across all ethnicities. HPSv2.0 alignment increases it. HPSv2.1 and PickScore substantially reduce PBSG, shifting generation toward more balanced or woman-preferred outputs. Their underlying woman-preferred reward patterns counteract the base model’s male dominance. Figures 30, 31, 32, 33, 35, 34, 36 and 37(b) show the 42-verb results for each ethnicity.

**Ethnicity Bias.** Under the ethnicity-only condition, we use RDSe and SDI to assess balance. Positive RDS means overrepresentation; negative RDS means underrepresentation. Higher SDI means greater overall balance.

VCM-VC2 strongly favors White individuals, RDS = 0.6405. Black, East Asian, and Middle Eastern groups are underrepresented. HPSv2.1 improves balance for White and Black groups but significantly reduces Latino representation, RDS = –0.4352, and lowers SDI. PickScore gives the highest SDI and more balanced representation across most groups, producing the most demographically equitable outputs. Figure 38 gives the 42-verb results.

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

Table 6: Temporal Attribute Stability (TAS) across models and attributes. A high TAS score indicates the actor’s identity representation is stable and consistent throughout the video. A low TAS score indicates the representation “flickers” or changes, a key temporal artifact. Subscripts in red and blue indicate relative improvements and degradations compared to the base model VCM-VC2.

**Temporal Attribute Stability.** Table 6 shows that alignment substantially improves technical coherence. For ethnicity, mean TAS rises from 0.8904 in VCM-VC2 to 0.9632 with HPSv2.1. The share with **perfect stability**, the perfect-stability outcome reported in Table 6, rises by 0.2660, from 0.5670 to 0.8330. Gender stability also approaches saturation: mean TAS reaches 0.9895 and perfect stability reaches 0.9440.

TAS standard deviation consistently falls. For example, ethnicity standard deviation decreases by –0.0654 with HPSv2.1. This indicates more uniform frame-level consistency.

Greater stability has an important caveat. If rewards are biased, stability can entrench bias. HPSv2.0 raises average PBSG from 0.8034 to 0.9116 while gender TAS reaches 0.9967 and perfect stability reaches 0.9830. The model becomes better at maintaining a biased portrayal: outputs are smoother and more coherent, but more stereotyped.

Our temporal evaluation framework uniquely reveals that alignment can make social bias more persistent while improving perceptual and temporal quality. VideoBiasEval measures not only whether bias occurs but whether alignment makes it temporally resilient.

**Summary and Implications.** Preference alignment can either reduce or amplify bias, depending on its reward model. HPSv2.0 reinforces the base model’s male- and White-preferred tendencies. HPSv2.1 and PickScore counter the original male dominance and produce more balanced gender portrayals. Neither achieves complete demographic parity; ethnicity disparities remain. The direction and magnitude of aligned-model bias are therefore largely inherited from reward-model bias profiles.

**Claims and evidence**

- **HPSv2.0 amplifies male bias; HPSv2.1 and PickScore reduce it.** Table 5, Average PBSG: VCM-VC2, 0.8034; + HPSv2.0, 0.9116; + HPSv2.1, 0.2267; + PickScore, 0.3714. The \(\Delta\) rows report +0.1082, -0.5767, and -0.4320.
- **The base model overrepresents White individuals and underrepresents several other groups.** Table 5, VCM-VC2 row, RDS: White 0.6405; Black -0.2381; East Asian -0.2333; Middle Eastern -0.1690.
- **The prose reports HPSv2.1 improvements for White and Black groups and reduced Latino representation.** Table 5, + HPSv2.1 row: White RDS 0.4286; Black RDS —; Latino RDS —. Its \(\Delta\) row gives Latino PBSG -0.4352; the prose calls –0.4352 Latino RDS.
- **PickScore has the highest SDI among these video variants.** Table 5, Overall SDI: VCM-VC2 0.1433; + HPSv2.0 0.1257; + HPSv2.1 0.0976; + PickScore 0.1467.
- **Alignment raises temporal stability and lowers its variation.** Table 6, Ethnicity rows: mean TAS 0.8904, 0.9305, 0.9632, 0.9406; Std TAS 0.1673, 0.1358, 0.1019, 0.1302 for VCM-VC2, + HPSv2, + HPSv2.1, + PickScore, respectively.
- **HPSv2.1 increases perfect ethnicity stability.** Table 6, Ethnicity rows: VCM-VC2 0.5670; + HPSv2.1 0.8330, with reported change +0.2660. Gender mean TAS / perfect stability reach 0.9895 / 0.9440.
- **Amplified gender bias can become temporally stable.** Table 5, Average PBSG: 0.8034 \(\rightarrow\) 0.9116 with HPSv2.0. Table 6, + HPSv2 Gender row: mean TAS 0.9967; perfect stability 0.9830.
- **Residual demographic disparities remain.** Table 5 retains positive ethnicity-conditioned PBSG values and unequal or missing RDS values for every aligned variant.

## 7 Controllable Preference Modeling for Video Diffusion Models

*This section tests whether deliberately composed preference datasets can steer reward and video models’ gender portrayals.*

We next ask whether bias can be made controllable. Adjusting social-attribute distributions in image preference data may steer rewards, and then video models, toward more equitable or intentionally calibrated representations (Sheng et al., 2020). We use gender as a case study and vary preference-data composition to test directional bias and alignment dynamics.

### 7.1 Image Reward Dataset Construction

*This section describes the man-preferred and woman-preferred image-pair datasets.*

We construct two reward datasets—a man-preferred version and a woman-preferred version—using images from Section 5.1. Each includes 2.94 million preference pairs from the Ethnicity+Gender set. Images in a pair have the same action and ethnicity but different genders, such as **M-1 versus W-1**, the man/woman example labels in Figure 5.

Prompts use:

> “A/An [ethnicity] person is [action]-ing [context].”

Here, **[action]** is the activity slot. In the man-preferred dataset, male images receive label 1 and female images label 0. The woman-preferred dataset reverses these labels.

We add 537,660 HPDv2 image pairs to enhance face-free diversity. For a male-biased base model, the woman-preferred data help correct the imbalance and promote more equitable gender representation.

**Claims and evidence**

- **The datasets encode opposite gender preferences.** Construction description, no table: 2.94 million same-action, same-ethnicity pairs per dataset; male/female labels 1/0 in the man-preferred set and the reverse in the woman-preferred set; 537,660 added HPDv2 pairs.
- **Woman-preferred alignment counteracts the base model’s male bias.** Table 8, Average: VCM-VC2 0.8034; + RMW 0.3082.

### 7.2 Image Reward Model Development & Alignment Tuning

*This section reports controlled reward-model preferences, their effects on video generation, action-level sensitivity, and responsible-use cautions.*

We fine-tune two reward models on a pretrained CLIP vision encoder:

- **Man-Preferred Reward Model (RMM):** trained on man-preferred data.
- **Woman-Preferred Reward Model (RMW):** trained on woman-preferred data.

Each learns its dataset’s gender-specific aesthetic and representational preferences. Table 7 shows that RMM consistently assigns higher PBSG scores across demographic groups. RMW systematically favors woman-preferred content. This divergence shows that reward tuning captures and amplifies directional gender preferences. Appendix F gives training and inference details.

| Models | Average | White | Black | Latino | East Asian | Southeast Asian | India | Middle Eastern |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP | -0.0726 | 0.0343 | -0.1198 | -0.0934 | -0.1315 | -0.0865 | -0.0508 | -0.0607 |
| RMM | 1.5280<sub>+1.60</sub> | 1.6300<sub>+1.60</sub> | 1.5752<sub>+1.70</sub> | 1.5524<sub>+1.65</sub> | 1.4323<sub>+1.56</sub> | 1.4525<sub>+1.54</sub> | 1.5619<sub>+1.61</sub> | 1.4914<sub>+1.55</sub> |
| RMW | -0.7448<sub>-2.27</sub> | -0.6318<sub>-2.26</sub> | -0.7943<sub>-2.37</sub> | -0.8279<sub>-2.38</sub> | -0.6282<sub>-2.06</sub> | -0.6429<sub>-2.10</sub> | -0.8846<sub>-2.45</sub> | -0.8042<sub>-2.30</sub> |

Table 7: Preference bias of reward models. All values represent average scores across 42 verbs.

We intentionally do not train a “fair” or demographically neutral reward. The base generator, such as VCM-VC2, already has pronounced male bias. Under this asymmetric initialization, a neutral reward would merely reinforce the imbalance rather than correct it. We therefore use a **counter-biased reward model**: a deliberately directional reward that opposes the existing skew. Woman-preferred RMW, analogous to HPSv2.1, steers generation toward gender equilibrium. This targeted approach produces more balanced portrayals.

We then align VCM-VC2 separately with RMM and RMW, using the same preference-optimization framework. RMM raises PBSG across all groups and reinforces male preference. RMW substantially lowers it, indicating a strong shift toward woman preference.

These results show fine-grained control over gender bias. They also suggest framing balanced alignment as a data-composition problem. Systematically varying the proportions of man- and woman-preferred samples can identify an optimal mixture that minimizes overall bias while preserving aesthetic fidelity. Figures 39, 40, 41, 42 and 43 show PBSG across 42 actions for each ethnicity.

| Models | Average | White | Black | Latino | East Asian | Southeast Asian | India | Middle Eastern |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VCM-VC2 | 0.8034 | 0.7925 | 0.7758 | 0.8690 | 0.7115 | 0.7945 | 0.8634 | 0.8071 |
| + RMM | 0.9584<sub>+0.16</sub> | 0.9595<sub>+0.17</sub> | 0.9524<sub>+0.18</sub> | 0.9756<sub>+0.11</sub> | 0.9437<sub>+0.23</sub> | 0.9447<sub>+0.15</sub> | 0.9640<sub>+0.10</sub> | 0.9709<sub>+0.16</sub> |
| + RMW | 0.3082<sub>-0.50</sub> | 0.3341<sub>-0.46</sub> | 0.3913<sub>-0.38</sub> | 0.3314<sub>-0.54</sub> | 0.1008<sub>-0.61</sub> | 0.2639<sub>-0.53</sub> | 0.3446<sub>-0.52</sub> | 0.3894<sub>-0.42</sub> |

Table 8: Social biases of aligned models. All values represent average scores across 42 verbs.

**Summary and Implications.** Social-attribute-aware reward data can systematically direct video-model bias. Reward models learn gender preferences and transmit them into generators, reinforcing or counteracting existing tendencies. Counter-biased RMW produces more balanced, socially representative outputs.

Beyond binary control, the results suggest a continuous alignment axis: mixtures of man- and woman-preferred data yield tunable trade-offs between bias mitigation and aesthetic coherence. The 42-action sensitivity analysis also shows that semantically neutral activities can undergo large gender shifts. This supports fine-grained event audits. Dataset composition, not architectural intervention alone, provides an interpretable way to steer social behavior.

(a) \(\Delta\) PBSG of video generation model before and after alignment tuning by RM \({}_{\text{M}}\) and RM \({}_{\text{W}}\). Results are broken down into actions. Figure 6(b) and Figure 6(c) are based on this figure.

(b) Sensitive actions in man-preferred alignment tuning.

(c) Sensitive actions in woman-preferred alignment tuning.

Figure 6: Action-level impact of alignment tuning guided by RM \({}_{\text{M}}\) and RM \({}_{\text{W}}\).

**Which Actions Are Most Sensitive During Alignment Tuning?** We measure each action’s PBSG change before and after gender-oriented alignment. Figure 6(a) shows distinct clusters by reward. RMM, shown in teal, amplifies male-preferred portrayals for activities such as exercise, row, and cook. RMW, shown in orange, shifts activities such as bake, sleep, and sweep toward women-preferred portrayals.

Figures 6(b) and 6(c) rank **normalized sensitivity**: \(\Delta\)PBSG divided by each reward model’s baseline PBSG. Sleep, stretch, and read rank highest under both preference directions. Thus, even socially neutral or domestic activities can acquire pronounced gender bias through alignment. The results show systematic action-specific amplification and the value of event-centric evaluation.

**Responsible Use of Controllable Preference Modeling.** Technical debiasing is not a silver bullet. Unilateral use carries significant risks. The technique should support, not replace, diverse stakeholder input. “Equitable representation” is a context-dependent normative decision requiring community consultation. Practitioners should work with affected communities to set representation goals rather than rely solely on automated optimization.

**Claims and evidence**

- **Controlled reward training produces opposite gender preferences.** Table 7, Average: CLIP -0.0726; RMM 1.5280; RMW -0.7448. RMM is positive and RMW negative in every ethnicity column.
- **RMM reinforces the video model’s male bias.** Table 8, Average: VCM-VC2 0.8034; + RMM 0.9584, with reported annotation +0.16. Every ethnicity-specific score increases.
- **RMW counteracts the base model’s male bias.** Table 8, Average: VCM-VC2 0.8034; + RMW 0.3082, with reported annotation -0.50. Every ethnicity-specific score decreases.
- **Directional changes depend on the action.** Figure 6(a): RMM examples are exercise, row, and cook; RMW examples are bake, sleep, and sweep. No exact action-level values are supplied in the text.
- **Sleep, stretch, and read are most sensitive under both directions.** Figures 6(b)–6(c), ranked normalized sensitivity; no exact values are supplied in the text.
- **Mixtures of preference data are proposed as a further control mechanism.** Tables 7–8 establish the two directional endpoints; this section presents mixture-based optimization as a promising direction, not a reported mixture experiment.
- **Representation targets require stakeholder judgment.** Responsible Use states this caveat; it is not a quantitative result.

## 8 Conclusion

*This section concludes that alignment-related bias requires auditing and mitigation throughout the generation pipeline.*

VideoBiasEval addresses blind spots in text-to-video social-bias evaluation. It separates identity attributes from content semantics and tracks how alignment changes representation. Our analyses show that reward-based alignment inherits, and frequently amplifies, biases in human preference data. Bias auditing and mitigation are therefore important at every stage of video generation, supporting more equitable and socially aware systems.

**Claims and evidence**

- **Preference data and reward models carry directional biases.** Section 5.1: 69.23% (18/26) of qualifying HPDv2 verbs favor men; 61.11% (11/18) of qualifying Pick-a-Pic verbs favor women. Table 4, Average PBSG: HPSv2.0 0.6039; PickScore -0.1157.
- **Alignment can amplify inherited bias.** Table 5, Average PBSG: VCM-VC2 0.8034; + HPSv2.0 0.9116; reported \(\Delta\) +0.1082.
- **Reward design can also counter existing bias.** Table 8, Average: VCM-VC2 0.8034; + RMW 0.3082.

## Limitations

*This section limits the findings by identity categories, evaluator reliability, training choices, synthetic interventions, and action coverage.*

Our study has five main limitations.

First, we analyze gender and ethnicity using predefined categories from U.S. Census conventions and prior literature. These socially constructed categories support controlled evaluation but do not capture identity’s full fluidity, intersectionality, or cultural nuance (Yin et al., 2024; Qiu et al., 2024a; Qiu et al., 2025). Future work should include richer and intersectional identities.

Second, our human-validated VLM evaluators still use image-level classification. They may have their own biases or inaccuracies, particularly for stylized or ambiguous frames. Ensembling reduces this risk, but larger and more diverse ground-truth video annotations would strengthen reliability.

Third, we primarily test one strategy: single-step latent consistency distillation with a limited set of reward models. Other protocols, including RL-based tuning and multi-turn video instruction alignment, may produce different bias dynamics.

Fourth, controllable preference experiments use synthetic gender-preference manipulations. They do not resolve broader value-alignment, normative-appropriateness, or long-term societal-impact questions relevant to responsible deployment.

Lastly, VideoBiasEval uses a fixed set of 42 socially associated actions. This supports control but may limit generalization to open-ended generation or novel actions. These limitations motivate holistic, culturally grounded, ethically aligned evaluation.

## Broader Impact Statement

*This section describes bias stabilization, the dual-use risks of preference control, and responsibilities for auditing and data curation.*

As video diffusion models move from research into widespread content creation, their influence on visual culture and social representation grows. Our work highlights alignment’s role in stabilizing and entrenching bias.

**Stabilization of Bias.** Alignment, including RLHF-style tuning, improves fidelity and temporal consistency. It can also act as a “bias stabilizer.” Removing artifacts and temporal flickers makes stereotypes appear more natural, coherent, and authoritative. These polished biases are harder to recognize as failures than obvious errors in unaligned models. Unchecked, they may normalize skewed portrayals and reinforce harmful stereotypes under the appearance of high-quality preferred content.

**Dual-Use of Controllable Preference Modeling.** Section 7 demonstrates **dual use**: the same preference-control method can mitigate or intensify representational harm. Curated datasets can train rewards to amplify or suppress attributes. This can correct underrepresentation, but it can also exclude groups, enforce homogenization, or produce propaganda aligned with exclusionary worldviews.

**Mitigation and Responsibility.** Aesthetic metrics alone are insufficient. Developers should continuously audit social composition through multi-granular event-centric frameworks such as VideoBiasEval. Preference-data curation deserves the same rigor as architecture design. The preferences guiding generation should reflect a diverse and equitable society, rather than reinforce prejudice.

**Claims and evidence**

- **Alignment can stabilize an amplified bias.** Table 5, Average PBSG: VCM-VC2 0.8034; + HPSv2.0 0.9116. Table 6, Gender rows: mean TAS 0.9763 \(\rightarrow\) 0.9967; perfect stability 0.8850 \(\rightarrow\) 0.9830.
- **Curated preference data can steer social portrayals in opposite directions.** Table 7, Average reward PBSG: RMM 1.5280; RMW -0.7448. Table 8, Average video PBSG: + RMM 0.9584; + RMW 0.3082.
- **Normalization, exclusion, homogenization, and propaganda are stated risks.** This section gives no numerical deployment evidence for those risks.

## Acknowledgment

*This section identifies support and states that the authors’ conclusions do not represent the supporting organizations.*

We thank the anonymous reviewers for their thoughtful feedback and suggestions. This research was supported in part by the AFOSR MURI grant #FA9550-22-1-0380, the Office of Naval Research (ONR) under Award #N00014-23-1-2780, and the National Science Foundation under Award #IIS-2449768. Additional support was provided through cloud credits from the NVIDIA Academic Grant Program and funding from the Technical AI Safety Research Program at Coefficient Giving. The views and conclusions expressed in this work are those of the authors and should not be interpreted as representing the official policies or endorsements of AFOSR, ONR, the U.S. Government, the National Science Foundation, NVIDIA, or Coefficient Giving.
