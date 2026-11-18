**Summary**

The paper asks how preference alignment changes gender and ethnicity portrayals in video diffusion models. It introduces QX2, an event-based audit that separates actor identity from actions and contexts. The audit connects human image preferences, learned image rewards, and aligned video outputs across 42 verbs. Both preference datasets favor White representations and have directional gender preferences. Learned rewards inherit and amplify these patterns. Alignment changes generated demographic distributions and improves within-video identity stability, but it can also make stereotypes more persistent. Controlled man- and woman-preferred datasets show that preference composition can amplify or counteract gender imbalance. The paper calls for bias-aware evaluation and mitigation throughout alignment.

**Glossary**

*This glossary defines the paper’s terminology, notation, model codes, and dataset codes.*

- **Alignment; alignment tuning; preference alignment; post-training optimization** — Optimization after likelihood-based training to better match human preferences.
- **Aligned model** — A video diffusion model optimized using learned preference or aesthetic rewards.
- **Unaligned model; baseline** — A counterpart trained with likelihood objectives alone, used to isolate alignment effects.
- **Video diffusion model; generator** — A model that generates videos from natural-language prompts.
- **Human-preference learning** — Learning from human-labeled comparisons to guide model optimization.
- **Preference pair; comparison tuple** — Two images, a caption, and a human choice of the better caption match.
- **Reward model; image reward; learned reward** — A model that assigns preference scores to images or frames and guides alignment.
- **Reward-weighted fine-tuning** — A stated method for aligning a base model using rewards.
- **RLHF; RL-based tuning** — Reward-guided alignment approaches discussed as optimization methods.
- **Social bias; representational disparity** — Uneven gender or ethnicity portrayal, including who appears, how often, and in which roles.
- **Stereotype reproduction** — Learned identity associations appearing in generated activities and contexts.
- **Bias inheritance; amplification; stabilization** — Transfer from preferences to rewards and generation, strengthening of bias, and persistence across frames.
- **Perceptual quality; fidelity; coherence; controllability** — Generation properties that alignment seeks to improve.
- **Temporal consistency; attribute persistence** — Stability of depicted identity across video frames.
- **Attribute flicker** — Inconsistent or changing demographic identity within a video.
- **Demographic composition; attribute distribution** — The distribution of represented gender and ethnicity attributes.
- **Parity; uniform reference** — A diagnostic balance reference, not demographic ground truth or a universal normative target.
- **Representational harm** — The fairness framing used to motivate uniform/parity diagnostic baselines.
- **Intersectionality; intersectional groups** — Identity combinations not fully covered by the predefined attribute space.
- **Counter-biased reward** — A deliberately directional reward used to counteract an existing model imbalance.
- **Controllable preference modeling** — Manipulating preference-data composition to change reward and video portrayals.
- **Event-centric; multi-granular audit** — Evaluation organized around actor–verb–context events and multiple representation and temporal scales.
- **QX2; VideoBiasEval** — The paper’s event-prompt and multi-granular gender/ethnicity audit.
- **Event; \(\langle p,a,c\rangle\)** — An actor \(p\), performed verb \(a\), and context \(c\).
- **Actor; \(p\)** — The person performing an event.
- **Verb; action; activity; \(a\)** — The event’s performed activity.
- **Context; \(c\)** — The event’s surrounding situation.
- **Actor attributes; social attributes** — Gender and ethnicity in this audit.
- **Man; woman; person; non-binary person** — The four predefined gender-related actor categories; “person” is neutral.
- **White; Black; Indian; East Asian; Southeast Asian; Middle Eastern; Latino** — The seven predefined, socially constructed ethnic categories.
- **India** — The table label used for the Indian category.
- **bake; bike; call; clean; climb; cook; cough; cry; drink; drive; eat; exercise; fish; hit; jump; kick; kneel; laugh; lift; paint; pick; pitch; pray; read; ride; row; run; shop; shout; sit; skate; sleep; smile; stand; stare; stretch; study; sweep; throw; walk; wash; work** — The 42 curated event verbs.
- **Prompt; event template** — A fixed text instantiation of actor, verb, and context used for generation or scoring.
- **A/An; [actor]; [verb]; [action]; [context]; [ethnicity]; [gender]** — Article alternatives and substitution slots in the fixed prompt templates.
- **J7K; Person-only** — The 42-prompt condition using “person” without explicit demographic cues.
- **V9R; Ethnicity+Person** — The 294-prompt condition specifying ethnicity but not gender.
- **C4Z; Ethnicity+Gender** — The 1176-prompt image condition explicitly specifying ethnicity and gender.
- **T2V** — Text-to-video generation or evaluation.
- **T2I** — Text-to-image generation or evaluation.
- **VLM** — A model used here to classify gender and ethnicity in frames or images.
- **LLM** — The type of model explicitly not used to generate actors or contexts at runtime.
- **VLM ensemble; judges** — Independent attribute classifiers whose outputs are combined.
- **Frame majority vote; video label** — Aggregation of frame classifications into a video-level attribute label.
- **Qwen2-VL-7B; Qwen2.5-VL-7B; InternVL2.5-8B** — The three open-source attribute judges.
- **GPT-4o-mini** — The model used to extract caption gender, ethnicity, and verbs.
- **\(g\); \(G\)** — A classified gender and the output set \(\{\text{man,woman}\}\).
- **\(e\); \(E\); \(|E|\)** — An ethnic group, the seven-group output set, and its size.
- **\(N_{\text{man}}\); \(N_{\text{woman}}\)** — Counts of outputs classified as men and women.
- **\(N_{\text{total}}\)** — Men plus women for gender scoring; all identifiable-ethnicity outputs for ethnicity scoring.
- **\(N_e\); \(P_e\)** — The count and proportion of identifiable outputs classified as ethnic group \(e\).
- **L2V; PBSG; Proportion Bias Score for Gender** — The signed man–woman count difference divided by their total.
- **Adapted L2V** — Men’s minus women’s average standardized reward score at fixed ethnicity and verb; it can exceed one.
- **F6N; \(\mathrm{RDS}_e\); Representation Deviation Score for Ethnicity** — \(P_e-1/|E|\), the group’s deviation from uniform coverage.
- **W3H; SDI; Simpson’s Diversity Index** — \(1-\sum_{e\in E}P_e^2\), the probability that two random outputs belong to different groups.
- **T8Q; TAS; Temporal Attribute Stability** — The percentage of frames matching the video’s final majority-voted attribute.
- **\(\Delta\)** — A change relative to the corresponding baseline.
- **\(\Delta\mathrm{L2V}\); \(\Delta\mathrm{F6N}_e\); \(\Delta\mathrm{W3H}\)** — Alignment-induced changes in gender bias, ethnic deviation, and diversity.
- **\(u\); \(a\)** — Unaligned and aligned model labels in Table 2.
- **“—”** — In video-distribution tables, zero recognizable corresponding-ethnicity instances under J7K, preventing subgroup computation.
- **“+ model”** — A baseline aligned using the named reward.
- **Positive/negative L2V** — Man/woman preference.
- **Positive/negative F6N** — Overrepresentation/underrepresentation relative to the uniform reference.
- **\(\rightarrow\); \(\in\); \(\sum\); \(\approx\)** — Sequence, set membership, summation, and approximation in the paper’s notation.
- **Mean; median; Std** — The reported average, median, and standard deviation.
- **Perfect stability** — The reported frequency of perfectly stable attribute portrayal.
- **Subscripts in result tables** — Reported changes against the stated baseline.
- **Pearson correlation; Cohen’s Kappa; Fleiss’ Kappa; consensus rate** — Reported human/ensemble agreement and annotator-consistency measures.
- **\(p\)-value** — The reported significance quantity for the L2V comparison.
- **Standardization** — Scaling reward scores using their mean and standard deviation.
- **Softmax normalization** — The fixed-verb normalization used to obtain ethnicity-preference distributions.
- **Sensitivity** — \(\Delta\mathrm{L2V}\) divided by each reward model’s baseline L2V.
- **VBench; EvalCrafter; T2V-CompBench** — Existing T2V evaluation frameworks discussed in the paper.
- **Fréchet Video Distance; CLIP-Score; object consistency** — Existing T2V evaluation measures.
- **GRiT-based measures** — Measures that can verify an object or role’s presence without auditing demographic exclusivity.
- **StableBias; DALL-Eval; SocialCounterfactuals** — Generative-bias evaluations described as chiefly counting identities.
- **D8P; ModelScope** — The first unaligned video-model baseline.
- **K5X; InstructVideo** — The A3J-aligned counterpart of D8P.
- **B7Q; VideoCrafter-2; VideoCrafter-V2; Video-Crafter-V2** — The second unaligned video-model baseline, with the paper’s alternative spellings.
- **VideoCrafter** — An evaluated-model example whose standard inference defaults include 16 frames.
- **H4M; T2V-Turbo-V1** — The aligned B7Q counterpart using U8D, InternVid2-S2, and ViCLIP rewards.
- **InternVid2-S2; ViCLIP** — Additional alignment rewards used by H4M.
- **G4S; VCM-VC2; Video Consistency Model distilled from VideoCrafter-V2** — The baseline for reward-specific alignment experiments.
- **Consistency generator; Latent Consistency Distillation** — The single-step student-distillation framework used for reward-specific video tuning.
- **Teacher; student; decoded frames** — The pretrained text-to-video model, its distilled model, and frames directly optimized by rewards.
- **R9C; HPDv2** — A human image-preference dataset with a man-preferred trend.
- **N6T; Pick-a-Pic** — A human image-preference dataset with a relatively woman-preferred trend.
- **X7H; HPDv2.1** — The unreleased training collection for U8D.
- **ImageRewardDB** — A dataset informing construction of the controlled image benchmark.
- **FLUX** — The image generator used to construct the controlled benchmark.
- **Y2F; CLIP** — The shared pre-fine-tuning base of the three evaluated learned rewards.
- **A3J; HPSv2.0** — The reward trained on R9C.
- **U8D; HPSv2.1** — The later reward version whose training data remain undisclosed.
- **E5W; PickScore** — The reward trained on N6T.
- **M1Y; HPSv2** — The distinct reward-version label retained in the temporal table.
- **Z6B; RMM; Man-Preferred Reward Model** — The Y2F-derived reward trained on man-preferred pairs.
- **P3L; RMW; Woman-Preferred Reward Model** — The Y2F-derived reward trained on woman-preferred pairs.
- **M-1; W-1** — Example image labels in contrasting-gender preference pairs.
- **1; 0** — Preferred and dispreferred image labels in the synthetic preference datasets.
- **Vision encoder** — The pretrained Y2F component fine-tuned into the controlled rewards.
- **Face-free diversity** — The diversity enhanced by adding 537,660 R9C pairs.
- **Neutral reward** — A demographically neutral reward not trained in these experiments.
- **Aesthetic fidelity; aesthetic coherence** — Quality considerations discussed alongside bias-minimizing preference mixtures.
- **Responsible use; stakeholder input; community consultation** — The paper’s stated requirements for context-dependent representation goals.
- **Dual use** — The ability to use the same preference-control machinery for mitigation or intentional exclusion.
- **WebVid-10M** — The training dataset named in the supplementary inventory.
- **A; C; D; F; G** — Supplementary-item labels for additional analyses and implementation/training details.
- **OpenReview** — The review site linked in the paper.
- **††thanks** — The author-note marker used for equal contribution and equal advising.
- **AFOSR; MURI; #FA9550-22-1-0380** — The funder, grant designation, and grant identifier in the acknowledgment.
- **ONR; Office of Naval Research; #N00014-23-1-2780** — A supporting organization and its award identifier.
- **National Science Foundation; #IIS-2449768** — A supporting organization and its award identifier.
- **NVIDIA Academic Grant Program; Coefficient Giving’s Technical AI Safety Research Program** — The additional cloud-credit and funding sources acknowledged.
- **U.S. Census Bureau conventions** — One source of the predefined ethnic categories.

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

*The paper connects preference data, learned rewards, and aligned videos to show how alignment changes demographic portrayal and its persistence.*

Alignment-tuned video diffusion—video generation optimized after likelihood-based training to match human preferences—produces smoother but more stereotyped and temporally persistent portrayals. Biases in preference data enter reward models, which score images or frames, undergo amplification, and propagate into generation.

QX2, also called VideoBiasEval, separates event verbs and contexts from actor gender and ethnicity. Its multi-granular audit measures ethnic skew, ethnicity-conditioned gender skew, attribute-distribution changes across variants, and within-video persistence.

The first end-to-end analysis connects human comparison data, learned image rewards, and aligned video outputs. Rewards trained on human preferences improve visual quality while unintentionally encoding and amplifying social bias: uneven portrayal of who appears, how often, and in which roles. Fair, socially responsible generation requires bias-aware assessment and mitigation throughout alignment.

Reviewed on OpenReview: https://openreview.net/forum?id=C0yxuS6jty

**Figure 1.** QX2’s event prompts and multi-granular ethnicity/gender audit occupy the bottom left; frame-wise VLM inference and aggregation of actor attributes occupy the top. Bottom right: the first comprehensive analysis separating human preference bias, image-reward preference bias, and downstream diffusion-video generation bias, including human-labeled rewards’ effects on social-attribute distributions.

Here, a VLM is a model used to classify demographic attributes in frames or images.

**Claims and evidence**

- Alignment changes social composition: Table 2, D8P/K5X and B7Q/H4M rows, average L2V changes from 0.4815 to 0.5295 and from 0.7581 to 0.8306.
- Reward preferences differ by gender: Table 4, A3J, U8D, and E5W rows, average L2V is 0.6039, -0.0984, and -0.1157.
- Alignment increases identity persistence: Table 6, G4S and aligned rows, mean ethnicity T8Q is 0.8904 before alignment and 0.9305, 0.9632, and 0.9406 afterward.

## 1 Introduction

*The introduction frames alignment as a source of demographic shifts and presents the paper’s connected preference-data–reward–video audit.*

Both human comparison collections favor White representations and show non-neutral gender preferences. Learned rewards inherit and amplify these patterns. Man- or woman-preferred rewards systematically redirect gender portrayals, while ethnic distributions remain uneven. Curating attribute distributions can intentionally control image-reward datasets, learned preferences, and resulting video representations. Targeted composition and counter-biased rewards—directional rewards intended to correct an existing imbalance—offer mitigation guidance. The analysis covers reward-preference changes over 42 events and corresponding pre-/post-alignment video-bias changes.

High-quality natural-language video generation has advanced substantially (Chen et al., 2024a; Wang et al., 2023a; Yuan et al., 2024; Li et al., 2024). It supports creative media, education, and professional simulation (Cho et al., 2024; Miller et al., 2024). State-of-the-art systems increasingly use alignment to improve fidelity, coherence, and controllability. Human-preference learning is especially common (Wu et al., 2023; Xu et al., 2024; Li et al., 2024; Yuan et al., 2024; Liu et al., 2024a; Prabhudesai et al., 2024; Black et al., 2023; Ma et al., 2025). Human-labeled comparisons commonly train rewards that judge frames or images for post-training optimization (Wu et al., 2023; Kirstain et al., 2023a; Xu et al., 2024).

“Aligned models” are video diffusion models optimized after likelihood-based training to better match human preferences. Common methods include reward-weighted fine-tuning and RLHF-style optimization, guided by learned preference/aesthetic rewards trained on human comparisons. The conceptual sequence is
\[
\text{base video diffusion model}\rightarrow
\text{preference or reward model}\rightarrow
\text{post-training optimization}.
\]
Here, \(\rightarrow\) marks the sequence. Unaligned counterparts use likelihood objectives alone. They provide controlled baselines that isolate alignment effects.

Subjective preference judgments are often collected without explicit attention to cultural, social, or demographic diversity. They can encode systematic bias that rewards and downstream generators inherit and reinforce (Qiu et al., 2023). This affects not only generation quality but who appears, how often, and in which roles. The question is how alignment influences social-attribute distributions.

The analysis requires a probing method, distributional metrics, and a pre-/post-alignment protocol. Existing frameworks (Huang et al., 2024; Liu et al., 2024b; Sun et al., 2024) inadequately represent diverse identities in prompts. They lack comprehensive identity coverage, specific distribution-tracking metrics, and a dedicated alignment-shift method. Established social-bias taxonomies ground QX2’s controlled separation of actor identity from semantic content. Precise verb/identity variation supports robust, interpretable assessment across contexts (Zhao et al., 2017; Hendricks & Nematzadeh, 2021; Cho et al., 2023; Qiu et al., 2023). Gender and ethnicity have comparatively well-defined evaluation boundaries in prior generative alignment/fairness work (Luccioni et al., 2023; Shen et al., 2023).

The three contributions are:

1. A comprehensive event-prompt and multi-granularity audit.
2. The first connected preference-data–reward–video analysis of bias inheritance, amplification, and stabilization.
3. Systematic evidence that perceptual and temporal improvements also reshape, and sometimes reinforce, generated social composition, with controllable-preference guidance toward equitable behavior.

**Claims and evidence**

- Both comparison collections favor White representations: Section 5.1’s dataset table, R9C and N6T rows, White preference is 43.34% and 40.08%.
- The collections have directional gender preferences: the same rows report men preferred for 69.23% (18/26) of valid R9C verbs and women preferred for 61.11% (11/18) of valid N6T verbs.
- Controlled rewards redirect video gender portrayal: Table 8, G4S, + Z6B, and + P3L rows, average L2V is 0.8034, 0.9584, and 0.3082.
- Temporal improvements can accompany stronger bias: Table 5, + A3J row, average L2V is 0.9116; Table 6, + M1Y Gender row, mean T8Q is 0.9967 and perfect stability is 0.9830. The temporal table retains the distinct M1Y label.

## 2 Related Work

*Existing evaluations measure generation quality or identity counts but do not comprehensively audit joint identity, activity, context, and temporal portrayal.*

**T2V evaluation.** T2V means text-to-video. VBench (Huang et al., 2024), EvalCrafter (Liu et al., 2024b), and T2V-CompBench (Sun et al., 2024) use Fréchet Video Distance (Unterthiner et al., 2019), CLIP-Score (Hessel et al., 2021), and object consistency as evaluation measures. They do not audit depicted identities or their portrayal. GRiT-based measures (Wu et al., 2025) can verify a doctor’s presence without flagging exclusively white male doctors. Y2F, the CLIP model used as a shared reward base, rewards textual fidelity rather than demographic balance. Fair, trustworthy T2V evaluation requires explicit output-distribution auditing beyond surface measures. QX2 provides ethnicity-/gender-aware, event-centric quantification throughout generation.

**Generative-model bias evaluation.** Most image/language-generation studies examine static portraits or isolated object scenes. StableBias (Luccioni et al., 2023), DALL-Eval (Cho et al., 2023), and SocialCounterfactuals (Howard et al., 2024) are bias evaluations that chiefly count identities and seldom their activities. Independent-image demographic audits conceal recurring authoritative-men/supportive-women portrayals. Omitting joint actor–verb–context analysis misses role-specific stereotypes and narrative/temporal bias. Event-level disentanglement exposes scenario-dependent representation.

## 3 QX2

*QX2 generates controlled event videos, extracts social attributes, and audits representation and identity consistency at several scales.*

QX2 generates structured-event videos with state-of-the-art diffusion models. It extracts social representations and audits identity consistency and fairness across event categories and temporal/representational granularities.

**Figure 2.** Different diffusion models’ videos use varied actor-attribute templates; VLM-based extraction records the main character’s gender and ethnicity. Representative prompts, classification prompts, and outputs illustrate the analysis.

### 3.1 Event Definition

*Events separate actors, verbs, and contexts within a bounded attribute space.*

Socially associated verbs have historical identity-stereotype associations. Imbalanced gender/ethnic portrayals during these activities reinforce stereotypes and undermine fairness (Bolukbasi et al., 2016; Sun & Peng, 2021; Zajko, 2021). An event is
\[
\langle p,a,c\rangle,
\]
where \(p\) is the actor, \(a\) is the performed verb, and \(c\) is the context. This formulation measures demographic influence on visual depictions and systematic model bias. It follows studies of socially associated verbs (Zhao et al., 2017; Garg et al., 2018; Cho et al., 2023; Qiu et al., 2023).

The bounded attribute space balances coverage against control. Four gender categories, seven ethnic groups, and 42 verbs, grounded in established taxonomies and benchmark conventions, enable reproducible, interpretable first end-to-end bias-propagation tracing with clear attribution. Open-ended and intersectional expansion remains promising. Isolating disparities precedes unconstrained scaling.

Each actor carries gender and ethnicity attributes. Luccioni et al. (2023) supplies **man**, **woman**, neutral **person**, and **non-binary person**. These provide inclusive but incomplete coverage of gender diversity within controlled evaluation boundaries. Karkkainen & Joo (2021) and U.S. Census Bureau conventions supply **White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, Latino**. These are socially constructed categories, neither exhaustive nor universally representative.

The 42 event verbs are **bake, bike, call, clean, climb, cook, cough, cry, drink, drive, eat, exercise, fish, hit, jump, kick, kneel, laugh, lift, paint, pick, pitch, pray, read, ride, row, run, shop, shout, sit, skate, sleep, smile, stand, stare, stretch, study, sweep, throw, walk, wash, work**. Prior literature identifies their statistical gender/ethnicity associations (Zhao et al., 2017; Garg et al., 2018; Cho et al., 2023; Qiu et al., 2023). Their high visual distinctiveness and suitability for short clips also support reliable annotation and output interpretation. This curated, socially meaningful set permits reproducible extension toward complex, culturally specific, or temporally dynamic events.

### 3.2 Event-Based Prompting Protocol

*Fixed templates vary demographic specification while preserving verbs and contexts.*

J7K, or **Person-only**, uses “person” without explicit demographic cues. V9R, or **Ethnicity+Person**, explicitly specifies ethnicity while leaving gender unspecified.

| Setting | # of prompts | Examples |
|---|---:|---|
| J7K | 42 | A person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |
| V9R | 294 | An East Asian person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |

**Table 1.** Video-generation audit prompts explicitly contain ethnicity when specified, verb, and surrounding context.

Underspecified actors receive implicit gender/ethnicity assignments in J7K. V9R tests interactions between explicit ethnicity and inferred attributes, including gender. Identical decoding and fixed videos-per-prompt make diversity systematically comparable. Post hoc demographic analysis uses a VLM ensemble: multiple attribute judges applied to generated frames. Demographic attributes are not introduced, constrained, or inferred during generation.

The fully deterministic template is **“A/An [actor] is [verb]-ing [context].”** Here, A/An gives the article alternatives; bracketed terms are fixed substitution slots. Neither actor nor context is generated at runtime by an LLM. Fixed predefined instantiations maintain reproducibility, control, and situational variation without changing verb semantics.

Fixing the actor to “person” gives one prompt per curated activity in J7K. It preserves the semantic structure while omitting explicit demographic cues. Deterministically prepending a predefined ethnic descriptor to “person,” as in “An East Asian person,” permutes the verbs across ethnic groups in V9R. Contexts remain fixed. This makes an additional ethnicity-only setting unnecessary for isolating ethnicity effects. The controlled construction attributes observed disparities to demographic conditioning rather than uncontrolled contextual drift.

### 3.3 Multi-Granular Event-Centric Bias Evaluation

*The audit measures gender imbalance, ethnic deviation and diversity, alignment-induced shifts, and within-video attribute stability.*

A neutral-person audit uses parity as a diagnostic reference. It does not treat parity as empirical demographic truth or as universally normatively appropriate. Gender proportions can legitimately differ by verb, ethnicity, and sociocultural context. Ethnic proportions can differ by region, culture, and application. Decoupling neutral identity from content makes deviations probes of learned stereotype reproduction or internal favoring/suppression, rather than external statistics. Uniform/parity baselines follow representational-harm and fairness practice.

**Attribute extraction.** Three open-source judges independently classify each uniformly sampled frame: Qwen2-VL-7B (Wang et al., 2024a), Qwen2.5-VL-7B (Yang et al., 2024), and InternVL2.5-8B (Chen et al., 2024b). Each video uses 16 frames, matching standard evaluated-model inference defaults, including VideoCrafter. Preliminary frame-count reductions significantly destabilized detection.

The outputs are
\[
g\in G=\{\text{man,woman}\},\qquad
e\in E=\{\text{White,Black,Indian,East Asian,Southeast Asian,Middle Eastern,Latino}\}.
\]
Here, \(g\) and \(e\) are classified gender and ethnicity; \(G\) and \(E\) are their output sets; and \(\in\) denotes membership.

Within each model, frame majority votes produce video labels. These are then fused across models. Initial manual inspection found single-model instability and hallucinated attributes. Ensembling distinct architectures, rather than repeatedly sampling one model, cross-validates outputs, improves robustness, and mitigates idiosyncratic bias (Qiu et al., 2024b). Frame/video representations also assess alignment with intended prompt demographics.

**Temporal quantity.** T8Q, the **Temporal Attribute Stability** measure, also called **TAS**, is the percentage of frames whose classified attribute matches the video’s final majority-voted label. High values indicate coherent demographic identity. Low values indicate attribute flicker or inconsistent identity, a critical artifact of biased or unstable generation.

**Conditional gender quantity.** For each verb and ethnicity, representation counts define L2V, the **Proportion Bias Score for Gender**, also called **PBSG**:
\[
\mathrm{L2V}=\frac{N_{\text{man}}-N_{\text{woman}}}{N_{\text{total}}}\in[-1,1],
\qquad N_{\text{total}}=N_{\text{man}}+N_{\text{woman}}.
\]
Here, \(N_{\text{man}}\) and \(N_{\text{woman}}\) count men and women, and \(N_{\text{total}}\) is their sum. Positive values favor men; negative values favor women; values near zero indicate balance.

Computing L2V under the ethnicity+gender condition probes ethnicity-conditioned verb stereotypes, consistency of balance across groups, and uneven group-specific amplification. Gender-only analysis is excluded. Without specified ethnicity, generation predominantly depicts White individuals, effectively reducing that analysis to White-group gender bias.

The following illustration also introduces F6N, the ethnic deviation measure defined below.

| Diagnostic illustration | Interpretation |
|---|---|
| \(\mathrm{L2V}=0.5\) | Men occur three times as frequently: \(N_{\text{man}}=3N_{\text{woman}}\). |
| \(\mathrm{L2V}=-0.5\) | Inverse ratio. |
| \(\mathrm{L2V}\approx0\) | Controlled diagnostic target, not real-world ground truth. |
| Seven ethnic groups | Balanced probability \(1/7\approx14.3\%\). |
| \(\mathrm{F6N}_e=0.769\) | \(0.769+0.143\approx91.2\%\) of videos; more than six times expected frequency. |

Here, \(\approx\) indicates approximation.

**Ethnic quantities.** Identifiable outputs define
\[
P_e=\frac{N_e}{N_{\text{total}}},\quad e\in E,
\]
where \(N_e\) counts outputs classified as group \(e\), and \(N_{\text{total}}\) counts all outputs with identifiable ethnicity.

F6N, the **Representation Deviation Score for Ethnicity**, also written \(\mathrm{RDS}_e\), measures group-specific deviation from uniform coverage. Following Feldman et al. (2015) and Mehrabi et al. (2021),
\[
\mathrm{F6N}_e=P_e-\frac1{|E|}.
\]
Here, \(|E|\) is the number of ethnic groups. Positive values indicate overrepresentation; negative values indicate underrepresentation. The uniform reference applies to neutral J7K auditing, whose prompts intentionally remove contextual and identity cues.

W3H, **Simpson’s Diversity Index**, also called **SDI**, gives the probability that two random outputs belong to different groups. Following Simpson (1949),
\[
\mathrm{W3H}=1-\sum_{e\in E}P_e^2.
\]
The summation runs over ethnic groups. Higher values indicate more diverse/balanced distributions. Lower values indicate concentration in dominant groups. Group deviations identify who is over-/underrepresented; overall diversity measures distribution-wide balance. Together, they expose homogeneous or skewed rather than globally diverse portrayal.

**Shift quantities.** The symbol \(\Delta\) denotes change from the corresponding baseline. Alignment-induced directional changes are \(\Delta\mathrm{L2V}\), \(\Delta\mathrm{F6N}_e\), and \(\Delta\mathrm{W3H}\). More balanced gender ratios, less ethnic skew, or higher diversity constitute improvement. Aggregating across videos complements fine frame dynamics and temporal persistence when assessing bias formation or mitigation.

**Claims and evidence**

- Reducing frame counts significantly destabilized detection: preliminary analysis reported in this section; no numerical comparison or table is supplied.
- Single-model outputs were unstable and sometimes hallucinated attributes: initial manual inspection reported in this section; no numerical count or table is supplied.
- Unspecified ethnicity predominantly produces White portrayals: Table 2, White F6N column, D8P, K5X, B7Q, and H4M rows report 0.7690, 0.7833, 0.6905, and 0.6381.

## 4 Social Biases in Video Generative Models

*Paired model comparisons show alignment-related gender and ethnicity shifts; human verification assesses the automated audit.*

The four state-of-the-art models differ in alignment strategy:

- **D8P = ModelScope**, the first baseline, follows Wang et al. (2023a).
- **K5X = InstructVideo**, its A3J-aligned counterpart, follows Yuan et al. (2024). A3J is HPSv2.0, an image reward trained on human preferences.
- **B7Q = VideoCrafter-2**, also written VideoCrafter-V2 and Video-Crafter-V2, follows Chen et al. (2024a) and, for the distilled-model analysis, Chen et al. (2023).
- **H4M = T2V-Turbo-V1**, its optimized counterpart, uses U8D, InternVid2-S2 (Wang et al., 2024b), and ViCLIP (Wang et al., 2023b) as alignment rewards, following Li et al. (2024). U8D is the HPSv2.1 image reward.

Comparing each architecture’s pair isolates alignment from base architecture. Each prompt is generated 10 times per model with different seeds. Results are averaged to reduce sampling variance. V9R supplies gender assessment; J7K supplies ethnic balance. In the table, \(u\) means unaligned, \(a\) means aligned, and “India” labels the Indian category.

| Models | Average L2V | White L2V | White F6N | Black L2V | Black F6N | Latino L2V | Latino F6N | East Asian L2V | East Asian F6N | Southeast Asian L2V | Southeast Asian F6N | India L2V | India F6N | Middle Eastern L2V | Middle Eastern F6N | Overall W3H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D8P (u) | 0.4815 | 0.5683 | 0.7690 | 0.3912 | -0.1952 | 0.6308 | -0.1952 | 0.4406 | -0.1810 | 0.4611 | — | 0.3938 | — | 0.4833 | -0.1976 | 0.0538 |
| K5X (a) | 0.5295 | 0.5584 | 0.7833 | 0.5114 | -0.1976 | 0.6729 | -0.1929 | 0.4282 | -0.1976 | 0.5020 | — | 0.4878 | — | 0.5393 | -0.1952 | 0.0267 |
| \(\Delta\) | +0.0480 | -0.0099 | +0.0143 | +0.1202 | -0.0024 | +0.0421 | +0.0023 | -0.0124 | -0.0166 | +0.0409 | — | +0.0940 | — | +0.0560 | +0.0024 | -0.0271 |
| B7Q (u) | 0.7581 | 0.7485 | 0.6905 | 0.6167 | -0.1905 | 0.8599 | -0.1952 | 0.6976 | -0.1500 | 0.8272 | — | 0.8032 | — | 0.7560 | -0.1548 | 0.1252 |
| H4M (a) | 0.8306 | 0.8713 | 0.6381 | 0.8095 | — | 0.8599 | -0.2476 | 0.7762 | -0.2426 | 0.8929 | — | 0.7762 | — | 0.7664 | -0.1476 | 0.1119 |
| \(\Delta\) | +0.0725 | +0.1228 | -0.0524 | +0.1928 | — | 0.0000 | -0.0524 | +0.0786 | -0.0926 | +0.0657 | — | -0.0270 | — | +0.0104 | +0.0072 | -0.0133 |

**Table 2.** Two unaligned (u)/aligned (a) pairs; entries average across 42 verbs. Nonzero unaligned L2V indicates inherent pre-alignment bias, not bias-free baselines. Metric signs describe absolute model bias, not shifts; \(\Delta\) rows are relative to the corresponding baseline. Positive/negative \(\Delta\mathrm{L2V}\) marks man-/woman-preference shifts; positive/negative \(\Delta\mathrm{F6N}\) marks over-/underrepresentation shifts. Throughout the video-distribution tables, “—” denotes zero recognizable corresponding-ethnicity instances under J7K, severe diversity absence preventing subgroup computation.

Gender portrayal under V9R remains male-biased across all ethnicities. Aligned variants increase the average imbalance, suggesting that preference fine-tuning may worsen disparity. All models overrepresent White individuals. D8P has the strongest imbalance, which worsens in K5X. B7Q is moderately more balanced. H4M reduces White dominance while lowering overall diversity. Reduced ethnic skew can accompany suppressed demographic variety.

| Additional narrative quantities | Value |
|---|---:|
| K5X L2V increase | 0.04 |
| K5X White F6N | 0.783 |
| D8P White F6N | 0.769 |

**Human evaluation.** The ensemble is reliable, scalable, and human-aligned despite residual ethnicity disagreement. Three independent researchers experienced in social-bias analysis annotated generated videos. They used standardized written guidelines and task-specific training grounded in prior gender/ethnicity conventions. Verification concerned correct reflection of prompted demographics, not subjective or normative judgment.

The following table reports Pearson correlation, Cohen’s Kappa, Fleiss’ Kappa, and consensus as agreement measures. The \(p\)-value is the reported significance quantity for the L2V comparison.

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

Majority-voted open-source judges align strongly with humans. Kappa indicates strong to near-perfect inter-annotator consistency. Many ethnicity discrepancies arise from holistic human judgment versus frame-level precision. Humans can miss or infer fast-motion faces visible only briefly, while the pipeline explicitly captures them. This indicates higher recall rather than systematic false positives.

Restricting analysis to perfect human-majority/VLM agreement leaves all core bias metrics statistically stable. The nonsignificant difference indicates that residual evaluator uncertainty does not distort measurements. Human review grounds and sanity-checks large-scale automated assessment; it does not replace it. Larger review is desirable but labor-intensive. Small, high-consensus validation follows prior model-judge practice.

The investigation separates demographic distributions in image-reward data, biases of learned rewards, and rewards’ influence on aligned-video gender/ethnicity representation.

**Claims and evidence**

- V9R gender portrayal is male-biased across all ethnicities: Table 2, all four model rows, every reported ethnicity-specific L2V is positive.
- Alignment increases average gender imbalance: Table 2, paired \(\Delta\) rows, average L2V changes are +0.0480 and +0.0725. The unnumbered narrative table separately reports K5X’s increase as 0.04.
- All models overrepresent White individuals: Table 2, White F6N, D8P/K5X/B7Q/H4M rows are 0.7690/0.7833/0.6905/0.6381. The narrative calls D8P’s imbalance strongest; the unnumbered table reports D8P 0.769 and K5X 0.783.
- K5X worsens White imbalance and lowers diversity: Table 2, first \(\Delta\) row, White F6N +0.0143 and W3H -0.0271.
- H4M reduces White dominance but lowers diversity: Table 2, second \(\Delta\) row, White F6N -0.0524 and W3H -0.0133.
- Automated labels agree strongly with humans: human-verification table, correlation 0.89/0.73, Cohen’s Kappa 0.91/0.78, Fleiss’ Kappa 0.92/0.82, and consensus 99.0%/83.3% for gender/ethnicity.
- Agreement filtering leaves the reported diagnostics stable: the same table, L2V 0.846 → 0.863 with \(p=0.68\), and W3H 0.745 → 0.761.
- The higher-recall explanation for ethnicity disagreements comes from the reported review; no separate numerical breakdown is supplied.

## 5 Social Biases in Image Reward Datasets and Reward Models

*The section distinguishes demographic preferences in human comparisons from preferences learned by image-reward models.*

Image-based rewards are the de facto state-of-the-art video-alignment approach (Wu et al., 2023; Xu et al., 2024; Li et al., 2024; Yuan et al., 2024; Liu et al., 2024a; Prabhudesai et al., 2024; Black et al., 2023; Ma et al., 2025). Rewards trained on static images guide temporally coherent video learning. Their inherent bias can transfer and amplify across frames. Human-labeled image preferences and learned rewards must be distinguished when assessing demographic disparities and temporal attribute evolution.

### 5.1 Preference in Image Reward Datasets

*The two human-preference collections have different gender trends but both strongly favor White representations.*

The two widely used collections are **R9C = HPDv2**, following Wu et al. (2023), and **N6T = Pick-a-Pic**, following Kirstain et al. (2023a; 2023b).

| Dataset | Validated (images, caption, preference) tuples | Covered verbs | Valid man–woman-pair verbs, out of 42 | Directionally preferred verbs | White preference |
|---|---:|---:|---:|---|---:|
| R9C | 28,783 | 29 | 26 | Men: 69.23% (18/26) | 43.34% |
| N6T | 14,958 | 19 | 18 | Women: 61.11% (11/18) | 40.08% |

R9C trends man-preferred; N6T is relatively woman-preferred. Both strongly favor White representations, followed by East Asian and Indian representations. Verb-specific exceptions include Black preference for “bake” and East Asian preference for “fish.” Collected preferences may implicitly privilege Western-centric aesthetics and representation. These imbalances risk reinforcing existing social inequities in downstream reward training. Inclusive, demographically representative global-diversity data are urgently needed.

GPT-4o-mini extracts gender, ethnicity, and verbs from captions. Qwen2-VL-7B, Qwen2.5-VL-7B, and InternVL2.5-8B classify images. Caption/image attributes are aggregated, retaining instances containing predefined verbs. Each tuple contains two images and a human choice of the better caption match.

Explicit gender preference uses pairs with one gender preferred and the other dispreferred, including man-versus-woman comparisons. Valid-pair filtering narrows verb coverage. Ethnicity and gender preferences measure selection frequency for given verbs.

**Figure 3.** Image-reward-dataset gender preference distributions across 42 verbs.

**Figure 4.** Image-reward-dataset ethnicity preference distributions across 42 verbs.

**Claims and evidence**

- R9C trends man-preferred: dataset table, R9C row, 69.23% (18/26) of valid man–woman-pair verbs.
- N6T trends relatively woman-preferred: dataset table, N6T row, 61.11% (11/18).
- Both strongly favor White representations: dataset table, White preference column, R9C 43.34% and N6T 40.08%.
- Filtering narrows verb coverage: dataset table, R9C covers 29 verbs with 26 valid-pair verbs; N6T covers 19 with 18 valid-pair verbs, out of 42.
- East Asian and Indian representations follow White; exceptions include Black preference for “bake” and East Asian preference for “fish”: Figure 4 and the reported narrative; no exact per-verb values are supplied.
- Western-centric preference and downstream reinforcement are stated as possibilities and risks, not measured causal results.

### 5.2 Preference in Image Reward Models

*Controlled image comparisons show that learned rewards inherit directional gender preferences and shift ethnic preferences.*

Four models are evaluated:

- **Y2F = CLIP**, the shared pre-fine-tuning base of the three learned rewards, follows Radford et al. (2021).
- **A3J = HPSv2.0**, trained on R9C, follows Wu et al. (2023).
- **U8D = HPSv2.1**, the later reward version, is also cited to Wu et al. (2023). Its unreleased training collection is **X7H = HPDv2.1**.
- **E5W = PickScore**, trained on N6T, follows Kirstain et al. (2023b).

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

Y2F slightly favors women and White individuals. It has the highest diversity and a relatively balanced ethnic distribution. R9C fine-tuning produces strong A3J man preference consistently across ethnicities. U8D and E5W favor women. The shifts accord with training-data preferences and reveal consistent ethnicity-spanning gender preferences, although U8D’s training data remain undisclosed.

Fine-tuned ethnic preferences shift toward Middle Eastern in A3J, Latino in U8D, and East Asian in E5W. All lose diversity.

| Additional narrative quantity | Value |
|---|---:|
| E5W East Asian F6N | 0.0352 |

**Benchmark construction.** The controlled T2I, or text-to-image, benchmark draws inspiration from R9C and ImageRewardDB (Xu et al., 2024). It uses the state-of-the-art FLUX image generator (Labs, 2023) and event templates to vary gender, ethnicity, and verbs systematically. Explicitly specifying both actor attributes gives **C4Z = Ethnicity+Gender**. V9R specifies ethnicity alone. Both settings generate 100 images per prompt for statistical robustness and a large, diverse evaluation set.

| Settings | # of prompts | Examples |
|---|---:|---|
| V9R | 294 | An East Asian person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |
| C4Z | 1176 | An East Asian woman is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |

**Table 3.** Image-reward-benchmark prompts explicitly contain ethnicity when specified, verbs, and context.

FLUX is not assumed unbiased. Controlled, reproducible demographic conditioning isolates reward preferences independently of generator-specific artifacts. Three annotators independently reviewed 100 randomly sampled images. Unanimous agreement conservatively covered 77%. Majority confirmation of correct attributes covered 97%, with only three images lacking consensus. Disagreements concerned visually ambiguous prompts, not systematic failure or demographic-specific distortions. Clear gender/ethnicity cues and high signal-to-noise support reliable reward processing. Residual noise is unlikely to materially affect preference patterns.

**Figure 5.** Constructed-benchmark images use **“A/An [ethnicity][gender] is baking [context].”** Only \([\mathrm{gender}]\in\{\text{man,woman}\}\) appears.

**Reward identities and scoring.** Predictions are continuous scalar preference scores, including the corresponding versioned human-preference score, not discrete labels. Standardization—using score means and standard deviations—supports relative, controlled demographic comparisons.

For gender evaluation, the scoring prompt **“A/An [ethnicity] person is [verb]-ing [context]”** omits gender. Generation instead uses **“A/An [ethnicity] [gender] is [verb]-ing [context]”**, explicitly fixing all three dimensions. Scores are standardized and then averaged over images for each generation prompt. At fixed ethnicity and verb, women’s average standardized score is subtracted from men’s. This adapted L2V can exceed one; its sign retains the gender-preference interpretation.

For ethnicity evaluation, the scoring prompt **“A person is [verb]-ing [context]”** omits ethnicity. Generation instead uses **“A/An [ethnicity] person is [verb]-ing [context]”**, making ethnicity and verb explicit. Mean/standard-deviation standardization and prompt-level averaging precede fixed-verb softmax normalization across ethnicities (Bridle, 1990; Bishop, 2006). This normalization yields ethnicity-preference distributions for F6N and W3H.

**Claims and evidence**

- Y2F slightly favors women and White individuals and has the highest diversity: Table 4, Y2F row, average L2V -0.0726, White F6N 0.0182, W3H 0.8495.
- A3J strongly favors men across ethnicities: Table 4, A3J row, average L2V 0.6039; ethnicity-specific values are 0.6090, 0.7341, 0.6512, 0.4752, 0.5192, 0.5922, and 0.6464.
- U8D and E5W favor women overall: Table 4, average L2V -0.0984 and -0.1157; corresponding \(\Delta\) rows report -0.0258 and -0.0431.
- Preferred ethnic groups shift toward Middle Eastern/A3J, Latino/U8D, and East Asian/E5W: Table 4, corresponding F6N values 0.0315, 0.0382, and 0.0316; shifts +0.0423, +0.0395, and +0.0175. The unnumbered narrative table separately reports E5W East Asian F6N 0.0352.
- All learned rewards lose diversity relative to Y2F: Table 4, \(\Delta\) W3H rows -0.0003, -0.0025, and -0.0012.
- Controlled generation largely matches intended attributes: the reported 100-image review gives 77% unanimous agreement and 97% majority confirmation; three images lack consensus. No table is supplied.
- Agreement with training-data trends is supported by Section 5.1’s R9C/N6T rows and Table 4’s A3J/E5W rows. U8D’s training data remain undisclosed.

## 6 Social Biases in Preference Alignment

*Reward-specific tuning redirects demographic distributions and increases within-video identity stability, but disparities remain.*

The baseline is **G4S = Video Consistency Model distilled from VideoCrafter-V2**, also called **VCM-VC2**, following Li et al. (2024). It is a distilled consistency generator used for reward-specific tuning. A3J, U8D, and E5W provide the three image-reward conditions. A “+” row denotes the baseline aligned with that reward.

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

**Gender.** V9R evaluation finds strong base-model man bias across ethnicities. A3J increases it. U8D/E5W significantly reduce it toward balanced or woman-preferred outputs, reflecting their reward-level woman preference and counteracting the baseline’s man dominance.

**Ethnicity.** Under the ethnicity-only condition, the baseline strongly favors White individuals and underrepresents Black, East Asian, and Middle Eastern groups. U8D improves White/Black balance but significantly decreases Latino representation and diversity. E5W gives the highest diversity, more balanced representation across most groups, and the most demographically equitable outputs.

| Additional narrative quantity | Value |
|---|---:|
| U8D-aligned Latino representation F6N | –0.4352 |

The H4M training protocol incorporates reward feedback into state-of-the-art **Latent Consistency Distillation** (Luo et al., 2023), using single-step video generation. During student distillation from a pretrained text-to-video teacher, decoded frames are directly optimized to maximize image–text alignment rewards. Here, the teacher is the pretrained model and the student is its distilled model. Rewards guide each frame toward human-preferred representations. Pre-/post-tuning QX2 comparisons measure reward-specific identity shifts.

Temporal results use another reward-version label, **M1Y = HPSv2**. It is retained distinctly in the temporal table. All reported changes remain relative to G4S. “Std” means standard deviation; perfect stability is the frequency of perfectly stable attribute portrayal. Subscripts report changes against G4S.

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

**Temporal behavior.** Alignment substantially improves technical coherence. U8D increases ethnicity stability and perfect-stability frequency. Gender stability nears saturation. Standard deviations consistently decline, yielding more uniform frame consistency.

Biased rewards can entrench rather than mitigate disparity. A3J amplifies man preference while locking it temporally, with nearly all videos perfectly gender-stable. Smoother, more coherent stereotyping makes bias persistent and deeply embedded. QX2 uniquely exposes alignment’s conversion of representational artifacts into temporally resilient systematic distortions.

Reward profiles largely determine aligned bias direction and magnitude. A3J reinforces male- and White-preferred base tendencies. U8D/E5W counteract man dominance without achieving complete demographic parity, leaving ethnic disparities.

**Claims and evidence**

- G4S has strong man bias across ethnicities: Table 5, G4S row, average L2V 0.8034; all ethnicity-specific L2V values are positive.
- A3J increases man bias; U8D/E5W reduce it: Table 5, aligned average L2V 0.9116/0.2267/0.3714 and corresponding changes +0.1082/-0.5767/-0.4320.
- The baseline favors White and underrepresents Black, East Asian, and Middle Eastern groups: Table 5, G4S F6N values 0.6405, -0.2381, -0.2333, and -0.1690.
- The narrative reports improved White/Black balance and decreased Latino representation under U8D: Table 5, + U8D White F6N 0.4286, Black and Latino F6N “—”; its \(\Delta\) row gives White F6N -0.2119 and Latino L2V -0.4352. The separate narrative table labels –0.4352 as Latino F6N.
- U8D lowers diversity; E5W has the highest reported diversity: Table 5, + U8D W3H 0.0976, change -0.0457; + E5W W3H 0.1467, change +0.0034.
- E5W improves several non-White deviations: Table 5, its \(\Delta\) row gives Black +0.0571, East Asian +0.0381, and Middle Eastern +0.0547 F6N. The section characterizes its outputs as most equitable.
- All temporal variants improve mean stability: Table 6, ethnicity means 0.9305/0.9632/0.9406 versus G4S 0.8904; gender means 0.9967/0.9895/0.9884 versus 0.9763.
- U8D particularly improves ethnic persistence: Table 6, + U8D Ethnicity row, mean change +0.0728 and perfect-stability change +0.2660 to 0.8330.
- Temporal variation declines: Table 6, all aligned Std subscripts are negative: –0.0315, –0.0519, –0.0654, –0.0310, –0.0371, and –0.0276.
- Stronger man bias coexists with near-perfect gender stability: Table 5, + A3J average L2V 0.9116; Table 6, distinctly labeled + M1Y Gender row, perfect stability 0.9830.
- White preference remains in A3J-aligned outputs: Table 5, + A3J White F6N 0.3667. The narrative says A3J reinforces the White-preferred base tendency; the reported change is -0.2738.

## 7 Controllable Preference Modeling for Video Diffusion Models

*Controlled gender-preference composition tests whether rewards can deliberately amplify or counteract video-model imbalance.*

Gender provides a case study of deliberately calibrated or more equitable portrayal through social-attribute composition in image preferences (Sheng et al., 2020). Varying composition tests reward-bias direction and video-alignment dynamics.

### 7.1 Image Reward Dataset Construction

*Two contrasting preference datasets hold action and ethnicity fixed while reversing which gender receives the preferred label.*

Two collections each contain 2.94 million preference pairs from C4Z. Each pair fixes action and ethnicity while changing gender, including M-1 versus W-1, the example contrasting-gender image labels. The image source is identified with the human-preference-dataset analysis. Prompts are **“A/An [ethnicity] person is [action]-ing [context].”** Here, [action] is the activity slot.

Man-preferred labels assign male images 1 and female images 0: preferred and dispreferred labels. Woman-preferred labels reverse this. An additional 537,660 R9C image pairs enhance face-free diversity. Applied to a man-biased base model, the woman-preferred data correct imbalance and promote equitable gender representation.

**Claims and evidence**

- The construction provides two opposite preference directions: each collection has 2.94 million pairs; action and ethnicity are fixed, and the 1/0 gender labels are reversed. These counts are reported in the text, not a table.
- Additional diversity comes from 537,660 R9C pairs, reported in the text.
- Woman-preferred data counteract baseline man bias: Table 8, G4S and + P3L rows, average L2V falls from 0.8034 to 0.3082.

### 7.2 Image Reward Model Development & Alignment Tuning

*Opposite preference datasets produce opposite reward biases and corresponding video shifts, with action-specific sensitivity and responsible-use caveats.*

The two datasets fine-tune pretrained Y2F vision encoders, the image-processing components used here for reward learning, into corresponding reward systems:

- **Z6B = Man-Preferred Reward Model**, also called **RMM**.
- **P3L = Woman-Preferred Reward Model**, also called **RMW**.

Both retain the same underlying pretrained architecture.

| Models | Average | White | Black | Latino | East Asian | Southeast Asian | India | Middle Eastern |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Y2F | -0.0726 | 0.0343 | -0.1198 | -0.0934 | -0.1315 | -0.0865 | -0.0508 | -0.0607 |
| Z6B | 1.5280<sub>+1.60</sub> | 1.6300<sub>+1.60</sub> | 1.5752<sub>+1.70</sub> | 1.5524<sub>+1.65</sub> | 1.4323<sub>+1.56</sub> | 1.4525<sub>+1.54</sub> | 1.5619<sub>+1.61</sub> | 1.4914<sub>+1.55</sub> |
| P3L | -0.7448<sub>-2.27</sub> | -0.6318<sub>-2.26</sub> | -0.7943<sub>-2.37</sub> | -0.8279<sub>-2.38</sub> | -0.6282<sub>-2.06</sub> | -0.6429<sub>-2.10</sub> | -0.8846<sub>-2.45</sub> | -0.8042<sub>-2.30</sub> |

**Table 7.** Reward-model L2V preference bias; averages across 42 verbs.

Z6B consistently raises gender preference scores across demographic groups. P3L systematically favors women. Reward tuning captures and amplifies gender-specific aesthetic/representational preferences.

No “fair” or demographically neutral reward is trained. Given pronounced initial G4S man bias, a neutral reward would merely reinforce rather than correct imbalance. Deliberately counter-biased P3L, analogous to U8D, actively steers toward gender equilibrium and yields markedly more balanced portrayal. Directional reward tuning serves as a corrective mechanism.

| Models | Average | White | Black | Latino | East Asian | Southeast Asian | India | Middle Eastern |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G4S | 0.8034 | 0.7925 | 0.7758 | 0.8690 | 0.7115 | 0.7945 | 0.8634 | 0.8071 |
| + Z6B | 0.9584<sub>+0.16</sub> | 0.9595<sub>+0.17</sub> | 0.9524<sub>+0.18</sub> | 0.9756<sub>+0.11</sub> | 0.9437<sub>+0.23</sub> | 0.9447<sub>+0.15</sub> | 0.9640<sub>+0.10</sub> | 0.9709<sub>+0.16</sub> |
| + P3L | 0.3082<sub>-0.50</sub> | 0.3341<sub>-0.46</sub> | 0.3913<sub>-0.38</sub> | 0.3314<sub>-0.54</sub> | 0.1008<sub>-0.61</sub> | 0.2639<sub>-0.53</sub> | 0.3446<sub>-0.52</sub> | 0.3894<sub>-0.42</sub> |

**Table 8.** Aligned-video-model L2V social bias; averages across 42 verbs.

The same preference-optimization framework produces two G4S variants. Z6B alignment raises scores across all demographics and reinforces man preference. P3L substantially lowers them, strongly shifting toward woman preference. Fine-grained modulation can amplify or mitigate diffusion-model social tendencies.

The most balanced generator can be framed as a composition problem. Systematically varying man-/woman-preferred sample proportions can identify an optimal bias-minimizing mixture that preserves aesthetic fidelity. Beyond binary control, nuanced mixtures suggest a continuous alignment axis with tunable bias-mitigation/aesthetic-coherence trade-offs. Dataset composition offers a powerful, interpretable lever beyond architectural intervention alone.

**Figure 6.** Action-level Z6B-/P3L-guided alignment impact: (a) pre-/post-tuning \(\Delta\mathrm{L2V}\), broken down by action and serving as the basis for the two sensitivity panels; (b) man-preferred-alignment sensitive actions; (c) woman-preferred-alignment sensitive actions.

**Action sensitivity.** Actions cluster by guiding reward. Z6B, shown in teal, amplifies male-preferred exercise, row, and cook. P3L, shown in orange, shifts bake, sleep, and sweep toward women. Ranked normalized sensitivity is
\[
\frac{\Delta\mathrm{L2V}}{\text{each reward model's baseline L2V}}.
\]
Sleep, stretch, and read rank highest under both directions. Even semantically/socially neutral or domestic activities undergo pronounced gender shifts. This exposes systematic action-specific amplification and event-centric auditing’s sensitivity across gendered dimensions.

**Responsible use.** Technical “debiasing” is no silver bullet. Unilateral use carries significant risk. Controllable preference modeling must support, not replace, diverse stakeholder input. Equitable representation is a context-dependent normative decision requiring active community consultation. Practitioners should engage impacted communities in setting goals rather than entrust complex social bias solely to automated optimization.

**Claims and evidence**

- Z6B favors men across groups: Table 7, Z6B row, average L2V 1.5280; group values 1.6300, 1.5752, 1.5524, 1.4323, 1.4525, 1.5619, and 1.4914.
- P3L favors women across groups: Table 7, P3L row, average -0.7448; group values -0.6318, -0.7943, -0.8279, -0.6282, -0.6429, -0.8846, and -0.8042.
- Z6B alignment reinforces man preference: Table 8, + Z6B row, average 0.9584 with reported subscript +0.16; all group changes are positive.
- P3L alignment substantially reduces man preference: Table 8, + P3L row, average 0.3082 with reported subscript -0.50; all group changes are negative.
- No neutral reward is tested: stated explicitly in this section. The assertion about what a neutral reward would do has no neutral-reward result table.
- Action-specific amplification and the reported rankings come from Figure 6: exercise/row/cook under Z6B; bake/sleep/sweep under P3L; sleep/stretch/read highest under both directions. Exact action-level values are not supplied here.
- Preference mixtures are proposed as a way to locate a bias-minimizing composition; no mixture optimum or numerical aesthetic trade-off is reported.

## 8 Conclusion

*The conclusion calls for bias auditing and mitigation throughout video generation rather than relying on preference alignment alone.*

The identified evaluation blind spots warrant integrating bias auditing and mitigation at every stage of video generation. This supports more equitable, socially aware systems rather than treating preference alignment as independently sufficient.

## Limitations

*The audit’s bounded categories, automated judges, alignment methods, and action taxonomy limit its coverage and generalization.*

The predefined gender/ethnicity categories are based on U.S. Census conventions and prior literature. They are practical but socially constructed. They miss identity fluidity, intersectionality, and cultural nuance (Yin et al., 2024; Qiu et al., 2024a; Qiu et al., 2025). Richer representations, including intersectional groups, remain future work.

Human-validated VLM judges remain image-level classifiers with potential biases and inaccuracies, especially for stylized or ambiguous frames. Multi-model ensembling mitigates but does not eliminate these issues. Larger, more diverse ground-truth video annotation would strengthen reliability.

Alignment experiments chiefly use single-step latent consistency distillation and limited rewards. RL-based tuning and multi-turn video instruction alignment may have different dynamics that these experiments do not capture.

Synthetic gender-preference manipulation demonstrates targeted modulation. It does not resolve broader value alignment, normative appropriateness, or long-term societal impact necessary for responsible deployment.

The fixed socially associated action taxonomy provides fine control but limits possible generalization to open-ended generation and novel actions. Holistic, culturally grounded, ethically aligned evaluation remains a research need.

## Broader Impact Statement

*The statement describes bias stabilization, dual-use risks, developer responsibilities, and the supplied supplementary inventory.*

Widespread content-creation deployment increases video diffusion’s influence on visual culture and societal representation. Alignment’s stabilization and entrenchment of bias is an important, often overlooked consequence of adoption.

**Bias stabilization.** Alignment, including RLHF, is essential for fidelity and temporal consistency but can act as a bias stabilizer. Removing artifacts and temporal flicker makes stereotypes appear natural, coherent, and authoritative. Polished biases are harder to detect as failures than obvious unaligned errors. This risks subtly normalizing skewed synthetic-media demographics and reinforcing harmful stereotypes through high-quality, preferred-content presentation.

**Dual use.** Curated preference data can engineer rewards to amplify or suppress social attributes. Historical underrepresentation correction and bias mitigation use the same machinery as intentional exclusion, homogenization, or propaganda aligned with exclusionary worldviews. The alignment process is double-edged.

**Responsibility.** Aesthetic metrics alone are insufficient. Developers must continually monitor generated social composition using multi-granular, event-centric audits such as QX2. Preference-data curation warrants the same rigor as architecture design, so guiding human preferences reflect a diverse, equitable society instead of reproducing prejudice.

The supplied supplementary inventory comprises per-verb gender/ethnicity analyses, training-data effects, statistical analyses, and implementation/training details. A, C, D, F, and G are supplementary-item labels. WebVid-10M is the training dataset named in item A.

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

**Claims and evidence**

- Alignment can stabilize biased portrayals: Table 5, + A3J average L2V 0.9116 versus G4S 0.8034; Table 6, + M1Y Gender perfect stability 0.9830 versus 0.8850. The temporal reward label remains distinct.
- Preference curation can amplify or suppress gender tendencies: Table 7, Z6B/P3L average L2V 1.5280/-0.7448; Table 8, aligned averages 0.9584/0.3082 versus G4S 0.8034.
- Normalization, intentional exclusion, homogenization, and propaganda are stated risks; the paper reports no numerical deployment-impact results for them.

## Acknowledgment

*The acknowledgment identifies support and states that the authors’ views do not represent the supporting organizations.*

We thank the anonymous reviewers for thoughtful feedback and suggestions. Partial support: AFOSR MURI grant #FA9550-22-1-0380; Office of Naval Research (ONR), Award #N00014-23-1-2780; National Science Foundation, Award #IIS-2449768. Additional support: NVIDIA Academic Grant Program cloud credits and Coefficient Giving’s Technical AI Safety Research Program funding. Authors’ views and conclusions do not represent official policies or endorsements of AFOSR, ONR, the U.S. Government, the National Science Foundation, NVIDIA, or Coefficient Giving.
