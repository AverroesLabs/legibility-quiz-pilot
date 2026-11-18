# Reasoning Boosts Opinion Alignment in LLMs

Frédéric Berdoz · Yann Billeter · Yann Vonlanthen · Roger Wattenhofer  
ETH Zurich · Switzerland  
{fberdoz, ybilleter, yvonlanthen, wattenhofer}@ethz.ch

###### Abstract

LLMs’ versatility, generalization and text-to-text successes make them candidates for individual/group political preference modeling; statistical operation and limited causal understanding yield biased opinions under naive prompting. Structured reasoning trained through reinforcement learning produces profile-consistent answers across U.S., European and Swiss politics, remains competitive with strong baselines, and leaves bias requiring additional mechanisms for faithful political digital twins. Released methods and datasets establish a research baseline; digital democracies could use opinion models to shape fairer, more popular policies.

## 1 Introduction

Reasoning consistently improves opinion alignment. Survey-trained agents generate short rationales followed by choices, receiving rewards for matching respondents’ answers and learning survey-consistent positions on unseen questions. Recent reinforcement-learning gains—including o3/o4-mini (OpenAI, 2025) and DeepSeek-R1 (Guo et al., 2025), with new benchmark marks—motivate this treatment of opinion formation.

Individual digital twins voting on every issue require accurate political-behavior simulation; such simulation offers opportunities for understanding elections and improving policy and democratic processes (Li et al., 2024). Sophisticated LLM political discourse does not reliably represent human viewpoint diversity (Santurkar et al., 2023; Qu and Wang, 2024; Yu et al., 2025; Qi et al., 2024). Capturing diverse individuals’ political reasoning remains difficult.

Demographic and affiliation prompts exploit learned demographic–opinion correlations; increasingly sophisticated variants still miss real distributions and respond unstably across prompts and demographic groups (Santurkar et al., 2023; Ball et al., 2025). Preference consistency instead uses known individual opinions rather than demographic proxies. Existing approaches do not readily simulate individual preferences; interview transcripts yield strong individual-persona performance (Park et al., 2024), with prohibitive collection costs and impractical interviewing at scale.

Large surveys and Voting Advice Applications—online voter–party/candidate matching platforms—capture political positions. They lack interviews’ narrative richness and textual signal but structurally represent population opinions. Their established use in bias evaluation (Santurkar et al., 2023; Argyle et al., 2023; Haller et al., 2025) and group simulation (Argyle et al., 2023; Cao et al., 2025) contrasts with limited adoption for individual-agent training.

The contributions comprise QX2-based explicit reasoning from survey responses, a released benchmark of authentic German-party, U.S.-voter and Swiss-candidate opinions, and answer-inversion analysis of individuals with known positions revealing systematic ideological variation in alignment quality.

**Figure 1: QX2 opinion alignment.** Public surveys supply individual preference profiles; initial fine-tuning on synthetic statements and ground-truth answers establishes the reasoning template without full opinion alignment, followed by QX2 with formatting/correctness rewards to improve reasoning.

## 2 Related Work

Political-persona adoption supports viewpoint-specific answers; demographic personas represent shared traits, including occupation or ethnicity, character personas represent well-known individuals, and individualized personas use personalized digital profiles (Chen et al., 2024; Tseng et al., 2024). Individual modeling makes the last category most relevant, although demographic work predominates and only a handful of studies employ individualized methods.

##### Demographic personas.

Basic affiliation prompting includes “You are a democrat. What is your opinion on…” (Santurkar et al., 2023); richer prompts add age, education and location (Argyle et al., 2023; Santurkar et al., 2023; Hwang et al., 2023; Sun et al., 2024; Wang et al., 2025), while belief-network demonstrations supply related political positions (Chuang et al., 2024). Gudiño et al. (2024) combine demographics and demonstrations, whereas Stammbach et al. (2024) integrate prompting with a preference-training procedure (N8R = odds ratio preference optimization (ORPO)) from Hong et al. (2024) to align outputs with target positions.

##### Demographic limitations.

Santurkar et al. (2023) distinguish representativeness—default distributions mismatch populations—steerability—demographic prompts inadequately control outputs—and consistency—bias varies unpredictably across topics and contexts. Systematic biases include unpredictable groupwise accuracy (Argyle et al., 2023; Santurkar et al., 2023); approximate overall distributions coexist with missing within-subgroup natural variance and prompt sensitivity (Sun et al., 2024; Ball et al., 2025).

##### Beyond demographics.

Park et al. (2024) outperform demographic baselines on public-opinion surveys using interviews, constrained by transcript availability. Haller et al. (2025) supply respondents’ question–answer pairs in context and predict answers to test questions, targeting bias identification rather than opinion modeling.

**Table 1: T9A trained on an X8J individual in M8C.** Training changes disagreement into agreement while retaining “equal opportunity” arguments under different framing.

| Item | Content |
|---|---|
| Question | Do you agree that it is not really that big a problem if some people have more of a chance in life than others? |
| Before: base-model prompting | No. I don’t think that’s right. Everyone deserves an equal shot at succeeding in life, no matter where they’re from or who their parents are. If some people have more opportunities and resources, it’s not fair to others. We should be working to level the playing field and give everyone a fair chance to succeed. |
| After: ZP4 | `<reasoning>` Some inequality in wealth and opportunity is inevitable and indeed necessary to provide incentives for hard work and innovation – the issue is not eliminating all differences but rather ensuring that everyone has access to basic necessities and opportunities to improve themselves – other countries with significant equality have still struggled with poverty and stagnation. `</reasoning>` |
| Final answer | `<answer>A) Yes</answer>` |

##### Training reasoning.

Chain-of-Thought prompting (Wei et al., 2022) and “Let’s think step by step” (Kojima et al., 2022) significantly increase accuracy; search and reinforcement learning increasingly train high-quality traces instead of merely prompting them (Xu et al., 2025). A Proximal policy optimization variant (PPO; Schulman et al., 2017), using group-relative reward normalization instead of a learned advantage-estimating value function (QX2 = Group-relative policy optimization (GRPO)), samples multiple outputs per prompt and subtracts their reward mean before dividing by their reward standard deviation (Shao et al., 2024). Mathematical reasoning benchmarks show particular effectiveness and state-of-the-art results (Guo et al., 2025).

##### Reasoning for alignment.

Yu et al. (2025) combine demographic, ideological and temporal factors in a persona-based, chain-of-thought-inspired prompting framework calibrated with M8C; individual voting simulations are summed into aggregate predictions. They do not use reinforcement learning and target voting rather than general preference profiles. The present method could theoretically address their task, but representative-sample modeling currently incurs prohibitive computation.

## 3 Method

Given policy question \(q\), persona \(p\)—respondent or party—and stances \(Y=\{\texttt{No},\texttt{Neutral},\texttt{Yes}\}\), individual alignment learns \(p\)’s stance on \(q\); “Should the government increase funding for renewable energy?” is one such question.

### 3.1 Political Reasoning with QX2

Only a country label appears in the system prompt; no explicit persona representation is used, and correct question answering alone aligns the model with individuals. Training observations contain final stances but no reasoning:
\[
\mathcal D=\{(p_j,q_i,y^*_{ij})\},
\]
where \(y^*_{ij}\) is persona \(p_j\)’s observed stance on question \(q_i\). Agents must generate both ground-truth-matching answers and format-compliant traces:
\[
\texttt{<reasoning>[justification text]</reasoning><answer>[stance]</answer>}.
\tag{1}
\]

The composite reward evaluates tag placement (E6Z = Format Reward), reasoning-token deviation (F2B = Length Reward), and observed-response agreement (G9L = Correctness Reward), with the last providing the primary signal and the first rewarding correctly located schema tags.

For reasoning trace \(x_i\), token length \(L\), desired length \(L^*\), predicted answer \(y_i\), and ground-truth response \(y_i^*\):
\[
\begin{aligned}
\mathrm{F2B}&=-|L-L^*|,\\
\mathrm{G9L}&=\mathds{1}[y_i=y_i^*],\\
R(r_i,y_i^*)&=\alpha_1\mathrm{E6Z}+\alpha_2\mathrm{F2B}
+\alpha_3\mathrm{G9L}.
\end{aligned}
\]

| Reward detail | Value |
|---|---|
| E6Z per correctly placed tag | 1 |
| Number of tags; E6Z maximum | 4; 4 |
| F2B maximum, attained only at \(L=L^*\) | 0 |
| \(\alpha_1\) | 0.25 |
| \(\alpha_2\) | 0.01 |
| \(\alpha_3\) | 1 |

##### Optional supervised initialization.

An optional warm start (V7K = supervised fine-tuning (SFT)) follows Shao et al. (2024) by constructing chain-of-thought demonstrations in the exact rewarded schema, accelerating convergence and improving initial reasoning quality while reducing E6Z’s subsequent burden through structure pretraining and supplying a reasonable political-reasoning initialization.

**Table 2: Data.** Question counts are per candidate/party/respondent.

| Setting | Unit | #Units | Labels | Train Q | Test Q |
|---|---|---:|---|---:|---|
| J4Q (CH) | Candidate | 18 | {Yes, No} | 48 | 12 (topic-strat.) |
| R2V (DE) | Party | 6 | {Yes, Neutral, No} | see Table 5 | 30 (B5X) |
| M8C 2020 (US) | Respondent | 21 | {Yes, Neutral, No} | 67 | 12 (random) |

## 4 Experiments

### 4.1 Datasets

Data cover different countries and political systems and are released for reproducibility and future research.

**Table 3: Reasoning requires large datasets for effectiveness.** At \(T=1.0\), each of 8 stochastic runs computes a per-unit classification metric (D3G = macro-F1), averages it across units, and contributes to the reported mean and standard deviation, in percent.

| Base model | Method | J4Q | R2V | M8C |
|---|---|---:|---:|---:|
| Untrained baselines | C6W | \(50.0\) | \(33.33\) | \(33.33\) |
| | U9F | \(37.43\) | \(27.44\) | \(22.98\) |
| T9A | H3D | \(55.97\pm 5\) .66) | \(28.17\pm 0\) .02) | \(23.2\pm 1\) .79) |
| | N8R | \(43.53\pm 2\) .81) | \(43.29\pm 5\) .16) | \(34.84\pm 5\) .41) |
| | V7K | \(63.44\pm 1\) .34) | \(48.95\pm 3\) .56) | \(42.77(1.23)\) |
| | QX2 | \(55.14\pm 1\) .25) | \(37.29\pm 2\) .06) | \(34.55\pm 1\) .32) |
| | ZP4 | \(66.88(2.18)\) | \(52.53(4.05)\) | \(40.66\pm 0\) .91) |
| K6Y | H3D | \(60.48\pm 4\) .14) | \(26.19\pm 0\) .35) | \(23.2\pm 1\) .79) |
| | N8R | \(23.87\pm 2\) .07) | \(25.25\pm 3\) .58) | \(26.95\pm 1\) .36) |
| | V7K | \(61.08\pm 2\) .74) | \(42.91\pm 3\) .44) | \(35.14\pm 1\) .70) |
| | QX2 | \(60.64\pm 2\) .01) | \(31.42\pm 1\) .46) | \(31.47\pm 1\) .06) |
| | ZP4 | \(65.11(3.33)\) | \(49.38(1.93)\) | \(38.44(0.40)\) |
| W2E | H3D | \(66.16\pm 4\) .2) | \(26.19\pm 0\) .35) | \(19.23\pm 2\) .91) |
| | N8R | \(23.31\pm 2\) .11) | \(24.73\pm 3\) .29) | \(24.25\pm 2\) .28) |
| | V7K | \(67.63\pm 1\) .91) | \(51.86\pm 2\) .58) | \(39.15\pm 0\) .66) |
| | QX2 | \(60.56\pm 1\) .93) | \(51.00\pm 3\) .10) | \(43.79\pm 1\) .06) |
| | ZP4 | \(70.73(2.21)\) | \(53.21(3.19)\) | \(45.43(1.11)\) |

##### German parties.

Official training data (R2V = Wahl-O-Mat (WoM)), version 26 March 2025, aggregate federal, state and European election party positions over 2021–2025, with agree/neutral/disagree labels mapped to {Yes, Neutral, No} and an explanatory comment for every item. The six major parties are CDU/CSU, SPD, Grüne, FDP, Die Linke and AfD. The out-of-domain test questionnaire (B5X = EU&I), from 2024 (Chalkidis, 2024), is recoded to the same labels. Per-party training counts: Appendix A.

##### U.S. respondents.

The 2020 Time Series source (M8C = American National Election Studies (ANES)), credited to American National Election Studies (2021), supplies 79 policy items after excluding demographics, knowledge, candidate evaluations and thermometers. Heterogeneous response formats are reduced to {Yes, Neutral, No} through a clearly-positive/negative-only agreement/disagreement scheme (I7V = conservative mapping) and a scheme collapsing most positive responses to Yes (O2K = aggressive mapping), with I7V used unless stated otherwise; scheme definitions: Appendix D; recoding robustness: Section E.2. Items are split randomly. Three respondents per answer option on the 7-point liberal–conservative self-placement question permit ideological analysis.

##### Swiss candidates.

The English 2023 Swiss national-election questionnaire (J4Q = smartvote) contains 75 items, including 60 policy questions across 12 topics, and its four-point Likert responses collapse Yes/Rather yes into Yes and Rather no/No into No. Sampling takes three candidates from each of six parties holding at least two National Council seats. One question per topic is held out; the remainder train the agents.

##### Spectrum groups.

Cross-context comparison assigns parties in J4Q/R2V and M8C ideologies to broad groups (P7N = Left), (A4U = Center), and (X8J = Right), with detailed party and ideology assignments and additional Swiss-party information in Appendix A.

**Figure 2: Agent positions.** The first two principal components use all standing Swiss candidates as small dots; included candidates are big dots, agents gold, individual displacements black lines, and group-average distortions big arrows. Axes represent left–right and conservative–liberal positions respectively.

### 4.2 Setup

One model is trained per candidate/party/respondent; the design tests scale and reasoning-pretraining advantages. Training details and hyperparameters: Appendix F.

##### Methods

Three configurations are reported: QX2 on survey questions and answers, and the initialized combination (ZP4 = SFT+GRPO), with synthetic arguments supporting both positive and negative stances for every J4Q/M8C policy question generated by a larger model (Y3H = Llama 3.1 70B) from Grattafiori et al. (2024) for V7K; synthetic-argument prompting: Section A.4.

##### Backbones

Open-weight backbones (T9A = Llama 3.1 8B), credited to Grattafiori et al. (2024), (K6Y = Qwen3 8B), credited to Yang et al. (2025), and (W2E = Magistral 24B), credited to Rastogi et al. (2025), test scale and prior reasoning training, with K6Y/W2E pretrained to reason and T9A a non-reasoning counterpart for comparable-scale comparison. Every training-involving experiment uses 4-bit quantization.

##### Baselines

The naive uniform-answer selector (C6W = random) and unit-training-set modal-answer predictor (U9F = majority) accompany the contextual comparator (H3D = icl), which follows Haller et al. (2025) by providing question–answer pairs before an unseen question and supplies all same-topic training questions, or a random subset under context-length limits. N8R uses Stammbach et al. (2024)’s hyperparameters.

### 4.3 Results

ZP4 outperforms naive and V7K baselines across datasets except T9A on M8C; it is comparable to or better than H3D on M8C/J4Q and better on R2V. J4Q scores are considerably higher than M8C/R2V, plausibly reflecting binary answers without Neutral; R2V exceeds M8C, plausibly reflecting substantially more training samples. J4Q peaks highest and M8C is hardest.

Native reasoning support gives no consistent gain: T9A exceeds K6Y on R2V/M8C, W2E performs best overall, and QX2 alone trails ZP4, suggesting improved training dynamics from supervised initialization. Neutral-containing datasets perform worse; uncertainty, social desirability and strategic non-commitment are aggregated into that class and may impede learning. Additional accuracy, significance and Neutral-excluded results: Section E.1.

**Figure 3: Unequal learnability.** F1 error bars show within-group variance; a more detailed version in Appendix C retains original parties and ideology groups. ZP4 typically works best, while every training method underperforms on centrist and right-leaning groups.

### 4.4 Agent reasoning

Agents generally produce coherent arguments for or against proposed measures implying their final answers. Failures include illogical or hallucinated elements and consistency with an individual on one topic but not another. Additional traces and failure examples: Appendix B.

### 4.5 Ideology

T9A/ZP4 groupwise F1 shows a clear J4Q advantage for P7N over A4U/X8J. R2V/M8C patterns are similar but weaker; R2V P7N/A4U are roughly equal, and M8C performance is generally lower. V7K averages exceed all other methods for X8J in J4Q/M8C, reinforcing difficulties with right-wing profiles.

Known general-purpose LLM left-leaning tendencies (Hartmann et al., 2023; Exler et al., 2025), T9A’s embedded biases, or intrinsic survey-signal learnability differences may account for disparities. Relatively strong R2V X8J performance may benefit from larger training sets (cf. Section E.4). Ideology affects preference learnability under either explanation.

### 4.6 Neutral

Neutral recall is worst for every M8C individual. X8J has the highest Neutral response rate and suffers most from prediction difficulty; A4U also tends toward more Neutral responses. Removing Neutral instances and recomputing F1 considerably improves all groups, narrows P7N–A4U differences and improves ZP4 relative to H3D, without eliminating P7N–X8J differences.

O2K reproduces the negative performance–Neutral-rate correlation but gives nearly identical A4U/X8J performance; recoding may partly produce the disparity. Numerous Neutral examples may depress the other classes during training; model bias and recoding artifacts remain possible influences.

**Figure 4: M8C, T9A/ZP4.** Left panels regress test-set Neutral base rate against F1 above and accuracy below; strong negative correlations are significant at the \(5\%\) level. Upper-right displays class recall averaged across individuals, lower-right F1 before Neutral removal as solid and after as shaded. Regression details: Section E.3; alternative-recoding analyses: Section E.2.

### 4.7 Representation

Agents do not perfectly represent their candidates. No overall left shift is observed despite documented LLM left-libertarian biases (Exler et al., 2025; Hartmann et al., 2023; Rozado, 2024): positions shift center-right on the horizontal spectrum and centerward on the vertical spectrum; group-average distortions show an overall conservative, negative-\(y\) trend without clear left–right bias. X8J individuals become more left-wing, P7N/A4U individuals more right-wing and conservative.

The semantic representativity analysis uses the political interpretation of J4Q principal components documented by Germann et al. (2015); agents are projected into the space computed from the full candidate population, with group-averaged human–agent distances displayed centrally.

**Figure 5: Biased initialization data.** T9A/ZP4 F1 under progressive-biased data occupies the left panel, default data the middle, conservative-biased data the right.

### 4.8 Initialization data

Ideological bias primarily harms underrepresented perspectives rather than systematically helping overrepresented ones. Progressive arguments strongly impair X8J with inconsistent P7N effects; conservative arguments impair P7N without consistent X8J gains. Adverse bias harms X8J more severely than P7N.

Two additional argument sets, generated with progressive and conservative prompts by another model (S5Q = Llama 3.3 70B), supply these conditions, with the synthetic-argument prompting strategy in Section A.4. Positional effects agree with F1 effects: underrepresented groups shift most, X8J more than P7N. Conservative-biased and default data produce highly similar shifts; base-model alignment rather than initialization-data bias may primarily drive the default centerward displacement.

**Figure 6: Biased-data positions.** Left displays candidates and agents in J4Q principal-component space; right displays group-mean agent displacement relative to corresponding candidates.

### 4.9 Synthetic positions

X8J benefits from inversion; P7N/A4U worsen on average. Inverted X8J still does not reach original P7N F1, while P7N falls roughly to original X8J levels. A purely original-model-political-bias account would make inverted right-leaning candidates nearly as predictable as left-leaning candidates; the remaining difference suggests potentially intrinsically easier left-leaning profiles, requiring investigation.

The counterfactual inverts every J4Q answer, trains T9A on the resulting data, and computes each politician’s before/after F1 difference. Inversion approximately rotates positions \(180^\circ\) about the origin in Fig. 2 (see Section E.6), including conversion of right-wing to left-wing positions.

**Figure 7: Inversion.** Points show original F1, individuals ordered horizontally by their first J4Q principal component; the left-most X8J candidate is correctly placed (see Fig. 6). Arrows encode changes after flipping both training and test answers. P7N worsens individually as well as on average; X8J improvement still leaves its average below P7N. Conversion of P7N into X8J does not remove the disparity, consistent with potentially harder right-wing learning.

## 5 Discussion

##### Limitations.

Same-respondent day-to-day answer variation imposes noise; consistently near-\(1.0\) F1 is unlikely. Computationally intensive per-profile training restricts analysis to few individuals/parties. Suitable surveys are scarce, often lack enough items for individual agents, emphasize generic themes instead of concrete issues/priorities, and rarely align question sets across countries, limiting comparison.

##### Future work.

Group disparities require bias-mitigation research. Replacing per-agent training with one model conditioned on compact persona representations (Ning et al., 2024) could reduce computation and improve data utilization. Deployment requires confidence estimates for answer reliability, obtainable through extended reasoning or a dedicated value head. Purpose-built questionnaires and improved uncertainty handling may mitigate Neutral difficulties.

## 6 Conclusion

Direct known-opinion training of persona-conditioned agents treats answers as structured reasoning; ZP4 generally outperforms all baselines across Swiss candidates, German parties and U.S. voters, with difficult Neutral prediction and significant political-spectrum accuracy variation. The benchmark supports systematic LLM opinion modeling as an early step toward more representative AI-driven democracy.

## Ethics statement

Individual-opinion modeling amid rising mis- and disinformation is not exclusively beneficial and creates downstream risks. The technology’s inevitable development and these risks motivate responsible alignment research. Only anonymized surveys and public figures’ positions are used for privacy.

Systematic ideological disparities, consistent centrist/right-wing underperformance and neutral-position difficulty are significant system risks: premature deployment could disenfranchise political viewpoints and undermine democratic representation. We commit to transparent bias reporting and strongly caution against real-world applications until fairness issues are resolved. Proper bias mitigation remains a condition of our conviction that opinion modeling can enable societally beneficial digital democracies.

## Reproducibility statement

Code and datasets: https://github.com/ETH-DISCO/reasoning-boosts-llm-alignment

Released documentation includes training details and hyperparameters, principal-component implementation details (all PCA results: Section E.5), per-party training counts, recoding definitions, training-sample-size analysis and inversion details.

#### Author Contributions

Frédéric Berdoz conceived/designed the project, led experimental planning and defined research direction. Equally contributing Yann Billeter led technical development: dataset collection, training, evaluation and analysis. Yann Vonlanthen advised methodology and experimental design. Roger Wattenhofer supervised as principal investigator.

#### Acknowledgments

We thank Prof. Dr. Hans Gersbach for valuable project insights.
