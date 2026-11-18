# Reasoning Boosts Opinion Alignment in LLMs

Frédéric Berdoz · Yann Billeter · Yann Vonlanthen · Roger Wattenhofer  
ETH Zurich · Switzerland  
{fberdoz, ybilleter, yvonlanthen, wattenhofer}@ethz.ch

###### Abstract

Large language models’ (LLMs’) versatility, generalization and text-to-text successes make LLMs candidates for modeling individual and group political preferences. However, LLMs’ statistical operation and limited causal understanding yield biased opinions under naive prompting. Structured reasoning trained through reinforcement learning produces profile-consistent answers across U.S., European and Swiss politics and remains competitive with strong baselines. Nevertheless, bias remains, so faithful political digital twins require additional mechanisms. The released methods and datasets establish a research baseline. Ultimately, digital democracies could use opinion models to shape fairer, more popular policies.

## 1 Introduction

Reasoning consistently improves opinion alignment: alignment between a model’s answers and an individual’s known opinions. Survey-trained agents generate short rationales followed by choices and receive rewards for matching respondents’ answers. Through this training, the agents learn survey-consistent positions on unseen questions. Recent reinforcement-learning gains—including o3/o4-mini (OpenAI, 2025) and DeepSeek-R1 (Guo et al., 2025), with new benchmark marks—motivate this treatment of opinion formation.

Individual digital twins that vote on every issue require accurate political-behavior simulation. Such simulation offers opportunities for understanding elections and improving policy and democratic processes (Li et al., 2024). However, sophisticated LLM political discourse does not reliably represent human viewpoint diversity (Santurkar et al., 2023; Qu and Wang, 2024; Yu et al., 2025; Qi et al., 2024). Capturing diverse individuals’ political reasoning therefore remains difficult.

Demographic and affiliation prompts exploit learned demographic–opinion correlations. However, increasingly sophisticated variants still miss real opinion distributions and respond unstably across prompts and demographic groups (Santurkar et al., 2023; Ball et al., 2025). Preference consistency instead uses known individual opinions rather than demographic proxies. Existing approaches do not readily simulate individual preferences. Interview transcripts yield strong individual-persona performance (Park et al., 2024), but transcript collection has prohibitive costs, making interviewing impractical at scale.

Large surveys and Voting Advice Applications—online platforms that match voters with parties or candidates—capture political positions. Although surveys and Voting Advice Applications lack interviews’ narrative richness and textual signal, these sources structurally represent population opinions. Surveys and Voting Advice Applications have an established role in bias evaluation (Santurkar et al., 2023; Argyle et al., 2023; Haller et al., 2025) and group simulation (Argyle et al., 2023; Cao et al., 2025). In contrast, adoption for individual-agent training remains limited.

The contributions comprise explicit reasoning from survey responses using Group-relative policy optimization (GRPO, denoted QX2), a released benchmark of authentic German-party, U.S.-voter and Swiss-candidate opinions, and an answer-inversion analysis of individuals with known positions. The answer-inversion analysis reveals systematic ideological variation in alignment quality. Group-relative policy optimization is the reinforcement-learning procedure described in Section 2 and used in Section 3.

**Figure 1: QX2 opinion alignment.** Public surveys supply individual preference profiles; initial fine-tuning on synthetic statements and ground-truth answers establishes the reasoning template without full opinion alignment, followed by QX2 with formatting/correctness rewards to improve reasoning.

Here, QX2 denotes Group-relative policy optimization. Thus, Figure 1 separates establishment of the reasoning template through initial fine-tuning from subsequent reinforcement learning for opinion alignment.

## 2 Related Work

Political-persona adoption supports viewpoint-specific answers. Demographic personas represent shared traits, including occupation or ethnicity; character personas represent well-known individuals; and individualized personas use personalized digital profiles (Chen et al., 2024; Tseng et al., 2024). Because the present work models individuals, individualized personas are the most relevant category. However, demographic work predominates, and only a handful of studies employ individualized methods.

##### Demographic personas.

Basic affiliation prompting includes “You are a democrat. What is your opinion on…” (Santurkar et al., 2023). Richer prompts add age, education and location (Argyle et al., 2023; Santurkar et al., 2023; Hwang et al., 2023; Sun et al., 2024; Wang et al., 2025), while belief-network demonstrations supply related political positions (Chuang et al., 2024). Gudiño et al. (2024) combine demographics and demonstrations. In contrast to prompting alone, Stammbach et al. (2024) integrate prompting with a preference-training procedure, odds ratio preference optimization (ORPO, denoted N8R), from Hong et al. (2024), to align outputs with target positions.

##### Demographic limitations.

Santurkar et al. (2023) distinguish three limitations:

- **Representativeness:** default output distributions mismatch populations.
- **Steerability:** demographic prompts inadequately control outputs.
- **Consistency:** bias varies unpredictably across topics and contexts.

These limitations include unpredictable groupwise accuracy (Argyle et al., 2023; Santurkar et al., 2023). Moreover, approximate agreement with overall distributions can coexist with missing natural variance within subgroups and with prompt sensitivity (Sun et al., 2024; Ball et al., 2025).

##### Beyond demographics.

Park et al. (2024) use interviews to outperform demographic baselines on public-opinion surveys, but transcript availability constrains the approach. Haller et al. (2025) instead supply respondents’ question–answer pairs in context and predict answers to test questions. However, Haller et al. target bias identification rather than opinion modeling.

Table 1 illustrates an individual-modeling result. In the table, T9A denotes the Llama 3.1 8B base model (Grattafiori et al., 2024); M8C denotes the American National Election Studies (ANES) dataset, here from 2020; and X8J denotes the broad Right political-spectrum group. ZP4 denotes supervised fine-tuning followed by Group-relative policy optimization (SFT+GRPO), where supervised fine-tuning (SFT) initializes the model before reinforcement learning. Table 1 contrasts base-model prompting with this initialized reinforcement-learning method.

**Table 1: T9A trained on an X8J individual in M8C.** Training changes disagreement into agreement while retaining “equal opportunity” arguments under different framing.

| Item | Content |
|---|---|
| Question | Do you agree that it is not really that big a problem if some people have more of a chance in life than others? |
| Before: base-model prompting | No. I don’t think that’s right. Everyone deserves an equal shot at succeeding in life, no matter where they’re from or who their parents are. If some people have more opportunities and resources, it’s not fair to others. We should be working to level the playing field and give everyone a fair chance to succeed. |
| After: ZP4 | `<reasoning>` Some inequality in wealth and opportunity is inevitable and indeed necessary to provide incentives for hard work and innovation – the issue is not eliminating all differences but rather ensuring that everyone has access to basic necessities and opportunities to improve themselves – other countries with significant equality have still struggled with poverty and stagnation. `</reasoning>` |
| Final answer | `<answer>A) Yes</answer>` |

##### Training reasoning.

Chain-of-Thought prompting, which elicits intermediate reasoning before an answer (Wei et al., 2022), and “Let’s think step by step” (Kojima et al., 2022) significantly increase accuracy. Search and reinforcement learning increasingly train high-quality reasoning traces rather than merely prompting the traces (Xu et al., 2025).

Group-relative policy optimization, denoted QX2, is a variant of Proximal policy optimization (PPO; Schulman et al., 2017). Instead of using a learned value function to estimate advantages, Group-relative policy optimization uses group-relative reward normalization: the procedure samples multiple outputs per prompt, subtracts the outputs’ mean reward from each reward, and divides by the outputs’ reward standard deviation (Shao et al., 2024). Mathematical reasoning benchmarks show particular effectiveness and state-of-the-art results (Guo et al., 2025).

##### Reasoning for alignment.

Yu et al. (2025) combine demographic, ideological and temporal factors in a persona-based, chain-of-thought-inspired prompting framework calibrated with American National Election Studies data (M8C). Individual voting simulations are then summed into aggregate predictions. Unlike the present work, Yu et al. do not use reinforcement learning and target voting rather than general preference profiles. The present method could theoretically address their task; however, modeling a representative sample currently incurs prohibitive computation.

## 3 Method

Let \(q\) denote a policy question, \(p\) a persona—either a respondent or a party—and \(Y=\{\texttt{No},\texttt{Neutral},\texttt{Yes}\}\) the set of possible stances. Individual alignment learns persona \(p\)’s stance on question \(q\). For example, \(q\) could be “Should the government increase funding for renewable energy?”

### 3.1 Political Reasoning with QX2

The system prompt contains only a country label. No explicit persona representation is used; instead, correct question answering alone aligns the model with individuals.

The training dataset is
\[
\mathcal D=\{(p_j,q_i,y^*_{ij})\},
\]
where \(j\) indexes personas, \(i\) indexes questions, \(p_j\) is a persona, \(q_i\) is a question, and \(y^*_{ij}\) is persona \(p_j\)’s observed stance on question \(q_i\). Each training observation therefore contains a final stance but no reasoning.

Although training observations contain no reasoning, agents must generate both a ground-truth-matching answer and a format-compliant reasoning trace. The required output schema is
\[
\texttt{<reasoning>[justification text]</reasoning><answer>[stance]</answer>}.
\tag{1}
\]

The composite reward has three components:

- **Format Reward (E6Z):** evaluates tag placement and rewards correctly located schema tags.
- **Length Reward (F2B):** evaluates deviation from the desired number of reasoning tokens.
- **Correctness Reward (G9L):** evaluates agreement with the observed response and provides the primary training signal.

For the reward definitions below, \(x_i\) denotes a reasoning trace, \(L\) its token length, \(L^*\) the desired token length, \(y_i\) the predicted answer, and \(y_i^*\) the ground-truth response. The symbol \(r_i\) denotes the generated response whose composite reward is \(R(r_i,y_i^*)\). The indicator \(\mathds{1}[\cdot]\) equals 1 when its condition holds and 0 otherwise. The coefficients \(\alpha_1,\alpha_2,\alpha_3\) weight the three reward components:
\[
\begin{aligned}
\mathrm{F2B}&=-|L-L^*|,\\
\mathrm{G9L}&=\mathds{1}[y_i=y_i^*],\\
R(r_i,y_i^*)&=\alpha_1\mathrm{E6Z}+\alpha_2\mathrm{F2B}
+\alpha_3\mathrm{G9L}.
\end{aligned}
\]

Thus, the Length Reward is greatest when the reasoning trace has exactly the desired token length, while the Correctness Reward is greatest when the predicted answer matches the observed response. The component values and weights are:

| Reward detail | Value |
|---|---|
| E6Z per correctly placed tag | 1 |
| Number of tags; E6Z maximum | 4; 4 |
| F2B maximum, attained only at \(L=L^*\) | 0 |
| \(\alpha_1\) | 0.25 |
| \(\alpha_2\) | 0.01 |
| \(\alpha_3\) | 1 |

##### Optional supervised initialization.

An optional warm start uses supervised fine-tuning, denoted V7K, following Shao et al. (2024). This initialization constructs chain-of-thought demonstrations in the exact rewarded schema. Consequently, supervised initialization accelerates convergence and improves initial reasoning quality. Structure pretraining also reduces the subsequent burden on the Format Reward and supplies a reasonable political-reasoning initialization.

Table 2 introduces the dataset codes used throughout the experiments: J4Q denotes smartvote from Switzerland (CH), R2V denotes Wahl-O-Mat from Germany (DE), and M8C denotes American National Election Studies from the United States (US). The German-party test questionnaire, EU&I, is denoted B5X. “Train Q” and “Test Q” count training and test questions, respectively; “topic-strat.” denotes a topic-stratified split.

**Table 2: Data.** Question counts are per candidate/party/respondent.

| Setting | Unit | #Units | Labels | Train Q | Test Q |
|---|---|---:|---|---:|---|
| J4Q (CH) | Candidate | 18 | {Yes, No} | 48 | 12 (topic-strat.) |
| R2V (DE) | Party | 6 | {Yes, Neutral, No} | see Table 5 | 30 (B5X) |
| M8C 2020 (US) | Respondent | 21 | {Yes, Neutral, No} | 67 | 12 (random) |

## 4 Experiments

### 4.1 Datasets

The datasets cover different countries and political systems and are released for reproducibility and future research. Table 2 gives the units, labels and question counts; Table 3 reports performance on these datasets.

To make Table 3 readable before the detailed setup in Section 4.2, the table uses the following model and method codes:

- **Base models:** T9A is Llama 3.1 8B (Grattafiori et al., 2024), K6Y is Qwen3 8B (Yang et al., 2025), and W2E is Magistral 24B (Rastogi et al., 2025).
- **Untrained baselines:** C6W is the random uniform-answer selector; U9F is the majority predictor, which selects the modal answer in the unit’s training set.
- **Contextual baseline:** H3D is in-context learning, abbreviated icl, which supplies training question–answer pairs before an unseen question.
- **Training methods:** N8R is odds ratio preference optimization; V7K is supervised fine-tuning; QX2 is Group-relative policy optimization; and ZP4 is supervised fine-tuning followed by Group-relative policy optimization.

The classification metric is macro-F1, denoted D3G: the unweighted average across classes of F1, where F1 is the harmonic mean of precision and recall. In Table 3, \(T\) denotes the sampling temperature. The three dataset columns are smartvote (J4Q), Wahl-O-Mat (R2V), and American National Election Studies (M8C).

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

Official training data come from Wahl-O-Mat (WoM, denoted R2V), version 26 March 2025. The data aggregate party positions from federal, state and European elections over 2021–2025. Agree/neutral/disagree labels are mapped to {Yes, Neutral, No}, and every item includes an explanatory comment.

The six major parties are CDU/CSU, SPD, Grüne, FDP, Die Linke and AfD. CDU/CSU denotes the Christian Democratic Union/Christian Social Union; SPD denotes the Social Democratic Party; FDP denotes the Free Democratic Party; and AfD denotes Alternative for Germany.

The out-of-domain test questionnaire is EU&I, denoted B5X, from 2024 (Chalkidis, 2024). The EU&I responses are recoded to the same {Yes, Neutral, No} labels. Per-party training counts appear in Appendix A; Table 2 points to Table 5 for these counts.

##### U.S. respondents.

The American National Election Studies 2020 Time Series source (ANES, denoted M8C), credited to American National Election Studies (2021), supplies 79 policy items after demographics, knowledge, candidate evaluations and thermometers are excluded.

Because the response formats are heterogeneous, responses are reduced to {Yes, Neutral, No} using two recoding schemes:

- **Conservative mapping (I7V):** uses a clearly-positive/negative-only agreement/disagreement scheme.
- **Aggressive mapping (O2K):** collapses most positive responses to Yes.

The conservative mapping is used unless stated otherwise. Appendix D defines the schemes, and Section E.2 examines recoding robustness.

The items are split randomly into training and test questions, giving the 67 training and 12 test questions per respondent shown in Table 2. To permit ideological analysis, the sample includes three respondents per answer option on the 7-point liberal–conservative self-placement question, yielding the 21 respondents shown in Table 2.

##### Swiss candidates.

The English 2023 Swiss national-election questionnaire comes from smartvote, denoted J4Q. The questionnaire contains 75 items, including 60 policy questions across 12 topics. Four-point Likert responses are collapsed into binary labels: Yes/Rather yes becomes Yes, and Rather no/No becomes No.

Sampling takes three candidates from each of six parties holding at least two National Council seats, yielding the 18 candidates shown in Table 2. One policy question per topic is held out, and the remaining policy questions train the agents. Consequently, Table 2 reports 12 topic-stratified test questions and 48 training questions per candidate.

##### Spectrum groups.

For cross-context comparison, parties in smartvote and Wahl-O-Mat and respondent ideologies in American National Election Studies are assigned to three broad groups: Left (P7N), Center (A4U), and Right (X8J). Appendix A gives the detailed party and ideology assignments and additional Swiss-party information.

Figure 2 represents Swiss candidates and agents using principal component analysis (PCA), which constructs the displayed component space from the full candidate population. The political interpretation of the smartvote principal components follows Germann et al. (2015): the first two components represent left–right and conservative–liberal positions, respectively. Agents are projected into this candidate-derived space.

**Figure 2: Agent positions.** The first two principal components use all standing Swiss candidates as small dots; included candidates are big dots, agents gold, individual displacements black lines, and group-average distortions big arrows. Axes represent left–right and conservative–liberal positions respectively.

### 4.2 Setup

One model is trained per candidate, party or respondent. This design tests the advantages of model scale and prior reasoning training. Appendix F gives training details and hyperparameters.

##### Methods

Three configurations are reported: supervised fine-tuning alone (V7K), Group-relative policy optimization on survey questions and answers (QX2), and their initialized combination, supervised fine-tuning followed by Group-relative policy optimization (ZP4).

For supervised fine-tuning, a larger model—Llama 3.1 70B, denoted Y3H, from Grattafiori et al. (2024)—generates synthetic arguments supporting both positive and negative stances for every smartvote and American National Election Studies policy question. These arguments provide the supervised initialization used in the combined method. Section A.4 gives the synthetic-argument prompting strategy.

##### Backbones

The open-weight backbones are Llama 3.1 8B (T9A), credited to Grattafiori et al. (2024); Qwen3 8B (K6Y), credited to Yang et al. (2025); and Magistral 24B (W2E), credited to Rastogi et al. (2025). The backbones test model scale and prior reasoning training. Qwen3 8B and Magistral 24B are pretrained to reason, whereas Llama 3.1 8B is a non-reasoning counterpart that permits a comparable-scale comparison with Qwen3 8B. Every experiment involving training uses 4-bit quantization.

##### Baselines

The naive random baseline (C6W) selects answers uniformly. The majority baseline (U9F) predicts the modal answer in the candidate’s, party’s or respondent’s training set.

The contextual comparator uses in-context learning (icl, denoted H3D), following Haller et al. (2025): question–answer pairs are provided before an unseen question. The comparator receives all same-topic training questions or, when context-length limits prevent this, a random subset.

The odds ratio preference optimization baseline (N8R) uses Stammbach et al. (2024)’s hyperparameters.

### 4.3 Results

The initialized combination of supervised fine-tuning and Group-relative policy optimization (ZP4) outperforms the naive baselines and supervised fine-tuning alone across datasets, except for Llama 3.1 8B on American National Election Studies. The relevant comparisons are Table 3’s ZP4 rows against the C6W, U9F and V7K rows; the exception is the T9A/M8C comparison between ZP4 and V7K. The initialized combination is also comparable to or better than in-context learning on American National Election Studies and smartvote, and better on Wahl-O-Mat, as shown by Table 3’s ZP4 and H3D rows for each backbone.

Smartvote scores are considerably higher than American National Election Studies and Wahl-O-Mat scores; compare the J4Q column with the M8C and R2V columns in Table 3. This difference plausibly reflects smartvote’s binary answers without Neutral, as specified in Table 2. Wahl-O-Mat exceeds American National Election Studies, plausibly because Wahl-O-Mat supplies substantially more training samples; per-party counts are given in Appendix A and Table 5. Thus, smartvote reaches the highest peak scores, while American National Election Studies is the hardest setting.

Native reasoning support gives no consistent gain. For example, Table 3’s initialized-combination rows show that the non-reasoning Llama 3.1 8B exceeds Qwen3 8B on Wahl-O-Mat and American National Election Studies. Magistral 24B performs best overall, as shown by the W2E/ZP4 row. Moreover, Group-relative policy optimization alone trails the initialized combination across the Table 3 QX2 and ZP4 rows. This comparison suggests improved training dynamics from supervised initialization.

The datasets containing Neutral perform worse than the binary smartvote dataset. Uncertainty, social desirability and strategic non-commitment are aggregated into the Neutral class and may therefore impede learning. Section E.1 provides additional accuracy, significance and Neutral-excluded results.

Figure 3 compares performance across the Left, Center and Right spectrum groups defined in Section 4.1.

**Figure 3: Unequal learnability.** F1 error bars show within-group variance; a more detailed version in Appendix C retains original parties and ideology groups. ZP4 typically works best, while every training method underperforms on centrist and right-leaning groups.

Here, ZP4 is supervised fine-tuning followed by Group-relative policy optimization. Section 4.5 discusses the group disparities shown in Figure 3.

### 4.4 Agent reasoning

Agents generally produce coherent arguments for or against proposed measures, and those arguments imply the agents’ final answers. However, failures include illogical or hallucinated elements. Agents can also be consistent with an individual on one topic but not another. Appendix B provides additional traces and failure examples.

### 4.5 Ideology

For Llama 3.1 8B trained with the initialized combination (T9A/ZP4), groupwise F1 shows a clear smartvote advantage for the Left group over the Center and Right groups; see Figure 3 and the more detailed version in Appendix C. Wahl-O-Mat and American National Election Studies show similar but weaker patterns. In Wahl-O-Mat, Left and Center performance are roughly equal, while American National Election Studies performance is generally lower.

Supervised fine-tuning alone has higher average performance than every other method for the Right group in smartvote and American National Election Studies; these are the V7K comparisons in Figure 3. This result reinforces the difficulty of aligning right-wing profiles.

Several possible explanations remain: known general-purpose LLM left-leaning tendencies (Hartmann et al., 2023; Exler et al., 2025), embedded biases in Llama 3.1 8B, or intrinsic differences in how learnable the survey signals are. The relatively strong Wahl-O-Mat performance for the Right group may benefit from larger training sets; see Section E.4. Whether disparities arise from model bias or from survey-signal learnability, ideology affects preference learnability.

### 4.6 Neutral

The analysis in Figure 4 uses American National Election Studies with Llama 3.1 8B trained by supervised fine-tuning followed by Group-relative policy optimization (M8C, T9A/ZP4). Class recall measures how often responses of a given ground-truth class are correctly predicted. For every individual, Neutral has the worst recall.

The Right group has the highest Neutral response rate and suffers most from Neutral-prediction difficulty. The Center group also tends toward more Neutral responses. Removing Neutral instances and recomputing F1 considerably improves all groups, narrows Left–Center differences, and improves the initialized combination relative to in-context learning. However, removing Neutral does not eliminate Left–Right differences; see Figure 4’s before/after F1 comparison.

The aggressive recoding scheme, which collapses most positive responses to Yes, reproduces the negative correlation between performance and Neutral response rate. However, aggressive recoding gives nearly identical Center and Right performance. Consequently, recoding may partly produce the disparity. Numerous Neutral examples may also depress the other classes during training; model bias and recoding artifacts remain possible influences. Section E.2 provides the alternative-recoding analyses.

**Figure 4: M8C, T9A/ZP4.** Left panels regress test-set Neutral base rate against F1 above and accuracy below; strong negative correlations are significant at the \(5\%\) level. Upper-right displays class recall averaged across individuals, lower-right F1 before Neutral removal as solid and after as shaded. Regression details: Section E.3; alternative-recoding analyses: Section E.2.

In this caption, the Neutral base rate is the proportion of test responses labeled Neutral. M8C denotes American National Election Studies, and T9A/ZP4 denotes Llama 3.1 8B trained with the initialized combination.

### 4.7 Representation

The semantic representativity analysis uses the political interpretation of smartvote principal components documented by Germann et al. (2015). As in Figure 2, agents are projected into the space computed from the full candidate population, and group-averaged human–agent distances are displayed centrally. This projection permits comparison of candidate and agent political positions.

Agents do not perfectly represent their candidates. Despite documented LLM left-libertarian biases (Exler et al., 2025; Hartmann et al., 2023; Rozado, 2024), Figure 2 shows no overall left shift: positions shift center-right on the horizontal spectrum and centerward on the vertical spectrum. Group-average distortions show an overall conservative, negative-\(y\) trend without clear left–right bias, where \(y\) denotes the vertical coordinate in the principal-component plot. Right-group individuals become more left-wing, whereas Left- and Center-group individuals become more right-wing and conservative.

Figure 5 introduces the initialization-data conditions examined in Section 4.8. The default condition uses the initialization arguments described in Section 4.2; the progressive-biased and conservative-biased conditions use two additional argument sets generated with corresponding prompts.

**Figure 5: Biased initialization data.** T9A/ZP4 F1 under progressive-biased data occupies the left panel, default data the middle, conservative-biased data the right.

Here, T9A/ZP4 denotes Llama 3.1 8B trained with supervised fine-tuning followed by Group-relative policy optimization.

### 4.8 Initialization data

To examine initialization-data bias, two additional argument sets are generated with progressive and conservative prompts by another model: Llama 3.3 70B, denoted S5Q. These argument sets supply the progressive-biased and conservative-biased conditions in Figure 5. Section A.4 describes the synthetic-argument prompting strategy.

The Figure 5 comparisons show that ideological bias primarily harms underrepresented perspectives rather than systematically helping overrepresented perspectives. Progressive arguments strongly impair the Right group, while effects on the Left group are inconsistent. Conversely, conservative arguments impair the Left group without consistent gains for the Right group. Adverse bias harms the Right group more severely than the Left group.

The positional effects in Figure 6 agree with the F1 effects in Figure 5: underrepresented groups shift most, and the Right group shifts more than the Left group. Conservative-biased and default initialization data produce highly similar shifts. This similarity suggests that base-model alignment, rather than initialization-data bias, may primarily drive the default centerward displacement.

**Figure 6: Biased-data positions.** Left displays candidates and agents in J4Q principal-component space; right displays group-mean agent displacement relative to corresponding candidates.

Here, J4Q denotes smartvote. The principal-component space is the candidate-derived political-position space described in Sections 4.1 and 4.7.

### 4.9 Synthetic positions

The counterfactual analysis inverts every smartvote answer, trains Llama 3.1 8B on the resulting data, and computes each politician’s before/after F1 difference. Both training and test answers are flipped. Inversion approximately rotates positions \(180^\circ\) about the origin in Figure 2, including conversion of right-wing to left-wing positions; see Section E.6.

Figure 7 shows that the Right group benefits from inversion, whereas the Left and Center groups worsen on average. However, the inverted Right group still does not reach the original Left group’s F1, while the Left group falls roughly to the original Right group’s levels.

A purely original-model-political-bias account would make inverted right-leaning candidates nearly as predictable as left-leaning candidates. Because a difference remains, the inversion results suggest potentially intrinsically easier left-leaning profiles. This possibility requires investigation.

**Figure 7: Inversion.** Points show original F1, individuals ordered horizontally by their first J4Q principal component; the left-most X8J candidate is correctly placed (see Fig. 6). Arrows encode changes after flipping both training and test answers. P7N worsens individually as well as on average; X8J improvement still leaves its average below P7N. Conversion of P7N into X8J does not remove the disparity, consistent with potentially harder right-wing learning.

In this caption, J4Q denotes smartvote, P7N denotes the Left group, and X8J denotes the Right group.

## 5 Discussion

##### Limitations.

Day-to-day answer variation within the same respondent imposes noise, so consistently near-\(1.0\) F1 is unlikely. In addition, computationally intensive per-profile training restricts analysis to few individuals or parties.

Suitable surveys are scarce and often lack enough items for individual-agent training. Surveys also emphasize generic themes rather than concrete issues and priorities, and question sets rarely align across countries. These survey limitations therefore restrict both individual-agent modeling and cross-country comparison.

##### Future work.

The observed group disparities require bias-mitigation research. Replacing per-agent training with one model conditioned on compact persona representations (Ning et al., 2024) could reduce computation and improve data utilization.

Deployment also requires confidence estimates for answer reliability. Such estimates could be obtained through extended reasoning or a dedicated value head. Finally, purpose-built questionnaires and improved uncertainty handling may mitigate the difficulties associated with Neutral responses.

## 6 Conclusion

Direct training on known opinions treats persona-conditioned agents’ answers as structured reasoning. The initialized combination of supervised fine-tuning and Group-relative policy optimization (ZP4) generally outperforms all baselines across Swiss candidates, German parties and U.S. voters, as reported in Table 3. However, Neutral prediction remains difficult, and accuracy varies significantly across the political spectrum, as shown in Figures 3 and 4. The benchmark therefore supports systematic LLM opinion modeling as an early step toward more representative artificial-intelligence-driven democracy.

## Ethics statement

Individual-opinion modeling amid rising mis- and disinformation is not exclusively beneficial and creates downstream risks. The technology’s inevitable development and these risks motivate responsible alignment research. To protect privacy, only anonymized surveys and public figures’ positions are used.

Systematic ideological disparities, consistent centrist/right-wing underperformance, and neutral-position difficulty are significant system risks. Because these disparities could affect whose opinions are represented, premature deployment could disenfranchise political viewpoints and undermine democratic representation. We commit to transparent bias reporting and strongly caution against real-world applications until fairness issues are resolved. Proper bias mitigation remains a condition of our conviction that opinion modeling can enable societally beneficial digital democracies.

## Reproducibility statement

Code and datasets: https://github.com/ETH-DISCO/reasoning-boosts-llm-alignment

Released documentation includes training details and hyperparameters, principal-component implementation details, per-party training counts, recoding definitions, training-sample-size analysis and inversion details. All principal component analysis (PCA) results appear in Section E.5.

#### Author Contributions

Frédéric Berdoz conceived/designed the project, led experimental planning and defined research direction. Equally contributing Yann Billeter led technical development: dataset collection, training, evaluation and analysis. Yann Vonlanthen advised methodology and experimental design. Roger Wattenhofer supervised as principal investigator.

#### Acknowledgments

We thank Prof. Dr. Hans Gersbach for valuable project insights.
