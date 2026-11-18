# Reasoning Boosts Opinion Alignment in LLMs

Frédéric Berdoz · Yann Billeter · Yann Vonlanthen · Roger Wattenhofer  
ETH Zurich · Switzerland  
{fberdoz, ybilleter, yvonlanthen, wattenhofer}@ethz.ch

###### Abstract

Large language models (LLMs) can handle many tasks, generalize to new settings, and perform well on text-to-text tasks. These strengths make them candidates for modeling the political preferences of individuals and groups. However, their statistical operation and limited understanding of causes produce biased opinions under simple prompting.

Structured reasoning trained through reinforcement learning—training that rewards desired outputs—produces answers consistent with individual profiles across U.S., European, and Swiss politics. The method remains competitive with strong baselines. It uses QX2, or Group-relative policy optimization (GRPO), which compares rewards across several outputs for the same prompt. Bias remains, so faithful political digital twins—computer models of individuals’ political behavior—need additional mechanisms. Our released methods and datasets provide a research baseline. Digital democracies could use opinion models to help shape fairer, more popular policies.

## 1 Introduction

Reasoning consistently improves opinion alignment: the match between a model’s answers and a person’s opinions. Agents trained on surveys generate short explanations followed by choices. They receive rewards for matching respondents’ answers and learn survey-consistent positions on unseen questions. Recent gains from reinforcement learning motivate this approach to opinion formation. These gains include new benchmark records from o3/o4-mini (OpenAI, 2025) and DeepSeek-R1 (Guo et al., 2025).

Individual digital twins that vote on every issue require accurate simulations of political behavior. Such simulations offer opportunities to understand elections and improve policy and democratic processes (Li et al., 2024). LLMs can produce sophisticated political discussion, but they do not reliably represent the diversity of human viewpoints (Santurkar et al., 2023; Qu and Wang, 2024; Yu et al., 2025; Qi et al., 2024). Capturing the political reasoning of diverse individuals remains difficult.

Prompts based on demographics and political affiliation use learned links between demographic traits and opinions. Even increasingly sophisticated versions miss real opinion distributions and respond inconsistently across prompts and demographic groups (Santurkar et al., 2023; Ball et al., 2025). Preference consistency instead uses a person’s known opinions rather than demographic substitutes. Existing approaches do not readily simulate individual preferences. Interview transcripts support strong performance for individual personas, or profiles that a model adopts (Park et al., 2024). However, collecting these transcripts costs too much, and interviewing at scale is impractical.

Large surveys and Voting Advice Applications—online platforms that match voters with parties or candidates—record political positions. They lack the narrative richness and textual information of interviews, but they represent population opinions in a structured form. Researchers have established their use for evaluating bias (Santurkar et al., 2023; Argyle et al., 2023; Haller et al., 2025) and simulating groups (Argyle et al., 2023; Cao et al., 2025). Few studies use them to train individual agents.

We contribute a QX2-based method that learns explicit reasoning from survey responses. We also release a benchmark of authentic opinions from German parties, U.S. voters, and Swiss candidates. Finally, we invert the answers of individuals with known positions to analyze systematic ideological differences in alignment quality.

**Figure 1: QX2 opinion alignment.** Public surveys provide individual preference profiles. Initial fine-tuning—further training on examples—uses synthetic, model-generated statements and ground-truth answers to establish the reasoning template without fully aligning opinions. QX2 then uses rewards for formatting and correctness to improve reasoning.

## 2 Related Work

Adopting a political persona helps a model answer from a specific viewpoint. Demographic personas represent shared traits, such as occupation or ethnicity. Character personas represent well-known individuals. Individualized personas use personalized digital profiles (Chen et al., 2024; Tseng et al., 2024). The last category matters most for individual modeling. However, demographic approaches dominate the literature, and only a handful of studies use individualized methods.

##### Demographic personas.

Basic affiliation prompts include “You are a democrat. What is your opinion on…” (Santurkar et al., 2023). Richer prompts add age, education, and location (Argyle et al., 2023; Santurkar et al., 2023; Hwang et al., 2023; Sun et al., 2024; Wang et al., 2025). Belief-network demonstrations provide examples of related political positions (Chuang et al., 2024). Gudiño et al. (2024) combine demographics with demonstrations. Stammbach et al. (2024) combine prompting with N8R, or odds ratio preference optimization (ORPO), a preference-training procedure from Hong et al. (2024) that aligns outputs with target positions.

##### Demographic limitations.

Santurkar et al. (2023) identify three problems. Representativeness concerns the mismatch between default model outputs and population distributions. Steerability concerns demographic prompts’ limited control over outputs. Consistency concerns bias that varies unpredictably across topics and contexts. Systematic biases include unpredictable accuracy across groups (Argyle et al., 2023; Santurkar et al., 2023). Models may approximate overall distributions while missing natural variation within subgroups and remaining sensitive to prompt wording (Sun et al., 2024; Ball et al., 2025).

##### Beyond demographics.

Park et al. (2024) use interviews to outperform demographic baselines on public-opinion surveys, but transcript availability limits their approach. Haller et al. (2025) place respondents’ question–answer pairs in the prompt and predict answers to test questions. They aim to identify bias rather than model opinions.

**Table 1: T9A trained on an X8J individual in M8C.** T9A denotes Llama 3.1 8B, X8J denotes Right, and M8C denotes the American National Election Studies (ANES). Training changes disagreement into agreement while retaining “equal opportunity” arguments under different framing. ZP4 combines supervised fine-tuning (SFT)—training on example outputs—with GRPO.

| Item | Content |
|---|---|
| Question | Do you agree that it is not really that big a problem if some people have more of a chance in life than others? |
| Before: base-model prompting | No. I don’t think that’s right. Everyone deserves an equal shot at succeeding in life, no matter where they’re from or who their parents are. If some people have more opportunities and resources, it’s not fair to others. We should be working to level the playing field and give everyone a fair chance to succeed. |
| After: ZP4 | `<reasoning>` Some inequality in wealth and opportunity is inevitable and indeed necessary to provide incentives for hard work and innovation – the issue is not eliminating all differences but rather ensuring that everyone has access to basic necessities and opportunities to improve themselves – other countries with significant equality have still struggled with poverty and stagnation. `</reasoning>` |
| Final answer | `<answer>A) Yes</answer>` |

##### Training reasoning.

Chain-of-Thought prompting asks models to produce intermediate reasoning steps (Wei et al., 2022). It and “Let’s think step by step” prompting (Kojima et al., 2022) significantly increase accuracy. Researchers increasingly use search and reinforcement learning to train high-quality reasoning traces, or written reasoning sequences, rather than only prompt for them (Xu et al., 2025).

QX2, or Group-relative policy optimization (GRPO), is a variant of Proximal policy optimization (PPO), a reinforcement-learning method for updating model behavior (Schulman et al., 2017). QX2 compares rewards within a group instead of using a learned value function to estimate how much better an output is than expected. It samples several outputs per prompt, subtracts their mean reward, and divides by their reward standard deviation, a measure of reward spread (Shao et al., 2024). The method works particularly well on mathematical reasoning benchmarks and achieves state-of-the-art results (Guo et al., 2025).

##### Reasoning for alignment.

Yu et al. (2025) combine demographic, ideological, and temporal factors in a persona-based prompting framework inspired by chain-of-thought reasoning. They calibrate the framework with M8C and sum individual voting simulations into aggregate predictions. They do not use reinforcement learning, and they target voting rather than general preference profiles. Our method could theoretically address their task, but modeling a representative sample currently requires prohibitive computation.

## 3 Method

We aim to learn a persona’s stance on a policy question. Let \(q\) denote the question and \(p\) the persona, which can be a respondent or party. The possible stances are \(Y=\{\texttt{No},\texttt{Neutral},\texttt{Yes}\}\). One example question is “Should the government increase funding for renewable energy?”

### 3.1 Political Reasoning with QX2

The system prompt contains only a country label. We use no explicit persona representation. Correct answers to questions alone align the model with individuals.

Training observations contain final stances but no reasoning:
\[
\mathcal D=\{(p_j,q_i,y^*_{ij})\},
\]
where \(\mathcal D\) is the training dataset and \(y^*_{ij}\) is persona \(p_j\)’s observed stance on question \(q_i\). Agents must generate answers that match these observed responses and reasoning traces that follow this format:
\[
\texttt{<reasoning>[justification text]</reasoning><answer>[stance]</answer>}.
\tag{1}
\]

The combined reward evaluates three features: tag placement through E6Z, the Format Reward; deviation from the desired reasoning length through F2B, the Length Reward; and agreement with the observed response through G9L, the Correctness Reward. Correctness provides the main training signal. The Format Reward rewards correctly placed tags.

For reasoning trace \(x_i\), let \(L\) be its length in tokens, the text units a model processes. Let \(L^*\) be the desired length, \(y_i\) the predicted answer, and \(y_i^*\) the ground-truth response:
\[
\begin{aligned}
\mathrm{F2B}&=-|L-L^*|,\\
\mathrm{G9L}&=\mathds{1}[y_i=y_i^*],\\
R(r_i,y_i^*)&=\alpha_1\mathrm{E6Z}+\alpha_2\mathrm{F2B}
+\alpha_3\mathrm{G9L}.
\end{aligned}
\]
The indicator \(\mathds{1}\) equals one when the stated condition holds and zero otherwise. The coefficients \(\alpha_1,\alpha_2,\alpha_3\) set the reward weights.

| Reward detail | Value |
|---|---|
| E6Z per correctly placed tag | 1 |
| Number of tags; E6Z maximum | 4; 4 |
| F2B maximum, attained only at \(L=L^*\) | 0 |
| \(\alpha_1\) | 0.25 |
| \(\alpha_2\) | 0.01 |
| \(\alpha_3\) | 1 |

##### Optional supervised initialization.

An optional warm start uses V7K, or supervised fine-tuning (SFT). Following Shao et al. (2024), we construct chain-of-thought demonstrations in the exact format that the rewards require. This step speeds convergence and improves initial reasoning quality. Training the structure in advance reduces the later burden on E6Z and provides a reasonable starting point for political reasoning.

**Table 2: Data.** Question counts are per candidate, party, or respondent. J4Q denotes smartvote in Switzerland (CH). R2V denotes Wahl-O-Mat (WoM) in Germany (DE). M8C covers the United States (US). B5X denotes EU&I, the test questionnaire from outside the training source. “Topic-strat.” means that the test split samples by topic.

| Setting | Unit | #Units | Labels | Train Q | Test Q |
|---|---|---:|---|---:|---|
| J4Q (CH) | Candidate | 18 | {Yes, No} | 48 | 12 (topic-strat.) |
| R2V (DE) | Party | 6 | {Yes, Neutral, No} | see Table 5 | 30 (B5X) |
| M8C 2020 (US) | Respondent | 21 | {Yes, Neutral, No} | 67 | 12 (random) |

## 4 Experiments

### 4.1 Datasets

The data cover different countries and political systems. We release them to support reproducibility and future research.

**Table 3: Reasoning requires large datasets for effectiveness.** At sampling temperature \(T=1.0\), which controls randomness in generated answers, each of 8 runs with random sampling computes D3G, or macro-F1, for each unit. F1 combines precision—the share of predicted class labels that are correct—and recall—the share of actual class labels recovered. Macro-F1 averages F1 across classes. Each run averages the metric across units and contributes to the reported mean and standard deviation, in percent. C6W denotes random answer selection; U9F predicts the most common training answer. H3D denotes in-context learning (icl), which provides examples in the prompt. K6Y denotes Qwen3 8B, and W2E denotes Magistral 24B.

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

The official R2V training data come from Wahl-O-Mat (WoM), version 26 March 2025. They combine party positions from federal, state, and European elections over 2021–2025. We map agree/neutral/disagree labels to {Yes, Neutral, No}. Every item includes an explanatory comment. The six major parties are CDU/CSU, SPD, Grüne, FDP, Die Linke, and AfD.

The out-of-domain test questionnaire, B5X or EU&I, comes from 2024 (Chalkidis, 2024). We recode it to the same labels. Appendix A gives per-party training counts.

##### U.S. respondents.

The M8C source is the 2020 Time Series from the American National Election Studies (ANES), credited to American National Election Studies (2021). It provides 79 policy items after we exclude demographics, knowledge questions, candidate evaluations, and thermometers.

We reduce varied response formats to {Yes, Neutral, No} using two schemes. I7V, the conservative mapping, maps only clearly positive or negative responses to agreement or disagreement. O2K, the aggressive mapping, collapses most positive responses into Yes. We use I7V unless stated otherwise. Appendix D defines the schemes, and Section E.2 tests robustness to recoding.

We split items randomly. We select three respondents per answer option on the 7-point liberal–conservative self-placement question to support ideological analysis.

##### Swiss candidates.

The English J4Q questionnaire, smartvote for the 2023 Swiss national election, contains 75 items. These include 60 policy questions across 12 topics. We collapse its four-point Likert responses—ordered degrees of agreement—by mapping Yes/Rather yes to Yes and Rather no/No to No.

We sample three candidates from each of six parties that hold at least two National Council seats. We hold out one question per topic and use the rest to train the agents.

##### Spectrum groups.

For comparisons across contexts, we assign J4Q/R2V parties and M8C ideologies to three broad groups: P7N, or Left; A4U, or Center; and X8J, or Right. Appendix A gives detailed party and ideology assignments and further information about Swiss parties.

**Figure 2: Agent positions.** The first two principal components—axes that summarize variation in the response data—use all standing Swiss candidates as small dots. Included candidates appear as big dots, agents in gold, individual displacements as black lines, and group-average distortions as big arrows. The axes represent left–right and conservative–liberal positions, respectively.

### 4.2 Setup

We train one model per candidate, party, or respondent. The design tests the advantages of model scale and prior reasoning training. Appendix F gives training details and hyperparameters.

##### Methods

We report three configurations: QX2 on survey questions and answers, V7K, and the initialized combination ZP4, or SFT+GRPO. For V7K, a larger model generates synthetic arguments that support both positive and negative stances for every J4Q/M8C policy question. This model is Y3H, or Llama 3.1 70B, from Grattafiori et al. (2024). Section A.4 gives the prompts used to generate synthetic arguments.

##### Backbones

We use three open-weight backbones, or base models whose learned numerical parameters are available: T9A, or Llama 3.1 8B, credited to Grattafiori et al. (2024); K6Y, or Qwen3 8B, credited to Yang et al. (2025); and W2E, or Magistral 24B, credited to Rastogi et al. (2025).

These models test scale and prior reasoning training. K6Y and W2E are pretrained to reason. T9A is a non-reasoning counterpart that supports comparison at a similar scale. Every experiment that involves training uses 4-bit quantization, which represents model values with 4 bits.

##### Baselines

C6W, or random, selects answers uniformly. U9F, or majority, predicts the most common answer in each unit’s training set.

The contextual comparator H3D, or icl, follows Haller et al. (2025). It provides question–answer pairs before an unseen question. It supplies all training questions on the same topic, or a random subset when context-length limits prevent this.

N8R uses the hyperparameters—the settings that control training—from Stammbach et al. (2024).

### 4.3 Results

ZP4 outperforms the naive and V7K baselines across datasets, except for T9A on M8C. It matches or outperforms H3D on M8C/J4Q and performs better on R2V.

J4Q scores are considerably higher than M8C/R2V scores. This may reflect its binary answers without Neutral. R2V scores exceed M8C scores, possibly because R2V has substantially more training samples. J4Q reaches the highest peak, and M8C is the hardest dataset.

Native reasoning support gives no consistent gain. T9A outperforms K6Y on R2V/M8C, while W2E performs best overall. QX2 alone trails ZP4, suggesting that supervised initialization improves training dynamics.

Datasets containing Neutral perform worse. That class combines uncertainty, social desirability, and strategic refusal to commit, which may hinder learning. Section E.1 gives additional accuracy results, significance tests, and results that exclude Neutral.

**Figure 3: Unequal learnability.** F1 error bars show variation within groups. A more detailed version in Appendix C retains the original parties and ideology groups. ZP4 typically performs best, while every training method underperforms on centrist and right-leaning groups.

### 4.4 Agent reasoning

Agents generally produce coherent arguments for or against proposed measures that imply their final answers. Failures include illogical elements and hallucinations, or invented details. Agents may also match an individual on one topic but not another. Appendix B gives additional traces and failure examples.

### 4.5 Ideology

Groupwise F1 for T9A/ZP4 shows a clear J4Q advantage for P7N over A4U/X8J. R2V/M8C show similar but weaker patterns. R2V P7N/A4U scores are roughly equal, and M8C performance is generally lower. For X8J in J4Q/M8C, V7K averages exceed those of all other methods. This reinforces the difficulty of modeling right-wing profiles.

Several explanations may account for these disparities. General-purpose LLMs have known left-leaning tendencies (Hartmann et al., 2023; Exler et al., 2025). T9A may contain embedded biases. Alternatively, survey signals may differ in how readily they can be learned. The relatively strong R2V X8J performance may benefit from larger training sets; see Section E.4. Under either explanation, ideology affects how readily preferences can be learned.

### 4.6 Neutral

Neutral has the lowest recall for every M8C individual. X8J has the highest Neutral response rate and suffers most from the difficulty of predicting it. A4U also tends to have more Neutral responses.

Removing Neutral instances and recomputing F1 considerably improves all groups. It narrows P7N–A4U differences and improves ZP4 relative to H3D, but it does not eliminate P7N–X8J differences.

O2K reproduces the negative relationship between performance and Neutral rate, but it gives nearly identical A4U/X8J performance. Recoding may therefore account for part of the disparity. Many Neutral examples may depress performance on the other classes during training. Model bias and recoding artifacts remain possible influences.

**Figure 4: M8C, T9A/ZP4.** The left panels use regression, a statistical fit between variables, to relate the test-set Neutral base rate—the share of Neutral responses—to F1 above and accuracy below. Strong negative correlations are significant at the \(5\%\) level. The upper-right panel shows class recall averaged across individuals. The lower-right panel shows F1 before Neutral removal as solid and after removal as shaded. Section E.3 gives regression details; Section E.2 gives analyses with alternative recoding.

### 4.7 Representation

Agents do not perfectly represent their candidates. We observe no overall left shift despite documented left-libertarian biases in LLMs (Exler et al., 2025; Hartmann et al., 2023; Rozado, 2024). Positions shift center-right on the horizontal spectrum and toward the center on the vertical spectrum. Group-average distortions show an overall conservative trend in the negative \(y\)-direction, without a clear left–right bias. X8J individuals become more left-wing. P7N/A4U individuals become more right-wing and conservative.

To assess how faithfully agents represent political positions, we use the political interpretation of J4Q principal components documented by Germann et al. (2015). We project agents into the space computed from the full candidate population. The central display shows human–agent distances averaged by group.

**Figure 5: Biased initialization data.** The left panel shows T9A/ZP4 F1 under progressive-biased data, the middle panel shows default data, and the right panel shows conservative-biased data.

### 4.8 Initialization data

Ideological bias mainly harms underrepresented perspectives rather than consistently helping overrepresented ones. Progressive arguments strongly impair X8J, with inconsistent effects on P7N. Conservative arguments impair P7N without consistent gains for X8J. Adverse bias harms X8J more severely than P7N.

We create these conditions with two additional argument sets. Another model, S5Q or Llama 3.3 70B, generates them using progressive and conservative prompts. Section A.4 gives the synthetic-argument prompting strategy.

Effects on positions agree with effects on F1. Underrepresented groups shift most, and X8J shifts more than P7N. Conservative-biased and default data produce highly similar shifts. Base-model alignment, rather than bias in initialization data, may therefore be the main cause of the default movement toward the center.

**Figure 6: Biased-data positions.** The left panel shows candidates and agents in J4Q principal-component space. The right panel shows each group’s mean agent displacement relative to its corresponding candidates.

### 4.9 Synthetic positions

X8J benefits from inversion, while P7N/A4U worsen on average. Inverted X8J still does not reach original P7N F1. P7N falls roughly to original X8J levels.

If the original model’s political bias fully explained the results, inverted right-leaning candidates would become nearly as predictable as left-leaning candidates. The remaining difference suggests that left-leaning profiles may be intrinsically easier to learn. This possibility requires investigation.

The counterfactual test inverts every J4Q answer, trains T9A on the resulting data, and computes each politician’s before/after F1 difference. Inversion approximately rotates positions \(180^\circ\) about the origin in Figure 2, including changing right-wing positions into left-wing positions. See Section E.6.

**Figure 7: Inversion.** Points show original F1. Individuals are ordered horizontally by their first J4Q principal component. The left-most X8J candidate is correctly placed; see Figure 6. Arrows show changes after flipping both training and test answers. P7N worsens individually as well as on average. X8J improves, but its average remains below P7N. Converting P7N into X8J does not remove the disparity, consistent with the possibility that right-wing positions are harder to learn.

## 5 Discussion

##### Limitations.

Day-to-day variation in the same respondent’s answers introduces noise. Consistently achieving F1 near \(1.0\) is unlikely. Training each profile requires substantial computation, which limits analysis to a few individuals or parties.

Suitable surveys are scarce and often lack enough items for individual agents. They emphasize broad themes rather than concrete issues and priorities. Their question sets rarely align across countries, which limits comparison.

##### Future work.

Group disparities require research on bias mitigation. One model conditioned on compact persona representations could replace per-agent training (Ning et al., 2024). This could reduce computation and make better use of data.

Deployment requires confidence estimates that indicate answer reliability. Extended reasoning or a dedicated value head—a model component that estimates reliability—could provide these estimates. Purpose-built questionnaires and better handling of uncertainty may reduce difficulties with Neutral.

## 6 Conclusion

Direct training on known opinions teaches persona-conditioned agents to produce answers as structured reasoning. ZP4 generally outperforms all baselines across Swiss candidates, German parties, and U.S. voters. Predicting Neutral remains difficult, and accuracy varies significantly across the political spectrum. The benchmark supports systematic LLM opinion modeling as an early step toward more representative AI-driven democracy.

## Ethics statement

Modeling individual opinions is not exclusively beneficial amid rising misinformation and disinformation. It creates downstream risks. The technology’s inevitable development and these risks motivate responsible alignment research. To protect privacy, we use only anonymized surveys and public figures’ positions.

Systematic ideological disparities, consistent underperformance on centrist and right-wing profiles, and difficulty with neutral positions pose significant system risks. Premature deployment could disenfranchise political viewpoints and undermine democratic representation. We commit to transparent bias reporting and strongly caution against real-world applications until fairness issues are resolved. Our conviction that opinion modeling can enable socially beneficial digital democracies remains conditional on proper bias mitigation.

## Reproducibility statement

Code and datasets: https://github.com/ETH-DISCO/reasoning-boosts-llm-alignment

The released documentation includes training details and hyperparameters, implementation details for principal-component analysis (PCA), per-party training counts, recoding definitions, training-sample-size analysis, and inversion details. Section E.5 contains all PCA results.

#### Author Contributions

Frédéric Berdoz conceived and designed the project, led experimental planning, and defined the research direction. Yann Billeter contributed equally and led technical development, including dataset collection, training, evaluation, and analysis. Yann Vonlanthen advised on methodology and experimental design. Roger Wattenhofer supervised the project as principal investigator.

#### Acknowledgments

We thank Prof. Dr. Hans Gersbach for valuable project insights.
