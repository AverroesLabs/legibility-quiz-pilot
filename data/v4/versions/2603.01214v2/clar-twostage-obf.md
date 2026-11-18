# Reasoning Boosts Opinion Alignment in LLMs

Frédéric Berdoz · Yann Billeter · Yann Vonlanthen · Roger Wattenhofer  
ETH Zurich · Switzerland  
{fberdoz, ybilleter, yvonlanthen, wattenhofer}@ethz.ch

###### Abstract

Large language models (LLMs) generate text from text inputs. Their versatility, ability to generalize to unseen inputs, and success on text-to-text tasks make them candidates for modeling individual and group political preferences. However, their statistical operation and limited understanding of cause and effect produce biased opinions under naive prompting: simply asking for a viewpoint without further training.

We train structured reasoning—an explicit rationale followed by an answer—through reinforcement learning, which updates a model using rewards for its outputs. This produces answers consistent with target political profiles across United States (U.S.), European and Swiss politics. The method remains competitive with strong baselines (Table 3). Bias nevertheless remains (§§4.5–4.9). Faithful political digital twins—models intended to simulate particular people’s political behavior—therefore require additional mechanisms. Released methods and datasets establish a research baseline. Digital democracies could use opinion models to shape fairer, more popular policies.

## 1 Introduction

Reasoning consistently improves opinion alignment: agreement between a model’s answers and a target person’s or party’s opinions. Survey-trained agents generate short rationales followed by choices. They receive rewards for matching respondents’ observed answers, treated as ground truth, and learn survey-consistent positions on unseen questions. Table 3 reports the comparisons; §4.3 states the exception for one model and dataset.

Recent reinforcement-learning gains motivate this treatment of opinion formation. Examples include the o3/o4-mini models (OpenAI, 2025) and DeepSeek-R1 (Guo et al., 2025), which have achieved new benchmark marks.

Individual digital twins that vote on every issue require accurate political-behavior simulation. Such simulation offers opportunities to understand elections and improve policy and democratic processes (Li et al., 2024). Yet sophisticated LLM political discourse does not reliably represent the diversity of human viewpoints (Santurkar et al., 2023; Qu and Wang, 2024; Yu et al., 2025; Qi et al., 2024). Capturing diverse individuals’ political reasoning remains difficult.

Demographic and affiliation prompts ask models to answer as members of demographic or political groups. They exploit learned correlations between group characteristics and opinions. Increasingly sophisticated variants still miss real opinion distributions. Their responses also vary unstably across prompts and demographic groups (Santurkar et al., 2023; Ball et al., 2025).

Preference consistency instead uses an individual’s known opinions, rather than demographic proxies. Existing approaches do not readily simulate individual preferences. Interview transcripts support strong performance for individual personas, or modeled identities, but collecting them is prohibitively costly and impractical at scale (Park et al., 2024).

Large surveys and Voting Advice Applications—online platforms that match voters to parties or candidates—capture political positions. They lack interviews’ narrative richness and textual signal but provide structured records of population opinions. They are established tools for bias evaluation (Santurkar et al., 2023; Argyle et al., 2023; Haller et al., 2025) and group simulation (Argyle et al., 2023; Cao et al., 2025). Their adoption for individual-agent training remains limited.

Our contributions are:
- Explicit reasoning learned from survey responses using QX2, the code for Group-relative policy optimization (GRPO), a reinforcement-learning method described in §2.
- A released benchmark of authentic German-party, U.S.-voter and Swiss-candidate opinions.
- An answer-inversion analysis, which flips known answers to their opposites, revealing systematic ideological variation in alignment quality (§4.9).

**Figure 1: QX2 opinion alignment.** Public surveys supply individual preference profiles. Initial fine-tuning—additional training—on synthetic, model-generated statements and ground-truth answers establishes the reasoning template without full opinion alignment. QX2 then uses formatting and correctness rewards to improve reasoning.

## 2 Related Work

Adopting a political persona supports viewpoint-specific answers. Demographic personas represent shared traits, including occupation or ethnicity. Character personas represent well-known individuals. Individualized personas use personalized digital profiles (Chen et al., 2024; Tseng et al., 2024). The last category is most relevant to individual modeling. However, demographic work predominates, and only a handful of studies use individualized methods.

##### Demographic personas.

Basic affiliation prompting includes “You are a democrat. What is your opinion on…” (Santurkar et al., 2023). Richer prompts add age, education and location (Argyle et al., 2023; Santurkar et al., 2023; Hwang et al., 2023; Sun et al., 2024; Wang et al., 2025). Belief-network demonstrations provide examples of related political positions (Chuang et al., 2024).

Gudiño et al. (2024) combine demographics and demonstrations. Stammbach et al. (2024) combine prompting with N8R, the code for odds ratio preference optimization (ORPO), a preference-training procedure from Hong et al. (2024), to align outputs with target positions.

##### Demographic limitations.

Santurkar et al. (2023) distinguish three problems:
- **Representativeness:** default model-answer distributions do not match population distributions.
- **Steerability:** demographic prompts do not adequately control outputs.
- **Consistency:** bias varies unpredictably across topics and contexts.

Systematic biases include unpredictable accuracy across groups (Argyle et al., 2023; Santurkar et al., 2023). Approximate agreement with overall opinion distributions can coexist with missing natural variation within subgroups and sensitivity to prompt wording (Sun et al., 2024; Ball et al., 2025).

##### Beyond demographics.

Park et al. (2024) use interviews to outperform demographic baselines on public-opinion surveys. Transcript availability constrains this approach. Haller et al. (2025) place respondents’ question–answer pairs in the prompt and predict answers to test questions. Their target is bias identification rather than opinion modeling.

Table 1 gives a qualitative before-and-after example. T9A denotes Llama 3.1 8B, where 8B identifies a model with 8 billion parameters. X8J denotes the Right ideological group. M8C denotes the American National Election Studies (ANES) dataset. ZP4 denotes supervised fine-tuning (SFT) followed by GRPO, or SFT+GRPO. SFT trains a model on example outputs.

In the displayed output, `<reasoning>` and `</reasoning>` mark the rationale’s beginning and end. `<answer>` and `</answer>` mark the final answer. The example retains the answer label `A)`.

**Table 1: T9A trained on an X8J individual in M8C.** Training changes disagreement into agreement while retaining “equal opportunity” arguments under different framing.

| Item | Content |
|---|---|
| Question | Do you agree that it is not really that big a problem if some people have more of a chance in life than others? |
| Before: base-model prompting | No. I don’t think that’s right. Everyone deserves an equal shot at succeeding in life, no matter where they’re from or who their parents are. If some people have more opportunities and resources, it’s not fair to others. We should be working to level the playing field and give everyone a fair chance to succeed. |
| After: ZP4 | `<reasoning>` Some inequality in wealth and opportunity is inevitable and indeed necessary to provide incentives for hard work and innovation – the issue is not eliminating all differences but rather ensuring that everyone has access to basic necessities and opportunities to improve themselves – other countries with significant equality have still struggled with poverty and stagnation. `</reasoning>` |
| Final answer | `<answer>A) Yes</answer>` |

##### Training reasoning.

Chain-of-Thought prompting asks a model to produce intermediate reasoning before answering. It significantly increases accuracy, as does “Let’s think step by step” (Wei et al., 2022; Kojima et al., 2022). Search and reinforcement learning increasingly train high-quality reasoning traces—the generated sequences of reasoning—instead of merely prompting for them (Xu et al., 2025).

QX2 is a variant of Proximal policy optimization (PPO; Schulman et al., 2017), a method for updating the model’s output-generating policy. It uses group-relative reward normalization instead of a learned value function that estimates how advantageous an output is. For each prompt, QX2 samples multiple outputs. It subtracts their mean reward from each reward and divides by their reward standard deviation, a measure of reward spread (Shao et al., 2024). This approach is particularly effective on mathematical-reasoning benchmarks and achieves state-of-the-art results (Guo et al., 2025).

##### Reasoning for alignment.

Yu et al. (2025) combine demographic, ideological and temporal factors in a persona-based, chain-of-thought-inspired prompting framework calibrated with M8C. They sum individual voting simulations into aggregate predictions. They do not use reinforcement learning and target voting rather than general preference profiles. The present method could theoretically address their task, but modeling a representative sample currently requires prohibitive computation.

## 3 Method

Let \(q\) be a policy question and \(p\) a persona, meaning a respondent or party. Let \(Y=\{\texttt{No},\texttt{Neutral},\texttt{Yes}\}\) be the possible stances: disagreement, a neutral response, and agreement. Individual alignment learns \(p\)’s stance on \(q\). An example question is “Should the government increase funding for renewable energy?”

### 3.1 Political Reasoning with QX2

One model is trained per candidate, party or respondent. Only a country label appears in the system prompt, the instruction supplied to the model before the question. No explicit persona representation is used. Correctly answering that individual’s questions alone aligns the model with the individual.

Training observations contain final stances but no reasoning:
\[
\mathcal D=\{(p_j,q_i,y^*_{ij})\},
\]
where \(\mathcal D\) is the training dataset, \(j\) indexes personas, \(i\) indexes questions, and \(y^*_{ij}\) is persona \(p_j\)’s observed stance on question \(q_i\).

Agents must generate both answers matching ground truth and reasoning traces that follow the required format. In Equation (1), `[justification text]` is the rationale and `[stance]` is the answer:
\[
\texttt{<reasoning>[justification text]</reasoning><answer>[stance]</answer>}.
\tag{1}
\]

The composite reward combines:
- E6Z, the **Format Reward**, for correctly placed tags.
- F2B, the **Length Reward**, for deviation from the desired reasoning length.
- G9L, the **Correctness Reward**, for agreement with the observed response.

G9L provides the primary signal. E6Z rewards correctly located output-schema tags.

Let \(x_i\) be a reasoning trace. Let \(L\) be its length in tokens, the pieces of text processed by the model, and \(L^*\) its desired token length. Let \(y_i\) be the predicted answer and \(y_i^*\) the ground-truth response. The absolute difference \(|L-L^*|\) measures length deviation. The indicator \(\mathds{1}[\cdot]\) equals 1 when its condition holds and 0 otherwise. The weights \(\alpha_1,\alpha_2,\alpha_3\) control the three rewards:
\[
\begin{aligned}
\mathrm{F2B}&=-|L-L^*|,\\
\mathrm{G9L}&=\mathds{1}[y_i=y_i^*],\\
R(r_i,y_i^*)&=\alpha_1\mathrm{E6Z}+\alpha_2\mathrm{F2B}
+\alpha_3\mathrm{G9L}.
\end{aligned}
\]
Here \(R\) is the composite reward. Its argument \(r_i\) is not separately defined; the reasoning trace above is denoted \(x_i\).

| Reward detail | Value |
|---|---|
| E6Z per correctly placed tag | 1 |
| Number of tags; E6Z maximum | 4; 4 |
| F2B maximum, attained only at \(L=L^*\) | 0 |
| \(\alpha_1\) | 0.25 |
| \(\alpha_2\) | 0.01 |
| \(\alpha_3\) | 1 |

##### Optional supervised initialization.

V7K denotes supervised fine-tuning (SFT). This optional warm start follows Shao et al. (2024): chain-of-thought demonstrations are constructed in the exact rewarded format. It accelerates convergence toward the trained behavior and improves initial reasoning quality. Pretraining the output structure reduces E6Z’s subsequent burden and supplies a reasonable political-reasoning initialization.

Table 2 summarizes the data. J4Q denotes smartvote, the Swiss questionnaire. R2V denotes Wahl-O-Mat (WoM), the German training source. B5X denotes EU&I, the German test questionnaire. CH, DE and US identify Switzerland, Germany and the United States. “#Units” counts modeled candidates, parties or respondents. “Train Q” and “Test Q” count training and held-out test questions. “Topic-strat.” means that the test split preserves topic coverage by holding out one question per topic.

**Table 2: Data.** Question counts are per candidate/party/respondent.

| Setting | Unit | #Units | Labels | Train Q | Test Q |
|---|---|---:|---|---:|---|
| J4Q (CH) | Candidate | 18 | {Yes, No} | 48 | 12 (topic-strat.) |
| R2V (DE) | Party | 6 | {Yes, Neutral, No} | see Table 5 | 30 (B5X) |
| M8C 2020 (US) | Respondent | 21 | {Yes, Neutral, No} | 67 | 12 (random) |

## 4 Experiments

### 4.1 Datasets

The data cover different countries and political systems. They are released for reproducibility and future research.

For Table 3, K6Y denotes Qwen3 8B and W2E denotes Magistral 24B; 24B identifies 24 billion parameters. C6W is the random baseline, which selects answer labels uniformly. U9F is the majority baseline, which predicts the most frequent label in each unit’s training set. H3D is `icl`, or in-context learning: predicting from question–answer examples included in the prompt. The other method codes are defined above.

D3G denotes macro-F1. F1 combines precision—the fraction of predictions for a class that are correct—and recall—the fraction of that class’s actual instances correctly predicted. Macro-F1 averages F1 equally across classes. Accuracy, used later, is the fraction of all predictions that are correct.

**Table 3: Reasoning requires large datasets for effectiveness.** At sampling temperature \(T=1.0\), which controls output randomness, each of 8 stochastic runs computes a per-unit classification metric (D3G = macro-F1), averages it across units, and contributes to the reported mean and standard deviation, in percent. Stochastic runs vary through random sampling; the mean is their average, and standard deviation measures their spread.

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

The official R2V training data use Wahl-O-Mat, version 26 March 2025. They aggregate party positions from federal, state and European elections over 2021–2025. Agree/neutral/disagree labels are mapped to {Yes, Neutral, No}. Every item includes an explanatory comment.

The six major parties are identified by the names or labels CDU/CSU, SPD, Grüne, FDP, Die Linke and AfD. The out-of-domain test questionnaire—drawn from a different source than training—is B5X, EU&I, from 2024 (Chalkidis, 2024). It is recoded to the same labels. Per-party training counts are in Appendix A.

##### U.S. respondents.

M8C uses the 2020 Time Series source from American National Election Studies (2021). It supplies 79 policy items after excluding demographics, knowledge, candidate evaluations and thermometers, or rating-scale evaluations.

Different response formats are reduced to {Yes, Neutral, No} using two schemes. I7V is the **conservative mapping**: only clearly positive or negative responses count as agreement or disagreement. O2K is the **aggressive mapping**, which collapses most positive responses to Yes. I7V is used unless stated otherwise. Appendix D defines the schemes; Section E.2 tests robustness to recoding.

Items are split randomly. To permit ideological analysis, we select three respondents per answer option on the 7-point liberal–conservative self-placement question, where respondents locate their own ideology.

##### Swiss candidates.

J4Q uses the English 2023 Swiss national-election smartvote questionnaire. It contains 75 items, including 60 policy questions across 12 topics. Its four-point Likert scale—ordered response options—collapses Yes/Rather yes into Yes and Rather no/No into No.

We sample three candidates from each of six parties holding at least two National Council seats. One question per topic is held out for testing. The remaining questions train the agents.

##### Spectrum groups.

For comparison across contexts, parties in J4Q/R2V and self-reported M8C ideologies are assigned to three broad groups: P7N = Left, A4U = Center, and X8J = Right. Appendix A gives detailed party and ideology assignments and additional Swiss-party information.

**Figure 2: Agent positions.** Principal components are axes summarizing variation in the questionnaire answers. The first two components use all standing Swiss candidates as small dots. Included candidates are big dots, agents gold, individual displacements black lines, and group-average distortions big arrows. Displacements connect candidate and agent positions. The axes represent left–right and conservative–liberal positions respectively.

### 4.2 Setup

One model is trained per candidate, party or respondent. The design tests model scale and the advantages of reasoning pretraining, meaning training to reason before the present task. Appendix F gives training details and hyperparameters, the settings controlling training.

##### Methods

Three configurations are reported. QX2 trains on survey questions and answers. The initialized combination, ZP4 = SFT+GRPO, first uses V7K and then QX2.

For V7K, a larger model generates synthetic arguments supporting both positive and negative stances for every J4Q/M8C policy question. That model is Y3H = Llama 3.1 70B, with 70 billion parameters (Grattafiori et al., 2024). Section A.4 gives the synthetic-argument prompts.

##### Backbones

The backbones, or starting models, have openly available weights: T9A = Llama 3.1 8B (Grattafiori et al., 2024), K6Y = Qwen3 8B (Yang et al., 2025), and W2E = Magistral 24B (Rastogi et al., 2025).

They test scale and prior reasoning training. K6Y/W2E are pretrained to reason. T9A is a non-reasoning counterpart for comparison at a similar scale. Every experiment involving training uses 4-bit quantization, which stores model values with reduced numerical precision.

##### Baselines

C6W selects answers uniformly at random. U9F predicts each unit’s most common training answer. H3D follows Haller et al. (2025): it supplies question–answer pairs before an unseen question. It includes all same-topic training questions, or a random subset when the model’s maximum input length prevents this.

N8R uses Stammbach et al. (2024)’s hyperparameters.

### 4.3 Results

ZP4 outperforms the naive and V7K baselines across datasets except T9A on M8C. Table 3’s V7K and ZP4 rows show this exception: \(42.77(1.23)\) for V7K versus \(40.66\pm 0\) .91) for ZP4. Comparing each backbone’s H3D and ZP4 rows, ZP4 is comparable to or better than H3D on M8C/J4Q and better on R2V.

J4Q scores are considerably higher than M8C/R2V scores. For example, Table 3’s W2E/ZP4 row reports \(70.73(2.21)\), \(53.21(3.19)\), and \(45.43(1.11)\), respectively. J4Q’s binary answers, without Neutral, plausibly explain its advantage. R2V exceeds M8C in every ZP4 row, plausibly because it has substantially more training samples. J4Q peaks highest, and M8C is hardest.

Native reasoning support gives no consistent gain. T9A exceeds K6Y on R2V/M8C: their ZP4 rows report \(52.53(4.05)\)/\(40.66\pm 0\) .91) and \(49.38(1.93)\)/\(38.44(0.40)\), respectively. W2E performs best overall; its ZP4 row gives the highest scores across the three datasets.

QX2 alone trails ZP4 in every corresponding Table 3 cell. This suggests that supervised initialization improves training dynamics. Datasets containing Neutral perform worse. That class aggregates uncertainty, social desirability—answers influenced by what seems acceptable—and strategic non-commitment. Combining these phenomena may impede learning. Section E.1 gives additional accuracy, significance and Neutral-excluded results.

**Figure 3: Unequal learnability.** F1 error bars show within-group variance, or variation among members of each group. A more detailed version in Appendix C retains the original parties and ideology groups. ZP4 typically works best, while every training method underperforms on centrist and right-leaning groups.

### 4.4 Agent reasoning

Agents generally produce coherent arguments for or against proposed measures that imply their final answers. Table 1 provides one example. Failures include illogical or hallucinated elements—generated assertions not grounded in the supplied information—and consistency with an individual on one topic but not another. Appendix B provides additional traces and failure examples.

### 4.5 Ideology

For T9A/ZP4, groupwise F1 clearly favors P7N over A4U/X8J in J4Q (Figure 3). R2V/M8C show similar but weaker patterns. R2V P7N/A4U are roughly equal, and M8C performance is generally lower. For X8J in J4Q/M8C, V7K averages exceed all other methods, reinforcing the difficulties of right-wing profiles.

Several possibilities may explain these disparities: known general-purpose LLM left-leaning tendencies (Hartmann et al., 2023; Exler et al., 2025), T9A’s embedded biases, or intrinsic differences in how learnable the survey signals are. Relatively strong R2V X8J performance may benefit from larger training sets; see Section E.4. Under either explanation, ideology affects preference learnability.

### 4.6 Neutral

The Neutral analysis compares recall across answer classes and relates performance to the Neutral response rate, the fraction of answers labeled Neutral. It also recomputes F1 after removing Neutral instances.

Neutral recall is worst for every M8C individual. X8J has the highest Neutral response rate and suffers most from prediction difficulty. A4U also tends to give more Neutral responses. Removing Neutral instances considerably improves F1 for all groups, narrows P7N–A4U differences, and improves ZP4 relative to H3D. It does not eliminate P7N–X8J differences (Figure 4).

O2K reproduces the negative relationship between performance and Neutral rate but gives nearly identical A4U/X8J performance (Section E.2). Recoding may therefore partly produce the disparity. Numerous Neutral examples may depress performance on other classes during training. Model bias and recoding artifacts remain possible influences.

**Figure 4: M8C, T9A/ZP4.** The left panels fit regressions—statistical relationships—between the test-set Neutral base rate and F1 above, and accuracy below. Strong negative correlations are significant at the \(5\%\) level. Upper-right displays class recall averaged across individuals. Lower-right shows F1 before Neutral removal as solid and after removal as shaded. Regression details: Section E.3; alternative-recoding analyses: Section E.2.

### 4.7 Representation

The semantic representativity analysis assesses how agents represent candidates’ political positions. It uses the political interpretation of J4Q principal components documented by Germann et al. (2015). The component space is computed from the full candidate population, and agents are projected into that space. Group-averaged human–agent distances are displayed centrally in Figure 2.

Agents do not perfectly represent their candidates. Despite documented LLM left-libertarian biases—left-leaning and socially liberal tendencies—no overall left shift is observed (Exler et al., 2025; Hartmann et al., 2023; Rozado, 2024). Positions shift center-right on the horizontal spectrum and centerward on the vertical spectrum. Group-average distortions show an overall conservative, negative-\(y\) trend without clear left–right bias; \(y\) denotes the vertical coordinate. X8J individuals become more left-wing. P7N/A4U individuals become more right-wing and conservative.

**Figure 5: Biased initialization data.** T9A/ZP4 F1 under progressive-biased data occupies the left panel, default data the middle, and conservative-biased data the right.

### 4.8 Initialization data

To test initialization-data bias, we generate two additional argument sets with progressive and conservative prompts. These prompts favor the respective political perspectives. They use another model, S5Q = Llama 3.3 70B. Section A.4 gives the synthetic-argument prompting strategy. Figures 5 and 6 compare these sets with the default data.

Ideological bias primarily harms underrepresented perspectives rather than systematically helping overrepresented ones. Progressive arguments strongly impair X8J, with inconsistent P7N effects. Conservative arguments impair P7N without consistent X8J gains. Adverse bias harms X8J more severely than P7N (Figure 5).

Positional effects agree with F1 effects. Underrepresented groups shift most, and X8J shifts more than P7N. Conservative-biased and default data produce highly similar shifts (Figure 6). Base-model alignment, rather than initialization-data bias, may therefore primarily drive the default centerward displacement.

**Figure 6: Biased-data positions.** Left displays candidates and agents in J4Q principal-component space. Right displays group-mean agent displacement relative to corresponding candidates.

### 4.9 Synthetic positions

The counterfactual analysis inverts every J4Q answer: Yes becomes No and No becomes Yes. It flips both training and test answers, trains T9A on the resulting data, and computes each politician’s before/after F1 difference. Inversion approximately rotates positions \(180^\circ\) about the origin in Figure 2, including converting right-wing positions to left-wing positions (Section E.6).

X8J benefits from inversion, while P7N/A4U worsen on average (Figure 7). Inverted X8J still does not reach original P7N F1. P7N falls roughly to original X8J levels.

A purely original-model-political-bias account would make inverted right-leaning candidates nearly as predictable as left-leaning candidates. The remaining difference suggests that left-leaning profiles may be intrinsically easier to learn. This possibility requires investigation.

**Figure 7: Inversion.** Points show original F1, with individuals ordered horizontally by their first J4Q principal component. The left-most X8J candidate is correctly placed (see Figure 6). Arrows encode changes after flipping both training and test answers. P7N worsens individually as well as on average. X8J improvement still leaves its average below P7N. Conversion of P7N into X8J does not remove the disparity, consistent with potentially harder right-wing learning.

## 5 Discussion

##### Limitations.

The same respondent may answer differently from day to day. This variation introduces noise, making consistently near-\(1.0\) F1 unlikely; \(1.0\) denotes perfect F1.

Computationally intensive per-profile training restricts analysis to few individuals or parties. Table 2 includes 18 candidates, 6 parties and 21 respondents. Suitable surveys are scarce and often lack enough items for individual agents. They emphasize generic themes rather than concrete issues and priorities. Question sets also rarely align across countries, limiting comparison.

##### Future work.

Group disparities require bias-mitigation research. Instead of training each agent separately, one model could receive compact persona representations, or short encodings of individual profiles (Ning et al., 2024). This could reduce computation and improve data utilization.

Deployment requires confidence estimates that indicate answer reliability. These could come from extended reasoning or a dedicated value head, an additional model component that estimates answer value or reliability. Purpose-built questionnaires and improved uncertainty handling may mitigate Neutral difficulties.

## 6 Conclusion

Direct training on known opinions treats persona-conditioned agents’ answers as structured reasoning. ZP4 generally outperforms all baselines across Swiss candidates, German parties and U.S. voters (Table 3). Neutral prediction remains difficult, and accuracy varies significantly across the political spectrum (§§4.5–4.6). The benchmark supports systematic LLM opinion modeling as an early step toward more representative AI-driven democracy.

## Ethics statement

Individual-opinion modeling is not exclusively beneficial. Amid rising misinformation and disinformation—false or misleading information, including deliberately misleading information—it creates downstream risks. The technology’s inevitable development and these risks motivate responsible alignment research. For privacy, we use only anonymized surveys and public figures’ positions.

Systematic ideological disparities, consistent centrist/right-wing underperformance and difficulty with neutral positions are significant system risks (§§4.5–4.9). Premature deployment could disenfranchise political viewpoints and undermine democratic representation. We commit to transparent bias reporting and strongly caution against real-world applications until fairness issues are resolved. Proper bias mitigation remains a condition of our conviction that opinion modeling can enable societally beneficial digital democracies.

## Reproducibility statement

Code and datasets: https://github.com/ETH-DISCO/reasoning-boosts-llm-alignment

Released documentation includes training details and hyperparameters, principal-component implementation details, per-party training counts, recoding definitions, training-sample-size analysis and inversion details. All principal component analysis (PCA) results are in Section E.5.

#### Author Contributions

Frédéric Berdoz conceived/designed the project, led experimental planning and defined research direction. Equally contributing Yann Billeter led technical development: dataset collection, training, evaluation and analysis. Yann Vonlanthen advised methodology and experimental design. Roger Wattenhofer supervised as principal investigator.

#### Acknowledgments

We thank Prof. Dr. Hans Gersbach for valuable project insights.
