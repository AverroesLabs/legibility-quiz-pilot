# Reasoning Boosts Opinion Alignment in LLMs

Frédéric Berdoz · Yann Billeter · Yann Vonlanthen · Roger Wattenhofer  
ETH Zurich · Switzerland  
{fberdoz, ybilleter, yvonlanthen, wattenhofer}@ethz.ch

###### Abstract

Large language models (LLMs) generate text from text inputs. Their versatility, ability to generalize to new tasks, and successes on text-to-text tasks make them candidates for modeling individual and group political preferences. However, their statistical operation and limited understanding of causes produce biased opinions under naive prompting: simply asking them to adopt a viewpoint.

Structured reasoning trained through reinforcement learning—training that rewards desired outputs—produces answers consistent with individuals’ preference profiles across United States (U.S.), European and Swiss politics. It remains competitive with strong comparison methods, or baselines (Table 3). Bias remains, so additional mechanisms are needed for faithful political digital twins: models that reproduce individuals’ political preferences. The released methods and datasets establish a research baseline. Digital democracies could use opinion models to shape fairer, more popular policies.

## 1 Introduction

Reasoning consistently improves opinion alignment: agreement between a model’s answers and a person’s observed opinions. Agents—models trained to represent particular respondents—generate short rationales followed by choices. They receive rewards for matching respondents’ answers and learn survey-consistent positions on unseen questions (Table 3). Recent reinforcement-learning gains motivate this approach to opinion formation. These include new benchmark results from o3/o4-mini, OpenAI’s models (OpenAI, 2025), and DeepSeek-R1 (Guo et al., 2025).

Individual digital twins that vote on every issue require accurate simulation of political behavior. Such simulation offers opportunities to understand elections and improve policy and democratic processes (Li et al., 2024). Although LLMs can produce sophisticated political discourse, they do not reliably represent the diversity of human viewpoints (Santurkar et al., 2023; Qu and Wang, 2024; Yu et al., 2025; Qi et al., 2024). Capturing different individuals’ political reasoning remains difficult.

Prompts that specify demographics or political affiliation use correlations between demographic characteristics and opinions learned by the model. More sophisticated variants still miss real opinion distributions. Their responses also vary unstably across prompts and demographic groups (Santurkar et al., 2023; Ball et al., 2025). Preference consistency instead uses an individual’s known opinions, rather than demographic proxies. Existing approaches do not readily simulate individual preferences. Interview transcripts support strong performance for individual personas—representations of particular people—but collecting interviews is prohibitively costly and impractical at scale (Park et al., 2024).

Large surveys and Voting Advice Applications, online platforms that match voters to parties or candidates, record political positions. They lack interviews’ narrative richness and textual detail. However, they represent population opinions in a structured form. They are established tools for evaluating bias (Santurkar et al., 2023; Argyle et al., 2023; Haller et al., 2025) and simulating groups (Argyle et al., 2023; Cao et al., 2025). Their use for training individual agents remains limited.

This paper makes three contributions:

- Explicit reasoning from survey responses using QX2, the paper’s label for group-relative policy optimization (GRPO), a reinforcement-learning method described in Section 2.
- A released benchmark containing authentic German-party, U.S.-voter and Swiss-candidate opinions.
- An analysis that reverses the answers of individuals with known positions. It reveals systematic ideological differences in alignment quality (Section 4.9 and Figure 7).

Fine-tuning means further training an existing model. Figure 1 distinguishes initial fine-tuning, which teaches an output structure, from subsequent reward-based training for opinion alignment.

**Figure 1: QX2 opinion alignment.** Public surveys supply individual preference profiles; initial fine-tuning on synthetic statements and ground-truth answers establishes the reasoning template without full opinion alignment, followed by QX2 with formatting/correctness rewards to improve reasoning.

## 2 Related Work

Adopting a political persona supports viewpoint-specific answers. Demographic personas represent shared traits, such as occupation or ethnicity. Character personas represent well-known individuals. Individualized personas use personalized digital profiles (Chen et al., 2024; Tseng et al., 2024). The last category is most relevant to individual modeling. However, demographic work predominates, and only a handful of studies use individualized methods.

##### Demographic personas.

Basic affiliation prompting includes “You are a democrat. What is your opinion on…” (Santurkar et al., 2023). Richer prompts add age, education and location (Argyle et al., 2023; Santurkar et al., 2023; Hwang et al., 2023; Sun et al., 2024; Wang et al., 2025). Belief-network demonstrations provide related political positions as examples (Chuang et al., 2024).

Gudiño et al. (2024) combine demographics and demonstrations. Stammbach et al. (2024) combine prompting with a preference-training procedure from Hong et al. (2024). This procedure, called N8R here, is odds ratio preference optimization (ORPO). It trains outputs toward target positions.

##### Demographic limitations.

Santurkar et al. (2023) distinguish three problems:

- **Representativeness:** default model-output distributions do not match population distributions.
- **Steerability:** demographic prompts do not adequately control outputs.
- **Consistency:** bias varies unpredictably across topics and contexts.

Systematic biases include unpredictable accuracy across groups (Argyle et al., 2023; Santurkar et al., 2023). A model can approximate an overall distribution while missing natural variation within subgroups. Its results can also remain sensitive to the prompt (Sun et al., 2024; Ball et al., 2025).

##### Beyond demographics.

Park et al. (2024) use interviews to outperform demographic baselines on public-opinion surveys. Their approach is constrained by transcript availability. Haller et al. (2025) provide respondents’ question–answer pairs in the prompt and predict their answers to test questions. Their goal is to identify bias rather than model opinions.

For Table 1, T9A denotes Llama 3.1 8B; B denotes billions of model parameters, the values learned during training. X8J denotes the broad Right ideological group. M8C denotes the American National Election Studies (ANES) dataset. ZP4 combines supervised fine-tuning (SFT), training on example outputs, with GRPO. The paper calls SFT alone V7K.

**Table 1: T9A trained on an X8J individual in M8C.** Training changes disagreement into agreement while retaining “equal opportunity” arguments under different framing.

| Item | Content |
|---|---|
| Question | Do you agree that it is not really that big a problem if some people have more of a chance in life than others? |
| Before: base-model prompting | No. I don’t think that’s right. Everyone deserves an equal shot at succeeding in life, no matter where they’re from or who their parents are. If some people have more opportunities and resources, it’s not fair to others. We should be working to level the playing field and give everyone a fair chance to succeed. |
| After: ZP4 | `<reasoning>` Some inequality in wealth and opportunity is inevitable and indeed necessary to provide incentives for hard work and innovation – the issue is not eliminating all differences but rather ensuring that everyone has access to basic necessities and opportunities to improve themselves – other countries with significant equality have still struggled with poverty and stagnation. `</reasoning>` |
| Final answer | `<answer>A) Yes</answer>` |

##### Training reasoning.

Chain-of-Thought prompting asks a model to produce intermediate reasoning before its answer (Wei et al., 2022). This approach and “Let’s think step by step” significantly increase accuracy (Kojima et al., 2022). Search and reinforcement learning increasingly train high-quality reasoning traces—the generated sequences of reasoning—instead of merely prompting for them (Xu et al., 2025).

QX2 is a variant of proximal policy optimization (PPO), a reinforcement-learning method (Schulman et al., 2017). It uses reward normalization within a group of sampled outputs instead of a learned value function that estimates an output’s advantage. For each prompt, QX2 samples multiple outputs. It subtracts their mean reward from each output’s reward, then divides by the rewards’ standard deviation (Shao et al., 2024). Mathematical-reasoning benchmarks show particular effectiveness and state-of-the-art results (Guo et al., 2025).

##### Reasoning for alignment.

Yu et al. (2025) combine demographic, ideological and temporal factors in a persona-based prompting framework inspired by chain of thought. They calibrate it with M8C and sum individual voting simulations into aggregate predictions. They do not use reinforcement learning. They target voting rather than general preference profiles. The present method could theoretically address their task, but modeling a representative sample currently requires prohibitive computation.

## 3 Method

Let \(q\) be a policy question and \(p\) a persona, either a respondent or a party. Let
\[
Y=\{\texttt{No},\texttt{Neutral},\texttt{Yes}\}
\]
be the set of possible stances. Individual alignment learns persona \(p\)’s stance on question \(q\). One example question is “Should the government increase funding for renewable energy?”

### 3.1 Political Reasoning with QX2

The system prompt—the instruction that sets the model’s behavior—contains only a country label. It does not explicitly represent the persona. Instead, correct answers to that persona’s questions provide the alignment signal. A separate model is trained for each persona, as specified in Section 4.2.

Training observations contain final stances but no reasoning:
\[
\mathcal D=\{(p_j,q_i,y^*_{ij})\}.
\]
Here, \(\mathcal D\) is the training dataset. The indices \(j\) and \(i\) identify personas and questions, respectively. The observed stance \(y^*_{ij}\) is persona \(p_j\)’s answer to question \(q_i\); the star marks the observed, or ground-truth, answer.

Agents must generate a reasoning trace and an answer in the following format. The tags mark the beginning and end of each part:
\[
\texttt{<reasoning>[justification text]</reasoning><answer>[stance]</answer>}.
\tag{1}
\]

The composite reward combines three components:

- E6Z, the **Format Reward**, rewards correctly located opening and closing tags.
- F2B, the **Length Reward**, penalizes deviation from the desired reasoning length.
- G9L, the **Correctness Reward**, rewards agreement with the observed response. This is the primary training signal.

Let \(x_i\) denote a reasoning trace and \(L\) its length in tokens, the units of text processed by the model. Let \(L^*\) be the desired length, \(y_i\) the predicted answer, and \(y_i^*\) the ground-truth response. The generated response is denoted \(r_i\). Its total reward is \(R(r_i,y_i^*)\):
\[
\begin{aligned}
\mathrm{F2B}&=-|L-L^*|,\\
\mathrm{G9L}&=\mathds{1}[y_i=y_i^*],\\
R(r_i,y_i^*)&=\alpha_1\mathrm{E6Z}+\alpha_2\mathrm{F2B}
+\alpha_3\mathrm{G9L}.
\end{aligned}
\]
The absolute-value signs measure the size of the length difference. The indicator \(\mathds{1}[y_i=y_i^*]\) equals 1 when the predicted and observed answers match and 0 otherwise. The coefficients \(\alpha_1,\alpha_2,\alpha_3\) set the weights of the three rewards.

| Reward detail | Value |
|---|---|
| E6Z per correctly placed tag | 1 |
| Number of tags; E6Z maximum | 4; 4 |
| F2B maximum, attained only at \(L=L^*\) | 0 |
| \(\alpha_1\) | 0.25 |
| \(\alpha_2\) | 0.01 |
| \(\alpha_3\) | 1 |

##### Optional supervised initialization.

An optional warm start uses V7K, supervised fine-tuning. Following Shao et al. (2024), it constructs chain-of-thought demonstrations in the exact format rewarded above. This accelerates convergence and improves initial reasoning quality. Teaching the structure beforehand also reduces the subsequent burden on E6Z and provides a reasonable starting point for political reasoning.

Table 2 summarizes the datasets. J4Q denotes smartvote, the Swiss dataset; R2V denotes Wahl-O-Mat (WoM), the German dataset; and B5X denotes EU&I, the German-party test questionnaire. CH, DE and US identify Switzerland, Germany and the United States. “#Units” is the number of candidates, parties or respondents. “Train Q” and “Test Q” count training and test questions. “Topic-strat.” means that the test questions are selected by topic.

**Table 2: Data.** Question counts are per candidate/party/respondent.

| Setting | Unit | #Units | Labels | Train Q | Test Q |
|---|---|---:|---|---:|---|
| J4Q (CH) | Candidate | 18 | {Yes, No} | 48 | 12 (topic-strat.) |
| R2V (DE) | Party | 6 | {Yes, Neutral, No} | see Table 5 | 30 (B5X) |
| M8C 2020 (US) | Respondent | 21 | {Yes, Neutral, No} | 67 | 12 (random) |

## 4 Experiments

### 4.1 Datasets

The datasets cover different countries and political systems. They are released for reproducibility and future research.

Table 3 uses three model backbones, the existing models on which training starts: T9A is Llama 3.1 8B, K6Y is Qwen3 8B, and W2E is Magistral 24B. It compares the training methods with C6W, a uniform random-answer selector; U9F, which predicts each unit’s most frequent training answer; and H3D, in-context learning (icl), which supplies question–answer examples in the prompt.

D3G denotes macro-F1. For each answer class, precision measures how often predictions of that class are correct, and recall measures how often the model identifies examples belonging to that class. F1 combines precision and recall through their harmonic mean. Macro-F1 averages F1 across classes. Here, classification means selecting an answer class.

The sampling temperature \(T\) controls randomness in generated outputs. Each stochastic run uses random sampling. The reported mean is the average across runs; the standard deviation measures variation across those runs.

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

Official R2V training data, version 26 March 2025, aggregate party positions from federal, state and European elections over 2021–2025. The labels agree, neutral and disagree map to {Yes, Neutral, No}. Every item also has an explanatory comment.

The six major parties included are CDU/CSU, SPD, Grüne, FDP, Die Linke and AfD; these names identify the party units. The B5X test questionnaire is from 2024 (Chalkidis, 2024). It is out of domain: it comes from a different questionnaire than the training data. Its answers are recoded to the same labels. Appendix A gives per-party training counts.

##### U.S. respondents.

The M8C source is the 2020 Time Series, credited to American National Election Studies (2021). It supplies 79 policy items after excluding demographics, knowledge questions, candidate evaluations and thermometers, which are rating items.

The original response formats differ across questions. Two schemes reduce them to {Yes, Neutral, No}:

- I7V, the **conservative mapping**, maps only clearly positive or negative responses to agreement or disagreement.
- O2K, the **aggressive mapping**, collapses most positive responses to Yes.

I7V is used unless stated otherwise. Appendix D defines both schemes. Section E.2 tests robustness to recoding. Items are split randomly into training and test questions. To permit ideological analysis, the sample takes three respondents from each answer option on the 7-point liberal–conservative self-placement question.

##### Swiss candidates.

The English 2023 J4Q questionnaire for the Swiss national election contains 75 items. These include 60 policy questions across 12 topics. Its four-point Likert scale—an ordered response scale—collapses Yes/Rather yes into Yes and Rather no/No into No.

The sample takes three candidates from each of six parties holding at least two National Council seats. One question per topic is held out for testing. The remaining questions train the agents.

##### Spectrum groups.

For comparison across contexts, parties in J4Q/R2V and ideologies in M8C are assigned to three broad groups: P7N, Left; A4U, Center; and X8J, Right. Appendix A gives the party and ideology assignments and additional Swiss-party information.

Figure 2 uses principal component analysis (PCA), which summarizes variation in answer profiles along derived axes called principal components. The first two axes are interpreted politically as described in Section 4.7.

**Figure 2: Agent positions.** The first two principal components use all standing Swiss candidates as small dots; included candidates are big dots, agents gold, individual displacements black lines, and group-average distortions big arrows. Axes represent left–right and conservative–liberal positions respectively.

### 4.2 Setup

One model is trained per candidate, party or respondent. The design tests the effects of model scale and prior reasoning training. Appendix F gives training details and hyperparameters.

##### Methods

Three configurations are reported: V7K, QX2 on survey questions and answers, and ZP4, the initialized combination SFT+GRPO.

For V7K, a larger model generates synthetic arguments supporting both positive and negative stances for every J4Q/M8C policy question. Synthetic means model-generated rather than supplied by respondents. The generating model is Y3H, Llama 3.1 70B, from Grattafiori et al. (2024). Section A.4 gives the synthetic-argument prompts.

##### Backbones

The open-weight backbones—models whose learned weights are available—are T9A, Llama 3.1 8B (Grattafiori et al., 2024); K6Y, Qwen3 8B (Yang et al., 2025); and W2E, Magistral 24B (Rastogi et al., 2025).

These models test scale and prior reasoning training. K6Y/W2E are pretrained to reason. T9A is a non-reasoning counterpart that permits a comparison at similar scale. Every experiment involving training uses 4-bit quantization, a reduced-precision representation of model values.

##### Baselines

C6W selects answers uniformly at random. U9F predicts the modal answer—the most frequent answer—in each unit’s training set.

H3D follows Haller et al. (2025). It provides question–answer pairs before asking an unseen question. It supplies all training questions from the same topic, or a random subset when the model’s context-length limit—the amount of text it can receive at once—prevents this. N8R uses the hyperparameters, or training settings, of Stammbach et al. (2024).

### 4.3 Results

ZP4 outperforms the naive and V7K baselines across datasets, except for T9A on M8C. It is comparable to or better than H3D on M8C/J4Q and better on R2V (Table 3, ZP4 rows compared with C6W, U9F, V7K and H3D rows).

J4Q scores are considerably higher than M8C/R2V scores (Table 3). This plausibly reflects J4Q’s binary answers, which exclude Neutral. R2V scores exceed M8C scores, plausibly because R2V has substantially more training samples (Table 3; per-party counts in Appendix A). J4Q reaches the highest peak score, and M8C is the hardest dataset (Table 3, W2E/ZP4 row and dataset columns).

Native reasoning support gives no consistent gain. T9A exceeds K6Y on R2V/M8C, while W2E performs best overall (Table 3, corresponding model rows). QX2 alone trails ZP4 for every backbone and dataset in Table 3. This suggests that supervised initialization improves training dynamics.

Datasets containing Neutral perform worse (Table 3, R2V and M8C columns compared with J4Q). Neutral aggregates uncertainty, social desirability and strategic non-commitment. Combining these different reasons for answering Neutral may impede learning. Section E.1 reports additional accuracy results, statistical-significance tests and results excluding Neutral. Accuracy is the fraction of answers predicted correctly.

**Figure 3: Unequal learnability.** F1 error bars show within-group variance; a more detailed version in Appendix C retains original parties and ideology groups. ZP4 typically works best, while every training method underperforms on centrist and right-leaning groups.

### 4.4 Agent reasoning

Agents generally produce coherent arguments for or against proposed measures that imply their final answers. Failures include illogical elements and hallucinated elements—generated content that is not grounded. An agent can also match an individual on one topic but not another. Appendix B provides additional traces and failure examples.

### 4.5 Ideology

Groupwise F1 for T9A/ZP4 shows a clear J4Q advantage for P7N over A4U/X8J (Figure 3). R2V/M8C show similar but weaker patterns. R2V P7N/A4U performance is roughly equal, and M8C performance is generally lower (Figure 3).

For X8J in J4Q/M8C, V7K’s average scores exceed those of all other methods (Figure 3). This reinforces the difficulty of modeling right-wing profiles.

Several explanations may account for these disparities: known left-leaning tendencies in general-purpose LLMs (Hartmann et al., 2023; Exler et al., 2025), biases embedded in T9A, or intrinsic differences in how readily survey signals can be learned. Relatively strong performance for R2V X8J may benefit from larger training sets (see Section E.4). Under either explanation, ideology affects how readily preferences can be learned.

### 4.6 Neutral

Neutral has the worst recall for every M8C individual (Figure 4). X8J has the highest Neutral response rate and suffers most from the difficulty of predicting it. A4U also tends to have more Neutral responses.

Removing Neutral instances and recomputing F1 considerably improves all groups. It narrows P7N–A4U differences and improves ZP4 relative to H3D, but does not eliminate P7N–X8J differences (Figure 4; Neutral-excluded results in Section E.1).

O2K reproduces the negative relationship between performance and the Neutral response rate. However, it yields nearly identical A4U/X8J performance (Section E.2). Recoding may therefore partly produce the disparity. Numerous Neutral examples may depress performance on the other classes during training. Model bias and recoding artifacts remain possible influences.

In Figure 4, the Neutral base rate is the share of test answers labeled Neutral. The regressions assess its relationship with F1 and accuracy. Statistical significance is reported at the \(5\%\) level.

**Figure 4: M8C, T9A/ZP4.** Left panels regress test-set Neutral base rate against F1 above and accuracy below; strong negative correlations are significant at the \(5\%\) level. Upper-right displays class recall averaged across individuals, lower-right F1 before Neutral removal as solid and after as shaded. Regression details: Section E.3; alternative-recoding analyses: Section E.2.

### 4.7 Representation

Agents do not perfectly represent their candidates. Despite documented LLM left-libertarian biases (Exler et al., 2025; Hartmann et al., 2023; Rozado, 2024), no overall left shift is observed. Positions shift center-right on the horizontal spectrum and toward the center on the vertical spectrum (Figure 2).

Group-average distortions show an overall conservative trend in the negative-\(y\) direction, where \(y\) is the vertical coordinate. They show no clear left–right bias. X8J individuals become more left-wing. P7N/A4U individuals become more right-wing and conservative (Figure 2).

This analysis of semantic representativity—how closely agents’ political positions represent candidates’ positions—uses the political interpretation of J4Q principal components documented by Germann et al. (2015). The component space is computed from the full candidate population. Agents are then projected into that same space. Group-averaged human–agent distances are displayed centrally in Figure 2.

**Figure 5: Biased initialization data.** T9A/ZP4 F1 under progressive-biased data occupies the left panel, default data the middle, conservative-biased data the right.

### 4.8 Initialization data

Ideological bias in initialization data primarily harms underrepresented perspectives rather than systematically helping overrepresented ones. Progressive arguments strongly impair X8J, with inconsistent effects on P7N. Conservative arguments impair P7N without consistently improving X8J. Adverse bias harms X8J more severely than P7N (Figure 5).

These conditions use two additional argument sets. Another model, S5Q, Llama 3.3 70B, generates them using progressive and conservative prompts. Section A.4 gives the synthetic-argument prompting strategy.

Changes in positions agree with the F1 results. Underrepresented groups shift most, and X8J shifts more than P7N (Figure 6). Conservative-biased and default data produce highly similar shifts. Thus, the base model’s alignment, rather than bias in the initialization data, may primarily drive the default displacement toward the center.

**Figure 6: Biased-data positions.** Left displays candidates and agents in J4Q principal-component space; right displays group-mean agent displacement relative to corresponding candidates.

### 4.9 Synthetic positions

Answer inversion benefits X8J, while P7N/A4U worsen on average (Figure 7). Inverted X8J still does not reach original P7N F1. P7N falls roughly to original X8J levels.

An explanation based purely on the original model’s political bias would make inverted right-leaning candidates nearly as predictable as left-leaning candidates. The remaining difference suggests that left-leaning profiles may be intrinsically easier to learn. This possibility requires investigation.

The counterfactual experiment reverses every J4Q answer, exchanging Yes and No. It trains T9A on the resulting data and computes each politician’s F1 difference before and after inversion. Both training and test answers are flipped. Inversion approximately rotates positions \(180^\circ\) about the origin in Figure 2, including converting right-wing positions to left-wing positions (Section E.6).

**Figure 7: Inversion.** Points show original F1, individuals ordered horizontally by their first J4Q principal component; the left-most X8J candidate is correctly placed (see Fig. 6). Arrows encode changes after flipping both training and test answers. P7N worsens individually as well as on average; X8J improvement still leaves its average below P7N. Conversion of P7N into X8J does not remove the disparity, consistent with potentially harder right-wing learning.

## 5 Discussion

##### Limitations.

The same respondent can answer differently from day to day. This introduces noise, making consistently near-\(1.0\) F1 unlikely.

Training a separate model for each profile is computationally intensive. It restricts the analysis to a small number of individuals or parties.

Suitable surveys are scarce. They often lack enough items to train individual agents. They emphasize generic themes rather than concrete issues and priorities. Question sets also rarely align across countries, limiting comparisons.

##### Future work.

Group disparities require research on bias mitigation. Replacing separate agent training with one model conditioned on compact persona representations could reduce computation and improve data use (Ning et al., 2024). In that approach, the model would receive a compact representation of the persona whose answer it should predict.

Deployment requires confidence estimates that indicate answer reliability. These could be obtained through extended reasoning or a dedicated value head, a model component that estimates answer value or reliability. Purpose-built questionnaires and better uncertainty handling may mitigate the difficulties with Neutral.

## 6 Conclusion

Direct training on known opinions produces persona-conditioned agents that answer through structured reasoning. ZP4 generally outperforms all baselines across Swiss candidates, German parties and U.S. voters (Table 3). Predicting Neutral remains difficult, and accuracy varies significantly across the political spectrum (Figures 3 and 4).

The benchmark supports systematic LLM opinion modeling. It is an early step toward more representative democracy driven by artificial intelligence (AI).

## Ethics statement

Individual-opinion modeling is not exclusively beneficial, especially amid rising misinformation and disinformation. It creates downstream risks. The technology’s inevitable development and these risks motivate responsible alignment research. To protect privacy, this work uses only anonymized surveys and public figures’ positions.

Systematic ideological disparities, consistent centrist/right-wing underperformance and difficulty with neutral positions are significant system risks (Figures 3 and 4). Premature deployment could disenfranchise political viewpoints and undermine democratic representation. We commit to transparent bias reporting. We strongly caution against real-world applications until fairness issues are resolved. Our conviction that opinion modeling can enable societally beneficial digital democracies remains conditional on proper bias mitigation.

## Reproducibility statement

Code and datasets: https://github.com/ETH-DISCO/reasoning-boosts-llm-alignment

Released documentation includes training details and hyperparameters, principal-component implementation details (all PCA results: Section E.5), per-party training counts, recoding definitions, training-sample-size analysis and inversion details.

#### Author Contributions

Frédéric Berdoz conceived/designed the project, led experimental planning and defined research direction. Equally contributing Yann Billeter led technical development: dataset collection, training, evaluation and analysis. Yann Vonlanthen advised methodology and experimental design. Roger Wattenhofer supervised as principal investigator.

#### Acknowledgments

We thank Prof. Dr. Hans Gersbach for valuable project insights.
