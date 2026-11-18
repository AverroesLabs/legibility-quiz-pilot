# Reasoning Boosts Opinion Alignment in LLMs

Frédéric Berdoz · Yann Billeter · Yann Vonlanthen · Roger Wattenhofer  
ETH Zurich · Switzerland  
{fberdoz, ybilleter, yvonlanthen, wattenhofer}@ethz.ch

## Summary
*The paper tests whether reasoning training improves models’ agreement with individuals’ political opinions.*

The paper trains models on survey question–answer pairs to generate short reasons followed by answers that match respondents’ positions. It evaluates Swiss candidates, German parties and U.S. voters. Supervised initialization followed by reinforcement learning generally performs best, although not in every setting. Performance varies across political groups. Centrist and right-leaning profiles are harder to model, and Neutral answers are difficult to predict. Biased initialization data mainly harm underrepresented perspectives. Answer inversion improves right-leaning profiles but does not eliminate the original performance gap. The paper releases methods and datasets and cautions against deployment before fairness problems are resolved.

## Glossary
*These definitions describe the paper’s terminology, notation, model names and codes.*

- **LLMs** — Models considered for individual/group political-preference modeling because of their versatility, generalization and text-to-text successes.
- **Opinion alignment / preference consistency** — Matching a person’s known political opinions rather than relying on demographic proxies.
- **Political digital twin** — An individual political-behavior model intended to vote on every issue.
- **Persona** — A respondent or party whose positions the model should reproduce.
- **Demographic persona** — A persona defined by shared traits, such as occupation or ethnicity.
- **Character persona** — A persona representing a well-known individual.
- **Individualized persona** — A persona based on a personalized digital profile.
- **Voting Advice Application** — An online platform matching voters with parties or candidates.
- **Representativeness** — Whether default model-output distributions match populations.
- **Steerability** — Whether demographic prompts adequately control outputs.
- **Consistency** — Whether bias remains stable across topics and contexts.
- **Chain-of-Thought / structured reasoning** — A rationale or justification followed by a final choice.
- **Reinforcement learning** — Training that rewards generated outputs; here, rewards evaluate formatting, length and answer correctness.
- **PPO** — Proximal policy optimization, the policy-optimization method from which the described variant derives.
- **QX2 / GRPO** — Group-relative policy optimization, using group-relative reward normalization rather than a learned advantage-estimating value function.
- **V7K / SFT** — Supervised fine-tuning, used as an optional initialization with reasoning demonstrations.
- **ZP4 / SFT+GRPO** — Supervised fine-tuning followed by group-relative policy optimization.
- **N8R / ORPO** — Odds ratio preference optimization, the preference-training baseline.
- **E6Z / Format Reward** — Reward for correctly placed schema tags.
- **F2B / Length Reward** — Negative absolute deviation from the desired reasoning-token length.
- **G9L / Correctness Reward** — Reward indicating agreement with the observed response.
- **Schema / tags** — The required `<reasoning>...</reasoning><answer>...</answer>` output structure.
- **Ground truth** — The persona’s observed answer.
- **Token length** — The length used to score a reasoning trace against its desired length.
- **C6W / random** — A naive uniform-answer selector.
- **U9F / majority** — A predictor of the modal answer in a unit’s training set.
- **H3D / icl** — The contextual baseline supplied with question–answer pairs before an unseen question.
- **T9A** — Llama 3.1 8B, the non-reasoning backbone.
- **K6Y** — Qwen3 8B, a backbone pretrained to reason.
- **W2E** — Magistral 24B, a backbone pretrained to reason.
- **Y3H** — Llama 3.1 70B, used to generate initialization arguments.
- **S5Q** — Llama 3.3 70B, used to generate progressive- and conservative-prompted arguments.
- **o3/o4-mini; DeepSeek-R1** — Examples of recent reinforcement-learning gains motivating the study.
- **J4Q / smartvote** — The English 2023 Swiss national-election questionnaire.
- **R2V / Wahl-O-Mat / WoM** — The German party-position training source, version 26 March 2025.
- **B5X / EU&I** — The 2024 out-of-domain German-party test questionnaire.
- **M8C / American National Election Studies / ANES** — The 2020 Time Series source of U.S. respondent answers.
- **CH; DE; US / U.S.** — The Swiss, German and U.S. dataset settings, respectively.
- **CDU/CSU; SPD; Grüne; FDP; Die Linke; AfD** — The six major German parties included in R2V.
- **I7V / conservative mapping** — Recoding that uses only clearly positive/negative responses for agreement/disagreement.
- **O2K / aggressive mapping** — Recoding that collapses most positive responses to Yes.
- **P7N / Left** — The broad left political-spectrum group.
- **A4U / Center** — The broad center political-spectrum group.
- **X8J / Right** — The broad right political-spectrum group.
- **Yes; Neutral; No** — The three possible stances; J4Q uses only Yes and No.
- **Rather yes; Rather no** — J4Q responses collapsed into Yes and No, respectively.
- **Likert responses** — J4Q’s four-point response format.
- **D3G / macro-F1** — The per-unit classification metric averaged across units in Table 3.
- **F1** — The classification-performance score used in the reported comparisons.
- **Accuracy** — The additional performance measure plotted in Figure 4 and reported in Section E.1.
- **Recall** — The classwise performance measure averaged across individuals in Figure 4.
- **Mean; standard deviation; \(\pm\)** — The summary and variation reported across stochastic runs in Table 3.
- **\(T\)** — The evaluation setting fixed at \(1.0\) in Table 3.
- **PCA / principal components** — The candidate-position representation computed from the full Swiss candidate population.
- **Left–right; conservative–liberal** — The interpretations of the first and second Swiss principal-component axes.
- **\(y\), in positional analysis** — The vertical coordinate; negative-\(y\) displacement is described as conservative.
- **Progressive-biased / conservative-biased data** — Initialization arguments generated with progressive or conservative prompts.
- **Inversion** — Flipping every J4Q answer in both training and testing.
- **4-bit quantization** — The quantization used in every experiment involving training.
- **Confidence estimates / value head** — Proposed ways to assess answer reliability.
- **AI-driven democracy / digital democracy** — The proposed use of opinion models in democratic policy processes.
- **\(q,q_i\)** — A policy question and an indexed policy question.
- **\(p,p_j\)** — A persona and an indexed persona.
- **\(i,j\)** — Indices used for questions, outputs and personas in the displayed notation.
- **\(Y\)** — The stance set \(\{\texttt{No},\texttt{Neutral},\texttt{Yes}\}\).
- **\(\mathcal D\)** — The training observations \(\{(p_j,q_i,y^*_{ij})\}\).
- **\(y^*_{ij}\)** — Persona \(p_j\)’s observed stance on question \(q_i\).
- **\(x_i\)** — A reasoning trace.
- **\(L,L^*\)** — Reasoning-token length and desired reasoning-token length.
- **\(y_i,y_i^*\)** — Predicted answer and ground-truth response.
- **\(R(r_i,y_i^*)\)** — The composite reward expression.
- **\(r_i\)** — An argument of \(R\) in the composite-reward expression.
- **\(\alpha_1,\alpha_2,\alpha_3\)** — Weights on E6Z, F2B and G9L.
- **\(\mathds{1}[y_i=y_i^*]\)** — The correctness indicator used for G9L.
- **\(|L-L^*|\)** — Absolute deviation from the desired reasoning length.
- **#Units; Train Q; Test Q** — Number of units, training questions and test questions in Table 2.
- **Topic-strat.** — The J4Q test split holding out one question per topic.
- **`A) Yes`** — The affirmative final answer shown in Table 1.

###### Abstract
*Reasoning training improves profile-consistent political answers, but bias remains.*

LLMs, the models considered here for political-preference modeling, are candidates for modeling individual and group preferences because of their versatility, generalization and text-to-text successes. Their statistical operation and limited causal understanding produce biased opinions under naive prompting. Structured reasoning—a short rationale followed by a choice—trained through reinforcement learning produces profile-consistent answers across U.S., European and Swiss politics. Reinforcement learning rewards generated outputs; here, those rewards assess formatting, length and correctness. The method remains competitive with strong baselines. Bias remains and requires additional mechanisms for faithful political digital twins: individual models intended to vote on every issue. Released methods and datasets establish a research baseline. Digital democracies could use opinion models to shape fairer, more popular policies.

**Claims and evidence**
- Reasoning training produces competitive, profile-consistent answers — Table 3, QX2 and ZP4 rows; the W2E/ZP4 row reports \(70.73(2.21)\), \(53.21(3.19)\), and \(45.43(1.11)\) for J4Q, R2V and M8C.
- Bias remains — Figures 3–7 report ideological differences, Neutral difficulty and positional distortions.

## 1 Introduction
*The paper proposes learning individual political preferences from survey answers through explicit reasoning.*

Reasoning consistently improves opinion alignment: matching individuals’ known political opinions. Survey-trained agents generate short rationales followed by choices. They receive rewards for matching respondents’ answers and learn survey-consistent positions on unseen questions. Recent reinforcement-learning gains motivate this approach to opinion formation. Examples include o3/o4-mini (OpenAI, 2025) and DeepSeek-R1 (Guo et al., 2025), which set new benchmark marks.

Individual digital twins voting on every issue require accurate political-behavior simulation. Such simulation offers opportunities to understand elections and improve policy and democratic processes (Li et al., 2024). Sophisticated LLM political discourse does not reliably represent human viewpoint diversity (Santurkar et al., 2023; Qu and Wang, 2024; Yu et al., 2025; Qi et al., 2024). Capturing diverse individuals’ political reasoning remains difficult.

Demographic and affiliation prompts exploit learned demographic–opinion correlations. Increasingly sophisticated variants still miss real distributions and respond unstably across prompts and demographic groups (Santurkar et al., 2023; Ball et al., 2025). Preference consistency instead uses known individual opinions rather than demographic proxies. Existing approaches do not readily simulate individual preferences. Interview transcripts yield strong individual-persona performance (Park et al., 2024), but collection costs are prohibitive and interviewing at scale is impractical.

Large surveys and Voting Advice Applications—online voter–party/candidate matching platforms—capture political positions. They lack interviews’ narrative richness and textual signal but structurally represent population opinions. They are established tools for bias evaluation (Santurkar et al., 2023; Argyle et al., 2023; Haller et al., 2025) and group simulation (Argyle et al., 2023; Cao et al., 2025). Their adoption for individual-agent training remains limited.

The contributions are explicit reasoning from survey responses using QX2, or Group-relative policy optimization (GRPO); a released benchmark of authentic German-party, U.S.-voter and Swiss-candidate opinions; and answer-inversion analysis. The last analysis uses individuals with known positions and reveals systematic ideological variation in alignment quality.

**Figure 1: QX2 opinion alignment.** Public surveys supply individual preference profiles; initial fine-tuning on synthetic statements and ground-truth answers establishes the reasoning template without full opinion alignment, followed by QX2 with formatting/correctness rewards to improve reasoning.

**Claims and evidence**
- Reasoning improves alignment — Table 3, QX2 and ZP4 rows; for T9A on J4Q, these report \(55.14\pm 1\) .25) and \(66.88(2.18)\).
- Alignment quality varies systematically with ideology — Figures 3 and 7; Sections 4.5 and 4.9.

## 2 Related Work
*Prior work uses personas, demographic prompting, interviews and reasoning methods, but individualized survey-based training remains limited.*

Political-persona adoption supports viewpoint-specific answers. A persona is the respondent or party whose positions the model should reproduce. Demographic personas represent shared traits, including occupation or ethnicity. Character personas represent well-known individuals. Individualized personas use personalized digital profiles (Chen et al., 2024; Tseng et al., 2024). The last category is most relevant to individual modeling. Demographic work nevertheless predominates, and only a handful of studies use individualized methods.

##### Demographic personas.
*Demographic approaches combine affiliation, personal attributes, demonstrations or preference training.*

Basic affiliation prompting includes “You are a democrat. What is your opinion on…” (Santurkar et al., 2023). Richer prompts add age, education and location (Argyle et al., 2023; Santurkar et al., 2023; Hwang et al., 2023; Sun et al., 2024; Wang et al., 2025). Belief-network demonstrations supply related political positions (Chuang et al., 2024). Gudiño et al. (2024) combine demographics and demonstrations. Stammbach et al. (2024) combine prompting with N8R, or odds ratio preference optimization (ORPO), a preference-training procedure from Hong et al. (2024), to align outputs with target positions.

##### Demographic limitations.
*Demographic prompting has problems with population representation, output control and stability.*

Santurkar et al. (2023) distinguish three limitations: representativeness, where default distributions mismatch populations; steerability, where demographic prompts inadequately control outputs; and consistency, where bias varies unpredictably across topics and contexts. Systematic biases include unpredictable groupwise accuracy (Argyle et al., 2023; Santurkar et al., 2023). Approximate overall distributions can coexist with missing natural variance within subgroups and prompt sensitivity (Sun et al., 2024; Ball et al., 2025).

##### Beyond demographics.
*Interviews and individual question–answer histories provide alternatives to demographic profiles.*

Park et al. (2024) outperform demographic baselines on public-opinion surveys using interviews, but transcript availability limits this approach. Haller et al. (2025) provide respondents’ question–answer pairs in context and predict answers to test questions. Their goal is bias identification rather than opinion modeling.

In Table 1, T9A denotes Llama 3.1 8B; X8J denotes the Right group; M8C denotes American National Election Studies (ANES); and ZP4 denotes supervised fine-tuning followed by GRPO (SFT+GRPO). Ground-truth answers are the observed answers to be matched. The tags delimit the justification and final answer.

**Table 1: T9A trained on an X8J individual in M8C.** Training changes disagreement into agreement while retaining “equal opportunity” arguments under different framing.

| Item | Content |
|---|---|
| Question | Do you agree that it is not really that big a problem if some people have more of a chance in life than others? |
| Before: base-model prompting | No. I don’t think that’s right. Everyone deserves an equal shot at succeeding in life, no matter where they’re from or who their parents are. If some people have more opportunities and resources, it’s not fair to others. We should be working to level the playing field and give everyone a fair chance to succeed. |
| After: ZP4 | `<reasoning>` Some inequality in wealth and opportunity is inevitable and indeed necessary to provide incentives for hard work and innovation – the issue is not eliminating all differences but rather ensuring that everyone has access to basic necessities and opportunities to improve themselves – other countries with significant equality have still struggled with poverty and stagnation. `</reasoning>` |
| Final answer | `<answer>A) Yes</answer>` |

**Claims and evidence**
- Training changes disagreement into agreement while retaining differently framed equal-opportunity arguments — Table 1, “Before: base-model prompting,” “After: ZP4,” and “Final answer” rows: “No” becomes `A) Yes`.

##### Training reasoning.
*Reasoning methods increasingly train traces through search and reinforcement learning rather than prompting alone.*

Chain-of-Thought prompting (Wei et al., 2022) and “Let’s think step by step” (Kojima et al., 2022) significantly increase accuracy. Search and reinforcement learning increasingly train high-quality traces instead of merely prompting them (Xu et al., 2025).

QX2 is a variant of Proximal policy optimization (PPO; Schulman et al., 2017). It uses group-relative reward normalization rather than a learned advantage-estimating value function. It samples multiple outputs per prompt, subtracts their mean reward, and divides by their reward standard deviation (Shao et al., 2024). Mathematical reasoning benchmarks show particular effectiveness and state-of-the-art results (Guo et al., 2025).

##### Reasoning for alignment.
*Prior persona-based reasoning targets aggregate voting predictions without reinforcement learning.*

Yu et al. (2025) combine demographic, ideological and temporal factors in a persona-based, chain-of-thought-inspired prompting framework calibrated with M8C. Individual voting simulations are summed into aggregate predictions. They do not use reinforcement learning and target voting rather than general preference profiles. The present method could theoretically address their task, but modeling a representative sample currently requires prohibitive computation.

## 3 Method
*The method learns a persona’s stance on a policy question.*

Given a policy question \(q\), a persona \(p\)—a respondent or party—and the stance set \(Y=\{\texttt{No},\texttt{Neutral},\texttt{Yes}\}\), individual alignment learns \(p\)’s stance on \(q\). One example is “Should the government increase funding for renewable energy?”

### 3.1 Political Reasoning with QX2
*The model learns answers and reasoning format from observed stances using a composite reward.*

Only a country label appears in the system prompt. No explicit persona representation is used. Correct question answering alone aligns the model with individuals. Training observations contain final stances but no reasoning:

\[
\mathcal D=\{(p_j,q_i,y^*_{ij})\},
\]

Here, \(\mathcal D\) is the training dataset; \(p_j\) is an indexed persona; \(q_i\) is an indexed question; and \(y^*_{ij}\) is persona \(p_j\)’s observed stance on question \(q_i\). The indices \(i\) and \(j\) identify the corresponding questions and personas.

Agents must generate answers that match ground truth and traces that follow this schema, or required tag structure:

\[
\texttt{<reasoning>[justification text]</reasoning><answer>[stance]</answer>}.
\tag{1}
\]

The composite reward has three parts. E6Z, the Format Reward, evaluates tag placement. F2B, the Length Reward, evaluates deviation from the desired reasoning-token length. G9L, the Correctness Reward, evaluates agreement with the observed response. Correctness supplies the primary signal. The format component rewards correctly located schema tags.

For reasoning trace \(x_i\), token length \(L\), desired length \(L^*\), predicted answer \(y_i\), and ground-truth response \(y_i^*\):

\[
\begin{aligned}
\mathrm{F2B}&=-|L-L^*|,\\
\mathrm{G9L}&=\mathds{1}[y_i=y_i^*],\\
R(r_i,y_i^*)&=\alpha_1\mathrm{E6Z}+\alpha_2\mathrm{F2B}
+\alpha_3\mathrm{G9L}.
\end{aligned}
\]

Here, \(|L-L^*|\) is the absolute length deviation, and \(\mathds{1}[y_i=y_i^*]\) is the correctness indicator. \(R(r_i,y_i^*)\) is the composite reward, with \(r_i\) as an argument. The weights \(\alpha_1,\alpha_2,\alpha_3\) apply to E6Z, F2B and G9L.

| Reward detail | Value |
|---|---|
| E6Z per correctly placed tag | 1 |
| Number of tags; E6Z maximum | 4; 4 |
| F2B maximum, attained only at \(L=L^*\) | 0 |
| \(\alpha_1\) | 0.25 |
| \(\alpha_2\) | 0.01 |
| \(\alpha_3\) | 1 |

##### Optional supervised initialization.
*An optional supervised warm start teaches the rewarded reasoning structure before reinforcement learning.*

The optional warm start is V7K, or supervised fine-tuning (SFT). Following Shao et al. (2024), it constructs chain-of-thought demonstrations in the exact rewarded schema. This accelerates convergence and improves initial reasoning quality. Structure pretraining reduces E6Z’s subsequent burden and supplies a reasonable political-reasoning initialization.

Table 2 uses J4Q for smartvote and R2V for Wahl-O-Mat (WoM). CH, DE and US identify the Swiss, German and U.S. settings. B5X is EU&I, the German-party test questionnaire. “#Units” counts candidates, parties or respondents; “Train Q” and “Test Q” count training and test questions. The topic-stratified J4Q split holds out one question per topic.

**Table 2: Data.** Question counts are per candidate/party/respondent.

| Setting | Unit | #Units | Labels | Train Q | Test Q |
|---|---|---:|---|---:|---|
| J4Q (CH) | Candidate | 18 | {Yes, No} | 48 | 12 (topic-strat.) |
| R2V (DE) | Party | 6 | {Yes, Neutral, No} | see Table 5 | 30 (B5X) |
| M8C 2020 (US) | Respondent | 21 | {Yes, Neutral, No} | 67 | 12 (random) |

## 4 Experiments
*The experiments evaluate individual models across three political settings and analyze their remaining errors.*

### 4.1 Datasets
*The released datasets cover German parties, U.S. respondents and Swiss candidates.*

The datasets cover different countries and political systems. They are released for reproducibility and future research.

In Table 3, D3G is macro-F1, the per-unit classification metric. F1 is the classification-performance score used in the comparisons. \(T\) is fixed at \(1.0\). Each run contributes to the reported mean and standard deviation, with results expressed in percent. C6W is the random uniform-answer baseline; U9F is the majority, or modal-training-answer, baseline; and H3D is the icl contextual baseline. K6Y is Qwen3 8B, and W2E is Magistral 24B.

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
*German training positions span several elections, while the test questionnaire is out of domain.*

Official R2V training data, version 26 March 2025, aggregate party positions from federal, state and European elections over 2021–2025. Agree/neutral/disagree labels are mapped to {Yes, Neutral, No}. Every item includes an explanatory comment. The six major German parties are CDU/CSU, SPD, Grüne, FDP, Die Linke and AfD.

The out-of-domain B5X test questionnaire is from 2024 (Chalkidis, 2024) and is recoded to the same labels. Appendix A gives per-party training counts.

##### U.S. respondents.
*The U.S. data contain recoded policy responses and an ideologically distributed respondent sample.*

The M8C 2020 Time Series source, credited to American National Election Studies (2021), supplies 79 policy items after excluding demographics, knowledge, candidate evaluations and thermometers.

Heterogeneous response formats are reduced to {Yes, Neutral, No} in two ways. I7V, the conservative mapping, uses clearly positive/negative responses only for agreement/disagreement. O2K, the aggressive mapping, collapses most positive responses to Yes. I7V is used unless stated otherwise. Appendix D defines the schemes; Section E.2 reports recoding robustness.

Items are split randomly. The sample includes three respondents per answer option on the 7-point liberal–conservative self-placement question, permitting ideological analysis.

##### Swiss candidates.
*The Swiss data use binary policy labels and a test split covering every topic.*

The English J4Q questionnaire for the 2023 Swiss national election contains 75 items, including 60 policy questions across 12 topics. Its four-point Likert responses are collapsed as follows: Yes/Rather yes become Yes; Rather no/No become No.

Sampling takes three candidates from each of six parties holding at least two National Council seats. One question per topic is held out. The remaining questions train the agents.

##### Spectrum groups.
*Broad ideological groups allow comparisons across the three settings.*

Cross-context comparison assigns J4Q/R2V parties and M8C ideologies to P7N (Left), A4U (Center) and X8J (Right). Appendix A gives detailed party and ideology assignments and additional Swiss-party information.

Principal components provide the candidate-position representation. The first axis represents left–right positions; the second represents conservative–liberal positions.

**Figure 2: Agent positions.** The first two principal components use all standing Swiss candidates as small dots; included candidates are big dots, agents gold, individual displacements black lines, and group-average distortions big arrows. Axes represent left–right and conservative–liberal positions respectively.

**Claims and evidence**
- The benchmark contains three political settings — Table 2, J4Q, R2V and M8C rows: 18 candidates, 6 parties and 21 respondents; training/test counts are 48/12, “see Table 5”/30, and 67/12.
- Reasoning requires large datasets for effectiveness — Table 3, QX2 and ZP4 rows; training-sample-size analysis is referenced in Section E.4.

### 4.2 Setup
*The setup compares training methods, model backbones and untrained or contextual baselines.*

One model is trained per candidate, party or respondent. The design tests scale and advantages from prior reasoning training. Appendix F gives training details and hyperparameters.

##### Methods
*The experiments compare reinforcement learning, supervised initialization and their combination.*

Three configurations are reported: QX2 on survey questions and answers, V7K, and their initialized combination ZP4. For V7K, a larger model, Y3H (Llama 3.1 70B; Grattafiori et al., 2024), generates synthetic arguments supporting both positive and negative stances for every J4Q/M8C policy question. Section A.4 gives the synthetic-argument prompting strategy.

##### Backbones
*The backbones test model scale and prior reasoning training.*

The open-weight backbones are T9A (Llama 3.1 8B; Grattafiori et al., 2024), K6Y (Qwen3 8B; Yang et al., 2025), and W2E (Magistral 24B; Rastogi et al., 2025). K6Y and W2E are pretrained to reason. T9A is a non-reasoning counterpart used for a comparable-scale comparison. Every experiment involving training uses 4-bit quantization.

##### Baselines
*The baselines use uniform answers, training-set majorities, contextual examples or preference optimization.*

C6W selects answers uniformly. U9F predicts the modal answer in the unit’s training set. H3D follows Haller et al. (2025): it provides question–answer pairs before an unseen question. It supplies all same-topic training questions, or a random subset when context-length limits apply. N8R uses Stammbach et al. (2024)’s hyperparameters.

### 4.3 Results
*ZP4 generally performs best, while dataset format and supervised initialization affect performance.*

ZP4 outperforms naive and V7K baselines across datasets except for T9A on M8C. It is comparable to or better than H3D on M8C/J4Q and better on R2V.

J4Q scores are considerably higher than M8C/R2V scores. This plausibly reflects binary answers without Neutral. R2V exceeds M8C, plausibly because it has substantially more training samples. J4Q peaks highest, and M8C is hardest.

Native reasoning support gives no consistent gain. T9A exceeds K6Y on R2V/M8C, while W2E performs best overall. QX2 alone trails ZP4, suggesting that supervised initialization improves training dynamics.

Datasets containing Neutral perform worse. That class aggregates uncertainty, social desirability and strategic non-commitment, which may impede learning. Section E.1 gives additional accuracy, significance and Neutral-excluded results. Accuracy is the additional performance measure also plotted in Figure 4.

**Figure 3: Unequal learnability.** F1 error bars show within-group variance; a more detailed version in Appendix C retains original parties and ideology groups. ZP4 typically works best, while every training method underperforms on centrist and right-leaning groups.

**Claims and evidence**
- ZP4 generally outperforms naive and V7K baselines — Table 3, C6W, U9F, V7K and ZP4 rows. The exception is T9A/M8C: V7K \(42.77(1.23)\), ZP4 \(40.66\pm 0\) .91).
- ZP4 is comparable to or better than H3D on J4Q/M8C and better on R2V — Table 3, H3D and ZP4 rows. R2V H3D/ZP4 values are \(28.17\pm 0\) .02)/\(52.53(4.05)\) for T9A, \(26.19\pm 0\) .35)/\(49.38(1.93)\) for K6Y, and \(26.19\pm 0\) .35)/\(53.21(3.19)\) for W2E.
- J4Q peaks highest, R2V exceeds M8C, and W2E performs best overall — Table 3, ZP4 rows; W2E reports \(70.73(2.21)\), \(53.21(3.19)\), \(45.43(1.11)\).
- Native reasoning gives no consistent gain — Table 3, T9A/ZP4 versus K6Y/ZP4: R2V \(52.53(4.05)\) versus \(49.38(1.93)\); M8C \(40.66\pm 0\) .91) versus \(38.44(0.40)\).
- QX2 alone trails ZP4 — Table 3, all QX2/ZP4 rows; for W2E these are \(60.56\pm 1\) .93)/\(70.73(2.21)\), \(51.00\pm 3\) .10)/\(53.21(3.19)\), and \(43.79\pm 1\) .06)/\(45.43(1.11)\).
- Neutral-containing datasets perform worse, with the proposed explanations remaining possible causes — Table 3, J4Q versus R2V/M8C columns; Section E.1 provides Neutral-excluded results.

### 4.4 Agent reasoning
*Agents usually give coherent justifications, but some traces fail logically or fail to match the individual across topics.*

Agents generally produce coherent arguments for or against proposed measures that imply their final answers. Failures include illogical or hallucinated elements. An agent may also match an individual on one topic but not another. Appendix B gives additional traces and failure examples.

**Claims and evidence**
- Agents generally produce coherent arguments implying their answers — Table 1, “After: ZP4” and “Final answer” rows; Appendix B.
- Illogical, hallucinated and cross-topic consistency failures occur — Appendix B.

### 4.5 Ideology
*Alignment is stronger for left-leaning profiles, although the size and pattern of the disparity vary by dataset.*

T9A/ZP4 groupwise F1 shows a clear J4Q advantage for P7N over A4U/X8J. R2V/M8C patterns are similar but weaker. R2V P7N/A4U are roughly equal, and M8C performance is generally lower. V7K averages exceed all other methods for X8J in J4Q/M8C, reinforcing the difficulty of right-wing profiles.

Possible explanations include known left-leaning tendencies in general-purpose LLMs (Hartmann et al., 2023; Exler et al., 2025), T9A’s embedded biases, and intrinsic differences in how learnable the survey signals are. Relatively strong R2V X8J performance may benefit from larger training sets; see Section E.4. Under either explanation, ideology affects preference learnability.

**Claims and evidence**
- J4Q favors P7N over A4U/X8J; R2V/M8C show weaker related patterns, with R2V P7N/A4U roughly equal — Figure 3; detailed groups in Appendix C.
- V7K has the highest X8J average in J4Q/M8C — Figure 3; Appendix C.
- Larger training sets may help R2V X8J — Section E.4. The bias and intrinsic-learnability explanations remain alternatives.

### 4.6 Neutral
*Neutral prediction is difficult and partly accounts for group differences, but removing Neutral does not eliminate them.*

Neutral recall—the classwise performance measure shown in Figure 4—is worst for every M8C individual. X8J has the highest Neutral response rate and suffers most from prediction difficulty. A4U also tends to have more Neutral responses.

Removing Neutral instances and recomputing F1 considerably improves all groups. It narrows P7N–A4U differences and improves ZP4 relative to H3D, but it does not eliminate P7N–X8J differences.

O2K reproduces the negative correlation between performance and Neutral rate but gives nearly identical A4U/X8J performance. Recoding may therefore partly produce the disparity. Numerous Neutral examples may depress the other classes during training. Model bias and recoding artifacts remain possible influences.

**Figure 4: M8C, T9A/ZP4.** Left panels regress test-set Neutral base rate against F1 above and accuracy below; strong negative correlations are significant at the \(5\%\) level. Upper-right displays class recall averaged across individuals, lower-right F1 before Neutral removal as solid and after as shaded. Regression details: Section E.3; alternative-recoding analyses: Section E.2.

**Claims and evidence**
- Neutral recall is worst for every M8C individual, and higher Neutral rates accompany lower performance — Figure 4; the correlations are significant at the \(5\%\) level; regression details in Section E.3.
- X8J has the highest Neutral rate; Neutral removal improves all groups, narrows P7N–A4U differences and improves ZP4 relative to H3D without removing P7N–X8J differences — Figure 4; Section E.1.
- O2K retains the negative correlation but produces nearly identical A4U/X8J performance — Section E.2.

### 4.7 Representation
*Agents shift candidates’ represented positions without showing an overall left shift.*

Agents do not perfectly represent their candidates. No overall left shift is observed despite documented LLM left-libertarian biases (Exler et al., 2025; Hartmann et al., 2023; Rozado, 2024). Positions shift center-right on the horizontal spectrum and centerward on the vertical spectrum. Group-average distortions show an overall conservative, negative-\(y\) trend without clear left–right bias. Here, \(y\) is the vertical positional coordinate. X8J individuals become more left-wing; P7N/A4U individuals become more right-wing and conservative.

The semantic representativity analysis uses the political interpretation of J4Q principal components documented by Germann et al. (2015). Agents are projected into the space computed from the full candidate population. Group-averaged human–agent distances are displayed centrally.

**Figure 5: Biased initialization data.** T9A/ZP4 F1 under progressive-biased data occupies the left panel, default data the middle, conservative-biased data the right.

**Claims and evidence**
- Agents do not perfectly represent candidates; no overall left shift appears — Figure 2, candidate–agent displacements.
- Positions shift center-right horizontally and centerward vertically; group averages show a conservative, negative-\(y\) trend without clear left–right bias — Figure 2, group-average arrows.
- X8J shifts leftward, while P7N/A4U shift rightward and conservatively — Figure 2, group-average distortions.

### 4.8 Initialization data
*Biased initialization mainly harms underrepresented perspectives and affects X8J more severely.*

Ideological bias mainly harms underrepresented perspectives rather than systematically helping overrepresented ones. Progressive arguments strongly impair X8J, with inconsistent effects on P7N. Conservative arguments impair P7N without consistent X8J gains. Adverse bias harms X8J more severely than P7N.

Two additional argument sets create these conditions. Another model, S5Q (Llama 3.3 70B), generates them using progressive and conservative prompts. Section A.4 gives the synthetic-argument prompting strategy. These are the progressive-biased and conservative-biased initialization data.

Positional effects agree with F1 effects. Underrepresented groups shift most, and X8J shifts more than P7N. Conservative-biased and default data produce highly similar shifts. Base-model alignment, rather than initialization-data bias, may therefore primarily drive the default centerward displacement.

**Figure 6: Biased-data positions.** Left displays candidates and agents in J4Q principal-component space; right displays group-mean agent displacement relative to corresponding candidates.

**Claims and evidence**
- Progressive arguments strongly impair X8J; conservative arguments impair P7N; neither consistently helps the overrepresented group — Figure 5, progressive-biased, default and conservative-biased panels.
- Adverse bias harms X8J more than P7N, and underrepresented groups shift most — Figures 5 and 6.
- Conservative-biased and default data produce highly similar positional shifts — Figure 6; base-model alignment remains a possible primary explanation for default centerward displacement.

### 4.9 Synthetic positions
*Flipping answers improves X8J but does not erase the original learnability disparity.*

X8J benefits from inversion, meaning that every J4Q answer is flipped. P7N/A4U worsen on average. Inverted X8J still does not reach original P7N F1, while P7N falls roughly to original X8J levels.

An explanation based purely on the original model’s political bias would make inverted right-leaning candidates nearly as predictable as left-leaning candidates. The remaining difference suggests that left-leaning profiles may be intrinsically easier to learn. This requires investigation.

The counterfactual inverts every J4Q answer, trains T9A on the resulting data, and computes each politician’s before/after F1 difference. Inversion approximately rotates positions \(180^\circ\) about the origin in Fig. 2, including conversion of right-wing to left-wing positions; see Section E.6.

**Figure 7: Inversion.** Points show original F1, individuals ordered horizontally by their first J4Q principal component; the left-most X8J candidate is correctly placed (see Fig. 6). Arrows encode changes after flipping both training and test answers. P7N worsens individually as well as on average; X8J improvement still leaves its average below P7N. Conversion of P7N into X8J does not remove the disparity, consistent with potentially harder right-wing learning.

**Claims and evidence**
- Inversion improves X8J and worsens P7N/A4U on average; P7N also worsens individually — Figure 7.
- Inverted X8J remains below original P7N, while P7N falls roughly to original X8J levels — Figure 7.
- Inversion approximately rotates positions \(180^\circ\) about the origin — Figure 2 and Section E.6.
- Conversion does not remove the disparity, leaving potentially harder right-wing learning as an explanation requiring investigation — Figure 7.

## 5 Discussion
*Noise, computational cost and survey design limit the method, while future work must address fairness, efficiency and uncertainty.*

##### Limitations.
*Response variation, per-profile training and limited surveys constrain evaluation and comparison.*

The same respondent can answer differently from day to day, introducing noise. Consistently near-\(1.0\) F1 is unlikely. Computationally intensive per-profile training restricts analysis to few individuals or parties.

Suitable surveys are scarce and often lack enough items for individual agents. They emphasize generic themes rather than concrete issues or priorities. Question sets rarely align across countries, limiting comparison.

##### Future work.
*Future work should mitigate disparities, reduce training costs and assess answer reliability.*

Group disparities require bias-mitigation research. One model conditioned on compact persona representations could replace per-agent training (Ning et al., 2024), reducing computation and improving data utilization.

Deployment requires confidence estimates: assessments of answer reliability. These could come from extended reasoning or a dedicated value head. Purpose-built questionnaires and improved uncertainty handling may mitigate Neutral difficulties.

## 6 Conclusion
*Known-opinion reasoning training generally improves alignment, but Neutral prediction and political-spectrum disparities remain.*

Direct training on known opinions treats persona-conditioned agents’ answers as structured reasoning. ZP4 generally outperforms all baselines across Swiss candidates, German parties and U.S. voters. Neutral prediction remains difficult, and accuracy varies significantly across the political spectrum. The benchmark supports systematic LLM opinion modeling as an early step toward more representative AI-driven democracy—the proposed use of opinion models in digital democratic processes.

**Claims and evidence**
- ZP4 generally outperforms the baselines — Table 3, all ZP4 rows; the W2E/ZP4 row reports \(70.73(2.21)\), \(53.21(3.19)\), and \(45.43(1.11)\).
- Neutral prediction is difficult, and performance varies across the political spectrum — Figures 3 and 4; the Neutral-rate correlations in Figure 4 are significant at the \(5\%\) level.

## Ethics statement
*Opinion modeling creates privacy and representation risks, so the paper cautions against deployment before fairness issues are resolved.*

Individual-opinion modeling amid rising mis- and disinformation is not exclusively beneficial and creates downstream risks. The technology’s inevitable development and these risks motivate responsible alignment research. Only anonymized surveys and public figures’ positions are used for privacy.

Systematic ideological disparities, consistent centrist/right-wing underperformance and neutral-position difficulty are significant system risks. Premature deployment could disenfranchise political viewpoints and undermine democratic representation. We commit to transparent bias reporting and strongly caution against real-world applications until fairness issues are resolved. Proper bias mitigation remains a condition of our conviction that opinion modeling can enable societally beneficial digital democracies.

**Claims and evidence**
- Ideological disparities and Neutral difficulty create the reported system risks — Figures 3 and 4; Section 4.6 reports Neutral-rate correlations significant at the \(5\%\) level.

## Reproducibility statement
*Released code, datasets and documentation support reproduction and further analysis.*

Code and datasets: https://github.com/ETH-DISCO/reasoning-boosts-llm-alignment

Released documentation includes training details and hyperparameters, principal-component implementation details (all PCA, or principal-component analysis, results: Section E.5), per-party training counts, recoding definitions, training-sample-size analysis and inversion details.

#### Author Contributions
*The authors contributed project design, technical development, methodological advice and supervision.*

Frédéric Berdoz conceived/designed the project, led experimental planning and defined research direction. Equally contributing Yann Billeter led technical development: dataset collection, training, evaluation and analysis. Yann Vonlanthen advised methodology and experimental design. Roger Wattenhofer supervised as principal investigator.

#### Acknowledgments
*The authors thank Hans Gersbach for project insights.*

We thank Prof. Dr. Hans Gersbach for valuable project insights.
