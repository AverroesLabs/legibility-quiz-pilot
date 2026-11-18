# Reasoning Boosts Opinion Alignment in LLMs

Frédéric Berdoz

  
Yann Billeter

  
Yann Vonlanthen

  
Roger Wattenhofer

Affiliation: ETH Zurich

Affiliation: Switzerland

Affiliation: {fberdoz, ybilleter, yvonlanthen, wattenhofer}@ethz.ch

###### Abstract

Opinion modeling aims to capture individual or group political preferences. Capturing these preferences could enable applications such as digital democracies, where models could help shape fairer and more popular policies. Large language models (LLMs) are natural candidates for opinion modeling because of their versatility, strong generalization capabilities, and demonstrated success across diverse text-to-text applications. However, because of their statistical nature and limited causal understanding, LLMs tend to produce biased opinions when prompted naively.

In this work, we study whether reasoning can improve opinion alignment: the correspondence between a model’s answers and the opinions the model is intended to represent. Motivated by recent advances in mathematical reasoning enabled by reinforcement learning (RL), we train models to produce profile-consistent answers through structured reasoning. We evaluate this approach on three datasets covering U.S., European, and Swiss politics. The results indicate that reasoning enhances opinion modeling and is competitive with strong baselines. However, reasoning does not fully remove bias. Therefore, additional mechanisms are needed to build faithful political digital twins using LLMs. By releasing both our method and datasets,[^1] we establish a solid baseline to support future research on LLM opinion alignment.

[^1]: Code and data: https://github.com/ETH-DISCO/reasoning-boosts-llm-alignment

## 1 Introduction

Could artificial intelligence (AI) give rise to a new kind of democracy, where digital twins vote on our behalf and faithfully reflect our opinions on every issue? Accurate simulations of political behavior offer new opportunities to understand election outcomes and improve policy and democratic processes (Li et al., 2024). However, capturing how diverse individuals reason about political issues remains challenging. Although large language models (LLMs) can generate sophisticated political discourse, LLMs often fail to reflect the true diversity of human political viewpoints (Santurkar et al., 2023; Qu and Wang, 2024; Yu et al., 2025; Qi et al., 2024). This failure raises a fundamental question: how can agents reason about politics while faithfully representing human political diversity?

Current approaches to modeling opinions with LLMs predominantly rely on prompting with demographic information and political affiliations. These approaches therefore leverage learned correlations between demographics and opinions. Although prompt-based methods have grown increasingly sophisticated, these methods consistently fail to capture real opinion distributions and exhibit unstable, inconsistent responses across different prompts and demographic groups (Santurkar et al., 2023; Ball et al., 2025). In contrast, we seek a preference-consistent approach that directly uses an individual’s known opinions rather than demographic proxies.

The aforementioned methods do not lend themselves to simulating individual political preferences. Methods using richer data sources, such as interview transcripts, can achieve strong performance in modeling individual personas (Park et al., 2024). However, interview-based methods remain limited by prohibitive data collection costs because interviewing is impractical at scale. We therefore propose using political survey data as an alternative.

Large surveys, such as the American National Election Studies (ANES), and Voting Advice Applications (VAAs)—online platforms that match voters with parties or candidates—aim to capture how people position themselves on political issues. Although surveys lack the narrative richness and textual signal of interviews, surveys offer a structured representation of political opinions across populations. Surveys are already prevalent in bias evaluation (Santurkar et al., 2023; Argyle et al., 2023; Haller et al., 2025) and group-level simulation (Argyle et al., 2023; Cao et al., 2025). However, surveys have not yet been widely adopted for training individual-level political agents.

Reinforcement learning has recently driven major gains in reasoning, with models such as o3/o4-mini (OpenAI, 2025) and DeepSeek-R1 (Guo et al., 2025) setting new marks on various benchmarks. Motivated by these results, we use reinforcement learning, in particular Group-relative policy optimization (GRPO) (Shao et al., 2024), to train agents on survey data. We treat opinion formation as a reasoning problem: the agent writes a short rationale and a final choice, and the agent receives a reward when the choice matches the respondent’s answer. This reward teaches the agent to reason toward survey-consistent positions on unseen questions.

We summarize our contributions as follows:

- **Reinforcement learning for reasoning-based opinion alignment.** We introduce a method that encourages explicit reasoning through reinforcement learning, specifically GRPO, to improve opinion alignment from survey responses.

- **Benchmarking on real-world political data.** We evaluate our approach on three datasets of authentic political opinions: German parties, U.S. voters, and Swiss candidates. We release these datasets as a benchmark to foster future research. Our results show that reasoning consistently improves opinion alignment.

- **Analyzing ideological effects.** We investigate how political ideology influences performance by inverting the survey answers of individuals with known positions. This analysis reveals systematic variations in alignment quality across the political spectrum.

**Figure 1: GRPO for opinion alignment.** We use public opinion surveys to align LLMs with individual preference profiles. First, the LLM undergoes supervised fine-tuning (SFT) with (synthetic) statements and ground-truth answers so that the LLM adheres to the reasoning template. After fine-tuning, the model answers in the correct format but is not fully aligned with opinions. We therefore use GRPO with a reward model that rewards proper formatting and correct answers to further improve reasoning.

## 2 Related Work

Modeling political opinions requires LLMs to adopt specific political personas and answer from those particular viewpoints. Prior work has explored demographic personas, which represent groups with shared traits such as occupation or ethnicity; character personas, which represent well-known individuals; and individualized personas, which are digital profiles based on personalized data (Chen et al., 2024; Tseng et al., 2024). Because our focus is modeling individuals, individualized personas are the most relevant category. However, the literature has predominantly employed demographic approaches, while only a handful of studies represent individualized methods.

##### Modeling political opinions with demographic personas.

Political opinion modeling has primarily used demographic personas through various prompting strategies. Simple approaches use basic political affiliation, for example, “You are a democrat. What is your opinion on…” (Santurkar et al., 2023). More complex methods add demographic details such as age, education, and location (Argyle et al., 2023; Santurkar et al., 2023; Hwang et al., 2023; Sun et al., 2024; Wang et al., 2025). Recent work has also explored richer contexts. For example, prompts with demonstrations from belief networks provide related political positions to guide the model’s answer (Chuang et al., 2024). Gudiño et al. (2024) combine demographics with demonstrations, while Stammbach et al. (2024) integrate prompting with odds ratio preference optimization (ORPO) (Hong et al., 2024) to better align outputs with target positions. Despite this methodological diversity, demographic approaches face several limitations.

##### Limitations of demographic approaches.

Santurkar et al. (2023) identify three critical issues in prompt-based demographic simulation that undermine authentic political representation:

- **Representativeness:** LLMs’ default opinion distributions fail to reflect true population distributions.
- **Steerability:** Demographic prompting provides insufficient control over model outputs.
- **Consistency:** Political biases vary unpredictably across different topics and contexts.

These limitations manifest as systematic biases: accuracy varies unpredictably across demographic groups (Argyle et al., 2023; Santurkar et al., 2023). Although models can approximate overall opinion distributions, models fail to capture natural variance within demographic subgroups and remain sensitive to prompt variations (Sun et al., 2024; Ball et al., 2025).

##### Moving beyond demographics.

In contrast to the extensive literature on demographic prompting, few studies have explored representing individual opinion profiles. Park et al. (2024) use interview transcripts to achieve strong accuracy on public opinion surveys, outperforming demographic baselines. However, transcript availability limits this approach. Haller et al. (2025) propose an in-context learning method: the model receives respondents’ question-answer pairs in the prompt and then predicts the respondents’ answers to test questions. However, Haller et al. (2025) focus on bias identification rather than opinion modeling.

**Table 1: Answers obtained from Llama 3.1 8B trained on an individual in the Right group in ANES.** Here, “8B” denotes a model size of 8 billion parameters, and “Right” denotes the broad political group assigned from ANES ideological self-placement, as described in Section 4.1 and Appendix A. Before training, Llama disagreed; after SFT+GRPO training, Llama agreed. Interestingly, the reasoning trace still appeals to similar arguments (“equal opportunity”) but frames those arguments differently. Additional examples are provided in Appendix B.

| Item | Text |
|---|---|
| Question | Do you agree that it is not really that big a problem if some people have more of a chance in life than others? |
| Before (prompting on base model) | No. I don’t think that’s right. Everyone deserves an equal shot at succeeding in life, no matter where they’re from or who their parents are. If some people have more opportunities and resources, it’s not fair to others. We should be working to level the playing field and give everyone a fair chance to succeed. |
| After (SFT+GRPO) | `<reasoning>`<br>Some inequality in wealth and opportunity is inevitable and indeed necessary to provide incentives for hard work and innovation – the issue is not eliminating all differences but rather ensuring that everyone has access to basic necessities and opportunities to improve themselves – other countries with significant equality have still struggled with poverty and stagnation.<br>`</reasoning>`<br><br>`<answer>A) Yes</answer>` |

##### Training LLMs to reason.

Encouraging LLMs to produce reasoning steps before answering significantly increases accuracy. Examples include Chain-of-Thought prompting (Wei et al., 2022), which elicits intermediate reasoning steps, and simple prompts such as “Let’s think step by step” (Kojima et al., 2022). Recent advances have shifted from prompting-based reasoning to training models to generate high-quality reasoning traces through search algorithms or reinforcement learning (Xu et al., 2025).

One such training method is Group-relative policy optimization (GRPO) (Shao et al., 2024). GRPO is a variant of Proximal policy optimization (PPO) (Schulman et al., 2017). Rather than learning a value function to estimate advantages, GRPO uses group-relative reward normalization. Specifically, for each prompt, GRPO samples a group of outputs and normalizes the outputs’ rewards by subtracting the group mean and dividing by the group standard deviation. GRPO has proven particularly effective for mathematical reasoning tasks, achieving state-of-the-art results on benchmarks (Guo et al., 2025).

##### Reasoning for opinion alignment.

Yu et al. (2025) propose a persona-based, chain-of-thought-inspired prompting framework that combines demographic, ideological, and temporal factors calibrated using ANES. Yu et al. (2025) use this framework to simulate individual respondents’ voting behavior and then sum the simulated responses to make aggregate predictions.

Unlike our proposed method, the framework of Yu et al. (2025) does not use reinforcement learning. Moreover, our method focuses on learning general preference profiles, whereas Yu et al. (2025) target voting predictions. Although our method could theoretically be applied to voting prediction, the computational cost of modeling a representative sample size is currently prohibitive.

## 3 Method

We study individual-level opinion alignment from survey data. Consider a policy question such as “Should the government increase funding for renewable energy?” The possible answer stances form a set, for example, \(Y=\{\texttt{No},\texttt{Neutral},\texttt{Yes}\}\). Let \(q\) denote a policy question and \(p\) denote a persona, which may be a respondent or a political party. Given any question \(q\) and persona \(p\), our goal is to learn the stance of persona \(p\) on question \(q\).

### 3.1 Political Reasoning with GRPO

We employ GRPO (Shao et al., 2024) to train agents that generate a reasoning trace before providing an answer. Every generated response is intended to follow this fixed schema, where `[justification text]` is the reasoning trace and `[stance]` is the answer:

\[
\texttt{<reasoning>[justification text]</reasoning><answer>[stance]</answer>}.
\tag{1}
\]

Our training data consist of survey responses:

\[
\mathcal{D}=\{(p_j,q_i,y^*_{ij})\},
\]

where \(\mathcal{D}\) is the survey-response dataset, \(p_j\) is persona \(j\), \(q_i\) is question \(i\), and \(y^*_{ij}\) is the observed stance of persona \(p_j\) on question \(q_i\). The key challenge is that the survey-response dataset contains only final stances, not reasoning traces. Therefore, the model must learn to generate reasoning that both adheres to the format in Equation (1) and produces answers matching the ground-truth stances.

To address both requirements, we evaluate each generated response using a composite reward with three components:

**Format Reward (\(R_{\text{format}}\)).** We reward correctly placed tags in the schema in Equation (1). Each of the four tags yields a score of 1, for a maximum format reward of 4.

**Length Reward (\(R_{\text{length}}\)).** Let \(x_i\) be a generated reasoning trace, \(L\) be the token length of that trace, and \(L^*\) be the desired length. We define a symmetric penalty:

\[
R_{\text{length}}=-|L-L^*|.
\]

Thus, the length reward reaches its maximum of 0 only when the reasoning-trace length equals \(L^*\).

**Correctness Reward (\(R_{\text{correct}}\)).** The primary reward signal comes from matching the survey response. Let \(y_i\) be the model’s predicted answer and \(y_i^*\) the ground-truth survey response. Then

\[
R_{\text{correct}}=\mathds{1}[y_i=y_i^*],
\]

where \(\mathds{1}\) is the indicator function: the correctness reward is 1 when the predicted answer matches the ground truth and 0 otherwise.

For a generated response \(r_i\), containing the reasoning trace and predicted answer, the total reward is

\[
R(r_i,y_i^*)=
\alpha_1R_{\text{format}}
+\alpha_2R_{\text{length}}
+\alpha_3R_{\text{correct}},
\]

with reward weights \(\alpha_1=0.25\), \(\alpha_2=0.01\), and \(\alpha_3=1\).

We do not use an explicit persona representation. Instead, the system prompt includes only a country label, and the model is aligned with individuals solely through correctly answering questions.

##### Optional Initialization via Supervised Fine-tuning.

To accelerate convergence and improve initial reasoning quality, we optionally initialize the model through supervised fine-tuning (SFT) before GRPO training. Following Shao et al. (2024), we construct a dataset of chain-of-thought demonstrations, with each demonstration following the exact format in Equation (1) that our reward design incentivizes.

The SFT stage therefore serves as a warm start in two ways: (i) pre-training the model to use the correct output structure reduces the burden on the format reward, \(R_{\text{format}}\), during GRPO; and (ii) the demonstrations provide a reasonable initialization for political reasoning.

**Table 2: Dataset overview.** A unit is a candidate, party, or respondent, as specified in the second column. Train/test counts refer to questions per unit. The settings are smartvote in Switzerland (CH), Wahl-O-Mat (WoM) in Germany (DE), and ANES 2020 in the United States (US). “Topic-strat.” means topic-stratified selection, with one question per topic held out for testing. EU&I is the out-of-domain test questionnaire described in Section 4.1.

| Setting | Unit | #Units | Labels | Train Q | Test Q |
|---|---|---|---|---|---|
| smartvote (CH) | Candidate | 18 | {Yes, No} | 48 | 12 (topic-strat.) |
| WoM (DE) | Party | 6 | {Yes, Neutral, No} | see Table 5 | 30 (EU&I) |
| ANES 2020 (US) | Respondent | 21 | {Yes, Neutral, No} | 67 | 12 (random) |

## 4 Experiments

### 4.1 Datasets

We use three datasets from different countries and political systems. Table 2 provides an overview, and the following paragraphs describe dataset creation. We release all data for reproducibility and future research.

**Table 3: Reasoning requires large datasets to be effective.** Entries report mean macro-F1 (%) with standard deviations over 8 stochastic runs at sampling temperature \(T=1.0\). Macro-F1 is the unweighted average of the class-level F1 scores; each class-level F1 combines precision and recall through their harmonic mean. For each run, we compute macro-F1 separately for each unit and then average across units. Section E.1 reports accuracy scores—the proportions of correctly predicted answers—and statistical significance.

The methods are defined in Sections 3.1 and 4.2: `random` chooses answer options uniformly at random; `majority` always chooses the unit’s most frequent training answer; `icl` denotes in-context learning from question-answer pairs; ORPO denotes odds ratio preference optimization; SFT denotes supervised fine-tuning; GRPO denotes Group-relative policy optimization; and SFT+GRPO applies SFT before GRPO. Model-size suffixes 8B and 24B denote 8 and 24 billion parameters, respectively.

Scores on smartvote are considerably better than scores on ANES and WoM. This difference is likely due to smartvote’s binary labels and absence of a Neutral option (see Section 4.6). Performance on WoM is better than performance on ANES, likely because WoM provides substantially more training samples (see also Section E.4).

| Base model | Method | smartvote | WoM | ANES |
|---|---|---|---|---|
| Untrained baselines | random | \(50.0\) | \(33.33\) | \(33.33\) |
| Untrained baselines | majority | \(37.43\) | \(27.44\) | \(22.98\) |
| Llama 3.1 8B | icl | $55.97\pm 5$ .66) | $28.17\pm 0$ .02) | $23.2\pm 1$ .79) |
| Llama 3.1 8B | ORPO | $43.53\pm 2$ .81) | $43.29\pm 5$ .16) | $34.84\pm 5$ .41) |
| Llama 3.1 8B | SFT | $63.44\pm 1$ .34) | $48.95\pm 3$ .56) | \(42.77(1.23)\) |
| Llama 3.1 8B | GRPO | $55.14\pm 1$ .25) | $37.29\pm 2$ .06) | $34.55\pm 1$ .32) |
| Llama 3.1 8B | SFT+GRPO | \(66.88(2.18)\) | \(52.53(4.05)\) | $40.66\pm 0$ .91) |
| Qwen3 8B | icl | $60.48\pm 4$ .14) | $26.19\pm 0$ .35) | $23.2\pm 1$ .79) |
| Qwen3 8B | ORPO | $23.87\pm 2$ .07) | $25.25\pm 3$ .58) | $26.95\pm 1$ .36) |
| Qwen3 8B | SFT | $61.08\pm 2$ .74) | $42.91\pm 3$ .44) | $35.14\pm 1$ .70) |
| Qwen3 8B | GRPO | $60.64\pm 2$ .01) | $31.42\pm 1$ .46) | $31.47\pm 1$ .06) |
| Qwen3 8B | SFT+GRPO | \(65.11(3.33)\) | \(49.38(1.93)\) | \(38.44(0.40)\) |
| Magistral 24B | icl | $66.16\pm 4$ .2) | $26.19\pm 0$ .35) | $19.23\pm 2$ .91) |
| Magistral 24B | ORPO | $23.31\pm 2$ .11) | $24.73\pm 3$ .29) | $24.25\pm 2$ .28) |
| Magistral 24B | SFT | $67.63\pm 1$ .91) | $51.86\pm 2$ .58) | $39.15\pm 0$ .66) |
| Magistral 24B | GRPO | $60.56\pm 1$ .93) | $51.00\pm 3$ .10) | $43.79\pm 1$ .06) |
| Magistral 24B | SFT+GRPO | \(70.73(2.21)\) | \(53.21(3.19)\) | \(45.43(1.11)\) |

##### German party positions.

For training, we use the official Wahl-O-Mat (WoM) dataset, version 26 March 2025. The dataset aggregates party positions across federal, state, and European elections from 2021–2025. We focus on six major parties: CDU/CSU, SPD, Grüne, FDP, Die Linke, and AfD. Each item records an agree/neutral/disagree position and an explanatory comment. We map the positions to {Yes, Neutral, No}.

For testing, we use EU&I (2024) (Chalkidis, 2024) as an out-of-domain questionnaire containing 30 items. We recode the EU&I responses to the same three labels. Per-party training counts appear in Appendix A.

##### American National Election Studies.

From ANES 2020 Time Series (American National Election Studies, 2021), we extract 79 policy items. We exclude demographics, knowledge, candidate evaluations, and thermometers. Because the remaining items use heterogeneous response formats, we recode each item to {Yes, Neutral, No}.

We use two recoding schemes, detailed in Appendix D. The **conservative mapping** treats only clearly positive or negative options as agreement or disagreement. In contrast, the **aggressive mapping** collapses most positive responses to Yes. Unless otherwise stated, results use the conservative scheme. Section E.2 analyzes robustness to recoding.

We split the items at random into 67 training questions and 12 test questions. To analyze ideological variation, we sample respondents according to their answers to ANES’ 7-point liberal–conservative self-placement question. We include 3 respondents per answer option, for a total of 21 respondents.

##### Swiss candidates.

We use the English smartvote questionnaire for the 2023 Swiss national elections. The questionnaire contains \(75\) items, of which \(60\) are policy questions spanning 12 topics. Responses use a four-point Likert scale. We collapse the responses to {Yes, No}: Yes and Rather yes map to Yes, while Rather no and No map to No.

We sample three candidates from each of the six parties with at least two National Council seats, for 18 candidates in total. Appendix A provides additional information on these parties. We perform a topic-stratified split by holding out one question per topic for testing. This produces 12 test questions and leaves 48 questions for training.

##### Political Spectrum Groups.

For clarity and comparability across political contexts, we assign parties in smartvote and Wahl-o-Mat, and ideologies in ANES, to three broad political groups: Left, Center, and Right. Appendix A details the assignments.

**Figure 2: Agents are more centrist and conservative.** The figure shows the first two dimensions of a principal component analysis (PCA), which represents the survey positions in component dimensions. The PCA uses all candidates standing for election in the 2023 Swiss national elections in smartvote; these candidates appear as small dots. The x- and y-axes correspond to the left-right and conservative-liberal spectra, respectively. Big dots mark the candidates included in the smartvote dataset, gold markers show those candidates’ agents, and black lines connect each candidate to the corresponding agent. Unlike results in the literature indicating a left-libertarian bias (Exler et al., 2025; Hartmann et al., 2023; Rozado, 2024), we observe that our agents are shifted towards the (center-)right. The group-average distortions between ground-truth and agent positions, shown as big arrows, indicate an overall trend towards more conservative positions (negative y) and no clear left-right bias. Implementation details are given in Section E.5.

### 4.2 Experimental setup

We evaluate our approach on different models and against strong baselines. The experimental design targets two questions: (i) how performance scales with model size, and (ii) whether reasoning-pretrained backbones confer advantages over non-reasoning peers. We train one model per unit—that is, one model per candidate, party, or respondent. Appendix F gives further training details.

##### Methods

We report results for three configurations: GRPO on the survey questions and answers, as well as SFT+GRPO. For smartvote and ANES, we generate synthetic arguments supporting both positive and negative stances for each policy question using Llama 3.1 70B (Grattafiori et al., 2024), a 70-billion-parameter model. We use these synthetic arguments for SFT. Section A.4 details the prompting strategy used to produce the synthetic arguments.

##### Model backbones..

We evaluate three open-weight backbones to study (i) the effect of model scale and (ii) the impact of prior reasoning pre-training: Llama3.1 8B (Grattafiori et al., 2024), Qwen3 8B Yang et al. (2025), and Magistral 24B Rastogi et al. (2025).

Qwen3 and Magistral are pre-trained to reason. These backbones therefore allow us to compare reasoning-pretrained models against a non-reasoning counterpart, Llama3.1, at comparable scales. We use 4-bit quantization for all models in all experiments involving training.

##### Baselines

We report two naive baselines. The `random` baseline selects one answer uniformly at random. The `majority` baseline answers every question with the most frequently chosen answer option in the corresponding unit’s training set.

We also compare against an in-context learning baseline, `icl`. Following Haller et al. (2025), we prompt the LLM with question-answer pairs and then ask the LLM to answer another, unseen question. Specifically, the model receives all training questions from the same topic area as the test question, or a random subset of those training questions when context length is limiting.

Finally, we compare against ORPO Hong et al. (2024), using the same hyperparameters as Stammbach et al. (2024).

### 4.3 Results and analysis

Table 3 reports macro-F1 averages for all datasets. SFT+GRPO outperforms the naive and SFT baselines across datasets, except for Llama 3.1 8B on ANES. For that exception, the Llama 3.1 8B rows in Table 3 report \(40.66\) for SFT+GRPO versus \(42.77\) for SFT.

Performance varies by dataset. Scores peak on smartvote at \(70.73\), while ANES is hardest at \(45.43\); both values come from the Magistral 24B SFT+GRPO row in Table 3. SFT+GRPO performs comparably to or better than the `icl` baselines on ANES and smartvote, and outperforms the `icl` baselines on WoM; the relevant comparisons appear in the `icl` and SFT+GRPO rows for each backbone in Table 3.

Using base models with native reasoning support does not consistently improve performance. In particular, with SFT+GRPO, Llama 3.1 8B outperforms Qwen3 8B on WoM (\(52.53\) versus \(49.38\)) and ANES (\(40.66\) versus \(38.44\)), while Magistral 24B performs best overall (Table 3). GRPO alone underperforms SFT+GRPO in the corresponding rows of Table 3, suggesting that the supervised warm start improves training dynamics.

We observe lower performance on datasets with Neutral classes. A likely explanation is that Neutral aggregates multiple behaviors—uncertainty, social desirability, and strategic non-commitment—and is therefore difficult to learn. Section 4.5 further investigates the impact of ideology, and Section 4.6 investigates the role of Neutral. Section E.1 reports results without Neutral.

**Figure 3: Not all political positions are equally learnable.** F1 scores show that, although SFT+GRPO typically works best, every training method underperforms on center and right-leaning groups. Error bars show variance within groups. These disparities may stem from biases baked into Llama 3.1 8B or from inherent differences in how well various political preferences can be learned from survey data. Either way, the results demonstrate that ideology affects the learnability of political preferences. Appendix C presents a detailed figure with the original parties and ideology groups.

### 4.4 How do agents reason?

Table 1 presents an example of reasoning traces. Generally, agents succeed in making coherent arguments for or against a proposed measure, and those arguments imply the final answer.

However, we identify two failure modes. First, arguments sometimes contain illogical or hallucinated elements. Second, an agent may argue in line with an individual’s opinion on one topic but fail to argue in line with that individual’s opinion on other topics. Appendix B provides additional traces and examples of failures.

### 4.5 Political ideology matters

Figure 3 reports F1 by ideology group for Llama 3.1 8B (SFT+GRPO). On smartvote, Figure 3 shows a clear performance disparity between the Left group and the Center and Right groups. Wahl-o-Mat and ANES show similar but weaker patterns. Specifically, on Wahl-o-Mat, the Left and Center groups perform roughly equally well, while performance is generally lower on ANES.

For the Right group in smartvote and ANES, SFT also outperforms all other methods on average, as shown in Figure 3. This pattern further suggests difficulties with right-wing preference profiles. The disparities may reflect known left-leaning tendencies in general-purpose LLMs (Hartmann et al., 2023; Exler et al., 2025), or the disparities may indicate that Right profiles are intrinsically harder to learn from survey signals. In contrast, the relatively strong Right performance on WoM may be aided by larger training sets; see Section E.4.

### 4.6 Neutral reasoning poses a particular challenge

Figure 4 (top right) shows that recall—the proportion of ground-truth instances of a class that the model predicts correctly—is worst for the Neutral class for all individuals in ANES. Furthermore, the regressions in Figure 4 (left column) show that individuals in the Right group respond with Neutral the most. These individuals therefore have a higher **Neutral base rate**, meaning a higher proportion of Neutral answers. Consequently, difficulties predicting Neutral hurt performance most for these individuals.

To examine Neutral’s contribution to the performance gap, we remove all Neutral instances and recompute the F1 scores. Performance improves considerably, as shown in Figure 4 (bottom right), but removing Neutral does not rectify the gap between the Left and Right groups. SFT+GRPO also improves relative to the `icl` baselines when Neutral is removed; see Section E.1.

We repeat this analysis using the alternative recoding scheme introduced in Section 4.1 and detailed in Appendix D. Under the alternative scheme, we again observe a negative correlation between predictive performance and Neutral base rate. However, the Center and Right groups perform almost identically. This difference between recoding schemes indicates that the group-performance pattern could, to some extent, be an artifact of recoding ANES. Section E.2 reports the details.

**Figure 4: Learning neutral stances is hard.** On ANES, predictive performance is strongly correlated with the Neutral base rate on the test set. Individuals in the Right group, and to some extent in the Center group, tend to answer questions with Neutral more often. For both F1 (top left) and accuracy (bottom left), the correlation is negative and significant at the \(5\%\)-level. Section E.3 gives regression details. **Top right:** Recall by class on ANES. Averaged over all individuals, the model struggles most with predicting the Neutral class. **Bottom right:** ANES F1 scores before removing Neutral instances (solid) and after removing Neutral instances (shaded). All groups improve, and the gap between Left and Center becomes narrow. However, the difference between Left and Right remains. Although the large number of Neutral answers likely depressed performance on the other two classes during training, the remaining gap could also suggest that other factors affect the performance disparity, such as model bias or artifacts from recoding ANES. All results were obtained from Llama 3.1 8B trained with SFT+GRPO.

### 4.7 Whom do agents represent?

To better understand agents’ representativity in semantic terms, we compare the agents’ positions with the ground-truth positions of the corresponding human individuals in a two-dimensional model of politics. We use smartvote because the interpretation of the dataset’s principal components is documented (Germann et al., 2015).

Figure 2 presents the individuals included in the smartvote dataset in the space spanned by the first two principal components. These components are obtained from the full set of candidates in the 2023 Swiss national elections. We then project the agent positions, shown in gold, into the same space.

Not all agents perfectly represent their respective candidates, which appear as big markers in Figure 2. However, despite documented left-libertarian biases in most LLMs (Exler et al., 2025; Hartmann et al., 2023; Rozado, 2024), we do not observe an overall left shift. Instead, agent positions shift towards the center-right of the left-right spectrum, represented by the \(x\)-axis, and towards the center of the conservative-liberal spectrum, represented by the \(y\)-axis.

The group-averaged distances between humans and agents, shown in the center of Figure 2, further support this observation: Right individuals become more left-wing, while Left and Center individuals become more right-wing and conservative.

**Figure 5: Biased SFT data consistently impairs the performance of underrepresented groups.** The figure reports F1 scores for Llama trained with SFT+GRPO on differently biased datasets. **Left:** Data with a progressive bias strongly impairs the Right group without consistently benefiting the Left group. **Middle:** Performance on the default dataset. **Right:** Data with a conservative bias decreases the Left group’s performance without consistently improving the Right group’s performance. These asymmetric effects suggest that ideological bias in SFT data primarily harms the underrepresented perspective rather than systematically improving performance on the overrepresented perspective.

### 4.8 Impact of SFT data

To investigate the impact of biases in SFT data, we generate two additional sets of arguments using Llama 3.3 70B. For one set, we prompt the model to produce progressive arguments; for the other, we prompt the model to produce conservative arguments. Section A.4 provides details.

Figure 5 presents the resulting F1 scores. Progressive bias strongly impairs Right candidates while having inconsistent effects on Left candidates. Conversely, conservative bias impairs Left candidates without consistently benefiting Right candidates. These results suggest that ideological bias in SFT data primarily harms underrepresented perspectives rather than systematically improving overrepresented perspectives. Notably, adverse bias impairs Right candidates more severely than Left candidates.

Figure 6 shows how the same SFT-data biases affect agent positions in PCA space. Consistent with the F1 results in Figure 5, biased data primarily shifts underrepresented groups, and Right candidates are again affected more strongly than Left candidates. However, the positional shifts under conservative-biased and default SFT data are highly similar. This similarity suggests that the centerward shift observed in Figure 2 may be driven primarily by the agent’s base-model alignment rather than by bias in the SFT data itself.

**Figure 6: Biased SFT data primarily shifts the agents of underrepresented groups.** **Left:** Candidates and their agents in the smartvote PCA space. **Right:** Group-mean displacement vectors of agents relative to their respective candidates. As with the F1 scores in Figure 5, biased data mostly affects the underrepresented group, and Right candidates are again affected more strongly than Left candidates. Notably, conservative-biased and default SFT data produce similar displacement patterns. This similarity suggests that the centerward shift under default data may originate from the base model’s alignment rather than from the SFT data.

### 4.9 Performance on synthetic positions

To assess how individual preferences affect an associated agent’s performance, we create a counterfactual dataset by inverting every smartvote answer and train Llama-3.1-8B on the resulting dataset. Inversion roughly corresponds to a \(180^{\circ}\) rotation about the origin in Figure 2; see Section E.6. Thus, inversion converts, for example, right-wing positions into left-wing positions.

For each politician, we compute the difference between F1 before inversion and F1 after inversion. Figure 7 shows these differences. Members of the Right group benefit from inversion, while the Left and Center groups perform worse on average. Yet even after inversion, the Right group does not reach the Left group’s original F1, while the Left group drops to roughly the Right group’s original level.

If the effect were purely due to the model’s original political bias, flipping labels should have made right-leaning candidates nearly as easy to predict as left-leaning candidates. Because flipping labels does not produce that outcome, the results suggest that left-leaning preference profiles may be intrinsically easier to model. This possibility merits further investigation.

## 5 Discussion

##### Limitations.

Learning human opinions is inherently limited by noise because even the same respondent might answer a question differently on different days. Hence, consistently achieving an F1 score close to \(1.0\) is unlikely.

The analyses in this paper cover only a small number of individuals or parties. This limited coverage is mainly due to the method’s computational intensity: the method requires training one model per profile.

Data are another constraint. Suitable surveys are scarce, and existing surveys often contain too few items to train individualized agents or focus on generic themes rather than concrete issues and priorities. Moreover, question sets rarely align across countries, which limits comparability.

##### Future work.

The observed disparities between political groups warrant further investigation into bias mitigation. Further research should aim to replace per-agent training with a single model that conditions on compact persona representations, for example, as in Ning et al. (2024). Such a model would reduce computational cost and improve data utilization.

Real-world deployments will require confidence estimates to gauge answer reliability. These estimates could be obtained by extending the reasoning scheme or using a dedicated value head. Purpose-built questionnaires and improved uncertainty handling may also mitigate the observed difficulties with the Neutral class.

**Figure 7: Learning Right and Center positions is harder than Left ones.** Points depict F1 scores on the original data. Individuals are ordered along the x-axis by their first component in the smartvote PCA space. Note that the left-most Right candidate is indeed correctly placed; see Figure 6. Arrows show the difference in F1 after all answers in the training and test sets have been flipped. Left candidates perform worse after flipping, both individually and on average. Right candidates improve but still perform worse on average than Left candidates. Because converting Left to Right does not remove the disparity, the results could suggest that Right positions are more difficult to learn.

## 6 Conclusion

In this work, we explore individual opinion alignment by training persona-conditioned agents directly on known opinions and framing answer generation as a structured reasoning task. Across Swiss candidates in smartvote, German parties in Wahl-o-Mat, and U.S. voters in ANES, SFT+GRPO generally outperforms all baselines; Table 3 reports the dataset- and backbone-specific results.

However, performance remains uneven. Neutral stances remain challenging to predict, as shown in Figure 4, and accuracy varies significantly across the political spectrum, as examined in Sections 4.5 and 4.6. Our work provides a strong benchmark and baseline for systematic opinion modeling with LLMs, marking an early step toward the possibility of more representative, AI-driven forms of democracy.

## Ethics statement

We acknowledge that modeling individual opinions in a time of rising mis- and disinformation is not exclusively beneficial and may pose downstream risks. Precisely because of these risks and the technology’s inevitable development, we believe responsible research is essential to investigate how individual opinion alignment can be improved.

To address privacy concerns, we use only anonymized survey data and public figures’ positions. Our research reveals systematic performance disparities across ideological groups: center and right-wing profiles consistently underperform, and neutral positions are difficult to model. We commit to transparent reporting of these biases because the biases represent significant system risks and could disenfranchise certain political viewpoints if deployed prematurely.

We therefore strongly caution against real-world applications until these fairness issues are resolved. Biased political modeling systems risk undermining democratic representation. Ultimately, we are convinced that, with proper bias mitigation, opinion modeling can enable societally beneficial applications such as digital democracies, where models could help shape fairer and more popular policies.

## Reproducibility statement

For reproducibility, we release our code and datasets:

https://github.com/ETH-DISCO/reasoning-boosts-llm-alignment

Appendix F gives training details, including hyperparameters. Section E.5 describes the details for all PCA results.

#### Author Contributions

Frédéric Berdoz conceived and designed the project, led experimental planning, and defined its research direction.
Yann Billeter contributed equally, leading technical development, including dataset collection, model training, evaluation, and analysis. Yann Vonlanthen contributed as an advisor, offering guidance on methodology and experimental design. Roger Wattenhofer supervised the project as principal investigator.

#### Acknowledgments

We thank Prof. Dr. Hans Gersbach for his valuable insights during the project.
