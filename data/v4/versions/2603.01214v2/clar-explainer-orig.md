**Summary**

This paper asks whether reasoning improves large language models’ alignment with individual political opinions. The method trains one model per respondent, candidate, or party. Models produce a rationale and a final answer. Group-relative policy optimization rewards formatting, reasoning length, and agreement with survey answers. Supervised fine-tuning can provide a warm start. Experiments cover Swiss candidates, German parties, and U.S. voters. SFT+GRPO generally outperforms the baselines, with exceptions, and performs best on smartvote. Neutral answers are difficult to predict. Performance also varies by ideology. Biased fine-tuning data mainly harms underrepresented perspectives. The agents retain political distortions, so the method does not fully remove bias.

**Glossary**

- **Opinion modeling** — Modeling individual or group political preferences.
- **Opinion alignment** — Matching a model’s answers to an individual’s known political opinions.
- **AI** — The technology considered for digital twins that could vote on people’s behalf.
- **Digital twin** — A model intended to faithfully reflect an individual’s political opinions.
- **Digital democracy** — A proposed application in which models could help shape fairer and more popular policies.
- **Large language model (LLM)** — The type of text-to-text model studied for political opinion modeling.
- **Text-to-text applications** — Applications across which the paper says LLMs have demonstrated success.
- **Generalization** — A capability cited as a reason to use LLMs for opinion modeling.
- **Bias** — Systematic distortions in modeled opinions or performance across political groups.
- **Political agent** — A model trained to answer political questions for a preference profile.
- **Preference profile** — An individual’s or party’s political opinions.
- **Persona** — The respondent or political party whose stance the model is to learn.
- **Demographic persona** — A persona representing people with shared traits, such as occupation or ethnicity.
- **Character persona** — A persona representing a well-known individual.
- **Individualized persona** — A digital profile based on personalized data.
- **Demographic proxy** — Demographic information used in place of an individual’s known opinions.
- **Prompting** — Supplying information or instructions to guide a model’s answers.
- **In-context learning (icl)** — Prompting with question-answer pairs before requesting an answer to an unseen question.
- **Belief-network demonstrations** — Related political positions supplied to guide a model’s answer.
- **Representativeness** — Whether model opinion distributions reflect population distributions.
- **Steerability** — The control demographic prompting provides over model outputs.
- **Consistency** — Whether political biases remain stable across topics and contexts.
- **Reinforcement learning (RL)** — Training that rewards generated answers matching survey responses.
- **Group-relative policy optimization (GRPO)** — An RL method that normalizes rewards within groups of generated outputs.
- **Proximal policy optimization (PPO)** — The method of which GRPO is a variant.
- **Group-relative reward normalization** — Subtracting a group’s mean reward and dividing by its standard deviation.
- **Value function** — The advantage-estimation component that GRPO avoids learning.
- **Supervised fine-tuning (SFT)** — Training on formatted reasoning demonstrations before GRPO.
- **SFT+GRPO** — SFT initialization followed by GRPO training.
- **Warm start** — SFT initialization intended to improve formatting and political reasoning before GRPO.
- **Chain-of-Thought** — Producing reasoning steps before answering.
- **Reasoning trace** — The justification generated before the final stance.
- **Odds ratio preference optimization (ORPO)** — The preference-optimization method used in related work and as a baseline.
- **Reward model / reward function** — The mechanism that scores formatting, reasoning length, and answer correctness.
- **Synthetic arguments** — Model-generated arguments supporting policy stances and used for SFT.
- **Backbone** — A model used as the starting point for training.
- **Reasoning-pretrained backbone** — A model pre-trained to reason.
- **Open-weight backbones** — The three model backbones evaluated in the experiments.
- **4-bit quantization** — The quantization setting used in all experiments involving training.
- **American National Election Studies (ANES)** — The U.S. survey source used for respondent-level modeling.
- **ANES 2020 Time Series** — The ANES dataset from which 79 policy items are extracted.
- **Voting Advice Application (VAA)** — An online platform matching voters with parties or candidates.
- **Wahl-O-Mat / Wahl-o-Mat (WoM)** — The source of German party positions used for training.
- **EU&I (2024)** — The out-of-domain, 30-item questionnaire used to test German party models.
- **smartvote** — The Swiss candidate dataset based on the 2023 national-election questionnaire.
- **U.S. / US** — The country represented by ANES.
- **CH** — The country code used for the Swiss smartvote setting.
- **DE** — The country code used for the German WoM setting.
- **CDU/CSU** — One of the six German parties included in the WoM dataset.
- **SPD** — One of the six German parties included in the WoM dataset.
- **Grüne** — One of the six German parties included in the WoM dataset.
- **FDP** — One of the six German parties included in the WoM dataset.
- **Die Linke** — One of the six German parties included in the WoM dataset.
- **AfD** — One of the six German parties included in the WoM dataset.
- **Left, Center, Right** — The three broad political groups used to compare parties and respondent ideologies.
- **Liberal–conservative self-placement** — The ANES 7-point question used to sample respondents by ideology.
- **Policy item / policy question** — A survey question about a political issue.
- **Stance / label** — An answer such as Yes, Neutral, or No.
- **Yes** — The label used for agreement or positive responses under the stated recoding.
- **No** — The label used for disagreement or negative responses under the stated recoding.
- **Neutral** — A label that may combine uncertainty, social desirability, and strategic non-commitment.
- **Conservative recoding** — Mapping only clearly positive or negative ANES responses to agreement or disagreement.
- **Aggressive recoding** — Mapping most positive ANES responses to Yes.
- **Four-point Likert scale** — The Swiss response scale collapsed from Yes/Rather yes and Rather no/No to Yes and No.
- **Topic-stratified split** — Holding out one Swiss question per topic for testing.
- **Out-of-domain questionnaire** — EU&I, used for testing after training on WoM.
- **Unit** — A candidate, party, or respondent with a separately trained model.
- **#Units** — The number of candidates, parties, or respondents in a dataset.
- **Train Q** — The number of training questions per unit.
- **Test Q** — The number of test questions per unit.
- **topic-strat.** — The topic-stratified test split used for smartvote.
- **random** — The baseline that selects an answer uniformly at random.
- **majority** — The baseline that always chooses the most frequent answer in a unit’s training set.
- **Macro-F1** — The reported score computed per unit and then averaged across units for each run.
- **F1 / F1-score** — The performance score used in the result tables and figures.
- **Accuracy** — The additional predictive-performance measure reported in the analyses.
- **Recall** — The class-specific measure used to examine difficulty predicting Neutral.
- **Standard deviation** — The variation reported across 8 stochastic runs in Table 3.
- **Variance** — The within-group variation represented by error bars in Figure 3.
- **Neutral base rate** — The frequency of Neutral answers on the test set.
- **Regression** — The analysis relating predictive performance to the Neutral base rate.
- **Statistical significance** — The reported significance of performance comparisons or correlations.
- **Principal component analysis (PCA)** — The analysis used to place candidates and agents in a political-position space.
- **Principal components** — The dimensions defining the PCA space used for smartvote.
- **Left-right spectrum** — The political dimension represented by the PCA x-axis.
- **Conservative-liberal spectrum** — The political dimension represented by the PCA y-axis.
- **Displacement / distortion** — The difference between a candidate’s position and the corresponding agent’s position.
- **Counterfactual inversion** — Flipping every smartvote answer before training and testing.
- **Progressive-biased SFT data** — Arguments generated by prompting for progressive arguments.
- **Conservative-biased SFT data** — Arguments generated by prompting for conservative arguments.
- **Underrepresented perspective** — The political perspective harmed by opposing bias in the SFT data.
- **Hallucinated element** — An element identified as a failure in an agent’s reasoning.
- **Confidence estimate** — A proposed measure of answer reliability.
- **Value head** — A proposed dedicated component for estimating answer reliability.
- **o3/o4-mini** — OpenAI reasoning models cited as motivation.
- **DeepSeek-R1** — A reasoning model cited as motivation and evidence of GRPO’s effectiveness.
- **Llama 3.1 8B / Llama3.1 8B / Llama-3.1-8B** — The non-reasoning backbone evaluated in the experiments.
- **Qwen3 8B** — A reasoning-pretrained backbone evaluated in the experiments.
- **Magistral 24B** — A reasoning-pretrained backbone evaluated in the experiments.
- **Llama 3.1 70B** — The model used to generate the default synthetic SFT arguments.
- **Llama 3.3 70B** — The model used to generate progressive and conservative SFT arguments.
- **ETH Zurich** — The authors’ affiliation in Switzerland.
- **\(Y\)** — The set of possible answer stances.
- **\(q\), \(q_i\)** — A policy question.
- **\(p\), \(p_j\)** — A persona: a respondent or political party.
- **\(i,j\)** — The indices used for questions and personas in the survey data.
- **\(\mathcal D\)** — The collection of persona, question, and observed-stance triples.
- **\(y^*_{ij}\)** — Persona \(p_j\)’s observed stance on question \(q_i\).
- **\(y_i\)** — The model’s predicted answer.
- **\(y_i^*\)** — The ground-truth survey response.
- **\(x_i\)** — A reasoning trace.
- **\(L\)** — The token length of a reasoning trace.
- **\(L^*\)** — The desired reasoning-trace length.
- **\(R_{\text{format}}\)** — The reward for correctly placed output tags.
- **\(R_{\text{length}}\)** — The negative absolute difference between actual and desired reasoning length.
- **\(R_{\text{correct}}\)** — The reward for matching the ground-truth response.
- **\(R(r_i,y_i^*)\)** — The weighted total reward.
- **\(r_i\)** — The first argument of the total-reward expression.
- **\(\alpha_1,\alpha_2,\alpha_3\)** — The weights on formatting, length, and correctness rewards.
- **\(\mathds{1}[y_i=y_i^*]\)** — The indicator used to reward a correct answer.
- **\(|L-L^*|\)** — The absolute difference between actual and desired reasoning length.
- **\(T\)** — The setting fixed at \(1.0\) for the stochastic runs in Table 3.
- **\(\pm\)** — The notation used for reported means and standard deviations in Table 3.
- **\(x\)-axis, \(y\)-axis** — The left-right and conservative-liberal axes in the smartvote PCA space.
- **\(180^\circ\)** — The approximate rotation associated with answer inversion.
- **`<reasoning>` and `</reasoning>`** — The opening and closing tags around a justification.
- **`<answer>` and `</answer>`** — The opening and closing tags around a final stance.
- **`[justification text]`** — The reasoning placeholder in the required output schema.
- **`[stance]`** — The final-answer placeholder in the required output schema.
- **`A) Yes`** — The final answer shown in Table 1’s trained-model example.
- **Code and data repository** — https://github.com/ETH-DISCO/reasoning-boosts-llm-alignment.

# Reasoning Boosts Opinion Alignment in LLMs

Frédéric Berdoz

  
Yann Billeter

  
Yann Vonlanthen

  
Roger Wattenhofer

Affiliation: ETH Zurich

Affiliation: Switzerland

Affiliation: {fberdoz, ybilleter, yvonlanthen, wattenhofer}@ethz.ch

###### Abstract

*Reasoning-based training improves political opinion modeling, but does not fully remove bias.*

Opinion modeling aims to capture individual or group political preferences. It could support digital democracies, where models help shape fairer and more popular policies. Large language models (LLMs), the text-to-text models studied here, are natural candidates because of their versatility, generalization capabilities, and success across diverse text-to-text applications. However, due to their statistical nature and limited causal understanding, they tend to produce biased opinions when prompted naively.

We ask whether reasoning improves opinion alignment: matching a model’s answers to known political opinions. Motivated by advances in mathematical reasoning through reinforcement learning (RL), we train models to produce a rationale followed by a profile-consistent answer. RL rewards answers that match survey responses.

We evaluate on three datasets covering U.S., European, and Swiss politics. Reasoning improves opinion modeling and is competitive with strong baselines. It does not fully remove bias. Additional mechanisms are therefore needed for faithful political digital twins—models of individuals’ political opinions. We release our method and datasets¹ to establish a baseline for future research.

¹ Code and data: https://github.com/ETH-DISCO/reasoning-boosts-llm-alignment

**Claims and evidence**
- Reasoning improves alignment but leaves disparities: Table 3, SFT+GRPO rows; Figures 3–4.
- The method does not fully remove political distortions: Figure 2, candidate-to-agent shifts.

## 1 Introduction

*The paper proposes learning individual political preferences from survey answers through rewarded reasoning.*

Could AI, used here to create digital twins, enable a democracy in which those twins vote on our behalf and faithfully reflect our opinions on every issue?

Accurate political-behavior simulations could help explain election outcomes and improve policies and democratic processes (Li et al., 2024). Yet modeling how diverse individuals reason about politics remains difficult. LLMs can generate sophisticated political discourse, but often fail to reflect the diversity of human political viewpoints (Santurkar et al., 2023; Qu and Wang, 2024; Yu et al., 2025; Qi et al., 2024). The question is how to build agents that reason about politics while faithfully representing that diversity.

Most current methods prompt LLMs with demographic information and political affiliations. They use learned correlations between demographics and opinions. Despite increasingly sophisticated prompts, these methods fail to capture real opinion distributions and give unstable, inconsistent responses across prompts and demographic groups (Santurkar et al., 2023; Ball et al., 2025). We instead seek preference consistency: directly using an individual’s known opinions rather than demographic proxies.

These existing methods are not well suited to simulating individual preferences. Interview transcripts can support strong individual-persona modeling (Park et al., 2024), but collecting interviews at scale is prohibitively costly. We propose political surveys as an alternative. The American National Election Studies (ANES), a U.S. survey source, and Voting Advice Applications (VAAs), online platforms matching voters with parties or candidates, capture positions on political issues. Surveys lack interviews’ narrative richness and textual signal, but provide structured political opinions across populations. They are common in bias evaluation (Santurkar et al., 2023; Argyle et al., 2023; Haller et al., 2025) and group-level simulation (Argyle et al., 2023; Cao et al., 2025), but have not been widely adopted to train individual political agents.

RL has recently improved reasoning. Models such as o3/o4-mini (OpenAI, 2025) and DeepSeek-R1 (Guo et al., 2025) have set new benchmark marks. Motivated by these results, we use Group-relative policy optimization (GRPO) (Shao et al., 2024), an RL method that normalizes rewards within groups of generated outputs. We treat opinion formation as a reasoning problem. An agent writes a short rationale and final choice, and receives a reward when its choice matches the respondent’s answer. This trains it to reason toward survey-consistent positions on unseen questions.

Our contributions are:

- **Reinforcement learning for reasoning-based opinion alignment.** We introduce GRPO training that encourages explicit reasoning to improve alignment from survey responses.
- **Benchmarking on real-world political data.** We evaluate authentic opinions from German parties, U.S. voters, and Swiss candidates. We release these datasets as a benchmark. Our results show that reasoning consistently improves alignment.
- **Analyzing ideological effects.** We invert survey answers from individuals with known political positions. This reveals systematic differences in alignment across the political spectrum.

Figure 1: GRPO for opinion alignment. We use public opinion surveys to align LLMs with individual preference profiles. First, the LLM is fine-tuned with (synthetic) statements and ground-truth answers to adhere to the reasoning template. After fine-tuning, the model answers in the correct format, but is not fully aligned with opinions. We use GRPO with a reward model that rewards proper formatting and correct answers to further improve reasoning.

**Claims and evidence**
- Reasoning-based training improves alignment: Table 3, SFT+GRPO rows; the Llama 3.1 8B ANES exception is discussed in Section 4.3.
- Alignment varies with ideology and answer inversion: Figures 3 and 7.

## 2 Related Work

*Prior work largely uses demographic personas; individual profiles and trained reasoning motivate this method.*

Political opinion modeling requires models to adopt political personas and answer from those viewpoints. A **demographic persona** represents people sharing traits such as occupation or ethnicity. A **character persona** represents a well-known individual. An **individualized persona** is a digital profile based on personalized data (Chen et al., 2024; Tseng et al., 2024). Individualized personas are most relevant here, but only a handful of studies use them. Most use demographic personas.

##### Modeling political opinions with demographic personas.

*Demographic methods use political affiliations, demographic details, demonstrations, and preference optimization.*

Simple demographic prompts give a political affiliation, such as “You are a democrat. What is your opinion on…” (Santurkar et al., 2023). More complex prompts add age, education, and location (Argyle et al., 2023; Santurkar et al., 2023; Hwang et al., 2023; Sun et al., 2024; Wang et al., 2025).

Other methods provide belief-network demonstrations: related political positions that guide an answer (Chuang et al., 2024). Gudiño et al. (2024) combine demographics with demonstrations. Stammbach et al. (2024) combine prompting with odds ratio preference optimization (ORPO) (Hong et al., 2024), a preference-optimization method, to align outputs with target positions. These varied methods still face limitations.

##### Limitations of demographic approaches.

*Demographic prompting has problems with representativeness, steerability, and consistency.*

Santurkar et al. (2023) identify three problems:

- **Representativeness:** default model opinion distributions do not reflect population distributions.
- **Steerability:** demographic prompts provide insufficient control over outputs.
- **Consistency:** political biases vary unpredictably across topics and contexts.

Accuracy varies unpredictably across demographic groups (Argyle et al., 2023; Santurkar et al., 2023). Models may approximate overall opinion distributions, but fail to capture natural variation within demographic subgroups. They also remain sensitive to prompt changes (Sun et al., 2024; Ball et al., 2025).

##### Moving beyond demographics.

*Few studies represent individual opinion profiles, and existing approaches have data or scope limitations.*

Park et al. (2024) use interview transcripts to outperform demographic baselines on public-opinion surveys, but transcript availability limits this approach. Haller et al. (2025) use **in-context learning**: prompting with a respondent’s question-answer pairs, then predicting answers to test questions. Their focus is bias identification rather than opinion modeling.

The following example uses Llama 3.1 8B, a backbone evaluated here. **SFT+GRPO** means supervised fine-tuning (SFT)—training on formatted reasoning demonstrations—followed by GRPO. **Left, Center, and Right** are the broad political groups used in the experiments.

Table 1: Answers obtained from Llama 3.1 8B trained on an individual in the Right group in ANES. Before training, Llama disagreed; after training, it agreed. Interestingly, the reasoning trace still appeals to similar arguments (“equal opportunity”) but frames them differently. Additional examples are provided in Appendix B.

| Item | Content |
|---|---|
| Question: | Do you agree that it is not really that big a problem if some people have more of a chance in life than others? |
| Before (prompting on base model) | No. I don’t think that’s right. Everyone deserves an equal shot at succeeding in life, no matter where they’re from or who their parents are. If some people have more opportunities and resources, it’s not fair to others. We should be working to level the playing field and give everyone a fair chance to succeed. |
| After (SFT+GRPO) | `<reasoning>`<br>Some inequality in wealth and opportunity is inevitable and indeed necessary to provide incentives for hard work and innovation – the issue is not eliminating all differences but rather ensuring that everyone has access to basic necessities and opportunities to improve themselves – other countries with significant equality have still struggled with poverty and stagnation.<br>`</reasoning>` |
| | `<answer>A) Yes</answer>` |

##### Training LLMs to reason.

*Reasoning can be encouraged by prompts or trained through search and reinforcement learning.*

Producing reasoning steps before answering significantly increases accuracy. Examples include Chain-of-Thought prompting (Wei et al., 2022) and “Let’s think step by step” (Kojima et al., 2022). Recent methods train high-quality reasoning traces through search or RL rather than prompting alone (Xu et al., 2025).

GRPO is a variant of Proximal policy optimization (PPO) (Schulman et al., 2017). Instead of learning a value function to estimate advantages, it normalizes rewards within a group. For each prompt, it samples outputs, subtracts their mean reward, and divides by their reward standard deviation. GRPO has been particularly effective in mathematical reasoning, achieving state-of-the-art benchmark results (Guo et al., 2025).

##### Reasoning for opinion alignment.

*Prior reasoning-based opinion work uses prompting for voting prediction rather than RL for general preference profiles.*

Yu et al. (2025) propose persona-based, chain-of-thought-inspired prompting. It combines demographic, ideological, and temporal factors calibrated using ANES. They simulate individual voting behavior and sum responses for aggregate predictions.

Unlike our method, they do not use RL. They target voting prediction; we learn general preference profiles. Our method could theoretically address their task, but modeling a representative sample is currently computationally prohibitive.

**Claims and evidence**
- Training changes the example answer from disagreement to agreement: Table 1, “Before (prompting on base model)” and “After (SFT+GRPO)” rows; the latter ends with `A) Yes`.
- Interview-based individual modeling can outperform demographic baselines: Park et al. (2024).
- Prompted and trained reasoning improve performance in prior work: Wei et al. (2022), Kojima et al. (2022), and Guo et al. (2025).

## 3 Method

*The method learns survey stances by rewarding structured reasoning and correct final answers.*

We study individual-level alignment from survey data. A policy question might ask, “Should the government increase funding for renewable energy?” Its possible stances form a set, such as
\[
Y=\{\texttt{No},\texttt{Neutral},\texttt{Yes}\}.
\]
Given a policy question \(q\) and persona \(p\), meaning a respondent or political party, we aim to learn \(p\)’s stance on \(q\).

### 3.1 Political Reasoning with GRPO

*GRPO rewards output formatting, reasoning length, and agreement with observed survey answers.*

We train agents with GRPO (Shao et al., 2024). Each agent generates a reasoning trace—a justification before its answer—in this fixed schema:

\[
\texttt{<reasoning>[justification text]</reasoning><answer>[stance]</answer>}.
\tag{1}
\]

The tags `<reasoning>` and `</reasoning>` enclose the justification. The tags `<answer>` and `</answer>` enclose the stance. `[justification text]` and `[stance]` mark those two contents.

The training data are
\[
\mathcal D=\{(p_j,q_i,y^*_{ij})\},
\]
where \(p_j\) indexes personas, \(q_i\) indexes questions, and \(y^*_{ij}\) is persona \(p_j\)’s observed stance on question \(q_i\). The dataset contains final stances but no reasoning traces. The model must therefore learn both the required format and reasoning that leads to ground-truth answers.

Each generation receives a composite reward:

**Format Reward (\(R_{\text{format}}\)).** Correctly placed tags in schema (1) receive 1 point each. There are four tags, so the maximum is 4.

**Length Reward (\(R_{\text{length}}\)).** Let \(x_i\) be a reasoning trace, \(L\) its token length, and \(L^*\) the desired length. The symmetric penalty is
\[
R_{\text{length}}=-|L-L^*|.
\]
The absolute difference \(|L-L^*|\) measures the length mismatch. The maximum reward is 0, reached only when \(L=L^*\).

**Correctness Reward (\(R_{\text{correct}}\)).** Let \(y_i\) be the predicted answer and \(y_i^*\) the ground-truth survey response:
\[
R_{\text{correct}}=\mathds{1}[y_i=y_i^*].
\]
The indicator \(\mathds{1}[y_i=y_i^*]\) rewards a matching answer.

The total reward is
\[
R(r_i,y_i^*)=
\alpha_1R_{\text{format}}+
\alpha_2R_{\text{length}}+
\alpha_3R_{\text{correct}},
\]
with
\[
\alpha_1=0.25,\qquad \alpha_2=0.01,\qquad \alpha_3=1.
\]
Here, \(\alpha_1,\alpha_2,\alpha_3\) weight the three rewards; \(r_i\) is the first argument of the total-reward expression.

We use no explicit persona representation. The system prompt includes only a country label. Alignment with an individual comes solely from correctly answering questions.

##### Optional Initialization via Supervised Fine-tuning.

*SFT provides a formatted reasoning warm start before GRPO.*

To accelerate convergence and improve initial reasoning, we optionally apply SFT before GRPO. Following Shao et al. (2024), we construct chain-of-thought demonstrations in exactly the format of (1).

This warm start has two purposes: (i) teaching the format reduces the burden on \(R_{\text{format}}\) during GRPO; and (ii) it provides a reasonable initialization for political reasoning.

In Table 2, a **unit** is a candidate, party, or respondent. **#Units** counts these units. **Train Q** and **Test Q** count questions per unit. The country codes **CH**, **DE**, and **US** identify Switzerland, Germany, and the U.S. **topic-strat.** means topic-stratified.

Table 2: Dataset overview. Train/test counts refer to questions per unit (candidate/party/respondent).

| Setting | Unit | #Units | Labels | Train Q | Test Q |
|---|---|---|---|---|---|
| smartvote (CH) | Candidate | 18 | {Yes, No} | 48 | 12 (topic-strat.) |
| WoM (DE) | Party | 6 | {Yes, Neutral, No} | see Table 5 | 30 (EU&I) |
| ANES 2020 (US) | Respondent | 21 | {Yes, Neutral, No} | 67 | 12 (random) |

## 4 Experiments

*The experiments compare methods, datasets, model backbones, ideological groups, and training-data biases.*

### 4.1 Datasets

*Three datasets provide political opinions from German parties, U.S. respondents, and Swiss candidates.*

We use three datasets from different countries and political systems. The following paragraphs explain their construction. We release all data for reproducibility and future research. Table 2 gives an overview.

Table 3 reports **macro-F1**, the performance score computed per unit and then averaged across units in each run. **F1** is the score used throughout the result figures. Standard deviations describe variation across the stochastic runs; \(\pm\) is used in the reported values. \(T\) is the run setting fixed at \(1.0\).

Table 3: Reasoning requires large datasets to be effective. Mean macro-F1 (%) with standard deviations over 8 stochastic runs at \(T{=}1.0\). For each run, we compute per-unit macro-F1 and then average across units. We report accuracy scores and statistical significance in Section E.1. Scores on smartvote are considerably better than on ANES and WoM. This is likely due to the binary nature of smartvote and the absence of a Neutral option (see Section 4.6). Performance on WoM is better than on ANES, likely because of the substantially bigger number of training samples (see also Section E.4).

| Base model | Method | Dataset: smartvote | Dataset: WoM | Dataset: ANES |
|---|---|---|---|---|
| Untrained baselines | random | \(50.0\) | \(33.33\) | \(33.33\) |
| | majority | \(37.43\) | \(27.44\) | \(22.98\) |
| Llama 3.1 8B | icl | \(55.97\pm 5\) .66) | \(28.17\pm 0\) .02) | \(23.2\pm 1\) .79) |
| | ORPO | \(43.53\pm 2\) .81) | \(43.29\pm 5\) .16) | \(34.84\pm 5\) .41) |
| | SFT | \(63.44\pm 1\) .34) | \(48.95\pm 3\) .56) | \(42.77(1.23)\) |
| | GRPO | \(55.14\pm 1\) .25) | \(37.29\pm 2\) .06) | \(34.55\pm 1\) .32) |
| | SFT+GRPO | \(66.88(2.18)\) | \(52.53(4.05)\) | \(40.66\pm 0\) .91) |
| Qwen3 8B | icl | \(60.48\pm 4\) .14) | \(26.19\pm 0\) .35) | \(23.2\pm 1\) .79) |
| | ORPO | \(23.87\pm 2\) .07) | \(25.25\pm 3\) .58) | \(26.95\pm 1\) .36) |
| | SFT | \(61.08\pm 2\) .74) | \(42.91\pm 3\) .44) | \(35.14\pm 1\) .70) |
| | GRPO | \(60.64\pm 2\) .01) | \(31.42\pm 1\) .46) | \(31.47\pm 1\) .06) |
| | SFT+GRPO | \(65.11(3.33)\) | \(49.38(1.93)\) | \(38.44(0.40)\) |
| Magistral 24B | icl | \(66.16\pm 4\) .2) | \(26.19\pm 0\) .35) | \(19.23\pm 2\) .91) |
| | ORPO | \(23.31\pm 2\) .11) | \(24.73\pm 3\) .29) | \(24.25\pm 2\) .28) |
| | SFT | \(67.63\pm 1\) .91) | \(51.86\pm 2\) .58) | \(39.15\pm 0\) .66) |
| | GRPO | \(60.56\pm 1\) .93) | \(51.00\pm 3\) .10) | \(43.79\pm 1\) .06) |
| | SFT+GRPO | \(70.73(2.21)\) | \(53.21(3.19)\) | \(45.43(1.11)\) |

##### German party positions.

*WoM supplies German party training positions, and EU&I supplies an out-of-domain test.*

We train on the official Wahl-O-Mat (WoM) dataset, version 26 March 2025. It aggregates party positions from federal, state, and European elections in 2021–2025. We use six major German parties: CDU/CSU, SPD, Grüne, FDP, Die Linke, and AfD.

Each item contains agree/neutral/disagree and an explanatory comment. We map these to {Yes, Neutral, No}: the agreement, neutral, and disagreement labels used here. For testing, EU&I (2024) (Chalkidis, 2024) provides an out-of-domain questionnaire of 30 items, recoded to the same labels. Appendix A gives per-party training counts.

##### American National Election Studies.

*ANES provides 79 recoded policy items and 21 respondents sampled across self-reported ideology.*

From ANES 2020 Time Series (American National Election Studies, 2021), we extract 79 policy items. We exclude demographics, knowledge, candidate evaluations, and thermometers.

Response formats differ, so we use two mappings to {Yes, Neutral, No}, detailed in Appendix D. The **conservative mapping** treats only clearly positive or negative options as agreement or disagreement. The **aggressive mapping** collapses most positive responses to Yes. Unless stated otherwise, results use conservative recoding. Section E.2 examines robustness to recoding.

We randomly split the items into 67 training and 12 test questions. To study ideology, we use ANES’ 7-point liberal–conservative self-placement question. We sample 3 respondents per answer option, totaling 21 respondents.

##### Swiss candidates.

*smartvote supplies binary policy answers from 18 candidates, split by topic.*

We use the English questionnaire for the 2023 Swiss national elections, called smartvote here. It contains \(75\) items, including \(60\) policy questions across 12 topics. Its four-point Likert response scale is collapsed to {Yes, No}: Yes/Rather yes becomes Yes, and Rather no/No becomes No.

We sample three candidates from each of the six parties holding at least two National Council seats, totaling 18 candidates. Appendix A gives more information on these parties.

The topic-stratified split holds out one question per topic for testing: 12 test questions and 48 training questions.

##### Political Spectrum Groups.

*Parties and respondent ideologies are assigned to Left, Center, or Right for comparison.*

For comparability, we assign parties in smartvote and Wahl-o-Mat, and ANES ideologies, to Left, Center, or Right. Appendix A details the assignments.

The positional analysis below uses **principal component analysis (PCA)**. Its first two principal components define the political-position space. The **x-axis** represents left-right positions; the **y-axis** represents conservative-liberal positions. Candidate-to-agent differences are called shifts or distortions.

Figure 2: Agents are more centrist and conservative. First two dimensions of the principal component analysis of all candidates (small dots) standing for election in the 2023 Swiss national elections (smartvote). The x- and y-axes correspond to the left-right and conservative-liberal spectra, respectively. The shifts between the positions of the candidates (big dots) included in the smartvote dataset and their agents (gold) are depicted by black lines. Unlike results in the literature which indicate a left-libertarian bias (Exler et al., 2025; Hartmann et al., 2023; Rozado, 2024), we observe that our agents are shifted towards the (center-)right. The average distortions between ground-truth and agent position for each group (big arrows) show an overall trend towards more conservative (negative y) and no clear left-right bias. Implementation details are given in Section E.5.

**Claims and evidence**
- Dataset sizes and labels differ: Table 2, smartvote row—18 units, 48 train questions, 12 test questions, {Yes, No}; WoM row—6 units, 30 test questions, {Yes, Neutral, No}; ANES row—21 units, 67 train questions, 12 test questions, {Yes, Neutral, No}.
- smartvote scores exceed WoM and ANES scores: Table 3, Magistral 24B/SFT+GRPO row—\(70.73(2.21)\), \(53.21(3.19)\), \(45.43(1.11)\). The caption offers binary labels and training-set size as likely explanations.
- Agents show positional distortions: Figure 2, candidate-to-agent lines and group-average arrows.

### 4.2 Experimental setup

*The setup compares training methods and backbones, with one trained model per unit.*

We compare models and strong baselines to ask: (i) how performance scales with model size, and (ii) whether reasoning-pretrained backbones improve on non-reasoning peers. A backbone is the model used as the training starting point. We train one model per candidate, party, or respondent. Appendix F gives further training details.

##### Methods

*The reported configurations use survey-answer GRPO and SFT followed by GRPO.*

We report results for three configurations: GRPO on survey questions and answers, as well as SFT+GRPO.

For smartvote and ANES, Llama 3.1 70B (Grattafiori et al., 2024) generates synthetic arguments supporting both positive and negative stances for each question. These model-generated arguments supply SFT data. Section A.4 describes the prompting strategy.

##### Model backbones..

*Three open-weight backbones test model scale and prior reasoning training.*

We evaluate Llama3.1 8B (Grattafiori et al., 2024), Qwen3 8B (Yang et al., 2025), and Magistral 24B (Rastogi et al., 2025). Qwen3 and Magistral are pre-trained to reason. They allow comparison with the non-reasoning Llama3.1 at comparable scales.

All experiments involving training use 4-bit quantization, the quantization setting applied to every model.

##### Baselines

*The baselines are random answers, majority answers, in-context learning, and ORPO.*

The **random** baseline selects an answer uniformly at random. The **majority** baseline answers every question with the most frequent answer in that unit’s training set.

The **icl** baseline follows Haller et al. (2025). It prompts the model with question-answer pairs, then requests an answer to an unseen question. We supply all training questions from the test question’s topic area, or a random subset when context length is limiting.

We also compare ORPO (Hong et al., 2024), using the same hyperparameters as Stammbach et al. (2024).

### 4.3 Results and analysis

*Combined SFT and GRPO generally performs best, but performance differs across datasets and backbones.*

Table 3 reports mean macro-F1. SFT+GRPO outperforms naive and SFT baselines across datasets, except for Llama 3.1 8B on ANES. Scores peak on smartvote (\(70.73\)); ANES is hardest (\(45.43\)).

SFT+GRPO is comparable to or better than icl on ANES and smartvote, and outperforms it on WoM. Native reasoning support does not consistently improve results: Llama 3.1 8B outperforms Qwen3 8B on WoM and ANES, while Magistral 24B performs best overall.

GRPO alone underperforms SFT+GRPO. This suggests that the supervised warm start improves training dynamics.

Datasets with a Neutral class perform worse. Neutral may combine uncertainty, social desirability, and strategic non-commitment, making it hard to learn. Section 4.5 examines ideology; Section 4.6 examines Neutral. Section E.1 reports results without Neutral.

In Figure 3, error bars show **variance**, meaning variation within political groups.

Figure 3: Not all political positions are equally learnable. F1 scores reveal that while SFT+GRPO typically works best, every training method underperforms on center and right-leaning groups. Error bars show variance within groups. These disparities may stem from biases baked into Llama 3.1 8B or from inherent differences in how well various political preferences can be learned from survey data. Either way, the results demonstrate that ideology impacts the learnability of political preferences. A detailed figure with the original parties and ideology groups is presented in Appendix C.

**Claims and evidence**
- SFT+GRPO generally exceeds SFT, with a Llama ANES exception: Table 3, Llama 3.1 8B/SFT+GRPO versus SFT—smartvote \(66.88(2.18)\) versus \(63.44\pm 1\) .34); WoM \(52.53(4.05)\) versus \(48.95\pm 3\) .56); ANES \(40.66\pm 0\) .91) versus \(42.77(1.23)\). Qwen3 and Magistral show improvements in all three columns.
- smartvote peaks highest and ANES lowest: Table 3, Magistral 24B/SFT+GRPO—\(70.73(2.21)\), \(53.21(3.19)\), \(45.43(1.11)\).
- SFT+GRPO exceeds icl on WoM: Table 3, Llama—\(52.53(4.05)\) versus \(28.17\pm 0\) .02); Qwen3—\(49.38(1.93)\) versus \(26.19\pm 0\) .35); Magistral—\(53.21(3.19)\) versus \(26.19\pm 0\) .35).
- Reasoning pre-training does not consistently help: Table 3, Llama and Qwen3 SFT+GRPO rows—WoM \(52.53(4.05)\) versus \(49.38(1.93)\); ANES \(40.66\pm 0\) .91) versus \(38.44(0.40)\).
- The warm start helps relative to GRPO alone: Table 3, every backbone’s SFT+GRPO and GRPO rows.
- Ideological disparities remain: Figure 3, Left, Center, and Right groups.

### 4.4 How do agents reason?

*Agents generally produce coherent arguments, but can hallucinate or fail across topics.*

Table 1 gives an example reasoning trace. Agents generally make coherent arguments for or against a measure that imply their final answer.

We identify two failure modes. First, arguments sometimes contain illogical or hallucinated elements. Second, an agent can match an individual’s opinion on one topic but fail on another. Appendix B provides further traces and failure examples.

**Claims and evidence**
- A trace can support a changed answer: Table 1, “Before” and “After” rows; the trained model answers `A) Yes`.
- Illogical or hallucinated arguments and cross-topic failures occur: Appendix B examples.

### 4.5 Political ideology matters

*Performance differs across political groups, with particular difficulties for right-wing profiles.*

Figure 3 reports F1 by ideology for Llama 3.1 8B (SFT+GRPO). On smartvote, Left clearly outperforms Center and Right. Wahl-o-Mat and ANES show similar but weaker patterns. On Wahl-o-Mat, Left and Center perform roughly equally; ANES performance is generally lower.

For Right in smartvote and ANES, SFT also outperforms every other method on average. This further suggests difficulty with right-wing preference profiles.

These differences may reflect general-purpose LLMs’ known left-leaning tendencies (Hartmann et al., 2023; Exler et al., 2025). Alternatively, Right profiles may be intrinsically harder to learn from surveys. Larger training sets may help explain the relatively strong Right performance on WoM (Section E.4).

**Claims and evidence**
- Left exceeds Center and Right on smartvote; patterns are weaker elsewhere: Figure 3, dataset panels and ideology groups.
- SFT performs best on average for Right in smartvote and ANES: Figure 3, Right groups.
- Larger WoM training sets may aid Right performance: Section E.4.

### 4.6 Neutral reasoning poses a particular challenge

*Neutral is difficult to predict, and recoding affects the observed ideological performance gaps.*

In Figure 4, **recall**, the class-specific performance measure, is worst for Neutral for every ANES individual (top right). The **Neutral base rate**, meaning the frequency of Neutral answers, is highest for Right respondents. The regressions—analyses relating base rate to performance—in the left column show this relationship. Difficulty predicting Neutral therefore harms these respondents most.

Removing Neutral instances and recomputing F1 substantially improves performance (bottom right). It does not remove the Left–Right gap. SFT+GRPO also improves relative to icl when Neutral is removed (Section E.1).

We repeat the analysis with the alternative recoding scheme from Section 4.1 and Appendix D. Predictive performance still correlates negatively with the Neutral base rate. Center and Right perform almost identically, however. This indicates that the observed difference could partly be an ANES recoding artifact. Section E.2 gives details.

Figure 4: Learning neutral stances is hard. On ANES, there is a strong correlation between predictive performance and the neutral base rate on the test set. Individuals in the Right group, and to some extent in the Center group, tend to answer questions with Neutral more often. Both in terms of F1 (top left) and accuracy (bottom left), we observe a negative correlation significant at the \(5\%\)-level. Regression details in Section E.3. Top right: Recall by class on ANES. Averaged over all individuals, the model struggles the most with predicting the Neutral class. Bottom right: Performance in terms of F1 score on ANES before (solid) and after (shaded) removing Neutral instances. All groups improve, and the gap between Left and Center becomes narrow. However, the difference between Left and Right remains. While the large number of Neutrals likely depressed performance on the other two classes during training, this could also suggest that the performance disparity is affected by other factors, such as model bias or artifacts from recoding ANES.
All results were obtained from Llama 3.1 8B trained with SFT+GRPO.

**Claims and evidence**
- Neutral is hardest to predict: Figure 4, top-right class-recall panel.
- Performance falls as Neutral base rate rises: Figure 4, top-left F1 and bottom-left accuracy regressions; both are significant at the \(5\%\)-level, with details in Section E.3.
- Removing Neutral improves all groups but preserves the Left–Right gap: Figure 4, bottom-right panel.
- Alternative recoding preserves the negative correlation but makes Center and Right performance almost identical: Section E.2.

### 4.7 Whom do agents represent?

*Agents do not perfectly reproduce candidates’ positions and show center-right positional shifts.*

We compare agents with their corresponding people in a two-dimensional political model. We use smartvote because the principal components’ interpretation is documented (Germann et al., 2015).

Figure 2 places the included individuals in the first two principal components computed from all candidates in the 2023 Swiss national elections. We project agent positions, shown in gold, into that space.

Agents do not perfectly represent their candidates, shown as large markers. Despite documented left-libertarian biases in most LLMs (Exler et al., 2025; Hartmann et al., 2023; Rozado, 2024), we find no overall left shift. Instead, agents shift toward the center-right on the left-right \(x\)-axis and toward the center on the conservative-liberal \(y\)-axis.

Group-averaged human-to-agent distances in the center of Figure 2 support this result. Right individuals become more left-wing. Left and Center individuals become more right-wing and conservative.

The next figure examines progressive- and conservative-biased SFT data: arguments generated by prompting for those respective political perspectives.

Figure 5: Biased SFT data consistently impairs the performance of underrepresented groups. F1-scores for Llama trained with SFT+GRPO on differently biased datasets. Left: Data with a progressive bias strongly impairs the Right group, without consistently benefiting the Left group. Middle: Performance on the default dataset. Right: Data with a conservative bias decreases the performance of the Left group, while showing no consistent improvement for the Right group. These asymmetric effects suggest that ideological bias in SFT data primarily harms the underrepresented perspective rather than systematically improving performance on the overrepresented one.

**Claims and evidence**
- Agents do not perfectly reproduce candidates and show no overall left shift: Figure 2, candidate positions, agent positions, and connecting lines.
- Right moves leftward; Left and Center move rightward and conservatively: Figure 2, group-average arrows.
- Biased SFT data mainly harms underrepresented perspectives: Figure 5, progressive, default, and conservative panels.

### 4.8 Impact of SFT data

*Ideological SFT bias mainly harms and shifts underrepresented groups, especially Right candidates.*

To test SFT-data bias, we use Llama 3.3 70B to generate two additional argument sets, prompted to be progressive and conservative respectively. Section A.4 gives details.

Figure 5 reports F1 effects. Progressive bias strongly harms Right candidates, with inconsistent effects on Left candidates. Conservative bias similarly harms Left candidates without consistently benefiting Right candidates. Thus, bias mainly harms the underrepresented perspective rather than systematically helping the overrepresented one. Right candidates suffer more from adverse bias than Left candidates.

Figure 6 shows effects on agent positions in PCA space. Consistent with F1, biased data mainly shifts underrepresented groups, again affecting Right more strongly than Left.

Conservative-biased and default SFT data produce highly similar shifts. This suggests that the centerward shift in Figure 2 may mainly come from the base model’s alignment rather than SFT-data bias.

Figure 6: Biased SFT data primarily shifts the agents of underrepresented groups. Left: Candidates and their agents in the smartvote PCA space. Right: Group-mean displacement vectors of the agents relative to their respective candidates. Similar to the F1 scores presented in Fig. 5, biased data mostly affects the underrepresented group, and Right candidates are again affected more strongly than Left candidates. Notably, conservative-biased and default SFT data produce similar displacement patterns, suggesting the centerward shift in the default data may originate from the base model’s alignment rather than the SFT data.

**Claims and evidence**
- Progressive bias harms Right; conservative bias harms Left, without consistent gains for the opposite group: Figure 5, left and right panels.
- Right is more strongly harmed and shifted: Figures 5–6, Right versus Left groups.
- Conservative-biased and default positional shifts are similar: Figure 6, group-mean displacement vectors; this supports the proposed base-model explanation.

### 4.9 Performance on synthetic positions

*Flipping all answers helps Right profiles but does not eliminate the original performance disparity.*

We create a counterfactual—invert every smartvote answer—and train Llama-3.1-8B on the resulting data. Inversion roughly corresponds to a \(180^\circ\) rotation about Figure 2’s origin (Section E.6). For example, it converts right-wing positions into left-wing ones.

Figure 7 shows each politician’s F1 difference before and after inversion. Right benefits; Left and Center perform worse on average. Even after inversion, Right does not reach Left’s original F1. Left drops to roughly Right’s original level.

If the effect came purely from the model’s original political bias, flipping labels should make right-leaning candidates nearly as predictable as left-leaning ones. This does not occur. The result suggests that left-leaning preference profiles may be intrinsically easier to model. That possibility needs further investigation.

**Claims and evidence**
- Answer inversion approximately rotates positions by \(180^\circ\): Section E.6.
- Inversion helps Right but worsens Left and Center on average: Figure 7, F1-change arrows.
- Inversion does not eliminate the disparity: Figure 7, original scores and post-inversion changes; Right remains below Left’s original average.

**Claims and evidence — Section 4**
- SFT+GRPO generally performs best: Table 3, SFT+GRPO rows—smartvote \(66.88(2.18)\), \(65.11(3.33)\), \(70.73(2.21)\); WoM \(52.53(4.05)\), \(49.38(1.93)\), \(53.21(3.19)\); ANES \(40.66\pm 0\) .91), \(38.44(0.40)\), \(45.43(1.11)\). Llama’s ANES SFT score is \(42.77(1.23)\).
- Neutral and ideology affect performance: Figures 3–4.
- Positional distortions and sensitivity to biased data remain: Figures 2 and 5–7.

## 5 Discussion

*Noise, computational cost, and limited survey data constrain the method; future work should address bias and uncertainty.*

##### Limitations.

*Opinion noise, per-profile training, and scarce or mismatched surveys limit learning and comparison.*

Human opinions contain noise. The same respondent may answer differently on different days, so consistently achieving F1 close to \(1.0\) is unlikely.

We study only a small number of individuals or parties. The main reason is computational cost: each profile needs a separately trained model.

Data are another constraint. Suitable surveys are scarce. Existing surveys often have too few items for individualized training, or address generic themes rather than concrete issues and priorities. Question sets also rarely align across countries, limiting comparison.

##### Future work.

*Future work should reduce training costs, estimate reliability, and improve bias mitigation and Neutral handling.*

Political-group disparities require further bias-mitigation research. A single model conditioned on compact persona representations could replace per-agent training, reduce cost, and improve data use; see Ning et al. (2024).

Deployment will require **confidence estimates**, measures of answer reliability. These could come from an extended reasoning scheme or a dedicated **value head**, a component for estimating reliability. Purpose-built questionnaires and better uncertainty handling may also reduce difficulty with Neutral.

Figure 7: Learning Right and Center positions is harder than Left ones. Points depict F1 scores on the original data, with individuals ordered along the x-axis by their first component in the smartvote PCA space. Note that the left-most Right candidate is indeed correctly placed (see Fig. 6). Arrows show the difference in F1 scores after all answers in the train and test sets have been flipped. Left candidates are worse after flipping, both individually and on average. Right candidates improve, but still perform worse on average than Left candidates. As the disparity is not removed by converting Left to Right, this could suggest that Right positions are more difficult to learn.

**Claims and evidence**
- Flipping worsens Left and improves Right, but does not remove the disparity: Figure 7, original F1 points and change arrows.
- These results could suggest greater difficulty learning Right positions: Figure 7, remaining post-inversion disparity.

## 6 Conclusion

*Structured reasoning improves individual opinion alignment, but Neutral and ideological disparities remain.*

We study individual opinion alignment by training persona-conditioned agents directly on known opinions and treating answer generation as structured reasoning.

Across Swiss candidates (smartvote), German parties (Wahl-o-Mat), and U.S. voters (ANES), SFT+GRPO generally outperforms all baselines. Performance remains uneven. Neutral stances are difficult to predict, and accuracy varies significantly across the political spectrum.

The work provides a strong benchmark and baseline for systematic LLM opinion modeling. It is an early step toward potentially more representative, AI-driven forms of democracy.

**Claims and evidence**
- SFT+GRPO generally outperforms the baselines: Table 3, SFT+GRPO rows; the Llama ANES exception is SFT \(42.77(1.23)\) versus SFT+GRPO \(40.66\pm 0\) .91).
- Neutral and ideological disparities remain: Figures 3–4; Figure 4’s performance–Neutral correlations are significant at the \(5\%\)-level.

## Ethics statement

*The paper warns against deployment before ideological fairness problems are resolved.*

Modeling individual opinions during rising mis- and disinformation is not exclusively beneficial and may create downstream risks. Because of these risks and the technology’s inevitable development, we believe responsible research must investigate better individual alignment.

We use only anonymized survey data and public figures’ positions to address privacy concerns.

We find systematic ideological performance disparities: consistent underperformance on center and right-wing profiles, and difficulty modeling neutral positions. We commit to reporting these biases transparently. Premature deployment could disenfranchise political viewpoints.

We strongly caution against real-world applications until these fairness problems are resolved. Biased political models could undermine democratic representation. With proper bias mitigation, we believe opinion modeling could support beneficial applications such as digital democracies, where models help shape fairer and more popular policies.

**Claims and evidence**
- Center and right-wing profiles face performance disparities: Figure 3, ideology groups.
- Neutral positions are difficult to model: Figure 4, top-right recall panel and performance–Neutral regressions.

## Reproducibility statement

*Code, datasets, training details, and PCA implementation details are made available.*

We release our code and datasets:

https://github.com/ETH-DISCO/reasoning-boosts-llm-alignment

Appendix F gives training details, including hyperparameters. Section E.5 describes all PCA results.

#### Author Contributions

*The authors’ roles cover project design, technical development, advice, and supervision.*

Frédéric Berdoz conceived and designed the project, led experimental planning, and defined its research direction.
Yann Billeter contributed equally, leading technical development, including dataset collection, model training, evaluation, and analysis. Yann Vonlanthen contributed as an advisor, offering guidance on methodology and experimental design. Roger Wattenhofer supervised the project as principal investigator.

#### Acknowledgments

*The authors thank Prof. Dr. Hans Gersbach for his insights.*

We thank Prof. Dr. Hans Gersbach for his valuable insights during the project.
