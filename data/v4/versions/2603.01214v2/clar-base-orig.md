# Reasoning Boosts Opinion Alignment in LLMs

Frédéric Berdoz

  
Yann Billeter

  
Yann Vonlanthen

  
Roger Wattenhofer

Affiliation: ETH Zurich

Affiliation: Switzerland

Affiliation: {fberdoz, ybilleter, yvonlanthen, wattenhofer}@ethz.ch

###### Abstract

Opinion modeling aims to capture the political preferences of individuals or groups. It could support applications such as digital democracies, where models help shape fairer and more popular policies. Large language models (LLMs), which generate text from textual inputs, are natural candidates for this task. They are versatile, generalize well, and have succeeded across diverse text-to-text applications. However, their statistical nature and limited causal understanding mean they tend to produce biased opinions when prompted naively. A prompt is the input used to guide a model’s response.

We study whether reasoning can improve opinion alignment: making a model’s answers match the preferences of a target individual or group. Recent advances in mathematical reasoning through reinforcement learning (RL), which trains models using rewards, motivate our approach. We train models to give answers consistent with a preference profile through structured reasoning. We evaluate the approach on three datasets covering U.S., European, and Swiss politics. Reasoning improves opinion modeling and is competitive with strong baselines, meaning comparison methods, but does not fully remove bias (Table 3 and Sections 4.5–4.9). Additional mechanisms are therefore needed to build faithful political digital twins: models that represent individuals’ political preferences. We release our method and datasets[^1] as a baseline for future research on LLM opinion alignment.

[^1]: Code and data: https://github.com/ETH-DISCO/reasoning-boosts-llm-alignment

## 1 Introduction

Could artificial intelligence (AI) enable a new kind of democracy, where digital twins vote on our behalf and faithfully reflect our opinions on every issue? Accurate simulations of political behavior could help explain election outcomes and improve policy and democratic processes (Li et al., 2024). However, capturing how diverse individuals reason about political issues remains difficult. LLMs can generate sophisticated political discourse, but they often fail to reflect the diversity of human political viewpoints (Santurkar et al., 2023; Qu and Wang, 2024; Yu et al., 2025; Qi et al., 2024). This raises a fundamental question: how can agents reason about politics while faithfully representing human political diversity?

Most current approaches prompt LLMs with demographic information and political affiliations. They therefore rely on learned correlations between demographic traits and opinions. These methods have become increasingly sophisticated. Yet they consistently fail to capture real opinion distributions and give unstable, inconsistent responses across prompts and demographic groups (Santurkar et al., 2023; Ball et al., 2025). We instead seek preference consistency by using an individual’s known opinions directly, rather than using demographics as a substitute.

These demographic methods are not well suited to simulating individual political preferences. Richer sources, such as interview transcripts, can support strong models of individual personas, or representations of particular people (Park et al., 2024). However, interviews are prohibitively costly to collect at scale. We propose political surveys as an alternative. Large surveys, such as the American National Election Studies (ANES), and Voting Advice Applications (VAAs) capture people’s positions on political issues. VAAs are online platforms that match voters with parties or candidates. Surveys lack the narrative richness and textual information of interviews, but provide structured political opinions across populations. They are widely used to evaluate bias (Santurkar et al., 2023; Argyle et al., 2023; Haller et al., 2025) and simulate groups (Argyle et al., 2023; Cao et al., 2025). They have not yet been widely adopted to train individual-level political agents.

RL has recently produced major gains in reasoning. Models such as o3/o4-mini (OpenAI, 2025) and DeepSeek-R1 (Guo et al., 2025) have set new marks on various benchmarks, or evaluation tasks. Motivated by these results, we use RL, specifically Group-relative policy optimization (GRPO; Shao et al., 2024), to train agents on survey data. We treat opinion formation as a reasoning problem. The agent writes a short rationale and a final choice. It receives a reward when that choice matches the respondent’s answer. This trains it to reason toward survey-consistent positions on unseen questions.

Our contributions are:

- **Reinforcement learning for reasoning-based opinion alignment.**  
  We introduce a method that uses GRPO to encourage explicit reasoning and improve opinion alignment from survey responses.

- **Benchmarking on real-world political data.**  
  We evaluate the method on three datasets of authentic political opinions: German parties, U.S. voters, and Swiss candidates. We release them as a benchmark for future research. Our results show that reasoning consistently improves opinion alignment (Table 3).

- **Analyzing ideological effects.**  
  We investigate how political ideology, meaning a person’s political outlook, affects performance. We invert the survey answers of individuals with known positions. This reveals systematic differences in alignment quality across the political spectrum (Section 4.9 and Figure 7).

**Figure 1: GRPO for opinion alignment.** We use public opinion surveys to align LLMs with individual preference profiles. First, we fine-tune the LLM—continue training it—with (synthetic) statements and ground-truth answers, meaning the recorded target answers. This teaches it to follow the reasoning template. After fine-tuning, the model uses the correct format but is not fully aligned with the target opinions. We then use GRPO with a reward model that rewards correct formatting and correct answers to further improve reasoning.

## 2 Related Work

Political opinion modeling requires LLMs to adopt particular political personas and answer from those viewpoints. Prior work has studied demographic personas, which represent groups sharing traits such as occupation or ethnicity; character personas, which represent well-known individuals; and individualized personas, which are digital profiles based on personal data (Chen et al., 2024; Tseng et al., 2024). Individualized personas are most relevant to our focus on individuals. However, most research uses demographic approaches. Only a handful of studies use individualized methods.

##### Modeling political opinions with demographic personas.

Political opinion modeling has mainly used demographic personas through prompting. Simple prompts specify a political affiliation, such as “You are a democrat. What is your opinion on…” (Santurkar et al., 2023). More complex prompts add age, education, location, and other demographic details (Argyle et al., 2023; Santurkar et al., 2023; Hwang et al., 2023; Sun et al., 2024; Wang et al., 2025). Recent work supplies richer context. For example, demonstrations from belief networks give models related political positions to guide their answers (Chuang et al., 2024). Gudiño et al. (2024) combine demographics with demonstrations. Stammbach et al. (2024) combine prompting with odds ratio preference optimization (ORPO), a training method used to align outputs with target positions (Hong et al., 2024). Despite this variety, demographic approaches have several limitations.

##### Limitations of demographic approaches.

Santurkar et al. (2023) identify three problems with prompt-based demographic simulation:

- **Representativeness:** default model opinion distributions do not reflect actual population distributions.
- **Steerability:** demographic prompts do not provide enough control over model outputs.
- **Consistency:** political biases vary unpredictably across topics and contexts.

These problems produce systematic biases. Accuracy varies unpredictably across demographic groups (Argyle et al., 2023; Santurkar et al., 2023). Models can approximate overall opinion distributions, but fail to capture natural variation within demographic subgroups. They also remain sensitive to prompt variations (Sun et al., 2024; Ball et al., 2025).

##### Moving beyond demographics.

Few studies model individual opinion profiles, in contrast to the extensive work on demographic prompting. Park et al. (2024) use interview transcripts to achieve strong accuracy on public opinion surveys, outperforming demographic baselines. Transcript availability limits this approach. Haller et al. (2025) propose in-context learning: the model receives a respondent’s question-answer pairs in its prompt and then predicts that respondent’s answers to test questions. Their focus, however, is identifying bias rather than modeling opinions.

**Table 1: Answers obtained from Llama 3.1 8B trained on an individual in the Right group in ANES.** Here, 8B denotes 8 billion model parameters; parameters are the learned numerical values of the model. Right is one of the broad political groups defined in Section 4.1. Before training, Llama disagreed; after training, it agreed. The reasoning trace—the written justification preceding the answer—still appeals to similar arguments (“equal opportunity”) but frames them differently. SFT means supervised fine-tuning, or training on supplied target outputs. SFT+GRPO applies SFT before GRPO. Additional examples appear in Appendix B.

| Part | Text |
|---|---|
| Question: | Do you agree that it is not really that big a problem if some people have more of a chance in life than others? |
| Before (prompting on base model) | No. I don’t think that’s right. Everyone deserves an equal shot at succeeding in life, no matter where they’re from or who their parents are. If some people have more opportunities and resources, it’s not fair to others. We should be working to level the playing field and give everyone a fair chance to succeed. |
| After (SFT+GRPO) | `<reasoning>`<br>Some inequality in wealth and opportunity is inevitable and indeed necessary to provide incentives for hard work and innovation – the issue is not eliminating all differences but rather ensuring that everyone has access to basic necessities and opportunities to improve themselves – other countries with significant equality have still struggled with poverty and stagnation.<br>`</reasoning>`<br><br>`<answer>A) Yes</answer>` |

##### Training LLMs to reason.

Asking LLMs to produce reasoning steps before answering significantly increases accuracy. Examples include Chain-of-Thought prompting, which asks for intermediate reasoning steps (Wei et al., 2022), and the simple instruction “Let’s think step by step” (Kojima et al., 2022). Recent work has moved from prompting to training models to generate high-quality reasoning traces using search algorithms or RL (Xu et al., 2025).

GRPO is one such method (Shao et al., 2024). It is a variant of Proximal policy optimization (PPO), an RL training method (Schulman et al., 2017). Instead of learning a value function to estimate advantages, GRPO normalizes rewards within a group. A value function estimates expected rewards; an advantage measures how favorable an output is relative to a reference value. For each prompt, GRPO samples a group of outputs. It subtracts the group’s mean reward from each reward, then divides by the group’s standard deviation, which measures the spread of those rewards. GRPO has been particularly effective for mathematical reasoning and has achieved state-of-the-art benchmark results (Guo et al., 2025).

##### Reasoning for opinion alignment.

Yu et al. (2025) propose a persona-based prompting framework inspired by chain-of-thought reasoning. It combines demographic, ideological, and temporal factors calibrated using ANES. The authors simulate individual respondents’ voting behavior and sum those responses to make aggregate predictions.

Unlike our method, their method does not use RL. We also learn general preference profiles, whereas Yu et al. (2025) target voting predictions. Our method could theoretically address the same problem. However, modeling a representative sample is currently prohibitively expensive computationally.

## 3 Method

We study individual-level opinion alignment using survey data. A policy question might ask, “Should the government increase funding for renewable energy?” Its possible stances, or answer positions, might be

\[
Y=\{\texttt{No},\texttt{Neutral},\texttt{Yes}\},
\]

where \(Y\) is the set of allowed answers. For any policy question \(q\) and persona \(p\), our goal is to learn \(p\)’s stance on \(q\). A persona is a survey respondent or political party.

### 3.1 Political Reasoning with GRPO

We use GRPO (Shao et al., 2024) to train agents to write reasoning before answering. Every response must follow this fixed schema:

```text
<reasoning>[justification text]</reasoning><answer>[stance]</answer>
```

\[
\tag{1}
\]

The four tags mark the beginning and end of the reasoning and answer.

Our training data are survey responses:

\[
\mathcal{D}=\{(p_j,q_i,y^*_{ij})\}.
\]

Here, \(\mathcal{D}\) is the dataset, \(j\) indexes personas, and \(i\) indexes questions. The value \(y^*_{ij}\) is persona \(p_j\)’s observed stance on question \(q_i\). These observed answers serve as ground truth. The dataset contains final stances but no reasoning traces. The model must therefore learn reasoning that both follows Equation (1) and leads to the recorded answer.

We evaluate each generated response with three reward components.

**Format Reward (\(R_{\text{format}}\)).**  
The model receives a score of 1 for each correctly placed tag in Equation (1). There are four tags, so the maximum format reward is 4.

**Length Reward (\(R_{\text{length}}\)).**  
Let \(x_i\) be a generated reasoning trace. Let \(L\) be its length in tokens, the units of text processed by the model, and let \(L^*\) be the desired length. We use the symmetric penalty

\[
R_{\text{length}}=-|L-L^*|,
\]

where the vertical bars denote absolute value. The maximum reward is 0, achieved only when \(L=L^*\).

**Correctness Reward (\(R_{\text{correct}}\)).**  
The main reward checks whether the predicted answer matches the survey response:

\[
R_{\text{correct}}=\mathds{1}[y_i=y_i^*].
\]

Here, \(y_i\) is the model’s predicted answer, \(y_i^*\) is the ground-truth survey answer, and \(\mathds{1}\) is an indicator: it equals 1 when the condition in brackets holds and 0 otherwise.

For a generated response \(r_i\), containing its reasoning and predicted answer, the total reward is

\[
R(r_i,y_i^*)=
\alpha_1R_{\text{format}}+
\alpha_2R_{\text{length}}+
\alpha_3R_{\text{correct}},
\]

with reward weights

\[
\alpha_1=0.25,\qquad \alpha_2=0.01,\qquad \alpha_3=1.
\]

We do not provide an explicit persona representation. The system prompt—the instruction specifying the model’s task—contains only a country label. Alignment with an individual occurs solely through training the model to answer that individual’s questions correctly.

##### Optional Initialization via Supervised Fine-tuning.

Before GRPO, we may initialize the model using SFT to accelerate convergence and improve its initial reasoning quality. Following Shao et al. (2024), we construct chain-of-thought demonstrations in the exact format of Equation (1).

This SFT stage provides a warm start, meaning an initial model already trained for part of the task. It serves two purposes:

1. It teaches the required output structure, reducing the burden on \(R_{\text{format}}\) during GRPO.
2. It provides a reasonable starting point for political reasoning.

**Table 2: Dataset overview.** Train/test counts are questions per unit: candidate, party, or respondent. Training questions are used to train the model; test questions are held out for evaluation. CH denotes Switzerland, DE Germany, and US the United States. smartvote is the Swiss candidate dataset. WoM abbreviates Wahl-O-Mat, the German party dataset. EU&I is the questionnaire used to test German-party models. “Q” means questions; “topic-strat.” means topic-stratified, with questions held out by topic.

| Setting | Unit | #Units | Labels | Train Q | Test Q |
|---|---|---|---|---|---|
| smartvote (CH) | Candidate | 18 | {Yes, No} | 48 | 12 (topic-strat.) |
| WoM (DE) | Party | 6 | {Yes, Neutral, No} | see Table 5 | 30 (EU&I) |
| ANES 2020 (US) | Respondent | 21 | {Yes, Neutral, No} | 67 | 12 (random) |

## 4 Experiments

### 4.1 Datasets

We use three datasets from different countries and political systems. The following paragraphs describe their construction. We release all data for reproducibility and future research. Table 2 summarizes the datasets.

**Table 3: Reasoning requires large datasets to be effective.** Values are mean macro-F1 (%) with standard deviations over 8 stochastic runs, meaning runs with random output sampling, at temperature \(T=1.0\). Temperature \(T\) is the text-generation sampling setting. F1 combines precision and recall through their harmonic mean. For a label, precision is the fraction of predictions of that label that are correct; recall is the fraction of actual instances of that label predicted correctly. Macro-F1 averages F1 across labels. For each run, we compute macro-F1 for each unit and then average across units. Standard deviations describe variation across runs. We report accuracy—the fraction of correct answers—and statistical significance in Section E.1. Scores on smartvote are considerably better than on ANES and WoM. This is likely because smartvote is binary and has no Neutral option (Section 4.6). WoM performs better than ANES, likely because it has substantially more training samples (Section E.4). The method label `icl` denotes the in-context learning baseline.

| Base model | Method | smartvote | WoM | ANES |
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

For training, we use the official Wahl-O-Mat (WoM) dataset, version 26 March 2025. It combines party positions from federal, state, and European elections during 2021–2025. We focus on six major parties, identified as CDU/CSU, SPD, Grüne, FDP, Die Linke, and AfD. Each item records an agree/neutral/disagree answer and an explanatory comment. We map these answers to {Yes, Neutral, No}.

For testing, we use EU&I (2024) (Chalkidis, 2024), a questionnaire outside the training source. It contains 30 items, which we recode to the same three labels. Appendix A gives the training counts for each party.

##### American National Election Studies.

From the ANES 2020 Time Series (American National Election Studies, 2021), we extract 79 policy items. We exclude demographics, knowledge questions, candidate evaluations, and thermometers, which are rating items.

The response formats differ across items. We therefore recode each item to {Yes, Neutral, No} using two schemes described in Appendix D:

- A **conservative mapping** treats only clearly positive or negative options as agreement or disagreement.
- An **aggressive mapping** collapses most positive responses to Yes.

Unless stated otherwise, results use the conservative scheme. Section E.2 tests robustness to recoding, meaning whether the findings persist under the other scheme.

We randomly split the items into 67 training questions and 12 test questions. To study ideology, we sample respondents according to their answers to ANES’ 7-point liberal–conservative self-placement question. We include 3 respondents for each answer option, giving 21 respondents.

##### Swiss candidates.

We use the English questionnaire for the 2023 Swiss national elections. It contains \(75\) items, including \(60\) policy questions across 12 topics. Responses use a four-point Likert scale, an ordered set of response options. We collapse them to {Yes, No}: Yes/Rather yes becomes Yes, and Rather no/No becomes No.

We sample three candidates from each of the six parties with at least two National Council seats, giving 18 candidates. Appendix A provides further information about these parties. We split the questions by topic: one question from each topic is held out for testing, giving 12 test questions. The remaining 48 questions are used for training.

##### Political Spectrum Groups.

For clarity and comparison across political settings, we assign parties in smartvote and Wahl-o-Mat, and ideologies in ANES, to three broad groups: Left, Center, and Right. Appendix A details the assignments.

**Figure 2: Agents are more centrist and conservative.** The figure shows the first two dimensions from principal component analysis (PCA) of all candidates standing in the 2023 Swiss national elections (smartvote). PCA represents the response data along component directions; here the first two provide a two-dimensional political space. Small dots show all candidates. The horizontal \(x\)-axis represents the left-right spectrum, and the vertical \(y\)-axis the conservative-liberal spectrum. Big dots show candidates included in our smartvote dataset. Gold markers show their agents. Black lines connect each candidate to its agent. Unlike work reporting a left-libertarian bias (Exler et al., 2025; Hartmann et al., 2023; Rozado, 2024), we observe agents shifted toward the (center-)right. Big arrows show each group’s average displacement between ground-truth and agent positions. They indicate an overall shift toward conservatism, corresponding to negative \(y\), and no clear left-right bias. Section E.5 gives implementation details.

### 4.2 Experimental setup

We evaluate our approach across models and against strong baselines. The design addresses two questions:

1. How does performance scale with model size?
2. Do models previously trained to reason provide an advantage over models without that training?

We train one model per unit: candidate, party, or respondent. Appendix F gives further training details.

##### Methods

We report results for three configurations: GRPO on survey questions and answers, as well as SFT+GRPO. For smartvote and ANES, we use Llama 3.1 70B (Grattafiori et al., 2024) to generate synthetic arguments supporting both positive and negative stances on each policy question. These model-generated arguments are used for SFT. Section A.4 details the prompts used to generate them.

##### Model backbones..

A backbone is the base model on which training is performed. We evaluate three open-weight backbones, meaning models with available learned weights:

- Llama3.1 8B (Grattafiori et al., 2024);
- Qwen3 8B (Yang et al., 2025);
- Magistral 24B (Rastogi et al., 2025).

The comparison examines model scale and prior reasoning training. Qwen3 and Magistral are pre-trained to reason. This allows comparison with Llama3.1, their non-reasoning counterpart, at comparable scales.

All experiments involving training use 4-bit quantization: representing model weights at reduced numerical precision using 4 bits.

##### Baselines

We use two naive baselines:

- **random** selects an answer uniformly at random, giving each answer option equal probability.
- **majority** answers every question with the answer option chosen most often in that unit’s training set.

We also use an in-context learning baseline, **icl**, following Haller et al. (2025). The model receives question-answer pairs in its prompt and then answers an unseen question. We give it all training questions from the same topic area as the test question. When the context length—the amount of text the model can receive—does not permit this, we use a random subset.

Finally, we compare with ORPO (Hong et al., 2024), using the same hyperparameters as Stammbach et al. (2024). Hyperparameters are settings chosen for the training procedure.

### 4.3 Results and analysis

Table 3 reports average macro-F1 for all datasets. SFT+GRPO outperforms the naive and SFT baselines across datasets, except for Llama 3.1 8B on ANES. In that comparison, SFT obtains \(42.77(1.23)\), while SFT+GRPO obtains \(40.66\pm 0\) .91) (Table 3, Llama 3.1 8B, ANES column).

Performance varies by dataset. Scores peak on smartvote at \(70.73\), while ANES is hardest at \(45.43\) (Table 3, Magistral 24B, SFT+GRPO row). SFT+GRPO performs comparably to or better than the icl baselines on ANES and smartvote, and outperforms them on WoM (Table 3, icl and SFT+GRPO rows for each model).

Native reasoning support does not consistently improve performance. Llama 3.1 8B outperforms Qwen3 8B on WoM and ANES, while Magistral 24B performs best overall (Table 3, SFT+GRPO rows). GRPO alone underperforms SFT+GRPO for every model-dataset combination in Table 3. This suggests that supervised initialization improves training dynamics.

Performance is lower on datasets with a Neutral class (Table 3, WoM and ANES versus smartvote). Neutral likely combines several behaviors: uncertainty, socially desirable responding, and strategic non-commitment. This makes it difficult to learn. Section 4.5 examines ideology, and Section 4.6 examines Neutral. Section E.1 reports results without Neutral.

**Figure 3: Not all political positions are equally learnable.** F1 scores show that SFT+GRPO typically works best, but every training method underperforms on center and right-leaning groups. Error bars show variation within groups. These differences may come from biases already present in Llama 3.1 8B or from inherent differences in how well survey data allow different political preferences to be learned. Either way, the results show that ideology affects the learnability of political preferences. Appendix C provides a more detailed figure with the original parties and ideology groups.

### 4.4 How do agents reason?

Table 1 gives an example of the reasoning traces. In general, agents produce coherent arguments for or against a proposed measure, and those arguments imply the final answer.

We identify two failure modes. First, arguments sometimes contain illogical or hallucinated elements, meaning generated content not grounded in the available information. Second, an agent may argue consistently with an individual’s opinion on one topic but fail on another. Appendix B provides additional traces and failure examples.

### 4.5 Political ideology matters

Figure 3 reports F1 by ideology group for Llama 3.1 8B (SFT+GRPO). On smartvote, the Left group clearly outperforms the Center and Right groups. Wahl-o-Mat and ANES show similar but weaker patterns. On Wahl-o-Mat, Left and Center perform roughly equally well. Performance is generally lower on ANES.

For the Right group in smartvote and ANES, SFT also outperforms every other method on average (Figure 3). This further suggests difficulties with right-wing preference profiles. These differences may reflect known left-leaning tendencies in general-purpose LLMs (Hartmann et al., 2023; Exler et al., 2025). Alternatively, Right profiles may be intrinsically harder to learn from survey responses. Larger training sets may help explain the relatively strong Right performance on WoM (Section E.4).

### 4.6 Neutral reasoning poses a particular challenge

On ANES, recall is worst for Neutral for all individuals (Figure 4, top right). The regressions in Figure 4’s left column also show that Right individuals answer Neutral most often. Their **Neutral base rate**, the fraction of their test answers labeled Neutral, is higher. Their performance therefore suffers most from the difficulty of predicting Neutral.

Removing all Neutral instances and recomputing F1 considerably improves performance. It does not eliminate the Left–Right gap (Figure 4, bottom right). SFT+GRPO also improves relative to the icl baselines when Neutral is removed (Section E.1).

We repeat the analysis using the alternative recoding scheme introduced in Section 4.1 and detailed in Appendix D. We again observe a negative correlation between predictive performance and Neutral base rate. However, Center and Right performance becomes almost identical. This indicates that the difference could partly be an artifact of how ANES responses are recoded. Section E.2 gives details.

**Figure 4: Learning neutral stances is hard.** On ANES, predictive performance is strongly correlated with the Neutral base rate in the test set. Right individuals, and to some extent Center individuals, answer Neutral more often. Both F1 (top left) and accuracy (bottom left) have negative correlations significant at the \(5\%\)-level. Section E.3 gives regression details; regression here describes the relationship between Neutral base rate and performance. Top right: recall by class on ANES. Averaged across individuals, the model struggles most with Neutral. Bottom right: ANES F1 before removing Neutral instances (solid) and after removing them (shaded). All groups improve, and the Left–Center gap narrows. The Left–Right difference remains. The large number of Neutral answers likely depressed performance on the other two classes during training. The remaining gap could also reflect other factors, including model bias or ANES recoding artifacts. All results use Llama 3.1 8B trained with SFT+GRPO.

### 4.7 Whom do agents represent?

To understand what the agents represent politically, we compare their positions with the ground-truth positions of their corresponding humans in a two-dimensional political model. We use smartvote because the interpretation of its principal components is documented (Germann et al., 2015).

Figure 2 shows the sampled individuals in the space defined by the first two principal components from all candidates in the 2023 Swiss national elections. We project the agents’ responses into the same space; gold markers show the resulting agent positions.

Not every agent perfectly represents its candidate, shown by a big marker. Despite documented left-libertarian biases in most LLMs (Exler et al., 2025; Hartmann et al., 2023; Rozado, 2024), we do not observe an overall left shift. Instead, agents shift toward the center-right on the left-right \(x\)-axis and toward the center on the conservative-liberal \(y\)-axis (Figure 2).

The group-averaged human-to-agent distances in the center of Figure 2 support this pattern. Right individuals become more left-wing. Left and Center individuals become more right-wing and conservative.

**Figure 5: Biased SFT data consistently impairs the performance of underrepresented groups.** The figure shows F1 scores for Llama trained with SFT+GRPO on datasets with different ideological biases. Left: progressive-biased data strongly impairs the Right group without consistently benefiting the Left group. Middle: performance with the default dataset. Right: conservative-biased data lowers Left performance without consistently improving Right performance. These asymmetric effects suggest that ideological bias in SFT data mainly harms the underrepresented perspective rather than systematically improving the overrepresented one. Here, underrepresented and overrepresented refer to the perspectives disfavored and favored by the biased arguments.

### 4.8 Impact of SFT data

To study bias in SFT data, we use Llama 3.3 70B to generate two additional argument sets. One prompt asks for progressive arguments; the other asks for conservative arguments. Section A.4 provides details.

Figure 5 shows the effects on F1. Progressive bias strongly impairs Right candidates and has inconsistent effects on Left candidates. Conservative bias similarly impairs Left candidates without consistently benefiting Right candidates. This suggests that ideological bias mainly harms underrepresented perspectives rather than systematically improving overrepresented ones. Adverse bias harms Right candidates more severely than Left candidates (Figure 5).

Figure 6 shows how the same biases affect positions in PCA space. Consistent with the F1 results, biased data mainly shifts underrepresented groups. Right candidates are again affected more strongly than Left candidates. The positional shifts under conservative-biased and default SFT data are highly similar. This suggests that the centerward shift in Figure 2 may mainly reflect the base model’s alignment rather than bias in the SFT data.

**Figure 6: Biased SFT data primarily shifts the agents of underrepresented groups.** Left: candidates and agents in the smartvote PCA space. Right: group-mean displacement vectors, meaning average changes in position from candidates to their agents. As with the F1 scores in Figure 5, biased data mainly affects the underrepresented group. Right candidates are again affected more strongly than Left candidates. Conservative-biased and default SFT data produce similar displacement patterns. This suggests that the default data’s centerward shift may originate in the base model’s alignment rather than in the SFT data.

### 4.9 Performance on synthetic positions

To test how individual preferences affect agent performance, we construct a counterfactual dataset: an alternative in which every smartvote answer is inverted. We train Llama-3.1-8B on these inverted answers. Inversion roughly corresponds to a \(180^{\circ}\) rotation around the origin in Figure 2 (Section E.6). For example, it converts right-wing positions into left-wing ones.

For each politician, we compute the F1 difference before and after inversion (Figure 7). Right members benefit from inversion. Left and Center members perform worse on average. However, after inversion, the Right still does not reach the Left’s original F1. The Left falls to roughly the Right’s original level.

If the effect came solely from the model’s original political bias, flipping the labels should have made right-leaning candidates nearly as easy to predict as left-leaning candidates. This does not happen (Figure 7). The result suggests that left-leaning preference profiles may be intrinsically easier to model. This possibility requires further investigation.

## 5 Discussion

##### Limitations.

Learning human opinions is inherently limited by noise. Even the same respondent may answer a question differently on different days. Consistently achieving an F1 score near \(1.0\) is therefore unlikely.

Our analyses cover only a small number of individuals or parties. The main reason is computational cost: the method trains one model per profile. Data are another constraint. Suitable surveys are scarce. Existing surveys often contain too few items to train individual agents, or address generic themes rather than concrete issues and priorities. Question sets also rarely align across countries, limiting comparisons.

##### Future work.

The performance differences between political groups require further investigation into bias mitigation, meaning ways to reduce those differences. Future work should aim to replace per-agent training with one model conditioned on compact persona representations (for example, Ning et al., 2024). This could reduce computational cost and improve data use.

Real-world deployment will require confidence estimates to indicate how reliable an answer is. These could come from extending the reasoning scheme or adding a dedicated value head, a model component that produces an estimate. Purpose-built questionnaires and better handling of uncertainty may also reduce the difficulties with Neutral.

**Figure 7: Learning Right and Center positions is harder than Left ones.** Points show F1 on the original data. Individuals are ordered along the \(x\)-axis by their first component in smartvote PCA space. The left-most Right candidate is correctly placed (Figure 6). Arrows show the F1 change after every answer in both the training and test sets is flipped. Left candidates perform worse after flipping, individually and on average. Right candidates improve but still perform worse on average than Left candidates. Because converting Left to Right does not remove the disparity, Right positions may be more difficult to learn.

## 6 Conclusion

We study individual opinion alignment by training persona-conditioned agents directly on known opinions and treating answer generation as structured reasoning. Across Swiss candidates (smartvote), German parties (Wahl-o-Mat), and U.S. voters (ANES), SFT+GRPO generally outperforms all baselines (Table 3).

Performance remains uneven. Neutral stances are difficult to predict (Section 4.6 and Figure 4), and accuracy varies significantly across the political spectrum (Sections 4.5–4.9). Our work provides a strong benchmark and baseline for systematic opinion modeling with LLMs. It is an early step toward potentially more representative, AI-driven forms of democracy.

## Ethics statement

Modeling individual opinions is not exclusively beneficial, especially during a period of rising misinformation and disinformation. It may create downstream risks. Precisely because of these risks and the technology’s inevitable development, we believe responsible research is essential to understand how individual opinion alignment can improve.

To address privacy concerns, we use only anonymized survey data and public figures’ positions. Our research finds systematic performance disparities across ideological groups. Center and right-wing profiles consistently underperform, and neutral positions are difficult to model (Sections 4.5–4.9). We commit to reporting these biases transparently. They are significant system risks: premature deployment could disenfranchise particular political viewpoints.

We strongly caution against real-world applications until these fairness issues are resolved. Biased political models risk undermining democratic representation. With proper bias mitigation, however, we are convinced that opinion modeling can ultimately support socially beneficial applications such as digital democracies, where models help shape fairer and more popular policies.

## Reproducibility statement

We release our code and datasets to support reproducibility:

https://github.com/ETH-DISCO/reasoning-boosts-llm-alignment

Appendix F provides training details, including hyperparameters. Section E.5 describes the methods for all PCA results.

#### Author Contributions

Frédéric Berdoz conceived and designed the project, led experimental planning, and defined its research direction.
Yann Billeter contributed equally, leading technical development, including dataset collection, model training, evaluation, and analysis. Yann Vonlanthen contributed as an advisor, offering guidance on methodology and experimental design. Roger Wattenhofer supervised the project as principal investigator.

#### Acknowledgments

We thank Prof. Dr. Hans Gersbach for his valuable insights during the project.
