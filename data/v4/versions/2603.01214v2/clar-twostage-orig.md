# Reasoning Boosts Opinion Alignment in LLMs

Frédéric Berdoz

  
Yann Billeter

  
Yann Vonlanthen

  
Roger Wattenhofer

Affiliation: ETH Zurich

Affiliation: Switzerland

Affiliation: {fberdoz, ybilleter, yvonlanthen, wattenhofer}@ethz.ch

###### Abstract

Opinion modeling aims to capture individual or group political preferences. It could support digital democracies, in which digital representatives help shape fairer and more popular policies. Large language models (LLMs) are natural candidates because they can perform many text-to-text tasks and generalize to new inputs. However, their statistical nature and limited understanding of causal relationships mean that simple prompts tend to produce biased opinions.

We study whether reasoning can improve opinion alignment: matching a model’s answers to a person’s or group’s political positions. Motivated by recent advances in mathematical reasoning through reinforcement learning (RL), which trains models using rewards for their outputs, we train models to give answers consistent with a political profile through structured reasoning.

We evaluate the approach on three datasets covering United States (U.S.), European, and Swiss politics. Reasoning improves opinion modeling and is competitive with strong baselines, as shown in Table 3. It does not fully remove bias, as the ideological analyses in Sections 4.5–4.9 show. Additional mechanisms are therefore needed to build faithful political digital twins: models that represent particular people’s political preferences. We release our method and datasets[^1] as a baseline for future research on LLM opinion alignment.

[^1]: Code and data: https://github.com/ETH-DISCO/reasoning-boosts-llm-alignment

## 1 Introduction

Could artificial intelligence (AI) create a new kind of democracy in which digital twins vote on our behalf and faithfully reflect our opinions on every issue?

Accurate simulations of political behavior could help explain election outcomes and improve policies and democratic processes (Li et al., 2024). Yet capturing how different individuals reason about political issues remains difficult. LLMs can generate sophisticated political discussion, but they often fail to reflect the diversity of human political viewpoints (Santurkar et al., 2023; Qu and Wang, 2024; Yu et al., 2025; Qi et al., 2024). This raises a basic question: how can agents reason about politics while faithfully representing that diversity?

Current approaches mainly prompt LLMs with demographic information and political affiliations. A prompt is the text supplied to guide a model’s response. These approaches rely on learned correlations between demographic traits and opinions. Even increasingly sophisticated prompts fail to capture real opinion distributions. Responses also remain unstable across prompts and demographic groups (Santurkar et al., 2023; Ball et al., 2025). We instead seek a preference-consistent approach: one that directly uses a person’s known opinions rather than demographic traits as substitutes for those opinions.

The methods described above do not lend themselves to simulating individual political preferences. Richer sources, such as interview transcripts, can support strong individual-persona modeling (Park et al., 2024). A persona is the person or group viewpoint that a model is meant to represent. However, collecting interviews is prohibitively costly at scale.

We propose political survey data as an alternative. Large surveys, such as the American National Election Studies (ANES), and Voting Advice Applications (VAAs)—online platforms that match voters with parties or candidates—record people’s positions on political issues. Surveys lack the narrative richness and textual information of interviews. They nevertheless provide structured political opinions across populations. Surveys are already common in bias evaluation (Santurkar et al., 2023; Argyle et al., 2023; Haller et al., 2025) and group-level simulation (Argyle et al., 2023; Cao et al., 2025). They have not yet been widely adopted for training individual-level political agents.

RL has recently produced major reasoning gains. Models such as o3/o4-mini (OpenAI, 2025) and DeepSeek-R1 (Guo et al., 2025) have set new marks on several benchmarks, or standardized evaluation tasks. Motivated by these results, we use Group-relative policy optimization (GRPO) (Shao et al., 2024), an RL method, to train agents on survey data. We treat opinion formation as a reasoning task. The agent writes a short justification and a final choice. It receives a reward when that choice matches the respondent’s answer. The aim is to teach it to reason toward survey-consistent positions on questions not used in training.

Our contributions are:

- **Reinforcement learning for reasoning-based opinion alignment.**  
  We introduce a method that uses GRPO to encourage explicit reasoning and improve alignment with survey responses.

- **Benchmarking on real-world political data.**  
  We evaluate authentic opinions from German parties, U.S. voters, and Swiss candidates. We release these datasets as a benchmark for future research. Our results show that reasoning consistently improves opinion alignment; Table 3 reports the comparisons, and Section 4.3 discusses the exception for Llama 3.1 8B on ANES.

- **Analyzing ideological effects.**  
  We examine how political ideology affects performance by reversing the survey answers of individuals with known positions. Section 4.9 and Figure 7 show systematic differences in alignment quality across the political spectrum.

**Figure 1: GRPO for opinion alignment.** We use public opinion surveys to align LLMs with individual preference profiles. First, we fine-tune the LLM—that is, further train an existing model—with (synthetic) statements and ground-truth answers. Synthetic statements are model-generated; ground-truth answers are the recorded survey answers used as targets. This stage teaches the reasoning template. After fine-tuning, the model answers in the correct format but is not fully aligned with opinions. We then use GRPO with a reward model that rewards correct formatting and correct answers to further improve reasoning.

## 2 Related Work

Political opinion modeling requires an LLM to adopt a political persona and answer from that viewpoint. Prior work distinguishes demographic personas, which represent groups sharing traits such as occupation or ethnicity; character personas, which represent well-known individuals; and individualized personas, which are digital profiles based on personal data (Chen et al., 2024; Tseng et al., 2024). Individualized personas are most relevant to our goal. However, most of the literature uses demographic personas, with only a handful of individualized studies.

##### Modeling political opinions with demographic personas.

Political opinion modeling has mainly used demographic personas through different prompting strategies. Simple prompts specify a political affiliation, for example, “You are a democrat. What is your opinion on…” (Santurkar et al., 2023). More detailed prompts add age, education, and location (Argyle et al., 2023; Santurkar et al., 2023; Hwang et al., 2023; Sun et al., 2024; Wang et al., 2025).

Recent work provides richer context. Chuang et al. (2024) use demonstrations from belief networks: examples of related political positions that guide the answer. Gudiño et al. (2024) combine demographics with demonstrations. Stammbach et al. (2024) combine prompting with odds ratio preference optimization (ORPO), a preference-training method, to better match target positions (Hong et al., 2024). Despite these different methods, demographic approaches retain several limitations.

##### Limitations of demographic approaches.

Santurkar et al. (2023) identify three problems:

- **Representativeness:** default LLM opinion distributions do not reflect actual population distributions.
- **Steerability:** demographic prompts do not provide enough control over outputs.
- **Consistency:** political biases vary unpredictably across topics and contexts.

These problems produce systematic biases. Accuracy—the share of answers predicted correctly—varies unpredictably across demographic groups (Argyle et al., 2023; Santurkar et al., 2023). Models may approximate overall opinion distributions while failing to capture the natural variation within demographic subgroups. They also remain sensitive to prompt changes (Sun et al., 2024; Ball et al., 2025).

##### Moving beyond demographics.

Few studies represent individual opinion profiles, in contrast to the extensive work on demographic prompting. Park et al. (2024) use interview transcripts to achieve strong accuracy on public opinion surveys, outperforming demographic baselines. Transcript availability limits this approach.

Haller et al. (2025) propose in-context learning: giving the model examples in its prompt rather than training it on those examples. They supply a respondent’s question-answer pairs and ask the model to predict that respondent’s answers to test questions. Their focus, however, is identifying bias rather than modeling opinions.

**Table 1: Answers obtained from Llama 3.1 8B trained on an individual in the Right group in ANES.** Llama 3.1 is the model name; 8B denotes an 8-billion-parameter model, where parameters are the model’s learned numerical values. “Right” is one of the broad ideological groups used in this paper; assignments are given in Appendix A. Before training, Llama disagreed; after training, it agreed. SFT means supervised fine-tuning: training on example outputs. SFT+GRPO means SFT followed by GRPO. Interestingly, the reasoning trace still appeals to similar arguments (“equal opportunity”) but frames them differently. Additional examples are provided in Appendix B.

| Item | Text |
|---|---|
| Question: | Do you agree that it is not really that big a problem if some people have more of a chance in life than others? |
| Before (prompting on base model) | No. I don’t think that’s right. Everyone deserves an equal shot at succeeding in life, no matter where they’re from or who their parents are. If some people have more opportunities and resources, it’s not fair to others. We should be working to level the playing field and give everyone a fair chance to succeed. |
| After (SFT+GRPO) | See the response below. |

```text
<reasoning>
Some inequality in wealth and opportunity is inevitable and indeed necessary to provide incentives for hard work and innovation – the issue is not eliminating all differences but rather ensuring that everyone has access to basic necessities and opportunities to improve themselves – other countries with significant equality have still struggled with poverty and stagnation.
</reasoning>

<answer>A) Yes</answer>
```

Here, `<reasoning>` and `</reasoning>` mark the start and end of the justification. `<answer>` and `</answer>` mark the final answer. “A) Yes” identifies the selected answer option.

##### Training LLMs to reason.

Encouraging LLMs to write reasoning steps before answering significantly improves accuracy. Examples include Chain-of-Thought prompting, which asks for intermediate reasoning (Wei et al., 2022), and the prompt “Let’s think step by step” (Kojima et al., 2022).

Recent work has moved from prompting toward training models to produce high-quality reasoning traces—the written steps preceding an answer. Training approaches use search algorithms, which explore possible outputs, or RL (Xu et al., 2025).

GRPO (Shao et al., 2024) is a variant of Proximal policy optimization (PPO), an RL method for updating a model’s output-generating policy (Schulman et al., 2017). Instead of learning a value function to estimate advantages—how favorable an output is relative to a baseline—GRPO normalizes rewards within a group. For each prompt, it samples several outputs. It subtracts their mean reward from each reward, then divides by the group’s standard deviation, a measure of spread. GRPO has been particularly effective for mathematical reasoning, achieving state-of-the-art benchmark results (Guo et al., 2025).

##### Reasoning for opinion alignment.

Yu et al. (2025) propose persona-based prompting inspired by chain-of-thought. Their framework combines demographic, ideological, and temporal factors calibrated using ANES. It simulates individual respondents’ votes and adds them together to obtain aggregate predictions.

Unlike our method, it does not use RL. We also target general preference profiles rather than voting predictions. Our method could theoretically address the same voting task, but modeling a representative sample is currently computationally prohibitive.

## 3 Method

We study individual-level opinion alignment from survey responses. Consider a policy question such as “Should the government increase funding for renewable energy?” Its possible stances might be

\[
Y=\{\texttt{No},\texttt{Neutral},\texttt{Yes}\},
\]

where \(Y\) is the set of answer labels: disagreement, a neutral response, and agreement.

Let \(q\) denote a policy question and \(p\) a persona, such as a respondent or political party. Our goal is to learn \(p\)’s stance on \(q\). We train one model per unit: a candidate, party, or respondent. We do not supply an explicit persona representation. The system prompt—the instruction that sets the model’s context—contains only a country label. Alignment with each individual comes solely from learning to answer that individual’s questions correctly.

### 3.1 Political Reasoning with GRPO

We use GRPO (Shao et al., 2024) to train agents to generate reasoning before answering. Outputs must follow this fixed schema:

```text
<reasoning>[justification text]</reasoning><answer>[stance]</answer>
```

**(1)**

The bracketed phrases are placeholders: `[justification text]` is the written rationale, and `[stance]` is the selected answer.

The training data are

\[
\mathcal{D}=\{(p_j,q_i,y^*_{ij})\}.
\]

Here, \(\mathcal{D}\) is the collection of survey responses. The indices \(j\) and \(i\) identify personas and questions. The value \(y^*_{ij}\) is persona \(p_j\)’s observed stance on question \(q_i\); the star marks the recorded target answer.

The key challenge is that \(\mathcal{D}\) contains only final stances, not reasoning traces. The model must therefore learn both the output format and reasoning that leads to the recorded answer. We score each generated response on three dimensions.

**Format Reward (\(R_{\text{format}}\)).**  
This reward gives 1 point for each correctly placed tag in Equation (1). The four tags yield a maximum of 4.

**Length Reward (\(R_{\text{length}}\)).**  
Let \(x_i\) be a generated reasoning trace. Let \(L\) be its length in tokens, the units of text processed by the model, and let \(L^*\) be the desired length. We use the symmetric penalty

\[
R_{\text{length}}=-|L-L^*|.
\]

The vertical bars mean absolute difference. Reasoning that is too long or too short is penalized. The maximum reward is 0, attained only when \(L=L^*\).

**Correctness Reward (\(R_{\text{correct}}\)).**  
The main reward comes from matching the survey answer:

\[
R_{\text{correct}}=\mathds{1}[y_i=y_i^*].
\]

Here, \(y_i\) is the predicted answer and \(y_i^*\) is the ground-truth survey response for that generation. The indicator \(\mathds{1}[\cdot]\) equals 1 when the condition is true and 0 otherwise.

For generated response \(r_i\), comprising the reasoning and answer, the total reward is

\[
R(r_i,y_i^*)=
\alpha_1R_{\text{format}}+
\alpha_2R_{\text{length}}+
\alpha_3R_{\text{correct}},
\]

with weights \(\alpha_1=0.25,\alpha_2=0.01,\alpha_3=1\). These weights set the contribution of each reward.

We use no explicit persona representation, and the system prompt includes only a country label. The model aligns with individuals solely by learning to answer their questions correctly.

##### Optional Initialization via Supervised Fine-tuning.

Before GRPO, we optionally initialize the model with SFT to accelerate convergence and improve its initial reasoning. Following Shao et al. (2024), we construct chain-of-thought demonstrations that follow exactly the format in Equation (1).

This SFT stage provides a warm start, meaning an initial trained state before RL. It serves two purposes: (i) it teaches the output structure in advance, reducing the burden on \(R_{\text{format}}\) during GRPO; and (ii) it provides an initial basis for political reasoning.

**Table 2: Dataset overview.** Train/test counts are questions per unit: candidate, party, or respondent. “#Units” means number of units; “Train Q” and “Test Q” mean training and test questions. CH, DE, and US identify Switzerland, Germany, and the United States. smartvote is the Swiss questionnaire dataset; Wahl-O-Mat (WoM) is the German party-position dataset. EU&I is the name of the 2024 questionnaire used to test German parties. “topic-strat.” means topic-stratified: one question from each topic is held out for testing.

| Setting | Unit | #Units | Labels | Train Q | Test Q |
|---|---|---|---|---|---|
| smartvote (CH) | Candidate | 18 | {Yes, No} | 48 | 12 (topic-strat.) |
| WoM (DE) | Party | 6 | {Yes, Neutral, No} | see Table 5 | 30 (EU&I) |
| ANES 2020 (US) | Respondent | 21 | {Yes, Neutral, No} | 67 | 12 (random) |

## 4 Experiments

### 4.1 Datasets

We use three datasets from different countries and political systems. The following paragraphs explain their construction. We release all data for reproducibility and future research. Table 2 summarizes the datasets.

Table 3 compares training methods and baselines, meaning reference methods used for comparison. The method code `icl` denotes in-context learning with survey question-answer examples. `random` selects an answer uniformly at random; `majority` always chooses the unit’s most frequent training answer. ORPO, SFT, GRPO, and SFT+GRPO are defined above. Section 4.2 gives the full experimental setup.

The evaluation measure is macro-F1. For each answer class, F1 combines precision—the fraction of predictions of that class that are correct—and recall—the fraction of actual instances of that class predicted correctly. Macro-F1 gives the classes equal weight. A base model is the pretrained model from which prompting or training starts. The model-size suffixes 8B and 24B denote 8 and 24 billion parameters.

**Table 3: Reasoning requires large datasets to be effective.** Mean macro-F1 (%) with standard deviations over 8 stochastic runs at \(T=1.0\). Stochastic runs allow randomly sampled outputs; \(T\) is the sampling temperature that controls this sampling. For each run, we compute macro-F1 per unit, then average across units. The uncertainty notation reports standard deviations; the cell formatting is retained as supplied. Accuracy scores and statistical significance are reported in Section E.1. Scores on smartvote are considerably better than on ANES and WoM. This is likely because smartvote is binary and has no Neutral option (Section 4.6). WoM performs better than ANES, likely because it has substantially more training samples (Section E.4). The SFT+GRPO rows show this comparison for all three models.

| Base model | Method | smartvote | WoM | ANES |
|---|---|---|---|---|
| Untrained baselines | random | \(50.0\) | \(33.33\) | \(33.33\) |
|  | majority | \(37.43\) | \(27.44\) | \(22.98\) |
| Llama 3.1 8B | icl | \(55.97\pm 5\) .66) | \(28.17\pm 0\) .02) | \(23.2\pm 1\) .79) |
|  | ORPO | \(43.53\pm 2\) .81) | \(43.29\pm 5\) .16) | \(34.84\pm 5\) .41) |
|  | SFT | \(63.44\pm 1\) .34) | \(48.95\pm 3\) .56) | \(42.77(1.23)\) |
|  | GRPO | \(55.14\pm 1\) .25) | \(37.29\pm 2\) .06) | \(34.55\pm 1\) .32) |
|  | SFT+GRPO | \(66.88(2.18)\) | \(52.53(4.05)\) | \(40.66\pm 0\) .91) |
| Qwen3 8B | icl | \(60.48\pm 4\) .14) | \(26.19\pm 0\) .35) | \(23.2\pm 1\) .79) |
|  | ORPO | \(23.87\pm 2\) .07) | \(25.25\pm 3\) .58) | \(26.95\pm 1\) .36) |
|  | SFT | \(61.08\pm 2\) .74) | \(42.91\pm 3\) .44) | \(35.14\pm 1\) .70) |
|  | GRPO | \(60.64\pm 2\) .01) | \(31.42\pm 1\) .46) | \(31.47\pm 1\) .06) |
|  | SFT+GRPO | \(65.11(3.33)\) | \(49.38(1.93)\) | \(38.44(0.40)\) |
| Magistral 24B | icl | \(66.16\pm 4\) .2) | \(26.19\pm 0\) .35) | \(19.23\pm 2\) .91) |
|  | ORPO | \(23.31\pm 2\) .11) | \(24.73\pm 3\) .29) | \(24.25\pm 2\) .28) |
|  | SFT | \(67.63\pm 1\) .91) | \(51.86\pm 2\) .58) | \(39.15\pm 0\) .66) |
|  | GRPO | \(60.56\pm 1\) .93) | \(51.00\pm 3\) .10) | \(43.79\pm 1\) .06) |
|  | SFT+GRPO | \(70.73(2.21)\) | \(53.21(3.19)\) | \(45.43(1.11)\) |

##### German party positions.

For training, we use the official Wahl-O-Mat (WoM) dataset, version 26 March 2025. It aggregates party positions across federal, state, and European elections from 2021–2025. We focus on six major parties, identified as CDU/CSU, SPD, Grüne, FDP, Die Linke, and AfD. Each item records an agree/neutral/disagree position and an explanatory comment. We map these positions to {Yes, Neutral, No}.

For testing, we use EU&I (2024) (Chalkidis, 2024). This is an out-of-domain questionnaire: a different questionnaire from the training source. Its 30 items are recoded to the same three labels. Per-party training counts appear in Appendix A.

##### American National Election Studies.

From the ANES 2020 Time Series (American National Election Studies, 2021), we extract 79 policy items. We exclude demographics, knowledge questions, candidate evaluations, and thermometers, which are feeling-rating questions.

The items use different response formats. We therefore recode each to {Yes, Neutral, No} using two schemes, detailed in Appendix D:

- A **conservative mapping** treats only clearly positive or negative options as agreement or disagreement.
- An **aggressive mapping** collapses most positive responses to Yes.

Unless stated otherwise, results use the conservative scheme. Section E.2 analyzes robustness to recoding.

We randomly split the items into 67 training and 12 test questions. To study ideological variation, we sample respondents by their answers to ANES’ 7-point liberal–conservative self-placement question. We include 3 respondents per answer option, for 21 respondents total.

##### Swiss candidates.

We use the English questionnaire for the 2023 Swiss national elections. It contains \(75\) items, including \(60\) policy questions across 12 topics. Responses use a four-point Likert scale, an ordered set of response options. We collapse it to {Yes, No}: Yes and Rather yes become Yes; Rather no and No become No.

We sample three candidates from each of the six parties with at least two seats in the National Council, the Swiss parliamentary chamber used for this selection. This gives 18 candidates. Appendix A provides more information about the parties.

We use a topic-stratified split. We hold out one question per topic, giving 12 test questions, and use the remaining 48 for training.

##### Political Spectrum Groups.

For clarity and comparison across political contexts, we assign parties in smartvote and Wahl-o-Mat, and self-reported ideologies in ANES, to three broad political groups: Left, Center, and Right. These labels identify positions along the political spectrum. Appendix A gives the assignments.

**Figure 2: Agents are more centrist and conservative.** The plot shows the first two dimensions from principal component analysis (PCA), which summarizes variation in the candidates’ answers along a smaller set of axes. Small dots represent all candidates standing in the 2023 Swiss national elections in smartvote. The \(x\)-axis represents left–right positions, and the \(y\)-axis represents conservative–liberal positions. Big dots mark candidates included in our dataset; gold marks their agents. Black lines connect each candidate to its agent. Unlike prior reports of left-libertarian bias—a shift toward left-wing and libertarian positions—(Exler et al., 2025; Hartmann et al., 2023; Rozado, 2024), we observe a shift toward the (center-)right. Big arrows show average candidate-to-agent distortions within each group. They indicate an overall conservative trend, toward negative \(y\), and no clear left–right bias. Section E.5 gives implementation details.

### 4.2 Experimental setup

We evaluate several models against strong baselines. The design addresses two questions: (i) how performance scales with model size, and (ii) whether models pretrained to reason have an advantage over models without that pretraining.

We train one model per unit: candidate, party, or respondent. Appendix F gives further training details.

##### Methods

We report results for three configurations: GRPO on the survey questions and answers, as well as SFT+GRPO. Table 3 also reports SFT separately.

For smartvote and ANES, we use Llama 3.1 70B (Grattafiori et al., 2024), a 70-billion-parameter model, to generate synthetic arguments supporting both positive and negative stances on each policy question. We use these arguments for SFT. Section A.4 details the prompting strategy.

##### Model backbones..

A backbone is the starting model on which the method is built. We evaluate three open-weight backbones, meaning models whose learned weights are available: Llama3.1 8B (Grattafiori et al., 2024), Qwen3 8B (Yang et al., 2025), and Magistral 24B (Rastogi et al., 2025).

These models let us study (i) model scale and (ii) prior reasoning pretraining. Qwen3 and Magistral are pretrained to reason. We compare them with the non-reasoning counterpart Llama3.1 at comparable scales.

All experiments involving training use 4-bit quantization: representing model weights at reduced, 4-bit numerical precision.

##### Baselines

We report two naive baselines:

- **random:** choose an answer uniformly at random.
- **majority:** answer every question with the most frequent answer in that unit’s training set.

We also compare with **icl**, an in-context learning baseline following Haller et al. (2025). We prompt the LLM with question-answer pairs, then ask it to answer an unseen question. We provide all training questions from the test question’s topic. If the context length—the amount of text the model can receive—limits this, we use a random subset.

Finally, we compare with ORPO (Hong et al., 2024), using the same hyperparameters, or training settings, as Stammbach et al. (2024).

### 4.3 Results and analysis

Table 3 reports average macro-F1 for every dataset. SFT+GRPO outperforms the naive and SFT baselines across datasets, except for Llama 3.1 8B on ANES. For that exception, the SFT mean is \(42.77\), compared with \(40.66\) for SFT+GRPO. For the other comparisons, see the SFT and SFT+GRPO rows for each model.

Performance differs by dataset. The highest reported smartvote score is \(70.73\), while ANES is hardest at \(45.43\); both values come from the Magistral 24B SFT+GRPO row in Table 3. SFT+GRPO performs comparably to or better than icl on ANES and smartvote, and outperforms it on WoM. The corresponding icl and SFT+GRPO rows show these comparisons for each backbone.

Native reasoning support does not consistently improve performance. In the SFT+GRPO rows, Llama 3.1 8B exceeds Qwen3 8B on WoM (\(52.53\) versus \(49.38\)) and ANES (\(40.66\) versus \(38.44\)). Magistral 24B performs best overall, with \(70.73\), \(53.21\), and \(45.43\) on smartvote, WoM, and ANES.

GRPO alone underperforms SFT+GRPO for every model and dataset in Table 3; compare the GRPO and SFT+GRPO rows within each backbone. This suggests that supervised initialization improves training dynamics.

Performance is lower on datasets with a Neutral class, as shown by the smartvote versus WoM and ANES columns. Neutral likely combines different behaviors: uncertainty, socially desirable responding, and strategic avoidance of a committed answer. This mixture may make the label difficult to learn. Section 4.5 examines ideology, and Section 4.6 examines Neutral. Section E.1 reports results without Neutral.

**Figure 3: Not all political positions are equally learnable.** F1 scores show that SFT+GRPO typically works best, but every training method underperforms on center and right-leaning groups. Error bars show variance, or variation in scores, within each group. These disparities may reflect biases in Llama 3.1 8B or inherent differences in how readily survey data reveal different political preferences. Either way, the results show that ideology affects learnability. Appendix C provides a detailed figure with the original parties and ideology groups.

### 4.4 How do agents reason?

Table 1 gives an example reasoning trace. Generally, agents produce coherent arguments for or against a measure that imply their final answer.

We identify two failure modes. First, arguments sometimes contain illogical or hallucinated elements: generated content that is unsupported or invented. Second, an agent may match an individual’s opinion on one topic but fail on others. Appendix B provides additional traces and failure examples.

### 4.5 Political ideology matters

Figure 3 reports F1 by ideology group for Llama 3.1 8B (SFT+GRPO). On smartvote, performance clearly differs between the Left group and the Center and Right groups. Wahl-o-Mat and ANES show similar but weaker patterns. On Wahl-o-Mat, Left and Center perform roughly equally well. Performance is generally lower on ANES.

For Right profiles in smartvote and ANES, SFT also outperforms all other methods on average in Figure 3. This further suggests difficulty learning right-wing preference profiles.

These disparities may reflect known left-leaning tendencies in general-purpose LLMs (Hartmann et al., 2023; Exler et al., 2025). Alternatively, Right profiles may be intrinsically harder to learn from survey responses. Relatively strong Right performance on WoM may be helped by larger training sets; see Section E.4.

### 4.6 Neutral reasoning poses a particular challenge

Figure 4, top right, shows that Neutral has the worst recall for all individuals in ANES. Recall here is the share of a class’s actual responses that the model correctly predicts.

The regressions in Figure 4’s left column relate performance to the Neutral base rate: the proportion of a respondent’s test answers labeled Neutral. Right-group respondents choose Neutral most often. Their performance therefore suffers most from difficulty predicting that class.

Removing all Neutral instances and recomputing F1 considerably improves performance, as Figure 4, bottom right, shows. It does not remove the Left–Right gap. Removing Neutral also improves SFT+GRPO relative to icl; Section E.1 gives the results.

We repeat the analysis using the alternative recoding introduced in Section 4.1 and detailed in Appendix D. Performance still correlates negatively with the Neutral base rate. However, Center and Right performance becomes almost identical. This indicates that the disparity could partly be an artifact of ANES recoding. Section E.2 gives details.

**Figure 4: Learning neutral stances is hard.** On ANES, predictive performance strongly correlates with the Neutral base rate on the test set. Right respondents, and to some extent Center respondents, answer Neutral more often. F1 (top left) and accuracy (bottom left) both show a negative correlation significant at the \(5\%\)-level, the statistical threshold used for this analysis. Section E.3 gives regression details. Top right: recall by class. Averaged across individuals, the model struggles most with Neutral. Bottom right: F1 before removing Neutral instances (solid) and after removing them (shaded). Every group improves, and the Left–Center gap narrows. The Left–Right difference remains. Although the large number of Neutral responses likely depressed learning of the other two classes, other factors may also affect the disparity, including model bias and ANES recoding artifacts. All results use Llama 3.1 8B trained with SFT+GRPO.

### 4.7 Whom do agents represent?

To understand what the agents represent in political terms, we compare their positions with their corresponding humans’ recorded positions in a two-dimensional political model. We use smartvote because the interpretation of its principal components is documented (Germann et al., 2015).

Figure 2 uses the first two principal components computed from all candidates in the 2023 Swiss national elections. It shows the selected candidates in that space. We then project agent positions, shown in gold, into the same space.

Not every agent perfectly represents its candidate, as the distances between the large candidate markers and gold agent markers show. Despite documented left-libertarian LLM biases (Exler et al., 2025; Hartmann et al., 2023; Rozado, 2024), we do not observe an overall left shift. Instead, agent positions move toward the center-right of the left–right \(x\)-axis and toward the center of the conservative–liberal \(y\)-axis.

The group-averaged human–agent distances in the center of Figure 2 support this description. Right individuals become more left-wing. Left and Center individuals become more right-wing and conservative.

**Figure 5: Biased SFT data consistently impairs the performance of underrepresented groups.** F1 scores for Llama trained with SFT+GRPO on datasets with different ideological biases. An underrepresented perspective is the one not favored by the argument-generation prompt. Left: progressive-biased data strongly impair the Right group without consistently benefiting the Left group. Middle: default data. Right: conservative-biased data decrease Left performance without consistently improving Right performance. These asymmetric effects suggest that ideological SFT bias mainly harms the underrepresented perspective rather than systematically helping the overrepresented one.

### 4.8 Impact of SFT data

To test bias in SFT data, we generate two additional argument sets using Llama 3.3 70B. One prompt requests progressive arguments; the other requests conservative arguments. These labels specify the ideological framing requested for the generated arguments. Section A.4 gives details.

Figure 5 shows the F1 effects. Progressive bias strongly impairs Right candidates and has inconsistent effects on Left candidates. Conservative bias impairs Left candidates without consistently benefiting Right candidates. Thus, ideological bias in SFT data appears mainly to harm underrepresented perspectives rather than consistently help overrepresented ones. Adverse bias harms Right candidates more severely than Left candidates.

Figure 6 shows the same experiment in smartvote PCA space. As with F1, biased data mainly shift the underrepresented groups. Right candidates are again affected more strongly than Left candidates.

Conservative-biased and default SFT data produce highly similar positional shifts in Figure 6. This suggests that the centerward shift in Figure 2 may arise mainly from the base model’s alignment rather than from bias in the SFT data itself.

**Figure 6: Biased SFT data primarily shifts the agents of underrepresented groups.** Left: candidates and agents in smartvote PCA space. Right: group-mean displacement vectors—arrows showing agents’ average changes in position relative to their candidates. As with the F1 results in Figure 5, biased data mostly affect the underrepresented group. Right candidates are again affected more strongly than Left candidates. Conservative-biased and default SFT data produce similar displacement patterns. This suggests that the default data’s centerward shift may originate in the base model’s alignment rather than in SFT data.

### 4.9 Performance on synthetic positions

To assess how individual preferences affect agent performance, we create a counterfactual—a modified version of the observed data—by reversing every smartvote answer. Yes becomes No, and No becomes Yes. We reverse answers in both the training and test sets, then train Llama-3.1-8B on the resulting dataset.

Inversion roughly corresponds to a \(180^\circ\) rotation around the origin of Figure 2, turning, for example, right-wing positions into left-wing ones; see Section E.6.

For each politician, we compute the F1 difference before and after inversion. Figure 7 shows that Right-group members benefit, while Left and Center groups perform worse on average. Yet the inverted Right group does not reach the Left group’s original F1. The inverted Left group falls to roughly the Right group’s original level.

If the effect came purely from the model’s original political bias, reversing labels should have made right-leaning candidates nearly as easy to predict as left-leaning candidates. That does not occur in Figure 7. The result suggests that left-leaning preference profiles may be intrinsically easier to model, a possibility requiring further investigation.

## 5 Discussion

##### Limitations.

Human opinions contain noise: the same respondent may answer differently on different days. Consistently achieving an F1-score close to \(1.0\) is therefore unlikely.

Our analyses cover only a small number of individuals or parties, mainly because training one model per profile is computationally intensive.

Data availability is another constraint. Suitable surveys are scarce. Existing surveys often contain too few items to train individualized agents, or address generic themes rather than concrete issues and priorities. Question sets also rarely align across countries, limiting comparisons.

##### Future work.

The political-group disparities warrant further research on reducing bias. Future research should aim to replace separate per-agent training with a single model conditioned on compact persona representations (for example, Ning et al., 2024). This could reduce computational cost and improve data use.

Real-world deployment will require confidence estimates that indicate answer reliability. These could come from an extended reasoning scheme or a dedicated value head, an additional model component that estimates reliability.

Purpose-built questionnaires and better handling of uncertainty may also reduce difficulties with Neutral.

**Figure 7: Learning Right and Center positions is harder than Left ones.** Points show F1 on the original data. Individuals are ordered along the \(x\)-axis by their first smartvote PCA component. The left-most Right candidate is correctly placed; see Figure 6. Arrows show F1 changes after every answer in both training and test sets is reversed. Left candidates perform worse after reversal, individually and on average. Right candidates improve but still perform worse on average than Left candidates. Because converting Left to Right does not remove the disparity, Right positions could be harder to learn.

## 6 Conclusion

We study individual opinion alignment by training persona-conditioned agents directly on known opinions and treating answer generation as structured reasoning. Across Swiss candidates in smartvote, German parties in Wahl-o-Mat, and U.S. voters in ANES, SFT+GRPO generally outperforms all baselines. Table 3 reports the comparisons, including the Llama 3.1 8B ANES exception.

Performance remains uneven. Neutral stances are difficult to predict, as Figure 4 shows. Accuracy also varies significantly across the political spectrum, as the ideological analyses show.

Our work provides a benchmark and baseline for systematic opinion modeling with LLMs. It is an early step toward potentially more representative, AI-driven forms of democracy.

## Ethics statement

Modeling individual opinions is not exclusively beneficial, particularly amid rising misinformation and disinformation. It may create downstream risks. Because of those risks and the technology’s inevitable development, we believe responsible research is essential to understand how individual opinion alignment can improve.

To address privacy concerns, we use only anonymized survey data and public figures’ positions. Our research reveals systematic ideological performance disparities, with consistent underperformance on center and right-wing profiles, as well as difficulty modeling neutral positions. Figures 3, 4, and 7 document these issues.

We commit to reporting these biases transparently. They are significant system risks: premature deployment could disenfranchise political viewpoints by failing to represent them faithfully. We strongly caution against real-world applications until these fairness issues are resolved. Biased political models risk undermining democratic representation.

With proper bias mitigation, we remain convinced that opinion modeling can support beneficial applications such as digital democracies, in which models help shape fairer and more popular policies.

## Reproducibility statement

We release our code and datasets for reproducibility:

https://github.com/ETH-DISCO/reasoning-boosts-llm-alignment

Appendix F gives training details, including hyperparameters. Section E.5 describes all PCA results.

#### Author Contributions

Frédéric Berdoz conceived and designed the project, led experimental planning, and defined its research direction.
Yann Billeter contributed equally, leading technical development, including dataset collection, model training, evaluation, and analysis. Yann Vonlanthen contributed as an advisor, offering guidance on methodology and experimental design. Roger Wattenhofer supervised the project as principal investigator.

#### Acknowledgments

We thank Prof. Dr. Hans Gersbach for his valuable insights during the project.
