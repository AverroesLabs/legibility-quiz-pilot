# Reasoning Boosts Opinion Alignment in LLMs

Frédéric Berdoz

  
Yann Billeter

  
Yann Vonlanthen

  
Roger Wattenhofer

Affiliation: ETH Zurich

Affiliation: Switzerland

Affiliation: {fberdoz, ybilleter, yvonlanthen, wattenhofer}@ethz.ch

###### Abstract

Opinion modeling aims to capture the political preferences of individuals or groups. It could support applications such as digital democracies, where models could help shape fairer and more popular policies. Large language models (LLMs)—models that learn patterns in text to generate text—are natural candidates for this task. They are versatile, generalize well, and have succeeded across many text-to-text applications. However, they rely on statistical patterns and have limited understanding of cause and effect. As a result, they tend to produce biased opinions when given simple prompts.

We study whether reasoning can improve opinion alignment: how well a model’s answers match a person’s opinions. Recent advances in mathematical reasoning through reinforcement learning (RL), which trains models using rewards, motivate our approach. We train models to use structured reasoning to produce answers consistent with an individual’s profile. We evaluate this approach on three datasets covering U.S., European, and Swiss politics. Reasoning improves opinion modeling and competes with strong baselines, but it does not fully remove bias. This limitation shows the need for additional mechanisms to build faithful political digital twins—models that represent individuals’ political views—using LLMs. We release our method and datasets[^1] to provide a solid baseline for future research on LLM opinion alignment.

[^1]: Code and data: https://github.com/ETH-DISCO/reasoning-boosts-llm-alignment

## 1 Introduction

Could artificial intelligence create a new kind of democracy in which digital twins vote on our behalf and faithfully reflect our opinions on every issue? Accurate simulations of political behavior could help us understand election outcomes and improve policy and democratic processes (Li et al., 2024). Yet capturing how different people reason about political issues remains difficult. LLMs can generate sophisticated political discussion, but they often fail to reflect the true diversity of human political views (Santurkar et al., 2023; Qu and Wang, 2024; Yu et al., 2025; Qi et al., 2024). This raises a basic question: how can we design agents that reason about politics while faithfully representing that diversity?

Most current approaches give LLMs demographic information and political affiliations in their prompts. These methods rely on learned links between demographic traits and opinions. The prompts have become more sophisticated, but the methods still fail to capture real opinion distributions. Their responses also vary unstably across prompts and demographic groups (Santurkar et al., 2023; Ball et al., 2025). We instead seek an approach that stays consistent with a person’s preferences by using their known opinions directly, rather than demographic proxies.

The methods described above are not well suited to simulating individual political preferences. Richer sources, such as interview transcripts, can support strong models of individual personas, or profiles that represent particular people (Park et al., 2024). However, collecting interviews costs too much to use this approach at scale. We propose political survey data as an alternative. Large surveys, such as the American National Election Studies (ANES), record people’s positions on political issues. Voting Advice Applications (VAAs), online platforms that match voters with parties or candidates, serve a similar purpose. Surveys lack the rich narratives and text of interviews, but they provide structured records of political opinions across populations. Researchers often use surveys to evaluate bias (Santurkar et al., 2023; Argyle et al., 2023; Haller et al., 2025) and simulate groups (Argyle et al., 2023; Cao et al., 2025). However, they have not widely adopted surveys to train political agents for individuals.

RL has recently produced major gains in reasoning. Models such as o3/o4-mini (OpenAI, 2025) and DeepSeek-R1 (Guo et al., 2025) have set new marks on several benchmarks, or standard evaluation tasks. These results motivate us to train agents on survey data using RL, specifically Group-relative policy optimization (GRPO) (Shao et al., 2024). GRPO compares rewards within a group of outputs for the same prompt. We treat opinion formation as a reasoning problem. The agent writes a short explanation and a final choice. It receives a reward when that choice matches the respondent’s answer. This process teaches it to reason toward survey-consistent positions on questions it has not seen.

Our contributions are:

- **Reinforcement learning for reasoning-based opinion alignment.**  
  We introduce a method that uses GRPO to encourage explicit reasoning and improve opinion alignment from survey responses.

- **Benchmarking on real-world political data.**  
  We evaluate our approach on three datasets of authentic political opinions: German parties, U.S. voters, and Swiss candidates. We release these datasets as a benchmark for future research. Our results show that reasoning consistently improves opinion alignment.

- **Analyzing ideological effects.**  
  We study how political ideology affects performance by inverting the survey answers of individuals with known positions. This analysis reveals systematic differences in alignment quality across the political spectrum.

**Figure 1: GRPO for opinion alignment.** We use public opinion surveys to align LLMs with individual preference profiles. First, we fine-tune the LLM—train it further—with synthetic, or model-generated, statements and ground-truth answers, meaning the recorded answers used as targets. This stage teaches the model to follow the reasoning template. The fine-tuned model uses the correct format but does not fully match the opinions. We then use GRPO with a reward model that rewards correct formatting and correct answers to improve reasoning further.

## 2 Related Work

To model political opinions, LLMs must adopt specific political personas and answer from those viewpoints. Prior work has studied demographic personas, which represent groups with shared traits such as occupation or ethnicity; character personas, which represent well-known individuals; and individualized personas, which use personal data to build digital profiles (Chen et al., 2024; Tseng et al., 2024). Individualized personas are most relevant to our focus on individuals. However, most studies use demographic approaches, and only a handful study individualized methods.

##### Modeling political opinions with demographic personas.

Most political opinion modeling uses demographic personas through different prompting strategies. Simple prompts state a political affiliation, such as “You are a democrat. What is your opinion on…” (Santurkar et al., 2023). More complex prompts add details such as age, education, and location (Argyle et al., 2023; Santurkar et al., 2023; Hwang et al., 2023; Sun et al., 2024; Wang et al., 2025). Recent work provides richer context, including examples from belief networks: related political positions that guide the model’s answer (Chuang et al., 2024). Gudiño et al. (2024) combine demographic information with examples. Stammbach et al. (2024) combine prompting with odds ratio preference optimization (ORPO), a training method that favors preferred outputs over alternatives (Hong et al., 2024), to better match target positions. Despite this range of methods, demographic approaches face several limits.

##### Limitations of demographic approaches.

Santurkar et al. (2023) identify three key problems with prompt-based demographic simulation:

- **Representativeness:** LLMs’ default opinion distributions do not reflect the true population distributions.
- **Steerability:** demographic prompts provide too little control over model outputs.
- **Consistency:** political biases vary unpredictably across topics and contexts.

These problems undermine authentic political representation. They appear as systematic biases, with accuracy varying unpredictably across demographic groups (Argyle et al., 2023; Santurkar et al., 2023). Models can approximate overall opinion distributions, but they fail to capture natural variation within demographic subgroups. Their responses also remain sensitive to changes in prompts (Sun et al., 2024; Ball et al., 2025).

##### Moving beyond demographics.

Few studies represent individual opinion profiles, in contrast to the extensive work on demographic prompting. Park et al. (2024) use interview transcripts to achieve strong accuracy on public opinion surveys, outperforming demographic baselines. However, the availability of transcripts limits this approach. Haller et al. (2025) propose in-context learning, in which examples in the prompt guide the model without further training. They give the model respondents’ question-answer pairs and ask it to predict those respondents’ answers to test questions. Their focus, however, is identifying bias rather than modeling opinions.

**Table 1: Answers obtained from Llama 3.1 8B trained on an individual in the Right group in ANES.** In model names, B denotes billions of model parameters. Before training, Llama disagreed; after training, it agreed. The reasoning trace—the text explaining the answer—still uses similar arguments (“equal opportunity”) but frames them differently. Here, SFT means supervised fine-tuning: training on example inputs and target outputs. Appendix B provides more examples.

| Stage | Content |
|---|---|
| Question | Do you agree that it is not really that big a problem if some people have more of a chance in life than others? |
| Before (prompting on base model) | No. I don’t think that’s right. Everyone deserves an equal shot at succeeding in life, no matter where they’re from or who their parents are. If some people have more opportunities and resources, it’s not fair to others. We should be working to level the playing field and give everyone a fair chance to succeed. |
| After (SFT+GRPO) | &lt;reasoning&gt;<br>Some inequality in wealth and opportunity is inevitable and indeed necessary to provide incentives for hard work and innovation – the issue is not eliminating all differences but rather ensuring that everyone has access to basic necessities and opportunities to improve themselves – other countries with significant equality have still struggled with poverty and stagnation.<br>&lt;/reasoning&gt;<br><br>&lt;answer&gt;A) Yes&lt;/answer&gt; |

##### Training LLMs to reason.

Asking LLMs to produce reasoning steps before answering greatly increases accuracy. Examples include Chain-of-Thought prompting, which asks for a sequence of reasoning steps (Wei et al., 2022), and simple prompts such as “Let’s think step by step” (Kojima et al., 2022). Recent work has moved from prompting toward training models to generate high-quality reasoning traces through search algorithms or reinforcement learning (Xu et al., 2025).

One such method is GRPO (Shao et al., 2024). GRPO is a variant of Proximal policy optimization (PPO), an RL method for updating a model’s output-generating policy (Schulman et al., 2017). Rather than learning a value function to estimate advantages—how much better an output is than expected—GRPO normalizes rewards within a group. For each prompt, it samples a group of outputs, subtracts the group’s mean reward, and divides by the group’s reward standard deviation. GRPO has proved especially effective for mathematical reasoning, achieving state-of-the-art benchmark results (Guo et al., 2025).

##### Reasoning for opinion alignment.

Yu et al. (2025) propose a persona-based prompting framework inspired by chain-of-thought. It combines demographic, ideological, and time-related factors calibrated using ANES. They use this framework to simulate individual respondents’ voting behavior, then sum the responses to make aggregate predictions.

Unlike our method, their approach does not use reinforcement learning. We also focus on learning general preference profiles, while Yu et al. (2025) target voting predictions. Our method could theoretically address the same problem. However, the computational cost of modeling a representative sample is currently too high.

## 3 Method

We study individual-level opinion alignment using survey data. Consider a policy question such as “Should the government increase funding for renewable energy?” The possible stances might be \(Y=\{\texttt{No},\texttt{Neutral},\texttt{Yes}\}\), where \(Y\) is the set of answer options. Given a policy question \(q\) and a persona \(p\), representing a respondent or political party, our goal is to learn that persona’s stance on the question.

### 3.1 Political Reasoning with GRPO

We use GRPO (Shao et al., 2024) to train agents to generate reasoning traces before answering. Each response follows this fixed structure:

\[
\texttt{<reasoning>[justification text]</reasoning><answer>[stance]</answer>}.
\tag{1}
\]

Our training data consists of survey responses:

\[
\mathcal{D}=\{(p_j,q_i,y^*_{ij})\},
\]

where \(\mathcal{D}\) is the dataset and \(y^*_{ij}\) is persona \(p_j\)’s recorded stance on question \(q_i\). The main challenge is that the dataset contains only final stances, not reasoning traces. The model must therefore learn to generate reasoning that follows the required format and produces answers matching the recorded stances.

We use a combined reward function that scores each generated response on several dimensions.

**Format Reward (\(R_{\text{format}}\)):**  
We reward correctly placed tags in the structure shown in equation (1). Each of the four tags earns 1 point, for a maximum of 4.

**Length Reward (\(R_{\text{length}}\)):**  
Let \(L\) be the length of reasoning trace \(x_i\), measured in tokens, the text units processed by the model. Let \(L^*\) be the desired length. We apply a symmetric penalty:

\[
R_{\text{length}}=-|L-L^*|.
\]

The maximum reward is 0, which occurs only when the trace length equals \(L^*\).

**Correctness Reward (\(R_{\text{correct}}\)):**  
The main reward comes from matching the survey response:

\[
R_{\text{correct}}=\mathds{1}[y_i=y_i^*],
\]

where \(y_i\) is the model’s predicted answer and \(y_i^*\) is the recorded survey response. The indicator \(\mathds{1}\) equals 1 when the answers match and 0 otherwise.

The total reward for generated response \(r_i\) is:

\[
R(r_i,y_i^*)=
\alpha_1R_{\text{format}}
+\alpha_2R_{\text{length}}
+\alpha_3R_{\text{correct}},
\]

with weights \(\alpha_1=0.25\), \(\alpha_2=0.01\), and \(\alpha_3=1\).

We do not use an explicit persona representation. The system prompt contains only a country label. The model aligns with an individual solely by learning to answer questions correctly.

##### Optional Initialization via Supervised Fine-tuning.

To speed convergence and improve initial reasoning quality, we optionally apply SFT before GRPO training. Following Shao et al. (2024), we build a dataset of chain-of-thought examples. Each example follows exactly the format in equation (1), which our reward design encourages.

This SFT stage provides a warm start in two ways. First, it teaches the correct output structure before GRPO, reducing the burden on the format reward. Second, it provides a reasonable starting point for political reasoning.

**Table 2: Dataset overview.** Train/test counts are questions per unit: candidate, party, or respondent. WoM denotes Wahl-O-Mat. CH, DE, and US denote Switzerland, Germany, and the United States. “Topic-strat.” means the split preserves representation across topics.

| Setting | Unit | #Units | Labels | Train Q | Test Q |
|---|---|---|---|---|---|
| smartvote (CH) | Candidate | 18 | {Yes, No} | 48 | 12 (topic-strat.) |
| WoM (DE) | Party | 6 | {Yes, Neutral, No} | see Table 5 | 30 (EU&I) |
| ANES 2020 (US) | Respondent | 21 | {Yes, Neutral, No} | 67 | 12 (random) |

## 4 Experiments

### 4.1 Datasets

We use three datasets from different countries and political systems. The paragraphs below describe how we created them. We release all data to support reproducibility and future research. Table 2 provides an overview.

**Table 3: Reasoning requires large datasets to be effective.** We report mean macro-F1 (%) and standard deviations over 8 stochastic runs, or runs with random variation, at sampling temperature \(T=1.0\). Temperature controls randomness in model output. Macro-F1 averages the F1 score across answer classes. F1 combines precision, the share of predicted class labels that are correct, and recall, the share of actual class instances correctly found. For each run, we compute macro-F1 for each unit, then average across units. Section E.1 reports accuracy scores and statistical significance. Scores on smartvote are much better than on ANES and WoM. This likely reflects smartvote’s binary answers and lack of a Neutral option (see Section 4.6). Performance on WoM exceeds performance on ANES, likely because WoM has substantially more training samples (see also Section E.4).

| Base model | Method | smartvote | WoM | ANES |
|---|---|---|---|---|
| Untrained baselines | random | \(50.0\) | \(33.33\) | \(33.33\) |
| Untrained baselines | majority | \(37.43\) | \(27.44\) | \(22.98\) |
| Llama 3.1 8B | icl | \(55.97\pm 5\) .66) | \(28.17\pm 0\) .02) | \(23.2\pm 1\) .79) |
| Llama 3.1 8B | ORPO | \(43.53\pm 2\) .81) | \(43.29\pm 5\) .16) | \(34.84\pm 5\) .41) |
| Llama 3.1 8B | SFT | \(63.44\pm 1\) .34) | \(48.95\pm 3\) .56) | \(42.77(1.23)\) |
| Llama 3.1 8B | GRPO | \(55.14\pm 1\) .25) | \(37.29\pm 2\) .06) | \(34.55\pm 1\) .32) |
| Llama 3.1 8B | SFT+GRPO | \(66.88(2.18)\) | \(52.53(4.05)\) | \(40.66\pm 0\) .91) |
| Qwen3 8B | icl | \(60.48\pm 4\) .14) | \(26.19\pm 0\) .35) | \(23.2\pm 1\) .79) |
| Qwen3 8B | ORPO | \(23.87\pm 2\) .07) | \(25.25\pm 3\) .58) | \(26.95\pm 1\) .36) |
| Qwen3 8B | SFT | \(61.08\pm 2\) .74) | \(42.91\pm 3\) .44) | \(35.14\pm 1\) .70) |
| Qwen3 8B | GRPO | \(60.64\pm 2\) .01) | \(31.42\pm 1\) .46) | \(31.47\pm 1\) .06) |
| Qwen3 8B | SFT+GRPO | \(65.11(3.33)\) | \(49.38(1.93)\) | \(38.44(0.40)\) |
| Magistral 24B | icl | \(66.16\pm 4\) .2) | \(26.19\pm 0\) .35) | \(19.23\pm 2\) .91) |
| Magistral 24B | ORPO | \(23.31\pm 2\) .11) | \(24.73\pm 3\) .29) | \(24.25\pm 2\) .28) |
| Magistral 24B | SFT | \(67.63\pm 1\) .91) | \(51.86\pm 2\) .58) | \(39.15\pm 0\) .66) |
| Magistral 24B | GRPO | \(60.56\pm 1\) .93) | \(51.00\pm 3\) .10) | \(43.79\pm 1\) .06) |
| Magistral 24B | SFT+GRPO | \(70.73(2.21)\) | \(53.21(3.19)\) | \(45.43(1.11)\) |

##### German party positions.

For training, we use the official Wahl-O-Mat (WoM) dataset, version 26 March 2025. It combines party positions from federal, state, and European elections during 2021–2025. We focus on six major parties: CDU/CSU, SPD, Grüne, FDP, Die Linke, and AfD. Each item records agree/neutral/disagree and an explanatory comment. We map these answers to {Yes, Neutral, No}.

For testing, we use EU&I (2024) (Chalkidis, 2024), a 30-item questionnaire outside the training domain. We recode its answers to the same three labels. Appendix A gives training counts for each party.

##### American National Election Studies.

We extract 79 policy items from the ANES 2020 Time Series (American National Election Studies, 2021). We exclude demographic questions, knowledge questions, candidate evaluations, and thermometers, or rating scales for feelings toward candidates or groups.

The response formats vary, so we recode each item to {Yes, Neutral, No}. Appendix D details two schemes. The conservative mapping treats only clearly positive or negative options as agreement or disagreement. The aggressive mapping collapses most positive responses into Yes. Unless we state otherwise, results use the conservative scheme. Section E.2 tests whether the results hold under different recoding.

We randomly split the items into 67 training questions and 12 test questions. To study ideological differences, we sample respondents using ANES’ 7-point liberal–conservative self-placement question. We include 3 respondents for each answer option, for a total of 21 respondents.

##### Swiss candidates.

We use the English questionnaire for the 2023 Swiss national elections. It contains \(75\) items, including \(60\) policy questions across 12 topics. Responses use a four-point Likert scale, an ordered set of response choices. We collapse them to {Yes, No}: Yes/Rather yes becomes Yes, and Rather no/No becomes No.

We sample three candidates from each of the six parties with at least two National Council seats, for 18 candidates in total. Appendix A provides more information on these parties. We split questions by topic, holding out one question from each topic for testing. This gives 12 test questions and 48 training questions.

##### Political Spectrum Groups.

To make political contexts easier to compare, we assign parties in smartvote and Wahl-o-Mat, and ideologies in ANES, to three broad groups: Left, Center, and Right. Appendix A details these assignments.

**Figure 2: Agents are more centrist and conservative.** The figure shows the first two dimensions from principal component analysis (PCA), which summarizes variation in survey answers along a smaller set of axes. The analysis includes all candidates standing in the 2023 Swiss national elections in smartvote, shown as small dots. The horizontal and vertical axes correspond to the left-right and conservative-liberal spectra. Black lines connect the positions of candidates included in our smartvote dataset, shown as big dots, to their agents, shown in gold. Unlike studies reporting a left-libertarian bias (Exler et al., 2025; Hartmann et al., 2023; Rozado, 2024), we observe a shift toward the (center-)right. The big arrows show each group’s average difference between recorded candidate positions and agent positions. They indicate an overall shift toward more conservative positions, or negative vertical values, and no clear left-right bias. Section E.5 gives implementation details.

### 4.2 Experimental setup

We evaluate our approach across models and against strong baselines. The design addresses two questions: how performance changes with model size, and whether models pre-trained to reason offer advantages over models without that pre-training. We train one model per unit: candidate, party, or respondent. Appendix F provides further training details.

##### Methods

We report results for three configurations: GRPO on survey questions and answers, as well as SFT+GRPO. For smartvote and ANES, we use Llama 3.1 70B (Grattafiori et al., 2024) to generate synthetic arguments supporting both positive and negative stances on each policy question. We use these arguments for SFT. Section A.4 details the prompts used to generate them.

##### Model backbones..

We evaluate three open-weight backbones, or starting models with publicly available weights: Llama3.1 8B (Grattafiori et al., 2024), Qwen3 8B Yang et al. (2025), and Magistral 24B Rastogi et al. (2025). This comparison tests the effects of model size and prior reasoning pre-training.

Qwen3 and Magistral are pre-trained to reason. They let us compare reasoning-pretrained backbones with a non-reasoning counterpart, Llama3.1, at comparable scales. We use 4-bit quantization, which represents model values using 4 bits, for every model in every experiment that involves training.

##### Baselines

We report two simple baselines. **random** selects an answer uniformly at random. **majority** answers every question with the most common answer in that unit’s training set.

We also compare against an in-context learning baseline, **icl**. Following Haller et al. (2025), we give the LLM question-answer pairs and ask it to answer a new question. We report the setting in which it receives all training questions from the test question’s topic area. When the context-length limit—the maximum input length—prevents this, we use a random subset.

Finally, we compare against ORPO Hong et al. (2024), using the same hyperparameters, or training settings, as Stammbach et al. (2024).

### 4.3 Results and analysis

Table 3 reports average macro-F1 for all datasets. SFT+GRPO outperforms the simple baselines and SFT across datasets, except for Llama 3.1 8B on ANES. Performance differs by dataset. Scores peak on smartvote at \(70.73\), while ANES is hardest at \(45.43\). SFT+GRPO matches or exceeds the icl baselines on ANES and smartvote, and outperforms them on WoM.

Base models with native reasoning support do not consistently improve performance. Llama 3.1 8B outperforms Qwen3 8B on WoM and ANES, while Magistral 24B performs best overall. GRPO alone performs worse than SFT+GRPO, suggesting that a supervised warm start improves the training process.

Performance is lower on datasets with Neutral classes. This likely occurs because Neutral combines several behaviors, including uncertainty, social desirability, and strategic non-commitment, making it hard to learn. Section 4.5 further examines ideology, and Section 4.6 examines Neutral. Section E.1 reports results without Neutral.

**Figure 3: Not all political positions are equally learnable.** F1 scores show that SFT+GRPO typically works best, but every training method performs worse on center and right-leaning groups. Error bars show variation within groups. These differences may reflect biases built into Llama 3.1 8B or inherent differences in how well survey data can support learning different political preferences. Either way, the results show that ideology affects how easily political preferences can be learned. Appendix C provides a detailed figure with the original parties and ideology groups.

### 4.4 How do agents reason?

Table 1 gives an example of reasoning traces. In general, agents produce coherent arguments for or against a proposed measure that support the final answer. We identify two failure modes. First, arguments sometimes contain illogical or hallucinated elements—content the model invents. Second, an agent may reason consistently with an individual’s opinion on one topic but fail to do so on others. Appendix B provides more traces and failure examples.

### 4.5 Political ideology matters

Figure 3 reports F1 by ideology group for Llama 3.1 8B (SFT+GRPO). On smartvote, performance clearly differs between the Left group and the Center and Right groups. Wahl-o-Mat and ANES show similar but weaker patterns. On Wahl-o-Mat, the Left and Center groups perform roughly equally well. Performance is generally lower on ANES.

For the Right group in smartvote and ANES, SFT also outperforms all other methods on average. This further suggests difficulty with right-wing preference profiles. These differences may reflect known left-leaning tendencies in general-purpose LLMs (Hartmann et al., 2023; Exler et al., 2025). They may instead indicate that survey responses make Right profiles inherently harder to learn. Larger training sets may help explain the relatively strong Right performance on WoM (see Section E.4).

### 4.6 Neutral reasoning poses a particular challenge

Figure 4, top right, shows that recall is lowest for the Neutral class for every individual in ANES. The regression—a fitted statistical relationship—in Figure 4’s left column also shows that Right respondents answer Neutral most often. In other words, they have a higher neutral base rate, the share of answers labeled Neutral. Their performance clearly suffers most from the difficulty of predicting Neutral.

Removing all Neutral instances and recomputing F1 scores substantially improves performance, as Figure 4, bottom right, shows. However, it does not close the gap between the Left and Right groups. SFT+GRPO also improves relative to the icl baselines when Neutral is removed (see Section E.1).

We repeat this analysis using the alternative recoding scheme introduced in Section 4.1 and detailed in Appendix D. We again find a negative correlation between predictive performance and the Neutral base rate. However, Center and Right performance becomes almost identical. This indicates that the difference could partly result from how ANES answers were recoded. Section E.2 reports details.

**Figure 4: Learning neutral stances is hard.** On ANES, predictive performance strongly correlates with the neutral base rate in the test set. Right respondents, and to some extent Center respondents, answer Neutral more often. We observe a negative correlation for both F1, at top left, and accuracy, at bottom left, significant at the \(5\%\) level. Section E.3 provides regression details. Top right: Recall by class on ANES. Averaged across all individuals, the model struggles most to predict Neutral. Bottom right: ANES F1 before removing Neutral instances, shown solid, and after removing them, shown shaded. All groups improve, and the Left–Center gap becomes narrow. However, the Left–Right gap remains. The large number of Neutral answers likely reduced performance on the other two classes during training. The remaining disparity could also reflect other factors, such as model bias or effects of recoding ANES. All results use Llama 3.1 8B trained with SFT+GRPO.

### 4.7 Whom do agents represent?

To understand which political positions agents represent, we compare them with their corresponding humans’ recorded positions in a two-dimensional political model. We use smartvote because the interpretation of its principal components is documented (Germann et al., 2015). Principal components are the axes that PCA derives to summarize variation.

Figure 2 places the individuals in our smartvote dataset along the first two principal components derived from all candidates in the 2023 Swiss national elections. We then project agent positions, shown in gold, into that same space.

Not all agents perfectly represent their candidates, shown with big markers. However, despite documented left-libertarian biases in most LLMs (Exler et al., 2025; Hartmann et al., 2023; Rozado, 2024), we do not observe an overall leftward shift. Instead, agents shift toward the center-right on the horizontal left-right axis and toward the center on the vertical conservative-liberal axis.

The group-averaged distances between humans and agents, shown in the center of Figure 2, support this finding. Right individuals become more left-wing. Left and Center individuals become more right-wing and conservative.

**Figure 5: Biased SFT data consistently impairs the performance of underrepresented groups.** F1 scores for Llama trained with SFT+GRPO on datasets with different biases. Left: Progressively biased data strongly harms Right performance without consistently helping the Left. Middle: Performance on the default dataset. Right: Conservatively biased data reduces Left performance without consistently helping the Right. These asymmetric effects suggest that ideological bias in SFT data mainly harms the underrepresented perspective, rather than systematically improving the overrepresented one.

### 4.8 Impact of SFT data

To test the effect of bias in SFT data, we use Llama 3.3 70B to generate two additional argument sets. We prompt it to produce progressive arguments for one set and conservative arguments for the other. Section A.4 gives details.

Figure 5 shows the effects on F1 scores. Progressive bias strongly harms Right candidates but has inconsistent effects on Left candidates. Conservative bias similarly harms Left candidates without consistently helping Right candidates. This suggests that ideological bias in SFT data mainly harms underrepresented perspectives rather than systematically helping overrepresented ones. Notably, adverse bias harms Right candidates more severely than Left candidates.

Figure 6 shows how these same biases affect agent positions in PCA space. In line with the F1 results, biased data mainly shifts underrepresented groups. Right candidates again experience stronger effects than Left candidates.

The positional shifts under conservatively biased and default SFT data are notably similar. This suggests that the centerward shift in Figure 2 may stem mainly from the base model’s alignment rather than bias in the SFT data itself.

**Figure 6: Biased SFT data primarily shifts the agents of underrepresented groups.** Left: Candidates and their agents in smartvote PCA space. Right: Group-mean displacement vectors, or arrows showing each group’s average shift from candidate to agent position. As with the F1 scores in Figure 5, biased data mainly affects the underrepresented group. Right candidates again experience stronger effects than Left candidates. Conservatively biased and default SFT data produce notably similar displacement patterns. This suggests that the centerward shift under default data may come from the base model’s alignment rather than the SFT data.

### 4.9 Performance on synthetic positions

To assess how an individual’s preferences affect their agent’s performance, we create a counterfactual: an altered dataset in which every smartvote answer is inverted. We train Llama-3.1-8B on this dataset. Inversion roughly corresponds to a \(180^\circ\) rotation around the origin in Figure 2 (see Section E.6). For example, it converts right-wing positions into left-wing ones.

For each politician, we compute the F1 difference before and after inversion, as Figure 7 shows. Right group members benefit from inversion, while the Left and Center groups perform worse on average. Yet the inverted Right still does not reach the Left’s original F1. Meanwhile, the inverted Left drops to roughly the Right’s original level.

If the effect came solely from the model’s original political bias, flipping labels should have made right-leaning candidates nearly as easy to predict as left-leaning ones. This does not happen. The result suggests that left-leaning preference profiles may be inherently easier to model, a possibility that warrants further study.

## 5 Discussion

##### Limitations.

Noise inherently limits the learning of human opinions. Even the same respondent might answer a question differently on different days. Consistently achieving an F1 score close to \(1.0\) is therefore unlikely.

Our analyses cover only a small number of individuals or parties. The main reason is computational cost: the method trains one model for each profile.

Data impose another constraint. Suitable surveys are scarce. Existing surveys often contain too few items to train individual agents, or they address general themes rather than specific issues and priorities. Question sets also rarely match across countries, limiting comparisons.

##### Future work.

The differences between political groups warrant more research on reducing bias. Future work should aim to replace training a separate model for each agent with a single model conditioned on compact persona representations, or short encodings of individual profiles (for example, Ning et al., 2024). This could reduce computational cost and make better use of data.

Real-world use will require confidence estimates to assess answer reliability. Researchers could obtain these by extending the reasoning scheme or adding a dedicated value head, a model component that estimates value. Purpose-built questionnaires and better handling of uncertainty may also reduce the difficulties with Neutral.

**Figure 7: Learning Right and Center positions is harder than Left ones.** Points show F1 scores on the original data. The horizontal axis orders individuals by their first component in smartvote PCA space. The left-most Right candidate is correctly placed (see Figure 6). Arrows show the change in F1 after flipping all answers in the training and test sets. Left candidates perform worse after flipping, both individually and on average. Right candidates improve but still perform worse on average than Left candidates. Converting Left to Right does not remove the disparity, which could suggest that Right positions are harder to learn.

## 6 Conclusion

We explore individual opinion alignment by training persona-conditioned agents directly on known opinions and treating answer generation as a structured reasoning task. Across Swiss candidates in smartvote, German parties in Wahl-o-Mat, and U.S. voters in ANES, SFT+GRPO generally outperforms all baselines.

Performance remains uneven. Neutral stances remain hard to predict, and accuracy differs significantly across the political spectrum. Our work provides a strong benchmark and baseline for systematic opinion modeling with LLMs. It marks an early step toward potentially more representative, AI-driven forms of democracy.

## Ethics statement

Modeling individual opinions is not exclusively beneficial, especially as misinformation and disinformation rise. It may create downstream risks. Precisely because of these risks and the technology’s inevitable development, we believe responsible research must investigate how to improve individual opinion alignment.

We use only anonymized survey data and public figures’ positions to address privacy concerns. Our research reveals systematic performance differences across ideological groups. Models consistently perform worse on center and right-wing profiles and struggle with neutral positions.

We commit to reporting these biases transparently. They pose significant system risks that could disenfranchise certain political viewpoints if the systems were deployed prematurely. We strongly caution against real-world applications until these fairness issues are resolved. Biased political models risk undermining democratic representation.

Ultimately, we remain convinced that with proper bias mitigation, opinion modeling can support socially beneficial applications. These include digital democracies, where models could help shape fairer and more popular policies.

## Reproducibility statement

We release our code and datasets to support reproducibility:

https://github.com/ETH-DISCO/reasoning-boosts-llm-alignment

Appendix F gives training details, including hyperparameters. Section E.5 describes the details for all PCA results.

#### Author Contributions

Frédéric Berdoz conceived and designed the project, led experimental planning, and defined its research direction. Yann Billeter contributed equally and led technical development, including dataset collection, model training, evaluation, and analysis. Yann Vonlanthen advised the project on methodology and experimental design. Roger Wattenhofer supervised the project as principal investigator.

#### Acknowledgments

We thank Prof. Dr. Hans Gersbach for his valuable insights during the project.
