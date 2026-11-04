# ACTR: Aligning Thoughts and Responses for Multilingual Safety in Reasoning LLMs

Xianhui Zhang

Affiliation: Nanjing University of Science and Technology, Nanjing, China

  
Jian Yu

Affiliation: Nanjing University of Science and Technology, Nanjing, China

  
Chengyu Xie

Affiliation: Nanjing University of Science and Technology, Nanjing, China

  
Chenhang Cui

Affiliation: National University of Singapore, Singapore

  
Shuyi Miao

Affiliation: Beihang University, Beijing, China

  
Pengyang Shao

Affiliation: National University of Singapore, Singapore

  
Yu Zheng

Affiliation: Nanjing University of Science and Technology, Nanjing, China

  
Fei Shen,
Tat-Seng Chua

Affiliation: National University of Singapore, Singapore

###### Abstract

Ensuring the safety of reasoning large language models (LLMs) across languages is essential for their reliable deployment.
However, when exposed to jailbreak attacks in non-high-resource languages, these models may generate unsafe responses even when their reasoning traces identify safety risks.
To address this issue, we propose aligning cross-lingual thoughts and responses (ACTR), a framework that improves multilingual safety alignment by strengthening the use of existing safety reasoning.
Specifically, we first present the think gap score (TGS) to compare the normalized contributions of reasoning traces to attention outputs during response generation across languages, and use reasoning-trace substitution to measure the cross-lingual safety gap.
Next, using a corpus of jailbreak queries, we assess neuron importance through changes in response representations caused by neuron masking and compare the high-importance neuron sets obtained with reasoning enabled and disabled to identify safety think neurons that support the use of safety reasoning.
Finally, we devise neuron-selective consistency optimization (NSCO), which uses a frozen judge model to reward agreement between the safety categories of reasoning traces and responses while updating only the parameters associated with the selected neurons, requiring no human-annotated responses or preference data.
Across two reasoning models, ACTR achieves lower average attack success rates than the evaluated state-of-the-art methods on AdvBench-X and MultiJail, with safety gains extending to unseen languages, while preserving or improving average performance on multilingual knowledge and mathematical reasoning tasks and limiting false refusals of benign requests.
Warning: this paper contains examples with unsafe content.

 
 

|

**footnotetext: Corresponding author.

## 1 Introduction

Reasoning large language models (LLMs) (Wei et al., 2022; Guo et al., 2025; Yoon et al., 2026) have made substantial progress on complex tasks by generating intermediate reasoning traces before their final responses.
These traces (Li et al., 2025; Wang et al., 2025a) offer an opportunity to recognize risks and formulate safety judgments, whose practical value depends on whether they effectively guide the final response.
Ensuring that such judgments remain effective across languages (Yang et al., 2025b) is therefore central to the safety alignment and reliable deployment of reasoning LLMs.

Existing LLM safety alignment (Ouyang et al., 2022; Rafailov et al., 2023; Bai et al., 2022) primarily shapes response behavior through supervised fine-tuning, preference learning, and reward optimization.
Research on reasoning LLMs (Guan et al., 2024; Wang et al., 2025b) further incorporates safety specifications into the reasoning process to guide risk recognition and safety judgments before answering.
In multilingual settings, existing work (Shen et al., 2024; Zhang et al., 2025) explores multilingual safety supervision and the transfer of safety knowledge from high-resource languages to extend alignment coverage.
However, as illustrated in Figure 1, a jailbreak prompt in a non-high-resource (NHR) language (Deng et al., 2024; Yong et al., 2023) may elicit English reasoning that identifies safety risks, yet still lead to unsafe response in the target language.
This disconnect shows that recognizing risks during reasoning does not ensure safe response generation, motivating alignment methods (Wei et al., 2023; Zou et al., 2023a; Qi et al., 2024a) that strengthen the use of existing safety judgments across languages.

Figure 1: Cross-lingual Thought–Response Safety Disconnect.
Models may identify risks in English reasoning yet produce unsafe responses in NHR languages.

To address this issue, we propose aligning cross-lingual thoughts and responses (ACTR), a framework that improves multilingual safety alignment by strengthening the use of existing safety reasoning.
We first introduce the think gap score (TGS) to quantify cross-lingual differences in the normalized contribution of reasoning traces to attention outputs during response generation.
Reasoning-trace substitution further tests whether reasoning content explains the safety gap: replacing NHR reasoning traces with those elicited by corresponding English prompts leaves NHR attack success rates largely unchanged.
We then assess neuron importance through changes in response representations caused by masking and compare the high-importance neuron sets obtained with reasoning enabled and disabled under matched prompts to identify safety think neurons that support the use of reasoning information.
Finally, ACTR applies neuron-selective consistency optimization (NSCO), which uses a frozen judge model to assign a single reward based on agreement between the safety categories of reasoning traces and final responses.
The proposed ACTR updates only the parameters associated with the selected safety think neurons and requires no human-annotated responses or preference data.
The main contributions are summarized as follows:

•

We present TGS to quantify cross-lingual gaps in reasoning contributions and combine reasoning-trace substitution with neuron interventions to characterize disconnects between safety reasoning and responses, identifying safety think neurons supporting reasoning use.

•

We propose ACTR and its NSCO strategy, which combines a model-judged thought-response consistency reward with selective updates to think-neuron parameters, enabling multilingual safety alignment without human-annotated responses or preference data.

•

Across two reasoning models, ACTR achieves lower average attack success rates than the state-of-the-art methods on AdvBench-X and MultiJail, while generalizing to unseen languages and preserving average task performance with low refusal rates on benign requests.

## 2 Related Work

Safety Alignment in Reasoning LLMs.
Safety alignment methods such as RLHF (Ouyang et al., 2022) and DPO (Rafailov et al., 2023) improve instruction-following and harmlessness, but primarily supervise final responses.
Reasoning-oriented LLMs (Yang et al., 2025a; Team, 2026), including OpenAI o1 (Jaech et al., 2024) and DeepSeek-R1 (Guo et al., 2025), generate explicit intermediate reasoning, and deliberative alignment (Guan et al., 2024) shows that reasoning over safety specifications can improve safety.
Nevertheless, models aligned mainly in English (Yong et al., 2023; Zou et al., 2023b) remain vulnerable to prompts in non-high-resource languages.
Existing defenses rely on multilingual supervision, data augmentation, or cross-lingual objectives, often requiring high-quality multilingual or parallel corpora that are costly and difficult to obtain.

Mechanistic Interpretability of Safety Behavior.
Mechanistic interpretability (Geva et al., 2021; Meng et al., 2022; Arditi et al., 2024; Olah et al., 2020; Elhage et al., 2021) has identified circuits, attention heads, and neurons associated with factual recall, refusal, and other safety behaviors.
Activation steering and probing (Turner et al., 2024; Zou et al., 2023a) further show that localized representations can influence model behavior.
However, most studies examine single-stage generation.
Reasoning LLMs instead generate responses after an intermediate think process, whose interaction with multilingual output remains poorly understood.
Although multilingual chain-of-thought studies (Shi et al., 2022) show that reasoning and answer languages can interact, we investigate the think-response interface and identify neurons underlying its cross-lingual safety gap.

Reinforcement Learning for Safety.
Reinforcement learning methods such as GRPO (Shao et al., 2024) improve reasoning and alignment through reward optimization.
Existing safety methods (Ouyang et al., 2022; Bai et al., 2022) typically reward complete responses, as in RLHF and Constitutional AI, which can cause reward overoptimization (Gao et al., 2023; Röttger et al., 2024), excessive refusal, or reduced utility.
In multilingual reasoning, a model (Yong et al., 2023; Guan et al., 2024) may produce safety-oriented thoughts but an inconsistent final response.
However, optimizing final responses alone does not ensure that safety judgments formed during reasoning are effectively incorporated into multilingual responses.

Figure 2: Overview of our mechanistic analysis and ACTR framework.
Top: TGS measures cross-lingual reasoning-utilization gaps; think-on/off comparisons localize safety think neurons.
Bottom: NSCO uses consistency rewards from a frozen judge to selectively optimize these neurons’ parameters.

## 3 Mechanistic Analysis of the Cross-lingual Gap

### 3.1 The “Think-Response” Attention Disconnect

Models and Languages.
We investigate two instruction-tuned large language models, Qwen3-8B (Yang et al., 2025a) and Gemma4-12B-it (Team, 2026), across seven languages: English (EN ), Chinese (ZH ), Korean (KO ), Thai (TH ), Afrikaans (AF ), Nepali (NE ), and Bengali (BN ). English is treated as the high-resource (HR) language, whereas the remaining languages are treated as non-high-resource (NHR) languages.

Motivation and Hypotheses. Reasoning LLMs often generate their internal reasoning traces in English, even when prompted in NHR languages. Nevertheless, they remain substantially more susceptible to jailbreak attacks in NHR languages than in HR languages. This discrepancy suggests two possible explanations: NHR prompts may elicit less safety-aware reasoning traces, or the model may fail to effectively incorporate otherwise safety-aware reasoning into its final responses. We first examine the former possibility through a
controlled reasoning-trace substitution experiment.

Controlled Reasoning Trace Substitution. Specifically, we conduct a controlled HR-think substitution experiment to rule out differences in the content or quality of the generated reasoning traces as the primary cause of the degraded safety performance in NHR languages. For each jailbreak query, we first obtain the reasoning trace elicited by its HR-language version and then replace the NHR-generated trace with the one elicited by its HR counterpart. This intervention holds the reasoning content constant across the HR and NHR conditions, allowing us to isolate whether inferior NHR reasoning traces account for the observed safety gap. If they were the primary cause, replacing them with HR-derived traces should substantially reduce the discrepancy in attack success rate (ASR) (Qi et al., 2024b) between HR and NHR languages. ASR calculation details are in Appendix  C.

Table 1: English Reasoning-Trace Substitution.
ASR on MultiJail with original (Default) vs. English-elicited traces (EN Think).

| | Qwen3-8B | Gemma4-12B-it

| Language | Default | EN Think | Default | EN Think

| EN | 11.75 | 9.21 | 4.76 | 4.13

| ZH | 9.21 | 10.48 | 6.67 | 6.03

| KO | 14.92 | 15.24 | 8.25 | 8.57

| TH | 13.33 | 11.43 | 6.98 | 6.98

| AF | 15.56 | 14.92 | 9.84 | 8.89

| NE | 27.62 | 27.94 | 7.62 | 6.67

| BN | 15.87 | 16.19 | 11.11 | 10.79

| Avg. | 15.46 | 15.06 | 7.89 | 7.44

Substitution Results.
Table 1 reports ASR before and after substitution on MultiJail (Deng et al., 2024) across seven languages.
Despite receiving the same English reasoning trace generated from the HR version of each jailbreak query, NHR ASR remains largely unchanged after substitution.
Specifically, the seven-language mean ASR decreases by only 0.40 and 0.45 percentage points for Qwen3-8B and Gemma4-12B-it, respectively.
This result indicates that the multilingual safety gap cannot be explained solely by the quality or content of the generated reasoning trace.
Instead, it points to a disconnect between the reasoning trace and response generation in NHR settings.

Quantifying Reasoning Trace Utilization. To quantify this utilization gap, we introduce the think gap score (TGS), which explicitly captures how much attention the final response allocates back to the reasoning steps. For each paired conversation $x\in\{\mathrm{HR},\mathrm{NHR}\}$ , let $\mathcal{I}_{T}^{x}$ and $\mathcal{I}_{A}^{x}$ denote the token positions of the reasoning trace and the final answer, respectively. We consider the latter half of the model’s $L$ layers, indexed from zero:

|
|

$\displaystyle\mathcal{L}_{\mathrm{half}}=\left\{\left\lfloor\frac{L}{2}\right\rfloor,\ldots,L-1\right\}.$

| | (1)

For an answer token at position $i$ , we use the attention
row at position $i-1$ , which predicts that token.
Let $\mathcal{K}_{i}^{x,\ell}$ denote the key positions
visible to this row under the attention mask of layer $\ell$ .
For any token-position set $\mathcal{S}$ , define its
contribution to the projected attention output as

|
|

$\displaystyle\begin{aligned} \mathbf{o}_{i,\mathcal{S}}^{x,\ell}={}&W_{O}^{\ell}\operatorname{Concat}_{h=1}^{H}\left(\sum_{k\in\mathcal{S}\cap\mathcal{K}_{i}^{x,\ell}}A_{h,i-1,k}^{x,\ell}\mathbf{V}_{h,k}^{x,\ell}\right),\end{aligned}$

| | (2)

where $A_{h,i-1,k}^{x,\ell}$ is the attention weight,
$\mathbf{V}_{h,k}^{x,\ell}$ is the value vector associated
with attention head $h$ , and $W_{O}^{\ell}$ is the output
projection matrix, with its bias omitted.
For grouped-query attention, value vectors are shared
across the corresponding query heads.
We measure the reasoning trace’s normalized contribution
using the ratio of its projected output energy to the
projected output energy from all visible tokens:

|
|

$\displaystyle r_{i,\ell}^{x}=\frac{\left\|\mathbf{o}_{i,\mathcal{I}_{T}^{x}}^{x,\ell}\right\|_{2}^{2}}{\left\|\mathbf{o}_{i,\mathcal{K}_{i}^{x,\ell}}^{x,\ell}\right\|_{2}^{2}+\epsilon},$

| | (3)

where $\epsilon>0$ ensures numerical stability.
Averaging these ratios over answer positions and the
selected layers gives

|
|

$\displaystyle R_{\mathrm{think}}^{x}=\frac{1}{|\mathcal{I}_{A}^{x}||\mathcal{L}_{\mathrm{half}}|}\sum_{\ell\in\mathcal{L}_{\mathrm{half}}}\sum_{i\in\mathcal{I}_{A}^{x}}r_{i,\ell}^{x}.$

| | (4)

Finally, we define the think gap score for a paired
HR/NHR conversation as

|
|

$\displaystyle\mathrm{TGS}^{\mathrm{NHR}\mid\mathrm{HR}}=R_{\mathrm{think}}^{\mathrm{HR}}-R_{\mathrm{think}}^{\mathrm{NHR}}.$

| | (5)

A positive TGS indicates a lower average normalized
think-value contribution during answer prediction
in the NHR conversation than in its paired HR conversation (see proof in Appendix B).

Figure 3: Reasoning Utilization and Multilingual Safety.
Comparison of the think gap score (TGS) and attack success rate (ASR) across languages.

The Think–Response Attention Disconnect.
To examine whether reduced safety accompanies weaker utilization of reasoning traces, we plot TGS as bars alongside benchmark ASR as lines for each language and model (Figure 3).
Across both models, larger TGS generally accompanies higher ASR, although this relationship is not strictly monotonic.
For example, Nepali in Qwen3-8B and Bengali in Gemma4-12B-it exhibit both the highest TGS and the highest MultiJail ASR among the evaluated languages.
Together with the substitution results, these patterns suggest a think–response attention disconnect as a possible contributor to the multilingual safety gap.
HR-think substitution leaves NHR ASR largely unchanged, while TGS reveals cross-lingual differences in the contribution of reasoning traces to attention outputs during response generation.
English-dominant reasoning may therefore be insufficient when its safety-relevant information is not effectively incorporated into the final NHR response.
This motivates targeted neuron interventions to test whether disrupting reasoning utilization also weakens multilingual safety.

Table 2: Think–Response Inconsistency Rate (%).
The proportion of examples with safe think traces but unsafe responses
on MultiJail and AdvBench-X.

| | Qwen3-8B | Gemma4-12B-it

| Language | MultiJail | AdvBench-X | MultiJail | AdvBench-X

| EN | 0.24 | 0.00 | 1.90 | 0.00

| ZH | 2.54 | 0.00 | 3.17 | 0.38

| KO | 3.81 | 3.81 | 5.40 | 1.35

| TH | 2.86 | 2.54 | 4.44 | 0.38

| AF | 4.76 | 4.44 | 6.03 | 1.35

| BN | 5.71 | 8.89 | 6.67 | 1.73

| NE | 6.67 | 13.97 | 3.81 | 0.77

To further quantify the think–response mismatch, we measure the inconsistency rate for Qwen3-8B and Gemma4-12B-it, defined as the proportion of examples in which the think trace is classified as safe while the final response is classified as unsafe. We report this rate separately on MultiJail and AdvBench-X. As shown in Table 2, the inconsistency rate generally increases for non-English languages. On MultiJail, it rises from only 0.24% in English to 6.67% in Nepali for Qwen3-8B. A similar pattern appears on AdvBench-X, where the Qwen3-8B rate reaches 13.97% for Nepali and 8.89% for Bengali, compared with 0% for English and Chinese. Gemma4-12B-it follows a similar trajectory, with its inconsistency rate rising from 1.90% in English to a peak of 6.67% in Bengali on MultiJail, and from 0% to 1.73% on AdvBench-X. These results indicate that a safe reasoning trace does not always lead to a safe final response, especially in lower-resource languages where the model’s safety guardrails are inherently more fragile. Thus, the multilingual safety gap may partly arise from a failure to transfer safety relevant information from the reasoning trace into the response, providing additional evidence for the proposed think–response attention disconnect and highlighting a critical vulnerability in current cross-lingual alignment strategies.

### 3.2 Identifying and Defining Safety Think Neuron

Prior work (Dalvi et al., 2019; Meng et al., 2022) shows that individual neurons and sparse feature groups can encode interpretable concepts and influence model outputs, including safety responses and refusal behavior. Motivated by this ability of local structures to steer global model behavior, we hypothesize that some neurons serve as continuous mediators that preserve and propagate safety-relevant information from the initial internal think span to subsequent response tokens. We call these units safety think neurons. This section presents our neuron-level detection method and examines whether they exhibit distinct activation patterns under HR and NHR conditions, aiming to empirically validate their functional role in the reasoning pipeline.

Paired Think/No-Think Probing Dataset.
We construct a separate multilingual probing dataset containing 700 jailbreak queries sampled from BeaverTails (Ji et al., 2023). For each query, we obtain its multilingual variants using Google Translate and retain only translations that pass an LLM-based semantic consistency check. To prevent data leakage, all queries are deduplicated against the evaluation examples in AdvBench-X and MultiJail, ensuring that none of the probing queries appears in either benchmark. For each query–language pair $q_{j}$ , we run the same model twice under identical prompt, system, and decoding settings. The only difference is whether think mode is enabled. The resulting model trajectories are denoted by

|
|

$\displaystyle s_{j}^{\mathrm{on}}=\left(q_{j},t_{j},a_{j}^{\mathrm{on}}\right),\qquad s_{j}^{\mathrm{off}}=\left(q_{j},a_{j}^{\mathrm{off}}\right),$

| | (6)

where $t_{j}$ is the reasoning trace generated in think mode, and $a_{j}^{\mathrm{on}}$ and $a_{j}^{\mathrm{off}}$ are the corresponding responses. We retain all generated responses, including safe responses and unsafe responses. This paired construction holds the jailbreak query, language, and model constant while varying the availability of the reasoning trace. The resulting contrast therefore targets neurons involved in using the think span during response generation.

Identification Strategy.
We define a neuron as a candidate think neuron if it is activated when think mode is enabled but not activated when think mode is disabled. Such neurons may be responsible for transmitting or using information from the model’s reasoning process.
At layer $\ell$ , we define a neuron $N$ as a specific row in one of the query, key, or value projection matrices, $\mathbf{W}_{Q}^{\ell}$ , $\mathbf{W}_{K}^{\ell}$ , or $\mathbf{W}_{V}^{\ell}$ , or as a specific column in the output projection matrix $\mathbf{W}_{O}^{\ell}$ .
For each trajectory $s_{j}^{m}$ , where $m\in\{\mathrm{on},\mathrm{off}\}$ , we identify the response-token positions $\mathcal{I}_{A,j}^{m}$ . Let $\mathbf{h}_{\theta}(s_{j}^{m},i)$ denote the output representation used to predict the response token at position $i$ . We then deactivate neuron $N$ during response generation and obtain the corresponding representation $\mathbf{h}_{\theta,-N}(s_{j}^{m},i)$ . The intervention is applied only at response-token positions, so that the resulting score measures the contribution of $N$ to response generation.
The importance of neuron $N$ for example $j$ under mode $m$ is quantified by the average representational shift:

|
|

$\displaystyle\Delta_{j}^{m}(N)=\frac{1}{\left|\mathcal{I}_{A,j}^{m}\right|}\sum_{i\in\mathcal{I}_{A,j}^{m}}\left\|\mathbf{h}_{\theta}(s_{j}^{m},i)-\mathbf{h}_{\theta,-N}(s_{j}^{m},i)\right\|_{2}^{2}.$

| | (7)

We aggregate this quantity across these paired queries to obtain the mode-specific importance score:

|
|

$\displaystyle I_{\ell}^{m}(N)=\frac{1}{\left|\mathcal{D}_{\mathrm{probe}}\right|}\sum_{j=1}^{\left|\mathcal{D}_{\mathrm{probe}}\right|}\Delta_{j}^{m}(N),$

| | (8)

where $\mathcal{D}_{\mathrm{probe}}$ denotes the same set of queries evaluated in both Think-on and Think-off modes.
For each layer, we select the top $p$ percent of neurons under each mode:

|
|

$\displaystyle\mathcal{S}_{\ell}^{m}=\operatorname{Top}_{p\%}\left\{N\in\mathcal{N}_{\ell}\;\middle|\;I_{\ell}^{m}(N)\right\},$

| | (9)

where $\mathcal{N}_{\ell}$ is the set of safety think neurons at layer $\ell$ . We set $p=3$ to obtain a sparse set of highly influential candidate units.
We then identify safety think neurons through mode-contrastive subtraction:

|
|

$\displaystyle\mathrm{TN}_{\ell}=\mathcal{S}_{\ell}^{\mathrm{on}}\setminus\mathcal{S}_{\ell}^{\mathrm{off}}.$

| | (10)

The resulting set comprises neurons highly influential when the reasoning trace is present but not in the matched Think-off condition. We therefore treat them as candidate safety think neurons for subsequent functional and causal analyses.

Table 3: Effects of Masking Safety Think Neurons.
Mean ASR across languages under random (R-Masking) and targeted (TN-Masking) neuron masking.

| AdvBench-X

| Model | Default | R-Masking | TN-Masking

| Qwen3-8B | 10.27 | 10.66+0.38 | 16.32+6.04

| Gemma4-12B-it | 2.30 | 3.43+1.13 | 40.52+38.22

| MultiJail

| Model | Default | R-Masking | TN-Masking

| Qwen3-8B | 15.46 | 15.87+0.41 | 31.07+15.60

| Gemma4-12B-it | 7.89 | 9.43+1.54 | 51.79+43.90

Causal Effect of Think-Neuron Masking.
We next test whether the identified safety think neurons causally sustain the safety-relevant effect of the reasoning trace. For each model and language, we compare the unmasked default with two interventions: masking the detected safety think neurons and masking a control set of neurons matched for count and layer-wise distribution.

Table 3 summarizes the ASR results. For Qwen3-8B, targeted masking raises ASR by 6.04 and 15.60 percentage points on AdvBench-X and MultiJail, respectively, whereas matched random masking changes it by only 0.38 and 0.41 points. This selective degradation indicates that the detected safety think neurons causally support model safety, beyond effects attributable to mask size or layer distribution. Additionally, we present qualitative examples of the model outputs before and after masking safety think neurons in Appendix F.

Figure 4: Effects of Masking Safety Think Neurons on TGS.
$\Delta$ TGS denotes the change relative to unmasked trajectories.
Targeted masking increases TGS more than random masking.

Effects on Reasoning Utilization. Figure 4 reports TGS changes from baseline.
Because higher TGS indicates less attention to think content, the consistent increase after think-neuron masking across all NHR languages (mean $+0.696$ ; largest in Thai, $+1.110$ ) shows reduced reliance on preceding reasoning. Matched random masking causes negligible change (mean $-0.045$ ), ruling out the possibility that this effect is driven solely by the number or layer distribution of masked neurons.
Together, the ASR and TGS results show that safety think neurons help connect the think span to the final response by carrying safety-related information from the reasoning process into response generation. Disrupting these neurons reduces the model’s use of this reasoning and makes it more vulnerable to jailbreak attacks.

### 3.3 The ACTR Framework

Motivation.

Building on our mechanistic findings, we propose ACTR, a parameter-efficient reinforcement learning framework that improves multilingual safety alignment by strengthening the use of existing safety reasoning.
Given a multilingual prompt $x$ , the policy generates $y=(t,a)$ , where $t$ denotes the think span and $a$ the final response.
ACTR applies neuron-selective consistency optimization (NSCO), which uses a frozen judge model to reward thought–response safety consistency while updating only the parameters associated with the identified safety think neurons.

Neuron-Selective Consistency Optimization.
Under neuron-selective consistency optimization (NSCO), the consistency rewards are normalized within each sampled group to obtain the relative consistency advantage:
$y_{i}=(t_{i},a_{i})\sim\pi_{\theta_{\mathrm{old}}}(\cdot\mid x)$ ,
where $i=1,\ldots,G$ . A frozen binary
consistency judge $C_{\phi}$ evaluates each trajectory. We define the
sequence-level consistency reward as

| | $r_{i}^{\mathrm{con}}=C_{\phi}(t_{i},a_{i})\in\{0,1\},$
| | (11)

where $r_{i}^{\mathrm{con}}=1$ indicates that the behavioral stance or
action expressed in the response $a_{i}$ is consistent with that implied
by the think span $t_{i}$ , while $r_{i}^{\mathrm{con}}=0$ indicates
inconsistency. The judge evaluates semantic and behavioral agreement
rather than surface-form similarity.
The consistency rewards within each group are normalized to derive the relative advantages:

|
|

$\displaystyle\widehat{A}_{i}=\frac{r_{i}^{\mathrm{con}}-\mu_{r}}{\sigma_{r}+\epsilon_{\mathrm{adv}}},$

| | (12)

where $\mu_{r}$ and $\sigma_{r}$ are the mean and standard deviation of the
$G$ rewards, respectively. Let $s_{i,k}=(x,y_{i,<k})$ denote the
decoding state at token position $k$ , and let
$\rho_{i,k}=\pi_{\theta}(y_{i,k}\mid s_{i,k})/\pi_{\theta_{\mathrm{old}}}(y_{i,k}\mid s_{i,k})$
denote the corresponding likelihood ratio.
Our ACTR framework optimizes:

|
|

$\displaystyle\begin{aligned} \mathcal{J}_{\mathrm{ACTR}}(\theta_{\mathrm{TN}})&=\mathbb{E}_{i,k}\left[\ell_{i,k}^{\mathrm{clip}}-\beta D_{\mathrm{KL}}\left(\pi_{\theta}\,\middle\|\,\pi_{\mathrm{ref}}\right)\right],~~\ell_{i,k}^{\mathrm{clip}}&=\min\left(\rho_{i,k}\widehat{A}_{i},\operatorname{clip}\left(\rho_{i,k},1-\epsilon,1+\epsilon\right)\widehat{A}_{i}\right).\end{aligned}$

| | (13)

Unlike standard full-parameter optimization, ACTR applies gradients
only to the parameters associated with the identified safety think neurons:

|
|

$\displaystyle\nabla_{\theta}\mathcal{J}_{\mathrm{ACTR}}\leftarrow M_{\mathrm{TN}}\odot\nabla_{\theta}\mathcal{J}_{\mathrm{ACTR}}.$

| | (14)

Here, $M_{\mathrm{TN}}$ is a binary mask that selects the parameters associated with the safety think neurons.
Thus, only $\theta_{\mathrm{TN}}$ is updated, while the remaining backbone parameters, the reference policy $\pi_{\mathrm{ref}}$ , and the consistency judge $C_{\phi}$ remain frozen throughout training.

## 4 Experiments

Training Details.
We train ACTR independently on two instruction-tuned reasoning LLMs: Qwen3-8B (Yang et al., 2025a) and Gemma4-12B-it (Team, 2026).
For each model, training covers six NHR languages: Chinese, Korean, Thai, Afrikaans, Nepali, and Bengali.
We use the multilingual probing dataset introduced in Section 3.2.
Only the default jailbreak queries serve as training prompts; the current rollout policy samples a new group of $G$ thought–response trajectories for each query.
Model-specific hyperparameters are provided in Appendix D.
For fair comparison, we use matched training settings across the compared training-based methods.
We periodically evaluate each model during training and select its best-performing checkpoint based on validation results.

Evaluation and Metrics.

We use GPT-4o-mini as a consistency reward model to evaluate sampled trajectories online and assign binary rewards.
A trajectory receives a positive reward when its final response agrees with the safety stance expressed in the think span, regardless of whether the response itself is safe.
The judging criteria are detailed in Appendix E.
For safety evaluation, we use attack success rate (ASR), with detailed evaluation protocols provided in Appendix C.

### 4.1 Main Results

Table 4: Comparison of Attack Success Rate (ASR $\downarrow$ , %) on AdvBench-X and MultiJail.
ACTR achieves the lowest average ASR across both benchmarks and model families among the currently evaluated methods.

| | AdvBench-X | MultiJail

| Method | EN | ZH | KO | BN | Avg. | EN | ZH | KO | BN | Avg.

| Qwen3-8B | 4.42 | 1.54 | 9.80 | 15.38 | 7.79 | 11.75 | 9.21 | 14.92 | 15.87 | 12.94

| SFT | 0.38-4.04 | 0.58-0.96 | 1.15-8.65 | 1.54-13.84 | 0.91-6.88 | 4.76-6.99 | 2.86-6.35 | 6.98-7.94 | 13.33-2.54 | 6.98-5.96

| DPO | 2.69-1.73 | 0.77-0.77 | 7.12-2.68 | 21.35+5.97 | 7.98+0.19 | 5.08-6.67 | 2.54-6.67 | 9.52-5.40 | 15.40-0.47 | 8.14-4.80

| MPO | 0.58-3.84 | 0.77-0.77 | 3.07-6.73 | 4.76-10.62 | 2.30-5.49 | 3.17-8.58 | 1.59-7.62 | 2.85-12.07 | 6.35-9.52 | 3.49-9.45

| Self Defense | 0.60-3.82 | 0.80-0.74 | 4.00-5.80 | 16.20+0.82 | 5.40-2.39 | 3.50-8.25 | 4.40-4.81 | 7.00-7.92 | 16.80+0.93 | 7.93-5.01

| SmoothLLM | 0.40-4.02 | 0.58-0.96 | 4.80-5.00 | 9.20-6.18 | 3.75-4.04 | 5.08-6.67 | 1.00-8.21 | 4.10-10.82 | 6.70-9.17 | 4.22-8.72

| TrajGuard | 2.31-2.11 | 1.54-0.00 | 9.62-0.18 | 14.62-0.76 | 7.02-0.77 | 4.13-7.62 | 2.86-6.35 | 11.75-3.17 | 20.95+5.08 | 9.92-3.02

| GUARD-SLM | 0.38-4.04 | 0.77-0.77 | 4.42-5.38 | 7.69-7.69 | 3.32-4.47 | 4.76-6.99 | 2.54-6.67 | 10.16-4.76 | 17.14+1.27 | 8.65-4.29

| Ours | 0.19-4.23 | 0.38-1.16 | 0.19-9.61 | 0.38-15.00 | 0.29-7.50 | 0.63-11.12 | 0.95-8.26 | 1.90-13.02 | 4.13-11.74 | 1.90-11.04

| Gemma4-12B-it | 0.96 | 1.34 | 2.11 | 3.46 | 1.97 | 4.76 | 6.67 | 8.25 | 11.11 | 7.70

| SFT | 0.58-0.38 | 1.15-0.19 | 1.92-0.19 | 3.08-0.38 | 1.68-0.29 | 5.40+0.64 | 2.86-3.81 | 4.76-3.49 | 3.81-7.30 | 4.21-3.49

| DPO | 0.77-0.19 | 0.96-0.38 | 2.31+0.20 | 2.31-1.15 | 1.59-0.38 | 3.49-1.27 | 4.76-1.91 | 4.76-3.49 | 3.17-7.94 | 4.05-3.65

| MPO | 0.38-0.58 | 0.58-0.76 | 1.73-0.38 | 2.69-0.77 | 1.35-0.62 | 2.22-2.54 | 2.53-4.14 | 1.27-6.98 | 1.59-9.52 | 1.90-5.80

| Self Defense | 0.58-0.38 | 1.70+0.36 | 3.50+1.39 | 2.69-0.77 | 2.12+0.15 | 3.80-0.96 | 5.40-1.27 | 5.70-2.55 | 3.20-7.91 | 4.53-3.17

| SmoothLLM | 0.96-0.00 | 0.60-0.74 | 1.73-0.38 | 2.30-1.16 | 1.40-0.57 | 1.00-3.76 | 3.50-3.17 | 3.80-4.45 | 1.59-9.52 | 2.47-5.23

| TrajGuard | 1.15+0.19 | 2.12+0.78 | 2.50+0.39 | 1.73-1.73 | 1.88-0.09 | 5.71+0.95 | 4.76-1.91 | 5.08-3.17 | 4.76-6.35 | 5.08-2.62

| GUARD-SLM | 0.38-0.58 | 0.96-0.38 | 1.54-0.57 | 1.92-1.54 | 1.20-0.77 | 1.59-3.17 | 2.53-4.14 | 1.59-6.66 | 1.90-9.21 | 1.90-5.80

| Ours | 0.20-0.76 | 0.00-1.34 | 1.35-0.76 | 1.35-2.11 | 0.72-1.23 | 1.27-3.49 | 0.95-5.72 | 0.63-7.62 | 1.27-9.84 | 1.03-6.67

Table 4 compares ACTR with representative alignment
and inference-time defense methods, including SFT, DPO (Rafailov et al., 2023),
MPO (Zhao et al., 2025), Self Defense (Phute et al., 2024), SmoothLLM (Robey et al., 2024), and the recently proposed
TrajGuard (Liu et al., 2026) and GUARD-SLM (Mia et al., 2026). Since TrajGuard and GUARD-SLM employ refusal-based
defenses, we treat an explicit refusal as a safe outcome when computing their
ASR; for other methods, safety is determined directly from the model’s
generated response. Under this evaluation protocol, ACTR achieves the lowest
average ASR for both model families on both benchmarks. For Qwen3-8B, ACTR
reduces the average ASR to $0.29\%$ and $1.90\%$ on AdvBench-X and MultiJail,
compared with $0.91\%$ and $3.49\%$ for the strongest competing methods. For
Gemma4-12B-it, ACTR further achieves $0.72\%$ on AdvBench-X (Yong et al., 2023) and $1.03\%$ on
MultiJail (Deng et al., 2024), improving over the strongest baselines, GUARD-SLM on AdvBench-X and
the tied MPO/GUARD-SLM baselines on MultiJail, by $40.0\%$ and $45.8\%$ ,
respectively.
Overall, these results demonstrate that ACTR provides robust improvements
across model architectures, benchmarks, and languages, suggesting that
targeted think-neuron updates more effectively connect internal safety
reasoning with the final response than existing alignment and inference-time
defense methods.

### 4.2 Ablation study

Figure 5: Effect of Selection Threshold $p$ .
MGSM accuracy and ASR (%) after neuron masking.

Selection Threshold $p$ . To validate $p=3$ in Eq. 9, we evaluate general capabilities on MGSM (Shi et al., 2022) and safety performance on MultiJail (Deng et al., 2024) across $p\in\{1,2,3,5,7\}$ . We aim to isolate neurons that utilize safety reasoning without compromising general utility during targeted interventions. As Figure 5 shows, the model maintains near-baseline MGSM scores when $p\leq 3\%$ . However, general capabilities degrade sharply for $p>3\%$ (e.g., at $p=5$ , MGSM drops by 83.60 for Gemma4-12B-it), indicating that higher thresholds incorrectly capture neurons essential for foundational reasoning and task execution. Thus, $p=3$ optimally isolates a sparse and highly specific set of candidate safety think neurons while reliably preserving general performance, ensuring that our downstream modifications remain minimally invasive.

Table 5: Effect of Neuron Selection.
Mean ASR ( $\downarrow$ , %) across languages after updating random neurons or safety think neurons.

| Model | Setting | AdvBench-X | MultiJail

| Qwen3-8B | Default | 11.25 | 16.08

| Random | 1.03-10.22 | 1.32-14.76

| ACTR | 0.45-10.80 | 1.27-14.81

| Gemma4-12B-it | Default | 2.31 | 7.89

| Random | 1.13-1.18 | 1.86-6.03

| ACTR | 0.93-1.38 | 1.72-6.17

Alignment Strategy.
To determine whether the safety gains come from sparse updates or the specific neurons identified by our probing procedure, we train a random-neuron baseline.
It selects the same number of neurons and uses the same update strategy, training data, optimization settings, and trainable parameter budget as ACTR.
As shown in Table 5, both strategies substantially reduce ASR relative to the original models, with only minor differences.
ACTR performs slightly better overall: for Qwen3-8B, it achieves lower ASR on AdvBench-X and comparable results on MultiJail; for Gemma4-12B-it, it obtains ASR of 0.93% and 1.72% on AdvBench-X and MultiJail, respectively.

Table 6: Refusal Behavior on XSTest.
Refusal rates (%) for benign and harmful requests under different neuron-selection strategies.

| Model | Setting | Safe refusal $\downarrow$ | Unsafe refusal $\uparrow$

| Qwen3-8B | Default | 4.40 | 77.00

| Random | 52.00+47.60 | 86.00+9.00

| ACTR | 6.40+2.00 | 89.50+12.50

| Gemma4-12B-it | Default | 4.80 | 82.00

| Random | 46.80+42.00 | 83.20+1.20

| ACTR | 4.84+0.04 | 89.20+7.20

Over-Refusal.
Low ASR may also result from excessive refusal and therefore does not fully capture a model’s ability to answer safe requests.
We further evaluate XSTest, which includes both safe and unsafe requests.
As shown in Table 6, random-neuron training raises Qwen3-8B’s safe-request refusal rate from 4.40% to 52.00%, whereas ACTR raises it only to 6.40% while achieving higher unsafe-request refusal (89.50% vs. 86.00%).
For Gemma4-12B-it, ACTR keeps safe-request refusal nearly unchanged (4.80% to 4.84%) while increasing unsafe-request refusal from 82.00% to 89.20%.
These results show that ACTR improves safety while remaining responsive to safe requests, avoiding the pronounced over-refusal observed with random-neuron training.

Table 7: Utility Preservation.
Accuracy ( $\uparrow$ , %) on MMMLU and MGSM before and after ACTR training.

|
| | MMMLU | MGSM

| Model | Setting | EN | ZH | JA | KO | TH | Avg. | EN | ZH | JA | KO | TH | Avg.

| Qwen3-8B | Default | 70.36 | 63.84 | 60.25 | 55.48 | 25.63 | 55.11 | 96.8 | 85.2 | 81.6 | 76.0 | 85.2 | 84.96

| Ours | 69.28 | 64.32 | 59.04 | 57.61 | 26.98 | 55.45 | 96.4 | 90.8 | 86.4 | 82.4 | 90.0 | 89.20

| Gemma4-12B-it | Default | 50.98 | 49.27 | 52.67 | 53.87 | 4.86 | 42.33 | 98.4 | 90.4 | 90.4 | 87.2 | 92.4 | 91.76

| Ours | 64.41 | 54.02 | 58.30 | 59.74 | 8.37 | 48.97 | 98.4 | 90.8 | 89.2 | 89.2 | 92.0 | 91.92

### 4.3 Utility Preservation

To evaluate whether safety-oriented training preserves general-purpose capabilities,
we conduct experiments on MMMLU (Hendrycks et al., 2020) and MGSM (Shi et al., 2022). MMMLU measures multilingual knowledge
and reasoning across a broad range of academic subjects, while MGSM evaluates
multilingual grade-school mathematical reasoning. For both benchmarks, we report
accuracy (%) in five languages: English, Chinese, Japanese,
Korean, and Thai.
Table 7 shows that our method does not degrade general capabilities and instead yields modest overall improvements. For Qwen3-8B (Yang et al., 2025a), the average accuracy increases from 55.11% to 55.45% on MMMLU and from 84.96% to 89.20% on MGSM. For Gemma4-12B-it, it increases from 42.33% to 48.97% on MMMLU and from 91.76% to 91.92% on MGSM.

### 4.4 Deeper Analysis

Table 8: Safety Generalization to Unseen Languages.

| Model | Setting | SW | JV | BG | VI

| Qwen3-8B | Default | 62.86 | 12.70 | 11.75 | 6.67

| Ours | 1.27 | 0.63 | 0.95 | 2.22

| Gemma4-12B-it | Default | 8.25 | 10.79 | 10.79 | 7.62

| Ours | 3.49 | 8.57 | 5.40 | 5.08

Out-of-Distribution Generalization. To test whether the safety improvements transfer beyond the languages seen during training, we evaluate both models on four held-out languages: Swahili (SW ), Javanese (JV ), Bulgarian (BG ), and Vietnamese (VI ).
None of these languages is included in the training data. Table 8 reports the ASR on MultiJail examples per language before and after training; lower values indicate safer behavior.
The results show consistent safety gains in out-of-distribution languages. Qwen3-8B’s overall ASR drops from 23.5% to 1.3%; in Swahili, ASR falls from 62.86% to 1.27%. Gemma4-12B-it shows similar improvements, reducing ASR from 9.4% to 5.6%. These results suggest that the method learns language-agnostic safety behavior rather than memorizing language-specific refusal patterns.

Figure 6: TGS Before and after ACTR Training.
ACTR lowers TGS across all evaluated NHR languages, indicating smaller cross-lingual gaps in reasoning utilization.

Training-Induced Changes in TGS.
To assess whether ACTR reduces the cross-lingual gap in internal reasoning, we compare the TGS before and after training. TGS measures the difference between English and NHR languages in the normalized contribution of think content, with lower values indicating a smaller gap. As shown in Fig. 6, ACTR reduces TGS across all six NHR languages for both models. For Qwen3-8B, mean TGS drops from 4.05 to 1.91 ( $\downarrow$ 52.7%); for Gemma4-12B-it, from 3.79 to 2.07 ( $\downarrow$ 45.3%). Combined with the safety gains, these results suggest that strengthening safety think neurons promotes internal safety reasoning in NHR responses, mitigating the think–response disconnect.

Figure 7: Thought–Response Safety Consistency after ACTR training.

Case Analysis.
Complementing the quantitative results, Figure 7 shows that ACTR resolves the thought-response disconnect seen in earlier Korean and Thai examples. Before training, models reasoned safely in English but responded unsafely in the target language. After ACTR, both the reasoning and final responses consistently express refusal. Alongside reduced TGS and ASR scores, this confirms that ACTR effectively preserves safety judgments across languages.
More results can be found in the appendix.

## 5 Conclusion

We investigate the multilingual thought–response safety disconnect in reasoning LLMs through TGS, reasoning-trace substitution, and neuron interventions, identifying safety think neurons that support safety reasoning utilization. Building on this, ACTR applies NSCO, combining a thought–response consistency reward with parameter updates restricted to these neurons, eliminating the need for human-annotated preference data.
Experiments on Qwen3-8B and Gemma4-12B-it show ACTR achieves lower average attack success rates than evaluated state-of-the-art methods on AdvBench-X and MultiJail, with safety gains generalizing to unseen languages. Furthermore, ACTR preserves general utility (MMMLU, MGSM) while maintaining low safe-request refusal rate.

### AI use statement

Generative AI tools were used to review the LaTeX formatting during manuscript preparation. To ensure consistency with the historical ASR computation methodology, GPT-4o was also used to calculate ASR. GPT-4o-mini was used as the frozen consistency judge described in Section 4. The authors are responsible for all claims, analyses, and final content.

### Ethics statement

This work studies multilingual jailbreak vulnerabilities to improve model safety. The same mechanistic interventions can weaken refusal behavior, so the findings should be interpreted in the context of defensive evaluation.

### Reproducibility statement

We provide the details required to reproduce our results. Section 3 defines the think gap score, the construction of the multilingual probing dataset, and the think neuron identification and intervention procedures. Section 3.3 specifies the consistency reward, the masked NSCO objective, and the training protocol. Appendix D reports the implementation and hyperparameter settings, while Appendix E documents the consistency-judge prompt, decision rule, and output parsing procedure.

## APPENDIX

The appendix is organized as follows. Appendix A summarizes the important notation used in our paper. Appendix B provides a theoretical justification for the Think Gap Score (TGS). Appendix C details the computation of the attack success rate (ASR) metric and the automated evaluation protocol. Appendix D provides the implementation details, including hyperparameters and training configurations. Appendix E describes the prompt and the binary reward formulation used for the consistency judge. Appendix G discusses the limitations of our current evaluation and future directions regarding code-switching. Appendix F shows the performance of the model before and after Mask safety think neurons. Finally, Appendix H presents additional discussions.

## Appendix A Notation Summary

Table  9 summarizes the notation used to quantify reasoning trace utilization, identify safety think neurons, and formulate the ACTR framework. A superscript $l$ indexes a model layer and does not denote exponentiation.

Table 9: Core Notation for Reasoning Utilization, Safety Think Neurons, and the ACTR Framework.

|

Symbol

|

Definition

|

$L,\mathcal{L}_{half}$

|

Model layers, indexed from zero, and the latter half of the model’s $L$ layers.

|

$x$

|

Language condition, where $x\in\{\text{HR},\text{NHR}\}$ denotes high-resource and non-high-resource languages.

|

$\mathcal{I}_{T}^{x},\mathcal{I}_{A}^{x}$

|

Token positions of the reasoning trace and the final answer, respectively.

|

$\mathcal{K}_{i}^{x,l}$

|

Key positions visible to the attention row predicting the token at position $i$ under the attention mask of layer $l$ .

|

$o_{i,S}^{x,l}$

|

Contribution of token-position set $S$ to the projected attention output.

|

$W_{Q}^{l},W_{K}^{l},W_{V}^{l},W_{O}^{l}$

|

Query, key, value, and output projection matrices at layer $l$ .

|

$r_{i,t}^{x},R_{think}^{x}$

|

Normalized contribution of the reasoning trace and its average over answer positions and selected layers.

|

$TGS$

|

Think gap score, measuring the cross-lingual difference in the normalized contribution of reasoning traces during answer prediction.

|

$s_{j}^{on},s_{j}^{off}$

|

Model trajectories for query $j$ with think mode enabled and disabled, respectively.

|

$t_{j},a_{j}^{on},a_{j}^{off}$

|

Reasoning trace generated in think mode $t_{j}$ , and the corresponding responses $a_{j}^{on}$ and $a_{j}^{off}$ .

|

$\Delta_{j}^{m}(N),I_{l}^{m}(N)$

|

Importance of neuron $N$ for example $j$ under mode $m$ , and its aggregated mode-specific importance score.

|

$\mathcal{S}_{l}^{m},TN_{l}$

|

The top $p$ percent of neurons under mode $m$ at layer $l$ , and the identified set of safety think neurons.

|

$C_{\phi},r_{i}^{con}$

|

Frozen binary consistency judge and the sequence-level consistency reward.

|

$\hat{A}_{i}$

|

Relative consistency advantage normalized within each sampled group.

|

$M_{TN},\mathcal{J}_{ACTR}$

|

Binary mask selecting the parameters associated with the safety think neurons, and the ACTR objective function.

## Appendix B Theoretical Justification of the Think Gap Score (TGS)

In Section 3.1, we introduced the Think Gap Score (TGS) to quantify the think–response attention disconnect. Here, we provide a theoretical justification for why a higher TGS (i.e., a drop in $R_{\mathrm{think}}^{\mathrm{NHR}}$ relative to $R_{\mathrm{think}}^{\mathrm{HR}}$ ) leads to degraded safety performance (higher ASR) under non-high-resource languages.

Assumption 1 (Linear Representation of Safety). Following prior mechanistic interpretability findings on linear concept representations (Meng et al., 2022; Zou et al., 2023a), we assume the existence of a linear “safety direction” $\mathbf{w}_{\mathrm{safe}}$ in the residual stream. A larger projection of the final hidden state $\mathbf{h}_{i}^{x,L}$ onto $\mathbf{w}_{\mathrm{safe}}$ increases the logit of generating a safe/refusal token at position $i$ :

| | $P(\text{Safe Token})\propto\exp\left(\langle\mathbf{h}_{i}^{x,L},\mathbf{w}_{\mathrm{safe}}\rangle\right).$
| | (15)

Assumption 2 (Safety Information is Encoded in the Think Trace). Based on our reasoning-trace substitution experiment (Table 1), we establish that the think traces for both HR and NHR queries correctly identify safety risks. Thus, the attention output sourced from the think span, $\mathbf{o}_{i,\mathcal{I}_{T}^{x}}^{x,\ell}$ , contains a strong component along the safety direction $\mathbf{w}_{\mathrm{safe}}$ . In contrast, the attention output sourced from the adversarial query itself (non-think tokens, denoted as $\mathcal{I}_{\setminus T}^{x}=\mathcal{K}_{i}^{x,\ell}\setminus\mathcal{I}_{T}^{x}$ ) aligns with the harmful intent, having a negative or zero projection on $\mathbf{w}_{\mathrm{safe}}$ .

Derivation.
In a transformer, the hidden state used to predict the token at position $i$ is updated additively by the attention and MLP layers. Focusing on the cumulative attention updates from the latter half layers $\mathcal{L}_{\mathrm{half}}$ (which directly steer the final output vocabulary distribution), the state update is:

| | $\Delta\mathbf{h}_{i,\mathrm{attn}}^{x}\approx\sum_{\ell\in\mathcal{L}_{\mathrm{half}}}\mathbf{o}_{i,\mathcal{K}_{i}^{x,\ell}}^{x,\ell}=\sum_{\ell\in\mathcal{L}_{\mathrm{half}}}\left(\mathbf{o}_{i,\mathcal{I}_{T}^{x}}^{x,\ell}+\mathbf{o}_{i,\mathcal{I}_{\setminus T}^{x}}^{x,\ell}\right).$
| | (16)

The total safety steering signal $S^{x}$ added to the residual stream is the projection of this update onto $\mathbf{w}_{\mathrm{safe}}$ :

| | $S^{x}=\langle\Delta\mathbf{h}_{i,\mathrm{attn}}^{x},\mathbf{w}_{\mathrm{safe}}\rangle=\sum_{\ell\in\mathcal{L}_{\mathrm{half}}}\Big(\langle\mathbf{o}_{i,\mathcal{I}_{T}^{x}}^{x,\ell},\mathbf{w}_{\mathrm{safe}}\rangle+\langle\mathbf{o}_{i,\mathcal{I}_{\setminus T}^{x}}^{x,\ell},\mathbf{w}_{\mathrm{safe}}\rangle\Big).$
| | (17)

By Assumption 2, $\langle\mathbf{o}_{i,\mathcal{I}_{\setminus T}^{x}}^{x,\ell},\mathbf{w}_{\mathrm{safe}}\rangle\leq 0$ , meaning the safety steering strictly relies on the positive contribution from the think span $\langle\mathbf{o}_{i,\mathcal{I}_{T}^{x}}^{x,\ell},\mathbf{w}_{\mathrm{safe}}\rangle$ .
By Cauchy-Schwarz inequality, the maximum possible safety steering provided by the think span at layer $\ell$ is bounded by its $L_{2}$ norm:

| | $\langle\mathbf{o}_{i,\mathcal{I}_{T}^{x}}^{x,\ell},\mathbf{w}_{\mathrm{safe}}\rangle\leq\left\|\mathbf{o}_{i,\mathcal{I}_{T}^{x}}^{x,\ell}\right\|_{2}\|\mathbf{w}_{\mathrm{safe}}\|_{2}.$
| | (18)

Recall our definition of the normalized think contribution in Eq. 3:

| | $r_{i,\ell}^{x}=\frac{\left\|\mathbf{o}_{i,\mathcal{I}_{T}^{x}}^{x,\ell}\right\|_{2}^{2}}{\left\|\mathbf{o}_{i,\mathcal{K}_{i}^{x,\ell}}^{x,\ell}\right\|_{2}^{2}+\epsilon}.$
| | (19)

Assuming the total energy of the layer output $\left\|\mathbf{o}_{i,\mathcal{K}_{i}^{x,\ell}}^{x,\ell}\right\|_{2}^{2}$ is bounded and relatively stable (due to LayerNorm/RMSNorm), the magnitude of the think span’s contribution $\left\|\mathbf{o}_{i,\mathcal{I}_{T}^{x}}^{x,\ell}\right\|_{2}$ is monotonically parameterized by $r_{i,\ell}^{x}$ . Therefore, the safety steering signal $S^{x}$ is directly constrained by $R_{\mathrm{think}}^{x}$ (the expectation of $r_{i,\ell}^{x}$ over positions and layers, defined in Eq. 4).

Conclusion.
For a paired query in HR and NHR languages, because the internal safety reasoning remains conceptually intact across languages, $\mathbf{w}_{\mathrm{safe}}$ remains constant. The difference in their safety steering relies primarily on the attention energy ratio:

| | $S^{\mathrm{HR}}-S^{\mathrm{NHR}}\propto R_{\mathrm{think}}^{\mathrm{HR}}-R_{\mathrm{think}}^{\mathrm{NHR}}\equiv\mathrm{TGS}^{\mathrm{NHR}\mid\mathrm{HR}}.$
| | (20)

When $\mathrm{TGS}^{\mathrm{NHR}\mid\mathrm{HR}}>0$ , it theoretically guarantees $S^{\mathrm{NHR}}<S^{\mathrm{HR}}$ . The model accumulates less safety steering signal in the residual stream during NHR generation, resulting in a lower probability of generating a safe response. This mathematically explains why a positive TGS correlates with the cross-lingual safety gap (higher ASR in NHR languages) observed in Section 3.1.

## Appendix C Attack Success Rate Computation

We quantify model safety using the attack success rate (ASR), a standard metric
in safety alignment research Qi et al. (2024b); Zeng et al. (2024).
ASR measures the proportion of queries for which the model fails to prevent the
intended harmful behavior. Given an instruction dataset $\mathcal{D}$ , we
define ASR as

|
|

$\displaystyle\mathrm{ASR}=\frac{1}{|\mathcal{D}|}\sum_{Q_{i}\in\mathcal{D}}\mathbb{I}(Q_{i}),$

| | (21)

where $Q_{i}$ denotes an input query and $\mathbb{I}(Q_{i})$ is an indicator
function specifying whether the model’s response constitutes a successful
attack. Specifically,

|
|

$\displaystyle\mathbb{I}(Q_{i})=\begin{cases}1,&\text{if the response fulfills the malicious intent of }Q_{i},\\
0,&\text{if the model refuses the request.}\end{cases}$

| | (22)

Following established practice Qi et al. (2024b), we use GPT-4o as an
automated judge to assess the safety of each response. The judge determines
whether the response meaningfully complies with the query’s malicious intent,
rather than merely matching specific refusal keywords. ASR is computed
independently for each language and evaluation condition using the same
procedure. Aggregate ASR is then calculated over the corresponding complete
query set.

## Appendix D Implementation Details

Table 10: Hyperparameters used for the ACTR strategy
across different base models. The Random baseline uses the same
training configuration.

| Hyperparameter | Qwen3-8B | Gemma4-12B-it

| Computing Device | $4\times$ A100 | $4\times$ A100

| Global Batch Size | 16 | 16

| Training Epochs | 3 | 3

| Learning Rate | $5\times 10^{-5}$ | $5\times 10^{-5}$

| Warmup Ratio | 0.03 | 0.03

| Optimizer | AdamW | AdamW

Table 10 summarizes the hyperparameters used
to optimize the models. Unless otherwise specified, we use the
same training configuration for both base models. For a fair comparison,
the Random baseline is trained using exactly the same hyperparameters.

## Appendix E Consistency Judge

We use GPT-4o-mini as a frozen judge to assign a binary consistency reward to each Think–Response pair. The judge follows this prompt:

Given a Think trace, and a Response, classify the Think trace and the Response separately into one of three categories:

Safe: The text contains no unsafe content or refuses to answer the unsafe query.

Unsafe: The text contains unsafe content or directly answers the unsafe query.

Invalid: The text does not form a natural sentence or is irrelevant to the given query.

First classify the Think trace and the Response. Then return 1 if both are classified as Safe or both are classified as Unsafe. Otherwise, return 0.

Thus, the binary consistency reward is

| | $r_{\mathrm{cons}}=\mathbb{1}\!\left[(c_{\mathrm{Think}},c_{\mathrm{Response}})\in\{(\mathrm{Safe},\mathrm{Safe}),(\mathrm{Unsafe},\mathrm{Unsafe})\}\right].$
|

The reward measures agreement between the two classifications, not response safety: a trajectory classified as Unsafe in both components also receives a reward of 1. Any trajectory with an Invalid classification receives 0.

## Appendix F Qualitative Analysis of Safety Think Neurons Masking

Qualitative Examples of Safety Degradation upon Masking safety think neurons. We present examples across two different models and NHR languages. For each instance, the green text indicates the safe content, while the red text highlights the harmful content generated after masking safety think neurons.

Figure 8: Qualitative Examples of Safety Degradation upon masking safety think neurons across different models and NHR languages.

## Appendix G Limitations

This work primarily evaluates multilingual safety under prompts written in a single language at a time, while real-world users may combine multiple languages, scripts, or language-specific expressions within the same interaction. In future work, we plan to build on ACTR to study think–response alignment under code-switching and dynamically changing language contexts. Such settings may provide a more realistic test of whether safety judgments can be consistently transferred when the language of the prompt, reasoning trace, and response are not fixed. We will further investigate whether language-switch-aware alignment can improve robustness while preserving natural multilingual interaction.

## Appendix H More Discussions

Q1. Does the consistency reward in NSCO fully align with safety objectives, and could optimizing consistency introduce potential biases?

A1. The consistency reward is not designed as a substitute for safety reward, but as a way to strengthen the alignment between safety reasoning and final responses. We intentionally avoid rewarding only safe responses, as directly optimizing refusal behavior may lead to over-refusal on benign requests. NSCO aims to improve safety by enhancing thought–response alignment rather than enforcing additional safety constraints.

Q2. Are the identified “safety think neurons” exclusively dedicated to safety, or do they serve general reasoning-transfer functions?

A2. The neurons identified by our pipeline likely represent a mixture of safety-specific circuits and general think-utilization pathways. Because we identify them using a probing dataset of jailbreak queries, the selected neurons are highly active during safety-critical contexts. However, neural representations in LLMs are inherently polysemantic. As shown in our ablation study (Figure 5), masking more than 3% of these neurons degrades general mathematical reasoning (MGSM). This suggests that while the top $3\%$ are highly specialized for transferring safety-related constraints, deeper expansion into the neuron population disrupts the model’s fundamental ability to utilize its reasoning traces for general tasks. ACTR succeeds precisely because it isolates the most safety-relevant subset without destroying the broader think-response bridge.

Q3. Could targeted masking simply damage the model globally?

A3. Random masking uses the same number of units and the same layer-wise distribution, yet changes ASR and TGS only marginally. Targeted masking instead produces large and selective increases, including a 43.90-point MultiJail increase for Gemma4-12B-it versus 1.54 points under random masking. The intervention is therefore not equivalent to generic capacity removal.

Q4. Does the random-neuron baseline show that neuron identification is unnecessary?

A4. No. The random baseline shows that sparse updates can help, but it does not match ACTR’s safety–utility trade-off. For Qwen3-8B, random-neuron training raises the benign refusal rate on XSTest from 4.40% to 52.00%, whereas ACTR raises it only to 6.40% while achieving higher unsafe-request refusal. The same pattern is stronger for Gemma4-12B-it.
