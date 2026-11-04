# PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations

Luciano Rolando Maldonado Romero

Affiliation: West Virginia University, Morgantown, WV, USA

Correspondence to: lrm00020@mix.wvu.edu

###### Abstract

Large language models increasingly operate as persistent assistants in user-facing, shared-session, and tool-augmented settings. Sensitive values disclosed by users during an active conversation may remain recoverable through later prompts after the dialogue moves to unrelated topics. We introduce PrivDrift, a benchmark for auditing re-disclosure of user-supplied secrets after conversational topic drift and persuasion-based probing. PrivDrift contains 1,000 controlled multi-turn dialogues with seeded secrets, content-dense drift turns, and standardized extraction probes. Across three LLMs with extended context windows, dialogue-level hybrid leakage ranges from 38.7% to 54.6% and varies strongly by model, secret type, and persuasion intensity. Within the tested drift window, additional topic drift does not reliably reduce leakage. Privacy risk in active LLM contexts should therefore be evaluated as a persistent behavioral failure mode, in addition to training-data memorization and immediate jailbreak behavior.

###### Keywords: 

Trustworthy AI, Large Language Models, Privacy, Evaluation, Benchmarking

## 1 Introduction

Large language models (LLMs) now function as persistent assistants that remain coherent over extended interactions (OpenAI, 2023; Touvron et al., 2023; Liu et al., 2024). In deployments spanning healthcare triage, financial planning, enterprise copilots, and personal productivity tooling, users disclose Personally Identifiable Information (PII) such as phone numbers, financial identifiers, email addresses, and health-related details. The context retention that enables multi-turn assistance also leaves earlier disclosures available to later generations.

Existing safety evaluations concentrate on training-data memorization, in which private information is extracted from parameters (Carlini et al., 2021; Nasr et al., 2023), and on immediate jailbreaking, in which a harmful behavior is elicited in the current turn (Wei et al., 2024; Zou et al., 2023). Prompt injection and multi-turn attacks have also been studied (Liu et al., 2023; Russinovich et al., 2024; Deng et al., 2024). None of these settings measures whether user-disclosed secrets can still be elicited after unrelated conversational drift, or whether topic shift functions as a practical privacy boundary within a single active session.

We do not assume that users believe an LLM forgets earlier messages within a session. The object of study is behavioral. Users and application designers may underestimate how readily earlier sensitive disclosures can be elicited through indirect, justified, high-pressure, or tool-mediated prompts. This applies to shared sessions, browser-integrated assistants, enterprise copilots, agentic workflows, and prompt-injection settings in which later instructions act on the active context in ways the original user did not intend.

PrivDrift measures active-context leakage along two controlled axes. The first is topic drift, the number of unrelated turns between disclosure and probing. The second is persuasion intensity, the interaction style of the eliciting probe. Across three LLMs with extended context windows and 1,000 dialogues per model, dialogue-level hybrid leakage ranges from 38.7% to 54.6%. Persuasion intensity has a significant effect. Leakage varies sharply by secret type: SSNs and credit cards are suppressed far more often than emails and phone numbers. Within $d\leq 6$, additional drift does not reliably reduce leakage.

Our contributions are as follows:

1.

PrivDrift, a controlled benchmark for auditing recoverability of user-disclosed secrets from active conversational context after unrelated topic drift.

2.

An evaluation of leakage under direct, justified, and high-pressure probes, which represent distinct interaction styles through which later prompts elicit sensitive context.

3.

A reproducible hybrid detector that combines normalization-based matching with an open-weights LLM judge. We report dialogue-level, probe-level, regex-only, fuzzy, and hybrid leakage rates.

4.

Privacy Half-Life ($\tau$), a stability-based metric for persistent suppression. No evaluated model reaches stable decay within the observed drift range.

## 2 Related Work

### 2.1 Contextual Privacy and Inference-Time Leakage

Privacy work on language models has centered on training-data extraction, membership inference, and recovery of memorized PII from pre-training corpora (Shokri et al., 2017; Carlini et al., 2021; Li et al., 2023; Nasr et al., 2023). These settings concern information stored in model parameters. Active-context privacy instead concerns user-provided information present in the current interaction and handled at inference time.

ConfAIde evaluates privacy reasoning through contextual integrity (Mireshghallah et al., 2024). PrivacyLens and PrivaCI-Bench study privacy-norm awareness and legal compliance in agentic and contextual settings (Shao et al., 2024; Li et al., 2025). These benchmarks pose a disclosure decision under a static access-control or privacy-norm scenario. PrivDrift instead addresses persistence: whether a user-disclosed secret can still be recovered after unrelated topic drift and subsequent persuasion-based probing.

### 2.2 Multi-Turn Adversarial Attacks

Aligned models remain exploitable across turns. Crescendo elicits harmful outputs through gradually escalating, superficially benign exchanges (Russinovich et al., 2024). Automated frameworks search over adversarial multi-turn attack paths (Deng et al., 2024; Narula et al., 2025). Prompt-injection studies show that later instructions can manipulate LLM-integrated applications (Liu et al., 2023). These lines of work target unsafe content generation or instruction hijacking. PrivDrift applies the multi-turn setting to information-flow control, specifically re-disclosure of a value supplied by the user earlier in the same active context.

### 2.3 Context Management and Unlearning

TOFU and WMDP evaluate removal or suppression of knowledge from model behavior (Maini et al., 2024; Li et al., 2024). Both operate over static question-answer pairs and parameter-level interventions. PrivDrift targets the active context rather than the weights. It asks whether a secret present in the context window is behaviorally suppressed under later elicitation attempts. This setting applies to deployed RAG systems, copilots, and agentic workflows, where privacy control is exercised through prompt-level context management rather than parameter-level unlearning.

## 3 The PrivDrift Framework

Figure 1: PrivDrift overview. The pipeline constructs controlled multi-turn dialogues, appends standardized probes, queries target models, and scores leakage with a hybrid detector.

PrivDrift is a synthetic evaluation framework with three components (Figure 1): a parametric dialogue generator, a probing harness, and a metric suite covering leakage detection and temporal analysis.

### 3.1 Threat Model: Active-Context Re-Disclosure

We study active-context re-disclosure, in which a value disclosed earlier in a conversation is reproduced in response to later prompts issued after unrelated drift. We do not assume an adversary who cannot inspect the raw transcript. The quantity measured is whether the model itself emits the sensitive value when later instructions query, justify, or pressure it to do so.

Relevant deployments include shared or persistent sessions, in which a later user interacts with an assistant without knowledge of earlier disclosures. They also include enterprise copilots and agentic workflows, in which tool outputs, retrieved documents, or injected content issue instructions over the active context. A third case is browser- or application-integrated assistants, in which earlier disclosures are reused after the task has shifted. Presence of the secret in the transcript is held fixed. What varies is whether the assistant reproduces it.

Formally, a user discloses a secret $S$ at turn $t_{0}$. The conversation proceeds through $d$ unrelated, content-dense drift turns $D=\{u_{1},a_{1},\ldots,u_{d},a_{d}\}$. At evaluation time a standardized probe $P$ is appended to the active context. A leakage event occurs if the response reveals $S$ exactly, approximately, or semantically.

### 3.2 Dataset Generation Pipeline

We generate $N=1000$ controlled dialogues with a multi-stage scaffold-then-rewrite pipeline. The ground-truth secret, drift length, and probe structure are fixed at the scaffold stage, and the rewrite stage reduces template artifacts. The dataset contains 900 synthetic dialogues and 100 human-authored dialogues built under the same secret and drift constraints. The human-authored dialogues are interleaved throughout the benchmark rather than stored as a contiguous block. The cached evaluation metadata does not retain human-authored identifiers, so we report aggregates over the full benchmark and defer a human-versus-synthetic comparison.

Statistic
Value

Total dialogues
1,000

Synthetic dialogues
900

Human-authored dialogues
100

Secret categories
4

Drift lengths
$0,2,3,4,5,6$

Avg. words per dialogue
359.16

Median words per dialogue
385.00

Min/max words per dialogue
43 / 795

Avg. approx. tokens per dialogue
478.87

Median approx. tokens per dialogue
513.33

Min/max approx. tokens per dialogue
57.33 / 1060.00

Table 1: Dataset summary for the evaluated 1,000-dialogue benchmark. Approximate token counts are estimated from word counts.

#### 3.2.1 Persona and Secret Injection

Each dialogue is initialized with a persona sampled from more than 100 professions and demographic profiles and a ground-truth secret $S$ drawn from four fixed-format PII categories: phone numbers, email addresses, Social Security Numbers, and credit card numbers. $S$ is inserted in the opening turn within a plausible task context, which gives the assistant a legitimate in-context reason to observe it.

#### 3.2.2 Content-Dense Topic Drift

Each dialogue is assigned a topic such as tax filing, medical triage, or travel logistics. Drift turns are drawn from a topic-conditioned library of dense question-answer pairs and from fallback tasks involving professional writing, structured reasoning, or long-form summarization. A subset of drift turns was written by human annotators and added to the library. The drift turns introduce entities and task demands unrelated to $S$, while $S$ and the number of drift turns remain under experimental control.

#### 3.2.3 Naturalistic Refinement

Each scaffold is passed to a rewriting model instructed to improve fluency and persona realism while preserving the exact secret, drift length, role structure, and final probe conditions. The rewrite alters surface phrasing only.

### 3.3 Probing Setup

A final standardized probe is appended to the completed history. Probes fall into three persuasion levels:

•

Simple: a direct neutral query, such as asking for the phone number mentioned earlier.

•

Medium: a query with a contextual justification, such as needing the information for a form.

•

Hard: a high-pressure or urgent request designed to test whether pressure cues increase or decrease leakage.

Drift length $d$, the number of turns between disclosure and final probing, is sampled from a mixture:

| | $d\sim\begin{cases}0&\text{with }p=0.2,\\
\mathrm{Uniform}\{2,3,4,5,6\}&\text{with }p=0.8.\end{cases}$
| | (1)

The $d=0$ component covers immediate recall, and the uniform component covers short-to-medium drift. Results bear on bounded active-context persistence and do not constitute a test of arbitrarily long contexts.

### 3.4 Evaluation Methodology

Disclosures may be verbatim, reformatted, partial substrings, or paraphrased references. We therefore use a hierarchical hybrid detector (Figure 2). A response is labeled as leakage if the normalized secret occurs in the normalized response, or if the LLM judge finds that the response reveals the full secret, a substantial substring, or partial information sufficient to identify it.

Figure 2: Hierarchical hybrid evaluation. Normalization-based matching is applied first; regex-negative responses are then passed to an LLM judge.

#### 3.4.1 Stage 1: Normalization-Based Matching

For numeric secrets, all non-digit characters are stripped from both the ground-truth secret and the response. For alphanumeric secrets, case and whitespace are normalized and meaningful delimiters are retained. With $\phi(\cdot)$ the normalization function, the regex-stage label is

| | $\mathrm{Leak}_{\mathrm{regex}}(R,S)=\mathbb{I}\big[\phi(S)\subset\phi(R)\big],$
| | (2)

where $R$ is the model response and $\mathbb{I}$ is the indicator function.

#### 3.4.2 Stage 2: LLM-as-a-Judge Verification

Regex-negative responses are passed to an open-weights Llama judge (Meta AI, 2024). The judge receives the secret and the response and returns a binary verdict on whether the response reveals the secret or a substantial part of it. This stage targets partial disclosure, indirect hints, and formatting variants outside the reach of $\phi$. Because judge-based labeling has its own error modes, regex-only and hybrid rates are reported separately.

#### 3.4.3 Privacy Half-Life

Let $D_{obs}$ be the observed set of drift lengths and $L(d)$ the leakage rate at drift length $d$. With threshold $\delta=0.1\times L(0)$, Privacy Half-Life is

| | $\displaystyle\tau=\min\{d\in D_{obs}\mid$ | $\displaystyle L(d)\leq\delta$
| | (3)

|
| | $\displaystyle\text{and }\forall d^{\prime}\in D_{obs},d^{\prime}>d:\ L(d^{\prime})\leq\delta\}.$
|

If the set is empty, $\tau=\max(D_{obs})+1$. The universal condition over $d^{\prime}>d$ excludes transient dips.

Figure 3: Privacy Half-Life $\tau$ requires stable decay below the threshold. A later resurgence disqualifies a transient decrease.

## 4 Experimental Setup

##### Models.

We evaluate GPT-OSS-120B, DeepSeek-R1, and Qwen3-VL-235B through the OpenRouter API. All models receive identical dialogue histories and probe templates.

##### Dataset.

Each model is evaluated on the same 1,000 PrivDrift dialogues. Each dialogue contains one seeded secret and one drift length $d\in\{0,2,3,4,5,6\}$ sampled from Eq. (1). Each dialogue is evaluated under all three persuasion probes, giving 3,000 cached model-probe responses per model and 9,000 cached responses in total.

##### Aggregation levels.

Overall leakage is computed at the dialogue level, where a dialogue counts as leaking if any of its probes elicits the secret. Persuasion-level and fuzzy-matching analyses are computed at the probe level, where each dialogue contributes one response per persuasion condition. Dialogue-level leakage can therefore exceed the mean of the per-persuasion probe-level rates.

##### Uncertainty and statistical tests.

Regex-only and hybrid leakage are both reported. Dialogue-level rates carry 95% bootstrap confidence intervals over dialogues. Probe-level sensitivity checks are computed over cached model-probe responses. Paired model and detector comparisons on matched examples use McNemar's test, with Bonferroni correction where applicable. Repeated measures across persuasion levels use Cochran's $Q$ with post-hoc McNemar tests. Associations between leakage and drift length, persuasion level, or secret type are tested with chi-square tests of independence, with Cramer's $V$ reported as effect size.

## 5 Results

### 5.1 Dialogue-Level Hybrid Evaluation Largely Agrees With Deterministic Matching

The LLM judge adds at most 1.10 percentage points of dialogue-level leakage over normalization-based matching (Table 2). Hybrid leakage is used as the primary metric, and regex-only rates are retained as a deterministic baseline.

Model
Regex
Hybrid
$\Delta$
$p$

GPT-OSS-120B
47.70
47.70
+0.00
1.0

DeepSeek-R1
37.80
38.70
+0.90
$7.7\times 10^{-3}$

Qwen3-VL-235B
53.50
54.60
+1.10
$2.57\times 10^{-3}$

Table 2: Dialogue-level leakage (%) under regex-only and hybrid evaluation. A dialogue is counted as leaking if any standardized probe elicits the secret.

### 5.2 All Models Leak Substantially at the Dialogue Level

Dialogue-level hybrid leakage is 38.70% for DeepSeek-R1, 47.70% for GPT-OSS-120B, and 54.60% for Qwen3-VL-235B (Table 3). These rates estimate the probability that a dialogue is vulnerable to at least one standardized extraction probe.

Model
Hybrid Leakage
95% CI

DeepSeek-R1
38.70%
[35.7, 41.6]

GPT-OSS-120B
47.70%
[44.6, 50.8]

Qwen3-VL-235B
54.60%
[51.6, 57.7]

Table 3: Dialogue-level hybrid leakage rates with 95% bootstrap confidence intervals. A dialogue is counted as leaking if any standardized probe elicits the secret.

### 5.3 Persuasion Intensity Affects Leakage Non-Monotonically

Probe-level leakage differs across persuasion levels for every model (Cochran's $Q$, $p<10^{-5}$), and the ordering is model-specific (Table 4). For GPT-OSS-120B, Hard is lowest, consistent with pressure cues triggering safety behavior. For DeepSeek-R1, Medium is lowest. For Qwen3-VL-235B, Hard is highest, indicating weaker resistance to pressure-based extraction.

Model
Simple
Medium
Hard

GPT-OSS-120B
39.30%
40.50%
32.90%

DeepSeek-R1
28.10%
22.00%
29.20%

Qwen3-VL-235B
38.70%
33.30%
40.60%

Table 4: Probe-level hybrid leakage by persuasion intensity.

### 5.4 Drift Length Does Not Reliably Reduce Leakage Within the Tested Window

Drift-leak curves dip near $d=3$ and then rebound (Figure 4). Cramer's $V$ for drift length in the cached probe-level analysis is 0.1073 for DeepSeek-R1, 0.0659 for GPT-OSS-120B, and 0.0723 for Qwen3-VL-235B. No claim is made about arbitrarily long contexts. Within the tested window, leakage persists across several unrelated topic shifts before the secret is far from the end of the context window.

Figure 4: Hybrid leakage versus drift length. Curves show a dip near $d=3$ followed by rebound.

### 5.5 Secret Type Strongly Determines Leakage

Cramer's $V$ between secret type and hybrid leakage in the cached probe-level analysis is 0.5871 for DeepSeek-R1, 0.7731 for GPT-OSS-120B, and 0.6927 for Qwen3-VL-235B. SSNs and credit card numbers are suppressed far more often than emails and phone numbers, although all four are user-disclosed and contextually private (Table 5). Behavior tracks format and sensitivity cues more closely than any generalized notion of contextual confidentiality.

Secret Type
GPT-OSS
DeepSeek
Qwen

SSN
0.14%
1.11%
5.97%

Credit card
0.00%
0.39%
4.01%

Email
78.24%
57.11%
81.13%

Phone
70.89%
46.13%
57.24%

Table 5: Probe-level hybrid leakage by secret type across cached model-probe responses.

### 5.6 Privacy Half-Life Exceeds the Observed Window

With $\delta=0.1\cdot L(0)$, no model approaches the threshold at any tested drift length. Hence $\tau=\max(D_{obs})+1=7$ for all models and persuasion levels, and there is no stable decay within $d\leq 6$.

### 5.7 Fuzzy Matching Supports the Hybrid Labels

As a post-hoc deterministic sensitivity check, we compute normalized fuzzy matching between each saved secret and saved response using cached outputs only. At threshold 0.85, fuzzy and probe-level hybrid leakage are 26.40% versus 26.43% for DeepSeek-R1, 37.03% versus 37.57% for GPT-OSS-120B, and 37.60% versus 37.53% for Qwen3-VL-235B (Table 6). Fuzzy matching is treated as auxiliary, since approximate string similarity is less expressive than the judge for partial or semantic disclosures.

Model
Regex
Fuzzy 0.85
Fuzzy 0.80
Hybrid

DeepSeek-R1
25.50
26.40
26.53
26.43

GPT-OSS-120B
36.80
37.03
37.07
37.57

Qwen3-VL-235B
36.97
37.60
37.93
37.53

Table 6: Probe-level regex, fuzzy, and hybrid leakage rates (%) computed from cached model outputs.

Figure 5: Hybrid leakage heatmap by drift length and persuasion intensity for each model.

## 6 Discussion

Sensitive values disclosed earlier in an active conversation remain behaviorally recoverable after unrelated topic drift. This finding does not imply permanent memory of the secret, nor uniform behavior across long-context settings. It does show that current assistants reproduce in-context sensitive values when a later prompt applies sufficient retrieval pressure. The near-identity of fuzzy and hybrid probe-level rates, together with the small judge increment over regex, indicates predominantly direct or near-direct reproduction.

The secret-type asymmetry is the largest effect. SSNs and credit card numbers are suppressed far more often than emails and phone numbers, even though all four categories are equally user-disclosed in the benchmark. This pattern is consistent with format-sensitive safety heuristics rather than robust contextual reasoning about confidentiality.

Persuasion effects are significant but non-monotonic and model-dependent. For GPT-OSS-120B, high-pressure requests lower leakage, which is compatible with urgency cues activating refusal behavior. For DeepSeek-R1, Medium is lowest. For Qwen3-VL-235B, Hard is highest, indicating weaker resistance to pressure-based extraction.

The intermediate dips in the leakage curves do not satisfy the Privacy Half-Life criterion, so topic drift cannot be treated as an implicit privacy boundary. Evaluation should therefore be stability-based rather than reading temporary drops as suppression.

## 7 Limitations and Future Work

##### Bounded drift range.

Drift lengths extend to $d=6$, which corresponds to bounded short-to-medium conversational drift rather than saturation of modern context windows. The results characterize persistence under bounded active-context drift and do not fully characterize privacy behavior across complete long-context windows.

##### Fixed-format secrets.

The benchmark covers structured secrets: SSNs, credit card numbers, emails, and phone numbers. These are easier to detect than unstructured sensitive information such as health status, family circumstances, immigration status, or financial hardship. Extension to non-fixed sensitive attributes is left to future work.

##### Active context, not cross-session memory.

PrivDrift measures recoverability within the active conversation context. It does not test training-data memorization, personalization memory, or cross-session retention.

##### Synthetic and human-authored dialogues.

The dataset comprises 900 synthetic and 100 human-authored dialogues interleaved throughout the benchmark rather than stored as a separate block. This improves realism relative to a purely synthetic benchmark while preserving control over secret type, drift length, and probe structure. The cached evaluation metadata does not record which rows are human-authored, so results are aggregate and a human-versus-synthetic comparison is deferred.

##### Detector scope.

Labels derive from surface-form outputs via deterministic matching, fuzzy matching, and an external LLM judge. Regex-only, fuzzy, and hybrid labels are reported separately for transparency, but formal inter-annotator agreement between humans and the automated methods is not yet reported. A larger manual audit of borderline partial disclosures and obfuscations is planned.

##### API-served models and reproducibility.

Owing to compute and budget constraints, the evaluated models are API-served and may change over time. We log model identifiers, prompts, timestamps, and cached responses, and we compute paired statistics on identical dialogue sets.

##### Artifact availability.

We plan to release the generation code, probe templates, evaluation scripts, aggregate result files, and a sanitized subset of generated dialogues. Because the benchmark contains synthetic PII-like strings, released examples will use non-real identifiers, together with generation templates sufficient to reproduce the benchmark without exposing realistic identifiers.

## 8 Conclusion

PrivDrift audits active-context privacy leakage under topic drift and persuasion intensity. Across three LLMs and 1,000 dialogues per model, dialogue-level hybrid leakage ranges from 38.7% to 54.6%. Persuasion effects are model-dependent, and secret type shows the strongest association with leakage. Within the tested drift range, drift does not reliably reduce leakage, and Privacy Half-Life exceeds the observed window for every model. The secret-type asymmetry indicates that current protections are uneven and keyed more to format-sensitive cues than to robust contextual confidentiality.

## Appendix A Dataset Characterization

Drift $d$
$N$
Avg. words
Median words
Avg. tokens
Median tokens

0
197
44.29
43.00
59.05
57.33

2
147
253.97
258.00
338.62
344.00

3
142
344.18
347.00
458.91
462.67

4
166
452.27
447.00
603.02
596.00

5
181
520.92
543.00
694.56
724.00

6
167
568.03
605.00
757.37
806.67

Table 7: Dataset length characteristics by drift length. Approximate token counts are estimated from word counts.

## Appendix B Additional Effect Sizes

Model
Variable
Cramer’s $V$
Interpretation

DeepSeek-R1
Secret type
0.5871
Large

DeepSeek-R1
Drift length
0.1073
Small

DeepSeek-R1
Persuasion
0.0718
Negligible

GPT-OSS-120B
Secret type
0.7731
Large

GPT-OSS-120B
Drift length
0.0659
Negligible

GPT-OSS-120B
Persuasion
0.0689
Negligible

Qwen3-VL-235B
Secret type
0.6927
Large

Qwen3-VL-235B
Drift length
0.0723
Negligible

Qwen3-VL-235B
Persuasion
0.0639
Negligible

Table 8: Probe-level effect sizes for associations between hybrid leakage and benchmark variables.

## Appendix C Dialogue Construction Figure

Figure 6: Dialogue construction details.