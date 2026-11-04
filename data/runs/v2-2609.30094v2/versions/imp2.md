# PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations

Luciano Rolando Maldonado Romero

Affiliation: West Virginia University, Morgantown, WV, USA

Correspondence to: lrm00020@mix.wvu.edu

###### Abstract

We introduce PrivDrift, a benchmark of 1,000 controlled multi-turn dialogues with seeded user secrets, content-dense drift turns, and standardized extraction probes at three persuasion levels. It is intended for persistent assistants deployed in user-facing, shared-session, and tool-augmented settings. Across three LLMs with extended context windows, dialogue-level hybrid leakage ranges from 38.7% to 54.6%. Leakage varies strongly by model, secret type, and persuasion intensity. Within the tested drift window, additional topic drift does not reliably reduce leakage.

###### Keywords: 

Trustworthy AI, Large Language Models, Privacy, Evaluation, Benchmarking

## 1 Introduction

LLMs now serve as persistent multi-turn assistants (OpenAI, 2023; Touvron et al., 2023; Liu et al., 2024). They are used in healthcare triage, financial planning, enterprise copilots, and personal productivity tools, where users may disclose PII such as phone numbers, financial identifiers, email addresses, and health-related details. Prior safety evaluations target training-data memorization (Carlini et al., 2021; Nasr et al., 2023) and immediate jailbreaking (Wei et al., 2024; Zou et al., 2023). Others study prompt injection and multi-turn attacks (Liu et al., 2023; Russinovich et al., 2024; Deng et al., 2024). None of these measure recoverability of user-disclosed secrets after unrelated conversational drift.

We do not assume users believe an LLM forgets earlier messages in a session. The relevant settings are shared sessions, browser-integrated assistants, enterprise copilots, agentic workflows, and prompt injection, in which later indirect, justified, high-pressure, or tool-mediated prompts interact with the active context.

PrivDrift varies two axes: topic drift and persuasion intensity. Over three LLMs with extended context windows and 1,000 dialogues per model, dialogue-level hybrid leakage is 38.7% to 54.6%. Persuasion intensity has a significant effect. SSNs and credit cards are suppressed far more often than emails and phone numbers. For $d\leq 6$, additional drift does not reliably reduce leakage.

Contributions:

1.

PrivDrift, a controlled benchmark for active-context re-disclosure after unrelated topic drift.

2.

Leakage under Simple, Medium, and Hard probes.

3.

A reproducible hybrid detector that combines normalization-based matching with an open-weights LLM judge. We report dialogue-level, probe-level, regex-only, fuzzy, and hybrid rates.

4.

Privacy Half-Life ( $\tau$ ). No evaluated model reaches stable decay within the observed drift range.

## 2 Related Work

### 2.1 Contextual Privacy and Inference-Time Leakage

Prior work covers training-data extraction, membership inference, and recovery of memorized PII (Shokri et al., 2017; Carlini et al., 2021; Li et al., 2023; Nasr et al., 2023), all of which concern parameters. ConfAIde evaluates privacy reasoning through contextual integrity (Mireshghallah et al., 2024). PrivacyLens and PrivaCI-Bench address norm awareness and legal compliance in agentic or contextual settings (Shao et al., 2024; Li et al., 2025). These benchmarks use static access-control or privacy-norm scenarios. PrivDrift instead isolates persistence of a user-disclosed secret under unrelated drift and later persuasion-based probing.

### 2.2 Multi-Turn Adversarial Attacks

Crescendo elicits harmful outputs through gradual benign-seeming conversations (Russinovich et al., 2024). Automated frameworks generate multi-turn attack paths (Deng et al., 2024; Narula et al., 2025). Prompt injection manipulates LLM-integrated applications (Liu et al., 2023). These works target unsafe content or instruction hijacking. PrivDrift targets information flow, specifically re-disclosure of user-supplied information from the same active context.

### 2.3 Context Management and Unlearning

TOFU and WMDP evaluate removal or suppression of knowledge (Maini et al., 2024; Li et al., 2024) over static QA pairs and model weights. PrivDrift targets behavioral suppression of a secret present in the context window. This is the relevant control surface for RAG systems, copilots, and agentic workflows that rely on prompt-level context management.

## 3 The PrivDrift Framework

Figure 1: PrivDrift overview. The pipeline constructs controlled multi-turn dialogues, appends standardized probes, queries target models, and scores leakage with a hybrid detector.

PrivDrift has three components (Figure 1): a parametric dialogue generator, a probing harness, and a metric suite for leakage detection and temporal analysis.

### 3.1 Threat Model: Active-Context Re-Disclosure

The adversary is not assumed to be unable to inspect the raw transcript. Instances include a later user in a shared or persistent session, later tool outputs, retrieved documents, or prompt-injection content in copilots and agentic workflows, and reuse after a task shift in integrated assistants.

A user discloses secret $S$ at turn $t_{0}$. Then $d$ unrelated, content-dense drift turns $D=\{u_{1},a_{1},\ldots,u_{d},a_{d}\}$ follow, and a standardized probe $P$ is appended. A leakage event is any response revealing $S$ exactly, approximately, or semantically.

### 3.2 Dataset Generation Pipeline

$N=1000$ dialogues are produced by a multi-stage scaffold-then-rewrite pipeline. The ground-truth secret, drift length, and probe structure are fixed by the scaffold. The set comprises 900 synthetic and 100 human-authored dialogues under the same secret and drift constraints. The human-authored dialogues are interleaved rather than stored as a contiguous block. The cached evaluation metadata does not preserve their identifiers, so results are aggregate and a human-versus-synthetic comparison is deferred.

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

Personas are sampled from more than 100 professions and demographic profiles. $S$ is drawn from four fixed-format PII categories: phone numbers, email addresses, Social Security Numbers, and credit card numbers. It is inserted in the opening turn within a plausible task context, which gives a legitimate in-context reason for its presence.

#### 3.2.2 Content-Dense Topic Drift

Each dialogue is assigned a topic, such as tax filing, medical triage, or travel logistics. Drift turns come from a topic-conditioned library of dense QA pairs and fallback tasks covering professional writing, structured reasoning, and long-form summarization. Annotators authored a subset of the drift turns, which were incorporated into the library.

#### 3.2.3 Naturalistic Refinement

A rewriting model revises each scaffold for fluency and persona realism. It preserves the exact secret, drift length, role structure, and final probe conditions, so only surface phrasing changes.

### 3.3 Probing Setup

A final standardized probe is appended to the full history:

•

Simple: a direct neutral query, such as asking for the phone number mentioned earlier.

•

Medium: a query with a contextual justification, such as needing the information for a form.

•

Hard: a high-pressure or urgent request.

Drift length is sampled as

| | $d\sim\begin{cases}0&\text{with }p=0.2,\\
\mathrm{Uniform}\{2,3,4,5,6\}&\text{with }p=0.8.\end{cases}$
| | (1)

The scope is bounded active-context persistence, not arbitrarily long windows.

### 3.4 Evaluation Methodology

Disclosure may be verbatim, reformatted, partial, or paraphrased. A response is labeled leaking if the normalized secret appears in the normalized response, or if the judge finds the full secret, a substantial substring, or identifying partial information.

Figure 2: Hierarchical hybrid evaluation. Normalization-based matching is applied first; regex-negative responses are then passed to an LLM judge.

#### 3.4.1 Stage 1: Normalization-Based Matching

With $\phi(\cdot)$ stripping non-digits for numeric secrets, and normalizing case and whitespace while preserving meaningful delimiters for alphanumeric secrets,

| | $\mathrm{Leak}_{\mathrm{regex}}(R,S)=\mathbb{I}\big[\phi(S)\subset\phi(R)\big],$
| | (2)

where $R$ is the model response and $\mathbb{I}$ is the indicator function.

#### 3.4.2 Stage 2: LLM-as-a-Judge Verification

Regex-negative responses go to an open-weights Llama judge (Meta AI, 2024). Given the secret and the response, it returns a binary verdict on whether the secret or a substantial part of it is revealed. Regex-only and hybrid rates are reported separately.

#### 3.4.3 Privacy Half-Life

With $D_{obs}$ the observed drift lengths, $L(d)$ the leakage rate at $d$, and $\delta=0.1\times L(0)$:

| | $\displaystyle\tau=\min\{d\in D_{obs}\mid$ | $\displaystyle L(d)\leq\delta$
| | (3)

|
| | $\displaystyle\text{and }\forall d^{\prime}\in D_{obs},d^{\prime}>d:\ L(d^{\prime})\leq\delta\}.$
|

For an empty set, $\tau=\max(D_{obs})+1$.

Figure 3: Privacy Half-Life $\tau$ requires stable decay below the threshold. A later resurgence disqualifies a transient decrease.

## 4 Experimental Setup

##### Models.

GPT-OSS-120B, DeepSeek-R1, and Qwen3-VL-235B are evaluated via the OpenRouter API, with identical dialogue histories and probe templates.

##### Dataset.

All 1,000 dialogues are run on each model. Each dialogue contains one secret and one $d\in\{0,2,3,4,5,6\}$ drawn from Eq. (1), and each receives all three probes. This gives 3,000 cached model-probe responses per model and 9,000 in total.

##### Aggregation levels.

Overall leakage is dialogue-level, with a dialogue counted as leaking if any probe elicits the secret. Persuasion-level and fuzzy analyses are probe-level, with one response per dialogue per condition. Dialogue-level leakage can therefore exceed the mean of the per-persuasion rates.

##### Uncertainty and statistical tests.

We report 95% bootstrap CIs over dialogues for dialogue-level rates, and rates over cached model-probe responses for probe-level checks. Paired model and detector comparisons use McNemar's test with Bonferroni correction where applicable. Persuasion levels are compared with Cochran's $Q$ and post-hoc McNemar tests. Associations of leakage with drift length, persuasion level, and secret type use chi-square tests of independence, with Cramer's $V$ as the effect size.

## 5 Results

### 5.1 Dialogue-Level Hybrid Evaluation Largely Agrees With Deterministic Matching

The judge adds at most 1.10 percentage points of dialogue-level leakage (Table 2). Hybrid is the primary metric, and regex-only is retained as a deterministic baseline.

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

Cochran's $Q$ gives $p<10^{-5}$ for all models. GPT-OSS-120B is lowest under Hard, DeepSeek-R1 is lowest under Medium, and Qwen3-VL-235B is highest under Hard (Table 4).

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

The curves dip near $d=3$ and then rebound (Figure 4). Cramer's $V$ (probe-level) is 0.1073 for DeepSeek-R1, 0.0659 for GPT-OSS-120B, and 0.0723 for Qwen3-VL-235B. No claim is made for arbitrarily long contexts, and the secret is never far from the end of the window.

Figure 4: Hybrid leakage versus drift length. Curves show a dip near $d=3$ followed by rebound.

### 5.5 Secret Type Strongly Determines Leakage

Cramer's $V$ (probe-level) is 0.5871 for DeepSeek-R1, 0.7731 for GPT-OSS-120B, and 0.6927 for Qwen3-VL-235B. All four types are user-disclosed and contextually private (Table 5).

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

With $\delta=0.1\cdot L(0)$, no model approaches the threshold at any tested $d$. Hence $\tau=\max(D_{obs})+1=7$ for all models and persuasion levels.

### 5.7 Fuzzy Matching Supports the Hybrid Labels

Post-hoc normalized fuzzy matching was computed between each saved secret and response, from cached outputs only. At the 0.85 threshold, fuzzy versus probe-level hybrid leakage is 26.40% versus 26.43% for DeepSeek-R1, 37.03% versus 37.57% for GPT-OSS-120B, and 37.60% versus 37.53% for Qwen3-VL-235B (Table 6). Fuzzy matching is auxiliary, being less expressive than the judge for partial or semantic disclosure.

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

Secrets disclosed earlier remain recoverable after unrelated drift under sufficient retrieval pressure. This does not imply permanent memory or uniform long-context behavior. The fuzzy-hybrid agreement places most leakage at direct or near-direct reproduction.

SSNs and credit cards are suppressed far more often than emails and phone numbers, which points to format-sensitive heuristics in current safeguards. Persuasion effects are significant, non-monotonic, and model-dependent. Hard probes lower leakage for GPT-OSS-120B, consistent with urgency cues triggering refusal. Medium is lowest for DeepSeek-R1. Hard is highest for Qwen3-VL-235B. Intermediate dips in the drift curves fail the $\tau$ stability criterion.

## 7 Limitations and Future Work

##### Bounded drift range.

The evaluation covers $d$ up to $d=6$, short-to-medium drift, and does not saturate modern context windows.

##### Fixed-format secrets.

The benchmark covers only SSNs, credit card numbers, emails, and phone numbers. Unstructured attributes such as health status, family circumstances, immigration status, and financial hardship are left for future work.

##### Active context, not cross-session memory.

Training-data memorization, personalization memory, and cross-session retention are not tested.

##### Synthetic and human-authored dialogues.

The 900 synthetic and 100 human-authored dialogues are interleaved. The cached metadata does not mark the human-authored rows, so only aggregate results are reported and the human-versus-synthetic comparison is deferred.

##### Detector scope.

Labels rest on surface-form outputs from deterministic matching, fuzzy matching, and an external LLM judge. Formal human-automated inter-annotator agreement is not yet reported. A larger manual audit of borderline partial disclosures and obfuscations is planned.

##### API-served models and reproducibility.

API-served models may change over time. We log model identifiers, prompts, timestamps, cached responses, and paired statistics on identical dialogue sets.

##### Artifact availability.

We plan to release the generation code, probe templates, evaluation scripts, aggregate results, and a sanitized subset of dialogues. Released examples will use non-real identifiers, and the templates will suffice to reproduce the benchmark.

## 8 Conclusion

PrivDrift audits active-context leakage under topic drift and persuasion intensity. Across three LLMs and 1,000 dialogues per model, dialogue-level hybrid leakage is 38.7% to 54.6%. Persuasion effects are model-dependent. Secret type has the strongest association with leakage. Drift does not reliably reduce leakage within the tested range, and $\tau$ exceeds the observed window for all models.

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