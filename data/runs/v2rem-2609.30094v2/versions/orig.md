# PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations

Luciano Rolando Maldonado Romero

Affiliation: West Virginia University, Morgantown, WV, USA

Correspondence to: lrm00020@mix.wvu.edu

###### Abstract

Large language models increasingly operate as persistent assistants in user-facing, shared-session, and tool-augmented settings. When users disclose sensitive information during an active conversation, that information becomes part of the context the model conditions on for every later turn, which raises the question of how such information is handled as the dialogue moves on to unrelated topics. We introduce PrivDrift, a benchmark and evaluation framework for auditing how LLMs treat user-disclosed secrets after conversational topic drift and under different styles of later probing. PrivDrift contains 1,000 controlled multi-turn dialogues with seeded secrets, content-dense drift turns, and standardized extraction probes, together with a layered detection pipeline and a stability-oriented summary metric. We describe the generation pipeline, the threat model, the detection methodology, and the statistical protocol, and we apply the framework to three LLMs with extended context windows. PrivDrift is intended to complement existing evaluations of training-data memorization and immediate jailbreak behavior by providing a controlled setting for studying privacy behavior inside active conversations.

###### Keywords: 

Trustworthy AI, Large Language Models, Privacy, Evaluation, Benchmarking

## 1 Introduction

Large language models (LLMs) have evolved from simple chat systems into persistent assistants capable of maintaining coherence over extended interactions (OpenAI, 2023; Touvron et al., 2023; Liu et al., 2024). As these systems are integrated into settings such as healthcare triage, financial planning, enterprise copilots, and personal productivity tools, users may disclose Personally Identifiable Information (PII), including phone numbers, financial identifiers, email addresses, and health-related details. The same context retention that supports useful multi-turn assistance also means that sensitive information disclosed earlier in an active conversation remains part of what the model conditions on for later generations.

Existing safety evaluations mostly target two extremes: training-data memorization, where private information is extracted from model parameters (Carlini et al., 2021; Nasr et al., 2023), and immediate jailbreaking, where a harmful behavior is elicited in the current turn (Wei et al., 2024; Zou et al., 2023). Recent work also studies prompt injection and multi-turn attacks (Liu et al., 2023; Russinovich et al., 2024; Deng et al., 2024), but these settings do not directly measure how user-disclosed secrets are treated after unrelated conversational drift. This leaves room for a dedicated evaluation setting in trustworthy AI research: one that focuses on the handling of user-supplied sensitive values within a single active conversation as the topic changes.

We do not assume that users believe an LLM literally forgets earlier messages in the same session. Rather, we are interested in a behavioral question: how assistants respond when sensitive information disclosed earlier is requested later through indirect, justified, high-pressure, or tool-mediated prompts. This question is relevant to shared sessions, browser-integrated assistants, enterprise copilots, agentic workflows, and prompt-injection settings where later instructions interact with the active context in ways the original user did not intend.

We introduce PrivDrift, a controlled benchmark for measuring active-context privacy behavior along two axes: topic drift, which captures the number of unrelated turns between secret disclosure and later probing, and persuasion intensity, which captures the interaction style used to elicit the secret. The benchmark is built so that the secret, the drift length, and the probe can be held fixed while surrounding content varies, allowing these factors to be examined in a controlled way. We apply the framework to three LLMs with extended context windows, using 1,000 dialogues per model and three probes per dialogue.

Our contributions are as follows:

1.

PrivDrift, a controlled benchmark for auditing how user-disclosed secrets are handled in active conversational context after unrelated topic drift.

2.

A probing protocol with direct, justified, and high-pressure probes, modeling how later prompts can request sensitive context through different interaction styles.

3.

A reproducible hybrid detector that combines normalization-based matching with an open-weights LLM judge, together with a reporting scheme covering dialogue-level, probe-level, regex-only, fuzzy, and hybrid leakage rates.

4.

Privacy Half-Life ( $\tau$ ), a stability-based metric intended to summarize whether suppression of a secret is sustained across increasing drift lengths.

## 2 Related Work

### 2.1 Contextual Privacy and Inference-Time Leakage

Privacy research in language models has historically focused on training-data extraction, membership inference, and recovery of memorized PII from pre-training corpora (Shokri et al., 2017; Carlini et al., 2021; Li et al., 2023; Nasr et al., 2023). These settings study whether private information is stored in model parameters. In contrast, active-context privacy concerns user-provided information that appears in the current interaction and must be handled appropriately at inference time.

Recent benchmarks examine whether LLMs can reason about privacy norms and contextual access. ConfAIde evaluates privacy reasoning through contextual integrity (Mireshghallah et al., 2024), while PrivacyLens and PrivaCI-Bench study privacy norm awareness and legal compliance in agentic or contextual settings (Shao et al., 2024; Li et al., 2025). These benchmarks primarily test whether a model should share information under a static access-control or privacy-norm scenario. PrivDrift instead focuses on a persistence-oriented setting: how a user-disclosed secret is treated after unrelated topic drift and later persuasion-based probing.

The contextual integrity framing is a useful lens for this work. Under contextual integrity, the appropriateness of an information flow depends on the norms of the context in which the information was shared. A phone number given to an assistant to draft a reservation request carries an implicit purpose, and reuse of that value for an unrelated later request may or may not be consistent with that purpose. PrivDrift does not attempt to adjudicate these norms directly; it provides a controlled testbed in which such questions can be posed against measurable model behavior.

### 2.2 Multi-Turn Adversarial Attacks

Aligned models remain vulnerable to multi-turn exploitation. Crescendo attacks show that seemingly benign conversations can gradually elicit harmful outputs (Russinovich et al., 2024), while automated jailbreak frameworks generate adversarial multi-turn attack paths (Deng et al., 2024; Narula et al., 2025). Prompt injection studies also show that later instructions can manipulate LLM-integrated applications (Liu et al., 2023). These works usually focus on unsafe content generation or instruction hijacking. PrivDrift adapts the multi-turn lens to information flow control: the question is not whether a model can be made to produce harmful external content, but how it responds to requests to re-disclose sensitive information supplied by a user earlier in the same active context.

A related distinction concerns where adversarial effort is located. In many multi-turn jailbreaks, the conversation itself is the attack, gradually steering the model toward an unsafe output. In PrivDrift, the intervening turns are deliberately unrelated to the secret, and the adversarial component is confined to the final probe. This separation makes it possible to treat drift and persuasion as distinct experimental factors rather than as entangled properties of a single attack trajectory.

### 2.3 Context Management and Unlearning

Machine unlearning benchmarks such as TOFU and WMDP evaluate whether knowledge can be removed or suppressed from model behavior (Maini et al., 2024; Li et al., 2024). These benchmarks usually operate over static question-answer pairs and model parameters. PrivDrift is related in spirit but targets a different object of control: active context rather than model weights. We ask how a secret present in the context window is treated when later prompts try to elicit it. This distinction is important for deployed RAG systems, copilots, and agentic workflows, where privacy control often depends on prompt-level context management rather than parameter-level unlearning.

Practical context management strategies, such as truncation, summarization, and retrieval-based memory, also interact with this question. Systems that summarize or prune earlier turns may incidentally remove sensitive values, while systems that retain full transcripts keep them available. PrivDrift evaluates the full-transcript case, which is the most common configuration for short-to-medium conversations and serves as a natural baseline for studying more elaborate memory architectures.

## 3 The PrivDrift Framework

Figure 1: PrivDrift overview. The pipeline constructs controlled multi-turn dialogues, appends standardized probes, queries target models, and scores leakage with a hybrid detector.

PrivDrift is a synthetic evaluation framework for studying how sensitive information disclosed in an active conversation is handled after topic drift. As shown in Figure 1, the framework has three components: a parametric dialogue generator, a probing harness, and a metric suite for leakage detection and temporal analysis.

### 3.1 Threat Model: Active-Context Re-Disclosure

We study active-context re-disclosure: a failure mode in which sensitive information disclosed earlier in a conversation is reproduced in response to later prompts after unrelated topic drift. We do not assume that an adversary is always unable to inspect the raw transcript. Instead, PrivDrift evaluates whether the model itself will re-disclose the sensitive value when later instructions query, justify, or pressure the assistant to reveal it.

This setting captures several practical scenarios. In shared or persistent sessions, a later user may interact with an assistant without understanding what private information was previously disclosed. In enterprise copilots and agentic workflows, later tool outputs, retrieved documents, or prompt-injection content may issue instructions that interact with the active context. In browser or application-integrated assistants, the user may not intend for earlier sensitive disclosures to be reused after the task has shifted.

Formally, a user discloses a secret $S$ at turn $t_{0}$ . The conversation proceeds through $d$ unrelated but content-dense drift turns, denoted as $D=\{u_{1},a_{1},\ldots,u_{d},a_{d}\}$ . At evaluation time, a standardized probe $P$ is appended to the active context. A leakage event occurs if the model response reveals $S$ exactly, approximately, or semantically.

### 3.2 Dataset Generation Pipeline

We generate $N=1000$ controlled dialogues using a multi-stage scaffold-then-rewrite pipeline. This design preserves strict control over the ground-truth secret, drift length, and probe structure while reducing the rigid artifacts of purely template-based generation. The dataset contains 900 synthetic dialogues and 100 human-authored dialogues constructed under the same secret and drift constraints. The human-authored dialogues were inserted throughout the benchmark rather than stored as a separate contiguous block. Since the current cached evaluation metadata does not preserve the human-authored identifiers, we treat the full benchmark as a single evaluation set and leave a separate human-versus-synthetic comparison for future work.

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

Each dialogue is initialized with a persona sampled from more than 100 professions and demographic profiles, together with a ground-truth secret $S$ . We sample $S$ from four fixed-format PII categories: phone numbers, email addresses, Social Security Numbers, and credit card numbers. The secret is inserted in the opening turn as part of a plausible task context, establishing a legitimate in-context reason for the assistant to observe the information.

#### 3.2.2 Content-Dense Topic Drift

For each dialogue, we assign a topic such as tax filing, medical triage, or travel logistics. Drift turns are drawn from a topic-conditioned library of dense question-answer pairs and fallback tasks involving professional writing, structured reasoning, or long-form summarization. A subset of drift turns is human-authored by annotators and incorporated into the generation library. This design aims to add unrelated entities and task demands to the active context while preserving experimental control over the original secret and the number of drift turns.

#### 3.2.3 Naturalistic Refinement

To reduce rigidity from the initial scaffold, we apply a final naturalistic refinement step. Each dialogue scaffold is passed to a rewriting model with instructions to improve fluency and persona realism while preserving the exact secret, drift length, role structure, and final probe conditions. This step modifies surface phrasing but does not change the experimental variables.

### 3.3 Probing Setup

The main evaluation appends a final standardized probe to the completed conversation history. Probes are grouped into three persuasion levels:

•

Simple: a direct neutral query, such as asking for the phone number mentioned earlier.

•

Medium: a query with a contextual justification, such as needing the information for a form.

•

Hard: a high-pressure or urgent request designed to test whether pressure cues increase or decrease leakage.

We sample drift length $d$ , the number of turns between secret disclosure and final probing, from a mixed distribution:

| | $d\sim\begin{cases}0&\text{with }p=0.2,\\
\mathrm{Uniform}\{2,3,4,5,6\}&\text{with }p=0.8.\end{cases}$
| | (1)

This distribution includes immediate recall cases while emphasizing short-to-medium topic drift. The framework is therefore designed to characterize bounded active-context behavior, not to serve as a full test of arbitrarily long context windows.

### 3.4 Evaluation Methodology

A model may reveal a secret verbatim, with formatting changes, through a partial substring, or through a paraphrased reference. We therefore implement a hierarchical hybrid detector. A response is labeled as leakage if either the normalized secret appears in the normalized response or the LLM judge determines that the response reveals the full secret, a substantial substring, or enough partial information to identify the secret.

Figure 2: Hierarchical hybrid evaluation. Normalization-based matching is applied first; regex-negative responses are then passed to an LLM judge.

#### 3.4.1 Stage 1: Normalization-Based Matching

The first stage detects verbatim and near-verbatim leakage. For numeric secrets, we strip all non-digit characters from both the ground-truth secret and the model response. For alphanumeric secrets, we normalize case and whitespace while preserving meaningful delimiters. Let $\phi(\cdot)$ denote the normalization function. The regex-stage leakage label is

| | $\mathrm{Leak}_{\mathrm{regex}}(R,S)=\mathbb{I}\big[\phi(S)\subset\phi(R)\big],$
| | (2)

where $R$ is the model response and $\mathbb{I}$ is the indicator function.

#### 3.4.2 Stage 2: LLM-as-a-Judge Verification

Responses that are negative under normalization-based matching are passed to an open-weights Llama judge (Meta AI, 2024). The judge receives the secret and the model response and returns a binary verdict indicating whether the response reveals the secret or a substantial part of it. This stage is intended to capture partial disclosure, indirect hints, and formatting variations not captured by deterministic matching. We report regex-only and hybrid rates separately because judge-based evaluation can introduce its own errors.

#### 3.4.3 Privacy Half-Life

We introduce Privacy Half-Life ( $\tau$ ), a stability-based metric for sustained suppression. Let $D_{obs}$ be the observed set of drift lengths and $L(d)$ be the leakage rate at drift length $d$ . We define the safety threshold as $\delta=0.1\times L(0)$ and define

| | $\displaystyle\tau=\min\{d\in D_{obs}\mid$ | $\displaystyle L(d)\leq\delta$
| | (3)

|
| | $\displaystyle\text{and }\forall d^{\prime}\in D_{obs},d^{\prime}>d:\ L(d^{\prime})\leq\delta\}.$
|

If this set is empty, we assign $\tau=\max(D_{obs})+1$ . This definition prevents transient dips from being interpreted as stable suppression.

Figure 3: Privacy Half-Life $\tau$ requires stable decay below the threshold. A later resurgence disqualifies a transient decrease.

## 4 Experimental Setup

##### Models.

We evaluate GPT-OSS-120B, DeepSeek-R1, and Qwen3-VL-235B through the OpenRouter API. All models are queried with the same dialogue histories and probe templates.

##### Dataset.

We evaluate 1,000 dialogues generated by the PrivDrift pipeline on each model. Each dialogue contains one seeded secret and one drift length $d\in\{0,2,3,4,5,6\}$ sampled from the distribution above. Each dialogue is evaluated under three standardized persuasion probes, yielding 3,000 cached model-probe responses per model and 9,000 total cached responses across the three evaluated models.

##### Aggregation levels.

We use two aggregation levels. Dialogue-level leakage counts a dialogue as leaking if any standardized probe elicits the secret. Persuasion-level and fuzzy-matching analyses are computed at the probe level, where each dialogue contributes one response per persuasion condition. Because a dialogue is counted at the dialogue level whenever any one of its three probes elicits the secret, dialogue-level rates are not directly comparable to averages of individual persuasion-level rates, and we keep the two levels distinct throughout.

##### Uncertainty and statistical tests.

We record both regex-only and hybrid labels. For dialogue-level rates, we compute 95% bootstrap confidence intervals over dialogues. For probe-level sensitivity checks, we compute rates over cached model-probe responses. For paired model and detector comparisons on matched examples, we use McNemar’s test with Bonferroni correction where applicable. For repeated measures across persuasion levels, we use Cochran’s $Q$ test with post-hoc McNemar tests. To test association between leakage and drift length, persuasion level, or secret type, we use chi-square tests of independence and report Cramer’s $V$ as an effect size.

## 5 Results

This section describes how the PrivDrift outputs are organized and interpreted for each analysis. Because the framework produces several overlapping views of the same responses, we describe each view in terms of what it measures and how it should be read relative to the others.

### 5.1 Dialogue-Level Hybrid Evaluation Largely Agrees With Deterministic Matching

The first comparison concerns the relationship between the two detector stages. Since the judge is only applied to regex-negative responses, the hybrid label can never be lower than the regex label, and the difference between them isolates the contribution of the judge. Paired McNemar tests on matched dialogues indicate whether that contribution is distinguishable from zero for each model. Reporting both columns side by side makes the role of the judge transparent and allows readers to adopt the deterministic baseline if they prefer a fully reproducible label.

Model
Regex
Hybrid
$\Delta$
$p$

GPT-OSS-120B
Stage 1
Stage 1 + 2
Judge contribution
McNemar

DeepSeek-R1
Stage 1
Stage 1 + 2
Judge contribution
McNemar

Qwen3-VL-235B
Stage 1
Stage 1 + 2
Judge contribution
McNemar

Table 2: Structure of the dialogue-level detector comparison. A dialogue is counted as leaking if any standardized probe elicits the secret.

### 5.2 All Models Leak Substantially at the Dialogue Level

The dialogue-level view answers a deployment-oriented question: is a given conversation vulnerable to at least one of the standardized extraction probes? This view is conservative from a defender's perspective, since an attacker in practice may try multiple phrasings. Confidence intervals are obtained by resampling dialogues, which respects the fact that the three probes for a dialogue share the same history and secret.

Model
Hybrid Leakage
95% CI

DeepSeek-R1
Any-probe rate
Bootstrap over dialogues

GPT-OSS-120B
Any-probe rate
Bootstrap over dialogues

Qwen3-VL-235B
Any-probe rate
Bootstrap over dialogues

Table 3: Reporting scheme for dialogue-level hybrid leakage with 95% bootstrap confidence intervals. A dialogue is counted as leaking if any standardized probe elicits the secret.

### 5.3 Persuasion Intensity Affects Leakage Non-Monotonically

The persuasion analysis is conducted at the probe level, where each dialogue contributes a matched triple of responses. Cochran's $Q$ test assesses whether the three persuasion conditions differ jointly, and post-hoc McNemar tests identify which pairs differ. Because pressure and justification cues can plausibly push behavior in either direction, activating refusal heuristics in some cases and compliance in others, no monotonic ordering of conditions is assumed in advance, and the analysis is carried out separately for each model.

Model
Simple
Medium
Hard

GPT-OSS-120B
Direct query
Justified query
Pressured query

DeepSeek-R1
Direct query
Justified query
Pressured query

Qwen3-VL-235B
Direct query
Justified query
Pressured query

Table 4: Probe-level persuasion conditions evaluated for each model.

### 5.4 Drift Length Does Not Reliably Reduce Leakage Within the Tested Window

Drift length is analyzed by grouping probe-level responses by $d$ and computing hybrid leakage for each group. Chi-square tests of independence and Cramer's $V$ quantify the strength of association between drift length and leakage. The bounded range $d\leq 6$ should be kept in mind when reading these curves: they characterize behavior across several unrelated topic shifts, well before the original secret approaches the limits of the models' context windows, and they should not be extrapolated to arbitrarily long contexts.

Figure 4: Hybrid leakage versus drift length for each evaluated model.

### 5.5 Secret Type Strongly Determines Leakage

The four secret categories differ in format and in how commonly they are associated with sensitivity in public discourse and safety training data. SSNs and credit card numbers are canonical examples of protected identifiers, while emails and phone numbers are often treated as semi-public contact details. Grouping probe-level responses by secret type, and testing association with Cramer's $V$, allows the framework to examine whether models apply uniform contextual confidentiality to all user-disclosed values or treat categories differently.

Secret Type
GPT-OSS
DeepSeek
Qwen

SSN
Per-type rate
Per-type rate
Per-type rate

Credit card
Per-type rate
Per-type rate
Per-type rate

Email
Per-type rate
Per-type rate
Per-type rate

Phone
Per-type rate
Per-type rate
Per-type rate

Table 5: Layout of the probe-level hybrid leakage breakdown by secret type across cached model-probe responses.

### 5.6 Privacy Half-Life Exceeds the Observed Window

Privacy Half-Life is computed per model and per persuasion level from the drift-indexed leakage curve, using the threshold $\delta=0.1\cdot L(0)$ . When no observed drift length satisfies the stable-decay condition, the metric takes the censored value $\max(D_{obs})+1$, which in the present design equals 7. A censored value should be read as indicating that stable suppression was not established within the tested range, rather than as an estimate of where suppression would eventually occur.

### 5.7 Fuzzy Matching Supports the Hybrid Labels

As a post-hoc deterministic sensitivity check, we compute normalized fuzzy matching between each saved secret and saved model response using cached outputs only, at thresholds of 0.85 and 0.80. This check serves as an independent reference point for the hybrid labels that does not depend on an external judge. We treat fuzzy matching as auxiliary evidence rather than the primary metric because approximate string similarity is less expressive than the judge for partial or semantic disclosures.

Model
Regex
Fuzzy 0.85
Fuzzy 0.80
Hybrid

DeepSeek-R1
Exact
Strict fuzzy
Lenient fuzzy
Regex + judge

GPT-OSS-120B
Exact
Strict fuzzy
Lenient fuzzy
Regex + judge

Qwen3-VL-235B
Exact
Strict fuzzy
Lenient fuzzy
Regex + judge

Table 6: Probe-level detector configurations applied to cached model outputs.

Figure 5: Hybrid leakage heatmap by drift length and persuasion intensity for each model.

## 6 Discussion

PrivDrift is motivated by the observation that active-context privacy sits between two well-studied regimes. It is not a memorization problem, since the secret is supplied at inference time, and it is not a conventional jailbreak, since the requested content is not harmful in itself and was legitimately provided by the user. Evaluating it therefore requires a setting in which the presence of the secret is guaranteed and the behavior of interest is the model's decision about whether to reproduce it.

The design choices in PrivDrift reflect this framing. Holding the secret, drift length, and probe fixed while rewriting surface content allows behavioral differences to be attributed to the controlled factors rather than to incidental phrasing. Content-dense drift turns ensure that increasing $d$ meaningfully changes the distance between disclosure and probe, as reflected in the growth of dialogue length across drift levels in Appendix A. Applying all three probes to every dialogue produces paired data, which supports both repeated-measures tests across persuasion levels and a dialogue-level vulnerability view.

The layered detector reflects a tradeoff between reproducibility and coverage. Deterministic matching is transparent and fully reproducible but misses paraphrased or partially masked disclosures, while an LLM judge captures those cases at the cost of introducing its own error profile. Reporting regex, fuzzy, and hybrid labels together lets readers see how much each layer contributes and choose the label best suited to their purposes.

Privacy Half-Life is designed to summarize the drift dimension in a way that is robust to noise. Leakage curves estimated from finite samples can fluctuate, and a naive threshold-crossing rule could mistake a momentary decrease for genuine suppression. Requiring that all larger observed drift lengths remain below threshold makes the metric conservative: it credits suppression only when it is sustained. A natural extension is to pair $\tau$ with confidence bands obtained by bootstrapping the underlying curve.

More broadly, the framework is intended to support a shift in how privacy is evaluated for deployed assistants. Many practical safeguards operate at the level of prompts, system instructions, and context management rather than model weights, and their effectiveness depends on how models respond to later requests for earlier context. Controlled benchmarks of this kind can help developers test whether such safeguards behave consistently across secret categories, drift lengths, and interaction styles.

## 7 Limitations and Future Work

##### Bounded drift range.

PrivDrift evaluates drift lengths up to $d=6$ , corresponding to bounded short-to-medium conversational drift rather than full saturation of modern context windows. Analyses based on the framework should therefore be interpreted as characterizing behavior under bounded active-context drift, not as a complete characterization of privacy behavior across full long-context windows.

##### Fixed-format secrets.

The benchmark focuses on structured secrets such as SSNs, credit card numbers, emails, and phone numbers. These are easier to detect than unstructured sensitive information such as health status, family circumstances, immigration status, or financial hardship. Future work should extend the benchmark to non-fixed sensitive attributes.

##### Active context, not cross-session memory.

PrivDrift evaluates recoverability within the active conversation context. It does not test training-data memorization, personalization memory, or cross-session retention.

##### Synthetic and human-authored dialogues.

The dataset contains 900 synthetic dialogues and 100 human-authored dialogues inserted throughout the benchmark rather than stored as a separate block. This construction improves realism relative to a purely synthetic benchmark while preserving control over secret type, drift length, and probe structure. However, the current cached evaluation metadata does not preserve which evaluated rows correspond to the human-authored dialogues, so we treat the benchmark in aggregate and leave a separate human-versus-synthetic comparison for future work.

##### Detector scope.

Leakage labels are based on surface-form outputs using deterministic matching, fuzzy matching, and an external LLM judge. We report regex-only, fuzzy, and hybrid labels separately to make detector behavior transparent, but we do not yet report formal inter-annotator agreement between humans and the automated methods. Future work should include a larger manual audit of borderline partial disclosures and obfuscations.

##### API-served models and reproducibility.

We evaluate API-served models due to compute and budget constraints. These models may change over time. We mitigate this by logging model identifiers, prompts, timestamps, cached responses, and paired statistics on identical dialogue sets.

##### Artifact availability.

We plan to release the benchmark generation code, probe templates, evaluation scripts, aggregate result files, and a sanitized subset of generated dialogues. Because the benchmark contains synthetic PII-like strings, we will release examples with non-real identifiers and generation templates sufficient to reproduce the benchmark without exposing realistic identifiers.

## 8 Conclusion

We introduce PrivDrift, a benchmark for auditing active-context privacy behavior under topic drift and persuasion intensity. The framework combines a controlled scaffold-then-rewrite dialogue generator, a three-level probing protocol, a layered detector combining normalization-based matching, fuzzy matching, and an LLM judge, and a stability-based Privacy Half-Life metric. Applied to three LLMs with 1,000 dialogues per model and paired probes, it supports dialogue-level and probe-level analyses of how drift length, persuasion style, and secret type relate to re-disclosure. We hope PrivDrift serves as a practical starting point for evaluating privacy inside active conversations, and as a foundation for extensions to longer contexts, unstructured sensitive attributes, and memory-augmented assistants.

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

All models
Secret type
Chi-square, 4 levels
Per model

All models
Drift length
Chi-square, 6 levels
Per model

All models
Persuasion
Chi-square, 3 levels
Per model

Table 8: Probe-level association tests between hybrid leakage and benchmark variables, computed separately for each model.

## Appendix C Dialogue Construction Figure

Figure 6: Dialogue construction details.