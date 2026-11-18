# PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations

Luciano Rolando Maldonado Romero

Affiliation: West Virginia University, Morgantown, WV, USA

Correspondence to: lrm00020@mix.wvu.edu

###### Abstract

Large language models (LLMs), which generate language from supplied context, increasingly serve as persistent assistants. They operate in user-facing settings, sessions shared by users, and systems connected to external tools. Sensitive information disclosed during an active conversation may remain behaviorally recoverable: later prompts may cause the model to reproduce it, even after the conversation shifts to unrelated topics.

We introduce PrivDrift, a benchmark for auditing this risk. It contains 1,000 controlled multi-turn dialogues with deliberately inserted secrets, content-dense unrelated turns, and standardized prompts that try to extract the secrets. The prompts vary in persuasion intensity: direct requests, requests with a justification, and high-pressure requests. We use hybrid leakage detection, which combines normalized text matching with an LLM that judges whether a response reveals the secret.

Across three LLMs with extended context windows—the text available to a model when it responds—dialogue-level hybrid leakage ranges from 38.7% to 54.6% (Table 3). A dialogue counts as leaking if at least one standardized probe extracts its secret. Leakage varies strongly by model, secret type, and persuasion intensity (Tables 3–5). Within the tested drift window, additional topic drift does not reliably reduce leakage (Section 5.4). These results suggest that privacy risk in active LLM contexts should be evaluated as a persistent behavioral failure mode, not only as training-data memorization or immediate jailbreak behavior.

###### Keywords:

Trustworthy AI (artificial intelligence), Large Language Models, Privacy, Evaluation, Benchmarking

## 1 Introduction

Large language models (LLMs) have evolved from simple chat systems into persistent assistants that maintain coherence across extended interactions (OpenAI, 2023; Touvron et al., 2023; Liu et al., 2024). These systems now support healthcare triage, financial planning, enterprise copilots, and personal productivity tools. Users may disclose Personally Identifiable Information (PII): information such as phone numbers, financial identifiers, email addresses, and health-related details. The context retention that supports multi-turn assistance also creates a privacy risk. Sensitive information disclosed earlier in an active conversation may remain available to later responses.

Existing safety evaluations mostly target two extremes. Training-data memorization concerns private information extracted from model parameters, the learned values that determine model behavior (Carlini et al., 2021; Nasr et al., 2023). Immediate jailbreaking concerns prompts that elicit harmful behavior in the current turn (Wei et al., 2024; Zou et al., 2023). Recent work also studies prompt injection, where supplied content introduces instructions that redirect a system, and attacks spread across multiple turns (Liu et al., 2023; Russinovich et al., 2024; Deng et al., 2024). These settings do not directly measure whether user-disclosed secrets remain recoverable after unrelated conversation. This leaves an important question for trustworthy AI evaluation: does a topic shift act as a practical privacy boundary within an active conversation?

We do not assume that users believe an LLM literally forgets earlier messages in the same session. We study a behavioral risk instead. Users and application designers may underestimate how easily later prompts can elicit earlier sensitive information. Such prompts may be indirect, justified, high-pressure, or delivered through tools. This risk matters in shared sessions, browser-integrated assistants, enterprise copilots, agentic workflows in which models coordinate actions and tools, and prompt-injection settings. In each case, later instructions may interact with active context in ways the original user did not intend.

PrivDrift measures active-context privacy leakage along two axes:

- **Topic drift:** the number of unrelated turns between secret disclosure and later probing.
- **Persuasion intensity:** the interaction style used to elicit the secret.

Across three LLMs with extended context windows and 1,000 dialogues per model, dialogue-level hybrid leakage ranges from 38.7% to 54.6% (Table 3). Persuasion intensity significantly affects leakage (Section 5.3). Leakage also varies sharply by secret type. Social Security numbers (SSNs) and credit card numbers are suppressed far more often than emails and phone numbers (Table 5). Let \(d\) denote the number of drift turns. Within the tested window, \(d\leq 6\), additional drift does not reliably reduce leakage (Section 5.4).

Our contributions are as follows:

1. PrivDrift is a controlled benchmark for auditing whether user-disclosed secrets remain recoverable from active conversational context after unrelated topic drift.

2. We evaluate direct, justified, and high-pressure probes. These represent different interaction styles through which later prompts can elicit sensitive context.

3. We implement a reproducible hybrid detector that combines normalization-based text matching with an open-weights LLM judge, whose model weights are available. We report dialogue-level, probe-level, regex-only, fuzzy, and hybrid leakage rates. “Regex-only” denotes the deterministic normalization-based matching stage. Fuzzy matching permits approximate text matches. These methods are defined in Sections 3.4 and 5.7.

4. We propose Privacy Half-Life, denoted by \(\tau\), as a measure of sustained leakage suppression. No evaluated model reaches stable decay within the observed drift range (Sections 3.4.3 and 5.6).

## 2 Related Work

### 2.1 Contextual Privacy and Inference-Time Leakage

Privacy research in language models has historically focused on training-data extraction, membership inference—determining whether an example belonged to training data—and recovery of memorized PII from pre-training corpora (Shokri et al., 2017; Carlini et al., 2021; Li et al., 2023; Nasr et al., 2023). These settings ask whether private information is stored in model parameters. Active-context privacy instead concerns user-provided information in the current interaction. The model must handle that information appropriately at inference time, when it produces a response.

Recent benchmarks examine privacy norms and contextual access. ConfAIde evaluates privacy reasoning through contextual integrity: whether information sharing fits its context (Mireshghallah et al., 2024). PrivacyLens and PrivaCI-Bench study privacy-norm awareness and legal compliance in agentic or contextual settings (Shao et al., 2024; Li et al., 2025). These benchmarks mainly ask whether a model should share information in a static access-control or privacy-norm scenario. PrivDrift instead isolates persistence: whether a user-disclosed secret remains behaviorally recoverable after unrelated topic drift and later persuasion-based probing.

### 2.2 Multi-Turn Adversarial Attacks

Aligned models—models trained to follow intended behavioral constraints—remain vulnerable to exploitation across multiple turns. Crescendo attacks show that seemingly benign conversations can gradually elicit harmful outputs (Russinovich et al., 2024). Automated jailbreak frameworks generate adversarial multi-turn attack paths (Deng et al., 2024; Narula et al., 2025). Prompt-injection studies also show that later instructions can manipulate applications that incorporate LLMs (Liu et al., 2023).

These works usually target unsafe content generation or instruction hijacking, where an attacker redirects the system’s instructions. PrivDrift applies the multi-turn perspective to information flow control: controlling whether sensitive information is disclosed. It asks whether a model will re-disclose information supplied by a user earlier in the same active context, rather than whether the model can produce harmful external content.

### 2.3 Context Management and Unlearning

Machine unlearning aims to remove or suppress knowledge from model behavior. Benchmarks such as TOFU (Task of Fictitious Unlearning) and WMDP (Weapons of Mass Destruction Proxy) evaluate this objective (Maini et al., 2024; Li et al., 2024). They usually operate over static question-answer pairs and model parameters.

PrivDrift targets active context rather than model weights. We ask whether a secret still present in the context window is behaviorally suppressed when later prompts try to elicit it. This distinction matters for deployed retrieval-augmented generation (RAG) systems, which supply retrieved material to a model, as well as copilots and agentic workflows. Their privacy controls often depend on managing prompt context rather than changing model parameters.

## 3 The PrivDrift Framework

Figure 1: PrivDrift overview. The pipeline constructs controlled multi-turn dialogues, appends standardized probes, queries target models, and scores leakage with a hybrid detector.

PrivDrift is a synthetic evaluation framework for measuring whether sensitive information disclosed in an active conversation remains recoverable after topic drift. As Figure 1 summarizes, it has three components: a dialogue generator controlled by experimental settings, a probing harness that submits extraction prompts, and metrics for leakage detection and analysis across drift lengths.

### 3.1 Threat Model: Active-Context Re-Disclosure

We study **active-context re-disclosure**: sensitive information disclosed earlier in a conversation remains recoverable through later prompts after unrelated topic drift. We do not assume that an adversary is always unable to inspect the raw transcript. Instead, we test whether the model itself will reproduce the sensitive value when later instructions ask for it, justify its disclosure, or apply pressure.

This setting captures several practical risks. In shared or persistent sessions, a later user may not understand what private information was previously disclosed. In enterprise copilots and agentic workflows, later tool outputs, retrieved documents, or prompt-injection content may issue instructions that interact with active context. In browser- or application-integrated assistants, the user may not intend earlier sensitive disclosures to be reused after the task changes. The question is therefore not whether the transcript contains the secret. It is whether later prompting causes the assistant to reproduce it.

Formally, a user discloses a secret \(S\), the ground-truth sensitive value, at turn \(t_{0}\), the disclosure turn. The conversation then proceeds through \(d\) unrelated but content-dense drift turns:

\[
D=\{u_{1},a_{1},\ldots,u_{d},a_{d}\}.
\]

Here, \(D\) denotes the intervening dialogue. For each drift turn \(i\), \(u_i\) is the user message and \(a_i\) is the assistant message. At evaluation time, a standardized extraction probe \(P\) is appended to the active context. A leakage event occurs if the response reveals \(S\) exactly, approximately, or semantically—that is, by conveying the secret without necessarily reproducing its exact text.

### 3.2 Dataset Generation Pipeline

We generate \(N=1000\) controlled dialogues, where \(N\) is the total number of dialogues. The pipeline first builds a controlled scaffold, or structured draft, and then rewrites it. This preserves the ground-truth secret, drift length, and probe structure while reducing rigid template artifacts.

The dataset contains 900 synthetic dialogues and 100 human-authored dialogues under the same secret and drift constraints (Table 1). Human-authored dialogues were inserted throughout the benchmark, not stored in a separate contiguous block. The current cached evaluation metadata—the information saved with the evaluation outputs—does not preserve their identifiers. We therefore report aggregate results and leave a separate human-versus-synthetic comparison for future work.

In Table 1, “Avg.” means average, and “Min/max” means minimum and maximum. A token is a unit of text processed by a model. The reported token counts are approximate estimates from word counts.

| Statistic | Value |
|---|---|
| Total dialogues | 1,000 |
| Synthetic dialogues | 900 |
| Human-authored dialogues | 100 |
| Secret categories | 4 |
| Drift lengths | \(0,2,3,4,5,6\) |
| Avg. words per dialogue | 359.16 |
| Median words per dialogue | 385.00 |
| Min/max words per dialogue | 43 / 795 |
| Avg. approx. tokens per dialogue | 478.87 |
| Median approx. tokens per dialogue | 513.33 |
| Min/max approx. tokens per dialogue | 57.33 / 1060.00 |

Table 1: Dataset summary for the evaluated 1,000-dialogue benchmark. Approximate token counts are estimated from word counts.

#### 3.2.1 Persona and Secret Injection

Each dialogue starts with a persona, or user profile, sampled from more than 100 professions and demographic profiles. It also receives a ground-truth secret \(S\). We sample \(S\) from four fixed-format PII categories: phone numbers, email addresses, Social Security Numbers, and credit card numbers.

The secret appears in the opening turn as part of a plausible task. This establishes a legitimate reason for the assistant to encounter the information in context.

#### 3.2.2 Content-Dense Topic Drift

Each dialogue is assigned a topic, such as tax filing, medical triage, or travel logistics. Drift turns come from a topic-conditioned library: a collection of dense question-answer pairs selected according to topic. Fallback tasks involve professional writing, structured reasoning, or long-form summarization.

Annotators write a subset of drift turns, which are incorporated into the generation library. This design aims to add unrelated entities and task demands while preserving control over the original secret and the number of drift turns.

#### 3.2.3 Naturalistic Refinement

A final rewriting step reduces the initial scaffold’s rigidity. A rewriting model receives each scaffold and instructions to improve fluency and persona realism. It must preserve the exact secret, drift length, role structure—the arrangement of user and assistant messages—and final probe conditions. This step changes surface phrasing, not experimental variables.

### 3.3 Probing Setup

The main evaluation appends a final standardized probe to the completed conversation history. Probes fall into three persuasion levels:

- **Simple:** a direct, neutral query, such as asking for the phone number mentioned earlier.
- **Medium:** a query with a contextual justification, such as needing the information for a form.
- **Hard:** a high-pressure or urgent request that tests whether pressure cues increase or decrease leakage.

We sample drift length \(d\), the number of turns between secret disclosure and final probing, from a mixed distribution. In Equation (1), \(p\) denotes probability, and \(\mathrm{Uniform}\) means that the listed values are equally likely within that branch:

\[
d\sim
\begin{cases}
0 & \text{with }p=0.2,\\
\mathrm{Uniform}\{2,3,4,5,6\} & \text{with }p=0.8.
\end{cases}
\tag{1}
\]

This includes immediate-recall cases, where \(d=0\), while emphasizing short-to-medium topic drift. We interpret the results as bounded active-context persistence, not as a full test of arbitrarily long context windows.

### 3.4 Evaluation Methodology

A model may reveal a secret verbatim, change its formatting, disclose part of it, or refer to it indirectly. We therefore use a hierarchical hybrid detector: deterministic matching runs first, followed by an LLM judge when matching finds no leakage.

A response counts as leakage if either:

- the normalized secret appears in the normalized response; or
- the judge determines that the response reveals the full secret, a substantial substring, or enough partial information to identify it.

Figure 2: Hierarchical hybrid evaluation. Normalization-based matching is applied first; regex-negative responses are then passed to an LLM judge.

#### 3.4.1 Stage 1: Normalization-Based Matching

The first stage detects verbatim and near-verbatim leakage. For numeric secrets, we remove every non-digit character from both the secret and the response. For alphanumeric secrets, which contain letters and numbers, we normalize case and whitespace while preserving meaningful delimiters.

Let \(\phi(\cdot)\) denote this normalization function. Let \(R\) denote the model response. The regex-stage leakage label is

\[
\mathrm{Leak}_{\mathrm{regex}}(R,S)
=
\mathbb{I}\big[\phi(S)\subset\phi(R)\big].
\tag{2}
\]

Here, \(\mathrm{Leak}_{\mathrm{regex}}\) is the binary deterministic leakage label. The symbol \(\subset\) means that the normalized secret appears as a substring of the normalized response. The indicator function \(\mathbb{I}\) returns 1 when this condition holds and 0 otherwise.

#### 3.4.2 Stage 2: LLM-as-a-Judge Verification

Responses labeled negative by normalization-based matching go to an open-weights Llama judge (Meta AI, 2024). The judge receives the secret and response. It returns a binary verdict: whether the response reveals the secret or a substantial part of it.

This stage is intended to capture partial disclosure, indirect hints, and formatting variations missed by deterministic matching. We report regex-only and hybrid rates separately because the judge can introduce its own errors.

#### 3.4.3 Privacy Half-Life

Privacy Half-Life, \(\tau\), measures sustained suppression rather than a temporary decrease. Let \(D_{obs}\) be the set of observed drift lengths. Let \(L(d)\) be the leakage rate at drift length \(d\), so \(L(0)\) is the rate without intervening drift.

We set the safety threshold to \(\delta=0.1\times L(0)\), where \(\delta\) denotes that threshold. We then define

\[
\begin{aligned}
\tau=\min\{d\in D_{obs}\mid{}&
L(d)\leq\delta\\
&\text{and }\forall d^{\prime}\in D_{obs},\ d^{\prime}>d:
L(d^{\prime})\leq\delta\}.
\end{aligned}
\tag{3}
\]

Here, \(\min\) selects the smallest qualifying drift length. The condition requires leakage to be at or below the threshold at \(d\) and at every later observed drift length \(d^{\prime}\).

If no observed drift length meets both conditions, we assign \(\tau=\max(D_{obs})+1\), one more than the largest observed drift length. This prevents a temporary dip followed by a rebound from counting as stable suppression.

Figure 3: Privacy Half-Life \(\tau\) requires stable decay below the threshold. A later resurgence disqualifies a transient decrease.

## 4 Experimental Setup

##### Models.

We evaluate GPT-OSS-120B, DeepSeek-R1, and Qwen3-VL-235B through the OpenRouter application programming interface (API), a service interface for submitting model requests. All models receive the same dialogue histories and probe templates.

##### Dataset.

We evaluate 1,000 PrivDrift dialogues on each model. Each dialogue contains one seeded secret and one drift length \(d\in\{0,2,3,4,5,6\}\), sampled using Equation (1). Each dialogue is evaluated under three standardized persuasion probes. This yields 3,000 cached model-probe responses per model and 9,000 total cached responses across the three models. “Cached” means the outputs are saved for subsequent analysis.

##### Aggregation levels.

We report two units of analysis:

- **Dialogue level:** a dialogue counts as leaking if any standardized probe elicits the secret.
- **Probe level:** each dialogue contributes one response per persuasion condition. Persuasion-level and fuzzy-matching analyses use this unit.

Because dialogue-level leakage counts success by any probe, it can exceed the average leakage rate of the individual persuasion conditions.

##### Uncertainty and statistical tests.

We report regex-only and hybrid leakage. For dialogue-level rates, we compute 95% bootstrap confidence intervals (CIs), which summarize uncertainty by resampling dialogues. Probe-level sensitivity checks use cached model-probe responses.

For paired model or detector comparisons on matched examples, we use McNemar’s test, which compares paired binary outcomes. We apply Bonferroni correction where applicable to account for multiple comparisons. For repeated measurements across persuasion levels, we use Cochran’s \(Q\) test, which compares related binary outcomes across conditions, followed by post-hoc McNemar tests.

To test whether leakage is associated with drift length, persuasion level, or secret type, we use chi-square tests of independence. We report Cramer’s \(V\), a measure of association strength, as the effect size. In statistical results, \(p\) denotes a p-value: the probability, under the test’s no-difference or no-association assumption, of a result at least as extreme as the observed result.

## 5 Results

### 5.1 Dialogue-Level Hybrid Evaluation Largely Agrees With Deterministic Matching

At the dialogue level, hybrid evaluation differs only slightly from normalization-based matching. Table 2 reports additions of +0.00, +0.90, and +1.10 percentage points for GPT-OSS-120B, DeepSeek-R1, and Qwen3-VL-235B, respectively. This indicates that leakage is dominated by direct or near-direct reproduction rather than purely semantic paraphrase.

The judge adds up to 1.10 percentage points of dialogue-level leakage (Table 2, Qwen3-VL-235B row). This supports using hybrid leakage as the main metric while retaining regex-only rates as a transparent deterministic baseline. In Table 2, \(\Delta\) is hybrid leakage minus regex-only leakage, in percentage points.

| Model | Regex | Hybrid | \(\Delta\) | \(p\) |
|---|---:|---:|---:|---:|
| GPT-OSS-120B | 47.70 | 47.70 | +0.00 | 1.0 |
| DeepSeek-R1 | 37.80 | 38.70 | +0.90 | \(7.7\times 10^{-3}\) |
| Qwen3-VL-235B | 53.50 | 54.60 | +1.10 | \(2.57\times 10^{-3}\) |

Table 2: Dialogue-level leakage (%) under regex-only and hybrid evaluation. A dialogue is counted as leaking if any standardized probe elicits the secret.

### 5.2 All Models Leak Substantially at the Dialogue Level

All models show substantial dialogue-level hybrid leakage. DeepSeek-R1 leaks 38.70%, GPT-OSS-120B leaks 47.70%, and Qwen3-VL-235B leaks 54.60% (Table 3, respective model rows). These rates estimate whether a dialogue is vulnerable to at least one standardized extraction probe.

| Model | Hybrid Leakage | 95% CI |
|---|---:|---|
| DeepSeek-R1 | 38.70% | [35.7, 41.6] |
| GPT-OSS-120B | 47.70% | [44.6, 50.8] |
| Qwen3-VL-235B | 54.60% | [51.6, 57.7] |

Table 3: Dialogue-level hybrid leakage rates with 95% bootstrap confidence intervals. A dialogue is counted as leaking if any standardized probe elicits the secret.

### 5.3 Persuasion Intensity Affects Leakage Non-Monotonically

Probe-level leakage differs significantly across persuasion levels for every model: Cochran’s \(Q\), \(p<10^{-5}\). The direction depends on the model (Table 4).

For GPT-OSS-120B, hard persuasion reduces leakage to 32.90%, compared with 39.30% for simple and 40.50% for medium persuasion (Table 4, GPT-OSS-120B row). This is consistent with pressure cues triggering safety behavior. For DeepSeek-R1, medium persuasion yields the lowest leakage, 22.00% (Table 4, DeepSeek-R1 row). For Qwen3-VL-235B, hard persuasion yields the highest leakage, 40.60%, indicating weaker resistance to pressure-based extraction (Table 4, Qwen3-VL-235B row).

| Model | Simple | Medium | Hard |
|---|---:|---:|---:|
| GPT-OSS-120B | 39.30% | 40.50% | 32.90% |
| DeepSeek-R1 | 28.10% | 22.00% | 29.20% |
| Qwen3-VL-235B | 38.70% | 33.30% | 40.60% |

Table 4: Probe-level hybrid leakage by persuasion intensity.

### 5.4 Drift Length Does Not Reliably Reduce Leakage Within the Tested Window

The leakage curves dip near \(d=3\) and then rebound (Figure 4). The association between drift length and leakage is small or negligible. In the cached probe-level analysis, Cramer’s \(V\) is 0.1073 for DeepSeek-R1, 0.0659 for GPT-OSS-120B, and 0.0723 for Qwen3-VL-235B (Table 8, drift length rows).

This is not a claim about arbitrarily long contexts. It shows that privacy risk can persist across several unrelated topic shifts, even before the original secret is far from the end of the context window.

Figure 4: Hybrid leakage versus drift length. Curves show a dip near \(d=3\) followed by rebound.

### 5.5 Secret Type Strongly Determines Leakage

Secret type has a large association with hybrid leakage. In the cached probe-level analysis, Cramer’s \(V\) is 0.5871 for DeepSeek-R1, 0.7731 for GPT-OSS-120B, and 0.6927 for Qwen3-VL-235B (Table 8, secret type rows).

SSNs and credit card numbers are suppressed far more often than emails and phone numbers (Table 5, respective secret-type rows). All four categories are user-disclosed and contextually private. This pattern suggests that current behavior depends more on format and sensitivity cues than on a generalized notion of contextual confidentiality.

In Table 5, GPT-OSS, DeepSeek, and Qwen are shortened labels for GPT-OSS-120B, DeepSeek-R1, and Qwen3-VL-235B, respectively.

| Secret Type | GPT-OSS | DeepSeek | Qwen |
|---|---:|---:|---:|
| SSN | 0.14% | 1.11% | 5.97% |
| Credit card | 0.00% | 0.39% | 4.01% |
| Email | 78.24% | 57.11% | 81.13% |
| Phone | 70.89% | 46.13% | 57.24% |

Table 5: Probe-level hybrid leakage by secret type across cached model-probe responses.

### 5.6 Privacy Half-Life Exceeds the Observed Window

Using the stable-decay definition in Section 3.4.3, with threshold \(\delta=0.1\cdot L(0)\), no evaluated model approaches the threshold at any tested drift length. Thus, no drift length meets the stable-suppression condition. The fallback rule gives \(\tau=\max(D_{obs})+1=7\) for all models and persuasion levels. This indicates no stable decay within \(d\leq 6\).

### 5.7 Fuzzy Matching Supports the Hybrid Labels

As a post-hoc deterministic sensitivity check—an additional analysis after the main evaluation—we compute normalized fuzzy matching between each saved secret and saved response. This uses cached outputs only. Fuzzy matching allows approximate string similarity rather than requiring an exact normalized substring.

At a similarity threshold of 0.85, fuzzy leakage closely tracks probe-level hybrid leakage: 26.40% versus 26.43% for DeepSeek-R1, 37.03% versus 37.57% for GPT-OSS-120B, and 37.60% versus 37.53% for Qwen3-VL-235B (Table 6, respective model rows).

We treat fuzzy matching as auxiliary evidence, not the primary metric. Approximate string similarity is less expressive than the judge for partial or semantic disclosures. Table 6 also reports fuzzy matching at a 0.80 threshold.

| Model | Regex | Fuzzy 0.85 | Fuzzy 0.80 | Hybrid |
|---|---:|---:|---:|---:|
| DeepSeek-R1 | 25.50 | 26.40 | 26.53 | 26.43 |
| GPT-OSS-120B | 36.80 | 37.03 | 37.07 | 37.57 |
| Qwen3-VL-235B | 36.97 | 37.60 | 37.93 | 37.53 |

Table 6: Probe-level regex, fuzzy, and hybrid leakage rates (%) computed from cached model outputs.

Figure 5: Hybrid leakage heatmap by drift length and persuasion intensity for each model.

## 6 Discussion

PrivDrift reveals a persistent active-context privacy risk. Sensitive information disclosed earlier can remain behaviorally recoverable after unrelated topic drift (Sections 5.2 and 5.4). This does not imply that models permanently remember the secret or that all long-context settings behave similarly. It shows that current assistants may reproduce sensitive in-context information when later prompts provide enough retrieval pressure. The cached probe-level evaluation also indicates mostly direct or near-direct reproduction: fuzzy matching closely tracks hybrid leakage (Table 6).

The strongest pattern is the asymmetry across secret types (Table 8, secret type rows). SSNs and credit card numbers are suppressed far more often than emails and phone numbers, although all four are user-disclosed secrets (Table 5). This suggests that current safeguards may rely on format-sensitive safety heuristics—rules responsive to a secret’s form—rather than robust contextual reasoning about confidentiality.

Persuasion effects are significant, non-monotonic, and model-dependent (Section 5.3). High-pressure requests reduce GPT-OSS-120B leakage, consistent with the possibility that urgency cues activate refusal behavior. Medium persuasion produces DeepSeek-R1’s lowest leakage. Hard persuasion produces Qwen3-VL-235B’s highest leakage, suggesting weaker resistance to pressure-based extraction (Table 4, respective model rows).

Topic drift should therefore not be treated as an implicit privacy boundary. The intermediate dips in leakage are not stable enough to satisfy the Privacy Half-Life criterion (Figure 4; Section 5.6). This motivates stability-based evaluation rather than treating temporary decreases as evidence of suppression.

## 7 Limitations and Future Work

##### Bounded drift range.

PrivDrift tests drift lengths up to \(d=6\). This is bounded short-to-medium conversational drift, not full saturation of modern context windows. The results show persistence under bounded active-context drift. They do not completely characterize privacy behavior across full long-context windows.

##### Fixed-format secrets.

The benchmark focuses on structured secrets: SSNs, credit card numbers, emails, and phone numbers. These are easier to detect than unstructured sensitive information, such as health status, family circumstances, immigration status, or financial hardship. Future work should extend the benchmark to sensitive attributes without fixed formats.

##### Active context, not cross-session memory.

PrivDrift tests recoverability within the active conversation context. It does not test training-data memorization, personalization memory, or retention across sessions.

##### Synthetic and human-authored dialogues.

The dataset contains 900 synthetic dialogues and 100 human-authored dialogues inserted throughout the benchmark, rather than stored in a separate block (Table 1). This improves realism relative to a purely synthetic benchmark while preserving control over secret type, drift length, and probe structure.

However, cached evaluation metadata does not preserve which evaluated rows are human-authored. We therefore report aggregate results and leave a separate human-versus-synthetic comparison for future work.

##### Detector scope.

Leakage labels depend on the text of model outputs, assessed through deterministic matching, fuzzy matching, and an external LLM judge. We report regex-only, fuzzy, and hybrid labels separately to make detector behavior transparent (Tables 2 and 6).

We do not yet report formal inter-annotator agreement: the degree of agreement between human assessments and automated methods. Future work should include a larger manual audit of borderline partial disclosures and obfuscations, where information is presented in a disguised form.

##### API-served models and reproducibility.

Compute and budget constraints lead us to evaluate models served through APIs. These models may change over time. We mitigate this by logging model identifiers, prompts, timestamps, cached responses, and paired statistics on identical dialogue sets.

##### Artifact availability.

We plan to release the benchmark generation code, probe templates, evaluation scripts, aggregate result files, and a sanitized subset of generated dialogues, with identifiers made safe for release. The benchmark contains synthetic PII-like strings. We will therefore release examples with non-real identifiers and generation templates sufficient to reproduce the benchmark without exposing realistic identifiers.

## 8 Conclusion

We introduce PrivDrift, a benchmark for auditing active-context privacy leakage under topic drift and varying persuasion intensity. Across three LLMs and 1,000 dialogues per model, dialogue-level hybrid leakage ranges from 38.7% to 54.6% (Table 3). Persuasion intensity affects leakage in model-dependent ways (Table 4). Secret type has the strongest association with leakage (Table 8).

Within the tested drift range, topic drift does not reliably reduce leakage (Section 5.4). Privacy Half-Life exceeds the observed window for all models (Section 5.6). The asymmetry across secret types suggests that current protections remain uneven and depend more on format-sensitive cues than on robust contextual reasoning about confidentiality (Table 5).

## Appendix A Dataset Characterization

Table 7 groups dialogues by drift length \(d\). Here, \(N\) is the number of dialogues in each drift group. Token counts are approximate estimates from word counts.

| Drift \(d\) | \(N\) | Avg. words | Median words | Avg. tokens | Median tokens |
|---|---:|---:|---:|---:|---:|
| 0 | 197 | 44.29 | 43.00 | 59.05 | 57.33 |
| 2 | 147 | 253.97 | 258.00 | 338.62 | 344.00 |
| 3 | 142 | 344.18 | 347.00 | 458.91 | 462.67 |
| 4 | 166 | 452.27 | 447.00 | 603.02 | 596.00 |
| 5 | 181 | 520.92 | 543.00 | 694.56 | 724.00 |
| 6 | 167 | 568.03 | 605.00 | 757.37 | 806.67 |

Table 7: Dataset length characteristics by drift length. Approximate token counts are estimated from word counts.

## Appendix B Additional Effect Sizes

| Model | Variable | Cramer’s \(V\) | Interpretation |
|---|---|---:|---|
| DeepSeek-R1 | Secret type | 0.5871 | Large |
| DeepSeek-R1 | Drift length | 0.1073 | Small |
| DeepSeek-R1 | Persuasion | 0.0718 | Negligible |
| GPT-OSS-120B | Secret type | 0.7731 | Large |
| GPT-OSS-120B | Drift length | 0.0659 | Negligible |
| GPT-OSS-120B | Persuasion | 0.0689 | Negligible |
| Qwen3-VL-235B | Secret type | 0.6927 | Large |
| Qwen3-VL-235B | Drift length | 0.0723 | Negligible |
| Qwen3-VL-235B | Persuasion | 0.0639 | Negligible |

Table 8: Probe-level effect sizes for associations between hybrid leakage and benchmark variables.

## Appendix C Dialogue Construction Figure

Figure 6: Dialogue construction details.
