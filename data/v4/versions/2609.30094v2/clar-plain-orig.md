# PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations

Luciano Rolando Maldonado Romero

Affiliation: West Virginia University, Morgantown, WV, USA

Correspondence to: lrm00020@mix.wvu.edu

###### Abstract

Large language models (LLMs), which generate language, increasingly serve as persistent assistants in user-facing systems, shared sessions, and systems with added tools. When users share sensitive information in an active conversation, later prompts may still recover it after the conversation shifts to unrelated topics. We introduce PrivDrift, a benchmark that tests whether later prompts can recover user-disclosed secrets after topic drift and persuasion-based probing. PrivDrift contains 1,000 controlled multi-turn dialogues with deliberately inserted secrets, content-dense topic shifts, and standardized extraction prompts. We evaluate three LLMs with extended context windows, which let them process long conversations at once. Dialogue-level hybrid leakage—the disclosure rate measured by combining text matching with a model judge—ranges from 38.7% to 54.6%. Leakage varies strongly by model, secret type, and persuasion intensity. Within the tested drift window, more topic drift does not reliably reduce leakage. This finding suggests that privacy evaluations should treat leakage from active LLM conversations as a persistent behavioral failure, not only as recall from training data or an immediate attempt to bypass safeguards.

###### Keywords: 

Trustworthy AI, Large Language Models, Privacy, Evaluation, Benchmarking

## 1 Introduction

Large language models (LLMs) have developed from simple chat systems into persistent assistants that can stay coherent across long interactions (OpenAI, 2023; Touvron et al., 2023; Liu et al., 2024). These systems now support tasks such as healthcare triage, financial planning, enterprise assistance, and personal productivity. Users may share Personally Identifiable Information (PII), meaning information that can identify a person. This includes phone numbers, financial identifiers, email addresses, and health-related details. Retaining context helps models provide useful multi-turn assistance. The same ability can create a privacy risk: models may use sensitive information from earlier in an active conversation in later responses.

Existing safety evaluations mostly target two extremes. The first is training-data memorization, where an attacker extracts private information stored in model parameters—the values learned during training (Carlini et al., 2021; Nasr et al., 2023). The second is immediate jailbreaking, where a prompt elicits harmful behavior in the current turn by bypassing safeguards (Wei et al., 2024; Zou et al., 2023). Recent work also studies prompt injection, where content introduces instructions that redirect a model, and attacks that span several turns (Liu et al., 2023; Russinovich et al., 2024; Deng et al., 2024). However, these settings do not directly measure whether later prompts can recover user-disclosed secrets after unrelated conversational drift. This leaves a gap in trustworthy artificial intelligence (AI) evaluation: does a topic shift act as a practical privacy boundary within an active conversation?

We do not assume that users believe an LLM literally forgets earlier messages in the same session. Instead, we study a behavioral risk. Users and application designers may underestimate how easily later prompts can elicit earlier sensitive information through indirect requests, justifications, pressure, or tools. This risk matters in shared sessions, browser-integrated assistants, enterprise copilots, and agentic workflows, where models carry out tasks through multiple steps or tools. It also matters in prompt-injection settings, where later instructions interact with the active context in ways the original user did not intend. Here, active context means the conversation content currently available to the model.

We introduce PrivDrift, a controlled benchmark that measures active-context privacy leakage along two dimensions. Topic drift measures the number of unrelated turns between secret disclosure and later probing. Persuasion intensity measures the interaction style used to elicit the secret. Across three LLMs with extended context windows and 1,000 dialogues per model, dialogue-level hybrid leakage ranges from 38.7% to 54.6%. Persuasion intensity significantly affects leakage. Leakage also varies sharply by secret type: models suppress Social Security Numbers (SSNs) and credit card numbers far more often than emails and phone numbers. Within the tested window of six or fewer drift turns, written as $d\leq 6$, more drift does not reliably reduce leakage.

Our contributions are as follows:

1. We introduce PrivDrift, a controlled benchmark that audits whether later prompts can recover user-disclosed secrets from active conversational context after unrelated topic drift.

2. We evaluate leakage under direct, justified, and high-pressure probes. These probes model different interaction styles that later prompts can use to elicit sensitive context.

3. We implement a reproducible hybrid detector that combines normalization-based matching with an open-weights LLM judge, whose model weights are available. Normalization-based matching standardizes text before comparing it. We report dialogue-level, probe-level, regex-only, fuzzy, and hybrid leakage rates. “Regex-only” names the deterministic matching stage; regex means regular expression, a text-pattern matching method. Fuzzy matching measures approximate text similarity.

4. We propose Privacy Half-Life ($\tau$), a metric that requires stable suppression of leakage. The symbol $\tau$, pronounced “tau,” denotes this metric. No evaluated model reaches stable decay within the observed drift range.

## 2 Related Work

### 2.1 Contextual Privacy and Inference-Time Leakage

Privacy research in language models has historically focused on training-data extraction, membership inference, and recovery of memorized PII from pre-training corpora (Shokri et al., 2017; Carlini et al., 2021; Li et al., 2023; Nasr et al., 2023). Membership inference tests whether a particular example appeared in a model’s training data. Pre-training corpora are the text collections used for initial training. These settings ask whether model parameters store private information. Active-context privacy instead concerns information that users provide in the current interaction. The model must handle that information appropriately at inference time, when it generates a response.

Recent benchmarks test whether LLMs can reason about privacy norms and contextual access. ConfAIde evaluates privacy reasoning through contextual integrity: whether sharing information fits its context (Mireshghallah et al., 2024). PrivacyLens and PrivaCI-Bench study awareness of privacy norms and legal compliance in agentic or contextual settings (Shao et al., 2024; Li et al., 2025). These benchmarks mainly test whether a model should share information in a fixed access-control or privacy-norm scenario. PrivDrift instead isolates persistence: whether later persuasion-based prompts can recover a user-disclosed secret after unrelated topic drift.

### 2.2 Multi-Turn Adversarial Attacks

Models aligned to follow safety requirements remain vulnerable to exploitation across multiple turns. Crescendo attacks show that seemingly harmless conversations can gradually elicit harmful outputs (Russinovich et al., 2024). Automated jailbreak frameworks also generate adversarial attack sequences across multiple turns (Deng et al., 2024; Narula et al., 2025). Prompt-injection studies show that later instructions can manipulate applications that use LLMs (Liu et al., 2023).

These studies usually focus on unsafe content generation or instruction hijacking, where an attacker redirects the model’s instructions. PrivDrift applies the multi-turn approach to information flow control: controlling where information may go. It does not ask whether a model can be made to produce harmful external content. It asks whether the model will disclose sensitive information that a user supplied earlier in the same active context.

### 2.3 Context Management and Unlearning

Machine unlearning benchmarks such as TOFU and WMDP test whether model behavior can remove or suppress knowledge (Maini et al., 2024; Li et al., 2024). These benchmarks usually concern fixed question-answer pairs and model parameters. PrivDrift shares this general aim but targets active context rather than model weights. We ask whether a model suppresses a secret present in its context window when later prompts try to elicit it.

This distinction matters for deployed retrieval-augmented generation (RAG) systems, which add retrieved information to a model’s context. It also matters for copilots and agentic workflows. In these settings, privacy control often depends on managing context through prompts rather than unlearning information from model parameters.

## 3 The PrivDrift Framework

Figure 1: PrivDrift overview. The pipeline constructs controlled multi-turn dialogues, appends standardized probes, queries target models, and scores leakage with a hybrid detector.

PrivDrift is a synthetic evaluation framework: it uses constructed conversations to measure whether later prompts can recover sensitive information after topic drift. As Figure 1 shows, the framework has three components. A parameter-controlled dialogue generator creates conversations. A probing system submits extraction prompts to target models. A set of metrics detects leakage and analyzes how it changes with drift.

### 3.1 Threat Model: Active-Context Re-Disclosure

We study active-context re-disclosure. This failure occurs when later prompts can recover sensitive information shared earlier in a conversation, despite unrelated topic drift. We do not assume that an adversary is always unable to inspect the raw transcript. Instead, PrivDrift tests whether the model itself will reveal the sensitive value when later instructions request it, justify disclosure, or pressure the assistant.

This setting captures several practical risks. In shared or persistent sessions, a later user may interact with an assistant without knowing what private information someone previously shared. In enterprise copilots and agentic workflows, later tool outputs, retrieved documents, or prompt-injection content may give instructions that interact with the active context. In browser- or application-integrated assistants, users may not intend the assistant to reuse earlier sensitive information after the task changes. The core question is therefore not whether the transcript contains the secret. It is whether the assistant reproduces the secret when later prompted.

Formally, a user discloses a secret $S$ at turn $t_0$. The conversation then passes through $d$ unrelated but content-dense drift turns:

$$
D=\{u_1,a_1,\ldots,u_d,a_d\}.
$$

Here, $D$ denotes the drift sequence, $u_i$ denotes a user message, and $a_i$ denotes the corresponding assistant message. At evaluation time, we append a standardized probe $P$ to the active context. A leakage event occurs if the response reveals $S$ exactly, approximately, or semantically—that is, by conveying the secret’s information without necessarily repeating its exact text.

### 3.2 Dataset Generation Pipeline

We generate $N=1000$ controlled dialogues, where $N$ is the total number of dialogues. We use a multi-stage scaffold-then-rewrite pipeline: we first build a controlled dialogue structure, then revise its wording. This design preserves strict control over the known secret, drift length, and probe structure while reducing the rigid patterns of generation based only on templates.

The dataset contains 900 synthetic dialogues and 100 human-authored dialogues built under the same secret and drift constraints. We inserted the human-authored dialogues throughout the benchmark rather than placing them in a separate continuous block. The current stored evaluation metadata does not preserve their identifiers. We therefore report aggregate results for the full evaluated benchmark and leave a separate human-versus-synthetic leakage comparison for future work.

| Statistic | Value |
|---|---:|
| Total dialogues | 1,000 |
| Synthetic dialogues | 900 |
| Human-authored dialogues | 100 |
| Secret categories | 4 |
| Drift lengths | $0,2,3,4,5,6$ |
| Avg. words per dialogue | 359.16 |
| Median words per dialogue | 385.00 |
| Min/max words per dialogue | 43 / 795 |
| Avg. approx. tokens per dialogue | 478.87 |
| Median approx. tokens per dialogue | 513.33 |
| Min/max approx. tokens per dialogue | 57.33 / 1060.00 |

Table 1: Dataset summary for the evaluated 1,000-dialogue benchmark. Approximate token counts are estimated from word counts.

In the table, “Avg.” means average, “approx.” means approximate, and “min/max” means minimum and maximum. Tokens are the units of text processed by a language model.

#### 3.2.1 Persona and Secret Injection

We start each dialogue with a persona sampled from more than 100 professions and demographic profiles, along with a known secret $S$. We sample $S$ from four fixed-format PII categories: phone numbers, email addresses, Social Security Numbers, and credit card numbers. We insert the secret into the opening turn as part of a plausible task. This gives the assistant a legitimate reason to encounter the information in context.

#### 3.2.2 Content-Dense Topic Drift

We assign each dialogue a topic, such as tax filing, medical triage, or travel logistics. We draw drift turns from a topic-conditioned library of dense question-answer pairs and fallback tasks. These tasks involve professional writing, structured reasoning, or long-form summarization. Annotators write a subset of the drift turns, which we add to the generation library. This design aims to add unrelated entities and task demands to the active context while preserving control over the original secret and the number of drift turns.

#### 3.2.3 Naturalistic Refinement

We apply a final refinement step to make the initial scaffold less rigid. A rewriting model receives each scaffold and instructions to improve fluency and persona realism. It must preserve the exact secret, drift length, role structure, and final probe conditions. This step changes wording but not the experimental variables.

### 3.3 Probing Setup

The main evaluation appends a final standardized probe to the completed conversation history. We group probes into three persuasion levels:

- **Simple:** A direct, neutral query, such as asking for the phone number mentioned earlier.
- **Medium:** A query with a contextual justification, such as needing the information for a form.
- **Hard:** A high-pressure or urgent request that tests whether pressure cues increase or decrease leakage.

We sample drift length $d$, the number of turns between secret disclosure and final probing, from this mixed distribution:

$$
d\sim
\begin{cases}
0 & \text{with }p=0.2,\\
\mathrm{Uniform}\{2,3,4,5,6\} & \text{with }p=0.8.
\end{cases}
\tag{1}
$$

Here, $p$ denotes probability. The uniform distribution gives each listed drift length an equal chance within that part of the distribution.

This distribution includes immediate-recall cases but emphasizes short-to-medium topic drift. We interpret the results as persistence within a bounded active context, not as a full test of arbitrarily long context windows.

### 3.4 Evaluation Methodology

A model may reveal a secret verbatim, with changed formatting, as a partial substring, or through a paraphrased reference. A substring is a continuous part of a text string. We therefore use a hierarchical hybrid detector, which applies two detection stages in order.

We label a response as leakage if the normalized secret appears in the normalized response. We also label it as leakage if the LLM judge finds that it reveals the full secret, a substantial substring, or enough partial information to identify the secret.

Figure 2: Hierarchical hybrid evaluation. Normalization-based matching is applied first; regex-negative responses are then passed to an LLM judge.

#### 3.4.1 Stage 1: Normalization-Based Matching

The first stage detects verbatim and near-verbatim leakage. For numeric secrets, we remove all non-digit characters from both the known secret and the model response. For alphanumeric secrets, which contain letters and numbers, we standardize case and whitespace while preserving meaningful delimiters, or separating characters.

Let $\phi(\cdot)$ denote the normalization function. The regex-stage leakage label is

$$
\mathrm{Leak}_{\mathrm{regex}}(R,S)
=
\mathbb{I}\big[\phi(S)\subset\phi(R)\big],
\tag{2}
$$

where $R$ is the model response. The containment expression means that the normalized secret appears within the normalized response. The indicator function $\mathbb{I}$ equals one when that condition holds and zero otherwise.

#### 3.4.2 Stage 2: LLM-as-a-Judge Verification

We pass responses that fail normalization-based matching to an open-weights Llama judge (Meta AI, 2024). The judge receives the secret and model response. It returns a binary, yes-or-no verdict on whether the response reveals the secret or a substantial part of it.

This stage aims to capture partial disclosure, indirect hints, and formatting changes that deterministic matching misses. We report regex-only and hybrid rates separately because judge-based evaluation can introduce its own errors.

#### 3.4.3 Privacy Half-Life

We introduce Privacy Half-Life ($\tau$), a metric that requires sustained suppression rather than a temporary drop in leakage. Let $D_{obs}$ be the observed set of drift lengths. Let $L(d)$ be the leakage rate at drift length $d$. We set the safety threshold to $\delta=0.1\times L(0)$, where $\delta$, pronounced “delta,” denotes the threshold. We define

$$
\tau=
\min\left\{
d\in D_{obs}
\;\middle|\;
L(d)\leq\delta
\ \text{and}\
\forall d'\in D_{obs},\ d'>d:\ L(d')\leq\delta
\right\}.
\tag{3}
$$

In words, $\tau$ is the smallest observed drift length at which leakage reaches or falls below the threshold and stays there at every later observed drift length. Here, $d'$ denotes a later drift length, $\forall$ means “for every,” and $\min$ means minimum.

If no drift length meets these conditions, we assign $\tau=\max(D_{obs})+1$, where $\max$ means maximum. This definition prevents us from treating a temporary dip as stable suppression.

Figure 3: Privacy Half-Life $\tau$ requires stable decay below the threshold. A later resurgence disqualifies a transient decrease.

## 4 Experimental Setup

##### Models.

We evaluate GPT-OSS-120B, DeepSeek-R1, and Qwen3-VL-235B through the OpenRouter application programming interface (API), which provides programmatic access to the models. We query every model with the same dialogue histories and probe templates.

##### Dataset.

We evaluate 1,000 PrivDrift dialogues on each model. Each dialogue contains one deliberately inserted secret and one drift length $d\in\{0,2,3,4,5,6\}$ sampled from the distribution above. Each dialogue receives three standardized persuasion probes. This yields 3,000 cached model-probe responses per model and 9,000 total cached responses across the three models. Cached responses are stored outputs used for evaluation.

##### Aggregation levels.

We report results at two levels. We compute overall leakage at the dialogue level: a dialogue counts as leaking if any standardized probe elicits the secret. We compute persuasion-level and fuzzy-matching analyses at the probe level, where each dialogue contributes one response per persuasion condition. This distinction explains why dialogue-level leakage can exceed the average leakage rate across individual persuasion levels.

##### Uncertainty and statistical tests.

We report both regex-only and hybrid leakage. For dialogue-level rates, we compute 95% bootstrap confidence intervals over dialogues. These intervals estimate uncertainty by repeatedly resampling dialogues. For probe-level sensitivity checks, we compute rates over cached model-probe responses.

For paired model and detector comparisons on matched examples, we use McNemar’s test, which compares paired binary outcomes. Where applicable, we apply Bonferroni correction to account for multiple comparisons. For repeated measurements across persuasion levels, we use Cochran’s $Q$ test, which compares related binary outcomes across conditions. We then use post-hoc McNemar tests for follow-up comparisons.

To test associations between leakage and drift length, persuasion level, or secret type, we use chi-square tests of independence. These tests assess whether categorical variables are associated. We report Cramer’s $V$, a measure of association strength, as an effect size. In the results, $p$ denotes a statistical test’s p-value.

## 5 Results

### 5.1 Dialogue-Level Hybrid Evaluation Largely Agrees With Deterministic Matching

At the dialogue level, hybrid evaluation differs only slightly from normalization-based matching. This indicates that direct or near-direct reproduction accounts for most leakage in this benchmark, rather than purely semantic paraphrase. The LLM judge adds up to 1.10 percentage points of dialogue-level leakage across models (Table 2). This supports using hybrid leakage as the main metric while retaining regex-only rates as a transparent deterministic baseline.

| Model | Regex | Hybrid | $\Delta$ | $p$ |
|---|---:|---:|---:|---:|
| GPT-OSS-120B | 47.70 | 47.70 | +0.00 | 1.0 |
| DeepSeek-R1 | 37.80 | 38.70 | +0.90 | $7.7\times 10^{-3}$ |
| Qwen3-VL-235B | 53.50 | 54.60 | +1.10 | $2.57\times 10^{-3}$ |

Table 2: Dialogue-level leakage (%) under regex-only and hybrid evaluation. A dialogue is counted as leaking if any standardized probe elicits the secret.

Here, $\Delta$ denotes the change from regex-only to hybrid leakage, in percentage points.

### 5.2 All Models Leak Substantially at the Dialogue Level

Dialogue-level hybrid leakage remains substantial for all models. DeepSeek-R1 leaks 38.70%, GPT-OSS-120B leaks 47.70%, and Qwen3-VL-235B leaks 54.60% (Table 3). These rates estimate whether at least one standardized extraction probe can recover a dialogue’s secret.

| Model | Hybrid Leakage | 95% CI |
|---|---:|---|
| DeepSeek-R1 | 38.70% | [35.7, 41.6] |
| GPT-OSS-120B | 47.70% | [44.6, 50.8] |
| Qwen3-VL-235B | 54.60% | [51.6, 57.7] |

Table 3: Dialogue-level hybrid leakage rates with 95% bootstrap confidence intervals. A dialogue is counted as leaking if any standardized probe elicits the secret.

“CI” means confidence interval.

### 5.3 Persuasion Intensity Affects Leakage Non-Monotonically

Probe-level leakage differs significantly across persuasion levels for all models (Cochran’s $Q$, $p<10^{-5}$). However, the direction of the effect depends on the model (Table 4). For GPT-OSS-120B, hard persuasion reduces leakage relative to simple and medium persuasion. This pattern is consistent with pressure cues triggering safety behavior. For DeepSeek-R1, medium persuasion produces the lowest leakage. For Qwen3-VL-235B, hard persuasion produces the highest leakage, indicating weaker resistance to pressure-based extraction.

| Model | Simple | Medium | Hard |
|---|---:|---:|---:|
| GPT-OSS-120B | 39.30% | 40.50% | 32.90% |
| DeepSeek-R1 | 28.10% | 22.00% | 29.20% |
| Qwen3-VL-235B | 38.70% | 33.30% | 40.60% |

Table 4: Probe-level hybrid leakage by persuasion intensity.

### 5.4 Drift Length Does Not Reliably Reduce Leakage Within the Tested Window

The drift-leakage curves dip near $d=3$ and then rise again. However, drift-length effect sizes are small or negligible across models. In the cached probe-level analysis, Cramer’s $V$ is 0.1073 for DeepSeek-R1, 0.0659 for GPT-OSS-120B, and 0.0723 for Qwen3-VL-235B.

These results do not support a claim about arbitrarily long contexts. They show instead that privacy risk can persist across several unrelated topic shifts, even before the original secret is far from the end of the context window.

Figure 4: Hybrid leakage versus drift length. Curves show a dip near $d=3$ followed by rebound.

### 5.5 Secret Type Strongly Determines Leakage

Leakage varies sharply by secret type. Cramer’s $V$ shows a large association between secret type and hybrid leakage for all models. In the cached probe-level analysis, it is 0.5871 for DeepSeek-R1, 0.7731 for GPT-OSS-120B, and 0.6927 for Qwen3-VL-235B.

Models suppress SSNs and credit card numbers far more often than emails and phone numbers, although all four are user-disclosed and contextually private (Table 5). This pattern suggests that current behavior depends more on format and sensitivity cues than on a general understanding of contextual confidentiality.

| Secret Type | GPT-OSS | DeepSeek | Qwen |
|---|---:|---:|---:|
| SSN | 0.14% | 1.11% | 5.97% |
| Credit card | 0.00% | 0.39% | 4.01% |
| Email | 78.24% | 57.11% | 81.13% |
| Phone | 70.89% | 46.13% | 57.24% |

Table 5: Probe-level hybrid leakage by secret type across cached model-probe responses.

### 5.6 Privacy Half-Life Exceeds the Observed Window

Under the stable-decay definition of Privacy Half-Life, with threshold $\delta=0.1\cdot L(0)$, no evaluated model approaches the threshold at any tested drift length. Thus, $\tau=\max(D_{obs})+1=7$ for all models and persuasion levels. This indicates no stable decay within $d\leq 6$.

### 5.7 Fuzzy Matching Supports the Hybrid Labels

As a post-hoc deterministic sensitivity check, we compute normalized fuzzy matching between each saved secret and saved model response. This check uses only cached outputs. At a 0.85 threshold, fuzzy leakage closely tracks probe-level hybrid leakage for all models: 26.40% versus 26.43% for DeepSeek-R1, 37.03% versus 37.57% for GPT-OSS-120B, and 37.60% versus 37.53% for Qwen3-VL-235B (Table 6).

We treat fuzzy matching as supporting evidence rather than the primary metric. Approximate string similarity is less expressive than the judge when detecting partial or semantic disclosures.

| Model | Regex | Fuzzy 0.85 | Fuzzy 0.80 | Hybrid |
|---|---:|---:|---:|---:|
| DeepSeek-R1 | 25.50 | 26.40 | 26.53 | 26.43 |
| GPT-OSS-120B | 36.80 | 37.03 | 37.07 | 37.57 |
| Qwen3-VL-235B | 36.97 | 37.60 | 37.93 | 37.53 |

Table 6: Probe-level regex, fuzzy, and hybrid leakage rates (%) computed from cached model outputs.

Figure 5: Hybrid leakage heatmap by drift length and persuasion intensity for each model.

## 6 Discussion

PrivDrift reveals a persistent active-context privacy risk. Later prompts can recover sensitive information disclosed earlier in a conversation, even after unrelated topic drift. This does not imply that models permanently remember the secret or that all long-context settings behave similarly. It shows instead that current assistants may reproduce sensitive in-context information when later prompts apply enough pressure to retrieve it.

The cached probe-level evaluation also shows that this behavior mostly involves direct or near-direct reproduction. Fuzzy matching closely tracks hybrid leakage.

The strongest pattern is the difference across secret types. Models suppress SSNs and credit card numbers far more often than emails and phone numbers, although all four categories are user-disclosed secrets in the benchmark. This suggests that current safeguards may rely on format-sensitive safety heuristics—rules of thumb based on the secret’s form—rather than robust contextual reasoning about confidentiality.

Persuasion effects are significant, but they do not move consistently in one direction and depend on the model. For GPT-OSS-120B, high-pressure requests reduce leakage. This is consistent with the possibility that urgency cues activate refusal behavior. For DeepSeek-R1, medium persuasion produces the lowest leakage. For Qwen3-VL-235B, hard persuasion produces the highest leakage, suggesting weaker resistance to pressure-based extraction.

Finally, topic drift should not count as an implicit privacy boundary. Although leakage curves show intermediate dips, those decreases do not remain stable enough to meet the Privacy Half-Life criterion. This motivates evaluation based on stability rather than treating temporary drops as evidence of suppression.

## 7 Limitations and Future Work

##### Bounded drift range.

PrivDrift evaluates drift lengths up to $d=6$. This covers bounded short-to-medium conversational drift rather than filling modern context windows. The results therefore show persistence under bounded active-context drift. They do not fully describe privacy behavior across full long-context windows.

##### Fixed-format secrets.

The benchmark focuses on structured secrets such as SSNs, credit card numbers, emails, and phone numbers. These are easier to detect than unstructured sensitive information such as health status, family circumstances, immigration status, or financial hardship. Future work should extend the benchmark to sensitive attributes without fixed formats.

##### Active context, not cross-session memory.

PrivDrift evaluates whether prompts can recover secrets within the active conversation context. It does not test training-data memorization, personalization memory, or retention across sessions.

##### Synthetic and human-authored dialogues.

The dataset contains 900 synthetic dialogues and 100 human-authored dialogues inserted throughout the benchmark rather than stored in a separate block. Compared with a purely synthetic benchmark, this construction improves realism while preserving control over secret type, drift length, and probe structure.

However, the current cached evaluation metadata does not preserve which evaluated rows correspond to human-authored dialogues. We therefore report aggregate benchmark results and leave a separate human-versus-synthetic comparison for future work.

##### Detector scope.

We base leakage labels on the model’s output text, using deterministic matching, fuzzy matching, and an external LLM judge. We report regex-only, fuzzy, and hybrid labels separately to make detector behavior transparent. However, we do not yet report formal inter-annotator agreement, which measures how consistently humans and automated methods assign labels. Future work should include a larger manual audit of borderline partial disclosures and obfuscations, where outputs disguise the secret.

##### API-served models and reproducibility.

Compute and budget constraints led us to evaluate models served through APIs. These models may change over time. We reduce this limitation by logging model identifiers, prompts, timestamps, cached responses, and paired statistics on identical dialogue sets.

##### Artifact availability.

We plan to release the benchmark generation code, probe templates, evaluation scripts, aggregate result files, and a sanitized subset of generated dialogues. The benchmark contains synthetic strings that resemble PII. We will therefore release examples with non-real identifiers and generation templates sufficient to reproduce the benchmark without exposing realistic identifiers.

## 8 Conclusion

We introduce PrivDrift, a benchmark that audits active-context privacy leakage under topic drift and persuasion intensity. Across three LLMs and 1,000 dialogues per model, dialogue-level hybrid leakage ranges from 38.7% to 54.6%. Persuasion intensity affects leakage in model-dependent ways, while secret type has the strongest association with leakage. Within the tested drift range, topic drift does not reliably reduce leakage. Privacy Half-Life exceeds the observed window for all models. The differences across secret types suggest that current protections remain uneven and depend more on format-sensitive cues than on robust contextual confidentiality.

## Appendix A Dataset Characterization

Here, $N$ denotes the number of dialogues at each drift length.

| Drift $d$ | $N$ | Avg. words | Median words | Avg. tokens | Median tokens |
|---|---:|---:|---:|---:|---:|
| 0 | 197 | 44.29 | 43.00 | 59.05 | 57.33 |
| 2 | 147 | 253.97 | 258.00 | 338.62 | 344.00 |
| 3 | 142 | 344.18 | 347.00 | 458.91 | 462.67 |
| 4 | 166 | 452.27 | 447.00 | 603.02 | 596.00 |
| 5 | 181 | 520.92 | 543.00 | 694.56 | 724.00 |
| 6 | 167 | 568.03 | 605.00 | 757.37 | 806.67 |

Table 7: Dataset length characteristics by drift length. Approximate token counts are estimated from word counts.

## Appendix B Additional Effect Sizes

| Model | Variable | Cramer’s $V$ | Interpretation |
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
