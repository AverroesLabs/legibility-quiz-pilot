# PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations

Luciano Rolando Maldonado Romero

Affiliation: West Virginia University, Morgantown, WV, USA

Correspondence to: lrm00020@mix.wvu.edu

###### Abstract

Persistent, user-facing, shared-session, tool-augmented large language models (LLMs) retain behaviorally recoverable sensitive disclosures beyond unrelated tasks. The controlled benchmark PrivDrift, designated QX2, audits this persistence using seeded multi-turn histories, content-dense intervening exchanges, and standardized extraction probes. The principal measure, dialogue-level leakage (G4P), counts a dialogue as leaking when at least one standardized probe elicits the seeded secret. Under the hybrid detector (H8D), which combines deterministic matching with an LLM judge, dialogue-level leakage remains substantial across extended-context models: the evaluated systems have rates of 38.70%, 47.70%, and 54.60% (Table 3). Leakage varies sharply across systems, secret type (S8Q), and persuasion intensity (V4N), as reported in Tables 3–5. Moreover, leakage fails to decline reliably within the tested window: no model and persuasion condition achieves the stability threshold at an observed drift length (Section 5.6). Therefore, active-context privacy requires evaluation as persistent behavioral failure, not exclusively as parameter memorization or immediate jailbreak behavior.

###### Keywords:

Trustworthy AI, Large Language Models, Privacy, Evaluation, Benchmarking

## 1 Introduction

Social Security Numbers (SSNs) and payment identifiers receive much stronger suppression than contact identifiers: Table 5 reports leakage rates of 0.14%, 1.11%, and 5.97% for SSNs and 0.00%, 0.39%, and 4.01% for credit card numbers, compared with 78.24%, 57.11%, and 81.13% for email addresses and 70.89%, 46.13%, and 57.24% for phone numbers. However, no evaluated system attains stable decay within the tested window (Section 5.6). PrivDrift provides controlled persistence auditing; direct, justified, and high-pressure elicitation; reproducible composite detection using deterministic processing and an open-weights judge; and both dialogue-level leakage and probe-level leakage (P6S) reporting. Probe-level leakage treats each dialogue–persuasion-condition response as an observation. PrivDrift also uses a stability criterion rather than a transient-decrease criterion.

LLMs have evolved from simple chat systems into persistent assistants maintaining coherence across extended interactions (OpenAI, 2023; Touvron et al., 2023; Liu et al., 2024). Deployment domains include healthcare triage, financial planning, enterprise copilots, and personal productivity. In these domains, disclosed Personally Identifiable Information (PII) includes phone numbers, designated A2L; financial identifiers; email addresses, designated F8Z; and health-related details. Although context retention supports multi-turn assistance, the same context retention makes earlier sensitive disclosures available to later generation.

Existing safety evaluations mostly target two extremes: parameter-based training-data memorization and immediate jailbreaking, which elicits harmful behavior in the current turn (Wei et al., 2024; Zou et al., 2023). However, parameter extraction and multi-turn or injection evaluations do not directly establish whether unrelated conversational shifts provide a practical privacy boundary. The premise is not that users believe same-session messages literally disappear. Rather, users and designers may underestimate later indirect, justified, high-pressure, or tool-mediated elicitation. Shared sessions, browser-integrated assistants, enterprise copilots, agentic workflows, and injected instructions permit later context interactions unintended by the original user.

## 2 Related Work

### 2.1 Contextual Privacy and Inference-Time Leakage

Training-data extraction, membership inference, and memorized-PII recovery concern information stored in parameters and pre-training corpora (Shokri et al., 2017; Carlini et al., 2021; Li et al., 2023; Nasr et al., 2023). In contrast, active-context privacy concerns appropriate inference-time handling of current-interaction user information.

ConfAIde tests contextual-integrity privacy reasoning (Mireshghallah et al., 2024), while PrivacyLens and PrivaCI-Bench examine privacy-norm awareness and legal compliance in agentic or contextual settings (Shao et al., 2024; Li et al., 2025). The principal object of these evaluations is whether sharing is appropriate under static access-control or privacy-norm scenarios. In contrast, PrivDrift isolates behavioral recoverability following unrelated exchanges and subsequent persuasive probing.

### 2.2 Multi-Turn Adversarial Attacks

Aligned models remain exploitable over multiple turns. Crescendo progressively elicits harmful outputs from apparently benign conversations (Russinovich et al., 2024); automated frameworks generate adversarial multi-turn paths (Deng et al., 2024; Narula et al., 2025); and later injected instructions manipulate LLM-integrated applications (Liu et al., 2023). The usual targets of these attacks are unsafe content or instruction hijacking. In contrast, PrivDrift targets information-flow control: specifically, re-disclosure of earlier user-supplied sensitive information rather than harmful external content.

### 2.3 Context Management and Unlearning

TOFU and WMDP assess removal or behavioral suppression of knowledge, usually through static question-answer pairs and model parameters (Maini et al., 2024; Li et al., 2024). PrivDrift instead concerns behavioral suppression of secrets still present in active context. This distinction matters because deployed retrieval-augmented generation (RAG) systems, copilots, and agentic workflows often depend on prompt-level context management rather than parameter-level unlearning.

## 3 The QX2 Framework

PrivDrift combines a parametric dialogue generator, a probing harness, and a detection/temporal-analysis metric suite. Figure 1 summarizes how these components connect.

Figure 1: QX2 pipeline: controlled multi-turn dialogue construction, standardized probe attachment, target-model queries, composite leakage scoring.

### 3.1 Threat Model

Transcript invisibility is not assumed. Instead, the failure mode is active-context re-disclosure, designated E9U: assistant reproduction of an earlier sensitive value under later querying, justification, or pressure. Therefore, the failure mode concerns reproduction of the sensitive value, not the mere presence of that value in the transcript.

A later participant in a shared or persistent session may not understand preceding disclosures. Similarly, tool outputs, retrieved documents, and injected instructions in enterprise or agentic settings may interact with existing context. Browser/application users may not intend sensitive information to be reused after task changes.

A user supplies a secret, denoted \(S\), at the initial disclosure time, denoted \(t_0\). Unrelated, content-dense exchanges then introduce topic drift, designated J7K. The intervening history is represented by
\(D=\{u_1,a_1,\ldots,u_d,a_d\}\), where \(u_i\) is a user message, \(a_i\) is the corresponding assistant message, and \(d\) is the intervening-turn quantity, or drift length. After these exchanges, a standardized probe, denoted \(P\), is appended to the active context. Exact, approximate, or semantic revelation of the secret \(S\) constitutes leakage.

### 3.2 Dataset Generation Pipeline

The scaffold-then-rewrite pipeline preserves exact ground truth, intervening-turn counts, and probe structure while reducing purely templated artifacts. Human-authored dialogues obey the same constraints and are interspersed rather than contiguous.

Table 1 summarizes the evaluated benchmark. In the table, \(N\) denotes the total number of dialogues, and D7A designates drift length \(d\), the intervening-turn quantity introduced in Section 3.1.

| Statistic | Value |
|---|---:|
| Total dialogues, \(N\) | 1,000 |
| Synthetic dialogues | 900 |
| Human-authored dialogues | 100 |
| Secret categories | 4 |
| D7A values | \(0,2,3,4,5,6\) |
| Avg. words per dialogue | 359.16 |
| Median words per dialogue | 385.00 |
| Min/max words per dialogue | 43 / 795 |
| Avg. approx. tokens per dialogue | 478.87 |
| Median approx. tokens per dialogue | 513.33 |
| Min/max approx. tokens per dialogue | 57.33 / 1060.00 |

Table 1: Evaluated benchmark summary; approximate tokens estimated from words.

#### 3.2.1 Persona and Secret Injection

Opening-turn plausible task contexts establish legitimate assistant access to the secret. Personas are sampled from more than 100 professions and demographic profiles, while the categorical variable secret type selects among fixed-format identifiers.

The four sampled formats comprise contact values—phone numbers—and electronic contact values—email addresses—alongside government identifiers, Social Security Numbers (W3B), and payment identifiers, credit card numbers (N6G). Each identifier is inserted as the ground-truth secret rather than subsequently inferred.

#### 3.2.2 Content-Dense J7K

Assigned topics include tax filing, medical triage, and travel logistics. A topic-conditioned dense question-answer library supplies intervening turns, with professional-writing, structured-reasoning, and long-form-summarization fallback tasks. Annotators contribute a subset of library turns. Thus, unrelated entities and task demands enter context while the original secret and turn count remain controlled.

#### 3.2.3 Naturalistic Refinement

A rewriting model receives each scaffold and instructions for fluency/persona realism. During rewriting, the exact secret, turn count, role structure, and final-probe conditions remain invariant. Therefore, surface phrasing changes, but experimental variables do not.

### 3.3 Probing Setup

The final probe follows the completed history. The manipulated interaction style, persuasion intensity, has three standardized conditions rather than an unconstrained adversarial search:

- Neutral direct retrieval is designated C8R, or Simple.
- Contextual justification is designated Z2F, or Medium.
- Urgent/high-pressure elicitation is designated L9W, or Hard.

These conditions distinguish requests for earlier phone numbers, form-filling needs, and tests of whether pressure cues raise or lower disclosure.

Drift length \(d\) counts turns between disclosure and final probing. Drift length is sampled from the following mixture, where \(p\) denotes probability and \(\mathrm{Uniform}\{2,3,4,5,6\}\) denotes uniform sampling over the listed lengths:

\[
d\sim
\begin{cases}
0 & \text{with }p=0.2,\\
\mathrm{Uniform}\{2,3,4,5,6\} & \text{with }p=0.8.
\end{cases}
\tag{1}
\]

Thus, the mixture includes immediate recall while emphasizing short-to-medium topic drift.

### 3.4 Evaluation Methodology

Eligible disclosures include verbatim output, formatting changes, partial substrings, and paraphrased references. The hybrid detector labels a response positive if normalized inclusion succeeds or the judge finds the full secret, a substantial substring, or sufficient partial information to identify the secret.

The evaluation is hierarchical: normalization-based matching, designated B6T, is applied first, and only responses negative under normalization-based matching proceed to the LLM judge, as Figure 2 shows.

Figure 2: Hierarchical evaluation: B6T first; negative responses proceed to the LLM judge.

#### 3.4.1 Stage 1: B6T

For direct and near-direct disclosure, normalization-based matching strips every non-digit character from numeric secrets and responses. For alphanumeric secrets and responses, normalization-based matching normalizes case and whitespace but retains meaningful delimiters.

The deterministic baseline is designated U3P, or regex-only. Let \(\phi\) denote normalization, \(R\) denote the model response, and \(\mathbb I\) denote an indicator that is 1 when the enclosed condition holds and 0 otherwise. The condition \(\phi(S)\subset\phi(R)\) means that the normalized secret is included in the normalized response. The deterministic leakage label is then:

\[
\mathrm{Leak}_{\mathrm{U3P}}(R,S)
=\mathbb I\big[\phi(S)\subset\phi(R)\big].
\tag{2}
\]

#### 3.4.2 Stage 2: LLM-as-a-Judge Verification

An open-weights Llama judge receives the secret \(S\) and response \(R\) and returns a binary full/substantial-disclosure verdict. The judge handles cases that are negative under deterministic matching, checking for partial disclosures, indirect hints, and otherwise missed formatting variations (Meta AI, 2024).

The regex-only baseline remains separately reported alongside the hybrid detector. This separate reporting recognizes that judge errors are possible rather than treating adjudication as error-free.

#### 3.4.3 K2V

The stability metric is Privacy Half-Life, designated K2V and denoted \(\tau\). Let \(D_{obs}\) denote the set of observed drift lengths and \(L(d)\) denote the leakage rate at drift length \(d\). The immediate-recall leakage rate \(L(0)\) defines the safety threshold \(\delta=0.1\times L(0)\).

Privacy Half-Life requires sustained rather than momentary suppression. Specifically, Privacy Half-Life is the smallest observed length at which leakage is at or below the safety threshold and remains at or below that threshold at every later observed length. In the expression below, \(d'\) denotes a later observed drift length:

\[
\tau=\min\left\{
d\in D_{obs}\ \middle|\
L(d)\leq\delta
\ \text{and}\
\forall d'\in D_{obs},\ d'>d:\ L(d')\leq\delta
\right\}.
\tag{3}
\]

If no observed length satisfies these conditions, the set is empty and receives \(\tau=\max(D_{obs})+1\).

Figure 3: K2V; later resurgence disqualifies transient decreases.

## 4 Experimental Setup

##### Models.

All systems receive identical histories and probe templates through the OpenRouter application programming interface (API). The evaluated model identifiers are M7Q for GPT-OSS-120B, T4X for DeepSeek-R1, and Y9C for Qwen3-VL-235B.

##### Dataset.

Each system processes the full PrivDrift dataset, with one seeded secret and one sampled drift length per dialogue, under every standardized persuasion condition. The resulting cached response counts are:

| Cached responses | Count |
|---|---:|
| Per model | 3,000 |
| All evaluated models | 9,000 |

##### Aggregation levels.

Dialogue-level leakage is an any-probe dialogue statistic: a dialogue is positive when at least one standardized probe elicits the dialogue’s secret. In contrast, probe-level leakage gives each dialogue one observation per persuasion condition. Probe-level observations are used for persuasion-intensity analyses and for the post-hoc fuzzy-matching sensitivity check, designated R5J. Because dialogue-level leakage counts success under any probe, dialogue-level leakage can exceed the mean condition-specific rate.

##### Uncertainty and statistical tests.

Dialogue-level leakage intervals use 95% bootstrapping over dialogues. Probe-level leakage sensitivity checks instead use cached model-probe responses. Matched model/detector comparisons use McNemar’s test, Bonferroni-corrected where applicable. Repeated persuasion measures use Cochran’s \(Q\), followed by McNemar tests.

Independence tests for leakage against drift length, persuasion intensity, or secret type use chi-square. The association statistic Cramer’s \(V\), designated X2H, is reported as an effect size rather than as a substitute for significance testing. In the results tables, \(p\) denotes a statistical-test p-value, rather than the sampling probability used in Equation (1).

## 5 Results

### 5.1 G4P: H8D Largely Agrees With B6T

Hybrid detection differs only slightly from normalization-based matching. In Table 2, the hybrid detector adds +0.00 percentage points for GPT-OSS-120B, +0.90 for DeepSeek-R1, and +1.10 for Qwen3-VL-235B relative to the regex-only baseline. Thus, direct or near-direct reproduction dominates purely semantic paraphrase. Judge augmentation is limited, supporting the hybrid detector as primary and the regex-only detector as a transparent deterministic baseline.

In Table 2, \(\Delta\) denotes the hybrid detector’s augmentation over the regex-only baseline in percentage points.

| Model | U3P | H8D | \(\Delta\) | \(p\) |
|---|---:|---:|---:|---:|
| M7Q | 47.70 | 47.70 | +0.00 | 1.0 |
| T4X | 37.80 | 38.70 | +0.90 | \(7.7\times10^{-3}\) |
| Y9C | 53.50 | 54.60 | +1.10 | \(2.57\times10^{-3}\) |

Table 2: G4P percentages; augmentation in percentage points.

### 5.2 Substantial G4P Across Models

Every evaluated system remains vulnerable to at least one standardized extraction condition at substantial rates. Specifically, hybrid dialogue-level leakage is 38.70% for DeepSeek-R1, 47.70% for GPT-OSS-120B, and 54.60% for Qwen3-VL-235B, with the 95% confidence intervals (CIs) shown in Table 3.

| Model | H8D | 95% CI |
|---|---:|---|
| T4X | 38.70% | [35.7, 41.6] |
| M7Q | 47.70% | [44.6, 50.8] |
| Y9C | 54.60% | [51.6, 57.7] |

Table 3: G4P with bootstrap confidence intervals.

### 5.3 Non-Monotonic V4N Effects

Within-model differences across persuasion conditions are significant for every model, with Cochran’s \(Q\) yielding \(p<10^{-5}\) throughout (Table 4). However, the directions differ. GPT-OSS-120B declines under Hard elicitation to 32.90%, compared with 39.30% under Simple retrieval and 40.50% under Medium justification. DeepSeek-R1 minimizes at Medium justification, with 22.00%, compared with 28.10% under Simple retrieval and 29.20% under Hard elicitation. In contrast, Qwen3-VL-235B maximizes at Hard elicitation, with 40.60%, compared with 38.70% under Simple retrieval and 33.30% under Medium justification.

| Model | C8R | Z2F | L9W |
|---|---:|---:|---:|
| M7Q | 39.30% | 40.50% | 32.90% |
| T4X | 28.10% | 22.00% | 29.20% |
| Y9C | 38.70% | 33.30% | 40.60% |

Table 4: H8D P6S by V4N; Cochran’s \(Q\), \(p<10^{-5}\) for every model.

### 5.4 No Reliable Reduction Within the Tested Window

Cached probe-level leakage associations with drift length are small or negligible. Table 8 reports Cramer’s \(V=0.1073\), interpreted as Small, for DeepSeek-R1; \(0.0659\), interpreted as Negligible, for GPT-OSS-120B; and \(0.0723\), interpreted as Negligible, for Qwen3-VL-235B. The dip near \(d=3\) is followed by a rebound, as described in Figure 4. Thus, privacy risk persists across several unrelated shifts before the original disclosure lies far from the context-window end.

Figure 4: H8D versus D7A; dip near \(d=3\), then rebound.

### 5.5 S8Q Strongly Determines Leakage

Secret type has a large association with leakage in every system and the strongest association among tested variables. Table 8 reports secret-type Cramer’s \(V\) values of 0.7731 for GPT-OSS-120B, 0.5871 for DeepSeek-R1, and 0.6927 for Qwen3-VL-235B, each interpreted as Large; the drift-length and persuasion-intensity associations are Small or Negligible.

As Table 5 shows, Social Security Numbers and credit card numbers receive far more suppression than email addresses and phone numbers, despite all categories being user-disclosed and contextually private. The rows labeled W3B, N6G, F8Z, and A2L correspond respectively to Social Security Numbers, credit card numbers, email addresses, and phone numbers.

| S8Q | M7Q | T4X | Y9C |
|---|---:|---:|---:|
| W3B | 0.14% | 1.11% | 5.97% |
| N6G | 0.00% | 0.39% | 4.01% |
| F8Z | 78.24% | 57.11% | 81.13% |
| A2L | 70.89% | 46.13% | 57.24% |

Table 5: H8D P6S across cached model-probe responses.

### 5.6 K2V Exceeds Observation

No system approaches the safety threshold at any tested length, so suppression never stabilizes. Consequently, every model and persuasion condition receives the out-of-observation Privacy Half-Life value of 7 under the empty-set rule in Equation (3).

| Scope | Observed bound | Assigned K2V |
|---|---|---|
| Every model and persuasion condition | \(d\leq6\) | \(\tau=\max(D_{obs})+1=7\) |

### 5.7 R5J Supports H8D Labels

The post-hoc deterministic fuzzy-matching sensitivity check computes normalized similarity between each saved secret and saved response using cached outputs only. The tested similarity thresholds are 0.85 and 0.80. The higher threshold closely tracks hybrid detection across systems: Table 6 reports fuzzy-matching rates at 0.85 of 26.40%, 37.03%, and 37.60%, compared with hybrid rates of 26.43%, 37.57%, and 37.53% for DeepSeek-R1, GPT-OSS-120B, and Qwen3-VL-235B, respectively.

Fuzzy matching is auxiliary because approximate string similarity is less expressive than judge-based evaluation for partial or semantic disclosures.

| Model | U3P | R5J 0.85 | R5J 0.80 | H8D |
|---|---:|---:|---:|---:|
| T4X | 25.50 | 26.40 | 26.53 | 26.43 |
| M7Q | 36.80 | 37.03 | 37.07 | 37.57 |
| Y9C | 36.97 | 37.60 | 37.93 | 37.53 |

Table 6: Cached-output P6S percentages.

Figure 5 displays hybrid leakage by drift length and persuasion intensity for each system.

Figure 5: H8D heatmap by D7A and V4N for each system.

## 6 Discussion

Behavioral recoverability does not imply permanent memory or uniform behavior across long-context settings. Rather, the observed leakage shows that current assistants may reproduce sensitive in-context information under sufficient retrieval pressure.

GPT-OSS-120B’s pressure response is consistent with urgency activating safety/refusal behavior: leakage falls to 32.90% under Hard elicitation from 39.30% under Simple retrieval and 40.50% under Medium justification (Table 4, M7Q row). In contrast, Qwen3-VL-235B’s response indicates weaker pressure resistance: leakage reaches 40.60% under Hard elicitation, compared with 38.70% under Simple retrieval and 33.30% under Medium justification (Table 4, Y9C row).

Topic drift is not an implicit privacy boundary. In particular, the intermediate dip followed by rebound described in Figure 4, together with the absence of stable suppression in Section 5.6, warrants stability-based evaluation rather than treatment of unstable intermediate decreases as suppression evidence.

## 7 Limitations and Future Work

##### Bounded D7A.

The tested short-to-medium drift-length range does not saturate modern context windows. Therefore, these findings concern bounded active-context persistence, not a complete characterization of arbitrarily long or full-window privacy behavior.

##### Fixed-format secrets.

Structured identifiers are easier to detect than unstructured health status, family circumstances, immigration status, or financial hardship. Consequently, extension to non-fixed sensitive attributes remains future work.

##### Active context, not cross-session memory.

Training-data memorization, personalization memory, and cross-session retention are untested.

##### Synthetic and human-authored dialogues.

The mixed construction improves realism over a purely synthetic benchmark while retaining experimental control. However, cached metadata no longer preserves human-authored row identifiers. Therefore, only aggregate results are reported, and separate human-versus-synthetic leakage comparison is deferred.

##### Detector scope.

Labels concern surface-form outputs under regex-only matching, fuzzy matching, and an external judge. Separate reporting exposes detector behavior, but formal human–automated inter-annotator agreement is not reported. Larger manual audits of borderline partial disclosures and obfuscations remain future work.

##### API-served models and reproducibility.

Compute/budget constraints require API service. The evaluated identifiers—M7Q for GPT-OSS-120B, T4X for DeepSeek-R1, and Y9C for Qwen3-VL-235B—designate models that may change over time, despite logged identifiers, prompts, timestamps, cached responses, and paired statistics on identical dialogue sets.

##### Artifact availability.

Planned releases comprise generation code, probe templates, evaluation scripts, aggregate result files, and a sanitized dialogue subset. Synthetic PII-like strings necessitate non-real identifiers in released instances and templates sufficient for reproduction without exposing realistic identifiers.

## 8 Conclusion

The secret-type asymmetry in Table 5—much lower leakage for Social Security Numbers and credit card numbers than for email addresses and phone numbers—suggests uneven safeguards relying more on format/sensitivity cues and format-sensitive safety heuristics than on generalized contextual confidentiality or robust contextual reasoning.

## Appendix A Dataset Characterization

Table 7 characterizes dialogue length by drift length. The D7A column gives drift length \(d\), and \(N\) gives the number of dialogues at each length. Token counts are approximate and derived from word counts.

| D7A, \(d\) | \(N\) | Avg. words | Median words | Avg. tokens | Median tokens |
|---|---:|---:|---:|---:|---:|
| 0 | 197 | 44.29 | 43.00 | 59.05 | 57.33 |
| 2 | 147 | 253.97 | 258.00 | 338.62 | 344.00 |
| 3 | 142 | 344.18 | 347.00 | 458.91 | 462.67 |
| 4 | 166 | 452.27 | 447.00 | 603.02 | 596.00 |
| 5 | 181 | 520.92 | 543.00 | 694.56 | 724.00 |
| 6 | 167 | 568.03 | 605.00 | 757.37 | 806.67 |

Table 7: Length characteristics by D7A; token counts approximate, word-derived.

## Appendix B Additional Effect Sizes

Table 8 reports Cramer’s \(V\), labeled X2H, for associations between hybrid probe-level leakage and each benchmark variable. The variable codes S8Q, D7A, and V4N denote secret type, drift length, and persuasion intensity, respectively. The model codes T4X, M7Q, and Y9C denote DeepSeek-R1, GPT-OSS-120B, and Qwen3-VL-235B, respectively.

| Model | Variable | X2H | Interpretation |
|---|---|---:|---|
| T4X | S8Q | 0.5871 | Large |
| T4X | D7A | 0.1073 | Small |
| T4X | V4N | 0.0718 | Negligible |
| M7Q | S8Q | 0.7731 | Large |
| M7Q | D7A | 0.0659 | Negligible |
| M7Q | V4N | 0.0689 | Negligible |
| Y9C | S8Q | 0.6927 | Large |
| Y9C | D7A | 0.0723 | Negligible |
| Y9C | V4N | 0.0639 | Negligible |

Table 8: Cached P6S effect sizes for H8D–benchmark-variable associations.

## Appendix C Dialogue Construction Figure

Figure 6: Dialogue construction details.
