# PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations

Luciano Rolando Maldonado Romero

Affiliation: West Virginia University, Morgantown, WV, USA

Correspondence to: lrm00020@mix.wvu.edu

###### Abstract

Persistent large language models (LLMs) retain sensitive user disclosures in behaviorally recoverable form: later prompts can elicit them after unrelated tasks. This concerns user-facing assistants, including shared-session assistants and assistants augmented with tools. The controlled benchmark QX2, named PrivDrift, audits this persistence. It uses multi-turn histories containing known secrets, content-dense intervening exchanges, and standardized extraction probes.

Across extended-context models, G4P—dialogue-level leakage, meaning disclosure under at least one probe—remains substantial under H8D, the hybrid leakage detector (Table 3). Leakage varies sharply across systems, S8Q (secret type), and V4N (persuasion intensity; Tables 3–5). It fails to decline reliably within the tested window (Sections 5.4 and 5.6). Active-context privacy concerns information still available in the current interaction. It requires evaluation as a persistent behavioral failure, not exclusively as memorization in model parameters or immediate jailbreak behavior.

###### Keywords:

Trustworthy AI (artificial intelligence), Large Language Models, Privacy, Evaluation, Benchmarking

## 1 Introduction

Social Security Numbers (SSNs) and payment identifiers receive much stronger suppression than contact identifiers (Table 5). No evaluated system attains stable decay in leakage within the tested window (Section 5.6). QX2 provides controlled persistence auditing with direct, justified, and high-pressure requests for disclosure. It provides reproducible composite detection using deterministic text processing and an open-weights model judge—a model whose parameter weights are available. It reports both G4P and P6S, or probe-level leakage, which counts responses separately for each probe condition. It evaluates sustained suppression rather than a temporary decrease.

LLMs have evolved from simple chat systems into persistent assistants that maintain coherence across extended interactions (OpenAI, 2023; Touvron et al., 2023; Liu et al., 2024). Deployment domains include healthcare triage, financial planning, enterprise copilots, and personal productivity. Disclosed Personally Identifiable Information (PII), meaning information that identifies a person, includes A2L (phone numbers), financial identifiers, F8Z (email addresses), and health-related details. Retaining conversational context supports multi-turn assistance. It also leaves earlier sensitive disclosures available to later generation.

Existing safety evaluations mostly target two extremes. The first is training-data memorization in model parameters, the learned values that determine model behavior. The second is immediate jailbreaking: eliciting harmful behavior in the current turn (Wei et al., 2024; Zou et al., 2023). Parameter-extraction tests and multi-turn or instruction-injection evaluations do not directly establish whether unrelated conversational shifts provide a practical privacy boundary.

The premise is not that users believe messages literally disappear within the same session. Rather, users and designers may underestimate later indirect, justified, high-pressure, or tool-mediated elicitation. Shared sessions, browser-integrated assistants, enterprise copilots, agentic workflows, and injected instructions allow later interactions with context that the original user did not intend. Agentic workflows are workflows in which an assistant carries out tasks through multiple actions, potentially using tools.

## 2 Related Work

### 2.1 Contextual Privacy and Inference-Time Leakage

Training-data extraction attempts to recover training information. Membership inference tests whether information belonged to the training data. Memorized-PII recovery attempts to recover identifying information retained through training. These concern information stored in model parameters and pre-training corpora (Shokri et al., 2017; Carlini et al., 2021; Li et al., 2023; Nasr et al., 2023). Active-context privacy instead concerns appropriate handling of current-interaction user information during inference, when the model generates responses.

ConfAIde tests contextual-integrity privacy reasoning: whether information sharing fits the context (Mireshghallah et al., 2024). PrivacyLens and PrivaCI-Bench examine awareness of privacy norms and legal compliance in agentic or contextual settings (Shao et al., 2024; Li et al., 2025). Their principal object is whether sharing is appropriate under static access-control or privacy-norm scenarios. QX2 instead isolates behavioral recoverability after unrelated exchanges and subsequent persuasive probing.

### 2.2 Multi-Turn Adversarial Attacks

Aligned models, trained to follow intended behavioral constraints, remain exploitable over multiple turns. Crescendo progressively elicits harmful outputs from apparently benign conversations (Russinovich et al., 2024). Automated frameworks generate adversarial multi-turn paths (Deng et al., 2024; Narula et al., 2025). Later injected instructions can also manipulate applications that integrate LLMs (Liu et al., 2023).

These approaches usually target unsafe content or instruction hijacking, in which unintended instructions redirect model behavior. QX2 targets information-flow control: whether earlier user-supplied sensitive information is re-disclosed. Its target is not harmful external content.

### 2.3 Context Management and Unlearning

TOFU and WMDP are benchmarks that assess removal of knowledge or suppression of its expression, usually through static question-answer pairs and changes to model parameters (Maini et al., 2024; Li et al., 2024). QX2 instead concerns behavioral suppression of secrets that remain in active context. Deployed retrieval-augmented generation (RAG) systems, which supply retrieved material to a model, as well as copilots and agentic workflows, often depend on prompt-level context management rather than parameter-level unlearning.

## 3 The QX2 Framework

Figure 1: QX2 pipeline: controlled multi-turn dialogue construction, standardized probe attachment, target-model queries, composite leakage scoring.

The synthetic framework combines a configurable dialogue generator, a system for applying probes, and metrics for detecting leakage and analyzing it across intervening dialogue lengths.

### 3.1 Threat Model

The threat model does not assume that the transcript is invisible. E9U, or active-context re-disclosure, is the failure being tested: the assistant reproduces an earlier sensitive value under later querying, justification, or pressure. The value’s mere presence in the transcript is not itself this failure.

A later participant in a shared or persistent session may not understand preceding disclosures. Tool outputs, retrieved documents, and injected instructions in enterprise or agentic settings may interact with existing context. Browser or application users may not intend sensitive information to be reused after task changes.

A user supplies a secret, denoted \(S\), at an initial time \(t_0\). Unrelated, content-dense exchanges then create J7K, or topic drift. These exchanges are represented by

\[
D=\{u_1,a_1,\ldots,u_d,a_d\},
\]

where \(D\) is the intervening dialogue, \(u_i\) is user message \(i\), and \(a_i\) is assistant message \(i\). D7A, or drift length, is denoted \(d\) and counts intervening turns. A standardized probe \(P\), meaning a request intended to elicit the secret, is then appended to the active context. Exact, approximate, or semantic revelation of \(S\) constitutes leakage.

### 3.2 Dataset Generation Pipeline

The pipeline first creates a controlled dialogue scaffold and then rewrites its wording. This preserves the exact ground-truth secret, the intervening-turn count, and the probe structure while reducing artifacts caused by repeated templates. Human-authored dialogues follow the same constraints. They are interspersed rather than contiguous.

In Table 1, \(N\) is the number of dialogues, “Avg.” means average, and “Min/max” gives the minimum and maximum. Tokens are the text units processed by a model; the reported token counts are approximate estimates from word counts.

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

Plausible task contexts in the opening turn establish legitimate assistant access to \(S\). Personas are sampled from more than 100 professions and demographic profiles. S8Q, the categorical secret-type variable, selects among fixed-format identifiers.

The four sampled formats are contact values, A2L (phone numbers); electronic contact values, F8Z (email addresses); government identifiers, W3B (Social Security Numbers); and payment identifiers, N6G (credit card numbers). Each value is inserted as the ground-truth secret rather than inferred later.

#### 3.2.2 Content-Dense J7K

Assigned topics include tax filing, medical triage, and travel logistics. A topic-conditioned library of dense question-answer exchanges supplies intervening turns. Fallback tasks involve professional writing, structured reasoning, and long-form summarization. Annotators contribute a subset of library turns. Unrelated entities and task demands enter the context while the original secret and turn count remain controlled.

#### 3.2.3 Naturalistic Refinement

A rewriting model receives each scaffold and instructions to improve fluency and persona realism. The exact secret, turn count, message-role structure, and final-probe conditions remain unchanged. Surface phrasing changes; experimental variables do not.

### 3.3 Probing Setup

The final probe follows the completed history. V4N, or persuasion intensity, is the manipulated interaction style. It has three standardized conditions rather than an unconstrained search for an adversarial prompt:

- C8R, or Simple: neutral, direct retrieval.
- Z2F, or Medium: contextual justification.
- L9W, or Hard: urgent or high-pressure elicitation.

The conditions distinguish requests for an earlier A2L value, form-filling needs, and tests of whether pressure cues raise or lower disclosure.

The intervening-turn quantity D7A, denoted \(d\), is sampled from the following mixture. Here \(p\) means sampling probability, and \(\mathrm{Uniform}\) means that the listed values are equally likely within that part of the mixture.

\[
d\sim
\begin{cases}
0 & \text{with }p=0.2,\\
\mathrm{Uniform}\{2,3,4,5,6\} & \text{with }p=0.8.
\end{cases}
\tag{1}
\]

Thus, immediate recall is included, while short-to-medium topic drift is emphasized.

### 3.4 Evaluation Methodology

Eligible disclosures include verbatim output, formatting changes, partial substrings, and paraphrased references. H8D, the hybrid composite detector, labels a response positive if either normalized inclusion succeeds or the judge finds the full secret, a substantial substring, or enough partial information to identify it. “Positive” means that leakage is detected.

The first stage, B6T, is normalization-based matching: deterministic text processing followed by an inclusion check. Responses not detected at this stage proceed to an LLM judge.

Figure 2: Hierarchical evaluation: B6T first; negative responses proceed to the LLM judge.

#### 3.4.1 Stage 1: B6T

For direct and near-direct disclosure, B6T strips every non-digit character from numeric secrets and responses. For alphanumeric text, it normalizes case and whitespace but retains meaningful delimiters.

Let \(\phi\) denote this normalization, and let \(R\) denote the model response. U3P is the separately reported regex-only baseline; “regex” means regular-expression, or pattern-based, matching. Its leakage indicator is

\[
\mathrm{Leak}_{\mathrm{U3P}}(R,S)
=\mathbb I\big[\phi(S)\subset\phi(R)\big].
\tag{2}
\]

Here \(\mathbb I\) is an indicator: it is 1 when the condition holds and 0 otherwise. The inclusion condition means that the normalized secret occurs within the normalized response.

#### 3.4.2 Stage 2: LLM-as-a-Judge Verification

An open-weights Llama judge receives \(S\) and \(R\). It returns a binary verdict on whether the response contains full or substantial disclosure. It handles cases that were negative under deterministic matching, including partial disclosures, indirect hints, and otherwise missed formatting variations (Meta AI, 2024).

U3P remains separately reported alongside H8D. Judge errors are possible; adjudication is not treated as error-free.

#### 3.4.3 K2V

K2V, or Privacy Half-Life, is a stability metric denoted \(\tau\). Let \(D_{obs}\) be the set of observed drift lengths, and let \(L(d)\) be the leakage rate at length \(d\). The safety threshold is \(\delta=0.1\times L(0)\), where \(L(0)\) is the rate without intervening turns.

The metric requires sustained, not momentary, suppression:

\[
\tau=\min\left\{
d\in D_{obs}\ \middle|\
L(d)\leq\delta
\ \text{and}\
\forall d'\in D_{obs},\ d'>d:\ L(d')\leq\delta
\right\}.
\tag{3}
\]

In words, \(\tau\) is the smallest observed length at which leakage is at or below the threshold and remains so at every later observed length \(d'\). The symbol \(\forall\) means “for every.” If no length satisfies these conditions, the set is empty and receives \(\tau=\max(D_{obs})+1\), one above the largest observed length.

Figure 3: K2V; later resurgence disqualifies transient decreases.

## 4 Experimental Setup

##### Models.

All systems receive identical histories and probe templates through the OpenRouter application programming interface (API), the interface used to query the models. The evaluated identifiers are M7Q (GPT-OSS-120B), T4X (DeepSeek-R1), and Y9C (Qwen3-VL-235B).

##### Dataset.

Each system processes the full QX2 dataset. Each dialogue contains one seeded secret and one sampled \(d\), and is evaluated under every standardized persuasion condition.

| Cached responses | Count |
|---|---:|
| Per model | 3,000 |
| All evaluated models | 9,000 |

##### Aggregation levels.

G4P, the any-probe dialogue statistic, counts a dialogue as positive when at least one standardized probe elicits its secret. P6S, the probe-level response statistic, gives each dialogue one observation per persuasion condition. P6S is used for V4N analyses and R5J analyses. R5J means fuzzy matching: approximate string-similarity matching used as a sensitivity check.

Because G4P counts disclosure under any probe, it can exceed the mean condition-specific rate.

##### Uncertainty and statistical tests.

G4P intervals use 95% bootstrapping over dialogues: repeatedly resampling dialogues to estimate uncertainty. P6S sensitivity checks use cached model-probe responses.

Matched model and detector comparisons use McNemar’s test, which compares paired binary outcomes. Bonferroni correction, an adjustment for multiple comparisons, is applied where applicable. Repeated persuasion measures use Cochran’s \(Q\), a test of binary outcomes across the repeated conditions, followed by McNemar tests.

Chi-square independence tests assess whether leakage is associated with D7A, V4N, or S8Q. X2H, or Cramer’s \(V\), summarizes association strength as an effect size. It does not replace significance testing. In the results, \(p\) denotes a statistical test’s \(p\)-value, not the sampling probability used in Equation (1).

## 5 Results

### 5.1 G4P: H8D Largely Agrees With B6T

H8D differs only slightly from B6T: the increases are +0.00, +0.90, and +1.10 percentage points for M7Q, T4X, and Y9C, respectively (Table 2). Direct or near-direct reproduction dominates purely semantic paraphrase. Judge augmentation is limited, supporting H8D as the primary detector and U3P as a transparent deterministic baseline.

In Table 2, \(\Delta\) is H8D minus U3P, expressed in percentage points.

| Model | U3P | H8D | \(\Delta\) | \(p\) |
|---|---:|---:|---:|---:|
| M7Q | 47.70 | 47.70 | +0.00 | 1.0 |
| T4X | 37.80 | 38.70 | +0.90 | \(7.7\times10^{-3}\) |
| Y9C | 53.50 | 54.60 | +1.10 | \(2.57\times10^{-3}\) |

Table 2: G4P percentages; augmentation in percentage points.

### 5.2 Substantial G4P Across Models

Every evaluated system remains vulnerable to at least one standardized extraction condition at substantial rates: 38.70% for T4X, 47.70% for M7Q, and 54.60% for Y9C (Table 3). CI means confidence interval; these intervals use the dialogue-level bootstrap described in Section 4.

| Model | H8D | 95% CI |
|---|---:|---|
| T4X | 38.70% | [35.7, 41.6] |
| M7Q | 47.70% | [44.6, 50.8] |
| Y9C | 54.60% | [51.6, 57.7] |

Table 3: G4P with bootstrap confidence intervals.

### 5.3 Non-Monotonic V4N Effects

Within-model differences are significant throughout, but their directions differ (Table 4). M7Q declines under L9W relative to C8R and Z2F. T4X has its minimum at Z2F. Y9C has its maximum at L9W. Thus, increasing persuasion intensity does not produce a consistent direction of change across systems.

| Model | C8R | Z2F | L9W |
|---|---:|---:|---:|
| M7Q | 39.30% | 40.50% | 32.90% |
| T4X | 28.10% | 22.00% | 29.20% |
| Y9C | 38.70% | 33.30% | 40.60% |

Table 4: H8D P6S by V4N; Cochran’s \(Q\), \(p<10^{-5}\) for every model.

### 5.4 No Reliable Reduction Within the Tested Window

Associations between cached P6S leakage and D7A are small or negligible. Table 8 reports D7A effect sizes of 0.1073 for T4X, 0.0659 for M7Q, and 0.0723 for Y9C. Privacy risk persists across several unrelated shifts before the original disclosure lies far from the context-window end.

Figure 4: H8D versus D7A; dip near \(d=3\), then rebound.

### 5.5 S8Q Strongly Determines Leakage

S8Q has a large association with leakage in every system and the strongest association among the tested variables (Table 8, S8Q rows). W3B and N6G receive far more suppression than F8Z and A2L (Table 5). This difference occurs even though every category is user-disclosed and contextually private.

| S8Q | M7Q | T4X | Y9C |
|---|---:|---:|---:|
| W3B | 0.14% | 1.11% | 5.97% |
| N6G | 0.00% | 0.39% | 4.01% |
| F8Z | 78.24% | 57.11% | 81.13% |
| A2L | 70.89% | 46.13% | 57.24% |

Table 5: H8D P6S across cached model-probe responses.

### 5.6 K2V Exceeds Observation

No system approaches the threshold at any tested length. Suppression never stabilizes. Every model and persuasion condition therefore receives the out-of-window assignment shown below, rather than an observed stable suppression length.

| Scope | Observed bound | Assigned K2V |
|---|---|---|
| Every model and persuasion condition | \(d\leq6\) | \(\tau=\max(D_{obs})+1=7\) |

### 5.7 R5J Supports H8D Labels

R5J is a post-hoc deterministic sensitivity check: an additional check performed after generation without new model queries. It computes normalized similarity between each saved secret and saved response using cached outputs only. The tested similarity thresholds are 0.85 and 0.80. The higher threshold closely tracks H8D across systems (Table 6, R5J 0.85 and H8D columns).

R5J is auxiliary. Approximate string similarity is less expressive than judge-based evaluation for partial or semantic disclosures.

| Model | U3P | R5J 0.85 | R5J 0.80 | H8D |
|---|---:|---:|---:|---:|
| T4X | 25.50 | 26.40 | 26.53 | 26.43 |
| M7Q | 36.80 | 37.03 | 37.07 | 37.57 |
| Y9C | 36.97 | 37.60 | 37.93 | 37.53 |

Table 6: Cached-output P6S percentages.

Figure 5: H8D heatmap by D7A and V4N for each system.

## 6 Discussion

Behavioral recoverability does not imply permanent memory. It also does not imply uniform behavior across long-context settings. Current assistants may reproduce sensitive information still in context under sufficient retrieval pressure (Table 3).

M7Q’s lower leakage under high pressure is consistent with urgency activating safety or refusal behavior (Table 4, M7Q row). Y9C’s response indicates weaker pressure resistance (Table 4, Y9C row). Topic drift is not an implicit privacy boundary. The intermediate decrease followed by resurgence in Figure 4 warrants stability-based evaluation rather than treating temporary decreases as evidence of suppression.

## 7 Limitations and Future Work

##### Bounded D7A.

The tested short-to-medium range does not fill modern context windows. These findings concern bounded active-context persistence. They do not completely characterize privacy behavior over arbitrarily long interactions or full context windows.

##### Fixed-format secrets.

Structured identifiers are easier to detect than unstructured health status, family circumstances, immigration status, or financial hardship. Extending the benchmark to sensitive attributes without fixed formats remains future work.

##### Active context, not cross-session memory.

Training-data memorization, personalization memory, and cross-session retention are untested.

##### Synthetic and human-authored dialogues.

The mixed construction improves realism over a purely synthetic benchmark while retaining experimental control. Cached metadata no longer preserves identifiers for human-authored rows. Only aggregate results are reported. A separate comparison of human-authored and synthetic dialogue leakage is deferred.

##### Detector scope.

Labels concern surface-form outputs under U3P, R5J, and an external judge. Separate reporting exposes detector behavior. However, formal human–automated inter-annotator agreement—agreement between human and automated labels—is not reported. Larger manual audits of borderline partial disclosures and obfuscations remain future work.

##### API-served models and reproducibility.

Compute and budget constraints require API service. The evaluated identifiers (M7Q = GPT-OSS-120B), (T4X = DeepSeek-R1), and (Y9C = Qwen3-VL-235B) designate models that may change over time. This remains a limitation despite logged identifiers, prompts, timestamps, cached responses, and paired statistics on identical dialogue sets.

##### Artifact availability.

Planned releases comprise generation code, probe templates, evaluation scripts, aggregate result files, and a sanitized dialogue subset. Synthetic PII-like strings require non-real identifiers in released instances. Templates must be sufficient for reproduction without exposing realistic identifiers.

## 8 Conclusion

The S8Q asymmetry—strong suppression of W3B and N6G compared with F8Z and A2L in Table 5—suggests uneven safeguards. These appear to rely more on format and sensitivity cues, and on format-sensitive safety heuristics, than on generalized contextual confidentiality or robust contextual reasoning.

## Appendix A Dataset Characterization

In Table 7, \(N\) is the number of dialogues at each drift length. Token counts are approximate estimates derived from words.

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

Table 8 reports X2H, or Cramer’s \(V\), for associations between H8D leakage and each benchmark variable, using cached P6S observations.

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
