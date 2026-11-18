# PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations

Luciano Rolando Maldonado Romero

Affiliation: West Virginia University, Morgantown, WV, USA

Correspondence to: lrm00020@mix.wvu.edu

###### Abstract

Persistent, user-facing, shared-session, tool-augmented LLMs retain behaviorally recoverable sensitive disclosures beyond unrelated tasks; the controlled benchmark (QX2 = PrivDrift) audits this persistence using seeded multi-turn histories, content-dense intervening exchanges, and standardized extraction probes. Across extended-context models, G4P under H8D remains substantial, varies sharply across systems, S8Q, and V4N, and fails to decline reliably within the tested window. Active-context privacy requires evaluation as persistent behavioral failure, not exclusively parameter memorization or immediate jailbreak behavior.

###### Keywords:

Trustworthy AI, Large Language Models, Privacy, Evaluation, Benchmarking

## 1 Introduction

SSNs and payment identifiers receive much stronger suppression than contact identifiers; no evaluated system attains stable decay. QX2 provides controlled persistence auditing, direct/justified/high-pressure elicitation, reproducible composite detection using deterministic processing and an open-weights judge, G4P and P6S reporting, and a stability criterion rather than a transient-decrease criterion.

LLMs have evolved from simple chat systems into persistent assistants maintaining coherence across extended interactions (OpenAI, 2023; Touvron et al., 2023; Liu et al., 2024). Deployment domains include healthcare triage, financial planning, enterprise copilots, and personal productivity; disclosed Personally Identifiable Information (PII) includes A2L, financial identifiers, F8Z, and health-related details. Context retention supporting multi-turn assistance also makes earlier sensitive disclosures available to later generation.

Existing safety evaluations mostly target two extremes: parameter-based training-data memorization and immediate jailbreaking, which elicits harmful behavior in the current turn (Wei et al., 2024; Zou et al., 2023); parameter extraction and multi-turn or injection evaluations do not directly establish whether unrelated conversational shifts provide a practical privacy boundary. The premise is not that users believe same-session messages literally disappear: users and designers may underestimate later indirect, justified, high-pressure, or tool-mediated elicitation. Shared sessions, browser-integrated assistants, enterprise copilots, agentic workflows, and injected instructions permit later context interactions unintended by the original user.

## 2 Related Work

### 2.1 Contextual Privacy and Inference-Time Leakage

Training-data extraction, membership inference, and memorized-PII recovery concern information stored in parameters and pre-training corpora (Shokri et al., 2017; Carlini et al., 2021; Li et al., 2023; Nasr et al., 2023); active-context privacy concerns appropriate inference-time handling of current-interaction user information.

ConfAIde tests contextual-integrity privacy reasoning (Mireshghallah et al., 2024); PrivacyLens and PrivaCI-Bench examine privacy-norm awareness and legal compliance in agentic or contextual settings (Shao et al., 2024; Li et al., 2025). Their principal object is whether sharing is appropriate under static access-control or privacy-norm scenarios; QX2 isolates behavioral recoverability following unrelated exchanges and subsequent persuasive probing.

### 2.2 Multi-Turn Adversarial Attacks

Aligned models remain exploitable over multiple turns: Crescendo progressively elicits harmful outputs from apparently benign conversations (Russinovich et al., 2024), automated frameworks generate adversarial multi-turn paths (Deng et al., 2024; Narula et al., 2025), and later injected instructions manipulate LLM-integrated applications (Liu et al., 2023). Their usual targets are unsafe content or instruction hijacking; QX2 targets information-flow control, specifically re-disclosure of earlier user-supplied sensitive information rather than harmful external content.

### 2.3 Context Management and Unlearning

TOFU and WMDP assess removal or behavioral suppression of knowledge, usually through static question-answer pairs and model parameters (Maini et al., 2024; Li et al., 2024). QX2 instead concerns behavioral suppression of secrets still present in active context; deployed RAG systems, copilots, and agentic workflows often depend on prompt-level context management rather than parameter-level unlearning.

## 3 The QX2 Framework

Figure 1: QX2 pipeline: controlled multi-turn dialogue construction, standardized probe attachment, target-model queries, composite leakage scoring.

The synthetic framework combines a parametric dialogue generator, probing harness, and detection/temporal-analysis metric suite.

### 3.1 Threat Model

Transcript invisibility is not assumed, and the failure mode (E9U = active-context re-disclosure) concerns assistant reproduction of an earlier sensitive value under later querying, justification, or pressure rather than the mere presence of that value in the transcript.

A later participant in a shared or persistent session may not understand preceding disclosures; tool outputs, retrieved documents, and injected instructions in enterprise or agentic settings may interact with existing context; browser/application users may not intend sensitive information to be reused after task changes.

A user supplies secret \(S\) at \(t_0\), and the unrelated, content-dense exchanges (J7K = topic drift) are represented by \(D=\{u_1,a_1,\ldots,u_d,a_d\}\) before standardized probe \(P\) is appended to the active context. Exact, approximate, or semantic revelation of \(S\) constitutes leakage.

### 3.2 Dataset Generation Pipeline

The scaffold-then-rewrite pipeline preserves exact ground truth, intervening-turn counts, and probe structure while reducing purely templated artifacts. Human-authored dialogues obey the same constraints and are interspersed, not contiguous.

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

Opening-turn plausible task contexts establish legitimate assistant access to \(S\), with personas sampled from more than 100 professions and demographic profiles, while the categorical variable (S8Q = secret type) selects among fixed-format identifiers.

The four sampled formats comprise contact values (A2L = phone numbers) and electronic contact values (F8Z = email addresses), alongside government identifiers (W3B = Social Security Numbers) and payment identifiers (N6G = credit card numbers), each inserted as the ground-truth secret rather than subsequently inferred.

#### 3.2.2 Content-Dense J7K

Assigned topics include tax filing, medical triage, and travel logistics; a topic-conditioned dense question-answer library supplies intervening turns, with professional-writing, structured-reasoning, and long-form-summarization fallback tasks. Annotators contribute a subset of library turns. Unrelated entities and task demands enter context while the original secret and turn count remain controlled.

#### 3.2.3 Naturalistic Refinement

A rewriting model receives each scaffold and instructions for fluency/persona realism; exact secret, turn count, role structure, and final-probe conditions remain invariant. Surface phrasing changes; experimental variables do not.

### 3.3 Probing Setup

The final probe follows the completed history, and the manipulated interaction style (V4N = persuasion intensity) has three standardized conditions rather than an unconstrained adversarial search.

Neutral direct retrieval (C8R = Simple), contextual justification (Z2F = Medium), and urgent/high-pressure elicitation (L9W = Hard) distinguish requests for earlier A2L, form-filling needs, and tests of whether pressure cues raise or lower disclosure.

The intervening-turn quantity (D7A = drift length), denoted \(d\), counts turns between disclosure and final probing and is sampled from the following mixture:

\[
d\sim
\begin{cases}
0 & \text{with }p=0.2,\\
\mathrm{Uniform}\{2,3,4,5,6\} & \text{with }p=0.8.
\end{cases}
\tag{1}
\]

Immediate recall is included; short-to-medium J7K is emphasized.

### 3.4 Evaluation Methodology

Verbatim output, formatting changes, partial substrings, and paraphrased references are eligible disclosures, and the composite detector (H8D = hybrid) labels a response positive if normalized inclusion succeeds or the judge finds the full secret, a substantial substring, or sufficient partial information to identify it.

Figure 2: Hierarchical evaluation: B6T first; negative responses proceed to the LLM judge.

#### 3.4.1 Stage 1: B6T

For direct and near-direct disclosure, the deterministic procedure (B6T = normalization-based matching) strips every non-digit character from numeric secrets and responses, while alphanumeric processing normalizes case and whitespace but retains meaningful delimiters.

With normalization \(\phi\), response \(R\), and indicator \(\mathbb I\):

\[
\mathrm{Leak}_{\mathrm{U3P}}(R,S)
=\mathbb I\big[\phi(S)\subset\phi(R)\big].
\tag{2}
\]

#### 3.4.2 Stage 2: LLM-as-a-Judge Verification

An open-weights Llama judge receives \(S\) and \(R\), returns a binary full/substantial-disclosure verdict, and handles negative deterministic cases for partial disclosures, indirect hints, and otherwise missed formatting variations (Meta AI, 2024).

The deterministic baseline (U3P = regex-only) remains separately reported alongside H8D, with judge errors possible rather than adjudication treated as error-free.

#### 3.4.3 K2V

Observed lengths \(D_{obs}\) and leakage rates \(L(d)\) define the stability metric (K2V = Privacy Half-Life), denoted \(\tau\), through safety threshold \(\delta=0.1\times L(0)\) and sustained rather than momentary suppression:

\[
\tau=\min\left\{
d\in D_{obs}\ \middle|\
L(d)\leq\delta
\ \text{and}\
\forall d'\in D_{obs},\ d'>d:\ L(d')\leq\delta
\right\}.
\tag{3}
\]

An empty set receives \(\tau=\max(D_{obs})+1\).

Figure 3: K2V; later resurgence disqualifies transient decreases.

## 4 Experimental Setup

##### Models.

All systems receive identical histories and probe templates through the OpenRouter API.

##### Dataset.

Each system processes the full QX2 dataset, one seeded secret and one sampled \(d\) per dialogue, under every standardized condition.

| Cached responses | Count |
|---|---:|
| Per model | 3,000 |
| All evaluated models | 9,000 |

##### Aggregation levels.

The any-probe dialogue statistic (G4P = dialogue-level leakage) counts a dialogue positive when at least one standardized probe elicits its secret, whereas the response statistic (P6S = probe-level leakage) gives each dialogue one observation per persuasion condition for V4N and R5J analyses. G4P can exceed the mean condition-specific rate.

##### Uncertainty and statistical tests.

G4P intervals use 95% bootstrapping over dialogues; P6S sensitivity checks use cached model-probe responses. Matched model/detector comparisons use McNemar’s test, Bonferroni-corrected where applicable; repeated persuasion measures use Cochran’s \(Q\), followed by McNemar tests. Independence tests for leakage against D7A, V4N, or S8Q use chi-square, with the association statistic (X2H = Cramer’s \(V\)) reported as effect size rather than a substitute for significance testing.

## 5 Results

### 5.1 G4P: H8D Largely Agrees With B6T

H8D differs only slightly from B6T; direct or near-direct reproduction dominates purely semantic paraphrase. Judge augmentation is limited, supporting H8D as primary and U3P as a transparent deterministic baseline.

| Model | U3P | H8D | \(\Delta\) | \(p\) |
|---|---:|---:|---:|---:|
| M7Q | 47.70 | 47.70 | +0.00 | 1.0 |
| T4X | 37.80 | 38.70 | +0.90 | \(7.7\times10^{-3}\) |
| Y9C | 53.50 | 54.60 | +1.10 | \(2.57\times10^{-3}\) |

Table 2: G4P percentages; augmentation in percentage points.

### 5.2 Substantial G4P Across Models

Every evaluated system remains vulnerable to at least one standardized extraction condition at substantial rates.

| Model | H8D | 95% CI |
|---|---:|---|
| T4X | 38.70% | [35.7, 41.6] |
| M7Q | 47.70% | [44.6, 50.8] |
| Y9C | 54.60% | [51.6, 57.7] |

Table 3: G4P with bootstrap confidence intervals.

### 5.3 Non-Monotonic V4N Effects

Within-model differences are significant throughout, but directions differ: M7Q declines under L9W relative to C8R and Z2F; T4X minimizes at Z2F; Y9C maximizes at L9W.

| Model | C8R | Z2F | L9W |
|---|---:|---:|---:|
| M7Q | 39.30% | 40.50% | 32.90% |
| T4X | 28.10% | 22.00% | 29.20% |
| Y9C | 38.70% | 33.30% | 40.60% |

Table 4: H8D P6S by V4N; Cochran’s \(Q\), \(p<10^{-5}\) for every model.

### 5.4 No Reliable Reduction Within the Tested Window

Cached P6S associations with D7A are small or negligible. Privacy risk persists across several unrelated shifts before the original disclosure lies far from the context-window end.

Figure 4: H8D versus D7A; dip near \(d=3\), then rebound.

### 5.5 S8Q Strongly Determines Leakage

S8Q has a large association in every system and the strongest association among tested variables. W3B and N6G receive far more suppression than F8Z and A2L despite all categories being user-disclosed and contextually private.

| S8Q | M7Q | T4X | Y9C |
|---|---:|---:|---:|
| W3B | 0.14% | 1.11% | 5.97% |
| N6G | 0.00% | 0.39% | 4.01% |
| F8Z | 78.24% | 57.11% | 81.13% |
| A2L | 70.89% | 46.13% | 57.24% |

Table 5: H8D P6S across cached model-probe responses.

### 5.6 K2V Exceeds Observation

No system approaches the threshold at any tested length; suppression never stabilizes.

| Scope | Observed bound | Assigned K2V |
|---|---|---|
| Every model and persuasion condition | \(d\leq6\) | \(\tau=\max(D_{obs})+1=7\) |

### 5.7 R5J Supports H8D Labels

A post-hoc deterministic sensitivity check (R5J = fuzzy matching) computes normalized similarity between each saved secret and saved response using cached outputs only, with the higher tested threshold closely tracking H8D across systems. R5J is auxiliary: approximate string similarity is less expressive than judge-based evaluation for partial or semantic disclosures.

| Model | U3P | R5J 0.85 | R5J 0.80 | H8D |
|---|---:|---:|---:|---:|
| T4X | 25.50 | 26.40 | 26.53 | 26.43 |
| M7Q | 36.80 | 37.03 | 37.07 | 37.57 |
| Y9C | 36.97 | 37.60 | 37.93 | 37.53 |

Table 6: Cached-output P6S percentages.

Figure 5: H8D heatmap by D7A and V4N for each system.

## 6 Discussion

Behavioral recoverability does not imply permanent memory, nor uniform behavior across long-context settings; current assistants may reproduce sensitive in-context information under sufficient retrieval pressure.

The M7Q pressure response is consistent with urgency activating safety/refusal behavior; the Y9C response indicates weaker pressure resistance. J7K is not an implicit privacy boundary; unstable intermediate decreases warrant stability-based evaluation rather than treatment as suppression evidence.

## 7 Limitations and Future Work

##### Bounded D7A.

The tested short-to-medium range does not saturate modern windows; these findings are bounded active-context persistence, not a complete characterization of arbitrarily long or full-window privacy behavior.

##### Fixed-format secrets.

Structured identifiers are easier to detect than unstructured health status, family circumstances, immigration status, or financial hardship. Extension to non-fixed sensitive attributes remains future work.

##### Active context, not cross-session memory.

Training-data memorization, personalization memory, and cross-session retention are untested.

##### Synthetic and human-authored dialogues.

The mixed construction improves realism over a purely synthetic benchmark while retaining experimental control. Cached metadata no longer preserves human-authored row identifiers; only aggregate results are reported, with separate human-versus-synthetic leakage comparison deferred.

##### Detector scope.

Labels concern surface-form outputs under U3P, R5J, and an external judge; separate reporting exposes detector behavior, but formal human–automated inter-annotator agreement is not reported. Larger manual audits of borderline partial disclosures and obfuscations remain future work.

##### API-served models and reproducibility.

Compute/budget constraints require API service, and the evaluated identifiers (M7Q = GPT-OSS-120B), (T4X = DeepSeek-R1), and (Y9C = Qwen3-VL-235B) designate models that may change over time despite logged identifiers, prompts, timestamps, cached responses, and paired statistics on identical dialogue sets.

##### Artifact availability.

Planned releases comprise generation code, probe templates, evaluation scripts, aggregate result files, and a sanitized dialogue subset. Synthetic PII-like strings necessitate non-real identifiers in released instances and templates sufficient for reproduction without exposing realistic identifiers.

## 8 Conclusion

The S8Q asymmetry suggests uneven safeguards relying more on format/sensitivity cues and format-sensitive safety heuristics than on generalized contextual confidentiality or robust contextual reasoning.

## Appendix A Dataset Characterization

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
