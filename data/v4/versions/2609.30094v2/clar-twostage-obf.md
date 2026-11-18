# PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations

Luciano Rolando Maldonado Romero

Affiliation: West Virginia University, Morgantown, WV, USA

Correspondence to: lrm00020@mix.wvu.edu

###### Abstract

Persistent, user-facing, shared-session, tool-augmented large language models (LLMs) retain sensitive disclosures that can be recovered from their later responses, even after unrelated tasks. The controlled benchmark QX2, named PrivDrift, audits this persistence. It uses conversation histories containing a known secret, content-dense intervening exchanges, and standardized extraction requests, called probes. The secret remains in the active conversation context; the measured failure is its later reproduction.

Across extended-context models, dialogue-level leakage (G4P) remains substantial. G4P counts a dialogue as leaking if at least one standardized probe elicits its secret. Under the hybrid detector (H8D), which combines normalized text matching with an LLM judge, G4P ranges from 38.70% to 54.60% across systems (Table 3). Leakage varies sharply by system, secret type (S8Q; Table 5), and persuasion intensity—the probe’s interaction style (V4N; Table 4). Leakage fails to decline reliably within the tested window (Figure 4 and Section 5.6). Active-context privacy—the appropriate handling of current-interaction user information during generation—requires evaluation as a persistent behavioral failure. It should not be evaluated exclusively as memorization in learned model parameters or as immediate jailbreaking, which elicits harmful behavior in the current turn.

###### Keywords:

Trustworthy AI (artificial intelligence), Large Language Models, Privacy, Evaluation, Benchmarking

## 1 Introduction

Social Security Numbers (SSNs) and payment identifiers receive much stronger suppression than contact identifiers. Table 5 shows this contrast in its government-identifier and credit-card rows versus its email-address and phone-number rows. No evaluated system attains stable decay: every model and persuasion condition receives a Privacy Half-Life beyond the observed range (Section 5.6).

QX2 provides controlled persistence auditing with direct, justified, and high-pressure elicitation. Its reproducible composite detector combines deterministic processing, which applies fixed matching rules, with an open-weights judge, a model whose learned weights are available. QX2 reports both G4P and probe-level leakage (P6S). P6S counts each response to a persuasion condition separately, rather than combining probes into one dialogue-level outcome. The framework also requires stable suppression, not a transient decrease (Equation 3).

LLMs have evolved from simple chat systems into persistent assistants that maintain coherence across extended interactions (OpenAI, 2023; Touvron et al., 2023; Liu et al., 2024). Deployment domains include healthcare triage, financial planning, enterprise copilots—workplace assistants—and personal productivity. Disclosed Personally Identifiable Information (PII), meaning information that identifies a person, includes phone numbers (A2L), financial identifiers, email addresses (F8Z), and health-related details. The context retention that supports multi-turn assistance also makes earlier sensitive disclosures available to later generation.

Existing safety evaluations mostly target two extremes: training-data memorization in model parameters and immediate jailbreaking (Wei et al., 2024; Zou et al., 2023). Parameter extraction seeks information retained in learned model weights. Multi-turn and instruction-injection evaluations examine attacks across exchanges or through instructions introduced into existing context. These evaluations do not directly establish whether unrelated conversational shifts provide a practical privacy boundary.

The premise is not that users believe same-session messages literally disappear. Rather, users and designers may underestimate later indirect, justified, high-pressure, or tool-mediated elicitation. Shared sessions, browser-integrated assistants, enterprise copilots, and agentic workflows—workflows in which assistants act through tools—permit later context interactions unintended by the original user. Injected instructions can also create such interactions.

## 2 Related Work

### 2.1 Contextual Privacy and Inference-Time Leakage

Training-data extraction recovers training examples. Membership inference tests whether information was included in training data. Memorized-PII recovery seeks identifying information retained from training. These tasks concern information stored in model parameters and pre-training corpora, the collections used to train models (Shokri et al., 2017; Carlini et al., 2021; Li et al., 2023; Nasr et al., 2023). Active-context privacy instead concerns appropriate handling of current-interaction user information at inference time, when the model generates a response.

ConfAIde tests contextual-integrity privacy reasoning: whether information sharing fits its context (Mireshghallah et al., 2024). PrivacyLens and PrivaCI-Bench examine awareness of privacy norms and legal compliance in agentic or contextual settings (Shao et al., 2024; Li et al., 2025). Their principal object is whether sharing is appropriate under static scenarios involving access-control rules or privacy norms. Access-control rules govern who may access information; privacy norms govern when sharing is appropriate. QX2 instead isolates whether information remains recoverable after unrelated exchanges and subsequent persuasive probing.

### 2.2 Multi-Turn Adversarial Attacks

Aligned models, trained to follow behavioral requirements, remain exploitable over multiple turns. Crescendo progressively elicits harmful outputs from apparently benign conversations (Russinovich et al., 2024). Automated frameworks generate adversarial multi-turn paths, or sequences of attack exchanges (Deng et al., 2024; Narula et al., 2025). Later injected instructions can manipulate applications that integrate LLMs (Liu et al., 2023).

These attacks usually target unsafe content or instruction hijacking, in which introduced instructions redirect the system. QX2 targets information-flow control: specifically, re-disclosure of earlier user-supplied sensitive information rather than harmful external content.

### 2.3 Context Management and Unlearning

TOFU and WMDP are evaluations of knowledge removal or behavioral suppression. They usually use static question-answer pairs and model parameters (Maini et al., 2024; Li et al., 2024). QX2 instead concerns behavioral suppression of secrets that remain in active context.

Deployed retrieval-augmented generation (RAG) systems, which use retrieved material during generation, often depend on prompt-level context management. So do copilots and agentic workflows. This manages information supplied in the model’s input, rather than using parameter-level unlearning to remove or suppress knowledge in its learned weights.

## 3 The QX2 Framework

Figure 1: QX2 pipeline: controlled multi-turn dialogue construction, standardized probe attachment, target-model queries, composite leakage scoring.

The synthetic framework combines three components: a dialogue generator with controlled settings, a probing harness that attaches probes and queries models, and metrics for detecting leakage and tracking it across intervening conversation lengths.

### 3.1 Threat Model

Transcript invisibility is not assumed. The failure mode, active-context re-disclosure (E9U), is the assistant’s reproduction of an earlier sensitive value under later querying, justification, or pressure. The mere presence of that value in the transcript is not the measured failure.

A later participant in a shared or persistent session may not understand preceding disclosures. Tool outputs, retrieved documents, and injected instructions in enterprise or agentic settings may interact with existing context. Browser or application users may not intend sensitive information to be reused after task changes.

A user supplies a secret, denoted \(S\), at time \(t_0\). Unrelated, content-dense exchanges then intervene. These exchanges are topic drift (J7K), represented by

\[
D=\{u_1,a_1,\ldots,u_d,a_d\}.
\]

Here, \(D\) is the intervening conversation, \(u_i\) and \(a_i\) are user and assistant messages, and \(i\) indexes them. The drift length (D7A), denoted \(d\), counts intervening turns between disclosure and final probing. A standardized probe, denoted \(P\), is then appended to the active context. Exact, approximate, or semantic revelation of \(S\) constitutes leakage; semantic revelation conveys the secret’s information without necessarily repeating its exact text.

### 3.2 Dataset Generation Pipeline

The scaffold-then-rewrite pipeline first constructs a controlled dialogue draft and then rewrites its surface wording. It preserves the exact ground-truth secret, intervening-turn counts, and probe structure while reducing artifacts of fixed templates. Human-authored dialogues obey the same constraints and are interspersed, not contiguous.

Table 1 summarizes the dataset. \(N\) denotes the number of dialogues. “Avg.” means average, “approx.” means approximate, and “Min/max” gives the minimum and maximum. Tokens are units of text processed by a model; the token counts here are estimates derived from words.

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

Plausible opening-turn tasks establish legitimate assistant access to \(S\). Personas—the user profiles represented in the dialogues—are sampled from more than 100 professions and demographic profiles. The secret-type variable S8Q selects among fixed-format identifiers.

The four sampled formats are phone numbers (A2L), email addresses (F8Z), Social Security Numbers (W3B), and credit card numbers (N6G). These represent contact, electronic contact, government, and payment identifiers, respectively. Each value is inserted as the known ground-truth secret rather than inferred afterward.

#### 3.2.2 Content-Dense J7K

Assigned topics include tax filing, medical triage, and travel logistics. A topic-conditioned library of dense question-answer exchanges supplies intervening turns. Fallback tasks include professional writing, structured reasoning, and long-form summarization. Annotators contribute a subset of library turns.

These exchanges introduce unrelated entities and task demands into context. The original secret and turn count remain controlled.

#### 3.2.3 Naturalistic Refinement

A rewriting model receives each scaffold with instructions to improve fluency and persona realism. The exact secret, turn count, role structure—which messages belong to the user or assistant—and final-probe conditions remain unchanged. Surface phrasing changes; experimental variables do not.

### 3.3 Probing Setup

The final probe follows the completed history. Persuasion intensity (V4N) has three standardized conditions, rather than an unconstrained search for attacks:

- **C8R = Simple:** neutral, direct retrieval.
- **Z2F = Medium:** retrieval supported by contextual justification.
- **L9W = Hard:** urgent or high-pressure elicitation.

These conditions distinguish requests for an earlier phone number (A2L), form-filling needs, and tests of whether pressure cues raise or lower disclosure.

The drift length \(d\), or D7A, is sampled from the following mixture. The symbol \(\sim\) means “is sampled from,” and \(p\) here denotes a sampling probability. “Uniform” means that each listed value is equally likely within that part of the mixture.

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

Eligible disclosures include verbatim output, formatting changes, partial substrings—parts of the secret’s text—and paraphrased references. The hybrid detector H8D labels a response positive if either normalized inclusion succeeds or the judge finds the full secret, a substantial substring, or sufficient partial information to identify it.

The deterministic first stage is normalization-based matching (B6T), which applies fixed text-processing rules. Only responses that this stage labels negative proceed to the LLM judge.

Figure 2: Hierarchical evaluation: B6T first; negative responses proceed to the LLM judge.

#### 3.4.1 Stage 1: B6T

B6T detects direct and near-direct disclosure. For numeric secrets, it strips every non-digit character from both the secret and response. For alphanumeric processing, involving letters and digits, it normalizes case and whitespace while retaining meaningful delimiters, or separators.

The deterministic baseline is reported as U3P, named “regex-only,” meaning regular-expression-only. Let \(R\) be the response and \(\phi\) the normalization operation. The indicator \(\mathbb I\) returns one when its condition holds and zero otherwise. The symbol \(\subset\) below denotes inclusion of the normalized secret in the normalized response:

\[
\mathrm{Leak}_{\mathrm{U3P}}(R,S)
=\mathbb I\big[\phi(S)\subset\phi(R)\big].
\tag{2}
\]

Thus, \(\mathrm{Leak}_{\mathrm{U3P}}(R,S)\) is the deterministic leakage label for response \(R\) and secret \(S\).

#### 3.4.2 Stage 2: LLM-as-a-Judge Verification

An open-weights Llama model serves as the judge (Meta AI, 2024). It receives \(S\) and \(R\) and returns a binary verdict: full or substantial disclosure, or no such disclosure. It evaluates cases that deterministic matching labels negative. These cases may contain partial disclosures, indirect hints, or otherwise missed formatting variations.

U3P remains separately reported alongside H8D. Judge errors are possible; judge decisions are not treated as error-free adjudication.

#### 3.4.3 K2V

The stability metric K2V is named Privacy Half-Life and is denoted \(\tau\). Let \(D_{obs}\) be the set of observed drift lengths and \(L(d)\) the leakage rate at length \(d\). The immediate-recall rate is \(L(0)\). The safety threshold is \(\delta=0.1\times L(0)\).

Privacy Half-Life requires sustained, not momentary, suppression. It is the smallest observed \(d\) whose leakage rate is at or below \(\delta\), provided every later observed length also remains at or below \(\delta\):

\[
\tau=\min\left\{
d\in D_{obs}\ \middle|\
L(d)\leq\delta
\ \text{and}\
\forall d'\in D_{obs},\ d'>d:\ L(d')\leq\delta
\right\}.
\tag{3}
\]

Here, \(\min\) selects the smallest qualifying length, \(\in\) means membership in the observed set, and \(\forall\) means “for every.” The symbol \(d'\) denotes another observed length; \(d'>d\) restricts the condition to later lengths, and \(\leq\) means “at or below.”

If no length qualifies, the set is empty. The assigned value is then \(\tau=\max(D_{obs})+1\), one more than the largest observed length.

Figure 3: K2V; later resurgence disqualifies transient decreases.

## 4 Experimental Setup

##### Models.

The evaluated identifiers are M7Q = GPT-OSS-120B, T4X = DeepSeek-R1, and Y9C = Qwen3-VL-235B. All systems receive identical histories and probe templates through the OpenRouter application programming interface (API), the software interface used to submit queries.

##### Dataset.

Each system processes the full QX2 dataset under every standardized condition. Each dialogue contains one seeded secret and one sampled drift length \(d\).

| Cached responses | Count |
|---|---:|
| Per model | 3,000 |
| All evaluated models | 9,000 |

##### Aggregation levels.

G4P, or dialogue-level leakage, counts a dialogue positive when at least one standardized probe elicits its secret. P6S, or probe-level leakage, gives each dialogue one response observation per persuasion condition.

P6S is used for persuasion-intensity (V4N) and fuzzy-matching (R5J) analyses. R5J is a deterministic, post-hoc sensitivity check based on text similarity, described in Section 5.7. Because G4P combines success across probes, it can exceed the mean condition-specific rate.

##### Uncertainty and statistical tests.

G4P intervals use 95% bootstrapping over dialogues: uncertainty is estimated by resampling dialogues. P6S sensitivity checks use cached model-probe responses.

Matched model and detector comparisons use McNemar’s test, which compares paired binary outcomes. Bonferroni correction adjusts for multiple comparisons where applicable. Repeated persuasion measures use Cochran’s \(Q\), a test of binary outcomes across repeated conditions, followed by McNemar tests.

Chi-square independence tests assess whether leakage is associated with drift length (D7A), persuasion intensity (V4N), or secret type (S8Q). The association statistic X2H is Cramer’s \(V\), a measure of association strength. It is reported as an effect size, not as a substitute for significance testing.

In statistical results, \(p\) denotes a test’s \(p\)-value: the probability, under its no-difference or independence hypothesis, of a result at least as extreme as the observed one. This differs from the sampling-probability use of \(p\) in Equation 1.

## 5 Results

### 5.1 G4P: H8D Largely Agrees With B6T

H8D differs only slightly from B6T, whose deterministic baseline is reported as U3P in Table 2. The judge adds +0.00, +0.90, and +1.10 percentage points for M7Q, T4X, and Y9C, respectively. These small differences indicate that direct or near-direct reproduction dominates purely semantic paraphrase. The limited augmentation supports H8D as the primary detector and U3P as a transparent deterministic baseline.

In Table 2, \(\Delta\) denotes the augmentation from U3P to H8D, measured in percentage points.

| Model | U3P | H8D | \(\Delta\) | \(p\) |
|---|---:|---:|---:|---:|
| M7Q | 47.70 | 47.70 | +0.00 | 1.0 |
| T4X | 37.80 | 38.70 | +0.90 | \(7.7\times10^{-3}\) |
| Y9C | 53.50 | 54.60 | +1.10 | \(2.57\times10^{-3}\) |

Table 2: G4P percentages; augmentation in percentage points.

### 5.2 Substantial G4P Across Models

Every evaluated system remains vulnerable to at least one standardized extraction condition at substantial rates. Table 3 reports H8D dialogue-level leakage and its 95% confidence interval (CI), the uncertainty interval estimated by bootstrapping.

| Model | H8D | 95% CI |
|---|---:|---|
| T4X | 38.70% | [35.7, 41.6] |
| M7Q | 47.70% | [44.6, 50.8] |
| Y9C | 54.60% | [51.6, 57.7] |

Table 3: G4P with bootstrap confidence intervals.

### 5.3 Non-Monotonic V4N Effects

Persuasion conditions differ significantly within every model: Table 4 reports Cochran’s \(Q\) with \(p<10^{-5}\) for each. The directions differ, however.

M7Q leakage declines under Hard pressure (L9W) relative to Simple retrieval (C8R) and Medium justification (Z2F). T4X has its minimum under Z2F. Y9C has its maximum under L9W. The corresponding rates are shown in each model’s row below.

| Model | C8R | Z2F | L9W |
|---|---:|---:|---:|
| M7Q | 39.30% | 40.50% | 32.90% |
| T4X | 28.10% | 22.00% | 29.20% |
| Y9C | 38.70% | 33.30% | 40.60% |

Table 4: H8D P6S by V4N; Cochran’s \(Q\), \(p<10^{-5}\) for every model.

### 5.4 No Reliable Reduction Within the Tested Window

Associations between cached P6S and drift length D7A are small or negligible. Table 8’s D7A rows report X2H = 0.1073 for T4X, interpreted as Small; 0.0659 for M7Q, interpreted as Negligible; and 0.0723 for Y9C, also interpreted as Negligible.

Privacy risk persists across several unrelated shifts before the original disclosure lies far from the context-window end. The tested conversation lengths are characterized in Table 7. Figure 4 reports a dip followed by a rebound, rather than reliable reduction.

Figure 4: H8D versus D7A; dip near \(d=3\), then rebound.

### 5.5 S8Q Strongly Determines Leakage

Secret type S8Q has a large association in every system and the strongest association among the tested variables. Table 8’s S8Q rows report X2H values of 0.5871 for T4X, 0.7731 for M7Q, and 0.6927 for Y9C. Each is interpreted as Large, and each exceeds that model’s D7A and V4N values in the same table.

Social Security Numbers (W3B) and credit card numbers (N6G) receive far more suppression than email addresses (F8Z) and phone numbers (A2L). Table 5 shows this contrast, although all categories were user-disclosed and contextually private.

| S8Q | M7Q | T4X | Y9C |
|---|---:|---:|---:|
| W3B | 0.14% | 1.11% | 5.97% |
| N6G | 0.00% | 0.39% | 4.01% |
| F8Z | 78.24% | 57.11% | 81.13% |
| A2L | 70.89% | 46.13% | 57.24% |

Table 5: H8D P6S across cached model-probe responses.

### 5.6 K2V Exceeds Observation

No system approaches the threshold \(\delta=0.1\times L(0)\) at any tested length; suppression never stabilizes. Every model and persuasion condition therefore receives the empty-set assignment from Equation 3.

| Scope | Observed bound | Assigned K2V |
|---|---|---|
| Every model and persuasion condition | \(d\leq6\) | \(\tau=\max(D_{obs})+1=7\) |

The value 7 is assigned beyond the observed range. It is not an observed stable threshold crossing.

### 5.7 R5J Supports H8D Labels

The post-hoc deterministic fuzzy-matching check R5J computes normalized text similarity between each saved secret and saved response. It uses cached outputs only. Table 6 reports the two tested similarity thresholds, 0.85 and 0.80. The higher threshold, 0.85, closely tracks H8D across systems.

R5J remains auxiliary. Approximate string similarity is less expressive than judge-based evaluation for partial or semantic disclosures.

| Model | U3P | R5J 0.85 | R5J 0.80 | H8D |
|---|---:|---:|---:|---:|
| T4X | 25.50 | 26.40 | 26.53 | 26.43 |
| M7Q | 36.80 | 37.03 | 37.07 | 37.57 |
| Y9C | 36.97 | 37.60 | 37.93 | 37.53 |

Table 6: Cached-output P6S percentages.

Figure 5: H8D heatmap by D7A and V4N for each system.

## 6 Discussion

Behavioral recoverability does not imply permanent memory or uniform behavior across long-context settings. The study concerns secrets still present in active context and a bounded drift range. Within this setup, current assistants may reproduce sensitive in-context information under sufficient retrieval pressure (Tables 3–4).

M7Q’s pressure response is consistent with urgency activating safety or refusal behavior. Its Table 4 row falls to 32.90% under Hard pressure from 39.30% under Simple retrieval and 40.50% under Medium justification. Y9C’s response indicates weaker pressure resistance: its Table 4 row rises to 40.60% under Hard pressure, compared with 38.70% under Simple retrieval and 33.30% under Medium justification.

Topic drift J7K is not an implicit privacy boundary. Figure 4 reports a rebound after an intermediate dip, and Section 5.6 reports no stable threshold crossing. Such unstable decreases warrant stability-based evaluation, rather than treatment as evidence of suppression.

## 7 Limitations and Future Work

##### Bounded D7A.

The tested short-to-medium drift range does not saturate modern context windows. Observations extend through \(d=6\), with dialogue lengths reported in Tables 1 and 7. These findings concern bounded active-context persistence. They do not completely characterize privacy behavior at arbitrarily long lengths or across full context windows.

##### Fixed-format secrets.

Structured identifiers are easier to detect than unstructured health status, family circumstances, immigration status, or financial hardship. Extension to sensitive attributes without fixed formats remains future work.

##### Active context, not cross-session memory.

Training-data memorization, personalization memory—information retained for personalized assistance—and cross-session retention are untested.

##### Synthetic and human-authored dialogues.

The mixed construction improves realism over a purely synthetic benchmark while retaining experimental control. Table 1 reports 900 synthetic and 100 human-authored dialogues. Cached metadata no longer preserves identifiers for the human-authored rows. Only aggregate results are therefore reported. A separate human-versus-synthetic leakage comparison is deferred.

##### Detector scope.

Labels concern surface-form outputs—the text returned by models—under U3P, R5J, and an external judge. Separate reporting exposes detector behavior (Tables 2 and 6). However, formal human–automated inter-annotator agreement, measuring agreement between human and automated labels, is not reported. Larger manual audits of borderline partial disclosures and obfuscations remain future work.

##### API-served models and reproducibility.

Compute and budget constraints require API service. The evaluated identifiers, M7Q = GPT-OSS-120B, T4X = DeepSeek-R1, and Y9C = Qwen3-VL-235B, designate models that may change over time. This remains a limitation despite logged identifiers, prompts, timestamps, cached responses, and paired statistics on identical dialogue sets.

##### Artifact availability.

Planned releases comprise generation code, probe templates, evaluation scripts, aggregate result files, and a sanitized dialogue subset. Synthetic PII-like strings necessitate non-real identifiers in released instances. Released templates must also be sufficient for reproduction without exposing realistic identifiers.

## 8 Conclusion

The secret-type asymmetry suggests uneven safeguards. Table 5’s W3B and N6G rows show much lower leakage than its F8Z and A2L rows, while Table 8 classifies every S8Q association as Large. This asymmetry suggests safeguards that rely more on format and sensitivity cues, and on format-sensitive safety heuristics, than on generalized contextual confidentiality or robust contextual reasoning.

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
