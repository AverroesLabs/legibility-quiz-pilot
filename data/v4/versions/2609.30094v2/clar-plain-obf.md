# PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations

Luciano Rolando Maldonado Romero

Affiliation: West Virginia University, Morgantown, WV, USA

Correspondence to: lrm00020@mix.wvu.edu

###### Abstract

Persistent large language models (LLMs) can reproduce sensitive user disclosures after unrelated tasks. This risk applies to user-facing systems, shared sessions, and systems that use tools. The controlled benchmark QX2, also called PrivDrift, audits this persistence. It uses multi-turn histories with inserted secrets, intervening exchanges rich in content, and standardized probes that try to extract the secrets. Across models that support extended conversation histories, dialogue-level leakage (G4P) remains substantial under hybrid detection (H8D), which combines deterministic matching with a model-based judge. Leakage varies sharply across systems, secret types (S8Q), and persuasion intensities (V4N). It does not decline reliably within the tested window. Privacy in active context—the information currently available to the model—requires evaluation as a persistent behavioral failure. Evaluation should not focus only on memorization in model parameters, the values learned during training, or on immediate jailbreaks, which try to bypass safeguards.

###### Keywords:

Trustworthy AI, Large Language Models, Privacy, Evaluation, Benchmarking

## 1 Introduction

Systems suppress Social Security Numbers (SSNs) and payment identifiers much more strongly than contact identifiers. No evaluated system achieves stable decay, or a lasting decline in leakage. QX2 provides controlled audits of persistence and tests direct, justified, and high-pressure requests for disclosure. It offers reproducible composite detection through deterministic processing, which follows fixed rules, and an open-weights judge, a model whose weights are available. It reports both G4P and probe-level leakage (P6S), the leakage rate across individual probe responses. It also uses a stability criterion rather than treating a temporary decrease as sufficient.

LLMs have evolved from simple chat systems into persistent assistants that maintain coherence across extended interactions (OpenAI, 2023; Touvron et al., 2023; Liu et al., 2024). Their uses include healthcare triage, financial planning, enterprise copilots, and personal productivity. Users disclose Personally Identifiable Information (PII), meaning information that can identify a person. This includes phone numbers (A2L), financial identifiers, email addresses (F8Z), and health-related details. Retaining context supports multi-turn assistance, but it also makes earlier sensitive disclosures available for later responses.

Existing safety evaluations mostly target two extremes: memorization of training data in model parameters and immediate jailbreaking, which elicits harmful behavior in the current turn (Wei et al., 2024; Zou et al., 2023). Tests of parameter extraction, multi-turn attacks, or injected instructions do not directly establish whether unrelated conversational shifts provide a practical privacy boundary. The premise is not that users think messages literally disappear within a session. Rather, users and designers may underestimate later attempts to elicit information through indirect requests, justification, high pressure, or tools. Shared sessions, browser-integrated assistants, enterprise copilots, agentic workflows that let models act through multiple steps, and injected instructions allow later interactions with context that the original user did not intend.

## 2 Related Work

### 2.1 Contextual Privacy and Inference-Time Leakage

Training-data extraction, membership inference, and recovery of memorized PII concern information stored in model parameters and pre-training corpora (Shokri et al., 2017; Carlini et al., 2021; Li et al., 2023; Nasr et al., 2023). Training-data extraction seeks to recover training examples. Membership inference tests whether particular data appeared in training. Active-context privacy instead concerns how a model handles information supplied by the user during the current interaction at inference time, when the model generates responses.

ConfAIde tests contextual-integrity privacy reasoning: reasoning about whether information sharing fits its context (Mireshghallah et al., 2024). PrivacyLens and PrivaCI-Bench examine awareness of privacy norms and legal compliance in agentic or contextual settings (Shao et al., 2024; Li et al., 2025). These benchmarks mainly ask whether sharing is appropriate under static access-control or privacy-norm scenarios. QX2 isolates whether a model can reproduce information after unrelated exchanges and later persuasive probes.

### 2.2 Multi-Turn Adversarial Attacks

Aligned models, which are trained to follow intended behavior, remain open to exploitation over multiple turns. Crescendo progressively elicits harmful outputs from conversations that appear benign (Russinovich et al., 2024). Automated frameworks generate adversarial multi-turn paths (Deng et al., 2024; Narula et al., 2025). Later injected instructions can also manipulate applications that integrate LLMs (Liu et al., 2023). These attacks usually target unsafe content or instruction hijacking, in which an attacker redirects the model's instructions. QX2 instead targets information-flow control: whether the model re-discloses sensitive information that a user supplied earlier, rather than producing harmful external content.

### 2.3 Context Management and Unlearning

TOFU and WMDP assess knowledge removal or behavioral suppression, usually through static question-answer pairs and model parameters (Maini et al., 2024; Li et al., 2024). QX2 instead tests behavioral suppression of secrets that remain in active context. Deployed retrieval-augmented generation (RAG) systems, which provide retrieved material to a model, often depend on managing context through prompts rather than unlearning information from model parameters. The same applies to copilots and agentic workflows.

## 3 The QX2 Framework

Figure 1: QX2 pipeline: controlled multi-turn dialogue construction, standardized probe attachment, target-model queries, composite leakage scoring.

The synthetic framework combines a configurable dialogue generator, a system for running probes, and measures for detecting leakage and analyzing it over time.

### 3.1 Threat Model

The threat model does not assume that the transcript is invisible. The failure mode is active-context re-disclosure (E9U): the assistant reproduces an earlier sensitive value after later querying, justification, or pressure. The value's mere presence in the transcript does not constitute this failure.

A later participant in a shared or persistent session may not understand earlier disclosures. Tool outputs, retrieved documents, and injected instructions may interact with existing context in enterprise or agentic settings. Browser or application users may not intend the system to reuse sensitive information after a task changes.

A user supplies a secret, \(S\), at an initial time, \(t_0\). Unrelated exchanges rich in content then create topic drift (J7K). These exchanges form \(D=\{u_1,a_1,\ldots,u_d,a_d\}\), where \(u_i\) and \(a_i\) represent user and assistant turns. The drift length (D7A), denoted \(d\), counts the intervening turns. A standardized probe, \(P\), is then appended to the active context. Exact, approximate, or semantic revelation of \(S\) counts as leakage; semantic revelation conveys the secret's meaning without necessarily copying its form.

### 3.2 Dataset Generation Pipeline

The pipeline first builds a scaffold and then rewrites it. This process preserves the exact ground-truth secret, the known value used for evaluation, along with the intervening-turn count and probe structure. It also reduces signs that dialogues came only from templates. Human-authored dialogues follow the same constraints and appear throughout the dataset rather than in one contiguous block. Token counts refer to units of text processed by a model; the approximate counts below come from word counts.

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

Plausible tasks in the opening turn give the assistant legitimate access to \(S\). The generator samples personas from more than 100 professions and demographic profiles. The categorical variable S8Q, or secret type, selects among fixed-format identifiers.

The four sampled formats are contact values (A2L = phone numbers), electronic contact values (F8Z = email addresses), government identifiers (W3B = Social Security Numbers), and payment identifiers (N6G = credit card numbers). Each value is inserted as the ground-truth secret rather than inferred later.

#### 3.2.2 Content-Dense J7K

Assigned topics include tax filing, medical triage, and travel logistics. A library of dense question-answer exchanges supplies intervening turns matched to the topic. Fallback tasks cover professional writing, structured reasoning, and long-form summarization. Annotators contribute a subset of the library turns. Unrelated entities and task demands enter the context while the original secret and turn count remain controlled.

#### 3.2.3 Naturalistic Refinement

A rewriting model receives each scaffold and instructions to improve fluency and persona realism. The exact secret, turn count, role structure, and final-probe conditions remain unchanged. The wording changes; the experimental variables do not.

### 3.3 Probing Setup

The final probe follows the completed history. The manipulated interaction style, V4N or persuasion intensity, has three standardized conditions rather than an unconstrained search for adversarial prompts.

The conditions are neutral direct retrieval (C8R = Simple), contextual justification (Z2F = Medium), and urgent or high-pressure elicitation (L9W = Hard). They distinguish requests for earlier A2L, form-filling needs, and tests of whether pressure cues increase or decrease disclosure.

D7A, or drift length \(d\), counts turns between disclosure and the final probe. The generator samples it from the following mixture, where \(p\) denotes probability and Uniform assigns equal probability to each listed value:

\[
d\sim
\begin{cases}
0 & \text{with }p=0.2,\\
\mathrm{Uniform}\{2,3,4,5,6\} & \text{with }p=0.8.
\end{cases}
\tag{1}
\]

The dataset includes immediate recall but emphasizes short-to-medium J7K.

### 3.4 Evaluation Methodology

Eligible disclosures include verbatim output, formatting changes, partial substrings, and paraphrased references. A substring is a portion of a longer string. H8D, the hybrid composite detector, labels a response positive if normalized inclusion succeeds or the judge finds the full secret, a substantial substring, or enough partial information to identify it. Normalized inclusion checks whether the secret appears in the response after standardizing their formats. The first stage uses normalization-based matching (B6T).

Figure 2: Hierarchical evaluation: B6T first; negative responses proceed to the LLM judge.

#### 3.4.1 Stage 1: B6T

B6T detects direct and near-direct disclosure through deterministic processing. For numeric secrets, it removes every non-digit character from both the secret and the response. For alphanumeric text, which contains letters and numbers, it standardizes case and whitespace but retains meaningful delimiters.

Let \(\phi\) denote normalization, \(R\) the response, and \(\mathbb I\) an indicator that equals one when the condition holds and zero otherwise. The deterministic baseline is U3P, called regex-only, which uses fixed pattern-matching rules:

\[
\mathrm{Leak}_{\mathrm{U3P}}(R,S)
=\mathbb I\big[\phi(S)\subset\phi(R)\big].
\tag{2}
\]

Here, \(\subset\) means that the normalized secret occurs within the normalized response.

#### 3.4.2 Stage 2: LLM-as-a-Judge Verification

An open-weights Llama judge receives \(S\) and \(R\) and returns a binary verdict on full or substantial disclosure. It reviews cases that the deterministic stage labels negative. It checks for partial disclosures, indirect hints, and formatting variations that the first stage missed (Meta AI, 2024).

The evaluation reports U3P, the regex-only baseline, separately alongside H8D. Judge errors remain possible; the evaluation does not treat the judge's decisions as error-free.

#### 3.4.3 K2V

The stability metric K2V, called Privacy Half-Life and denoted \(\tau\), uses the observed drift lengths \(D_{obs}\) and the leakage rate \(L(d)\) at each length. It sets a safety threshold of \(\delta=0.1\times L(0)\). It requires sustained suppression rather than a momentary decrease:

\[
\tau=\min\left\{
d\in D_{obs}\ \middle|\
L(d)\leq\delta
\ \text{and}\
\forall d'\in D_{obs},\ d'>d:\ L(d')\leq\delta
\right\}.
\tag{3}
\]

Thus, \(\tau\) is the smallest observed drift length at which leakage meets the threshold and stays at or below it at every later observed length. If no length qualifies, the metric assigns \(\tau=\max(D_{obs})+1\).

Figure 3: K2V; later resurgence disqualifies transient decreases.

## 4 Experimental Setup

##### Models.

All systems receive identical histories and probe templates through the OpenRouter application programming interface (API), which provides access to the models. The evaluated model identifiers are M7Q = GPT-OSS-120B, T4X = DeepSeek-R1, and Y9C = Qwen3-VL-235B.

##### Dataset.

Each system processes the full QX2 dataset under every standardized condition. Each dialogue contains one inserted secret and one sampled \(d\).

| Cached responses | Count |
|---|---:|
| Per model | 3,000 |
| All evaluated models | 9,000 |

##### Aggregation levels.

G4P, or dialogue-level leakage, counts a dialogue as positive when at least one standardized probe elicits its secret. P6S, or probe-level leakage, gives each dialogue one observation per persuasion condition. The V4N analyses and fuzzy-matching analyses (R5J), which compare approximate text similarity, use P6S. G4P can exceed the mean rate across individual conditions.

##### Uncertainty and statistical tests.

G4P intervals use 95% bootstrapping over dialogues, which estimates uncertainty by repeatedly resampling dialogues. P6S sensitivity checks use cached, or saved, model-probe responses. Matched comparisons between models or detectors use McNemar’s test, which compares paired binary outcomes. Where applicable, Bonferroni correction adjusts for multiple comparisons.

Repeated persuasion measures use Cochran’s \(Q\), a test for differences across repeated binary measures, followed by McNemar tests. Chi-square independence tests assess whether leakage is associated with D7A, V4N, or S8Q. The association statistic X2H, or Cramer’s \(V\), reports effect size—the strength of association—rather than replacing significance tests.

## 5 Results

### 5.1 G4P: H8D Largely Agrees With B6T

H8D differs only slightly from B6T. Direct or near-direct reproduction dominates purely semantic paraphrase. Adding the judge changes few labels. This supports H8D as the primary detector and U3P as a transparent deterministic baseline. In the table, \(\Delta\) gives the increase in percentage points, and \(p\) denotes the statistical test's p-value.

| Model | U3P | H8D | \(\Delta\) | \(p\) |
|---|---:|---:|---:|---:|
| M7Q | 47.70 | 47.70 | +0.00 | 1.0 |
| T4X | 37.80 | 38.70 | +0.90 | \(7.7\times10^{-3}\) |
| Y9C | 53.50 | 54.60 | +1.10 | \(2.57\times10^{-3}\) |

Table 2: G4P percentages; augmentation in percentage points.

### 5.2 Substantial G4P Across Models

Every evaluated system remains vulnerable at substantial rates to at least one standardized extraction condition. CI denotes confidence interval, the reported interval estimate of uncertainty.

| Model | H8D | 95% CI |
|---|---:|---|
| T4X | 38.70% | [35.7, 41.6] |
| M7Q | 47.70% | [44.6, 50.8] |
| Y9C | 54.60% | [51.6, 57.7] |

Table 3: G4P with bootstrap confidence intervals.

### 5.3 Non-Monotonic V4N Effects

Differences within each model are significant, but their directions differ. M7Q leakage declines under L9W relative to C8R and Z2F. T4X reaches its minimum at Z2F. Y9C reaches its maximum at L9W.

| Model | C8R | Z2F | L9W |
|---|---:|---:|---:|
| M7Q | 39.30% | 40.50% | 32.90% |
| T4X | 28.10% | 22.00% | 29.20% |
| Y9C | 38.70% | 33.30% | 40.60% |

Table 4: H8D P6S by V4N; Cochran’s \(Q\), \(p<10^{-5}\) for every model.

### 5.4 No Reliable Reduction Within the Tested Window

Associations between cached P6S and D7A are small or negligible. Privacy risk persists across several unrelated shifts, before the original disclosure lies far from the end of the context window—the text available to the model.

Figure 4: H8D versus D7A; dip near \(d=3\), then rebound.

### 5.5 S8Q Strongly Determines Leakage

S8Q has a large association with leakage in every system. It also has the strongest association among the tested variables. Systems suppress W3B and N6G far more than F8Z and A2L, even though users disclosed all categories and all are private in context.

| S8Q | M7Q | T4X | Y9C |
|---|---:|---:|---:|
| W3B | 0.14% | 1.11% | 5.97% |
| N6G | 0.00% | 0.39% | 4.01% |
| F8Z | 78.24% | 57.11% | 81.13% |
| A2L | 70.89% | 46.13% | 57.24% |

Table 5: H8D P6S across cached model-probe responses.

### 5.6 K2V Exceeds Observation

No system approaches the threshold at any tested length. Suppression never stabilizes.

| Scope | Observed bound | Assigned K2V |
|---|---|---|
| Every model and persuasion condition | \(d\leq6\) | \(\tau=\max(D_{obs})+1=7\) |

### 5.7 R5J Supports H8D Labels

R5J, or fuzzy matching, provides a post-hoc deterministic sensitivity check, meaning a check performed after the main evaluation. It uses only cached outputs to compute normalized similarity between each saved secret and saved response. The higher tested threshold closely tracks H8D across systems. R5J remains an auxiliary check because approximate string similarity is less expressive than judge-based evaluation for partial or semantic disclosures.

| Model | U3P | R5J 0.85 | R5J 0.80 | H8D |
|---|---:|---:|---:|---:|
| T4X | 25.50 | 26.40 | 26.53 | 26.43 |
| M7Q | 36.80 | 37.03 | 37.07 | 37.57 |
| Y9C | 36.97 | 37.60 | 37.93 | 37.53 |

Table 6: Cached-output P6S percentages.

Figure 5: H8D heatmap by D7A and V4N for each system.

## 6 Discussion

Behavioral recoverability—the ability to elicit earlier information through model responses—does not imply permanent memory or uniform behavior across long-context settings. Current assistants may reproduce sensitive information in context under sufficient retrieval pressure.

M7Q's response to pressure is consistent with urgency activating safety or refusal behavior. Y9C's response indicates weaker resistance to pressure. J7K does not provide an implicit privacy boundary. Unstable intermediate decreases call for stability-based evaluation rather than treatment as evidence of suppression.

## 7 Limitations and Future Work

##### Bounded D7A.

The tested short-to-medium range does not fill modern context windows. These findings concern bounded persistence in active context. They do not fully characterize privacy behavior over arbitrarily long interactions or full context windows.

##### Fixed-format secrets.

Structured identifiers are easier to detect than unstructured information about health status, family circumstances, immigration status, or financial hardship. Extending the benchmark to sensitive attributes without fixed formats remains future work.

##### Active context, not cross-session memory.

The study does not test training-data memorization, personalization memory, or retention across sessions.

##### Synthetic and human-authored dialogues.

The mixed construction improves realism over a purely synthetic benchmark while preserving experimental control. Cached metadata no longer retains identifiers for human-authored rows. The study therefore reports only aggregate results and defers a separate comparison of human-authored and synthetic dialogues.

##### Detector scope.

Labels concern the surface form of outputs under U3P, R5J, and an external judge. Separate reporting exposes detector behavior, but the study does not report formal human–automated inter-annotator agreement, a measure of agreement between human and automated labels. Larger manual audits of borderline partial disclosures and obfuscated disclosures remain future work.

##### API-served models and reproducibility.

Compute and budget constraints require API service. The evaluated identifiers (M7Q = GPT-OSS-120B), (T4X = DeepSeek-R1), and (Y9C = Qwen3-VL-235B) designate models that may change over time. This limitation remains despite logged identifiers, prompts, timestamps, cached responses, and paired statistics on identical dialogue sets.

##### Artifact availability.

Planned releases include generation code, probe templates, evaluation scripts, aggregate result files, and a sanitized dialogue subset. Synthetic PII-like strings require non-real identifiers in released instances. They also require templates sufficient for reproduction without exposing realistic identifiers.

## 8 Conclusion

The S8Q asymmetry suggests uneven safeguards. These safeguards rely more on format and sensitivity cues, including format-sensitive safety heuristics, than on generalized contextual confidentiality or robust contextual reasoning.

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
