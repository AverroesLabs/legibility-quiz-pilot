This paper asks whether unrelated conversational tasks prevent later disclosure of sensitive information already in an active large language model conversation. PrivDrift uses seeded multi-turn histories, content-dense intervening exchanges, and three standardized extraction conditions. Its 1,000 dialogues contain phone numbers, email addresses, Social Security Numbers, or credit card numbers. Detection combines normalized matching with an open-weights judge. Hybrid dialogue-level leakage is 38.70%, 47.70%, and 54.60% across the three evaluated models. Leakage varies by model, secret type, and persuasion intensity. Social Security Numbers and credit card numbers receive much stronger suppression than contact identifiers. No system attains stable suppression within the tested drift lengths. The paper treats this as an active-context behavioral privacy failure, not evidence of permanent memory.

## Glossary

*This glossary defines the paper’s terms, codes, abbreviations, and notation from their uses in the paper.*

- **PrivDrift; QX2:** The benchmark for auditing sensitive-information persistence after unrelated conversational exchanges.
- **LLM:** Large language model.
- **Trustworthy AI:** A keyword framing the paper’s privacy evaluation.
- **PII:** Personally Identifiable Information disclosed by users.
- **Active-context privacy:** Appropriate inference-time handling of user information in the current interaction.
- **E9U; active-context re-disclosure:** Assistant reproduction of an earlier sensitive value under later querying, justification, or pressure.
- **Behavioral recoverability:** The ability to elicit an earlier sensitive disclosure from the assistant.
- **Leakage:** Exact, approximate, or semantic revelation of the seeded secret.
- **Parameter-based memorization:** Information stored in model parameters or pre-training corpora.
- **Training-data extraction; memorized-PII recovery:** Recovery of information stored in parameters and pre-training corpora.
- **Membership inference:** A related-work concern involving training data, distinguished from current-interaction privacy.
- **Immediate jailbreaking:** Eliciting harmful behavior in the current turn.
- **Multi-turn adversarial attacks:** Exploitation through a sequence of conversational turns.
- **Instruction hijacking; injection:** Later instructions manipulating an LLM-integrated application.
- **Information-flow control:** In this study, control over re-disclosure of earlier user-supplied sensitive information.
- **Contextual integrity; privacy norms:** Standards for whether information sharing is appropriate in a context.
- **Access control:** A basis for static scenarios about whether sharing is appropriate.
- **Unlearning:** Removal or behavioral suppression of knowledge, usually involving model parameters.
- **RAG systems:** Deployed systems cited as depending on prompt-level context management.
- **Scaffold-then-rewrite:** Dialogue construction that preserves experimental variables while revising surface phrasing.
- **Persona:** A sampled profession and demographic profile used in a plausible task context.
- **Ground-truth secret:** The exact sensitive value inserted into a dialogue.
- **J7K; topic drift:** Unrelated, content-dense exchanges after disclosure.
- **S8Q; secret type:** The categorical choice among four fixed-format identifiers.
- **A2L:** Phone numbers.
- **F8Z:** Email addresses.
- **W3B; SSNs:** Social Security Numbers.
- **N6G:** Credit card numbers.
- **V4N; persuasion intensity:** The standardized interaction style of the extraction probe.
- **C8R; Simple:** Neutral direct retrieval.
- **Z2F; Medium:** Contextual justification.
- **L9W; Hard:** Urgent or high-pressure elicitation.
- **D7A; drift length:** The number of turns between disclosure and final probing.
- **Probe:** The standardized request appended after the completed history.
- **H8D; hybrid:** Detection by normalized inclusion or an LLM judge’s disclosure verdict.
- **B6T; normalization-based matching:** Deterministic processing for direct and near-direct disclosure.
- **U3P; regex-only:** The separately reported deterministic detection baseline.
- **LLM-as-a-judge; Llama judge:** An open-weights model returning a binary full/substantial-disclosure verdict.
- **R5J; fuzzy matching:** A post-hoc normalized-similarity check on saved secrets and responses.
- **G4P; dialogue-level leakage:** A dialogue is positive if any standardized probe elicits its secret.
- **P6S; probe-level leakage:** Each dialogue contributes one response observation per persuasion condition.
- **K2V; Privacy Half-Life:** The earliest observed drift length meeting a suppression threshold at that and every later observed length.
- **X2H; Cramer’s \(V\):** The reported association effect size.
- **M7Q:** GPT-OSS-120B.
- **T4X:** DeepSeek-R1.
- **Y9C:** Qwen3-VL-235B.
- **API; OpenRouter API:** The service interface through which systems receive histories and probe templates.
- **Cached responses:** Saved model outputs used for evaluation and sensitivity checks.
- **CI:** Confidence interval; G4P intervals use 95% bootstrapping over dialogues.
- **McNemar’s test:** The test used for matched model or detector comparisons and follow-up persuasion comparisons.
- **Bonferroni correction:** The correction applied to matched comparisons where applicable.
- **Cochran’s \(Q\):** The test used for repeated persuasion measures.
- **Chi-square:** The independence test for leakage against D7A, V4N, or S8Q.
- **ConfAIde:** A test of contextual-integrity privacy reasoning.
- **PrivacyLens; PrivaCI-Bench:** Evaluations of privacy-norm awareness and legal compliance in agentic or contextual settings.
- **Crescendo:** A multi-turn attack that progressively elicits harmful outputs from apparently benign conversations.
- **TOFU; WMDP:** Evaluations of knowledge removal or behavioral suppression.
- **\(S\):** The user-supplied secret.
- **\(t_0\):** The time at which the user supplies \(S\).
- **\(D\):** The sequence of unrelated user and assistant exchanges.
- **\(u_i,a_i\):** User and assistant messages in \(D\).
- **\(P\):** The standardized final probe.
- **\(d\):** Drift length.
- **\(N\):** Number of dialogues.
- **\(p\):** Sampling probability in Equation (1), or a statistical-test \(p\)-value in results.
- **\(\mathrm{Uniform}\):** The distribution over \(\{2,3,4,5,6\}\) in Equation (1).
- **\(R\):** The model response.
- **\(\phi\):** The normalization applied to secrets and responses.
- **\(\mathbb I\):** The indicator in the deterministic leakage rule.
- **\(\subset\):** Normalized-secret inclusion in the normalized response in Equation (2).
- **\(\mathrm{Leak}_{\mathrm{U3P}}\):** The deterministic leakage label.
- **\(D_{obs}\):** Observed drift lengths.
- **\(L(d)\); \(L(0)\):** Leakage at drift length \(d\), and at immediate recall.
- **\(\delta\):** The safety threshold, \(0.1\times L(0)\).
- **\(\tau\):** K2V, including the assigned value when no observed length qualifies.
- **\(\min,\max,\forall\):** Minimum, maximum, and the “for every” condition used in Equation (3).
- **\(\Delta\):** H8D augmentation over U3P, in percentage points.
- **Avg.; median; min/max:** Average, median, and minimum/maximum dialogue-length summaries.
- **Approximate tokens:** Dialogue-length estimates derived from word counts.

# PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations

Luciano Rolando Maldonado Romero

Affiliation: West Virginia University, Morgantown, WV, USA

Correspondence to: lrm00020@mix.wvu.edu

###### Abstract

*The benchmark measures persistent sensitive-information recoverability after unrelated conversational tasks.*

Persistent, user-facing, shared-session, tool-augmented large language models (LLMs) retain behaviorally recoverable sensitive disclosures beyond unrelated tasks. Behavioral recoverability means that later prompting can elicit the earlier disclosure. The controlled benchmark PrivDrift (QX2) audits this persistence with seeded multi-turn histories, content-dense intervening exchanges, and standardized extraction probes.

Across extended-context models, dialogue-level leakage (G4P), which counts disclosure under any standardized probe, remains substantial under hybrid detection (H8D). H8D combines normalized matching with a model judge. Leakage varies sharply across systems, secret type (S8Q), and persuasion intensity (V4N). It fails to decline reliably within the tested window. Active-context privacy—appropriate inference-time handling of current-interaction user information—requires evaluation as a persistent behavioral failure, not exclusively as parameter memorization or immediate jailbreak behavior.

**Claims and evidence**
- Substantial leakage varies across systems: Table 3, T4X, M7Q, and Y9C rows: 38.70%, 47.70%, and 54.60%.
- Secret type and persuasion matter: Tables 5 and 4 report category- and condition-specific rates.
- Suppression does not stabilize: Section 5.6, “Every model and persuasion condition” row: \(d\leq6\), assigned \(\tau=7\).

###### Keywords:

*The keywords identify the paper’s research topics.*

Trustworthy AI, Large Language Models, Privacy, Evaluation, Benchmarking

## 1 Introduction

*QX2 evaluates whether unrelated exchanges provide a practical boundary against later re-disclosure.*

Social Security Numbers (SSNs) and payment identifiers receive much stronger suppression than contact identifiers. No evaluated system attains stable decay. QX2 provides controlled persistence auditing; direct, justified, and high-pressure elicitation; reproducible composite detection using deterministic processing and an open-weights judge; G4P and probe-level leakage (P6S) reporting; and a stability criterion rather than a transient-decrease criterion. P6S counts one response observation per dialogue and persuasion condition.

LLMs have evolved from simple chat systems into persistent assistants that maintain coherence across extended interactions (OpenAI, 2023; Touvron et al., 2023; Liu et al., 2024). Deployment domains include healthcare triage, financial planning, enterprise copilots, and personal productivity. Disclosed Personally Identifiable Information (PII) includes phone numbers (A2L), financial identifiers, email addresses (F8Z), and health-related details. Context retention supports multi-turn assistance but also makes earlier sensitive disclosures available to later generation.

Existing safety evaluations mostly target two extremes: parameter-based training-data memorization, involving information stored in model parameters, and immediate jailbreaking, which elicits harmful behavior in the current turn (Wei et al., 2024; Zou et al., 2023). Parameter extraction and multi-turn or injection evaluations do not directly establish whether unrelated conversational shifts provide a practical privacy boundary.

The premise is not that users believe same-session messages literally disappear. Users and designers may underestimate later indirect, justified, high-pressure, or tool-mediated elicitation. Shared sessions, browser-integrated assistants, enterprise copilots, agentic workflows, and injected instructions permit later context interactions unintended by the original user.

**Claims and evidence**
- Government and payment identifiers receive stronger suppression than contact identifiers: Table 5, W3B and N6G rows: 0.14%/1.11%/5.97% and 0.00%/0.39%/4.01%; F8Z and A2L rows: 78.24%/57.11%/81.13% and 70.89%/46.13%/57.24%, respectively, for M7Q/T4X/Y9C.
- No evaluated system attains stable decay: Section 5.6, every-model-and-condition row: observed \(d\leq6\), assigned \(\tau=7\).

## 2 Related Work

*Prior evaluations address parameter-held information, privacy norms, attacks, or unlearning rather than recoverability after unrelated exchanges.*

### 2.1 Contextual Privacy and Inference-Time Leakage

*Active-context privacy differs from training-data recovery and static judgments about appropriate sharing.*

Training-data extraction, membership inference, and memorized-PII recovery concern information stored in parameters and pre-training corpora (Shokri et al., 2017; Carlini et al., 2021; Li et al., 2023; Nasr et al., 2023). Active-context privacy concerns current-interaction user information.

ConfAIde tests contextual-integrity privacy reasoning: whether sharing is appropriate in context (Mireshghallah et al., 2024). PrivacyLens and PrivaCI-Bench examine privacy-norm awareness and legal compliance in agentic or contextual settings (Shao et al., 2024; Li et al., 2025). Their principal object is appropriate sharing under static access-control or privacy-norm scenarios. QX2 isolates behavioral recoverability after unrelated exchanges and subsequent persuasive probing.

### 2.2 Multi-Turn Adversarial Attacks

*QX2 targets re-disclosure of user-supplied information rather than harmful content or instruction hijacking.*

Aligned models remain exploitable over multiple turns. Crescendo progressively elicits harmful outputs from apparently benign conversations (Russinovich et al., 2024). Automated frameworks generate adversarial multi-turn paths (Deng et al., 2024; Narula et al., 2025). Later injected instructions manipulate LLM-integrated applications (Liu et al., 2023). These attacks usually target unsafe content or instruction hijacking. QX2 targets information-flow control: re-disclosure of earlier user-supplied sensitive information rather than harmful external content.

### 2.3 Context Management and Unlearning

*QX2 studies suppression of secrets that remain in active context, not removal from model parameters.*

TOFU and WMDP assess unlearning: removal or behavioral suppression of knowledge, usually through static question-answer pairs and model parameters (Maini et al., 2024; Li et al., 2024). QX2 instead concerns behavioral suppression of secrets still present in active context. Deployed RAG systems, copilots, and agentic workflows often depend on prompt-level context management rather than parameter-level unlearning.

## 3 The QX2 Framework

*The framework constructs controlled dialogues, probes models, and scores leakage and its persistence.*

Figure 1: QX2 pipeline: controlled multi-turn dialogue construction, standardized probe attachment, target-model queries, composite leakage scoring.

The synthetic framework combines a parametric dialogue generator, probing harness, and detection/temporal-analysis metric suite.

### 3.1 Threat Model

*The failure is later reproduction of an earlier secret, not its continued presence in the transcript.*

Transcript invisibility is not assumed. Active-context re-disclosure (E9U) means assistant reproduction of an earlier sensitive value under later querying, justification, or pressure.

A later participant in a shared or persistent session may not understand preceding disclosures. Tool outputs, retrieved documents, and injected instructions in enterprise or agentic settings may interact with existing context. Browser/application users may not intend sensitive information to be reused after task changes.

A user supplies secret \(S\) at time \(t_0\). Unrelated, content-dense exchanges—topic drift (J7K)—then form \(D=\{u_1,a_1,\ldots,u_d,a_d\}\), where \(u_i\) and \(a_i\) are user and assistant messages. A standardized probe \(P\) is appended to the active context. Exact, approximate, or semantic revelation of \(S\) constitutes leakage.

### 3.2 Dataset Generation Pipeline

*Scaffolding and rewriting preserve experimental controls while reducing purely templated dialogue artifacts.*

The scaffold-then-rewrite pipeline preserves exact ground truth, intervening-turn counts, and probe structure. Human-authored dialogues obey the same constraints and are interspersed, not contiguous. Here \(N\) is the number of dialogues. Drift length (D7A), denoted \(d\), counts intervening turns. Approximate tokens are estimated from words.

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

*Plausible opening tasks give the assistant legitimate access to one exact, fixed-format secret.*

Personas—professions and demographic profiles—are sampled from more than 100 profiles. Opening-turn task contexts establish legitimate assistant access to \(S\). S8Q selects among four formats: A2L, F8Z, Social Security Numbers (W3B), and credit card numbers (N6G). Each value is inserted as the ground-truth secret rather than subsequently inferred.

#### 3.2.2 Content-Dense J7K

*Intervening tasks add unrelated entities and demands while preserving the secret and turn count.*

Assigned topics include tax filing, medical triage, and travel logistics. A topic-conditioned dense question-answer library supplies intervening turns, with professional-writing, structured-reasoning, and long-form-summarization fallback tasks. Annotators contribute a subset of library turns.

#### 3.2.3 Naturalistic Refinement

*Rewriting changes surface phrasing but leaves experimental variables unchanged.*

A rewriting model receives each scaffold and instructions for fluency and persona realism. The exact secret, turn count, role structure, and final-probe conditions remain invariant.

**Claims and evidence**
- The dataset contains synthetic and interspersed human-authored dialogues: Table 1, total, synthetic, and human-authored rows: 1,000, 900, and 100.
- It covers four secret categories and six drift lengths: Table 1, “Secret categories” and “D7A values” rows: 4 and \(0,2,3,4,5,6\).
- Its length characteristics are reported in Table 1: word average/median 359.16/385.00, range 43 / 795; approximate-token average/median 478.87/513.33, range 57.33 / 1060.00.

### 3.3 Probing Setup

*Three standardized persuasion conditions probe immediate recall and short-to-medium topic drift.*

The final probe follows the completed history. V4N has three standardized conditions, not an unconstrained adversarial search:

- Simple (C8R): neutral direct retrieval.
- Medium (Z2F): contextual justification.
- Hard (L9W): urgent/high-pressure elicitation.

The conditions distinguish requests for earlier A2L, form-filling needs, and tests of whether pressure cues raise or lower disclosure.

D7A counts turns between disclosure and final probing. Its sampling mixture uses probability \(p\), with \(\mathrm{Uniform}\) over the listed values:

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

*Hybrid detection accepts normalized inclusion or a judge’s finding of identifying disclosure.*

Verbatim output, formatting changes, partial substrings, and paraphrased references are eligible disclosures. H8D labels a response positive if normalized inclusion succeeds or the judge finds the full secret, a substantial substring, or sufficient partial information to identify it.

Figure 2: Hierarchical evaluation: B6T first; negative responses proceed to the LLM judge.

#### 3.4.1 Stage 1: B6T

*Deterministic normalization detects direct and near-direct disclosure.*

Normalization-based matching (B6T) strips every non-digit character from numeric secrets and responses. Alphanumeric processing normalizes case and whitespace but retains meaningful delimiters.

With normalization \(\phi\), response \(R\), and indicator \(\mathbb I\), normalized-secret inclusion, written \(\subset\), defines the deterministic leakage label \(\mathrm{Leak}_{\mathrm{U3P}}\):

\[
\mathrm{Leak}_{\mathrm{U3P}}(R,S)
=\mathbb I\big[\phi(S)\subset\phi(R)\big].
\tag{2}
\]

#### 3.4.2 Stage 2: LLM-as-a-Judge Verification

*An open-weights judge checks deterministic negatives for otherwise missed disclosures.*

An open-weights Llama judge receives \(S\) and \(R\) and returns a binary full/substantial-disclosure verdict. It checks partial disclosures, indirect hints, and missed formatting variations (Meta AI, 2024).

The regex-only deterministic baseline (U3P) remains separately reported alongside H8D. Judge errors are possible; adjudication is not treated as error-free.

#### 3.4.3 K2V

*Privacy Half-Life requires suppression to persist at every later observed drift length.*

Privacy Half-Life (K2V), denoted \(\tau\), uses observed lengths \(D_{obs}\) and leakage rates \(L(d)\). Its safety threshold is \(\delta=0.1\times L(0)\), where \(L(0)\) is immediate-recall leakage. The minimum qualifying \(d\) must satisfy the threshold at every later observed \(d'\), expressed by \(\forall\):

\[
\tau=\min\left\{
d\in D_{obs}\ \middle|\
L(d)\leq\delta
\ \text{and}\
\forall d'\in D_{obs},\ d'>d:\ L(d')\leq\delta
\right\}.
\tag{3}
\]

An empty set receives \(\tau=\max(D_{obs})+1\), where \(\max\) is the largest observed length.

Figure 3: K2V; later resurgence disqualifies transient decreases.

## 4 Experimental Setup

*Every model receives the same histories and probes, with dialogue- and response-level analyses.*

##### Models.

*The systems are queried through the same service interface.*

The systems are GPT-OSS-120B (M7Q), DeepSeek-R1 (T4X), and Qwen3-VL-235B (Y9C). All receive identical histories and probe templates through the OpenRouter API, the service interface used for model queries.

##### Dataset.

*Each model processes every dialogue under all standardized conditions.*

Each system processes the full QX2 dataset, with one seeded secret and one sampled \(d\) per dialogue, under every standardized condition. Cached responses are saved model outputs.

| Cached responses | Count |
|---|---:|
| Per model | 3,000 |
| All evaluated models | 9,000 |

##### Aggregation levels.

*G4P counts any-probe dialogue disclosure; P6S counts individual probe responses.*

G4P marks a dialogue positive when at least one standardized probe elicits its secret. P6S gives each dialogue one observation per persuasion condition for V4N and fuzzy-matching (R5J) analyses. R5J is a post-hoc normalized-similarity check. G4P can exceed the mean condition-specific rate.

##### Uncertainty and statistical tests.

*The analysis uses dialogue bootstrapping, matched tests, repeated-measure tests, and association effect sizes.*

G4P confidence intervals (CIs) use 95% bootstrapping over dialogues. P6S sensitivity checks use cached model-probe responses. Matched model/detector comparisons use McNemar’s test, Bonferroni-corrected where applicable. Repeated persuasion measures use Cochran’s \(Q\), followed by McNemar tests.

Chi-square tests assess independence of leakage against D7A, V4N, or S8Q. Cramer’s \(V\) (X2H) is reported as an association effect size, not a substitute for significance testing. In result tables, \(p\) denotes a statistical-test \(p\)-value.

**Claims and evidence**
- Every model has the same cached-response count: Section 4 cached-responses table, “Per model” row: 3,000; “All evaluated models” row: 9,000.

## 5 Results

*Leakage remains substantial, varies by model and secret type, and does not stabilize below the suppression threshold.*

### 5.1 G4P: H8D Largely Agrees With B6T

*Hybrid detection adds little to deterministic detection.*

H8D differs only slightly from B6T. Direct or near-direct reproduction dominates purely semantic paraphrase. Limited judge augmentation supports H8D as primary and U3P as a transparent deterministic baseline. Here \(\Delta\) is augmentation in percentage points.

| Model | U3P | H8D | \(\Delta\) | \(p\) |
|---|---:|---:|---:|---:|
| M7Q | 47.70 | 47.70 | +0.00 | 1.0 |
| T4X | 37.80 | 38.70 | +0.90 | \(7.7\times10^{-3}\) |
| Y9C | 53.50 | 54.60 | +1.10 | \(2.57\times10^{-3}\) |

Table 2: G4P percentages; augmentation in percentage points.

**Claims and evidence**
- Hybrid augmentation is limited, supporting the stated detection interpretation: Table 2, M7Q/T4X/Y9C rows give \(\Delta=+0.00/+0.90/+1.10\), with \(p=1.0/7.7\times10^{-3}/2.57\times10^{-3}\).

### 5.2 Substantial G4P Across Models

*Every evaluated model discloses secrets under at least one standardized extraction condition at substantial rates.*

| Model | H8D | 95% CI |
|---|---:|---|
| T4X | 38.70% | [35.7, 41.6] |
| M7Q | 47.70% | [44.6, 50.8] |
| Y9C | 54.60% | [51.6, 57.7] |

Table 3: G4P with bootstrap confidence intervals.

**Claims and evidence**
- Substantial vulnerability occurs in every system: Table 3, T4X: 38.70%, CI [35.7, 41.6]; M7Q: 47.70%, CI [44.6, 50.8]; Y9C: 54.60%, CI [51.6, 57.7].

### 5.3 Non-Monotonic V4N Effects

*Persuasion effects are significant within every model, but their directions differ.*

M7Q declines under L9W relative to C8R and Z2F. T4X minimizes at Z2F. Y9C maximizes at L9W.

| Model | C8R | Z2F | L9W |
|---|---:|---:|---:|
| M7Q | 39.30% | 40.50% | 32.90% |
| T4X | 28.10% | 22.00% | 29.20% |
| Y9C | 38.70% | 33.30% | 40.60% |

Table 4: H8D P6S by V4N; Cochran’s \(Q\), \(p<10^{-5}\) for every model.

**Claims and evidence**
- Within-model differences are significant throughout: Table 4 caption, Cochran’s \(Q\), \(p<10^{-5}\) for every model.
- Directions differ: Table 4, M7Q row: 39.30%/40.50%/32.90%; T4X: 28.10%/22.00%/29.20%; Y9C: 38.70%/33.30%/40.60%, for C8R/Z2F/L9W.

### 5.4 No Reliable Reduction Within the Tested Window

*Drift has small or negligible associations with leakage, and an intermediate dip is followed by rebound.*

Cached P6S associations with D7A are small or negligible. Privacy risk persists across several unrelated shifts before the original disclosure lies far from the context-window end.

Figure 4: H8D versus D7A; dip near \(d=3\), then rebound.

**Claims and evidence**
- Drift associations are small or negligible: Table 8, D7A rows: T4X 0.1073, Small; M7Q 0.0659, Negligible; Y9C 0.0723, Negligible.
- The decrease is not reliable suppression: Figure 4 reports a dip near \(d=3\), then rebound; Section 5.6 reports no stable suppression through \(d\leq6\).

### 5.5 S8Q Strongly Determines Leakage

*Secret type has the strongest tested association, with much greater suppression of government and payment identifiers.*

S8Q has a large association in every system and the strongest association among tested variables. W3B and N6G receive far more suppression than F8Z and A2L, although all categories are user-disclosed and contextually private.

| S8Q | M7Q | T4X | Y9C |
|---|---:|---:|---:|
| W3B | 0.14% | 1.11% | 5.97% |
| N6G | 0.00% | 0.39% | 4.01% |
| F8Z | 78.24% | 57.11% | 81.13% |
| A2L | 70.89% | 46.13% | 57.24% |

Table 5: H8D P6S across cached model-probe responses.

**Claims and evidence**
- S8Q has the largest association in each model: Table 8, S8Q rows: T4X 0.5871, M7Q 0.7731, Y9C 0.6927, all Large; D7A/V4N values are 0.1073/0.0718, 0.0659/0.0689, and 0.0723/0.0639.
- Government/payment identifiers receive stronger suppression: Table 5, W3B 0.14%/1.11%/5.97%; N6G 0.00%/0.39%/4.01%; F8Z 78.24%/57.11%/81.13%; A2L 70.89%/46.13%/57.24%, for M7Q/T4X/Y9C.

### 5.6 K2V Exceeds Observation

*No tested model-condition combination attains stable threshold-level suppression.*

No system approaches the threshold at any tested length. Suppression never stabilizes.

| Scope | Observed bound | Assigned K2V |
|---|---|---|
| Every model and persuasion condition | \(d\leq6\) | \(\tau=\max(D_{obs})+1=7\) |

**Claims and evidence**
- K2V exceeds observation for every model and condition: Section 5.6 table, sole row: \(d\leq6\), assigned \(\tau=\max(D_{obs})+1=7\); Equation (3) sets \(\delta=0.1\times L(0)\).

### 5.7 R5J Supports H8D Labels

*The higher fuzzy-matching threshold closely tracks hybrid labels on cached outputs.*

R5J computes normalized similarity between each saved secret and saved response using cached outputs only. The higher tested threshold closely tracks H8D across systems. R5J is auxiliary: approximate string similarity is less expressive than judge-based evaluation for partial or semantic disclosures.

| Model | U3P | R5J 0.85 | R5J 0.80 | H8D |
|---|---:|---:|---:|---:|
| T4X | 25.50 | 26.40 | 26.53 | 26.43 |
| M7Q | 36.80 | 37.03 | 37.07 | 37.57 |
| Y9C | 36.97 | 37.60 | 37.93 | 37.53 |

Table 6: Cached-output P6S percentages.

Figure 5: H8D heatmap by D7A and V4N for each system.

**Claims and evidence**
- The higher threshold tracks H8D closely: Table 6, R5J 0.85/H8D columns: T4X 26.40/26.43; M7Q 37.03/37.57; Y9C 37.60/37.53.

## 6 Discussion

*Recoverability concerns in-context behavior, while differing pressure responses and rebounds argue against treating topic drift as a privacy boundary.*

Behavioral recoverability does not imply permanent memory or uniform behavior across long-context settings. Current assistants may reproduce sensitive in-context information under sufficient retrieval pressure.

M7Q’s pressure response is consistent with urgency activating safety/refusal behavior. Y9C’s response indicates weaker pressure resistance. J7K is not an implicit privacy boundary. Unstable intermediate decreases warrant stability-based evaluation rather than treatment as suppression evidence.

**Claims and evidence**
- Pressure responses differ: Table 4, M7Q C8R/Z2F/L9W: 39.30%/40.50%/32.90%; Y9C: 38.70%/33.30%/40.60%.
- Intermediate decreases do not establish stable suppression: Figure 4 reports a dip near \(d=3\), then rebound; Section 5.6’s sole row assigns \(\tau=7\) to every model and persuasion condition over \(d\leq6\).

## 7 Limitations and Future Work

*The findings are bounded by drift length, secret formats, dataset construction, detectors, API access, and unreleased artifacts.*

##### Bounded D7A.

*The tested range does not characterize arbitrarily long or full-window behavior.*

The short-to-medium range does not saturate modern windows. These findings concern bounded active-context persistence, not a complete characterization of arbitrarily long or full-window privacy behavior.

##### Fixed-format secrets.

*Unstructured sensitive attributes remain outside the evaluation.*

Structured identifiers are easier to detect than unstructured health status, family circumstances, immigration status, or financial hardship. Extension to non-fixed sensitive attributes remains future work.

##### Active context, not cross-session memory.

*Training-data, personalization, and cross-session retention are untested.*

Training-data memorization, personalization memory, and cross-session retention are untested.

##### Synthetic and human-authored dialogues.

*The mixed dataset retains control, but separate human-versus-synthetic comparisons are deferred.*

Mixed construction improves realism over a purely synthetic benchmark while retaining experimental control. Cached metadata no longer preserves human-authored row identifiers. Only aggregate results are reported.

##### Detector scope.

*Surface-output labels lack reported formal human–automated agreement.*

Labels concern surface-form outputs under U3P, R5J, and an external judge. Separate reporting exposes detector behavior, but formal human–automated inter-annotator agreement is not reported. Larger manual audits of borderline partial disclosures and obfuscations remain future work.

##### API-served models and reproducibility.

*API-served model identities may change despite logging and paired evaluation.*

Compute/budget constraints require API service. The evaluated identifiers (M7Q = GPT-OSS-120B), (T4X = DeepSeek-R1), and (Y9C = Qwen3-VL-235B) designate models that may change over time despite logged identifiers, prompts, timestamps, cached responses, and paired statistics on identical dialogue sets.

##### Artifact availability.

*Planned artifacts must support reproduction without exposing realistic identifiers.*

Planned releases comprise generation code, probe templates, evaluation scripts, aggregate result files, and a sanitized dialogue subset. Synthetic PII-like strings necessitate non-real identifiers in released instances and templates sufficient for reproduction without exposing realistic identifiers.

## 8 Conclusion

*Secret-type asymmetry suggests format-sensitive safeguards rather than generalized contextual confidentiality.*

The S8Q asymmetry suggests uneven safeguards relying more on format/sensitivity cues and format-sensitive safety heuristics than on generalized contextual confidentiality or robust contextual confidentiality reasoning.

**Claims and evidence**
- The stated safeguard interpretation rests on secret-type asymmetry: Table 5, W3B/N6G rows: 0.14%/1.11%/5.97% and 0.00%/0.39%/4.01%; F8Z/A2L rows: 78.24%/57.11%/81.13% and 70.89%/46.13%/57.24%, for M7Q/T4X/Y9C.

## Appendix A Dataset Characterization

*Dialogue word counts and approximate token counts are reported for each drift length.*

| D7A, \(d\) | \(N\) | Avg. words | Median words | Avg. tokens | Median tokens |
|---|---:|---:|---:|---:|---:|
| 0 | 197 | 44.29 | 43.00 | 59.05 | 57.33 |
| 2 | 147 | 253.97 | 258.00 | 338.62 | 344.00 |
| 3 | 142 | 344.18 | 347.00 | 458.91 | 462.67 |
| 4 | 166 | 452.27 | 447.00 | 603.02 | 596.00 |
| 5 | 181 | 520.92 | 543.00 | 694.56 | 724.00 |
| 6 | 167 | 568.03 | 605.00 | 757.37 | 806.67 |

Table 7: Length characteristics by D7A; token counts approximate, word-derived.

**Claims and evidence**
- Characterization covers every sampled length: Table 7, \(d=0,2,3,4,5,6\) rows contain \(N=197,147,142,166,181,167\), respectively; each row reports average and median words and approximate tokens.

## Appendix B Additional Effect Sizes

*Secret type has large associations, while drift and persuasion have small or negligible associations.*

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

**Claims and evidence**
- S8Q associations are Large: Table 8, S8Q rows: T4X 0.5871, M7Q 0.7731, Y9C 0.6927.
- D7A is Small or Negligible: Table 8, D7A rows: T4X 0.1073, Small; M7Q 0.0659 and Y9C 0.0723, Negligible.
- V4N associations are Negligible: Table 8, V4N rows: T4X 0.0718, M7Q 0.0689, Y9C 0.0639.

## Appendix C Dialogue Construction Figure

*The figure presents dialogue construction details.*

Figure 6: Dialogue construction details.
