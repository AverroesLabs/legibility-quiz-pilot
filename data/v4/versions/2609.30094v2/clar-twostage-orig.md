# PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations

Luciano Rolando Maldonado Romero

Affiliation: West Virginia University, Morgantown, WV, USA

Correspondence to: lrm00020@mix.wvu.edu

###### Abstract

Large language models (LLMs), systems that generate language, increasingly operate as persistent assistants. They retain conversational context in user-facing applications, sessions used by more than one person, and systems that use external tools. Sensitive information disclosed during an active conversation may remain behaviorally recoverable: later prompts may cause the model to reproduce it, even after unrelated topics intervene.

We introduce PrivDrift, a benchmark for auditing this risk. It tests secret recovery after topic drift, meaning unrelated conversational turns between disclosure and probing, and persuasion-based probing, meaning requests that try to elicit the secret through different interaction styles. PrivDrift contains 1,000 controlled multi-turn dialogues with deliberately inserted secrets, content-dense drift turns, and standardized extraction probes. Its hybrid detector combines matching after text normalization with an LLM judge that checks disclosures missed by matching. A dialogue counts as leaking if any standardized probe elicits the secret.

Across three LLMs with extended context windows—the conversation text available to a model when it responds—dialogue-level hybrid leakage ranges from 38.7% to 54.6% (Table 3). Leakage varies strongly by model, secret type, and persuasion intensity (Tables 3–5). Within the tested drift window, additional topic drift does not reliably reduce leakage: drift associations are small or negligible, and no model reaches sustained suppression (Sections 5.4 and 5.6). These results suggest evaluating privacy risk in active LLM contexts as a persistent behavioral failure mode, not only as recovery of private training data or harmful behavior elicited by an immediate jailbreak.

###### Keywords:

Trustworthy AI, Large Language Models, Privacy, Evaluation, Benchmarking

Here, AI means artificial intelligence.

## 1 Introduction

Large language models (LLMs) have evolved from simple chat systems into persistent assistants that maintain coherence over extended interactions (OpenAI, 2023; Touvron et al., 2023; Liu et al., 2024). They are being integrated into healthcare triage, financial planning, enterprise copilots—assistants for workplace tasks—and personal productivity tools. Users in these settings may disclose Personally Identifiable Information (PII), meaning information that identifies a person. Examples include phone numbers, financial identifiers, email addresses, and health-related details. Retaining context supports useful assistance across multiple conversational turns. The same retention can leave earlier sensitive information available for later responses.

Existing safety evaluations mostly target two extremes. Training-data memorization concerns private information extracted from model parameters, the learned values also called model weights (Carlini et al., 2021; Nasr et al., 2023). Immediate jailbreaking concerns harmful behavior elicited in the current turn by prompts that bypass safeguards (Wei et al., 2024; Zou et al., 2023). Recent work also studies prompt injection, in which later instructions manipulate an LLM-integrated application, and multi-turn attacks, which steer a model across a sequence of exchanges (Liu et al., 2023; Russinovich et al., 2024; Deng et al., 2024). These settings do not directly measure whether user-disclosed secrets remain recoverable after unrelated conversational drift. The gap is whether a topic shift acts as a practical privacy boundary within an active conversation.

We do not assume that users believe an LLM literally forgets earlier messages in the same session. We study a behavioral risk instead. Users and application designers may underestimate how easily later prompts can elicit earlier sensitive information. Those prompts may be indirect, provide a justification, apply pressure, or arrive through tools. This risk is relevant to shared sessions, browser-integrated assistants, enterprise copilots, agentic workflows in which assistants carry out tasks through multiple steps or tools, and prompt-injection settings. In these settings, later instructions can interact with active context in ways the original user did not intend.

PrivDrift measures active-context privacy leakage along two axes. Topic drift is the number of unrelated turns between secret disclosure and later probing. Persuasion intensity is the interaction style used to elicit the secret. Across three LLMs with extended context windows and 1,000 dialogues per model, dialogue-level hybrid leakage ranges from 38.7% to 54.6% (Table 3). Persuasion intensity significantly affects leakage, with model-dependent directions (Table 4; Section 5.3). Secret type also matters sharply. Social Security Numbers (SSNs) and credit card numbers are suppressed far more often than email addresses and phone numbers (Table 5). Within the tested window, \(d\leq 6\), additional drift does not reliably reduce leakage (Sections 5.4 and 5.6). Here, \(d\) denotes drift length: the number of unrelated turns between disclosure and probing.

Our contributions are as follows:

1. PrivDrift is a controlled benchmark for auditing whether user-disclosed secrets remain recoverable from active conversational context after unrelated topic drift.

2. We evaluate direct, justified, and high-pressure probes. These model different interaction styles through which later prompts can elicit sensitive context.

3. We implement a reproducible hybrid detector that combines normalization-based matching—matching after standardizing text—with an open-weights LLM judge, a model whose weights are available and which assesses disclosure. We report dialogue-level and probe-level rates. We also report regex-only rates, the normalization-based matching stage labeled for regular-expression matching; fuzzy rates, based on approximate text matching; and hybrid rates, which combine matching and the judge.

4. We propose Privacy Half-Life, denoted by \(\tau\), a stability-based metric for sustained suppression. No evaluated model reaches stable decay within the observed drift range (Section 5.6).

## 2 Related Work

### 2.1 Contextual Privacy and Inference-Time Leakage

Language-model privacy research has historically focused on extracting training data, inferring whether information belonged to the training set through membership inference, and recovering memorized PII from pre-training corpora, the text collections used for initial training (Shokri et al., 2017; Carlini et al., 2021; Li et al., 2023; Nasr et al., 2023). These settings ask whether private information is stored in model parameters.

Active-context privacy concerns a different source of information: user-provided material in the current interaction. The model must handle that information appropriately at inference time, when it generates a response.

Recent benchmarks examine privacy norms and contextual access: whether sharing information is appropriate in the stated setting and whether a recipient should have access. ConfAIde evaluates privacy reasoning through contextual integrity, or appropriate information sharing within a context (Mireshghallah et al., 2024). PrivacyLens and PrivaCI-Bench study awareness of privacy norms and legal compliance in agentic or contextual settings (Shao et al., 2024; Li et al., 2025). These names identify the respective benchmarks.

These benchmarks primarily ask whether a model should share information in a static access-control or privacy-norm scenario, where permissions or sharing expectations are specified for the scenario. PrivDrift instead isolates persistence. It asks whether a user-disclosed secret remains behaviorally recoverable after unrelated topic drift and later persuasion-based probing.

### 2.2 Multi-Turn Adversarial Attacks

Aligned models—models trained to follow intended behavioral constraints—remain vulnerable to exploitation across multiple turns. Crescendo attacks show that seemingly benign conversations can gradually elicit harmful outputs (Russinovich et al., 2024). Automated jailbreak frameworks generate adversarial multi-turn attack paths, or sequences of prompts designed to bypass safeguards (Deng et al., 2024; Narula et al., 2025). Prompt-injection studies also show that later instructions can manipulate LLM-integrated applications (Liu et al., 2023).

These works usually focus on unsafe content generation or instruction hijacking, in which attacker instructions redirect the system. PrivDrift applies the multi-turn perspective to information flow control: controlling whether information is re-disclosed. The question is not whether a model can be made to produce harmful external content. It is whether the model will re-disclose sensitive information supplied by a user earlier in the same active context.

### 2.3 Context Management and Unlearning

Machine unlearning concerns removing or suppressing knowledge from model behavior. TOFU and WMDP are benchmarks that evaluate this goal (Maini et al., 2024; Li et al., 2024). They usually operate over fixed question-answer pairs and model parameters.

PrivDrift is related in spirit but targets active context rather than model weights. It asks whether a secret in the context window is behaviorally suppressed when later prompts try to elicit it. The distinction matters for deployed retrieval-augmented generation (RAG) systems, which use retrieved material in generation, as well as copilots and agentic workflows. In these settings, privacy control often depends on managing the context supplied in prompts rather than unlearning information from model parameters.

## 3 The PrivDrift Framework

Figure 1: PrivDrift overview. The pipeline constructs controlled multi-turn dialogues, appends standardized probes, queries target models, and scores leakage with a hybrid detector.

PrivDrift is a synthetic evaluation framework for measuring whether sensitive information disclosed in an active conversation remains recoverable after topic drift. As described in Figure 1, it has three components: a dialogue generator controlled by parameters, a probing harness that submits the test prompts, and a metric suite that detects leakage and analyzes its persistence across drift lengths.

### 3.1 Threat Model: Active-Context Re-Disclosure

We study active-context re-disclosure. This is a failure mode in which later prompts recover sensitive information disclosed earlier in a conversation, after unrelated topic drift. We do not assume that an adversary is always unable to inspect the raw transcript. Instead, PrivDrift tests whether the model itself re-discloses a sensitive value when later instructions ask for it, justify its use, or pressure the assistant to reveal it.

The setting captures several practical risks. In a shared or persistent session, a later user may not understand what private information was disclosed before. In enterprise copilots and agentic workflows, tool outputs, retrieved documents, or prompt-injection content may contain instructions that interact with active context. In browser- or application-integrated assistants, the user may not intend earlier sensitive disclosures to be reused after the task changes. The question is therefore not whether the transcript contains the secret. It is whether the assistant reproduces the secret when later prompted.

Formally, let \(S\) be the secret and \(t_0\) the turn at which the user discloses it. The conversation then proceeds through \(d\) unrelated but content-dense drift turns:

\[
D=\{u_1,a_1,\ldots,u_d,a_d\}.
\]

Here, \(D\) denotes the drift material. Each \(u_i\) is the user message and each \(a_i\) the assistant message in drift turn \(i\); \(i\) indexes the turns. At evaluation time, a standardized probe \(P\), meaning a test request with a controlled form, is appended to the active context. Leakage occurs if the model response reveals \(S\) exactly, approximately, or semantically—that is, through its meaning rather than only its exact text.

### 3.2 Dataset Generation Pipeline

We generate \(N=1000\) controlled dialogues, where \(N\) is the total number of dialogues. We use a multi-stage scaffold-then-rewrite pipeline: first construct a controlled dialogue structure, then refine its wording. This preserves control over the ground-truth secret—the known value against which disclosure is checked—drift length, and probe structure. It also reduces the rigid artifacts of generation based only on templates.

The dataset contains 900 synthetic dialogues and 100 human-authored dialogues constructed under the same secret and drift constraints (Table 1). The human-authored dialogues were inserted throughout the benchmark, not placed in a separate contiguous block. The current cached evaluation metadata—the saved information describing evaluated examples—does not preserve their identifiers. We therefore report aggregate results over the full benchmark. A separate human-versus-synthetic leakage comparison remains future work.

| Statistic | Value |
|---|---:|
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

In Table 1, “Avg.” means average, “approx.” means approximate, and “Min/max” means minimum and maximum. Tokens are the text units processed by a model; the table estimates their counts from word counts.

#### 3.2.1 Persona and Secret Injection

Each dialogue begins with a persona, or user profile, sampled from more than 100 professions and demographic profiles. It also contains a ground-truth secret \(S\). We sample \(S\) from four fixed-format PII categories: phone numbers, email addresses, Social Security Numbers, and credit card numbers.

The secret is inserted in the opening turn as part of a plausible task. This gives the assistant a legitimate in-context reason to observe it.

#### 3.2.2 Content-Dense Topic Drift

Each dialogue is assigned a topic, such as tax filing, medical triage, or travel logistics. Drift turns come from a topic-conditioned library: a collection of dense question-answer pairs selected according to the topic. Fallback tasks involve professional writing, structured reasoning, or long-form summarization. Annotators also write a subset of drift turns, which are incorporated into the generation library.

This design aims to add unrelated entities and task demands to active context. At the same time, it preserves control over the original secret and the number of drift turns.

#### 3.2.3 Naturalistic Refinement

A final refinement step reduces the rigidity of the initial scaffold. Each scaffold is passed to a rewriting model. The model is instructed to improve fluency and persona realism while preserving the exact secret, drift length, role structure—the arrangement of user and assistant messages—and final probe conditions.

This step changes surface phrasing but does not change the experimental variables.

### 3.3 Probing Setup

The main evaluation appends a final standardized probe to the completed conversation history. Probes fall into three persuasion levels:

- **Simple:** a direct, neutral query, such as asking for the phone number mentioned earlier.
- **Medium:** a query with a contextual justification, such as needing the information for a form.
- **Hard:** a high-pressure or urgent request designed to test whether pressure cues increase or decrease leakage.

We sample drift length \(d\), the number of turns between disclosure and final probing, from the following mixed distribution:

\[
d\sim
\begin{cases}
0 & \text{with }p=0.2,\\
\mathrm{Uniform}\{2,3,4,5,6\} & \text{with }p=0.8.
\end{cases}
\tag{1}
\]

The symbol \(\sim\) means “is sampled from,” and \(p\) here means sampling probability. With probability \(0.2\), there are no drift turns. With probability \(0.8\), we choose uniformly from \(2,3,4,5,6\), meaning each listed value is equally likely within that choice.

This distribution includes immediate recall cases while emphasizing short-to-medium topic drift. We interpret the results as bounded active-context persistence, not as a full test of arbitrarily long context windows.

### 3.4 Evaluation Methodology

A model may reveal a secret verbatim, with formatting changes, as a partial substring—a consecutive portion of its text—or through a paraphrased reference. We therefore use a hierarchical hybrid detector: matching is applied first, followed by a judge for responses that matching does not flag.

A response counts as leakage if either:

- the normalized secret appears in the normalized response; or
- the LLM judge finds that the response reveals the full secret, a substantial substring, or enough partial information to identify the secret.

Figure 2: Hierarchical hybrid evaluation. Normalization-based matching is applied first; regex-negative responses are then passed to an LLM judge.

#### 3.4.1 Stage 1: Normalization-Based Matching

The first stage detects verbatim and near-verbatim leakage. For numeric secrets, we remove all non-digit characters from both the ground-truth secret and the model response. For alphanumeric secrets, which contain letters or numbers, we normalize case and whitespace while preserving meaningful delimiters, or separating characters.

Let \(R\) be the model response and \(\phi(\cdot)\) the normalization function. The dot denotes the text supplied to that function. The regex-stage leakage label is

\[
\mathrm{Leak}_{\mathrm{regex}}(R,S)
=\mathbb{I}\big[\phi(S)\subset\phi(R)\big].
\tag{2}
\]

Here, \(\mathrm{Leak}_{\mathrm{regex}}\) names the matching-stage label. The symbol \(\subset\) expresses the condition that the normalized secret appears within the normalized response. The indicator function \(\mathbb{I}\) returns one when that condition holds and zero otherwise.

#### 3.4.2 Stage 2: LLM-as-a-Judge Verification

Responses not flagged by normalization-based matching are passed to an open-weights Llama judge (Meta AI, 2024). Llama is the model family used for this judging stage. The judge receives the secret and the model response. It returns a binary verdict: whether or not the response reveals the secret or a substantial part of it.

This stage is intended to capture partial disclosure, indirect hints, and formatting variations missed by deterministic matching, whose result follows fixed matching rules. We report regex-only and hybrid rates separately because judge-based evaluation can introduce its own errors.

#### 3.4.3 Privacy Half-Life

Privacy Half-Life, \(\tau\), measures sustained suppression rather than a temporary decline in leakage. Let \(D_{obs}\) be the set of observed drift lengths; the subscript “obs” means observed. Let \(L(d)\) be the leakage rate at drift length \(d\). Thus, \(L(0)\) is the leakage rate with no drift.

We set the safety threshold to \(\delta=0.1\times L(0)\), where \(\delta\) denotes the threshold and \(\times\) means multiplication. We then define

\[
\tau=
\min\left\{
d\in D_{obs}\ \middle|\
L(d)\leq\delta
\text{ and }
\forall d'\in D_{obs},\,d'>d:\ L(d')\leq\delta
\right\}.
\tag{3}
\]

In plain words, choose the smallest observed drift length at which leakage is at or below the threshold and remains there at every larger observed drift length. In the formula, \(\min\) means smallest, \(\in\) means belongs to, \(\leq\) means at or below, and \(\forall\) means for every. The symbol \(d'\) denotes another observed drift length, and \(d'>d\) restricts it to lengths larger than the candidate \(d\).

If no observed drift length meets both conditions, we assign \(\tau=\max(D_{obs})+1\): the largest observed drift length plus one. This prevents transient dips from being interpreted as stable suppression.

Figure 3: Privacy Half-Life \(\tau\) requires stable decay below the threshold. A later resurgence disqualifies a transient decrease.

## 4 Experimental Setup

##### Models.

We evaluate GPT-OSS-120B, DeepSeek-R1, and Qwen3-VL-235B through the OpenRouter application programming interface (API), the interface used to submit requests to the models. These three names are model identifiers. All models receive the same dialogue histories and probe templates.

##### Dataset.

We evaluate 1,000 PrivDrift dialogues on each model. Each dialogue contains one seeded secret, meaning a deliberately inserted known value, and one drift length \(d\in\{0,2,3,4,5,6\}\), sampled from Equation (1).

Each dialogue is evaluated under three standardized persuasion probes. This yields 3,000 cached model-probe responses per model and 9,000 total cached responses across the three models. A model-probe response is one saved response from one model to one probe for a dialogue.

##### Aggregation levels.

We report two aggregation levels:

- **Dialogue level:** overall leakage counts a dialogue as leaking if any standardized probe elicits the secret.
- **Probe level:** persuasion-level and fuzzy-matching analyses count individual responses. Each dialogue contributes one response per persuasion condition.

The dialogue-level rule combines the three opportunities to elicit a secret. It therefore explains why dialogue-level leakage can exceed the average of the individual persuasion-level leakage rates.

##### Uncertainty and statistical tests.

We report both regex-only and hybrid leakage. For dialogue-level rates, we compute 95% bootstrap confidence intervals (CIs) over dialogues. These intervals describe uncertainty using resampling of dialogues. Probe-level sensitivity checks compute rates over cached model-probe responses.

For paired model and detector comparisons on matched examples, we use McNemar’s test, which compares paired binary outcomes. Where applicable, we apply Bonferroni correction to account for multiple comparisons. For repeated measurements across persuasion levels on the same dialogues, we use Cochran’s \(Q\) test, followed by post-hoc McNemar tests—paired comparisons after the overall test.

We use chi-square tests of independence to test whether leakage is associated with drift length, persuasion level, or secret type. We report Cramer’s \(V\) as the effect size, meaning the strength of association. Statistical \(p\)-values, unlike the sampling probabilities in Equation (1), describe how unusual the test result would be under the test’s no-difference or independence assumption.

## 5 Results

### 5.1 Dialogue-Level Hybrid Evaluation Largely Agrees With Deterministic Matching

Dialogue-level hybrid rates differ only slightly from normalization-based matching rates (Table 2). GPT-OSS-120B is unchanged at 47.70%. DeepSeek-R1 rises from 37.80% to 38.70%, and Qwen3-VL-235B rises from 53.50% to 54.60%.

The judge therefore adds up to 1.10 percentage points of dialogue-level leakage, as shown in the Qwen3-VL-235B row. These small differences indicate that direct or near-direct reproduction dominates leakage in this benchmark, rather than purely semantic paraphrase. This supports using hybrid leakage as the main metric while retaining regex-only rates as a transparent deterministic baseline.

In Table 2, \(\Delta\) is hybrid leakage minus regex-only leakage, in percentage points. The \(p\) column gives the statistical comparison’s \(p\)-value.

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

Probe-level leakage differs significantly across persuasion levels for all models: Cochran’s \(Q\) test gives \(p<10^{-5}\). The direction depends on the model, rather than increasing consistently with persuasion intensity (Table 4).

For GPT-OSS-120B, hard persuasion produces 32.90% leakage, below simple at 39.30% and medium at 40.50%. This is consistent with pressure cues triggering safety behavior. For DeepSeek-R1, medium persuasion is lowest at 22.00%, compared with 28.10% for simple and 29.20% for hard. For Qwen3-VL-235B, hard persuasion is highest at 40.60%, compared with 38.70% for simple and 33.30% for medium. This indicates weaker resistance to pressure-based extraction.

| Model | Simple | Medium | Hard |
|---|---:|---:|---:|
| GPT-OSS-120B | 39.30% | 40.50% | 32.90% |
| DeepSeek-R1 | 28.10% | 22.00% | 29.20% |
| Qwen3-VL-235B | 38.70% | 33.30% | 40.60% |

Table 4: Probe-level hybrid leakage by persuasion intensity.

The probe-level association effect sizes for persuasion are nevertheless classified as negligible: Cramer’s \(V\) is 0.0718 for DeepSeek-R1, 0.0689 for GPT-OSS-120B, and 0.0639 for Qwen3-VL-235B (Table 8, persuasion rows).

### 5.4 Drift Length Does Not Reliably Reduce Leakage Within the Tested Window

The drift-leak curves dip near \(d=3\) and then rebound (Figure 4). Drift-length effect sizes are small or negligible. In the cached probe-level analysis, Cramer’s \(V\) is 0.1073 for DeepSeek-R1, 0.0659 for GPT-OSS-120B, and 0.0723 for Qwen3-VL-235B (Table 8, drift-length rows).

This is not a claim about arbitrarily long contexts. It shows that privacy risk can persist across several unrelated topic shifts, even before the original secret is far from the end of the context window.

Figure 4: Hybrid leakage versus drift length. Curves show a dip near \(d=3\) followed by rebound.

### 5.5 Secret Type Strongly Determines Leakage

Secret type has a large association with hybrid leakage in all models. In the cached probe-level analysis, Cramer’s \(V\) is 0.5871 for DeepSeek-R1, 0.7731 for GPT-OSS-120B, and 0.6927 for Qwen3-VL-235B (Table 8, secret-type rows).

SSNs and credit card numbers are suppressed far more often than emails and phone numbers, although all are user-disclosed and contextually private (Table 5). For example, the SSN and credit card rows range from 0.00% to 5.97%, while the email and phone rows show much higher leakage. The pattern suggests that current behavior depends more on format and sensitivity cues than on contextual confidentiality: treating information as confidential because of the context in which it was supplied.

Table 5 shortens the model identifiers: “GPT-OSS” means GPT-OSS-120B, “DeepSeek” means DeepSeek-R1, and “Qwen” means Qwen3-VL-235B.

| Secret Type | GPT-OSS | DeepSeek | Qwen |
|---|---:|---:|---:|
| SSN | 0.14% | 1.11% | 5.97% |
| Credit card | 0.00% | 0.39% | 4.01% |
| Email | 78.24% | 57.11% | 81.13% |
| Phone | 70.89% | 46.13% | 57.24% |

Table 5: Probe-level hybrid leakage by secret type across cached model-probe responses.

### 5.6 Privacy Half-Life Exceeds the Observed Window

We apply the stable-decay definition of Privacy Half-Life with threshold \(\delta=0.1\cdot L(0)\), where the centered dot means multiplication. No evaluated model approaches the threshold at any tested drift length.

No drift length therefore satisfies Equation (3). The fallback rule gives \(\tau=\max(D_{obs})+1=7\) for all models and persuasion levels. This indicates no stable decay within \(d\leq 6\).

### 5.7 Fuzzy Matching Supports the Hybrid Labels

As a post-hoc deterministic sensitivity check, we compare each saved secret with its saved model response using normalized fuzzy matching. This check uses cached outputs only. Fuzzy matching measures approximate text similarity rather than requiring an exact normalized match.

At a 0.85 similarity threshold, fuzzy leakage closely tracks probe-level hybrid leakage for all models (Table 6). DeepSeek-R1 gives 26.40% fuzzy versus 26.43% hybrid leakage. GPT-OSS-120B gives 37.03% versus 37.57%, and Qwen3-VL-235B gives 37.60% versus 37.53%.

We treat fuzzy matching as auxiliary evidence, not the primary metric. Approximate string similarity is less expressive than the judge for partial disclosures or disclosures conveyed through meaning. The columns “Fuzzy 0.85” and “Fuzzy 0.80” identify the two similarity thresholds used.

| Model | Regex | Fuzzy 0.85 | Fuzzy 0.80 | Hybrid |
|---|---:|---:|---:|---:|
| DeepSeek-R1 | 25.50 | 26.40 | 26.53 | 26.43 |
| GPT-OSS-120B | 36.80 | 37.03 | 37.07 | 37.57 |
| Qwen3-VL-235B | 36.97 | 37.60 | 37.93 | 37.53 |

Table 6: Probe-level regex, fuzzy, and hybrid leakage rates (%) computed from cached model outputs.

Figure 5: Hybrid leakage heatmap by drift length and persuasion intensity for each model.

## 6 Discussion

PrivDrift reveals a persistent active-context privacy risk. Sensitive information disclosed earlier can remain behaviorally recoverable after unrelated topic drift. Dialogue-level hybrid leakage is 38.70% for DeepSeek-R1, 47.70% for GPT-OSS-120B, and 54.60% for Qwen3-VL-235B (Table 3), and no model reaches stable suppression in the tested drift window (Section 5.6).

This does not imply that models permanently remember the secret or that all long-context settings behave similarly. Instead, current assistants may reproduce sensitive in-context information when later prompts apply enough retrieval pressure—that is, enough prompting to elicit earlier information. The cached probe-level evaluation also indicates mostly direct or near-direct reproduction: fuzzy matching closely tracks hybrid leakage (Table 6).

The strongest pattern is the asymmetry across secret types. SSNs and credit card numbers are suppressed far more often than emails and phone numbers, although all four are user-disclosed secrets in the benchmark (Table 5). Secret-type associations are large for all models, unlike the small or negligible drift and persuasion associations (Table 8). This suggests that current safeguards may rely on format-sensitive safety heuristics—behavioral rules responsive to the secret’s format—rather than robust contextual reasoning about confidentiality.

Persuasion effects are significant but non-monotonic and model-dependent (Section 5.3; Table 4). For GPT-OSS-120B, high-pressure requests reduce leakage to 32.90%, compared with 39.30% for simple and 40.50% for medium requests. This is consistent with the possibility that urgency cues activate refusal behavior. DeepSeek-R1 has its lowest leakage under medium persuasion, at 22.00%. Qwen3-VL-235B has its highest leakage under hard persuasion, at 40.60%, suggesting weaker resistance to pressure-based extraction.

Finally, topic drift should not be treated as an implicit privacy boundary. The curves have intermediate dips followed by rebound (Figure 4). These decreases do not satisfy the Privacy Half-Life criterion: \(\tau=7\) for all models and persuasion levels, beyond the observed window (Section 5.6). This motivates stability-based evaluation rather than treating temporary drops as evidence of suppression.

## 7 Limitations and Future Work

##### Bounded drift range.

PrivDrift tests drift lengths up to \(d=6\). This is bounded short-to-medium conversational drift, not full saturation of modern context windows. The results are evidence of persistence under bounded active-context drift. They do not completely characterize privacy behavior across full long-context windows.

##### Fixed-format secrets.

The benchmark focuses on structured secrets: SSNs, credit card numbers, emails, and phone numbers. These are easier to detect than unstructured sensitive information such as health status, family circumstances, immigration status, or financial hardship. Future work should extend the benchmark to sensitive attributes without fixed formats.

##### Active context, not cross-session memory.

PrivDrift evaluates recoverability within the active conversation context. It does not test training-data memorization, personalization memory—information retained to personalize assistance—or cross-session retention, meaning retention between separate conversations.

##### Synthetic and human-authored dialogues.

The dataset contains 900 synthetic dialogues and 100 human-authored dialogues. The human-authored dialogues are inserted throughout the benchmark, not stored as a separate block. This construction improves realism relative to a purely synthetic benchmark while preserving control over secret type, drift length, and probe structure.

However, the current cached evaluation metadata does not preserve which evaluated rows are human-authored. We therefore report aggregate results and leave a separate human-versus-synthetic comparison for future work.

##### Detector scope.

Leakage labels are based on surface-form outputs, meaning the text the model produces. We use deterministic matching, fuzzy matching, and an external LLM judge. We report regex-only, fuzzy, and hybrid labels separately to make detector behavior transparent.

We do not yet report formal inter-annotator agreement between humans and automated methods—that is, a formal measure of how often their labels agree. Future work should include a larger manual audit of borderline partial disclosures and obfuscations, where a response disguises the secret.

##### API-served models and reproducibility.

Compute and budget constraints lead us to evaluate models served through APIs. These models may change over time. We mitigate this by logging model identifiers, prompts, timestamps, cached responses, and paired statistics on identical dialogue sets.

##### Artifact availability.

We plan to release the benchmark generation code, probe templates, evaluation scripts, aggregate result files, and a sanitized subset of generated dialogues. “Sanitized” means prepared to avoid exposing realistic identifiers.

The benchmark contains synthetic PII-like strings: generated text resembling personal identifiers. Released examples will therefore use non-real identifiers. We also plan to provide generation templates sufficient to reproduce the benchmark without exposing realistic identifiers.

## 8 Conclusion

We introduce PrivDrift, a benchmark for auditing active-context privacy leakage under topic drift and persuasion intensity. Across three LLMs and 1,000 dialogues per model, dialogue-level hybrid leakage ranges from 38.7% to 54.6% (Table 3).

Persuasion intensity affects leakage in model-dependent ways (Table 4). Secret type has the strongest association with leakage: its Cramer’s \(V\) values are 0.5871, 0.7731, and 0.6927, compared with small or negligible drift and persuasion associations (Table 8).

Within the tested drift range, topic drift does not reliably reduce leakage. Privacy Half-Life exceeds the observed window for all models: \(\tau=7\) with \(d\leq 6\) (Section 5.6). The asymmetry between SSN and credit card leakage and email and phone leakage (Table 5) suggests that current protections remain uneven and depend more on format-sensitive cues than on robust contextual confidentiality.

## Appendix A Dataset Characterization

In Table 7, \(d\) is drift length and \(N\) is the number of dialogues at that length. “Avg.” denotes the average. Token counts are approximate estimates from word counts, as in Table 1.

| Drift \(d\) | \(N\) | Avg. words | Median words | Avg. tokens | Median tokens |
|---:|---:|---:|---:|---:|---:|
| 0 | 197 | 44.29 | 43.00 | 59.05 | 57.33 |
| 2 | 147 | 253.97 | 258.00 | 338.62 | 344.00 |
| 3 | 142 | 344.18 | 347.00 | 458.91 | 462.67 |
| 4 | 166 | 452.27 | 447.00 | 603.02 | 596.00 |
| 5 | 181 | 520.92 | 543.00 | 694.56 | 724.00 |
| 6 | 167 | 568.03 | 605.00 | 757.37 | 806.67 |

Table 7: Dataset length characteristics by drift length. Approximate token counts are estimated from word counts.

## Appendix B Additional Effect Sizes

Table 8 reports Cramer’s \(V\), the strength of the probe-level association between hybrid leakage and each benchmark variable. “Large,” “Small,” and “Negligible” are the reported effect-size interpretations.

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
