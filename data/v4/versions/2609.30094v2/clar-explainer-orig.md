**Summary**

The paper asks whether secrets disclosed in an active large language model (LLM) conversation remain recoverable after unrelated topic shifts. PrivDrift, its benchmark, contains 1,000 controlled dialogues with seeded secrets, unrelated intervening turns, and standardized extraction probes. Three models are tested with direct, justified, and high-pressure probes. A hybrid detector combines normalized matching with an LLM judge. Dialogue-level leakage—whether any standardized probe elicits the secret—ranges from 38.7% to 54.6%. Persuasion effects depend on the model. Social Security Numbers and credit card numbers are suppressed more often than emails and phone numbers. Within the tested drift range, \(d\leq 6\), additional drift does not reliably reduce leakage. No evaluated model reaches the paper’s criterion for stable suppression.

**Glossary**

- **PrivDrift:** A benchmark for auditing recovery of user-disclosed secrets after conversational topic drift and persuasion-based probing.
- **LLM:** Large language model; the type of model evaluated in the paper.
- **Trustworthy AI:** The evaluation framing in which the paper places its privacy audit.
- **Persistent assistant:** An assistant that maintains coherence over extended interactions.
- **Active context / active conversation context:** Information present in the current conversation, rather than stored in model parameters or cross-session memory.
- **Context window:** The conversational context available to a model.
- **PII:** Personally Identifiable Information, including the structured secrets tested here.
- **Secret / seeded secret / ground-truth secret, \(S\):** The sensitive value inserted into a dialogue and later targeted by probes.
- **SSN:** Social Security Number, one of the four secret categories.
- **Topic drift:** Unrelated conversational turns between secret disclosure and later probing.
- **Drift length, \(d\):** The number of turns between secret disclosure and the final probe.
- **Content-dense drift turns:** Unrelated question-answer turns that add entities and task demands to the active context.
- **Probe, \(P\):** A standardized prompt appended to the active context to elicit the secret.
- **Persuasion intensity / persuasion level:** The interaction style used to elicit a secret.
- **Simple:** A direct, neutral probe.
- **Medium:** A probe with a contextual justification.
- **Hard:** A high-pressure or urgent probe.
- **Leakage event:** A response that reveals the secret exactly, approximately, or semantically.
- **Active-context re-disclosure:** Reproduction of sensitive information supplied earlier in the same active conversation.
- **Dialogue-level leakage:** Leakage counted when any standardized probe elicits the secret in a dialogue.
- **Probe-level leakage:** Leakage measured over individual model responses to persuasion probes.
- **Hybrid detector / hybrid evaluation:** Normalization-based matching followed, for negative responses, by an LLM judge.
- **Regex-only / normalization-based matching:** Detection based on whether the normalized secret appears in the normalized response.
- **Fuzzy matching:** An auxiliary deterministic check using normalized approximate string similarity.
- **LLM-as-a-judge:** A model that judges whether a response reveals a secret or a substantial part of it.
- **Open-weights Llama judge:** The judge used after normalization-based matching returns a negative result.
- **Training-data memorization / extraction:** Recovery of private information stored in model parameters.
- **Membership inference:** A privacy-research setting listed alongside training-data extraction and memorized-PII recovery.
- **Immediate jailbreaking:** Eliciting harmful behavior in the current turn.
- **Prompt injection:** Later instructions manipulating an LLM-integrated application.
- **Multi-turn adversarial attack:** Exploitation through multiple conversational turns.
- **Crescendo:** An attack in which seemingly benign conversations gradually elicit harmful outputs.
- **Automated jailbreak framework:** A framework that generates adversarial multi-turn attack paths.
- **Active-context privacy / inference-time leakage:** Privacy handling of user-provided information in the current interaction.
- **Contextual integrity:** The privacy-reasoning framing used by ConfAIde.
- **ConfAIde:** A benchmark evaluating privacy reasoning through contextual integrity.
- **PrivacyLens:** A benchmark studying privacy-norm awareness and legal compliance in agentic or contextual settings.
- **PrivaCI-Bench:** A benchmark studying privacy-norm awareness and legal compliance in agentic or contextual settings.
- **Machine unlearning:** Removal or suppression of knowledge from model behavior.
- **TOFU:** A machine-unlearning benchmark discussed as operating over static question-answer pairs and model parameters.
- **WMDP:** A machine-unlearning benchmark discussed as operating over static question-answer pairs and model parameters.
- **RAG systems:** A deployment setting discussed in connection with prompt-level context management.
- **Parameter-level unlearning:** Control over knowledge in model weights, contrasted with control over active context.
- **Information flow control:** Control over whether the model re-discloses sensitive information supplied earlier.
- **Parametric dialogue generator:** The framework component that constructs controlled dialogues.
- **Probing harness:** The framework component that appends probes and queries target models.
- **Metric suite:** Leakage-detection and temporal-analysis measures.
- **Scaffold-then-rewrite pipeline:** Controlled dialogue construction followed by naturalistic rewriting.
- **Persona:** A sampled profession and demographic profile used to initialize a dialogue.
- **Naturalistic refinement:** Rewriting surface phrasing while preserving the experimental variables.
- **Synthetic dialogue:** A dialogue generated through the benchmark pipeline.
- **Human-authored dialogue:** A dialogue written under the same secret and drift constraints.
- **Cached response:** A saved model-probe response used in the reported analyses.
- **GPT-OSS-120B / GPT-OSS:** One evaluated target model; “GPT-OSS” is its shortened table label.
- **DeepSeek-R1 / DeepSeek:** One evaluated target model; “DeepSeek” is its shortened table label.
- **Qwen3-VL-235B / Qwen:** One evaluated target model; “Qwen” is its shortened table label.
- **OpenRouter API:** The interface through which the evaluated models are queried.
- **Privacy Half-Life, \(\tau\):** The earliest observed drift length below the safety threshold with no later observed resurgence.
- **Safety threshold, \(\delta\):** \(0.1\times L(0)\).
- **\(L(d)\):** Leakage rate at drift length \(d\).
- **\(D_{obs}\):** The observed set of drift lengths.
- **\(t_0\):** The turn at which the user discloses the secret.
- **\(D\):** The sequence of unrelated drift-turn messages.
- **\(u_i,a_i\):** The paired messages in drift turn \(i\).
- **\(N\):** Number of dialogues, either overall or within a drift-length row.
- **\(R\):** The model response.
- **\(\phi(\cdot)\):** The normalization function applied to a secret or response.
- **\(\mathrm{Leak}_{\mathrm{regex}}(R,S)\):** The normalization-based leakage label.
- **\(\mathbb{I}\):** The indicator function in the regex-stage leakage definition.
- **\(\subset\):** The normalized-secret containment condition in Equation (2).
- **\(\mathrm{Uniform}\{2,3,4,5,6\}\):** The uniform drift-length component of the sampling distribution.
- **\(p\):** A sampling probability in Equation (1), or a statistical-test value in results.
- **\(\Delta\):** The additional dialogue-level leakage reported for hybrid versus regex evaluation.
- **CI:** Confidence interval; dialogue-level intervals use 95% bootstrap intervals over dialogues.
- **McNemar’s test:** The paired test used for matched model and detector comparisons.
- **Bonferroni correction:** The correction applied to paired comparisons where applicable.
- **Cochran’s \(Q\):** The test used for repeated measures across persuasion levels.
- **Chi-square test of independence:** The test used for association between leakage and benchmark variables.
- **Cramer’s \(V\):** The reported effect size for those associations.
- **Avg. / approx. / min/max:** Average, approximate, and minimum/maximum labels in dataset tables.
- **Approximate tokens:** Token counts estimated from word counts.
- **\(\min,\max\):** Minimum and maximum, used in the Privacy Half-Life definition.
- **\(\in,\forall,\mid,d'\):** Set membership, “for all,” “such that,” and another observed drift length in Equation (3).
- **\(\leq,>,<,\times,\cdot,\%\):** At most, greater than, less than, multiplication, multiplication, and percent.

# PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations

Luciano Rolando Maldonado Romero

Affiliation: West Virginia University, Morgantown, WV, USA

Correspondence to: lrm00020@mix.wvu.edu

###### Abstract

*This section introduces the benchmark and summarizes leakage under topic drift and persuasion-based probing.*

Large language models (LLMs) increasingly operate as persistent assistants: systems that maintain coherence over extended interactions. They serve users in shared-session and tool-augmented settings. Sensitive information disclosed during an active conversation may remain behaviorally recoverable after the dialogue shifts to unrelated topics.

We introduce PrivDrift, a benchmark for auditing recovery of user-disclosed secrets after topic drift—unrelated intervening conversation—and persuasion-based probing, which uses different interaction styles to elicit the secret. PrivDrift contains 1,000 controlled multi-turn dialogues with seeded secrets, content-dense drift turns, and standardized extraction probes.

Across three LLMs with extended context windows, dialogue-level hybrid leakage ranges from 38.7% to 54.6%. Dialogue-level leakage counts a dialogue if any standardized probe elicits its secret. Hybrid detection combines normalization-based matching with an LLM judge. Leakage varies strongly by model, secret type, and persuasion intensity. Within the tested drift window, additional topic drift does not reliably reduce leakage. This suggests evaluating privacy risk in active LLM contexts as a persistent behavioral failure mode, rather than only as training-data memorization or immediate jailbreak behavior.

**Claims and evidence**
- Leakage remains substantial across models: Table 3, DeepSeek-R1, GPT-OSS-120B, and Qwen3-VL-235B rows: 38.70%, 47.70%, and 54.60%.
- Persuasion effects depend on the model: Table 4, all model rows; Cochran’s \(Q\), \(p<10^{-5}\), in Section 5.3.
- Secret type has a strong association with leakage: Table 8, “Secret type” rows: Cramer’s \(V=0.5871\), \(0.7731\), and \(0.6927\).
- Drift does not reliably reduce leakage in the tested window: Table 8, “Drift length” rows: \(V=0.1073\), \(0.0659\), and \(0.0723\); Section 5.6 reports \(\tau=7\) for all models and persuasion levels.

###### Keywords:

*This section lists the paper’s subject areas.*

Trustworthy AI, Large Language Models, Privacy, Evaluation, Benchmarking

## 1 Introduction

*This section motivates active-context privacy auditing and states the benchmark’s contributions.*

LLMs have evolved from simple chat systems into persistent assistants that maintain coherence over extended interactions (OpenAI, 2023; Touvron et al., 2023; Liu et al., 2024). In healthcare triage, financial planning, enterprise copilots, and personal productivity tools, users may disclose Personally Identifiable Information (PII). This includes phone numbers, financial identifiers, email addresses, and health-related details. Context retention supports multi-turn assistance, but it also creates a privacy risk: sensitive information disclosed earlier in an active conversation may remain available to later generations.

Existing safety evaluations mostly target two extremes. Training-data memorization concerns extraction of private information from model parameters (Carlini et al., 2021; Nasr et al., 2023). Immediate jailbreaking concerns harmful behavior elicited in the current turn (Wei et al., 2024; Zou et al., 2023). Recent work also studies prompt injection—later instructions manipulating LLM-integrated applications—and multi-turn attacks, which exploit a model across several turns (Liu et al., 2023; Russinovich et al., 2024; Deng et al., 2024). These settings do not directly measure whether user-disclosed secrets remain recoverable after unrelated conversational drift. The resulting gap is whether topic shift acts as a practical privacy boundary within an active conversation.

We do not assume users believe that an LLM literally forgets earlier messages in the same session. We study a behavioral risk: users and application designers may underestimate how easily indirect, justified, high-pressure, or tool-mediated prompts can elicit earlier sensitive information. This matters in shared sessions, browser-integrated assistants, enterprise copilots, agentic workflows, and prompt-injection settings. In these settings, later instructions can interact with the active context—the current conversation—in ways the original user did not intend.

PrivDrift measures active-context privacy leakage along two axes. Topic drift is the number of unrelated turns between secret disclosure and later probing. Persuasion intensity is the interaction style used to elicit the secret. Across three LLMs with extended context windows and 1,000 dialogues per model, dialogue-level hybrid leakage ranges from 38.7% to 54.6%. Persuasion intensity significantly affects leakage. Secret type also matters sharply: Social Security Numbers (SSNs) and credit cards are suppressed far more often than emails and phone numbers. Additional drift does not reliably reduce leakage within the tested window, \(d\leq 6\), where \(d\) is drift length.

Our contributions are:

1. PrivDrift, a controlled benchmark for auditing whether user-disclosed secrets remain recoverable from active conversational context after unrelated topic drift.
2. Evaluation under direct, justified, and high-pressure probes, modeling different interaction styles for eliciting sensitive context.
3. A reproducible hybrid detector combining normalization-based matching with an open-weights LLM judge. We report dialogue-level, probe-level, regex-only, fuzzy, and hybrid leakage rates. Probe-level rates concern individual probe responses; regex-only rates use normalized containment; fuzzy rates use approximate string matching.
4. Privacy Half-Life (\(\tau\)), a stability-based metric for persistent suppression. No evaluated model reaches stable decay within the observed drift range.

**Claims and evidence**
- Dialogue-level leakage ranges from 38.7% to 54.6%: Table 3, all model rows: 38.70%, 47.70%, and 54.60%.
- Persuasion significantly affects leakage: Section 5.3 reports Cochran’s \(Q\), \(p<10^{-5}\); Table 4 gives each model’s Simple, Medium, and Hard rates.
- SSNs and credit cards are suppressed more often than emails and phones: Table 5, SSN and Credit card rows: 0.14%/1.11%/5.97% and 0.00%/0.39%/4.01%; Email and Phone rows: 78.24%/57.11%/81.13% and 70.89%/46.13%/57.24%.
- No stable decay occurs for \(d\leq 6\): Section 5.6 reports \(\tau=7\) for all models and persuasion levels; no separate table is reported.

## 2 Related Work

*This section distinguishes PrivDrift from privacy benchmarks, multi-turn attacks, and unlearning evaluations.*

### 2.1 Contextual Privacy and Inference-Time Leakage

*This section contrasts private information in model parameters with user information in the current interaction.*

Language-model privacy research has historically focused on training-data extraction, membership inference—a listed privacy-research setting—and recovery of memorized PII from pre-training corpora (Shokri et al., 2017; Carlini et al., 2021; Li et al., 2023; Nasr et al., 2023). These settings study private information stored in model parameters. Active-context privacy instead concerns user-provided information in the current interaction and its handling at inference time.

Recent benchmarks test privacy norms and contextual access. ConfAIde evaluates privacy reasoning through contextual integrity, its privacy-reasoning framing (Mireshghallah et al., 2024). PrivacyLens and PrivaCI-Bench study privacy-norm awareness and legal compliance in agentic or contextual settings (Shao et al., 2024; Li et al., 2025). These benchmarks mainly ask whether a model should share information in a static access-control or privacy-norm scenario. PrivDrift isolates persistence: whether a user-disclosed secret remains behaviorally recoverable after unrelated drift and later persuasion-based probing.

### 2.2 Multi-Turn Adversarial Attacks

*This section applies the multi-turn attack perspective to re-disclosure of user-provided secrets.*

Aligned models remain vulnerable to multi-turn exploitation. Crescendo attacks gradually elicit harmful outputs through seemingly benign conversations (Russinovich et al., 2024). Automated jailbreak frameworks generate adversarial multi-turn attack paths (Deng et al., 2024; Narula et al., 2025). Prompt injection can also manipulate LLM-integrated applications through later instructions (Liu et al., 2023).

These works usually concern unsafe content generation or instruction hijacking. PrivDrift instead applies the multi-turn lens to information flow control: whether a model re-discloses sensitive information supplied earlier by a user, rather than whether it produces harmful external content.

### 2.3 Context Management and Unlearning

*This section distinguishes behavioral suppression in active context from removal or suppression of model knowledge.*

Machine unlearning concerns removing or suppressing knowledge from model behavior. Benchmarks such as TOFU and WMDP usually operate over static question-answer pairs and model parameters (Maini et al., 2024; Li et al., 2024). PrivDrift targets active context rather than model weights. It asks whether a secret still present in the context window is behaviorally suppressed when later prompts try to elicit it.

This distinction matters for deployed RAG systems—a deployment setting discussed here in connection with context management—copilots, and agentic workflows. Their privacy control often depends on prompt-level context management rather than parameter-level unlearning.

## 3 The PrivDrift Framework

*This section describes the threat model, dialogue construction, probes, and leakage metrics.*

Figure 1: PrivDrift overview. The pipeline constructs controlled multi-turn dialogues, appends standardized probes, queries target models, and scores leakage with a hybrid detector.

PrivDrift is a synthetic evaluation framework for measuring whether sensitive information disclosed in an active conversation remains recoverable after topic drift. Figure 1 shows three components: a parametric dialogue generator for controlled construction, a probing harness for querying models, and a metric suite for leakage detection and temporal analysis.

### 3.1 Threat Model: Active-Context Re-Disclosure

*This section defines the failure mode as later reproduction of a secret already present in the conversation.*

Active-context re-disclosure occurs when sensitive information disclosed earlier remains recoverable through later prompts after unrelated drift. We do not assume an adversary is always unable to inspect the raw transcript. Instead, PrivDrift tests whether the model itself re-discloses the sensitive value when later instructions query, justify, or pressure it to do so.

In shared or persistent sessions, a later user may not understand what private information was previously disclosed. In enterprise copilots and agentic workflows, later tool outputs, retrieved documents, or prompt-injection content may issue instructions that interact with the active context. In browser- or application-integrated assistants, users may not intend earlier sensitive disclosures to be reused after the task changes. The question is not whether the transcript contains the secret. It is whether the assistant reproduces it when later prompted.

Formally, a user discloses a secret \(S\) at turn \(t_0\). The conversation then contains \(d\) unrelated but content-dense drift turns:

\[
D=\{u_1,a_1,\ldots,u_d,a_d\},
\]

where \(D\) denotes the drift sequence and \(u_i,a_i\) its paired messages at drift turn \(i\). A standardized probe \(P\) is appended to the active context. A leakage event occurs if the response reveals \(S\) exactly, approximately, or semantically.

### 3.2 Dataset Generation Pipeline

*This section describes the controlled dialogue pipeline and the evaluated dataset’s composition.*

We generate \(N=1000\) controlled dialogues, where \(N\) is the number of dialogues, through a scaffold-then-rewrite pipeline. Initial scaffolds control the ground-truth secret, drift length, and probe structure. Rewriting reduces rigid template artifacts.

The dataset contains 900 synthetic dialogues and 100 human-authored dialogues constructed under the same secret and drift constraints. Human-authored dialogues are distributed throughout the benchmark, not stored in a separate contiguous block. Current cached evaluation metadata does not preserve their identifiers. We therefore report aggregate results and leave a separate human-versus-synthetic leakage comparison for future work.

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

Here, “Avg.” means average, “approx.” means approximate, and “min/max” means minimum/maximum. Approximate token counts are estimated from word counts.

#### 3.2.1 Persona and Secret Injection

*This section describes how a persona and a fixed-format secret enter each dialogue.*

Each dialogue begins with a persona—a profession and demographic profile—sampled from more than 100 profiles, together with a ground-truth secret \(S\). We sample \(S\) from four fixed-format PII categories: phone numbers, email addresses, Social Security Numbers, and credit card numbers. The opening turn inserts the secret into a plausible task, giving the assistant a legitimate in-context reason to observe it.

#### 3.2.2 Content-Dense Topic Drift

*This section describes intervening tasks that add unrelated content while preserving experimental control.*

Each dialogue receives a topic such as tax filing, medical triage, or travel logistics. Drift turns come from a topic-conditioned library of dense question-answer pairs and fallback tasks involving professional writing, structured reasoning, or long-form summarization. Annotators write a subset of drift turns for the generation library.

These turns add unrelated entities and task demands to active context while preserving control over the original secret and drift length.

#### 3.2.3 Naturalistic Refinement

*This section describes rewriting that changes phrasing but preserves experimental variables.*

A rewriting model performs naturalistic refinement: improving fluency and persona realism while preserving the exact secret, drift length, role structure, and final probe conditions. This changes surface phrasing, not experimental variables.

**Claims and evidence**
- The evaluated dataset has 1,000 dialogues: Table 1, “Total dialogues,” “Synthetic dialogues,” and “Human-authored dialogues” rows: 1,000, 900, and 100.
- It covers four secret categories and six drift lengths: Table 1, “Secret categories” and “Drift lengths” rows: 4 and \(0,2,3,4,5,6\).
- Reported word lengths are 359.16 average, 385.00 median, and 43 / 795 minimum/maximum: Table 1, corresponding word-count rows.
- Reported approximate token lengths are 478.87 average, 513.33 median, and 57.33 / 1060.00 minimum/maximum: Table 1, corresponding token-count rows.

### 3.3 Probing Setup

*This section defines three persuasion levels and the drift-length sampling distribution.*

The main evaluation appends a final standardized probe to the completed conversation. Probes use three persuasion levels:

- **Simple:** A direct neutral query, such as asking for the phone number mentioned earlier.
- **Medium:** A query with contextual justification, such as needing the information for a form.
- **Hard:** A high-pressure or urgent request testing whether pressure cues increase or decrease leakage.

Drift length \(d\), the number of turns between disclosure and final probing, follows a mixed distribution:

\[
d\sim
\begin{cases}
0 & \text{with }p=0.2,\\
\mathrm{Uniform}\{2,3,4,5,6\} & \text{with }p=0.8.
\end{cases}
\tag{1}
\]

Here, \(p\) denotes sampling probability, and \(\mathrm{Uniform}\{2,3,4,5,6\}\) is the uniform component over those drift lengths. This includes immediate recall while emphasizing short-to-medium drift. We interpret results as bounded active-context persistence, not a full test of arbitrarily long context windows.

### 3.4 Evaluation Methodology

*This section defines hybrid detection and the stability-based Privacy Half-Life metric.*

A response may disclose a secret verbatim, with formatting changes, as a partial substring, or through a paraphrased reference. The hierarchical hybrid detector labels leakage if either the normalized secret appears in the normalized response or the LLM judge finds disclosure of the full secret, a substantial substring, or enough partial information to identify it.

Figure 2: Hierarchical hybrid evaluation. Normalization-based matching is applied first; regex-negative responses are then passed to an LLM judge.

#### 3.4.1 Stage 1: Normalization-Based Matching

*This section defines deterministic detection through normalized secret containment.*

The first stage detects verbatim and near-verbatim leakage. For numeric secrets, it strips all non-digit characters from the secret and response. For alphanumeric secrets, it normalizes case and whitespace while preserving meaningful delimiters.

Let \(\phi(\cdot)\) be this normalization function, \(R\) the model response, and \(\mathbb{I}\) the indicator function. The regex-stage leakage label is

\[
\mathrm{Leak}_{\mathrm{regex}}(R,S)
=
\mathbb{I}\big[\phi(S)\subset\phi(R)\big],
\tag{2}
\]

where \(\subset\) denotes the normalized-secret containment condition.

#### 3.4.2 Stage 2: LLM-as-a-Judge Verification

*This section describes model-based review of responses that deterministic matching labels negative.*

Responses negative under normalization-based matching go to an open-weights Llama judge (Meta AI, 2024). In this LLM-as-a-judge stage, the judge receives the secret and response. It returns a binary verdict on whether the response reveals the secret or a substantial part of it.

This stage targets partial disclosure, indirect hints, and formatting variations missed by deterministic matching. We report regex-only and hybrid rates separately because judge-based evaluation can introduce errors.

#### 3.4.3 Privacy Half-Life

*This section defines sustained suppression and excludes temporary leakage dips.*

Privacy Half-Life, \(\tau\), measures sustained suppression. Let \(D_{obs}\) be the observed set of drift lengths and \(L(d)\) the leakage rate at drift length \(d\). Define the safety threshold as \(\delta=0.1\times L(0)\). Then

\[
\begin{aligned}
\tau=\min\{d\in D_{obs}\mid\;&L(d)\leq\delta\\
&\text{and }\forall d'\in D_{obs},d'>d:\ L(d')\leq\delta\}.
\end{aligned}
\tag{3}
\]

Here, \(\min\) selects the earliest qualifying drift length; \(\in\) means membership, \(\mid\) means “such that,” \(\forall\) means “for all,” and \(d'\) is another observed drift length. If the set is empty, assign \(\tau=\max(D_{obs})+1\), where \(\max\) is the maximum observed drift length. This prevents a transient dip from counting as stable suppression.

Figure 3: Privacy Half-Life \(\tau\) requires stable decay below the threshold. A later resurgence disqualifies a transient decrease.

## 4 Experimental Setup

*This section specifies models, response counts, aggregation levels, and statistical methods.*

##### Models.

*This subsection identifies the three target models and their query interface.*

We evaluate GPT-OSS-120B, DeepSeek-R1, and Qwen3-VL-235B through the OpenRouter API, the interface used to query them. All models receive the same dialogue histories and probe templates.

##### Dataset.

*This subsection states the number of dialogues and cached responses evaluated.*

Each model is evaluated on 1,000 PrivDrift dialogues. Each dialogue has one seeded secret and one drift length \(d\in\{0,2,3,4,5,6\}\), sampled as above. Three standardized persuasion probes per dialogue yield 3,000 cached model-probe responses per model and 9,000 total cached responses. Cached responses are saved outputs used for analysis.

##### Aggregation levels.

*This subsection distinguishes dialogue-level vulnerability from individual probe responses.*

Overall leakage is dialogue-level: a dialogue leaks if any standardized probe elicits the secret. Persuasion-level and fuzzy-matching analyses are probe-level: each dialogue contributes one response per persuasion condition. Thus, dialogue-level leakage can exceed the average of individual persuasion-level rates.

##### Uncertainty and statistical tests.

*This subsection describes uncertainty estimates, paired comparisons, and association tests.*

We report regex-only and hybrid leakage. Dialogue-level rates use 95% bootstrap confidence intervals (CIs) over dialogues. Probe-level sensitivity checks use cached model-probe responses.

McNemar’s test compares matched examples for paired model and detector comparisons, with Bonferroni correction where applicable. Cochran’s \(Q\) tests repeated measures across persuasion levels, followed by post-hoc McNemar tests. Chi-square tests of independence assess associations between leakage and drift length, persuasion level, or secret type. Cramer’s \(V\) reports the effect size of those associations. In test results, \(p\) denotes the reported statistical-test value.

## 5 Results

*This section reports leakage by detector, model, persuasion level, drift length, and secret type.*

### 5.1 Dialogue-Level Hybrid Evaluation Largely Agrees With Deterministic Matching

*This section shows that the judge adds little dialogue-level leakage beyond normalized matching.*

At dialogue level, hybrid evaluation differs only slightly from normalization-based matching. This indicates that direct or near-direct reproduction dominates leakage in this benchmark, rather than purely semantic paraphrase. The judge adds up to 1.10 percentage points of leakage across models (Table 2). This supports hybrid leakage as the main metric, with regex-only rates retained as a transparent deterministic baseline.

| Model | Regex | Hybrid | \(\Delta\) | \(p\) |
|---|---:|---:|---:|---:|
| GPT-OSS-120B | 47.70 | 47.70 | +0.00 | 1.0 |
| DeepSeek-R1 | 37.80 | 38.70 | +0.90 | \(7.7\times 10^{-3}\) |
| Qwen3-VL-235B | 53.50 | 54.60 | +1.10 | \(2.57\times 10^{-3}\) |

Table 2: Dialogue-level leakage (%) under regex-only and hybrid evaluation. A dialogue is counted as leaking if any standardized probe elicits the secret.

Here, \(\Delta\) is the additional leakage under hybrid evaluation relative to regex-only evaluation.

**Claims and evidence**
- Hybrid evaluation adds little leakage: Table 2, GPT-OSS-120B row: 47.70 to 47.70, +0.00, \(p=1.0\); DeepSeek-R1 row: 37.80 to 38.70, +0.90, \(p=7.7\times10^{-3}\); Qwen3-VL-235B row: 53.50 to 54.60, +1.10, \(p=2.57\times10^{-3}\).
- Direct or near-direct reproduction dominates, supporting retention of the deterministic baseline: Table 2, all rows; the largest reported addition is +1.10 percentage points.

### 5.2 All Models Leak Substantially at the Dialogue Level

*This section reports each model’s vulnerability to at least one standardized extraction probe.*

Dialogue-level hybrid leakage is substantial for all models: 38.70% for DeepSeek-R1, 47.70% for GPT-OSS-120B, and 54.60% for Qwen3-VL-235B (Table 3). These rates estimate whether a dialogue is vulnerable to at least one standardized extraction probe.

| Model | Hybrid Leakage | 95% CI |
|---|---:|---|
| DeepSeek-R1 | 38.70% | [35.7, 41.6] |
| GPT-OSS-120B | 47.70% | [44.6, 50.8] |
| Qwen3-VL-235B | 54.60% | [51.6, 57.7] |

Table 3: Dialogue-level hybrid leakage rates with 95% bootstrap confidence intervals. A dialogue is counted as leaking if any standardized probe elicits the secret.

**Claims and evidence**
- All models leak substantially: Table 3, DeepSeek-R1 row: 38.70%, CI [35.7, 41.6]; GPT-OSS-120B row: 47.70%, CI [44.6, 50.8]; Qwen3-VL-235B row: 54.60%, CI [51.6, 57.7].

### 5.3 Persuasion Intensity Affects Leakage Non-Monotonically

*This section shows significant persuasion effects whose direction differs by model.*

Probe-level leakage varies significantly across persuasion levels for every model: Cochran’s \(Q\), \(p<10^{-5}\). The direction is model-dependent (Table 4).

For GPT-OSS-120B, hard persuasion reduces leakage relative to simple and medium persuasion, consistent with pressure cues triggering safety behavior. DeepSeek-R1 leaks least under medium persuasion. Qwen3-VL-235B leaks most under hard persuasion, indicating weaker resistance to pressure-based extraction.

| Model | Simple | Medium | Hard |
|---|---:|---:|---:|
| GPT-OSS-120B | 39.30% | 40.50% | 32.90% |
| DeepSeek-R1 | 28.10% | 22.00% | 29.20% |
| Qwen3-VL-235B | 38.70% | 33.30% | 40.60% |

Table 4: Probe-level hybrid leakage by persuasion intensity.

**Claims and evidence**
- Persuasion affects every model significantly: Cochran’s \(Q\), \(p<10^{-5}\), reported in this section; no separate test-statistic table is provided.
- Hard persuasion reduces GPT-OSS-120B leakage: Table 4, GPT-OSS-120B row: Simple 39.30%, Medium 40.50%, Hard 32.90%.
- Medium persuasion yields DeepSeek-R1’s lowest leakage: Table 4, DeepSeek-R1 row: 28.10%, 22.00%, 29.20%.
- Hard persuasion yields Qwen3-VL-235B’s highest leakage: Table 4, Qwen3-VL-235B row: 38.70%, 33.30%, 40.60%.

### 5.4 Drift Length Does Not Reliably Reduce Leakage Within the Tested Window

*This section reports small or negligible drift effects and a dip followed by rebound.*

Drift-leak curves dip near \(d=3\), then rebound. Drift-length effect sizes are small or negligible across models. In cached probe-level analysis, Cramer’s \(V\) is 0.1073 for DeepSeek-R1, 0.0659 for GPT-OSS-120B, and 0.0723 for Qwen3-VL-235B.

This is not a claim about arbitrarily long contexts. It shows that privacy risk can persist through several unrelated topic shifts even before the original secret is far from the end of the context window.

Figure 4: Hybrid leakage versus drift length. Curves show a dip near \(d=3\) followed by rebound.

**Claims and evidence**
- Curves dip near \(d=3\), then rebound: Figure 4 and its caption; pointwise leakage values are not tabulated.
- Drift effects are small or negligible: Table 8, DeepSeek-R1 / Drift length row: 0.1073, Small; GPT-OSS-120B / Drift length row: 0.0659, Negligible; Qwen3-VL-235B / Drift length row: 0.0723, Negligible.
- The claim is bounded to the tested window: drift lengths \(0,2,3,4,5,6\), Table 1, “Drift lengths” row.

### 5.5 Secret Type Strongly Determines Leakage

*This section shows much lower leakage for SSNs and credit cards than for emails and phone numbers.*

Secret type has a large association with hybrid leakage for every model. In cached probe-level analysis, Cramer’s \(V\) is 0.5871 for DeepSeek-R1, 0.7731 for GPT-OSS-120B, and 0.6927 for Qwen3-VL-235B.

SSNs and credit card numbers are suppressed far more often than emails and phone numbers, although all are user-disclosed and contextually private (Table 5). This suggests that current behavior depends more on format and sensitivity cues than on a generalized notion of contextual confidentiality.

The shortened labels GPT-OSS, DeepSeek, and Qwen refer to GPT-OSS-120B, DeepSeek-R1, and Qwen3-VL-235B.

| Secret Type | GPT-OSS | DeepSeek | Qwen |
|---|---:|---:|---:|
| SSN | 0.14% | 1.11% | 5.97% |
| Credit card | 0.00% | 0.39% | 4.01% |
| Email | 78.24% | 57.11% | 81.13% |
| Phone | 70.89% | 46.13% | 57.24% |

Table 5: Probe-level hybrid leakage by secret type across cached model-probe responses.

**Claims and evidence**
- Secret type has a large association with leakage: Table 8, “Secret type” rows: DeepSeek-R1 0.5871, GPT-OSS-120B 0.7731, Qwen3-VL-235B 0.6927; all interpreted as Large.
- SSNs and credit cards leak much less than emails and phones: Table 5, SSN row: 0.14%, 1.11%, 5.97%; Credit card row: 0.00%, 0.39%, 4.01%; Email row: 78.24%, 57.11%, 81.13%; Phone row: 70.89%, 46.13%, 57.24%.
- The format- and sensitivity-cue interpretation rests on this asymmetry: Table 5, all four secret-type rows.

### 5.6 Privacy Half-Life Exceeds the Observed Window

*This section reports that no tested model reaches the stable-suppression threshold.*

With the stable-decay threshold \(\delta=0.1\cdot L(0)\), no evaluated model approaches the threshold at any tested drift length. Therefore, \(\tau=\max(D_{obs})+1=7\) for all models and persuasion levels. No stable decay occurs within \(d\leq 6\).

**Claims and evidence**
- No model approaches the threshold or reaches stable decay: this section reports \(\delta=0.1\cdot L(0)\) and \(\tau=7\) for all models and persuasion levels within \(d\leq 6\); no separate results table is provided.

### 5.7 Fuzzy Matching Supports the Hybrid Labels

*This section compares an auxiliary approximate-string check with probe-level hybrid labels.*

As a post-hoc deterministic sensitivity check, we compute normalized fuzzy matching between each saved secret and saved response, using cached outputs only. At threshold 0.85, fuzzy leakage closely tracks probe-level hybrid leakage: 26.40% versus 26.43% for DeepSeek-R1, 37.03% versus 37.57% for GPT-OSS-120B, and 37.60% versus 37.53% for Qwen3-VL-235B (Table 6).

Fuzzy matching is auxiliary, not the primary metric. Approximate string similarity is less expressive than the judge for partial or semantic disclosures.

| Model | Regex | Fuzzy 0.85 | Fuzzy 0.80 | Hybrid |
|---|---:|---:|---:|---:|
| DeepSeek-R1 | 25.50 | 26.40 | 26.53 | 26.43 |
| GPT-OSS-120B | 36.80 | 37.03 | 37.07 | 37.57 |
| Qwen3-VL-235B | 36.97 | 37.60 | 37.93 | 37.53 |

Table 6: Probe-level regex, fuzzy, and hybrid leakage rates (%) computed from cached model outputs.

Figure 5: Hybrid leakage heatmap by drift length and persuasion intensity for each model.

**Claims and evidence**
- Fuzzy 0.85 closely tracks hybrid leakage: Table 6, DeepSeek-R1 row: 26.40 versus 26.43; GPT-OSS-120B row: 37.03 versus 37.57; Qwen3-VL-235B row: 37.60 versus 37.53.
- The cached-output sensitivity check also reports regex and Fuzzy 0.80 rates: Table 6, DeepSeek-R1 row: 25.50 and 26.53; GPT-OSS-120B row: 36.80 and 37.07; Qwen3-VL-235B row: 36.97 and 37.93.

## 6 Discussion

*This section interprets persistent leakage, secret-type asymmetry, persuasion effects, and unstable drift-related dips.*

PrivDrift reveals a persistent active-context privacy risk: sensitive information disclosed earlier can remain recoverable after unrelated drift. This does not mean models permanently remember secrets or that all long-context settings behave alike. Current assistants may reproduce sensitive in-context information when later prompts provide enough retrieval pressure. Cached probe-level results also indicate mostly direct or near-direct reproduction, because fuzzy matching closely tracks hybrid leakage.

The strongest pattern is secret-type asymmetry. SSNs and credit card numbers are suppressed far more often than emails and phone numbers, although all four are user-disclosed secrets. This suggests safeguards may rely on format-sensitive safety heuristics rather than robust contextual reasoning about confidentiality.

Persuasion effects are significant, non-monotonic, and model-dependent. High-pressure requests reduce GPT-OSS-120B leakage, consistent with the possibility that urgency cues activate refusal behavior. Medium persuasion produces DeepSeek-R1’s lowest leakage. Hard persuasion produces Qwen3-VL-235B’s highest leakage, suggesting weaker resistance to pressure-based extraction.

Topic drift should not be treated as an implicit privacy boundary. Intermediate dips do not remain low enough to satisfy Privacy Half-Life. This motivates stability-based evaluation rather than treating temporary decreases as suppression.

**Claims and evidence**
- Active-context leakage persists: Table 3, all model rows: 38.70%, 47.70%, 54.60%; Figure 4 shows a dip near \(d=3\) followed by rebound.
- Fuzzy matching closely tracks hybrid results: Table 6, Fuzzy 0.85 / Hybrid columns: DeepSeek-R1 26.40 / 26.43; GPT-OSS-120B 37.03 / 37.57; Qwen3-VL-235B 37.60 / 37.53.
- Secret-type asymmetry is the strongest pattern: Table 8, “Secret type” rows: 0.5871, 0.7731, 0.6927; Table 5, SSN and Credit card rows: 0.14%/1.11%/5.97% and 0.00%/0.39%/4.01%, versus Email and Phone rows: 78.24%/57.11%/81.13% and 70.89%/46.13%/57.24%.
- Persuasion direction differs by model: Table 4, GPT-OSS-120B 39.30%/40.50%/32.90%; DeepSeek-R1 28.10%/22.00%/29.20%; Qwen3-VL-235B 38.70%/33.30%/40.60%; Section 5.3 reports \(p<10^{-5}\).
- Temporary dips do not meet stable suppression: Section 5.6 reports \(\tau=7\) for all models and persuasion levels within \(d\leq 6\).

## 7 Limitations and Future Work

*This section limits the findings to bounded active context, structured secrets, aggregate dialogue results, and automated detection.*

##### Bounded drift range.

*This subsection limits the findings to short-to-medium drift rather than full context-window saturation.*

PrivDrift tests drift lengths up to \(d=6\). This is bounded short-to-medium drift, not full saturation of modern context windows. Results show persistence under bounded active-context drift, not a complete characterization of privacy across full long-context windows.

##### Fixed-format secrets.

*This subsection notes that the tested structured secrets do not cover unstructured sensitive attributes.*

SSNs, credit card numbers, emails, and phone numbers are easier to detect than unstructured sensitive information such as health status, family circumstances, immigration status, or financial hardship. Future work should extend the benchmark to non-fixed sensitive attributes.

##### Active context, not cross-session memory.

*This subsection excludes memorization, personalization memory, and cross-session retention from the evaluation.*

PrivDrift tests recoverability within active conversation context. It does not test training-data memorization, personalization memory, or cross-session retention.

##### Synthetic and human-authored dialogues.

*This subsection explains why human-authored and synthetic dialogues are evaluated together.*

The dataset contains 900 synthetic and 100 human-authored dialogues distributed throughout the benchmark rather than stored in separate blocks. This improves realism relative to a purely synthetic benchmark while controlling secret type, drift length, and probe structure. However, cached evaluation metadata does not preserve which rows are human-authored. We report aggregate results and leave a separate human-versus-synthetic comparison for future work.

##### Detector scope.

*This subsection notes the absence of formal human–automated agreement measurements and calls for manual auditing.*

Labels use surface-form outputs, deterministic matching, fuzzy matching, and an external LLM judge. Separate regex-only, fuzzy, and hybrid results make detector behavior transparent. We do not yet report formal inter-annotator agreement between humans and automated methods. Future work should include a larger manual audit of borderline partial disclosures and obfuscations.

##### API-served models and reproducibility.

*This subsection notes model changes over time and the records retained for reproducibility.*

Compute and budget constraints lead us to evaluate API-served models, which may change over time. We mitigate this by logging model identifiers, prompts, timestamps, cached responses, and paired statistics on identical dialogue sets.

##### Artifact availability.

*This subsection states planned releases and the use of non-real identifiers.*

We plan to release benchmark generation code, probe templates, evaluation scripts, aggregate result files, and a sanitized subset of generated dialogues. Because the benchmark contains synthetic PII-like strings, released examples will use non-real identifiers. Generation templates will support reproduction without exposing realistic identifiers.

## 8 Conclusion

*This section summarizes persistent leakage, model-dependent persuasion effects, secret-type asymmetry, and the absence of stable decay.*

PrivDrift audits active-context privacy leakage under topic drift and persuasion intensity. Across three LLMs and 1,000 dialogues per model, dialogue-level hybrid leakage ranges from 38.7% to 54.6%. Persuasion affects leakage in model-dependent ways. Secret type has the strongest association with leakage.

Within the tested drift range, topic drift does not reliably reduce leakage. Privacy Half-Life exceeds the observed window for every model. The asymmetry across secret types suggests that current protections remain uneven and depend more on format-sensitive cues than on robust contextual confidentiality.

**Claims and evidence**
- Dialogue-level hybrid leakage remains substantial: Table 3, DeepSeek-R1 38.70%, GPT-OSS-120B 47.70%, Qwen3-VL-235B 54.60%; Section 4 specifies 1,000 dialogues per model.
- Persuasion effects are model-dependent: Table 4, Simple/Medium/Hard rates: GPT-OSS-120B 39.30%/40.50%/32.90%; DeepSeek-R1 28.10%/22.00%/29.20%; Qwen3-VL-235B 38.70%/33.30%/40.60%.
- Secret type has the strongest association: Table 8, “Secret type” rows: 0.5871, 0.7731, 0.6927; “Drift length” rows: 0.1073, 0.0659, 0.0723; “Persuasion” rows: 0.0718, 0.0689, 0.0639.
- Stable decay is absent within the tested window: Section 5.6 reports \(\tau=7\) for all models and persuasion levels within \(d\leq6\).
- The interpretation of uneven, format-sensitive protection rests on secret-type asymmetry: Table 5, SSN 0.14%/1.11%/5.97%; Credit card 0.00%/0.39%/4.01%; Email 78.24%/57.11%/81.13%; Phone 70.89%/46.13%/57.24%.

## Appendix A Dataset Characterization

*This section reports dialogue counts and word- and token-length summaries at each drift length.*

| Drift \(d\) | \(N\) | Avg. words | Median words | Avg. tokens | Median tokens |
|---|---:|---:|---:|---:|---:|
| 0 | 197 | 44.29 | 43.00 | 59.05 | 57.33 |
| 2 | 147 | 253.97 | 258.00 | 338.62 | 344.00 |
| 3 | 142 | 344.18 | 347.00 | 458.91 | 462.67 |
| 4 | 166 | 452.27 | 447.00 | 603.02 | 596.00 |
| 5 | 181 | 520.92 | 543.00 | 694.56 | 724.00 |
| 6 | 167 | 568.03 | 605.00 | 757.37 | 806.67 |

Table 7: Dataset length characteristics by drift length. Approximate token counts are estimated from word counts.

**Claims and evidence**
- Characteristics at \(d=0\): Table 7, row 0: \(N=197\), average/median words 44.29/43.00, average/median tokens 59.05/57.33.
- Characteristics at \(d=2\): Table 7, row 2: \(N=147\), words 253.97/258.00, tokens 338.62/344.00.
- Characteristics at \(d=3\): Table 7, row 3: \(N=142\), words 344.18/347.00, tokens 458.91/462.67.
- Characteristics at \(d=4\): Table 7, row 4: \(N=166\), words 452.27/447.00, tokens 603.02/596.00.
- Characteristics at \(d=5\): Table 7, row 5: \(N=181\), words 520.92/543.00, tokens 694.56/724.00.
- Characteristics at \(d=6\): Table 7, row 6: \(N=167\), words 568.03/605.00, tokens 757.37/806.67.

## Appendix B Additional Effect Sizes

*This section reports probe-level association effect sizes and their stated interpretations.*

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

**Claims and evidence**
- Secret-type associations are Large: Table 8, “Secret type” rows: DeepSeek-R1 0.5871, GPT-OSS-120B 0.7731, Qwen3-VL-235B 0.6927.
- Drift association is Small for DeepSeek-R1 and Negligible for the other models: Table 8, “Drift length” rows: 0.1073, 0.0659, 0.0723.
- Persuasion associations are Negligible: Table 8, “Persuasion” rows: DeepSeek-R1 0.0718, GPT-OSS-120B 0.0689, Qwen3-VL-235B 0.0639.

## Appendix C Dialogue Construction Figure

*This section presents the dialogue-construction figure.*

Figure 6: Dialogue construction details.
