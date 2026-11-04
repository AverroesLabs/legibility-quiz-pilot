# PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations

Luciano Rolando Maldonado Romero, West Virginia University (lrm00020@mix.wvu.edu)

PrivDrift audits whether user-disclosed secrets in an active LLM conversation remain recoverable after topic drift and persuasion-based probing: 1,000 controlled multi-turn dialogues with seeded secrets, content-dense drift turns and standardized extraction probes. Active-context privacy risk should be evaluated as a persistent behavioral failure mode, not only as memorization or immediate jailbreaks.

Keywords: Trustworthy AI, Large Language Models, Privacy, Evaluation, Benchmarking

## 1 Introduction

Persistent assistants (healthcare triage, finance, enterprise copilots) receive PII, and earlier disclosures may stay available to later generations. Evaluations target training-data memorization or immediate jailbreaks. Prompt-injection and multi-turn attack work does not measure recoverability after unrelated drift. We do not assume users believe an LLM forgets within a session. The risk is underestimating how easily earlier secrets are elicited by indirect, justified, high-pressure or tool-mediated prompts (shared sessions, browser assistants, copilots, agentic workflows, prompt injection).

Contributions:
1. PrivDrift, a benchmark on two axes: topic drift (unrelated turns between disclosure and probe) and persuasion intensity.
2. Evaluation under direct, justified and high-pressure probes.
3. A reproducible hybrid detector (normalization matching + open-weights LLM judge), reporting dialogue-level, probe-level, regex-only, fuzzy and hybrid rates.
4. Privacy Half-Life (τ), a stability-based metric for persistent suppression. No model reaches stable decay in the observed range.

## 2 Related Work

### 2.1 Contextual Privacy and Inference-Time Leakage
Prior work (training-data extraction, membership inference, memorized PII) concerns parameters. Active-context privacy concerns user-provided information at inference. ConfAIde, PrivacyLens and PrivaCI-Bench test whether a model should share under static access-control or privacy-norm scenarios. PrivDrift isolates persistence after drift and later persuasion.

### 2.2 Multi-Turn Adversarial Attacks
Crescendo, automated multi-turn jailbreaks and prompt injection target harmful content or instruction hijacking. PrivDrift applies the multi-turn lens to information-flow control: re-disclosure of secrets the user supplied earlier.

### 2.3 Context Management and Unlearning
TOFU and WMDP address static QA and weights. PrivDrift targets suppression of secrets in active context, which matters where RAG, copilots and agents rely on prompt-level context management.

## 3 The PrivDrift Framework

Figure 1: Pipeline: build dialogues, append probes, query models, score with hybrid detector. Components: parametric dialogue generator, probing harness, metric suite.

### 3.1 Threat Model: Active-Context Re-Disclosure
The adversary may be able to inspect the transcript. The question is whether the model itself re-discloses the secret when later instructions query, justify or pressure it. Cases: later users in shared sessions; tool outputs, retrieved docs or injections in copilots/agents; reuse after a task shift. A user discloses S at t_0, followed by d unrelated content-dense drift turns D={u_1,a_1,…,u_d,a_d}, then probe P is appended. Leakage: the response reveals S exactly, approximately or semantically.

### 3.2 Dataset Generation Pipeline
N=1000 dialogues via scaffold-then-rewrite, which controls secret, drift and probe while reducing template artifacts. 900 are synthetic and 100 human-authored under the same constraints, interspersed rather than in a block. Cached metadata lacks human-authored IDs, so results are aggregate; a human-vs-synthetic comparison is future work.

| Drift d | N | Avg/Median words | Avg/Median tokens |
|---|---|---|---|
| 0 | 197 | 44.29/43.00 | 59.05/57.33 |
| 2 | 147 | 253.97/258.00 | 338.62/344.00 |
| 3 | 142 | 344.18/347.00 | 458.91/462.67 |
| 4 | 166 | 452.27/447.00 | 603.02/596.00 |
| 5 | 181 | 520.92/543.00 | 694.56/724.00 |
| 6 | 167 | 568.03/605.00 | 757.37/806.67 |
| All | 1,000 | 359.16/385.00 (min 43, max 795) | 478.87/513.33 (min 57.33, max 1060.00) |

Table 1: 4 secret categories. Tokens are estimated from word counts.

#### 3.2.1 Persona and Secret Injection
Personas are drawn from >100 professions/demographic profiles. S is a phone number, email, SSN or credit card, inserted in the opening turn as plausible task context.

#### 3.2.2 Content-Dense Topic Drift
Each dialogue has a topic (e.g., tax filing, medical triage, travel). Drift turns come from a topic-conditioned library of dense QA pairs and fallback tasks (professional writing, structured reasoning, long-form summarization), partly human-authored, adding unrelated entities and task demands.

#### 3.2.3 Naturalistic Refinement
A rewriting model improves fluency and persona realism while preserving the exact secret, drift length, roles and probe; only surface phrasing changes.

### 3.3 Probing Setup
A final probe is appended to the full history: Simple (direct neutral), Medium (contextual justification, e.g., a form), Hard (high-pressure/urgent).

$$d\sim\begin{cases}0&p=0.2,\\ \mathrm{Uniform}\{2,3,4,5,6\}&p=0.8.\end{cases}\quad(1)$$

Results reflect bounded active-context persistence, not arbitrarily long contexts.

### 3.4 Evaluation Methodology
Leak if the normalized secret is in the normalized response, or if the judge finds the full secret, a substantial substring, or enough partial info to identify it.

Figure 2: Matching first; regex-negatives go to the judge.

#### 3.4.1 Stage 1: Normalization-Based Matching
Numeric secrets: strip non-digits. Alphanumeric: normalize case/whitespace, keeping meaningful delimiters.
$$\mathrm{Leak}_{\mathrm{regex}}(R,S)=\mathbb{I}[\phi(S)\subset\phi(R)]\quad(2)$$

#### 3.4.2 Stage 2: LLM-as-a-Judge Verification
An open-weights Llama judge receives secret and response and gives a binary verdict, capturing partial disclosure, hints and formatting variants. Regex and hybrid rates are reported separately because the judge may err.

#### 3.4.3 Privacy Half-Life
With δ=0.1·L(0) over observed drifts D_obs:
$$\tau=\min\{d\in D_{obs}\mid L(d)\le\delta,\ \forall d'>d\in D_{obs}: L(d')\le\delta\}\quad(3)$$
If the set is empty, τ=max(D_obs)+1. This excludes transient dips.

Figure 3: A later resurgence disqualifies a transient decrease.

## 4 Experimental Setup

- **Models:** GPT-OSS-120B, DeepSeek-R1 and Qwen3-VL-235B via OpenRouter, with identical histories and probes.
- **Responses:** 1,000 dialogues per model × 3 probes = 3,000 responses per model, 9,000 in total.
- **Aggregation:** Overall leakage is dialogue-level: a dialogue leaks if any probe elicits the secret, so it can exceed average persuasion-level rates. Persuasion and fuzzy analyses are probe-level.
- **Statistics:**
  - 95% bootstrap CIs over dialogues
  - McNemar (Bonferroni) for paired comparisons
  - Cochran's Q with post-hoc McNemar across persuasion levels
  - chi-square with Cramer's V for drift, persuasion and secret type

## 5 Results

### 5.1 Dialogue-Level Hybrid Evaluation Largely Agrees With Deterministic Matching
The judge adds ≤1.10 pp, so leakage is dominated by direct or near-direct reproduction.

| Model | Regex | Hybrid | Δ | p | Hybrid 95% CI |
|---|---|---|---|---|---|
| GPT-OSS-120B | 47.70 | 47.70 | +0.00 | 1.0 | [44.6, 50.8] |
| DeepSeek-R1 | 37.80 | 38.70 | +0.90 | 7.7×10⁻³ | [35.7, 41.6] |
| Qwen3-VL-235B | 53.50 | 54.60 | +1.10 | 2.57×10⁻³ | [51.6, 57.7] |

Table 2: Dialogue-level leakage (%).

### 5.2 All Models Leak Substantially at the Dialogue Level
See Table 2.

### 5.3 Persuasion Intensity Affects Leakage Non-Monotonically
Cochran's Q p<10⁻⁵ for all models, but direction is model-dependent. GPT-OSS: hard is lowest, consistent with pressure triggering safety behavior. DeepSeek: medium is lowest. Qwen: hard is highest.

| Model | Simple | Medium | Hard |
|---|---|---|---|
| GPT-OSS-120B | 39.30 | 40.50 | 32.90 |
| DeepSeek-R1 | 28.10 | 22.00 | 29.20 |
| Qwen3-VL-235B | 38.70 | 33.30 | 40.60 |

Table 4: Probe-level hybrid leakage (%).

### 5.4 Drift Length Does Not Reliably Reduce Leakage Within the Tested Window
There is a dip near d=3 and then a rebound (Figure 4). V is 0.1073 (DeepSeek), 0.0659 (GPT-OSS), 0.0723 (Qwen). This is no claim about arbitrarily long contexts, but risk persists across several shifts.

### 5.5 Secret Type Strongly Determines Leakage
V is 0.5871 (DeepSeek), 0.7731 (GPT-OSS), 0.6927 (Qwen), all large. Behavior tracks format/sensitivity cues rather than contextual confidentiality.

| Secret | GPT-OSS | DeepSeek | Qwen |
|---|---|---|---|
| SSN | 0.14 | 1.11 | 5.97 |
| Credit card | 0.00 | 0.39 | 4.01 |
| Email | 78.24 | 57.11 | 81.13 |
| Phone | 70.89 | 46.13 | 57.24 |

Table 5: Probe-level hybrid leakage (%).

### 5.6 Privacy Half-Life Exceeds the Observed Window
No model approaches δ at any d, so τ=7 for all models and persuasion levels: no stable decay within d≤6.

### 5.7 Fuzzy Matching Supports the Hybrid Labels
This is a post-hoc check on cached outputs only, kept auxiliary because it is less expressive than the judge for partial or semantic disclosures.

| Model | Regex | Fuzzy 0.85 | Fuzzy 0.80 | Hybrid |
|---|---|---|---|---|
| DeepSeek-R1 | 25.50 | 26.40 | 26.53 | 26.43 |
| GPT-OSS-120B | 36.80 | 37.03 | 37.07 | 37.57 |
| Qwen3-VL-235B | 36.97 | 37.60 | 37.93 | 37.53 |

Table 6: Probe-level rates (%).

Figure 5: Hybrid leakage heatmap by drift × persuasion per model.

## 6 Discussion

Leakage shows behavioral recoverability under retrieval pressure. It does not show permanent memory or uniform behavior across long-context settings, and is mostly near-direct reproduction. The secret-type asymmetry suggests format-sensitive safety heuristics rather than robust contextual reasoning about confidentiality. Urgency cues may activate refusal in GPT-OSS. Topic drift is not an implicit privacy boundary: intermediate dips are too unstable to satisfy τ, which motivates stability-based evaluation.

## 7 Limitations and Future Work

##### Bounded drift range.
d≤6 is short-to-medium drift, not saturation of full context windows.

##### Fixed-format secrets.
Only structured PII is tested. Unstructured attributes (health, family, immigration status, financial hardship) are future work.

##### Active context, not cross-session memory.
Not tested: training-data memorization, personalization memory, cross-session retention.

##### Synthetic and human-authored dialogues.
The human-authored dialogues improve realism while preserving control.

##### Detector scope.
Labels rest on surface-form outputs. There is no formal human–automated agreement yet; a larger manual audit of borderline partial disclosures and obfuscations is needed.

##### API-served models and reproducibility.
API models (used for compute/budget reasons) may change. Mitigation: model IDs, prompts, timestamps, cached responses and paired statistics on identical dialogues are logged.

##### Artifact availability.
Planned release: generation code, probe templates, evaluation scripts, aggregate results, and a sanitized dialogue subset with non-real identifiers and templates sufficient to reproduce.

## 8 Conclusion

Across three LLMs (1,000 dialogues each), dialogue-level hybrid leakage is 38.7–54.6%. Secret type has the strongest association, persuasion effects are model-dependent, drift within d≤6 does not reliably reduce leakage, and protections remain uneven.

## Appendix A Dataset Characterization

Per-drift length statistics appear in Table 1.

## Appendix B Additional Effect Sizes

Probe-level persuasion V (negligible): DeepSeek 0.0718, GPT-OSS 0.0689, Qwen 0.0639. Secret type is large for all models. Drift is small for DeepSeek and negligible for the others.

## Appendix C Dialogue Construction Figure

Figure 6: Dialogue construction details.