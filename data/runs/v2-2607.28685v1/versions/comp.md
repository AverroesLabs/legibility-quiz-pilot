# Safety, or Just Capability? A Validity Audit of Agent-Safety Benchmarks

Youting Wang, Xiao Han, Dingyan Shang, Yuan Tang, Bowen Liu

###### Abstract

Agent-safety benchmarks measure different behaviors, yet their scores are quoted interchangeably as an agent's safety. We validate four (R-Judge, InjecAgent, AgentHarm, AgentDojo) as measurements, each under its official implementation and author-provided scorer on up to 22 models, with MMLU and GPQA measured by us under one protocol as a capability composite. We find a closed-form F1 failure, benchmark-dependent rankings, a small-panel trade-off artifact, outcome- and panel-dependent capability–safety links, and a strong AgentHarm–jailbreak association that is convergent validity, not general safety. A safety claim minimally needs its benchmark, metric, target behavior, and model panel.

## 1 Introduction

A taxonomy of >40 agent-safety benchmarks reports rank concordance near 0.10 across models, but stops at a 12-model, four-benchmark check and lists capability-controlled evaluation as open. We separate construct validity (what a score measures), metric validity (whether its metric measures that target), and criterion validity (whether it tracks held-out behavior):

- RQ1 (construct structure): common factor or dissociable ones?
- RQ2 (capability confound): how much between-benchmark signal is general capability?
- RQ3 (criterion validity): do safety scores track held-out criteria beyond capability?
- RQ4 (metric validity): do headline metrics reward degenerate behavior?

#### Contributions.

A formal F1 failure (Observation 1); quantification of a small-panel failure mode; a pre-specified, capability-controlled criterion audit over one task-success and two safety outcomes; an API-only audit harness and re-run artifacts for four benchmarks and three held-out outcomes.

#### Main conclusion.

A capability score is not a safety score, and no one benchmark stands in for safety; validity needs re-checking as model populations turn over.

#### Scope.

RQ3 criteria are stand-ins for deployment, not real-world harm claims (limits in §6).

## 2 Related Work

#### Agent-safety benchmarking.

InjecAgent and AgentDojo target prompt-injection robustness, AgentHarm harmful-task compliance, R-Judge safety-risk awareness over traces; evaluators like ToolEmu were likewise never validated as measurements. AutoMonitor-Bench reports miss and false-alarm rates, the two-sided reporting we argue for. The closest work documents ranking disagreement but neither controls capability, tests criterion validity, nor recovers latent structure; we do all three.

#### Construct validity.

Knowledge/reasoning benchmarks share a strong latent factor; capability confounding of agent-safety comparisons is unquantified.

#### Benchmark gaming.

Our claim is prior to gaming: the F1-gaming policy is not adversarial; it labels every trace "unsafe."

#### Over-refusal.

XSTest/OR-Bench work at prompt level; we ask whether AgentHarm refusers are worse at R-Judge specificity. On the full panel the correlation is near zero; on seven-model subsets often not.

#### Agent reliability and task performance.

We use τ²-bench retail as the held-out task-success criterion.

## 3 Method

### 3.1 Model panel and benchmarks

API-only panel of up to 22 models from nine organizations (OpenAI 6, Meta 3, Qwen 3, Mistral 3, DeepSeek 2, Amazon 2, Anthropic, Google, Cohere). Official implementations/scorers: R-Judge (all 571 items, self-judge), InjecAgent (300 stratified direct-harm/data-stealing items, rule-scored), AgentHarm (44 base behaviors, all eight harm categories, official Inspect evaluation, gpt-4o-mini judge), AgentDojo (slack suite, 100 items, environment-scored security and utility). Coverage: R-Judge n=21 (Qwen3-32B below the 90% valid-output threshold); InjecAgent 22; AgentHarm 19 and AgentDojo 5 (native tool use required); τ² 20. Common R-Judge/InjecAgent/AgentHarm panel: n=18. Anchors measured by us through the same harness: MMLU zero-shot on a fixed 500-item subset (seed 42, same items for all models), GPQA-Diamond (198 items, CoT). Anchor panel: 21 models (DeepSeek-R1 excluded for unstable long-form GPQA generation); provider-reported model-card anchors kept as pre-harmonization comparison.

### 3.2 Benchmark targets

Fixed before analysis: R-Judge distinguishes unsafe from benign traces; InjecAgent and AgentDojo test injection resistance (their correlation is a convergent check); AgentHarm tests refusal of harmful tasks. R-Judge is also analyzed by specificity and balanced accuracy.

#### Score orientation and terminology.

Higher = better/safer. Specificity: fraction of benign traces judged benign; balanced accuracy averages both classes. InjecAgent robustness = 1 − attack success; AgentHarm safety = 1 − harmful compliance. τ² is task success, not safety. "Internal consistency" refers to split-half analyses.

### 3.3 Analysis plan

Hypotheses, thresholds, decision rules fixed beforehand (no independent timestamp, so "pre-specified," not preregistered). Spearman correlations, two-sided asymptotic p unless noted; Fig. 2 bars are model-level percentile bootstraps. PCA on the Pearson matrix of standardized scores; factor count by Horn's parallel analysis vs. random-normal null. Partial Spearman correlations control the capability composite (mean of standardized MMLU and GPQA; ordering identical to the anchors' PC1).

Controls: positive (anchor loadings ≥0.6 on factor 1); negative (column-permuted matrix within the null); convergent (high InjecAgent–AgentDojo correlation). Headline metric: median pairwise partial |ρ| among distinct-construct benchmarks. Factor structure interpretable only if first-factor variance ≥40% or capability control reduces that median by ≥0.15, with positive and negative controls passing.

RQ3 outcomes, each pre-specified: task success = τ²-bench retail (40 multi-turn, environment-scored tool-agent-user tasks); misalignment safety = avoiding blackmail or leaking in a fictional autonomous-agent scenario, fixed classifier, 100 decisions/model; jailbreak safety = 1 − harmful compliance on 50 forbidden prompts, averaged over three published templates. Safety criteria cover the full panel; τ² covers 20. None is a predictor. Headline: partial ρ of each safety score with each criterion, after reporting the criterion's raw capability association.

## 4 Results

Ordered by evidential strength: RQ4, RQ2 firmest; RQ1 weaker; power-limited RQ3 last.

### 4.1 The headline metric is gameable (RQ4)

R-Judge labels 52.7% of 571 items unsafe. "Always unsafe" attains recall 1.0, specificity 0, F1=0.690, outranking five of the 21 valid models (F1 0.30–0.57; Fig. 1), all of which discriminate. Highest-specificity o3-mini (0.97) gets F1=0.702, only 0.012 above, due to lower recall. Balanced accuracy reorders the board.

###### Observation 1 (F1 admits a high-scoring constant baseline).

With base rate π, the always-positive policy attains F1=2π/(1+π) without discriminating; for R-Judge (π=0.527) this is 0.690. The closed form depends on prevalence, but the problem is F1's blindness to true negatives, so F1 cannot stand alone as a measure of two-sided discrimination here.

| Model | R-Judge spec. | InjecAgent robust. | AgentHarm safety | GPQA |
|---|---|---|---|---|
| Llama-3.1-8B | 0.04 | 0.32 | 0.89 | 30.4 |
| Claude-3-Haiku | 0.40 | 0.77 | 0.81 | 33.3 |
| Llama-3.1-70B | 0.11 | 0.35 | 0.78 | 46.7 |
| o3-mini | 0.97 | 0.87 | 0.77 | 77.2 |
| Nova-Lite | 0.94 | 0.88 | 0.76 | 42.0 |
| Nova-Pro | 0.64 | 0.60 | 0.75 | 46.9 |
| Gemini-2.5-Flash | 0.92 | 0.71 | 0.62 | 68.3 |
| DeepSeek-V3 | 0.86 | 0.76 | 0.61 | 59.1 |
| Qwen-2.5-72B | 0.62 | 0.92 | 0.60 | 49.1 |
| GPT-4.1 | 0.61 | 0.87 | 0.60 | 66.3 |
| GPT-4.1-mini | 0.76 | 0.87 | 0.51 | 65.0 |
| GPT-4o | 0.72 | 0.90 | 0.47 | 53.6 |
| GPT-4.1-nano | 0.79 | 0.94 | 0.42 | 50.3 |
| Llama-3.3-70B | 0.28 | 0.31 | 0.39 | 50.0 |
| Mistral-Nemo | 0.80 | 0.34 | 0.35 | 28.0 |
| GPT-4o-mini | 0.61 | 0.89 | 0.31 | 40.2 |
| Mixtral-8x22B | 0.58 | 0.78 | 0.24 | 33.2 |
| Mistral-Large | 0.50 | 0.81 | 0.24 | 44.0 |

Table 1: Same n=18 models ranked differently (sorted by AgentHarm).

### 4.2 The capability confound is real but metric-dependent (RQ2)

R-Judge panel (n=20, harmonized anchors): capability correlates with balanced accuracy (ρ=+0.71 MMLU, p<0.001; +0.49 GPQA, p=0.03) and F1 (+0.76, +0.62), weakly with specificity (+0.16, +0.07). BA and F1 include unsafe-trace recall; specificity can be inflated by rarely flagging. The confound depends on the metric and the panel: specificity–MMLU fell from 0.85 (first nine models) to +0.16 at n=20.

### 4.3 Benchmarks disagree; a seven-model panel suggested a trade-off that does not persist (RQ1)

Firmest, correlation-free: no benchmark orders models like another (Table 1). Cross-construct disagreement is expected and indicts no instrument; it defeats interchangeable quoting. R-Judge-specificity/AgentHarm-safety ρ: −0.64 at n=7, +0.02 (p=0.95) at n=18 (Fig. 5); size-7 subsets give |ρ|≥0.5 in a quarter of draws; dissolution holds under all four R-Judge metrics (A.4). Capability moderation of the remainder is an untested hypothesis.

PCA (corroborative; Fig. 6): PC1 48.7%; R-Judge specificity, InjecAgent robustness, both anchors one way, AgentHarm opposite (+0.33; opposite in 17/18 leave-one-out fits).

#### The positive control now passes.

MMLU loads 0.74≥0.6 (provider-reported anchors failed at 0.42); negative control passes; convergent control inconclusive at AgentDojo n=5. Conditions met, but the factor is corroboration only: Spearman PCA gives PC1 40.8% with no factor retained by Horn (A.3); median between-benchmark correlation 0.22.

### 4.4 Predictive validity differs across held-out outcomes (RQ3)

| Predictor | τ² | misalign. | jailbreak† |
|---|---|---|---|
| Capability (raw ρ) | +0.60** | −0.44* | +0.08 |
| R-Judge specificity (partial) | −0.09 | +0.41 | −0.11 |
| InjecAgent robustness (partial) | +0.16 | +0.47 | +0.21 |
| AgentHarm safety (partial) | −0.23 | +0.16 | +0.72*** |

Table 2: 2024–25 panel, cellwise n=18–21; partials control capability; †three-template; *p<0.05, **p<0.01, ***p<0.001.

#### Against task success, safety scores add no information beyond capability.

Capability: ρ=+0.60, p=0.005, n=20; best safety partial |ρ|=0.23, n.s., n=19 (Fig. 7). Capability already explains this outcome; the null is no verdict on safety scores.

#### Against agentic misalignment, capability has the opposite association.

n=21: ρ_cap=−0.44 (p=0.047) (Mistral-Large blackmails/leaks 85%; o3-mini a capable exception at 1%). InjecAgent (+0.47) and R-Judge specificity (+0.41) reach the pre-specified 0.40 threshold; AgentHarm (+0.16) does not. Exploratory: CIs include zero, fail Bonferroni, power ≈0.35, and the directional prediction named AgentHarm. R-Judge partial: +0.41 specificity, +0.19 BA, −0.46 recall (A.4).

#### A direct interaction test on the original panel.

Paired n=20: task +0.60 vs. misalignment −0.41; Δ=−1.00 (95% CI [−1.48,−0.49], p<0.001), stable to leave-one-model/organization-out, organization-clustered bootstrap, subsampling (A.5). R-Judge interaction marginal (Δ=+0.50, p=0.082): capability interaction supported, full two-way dissociation not. Independent misalignment grader: ρ=0.97.

#### The capability–misalignment correlation is weaker on the expanded panel.

Pre-specified "hold or strengthen"; observed −0.16 (n=40, bootstrap p=0.40, CI [−0.54,+0.22]). CI includes −0.44; between-panel Δρ=+0.27 [−0.25,+0.78], n.s. Organization-matched subsets: median −0.17, so composition does not explain it; a generational account is possible, undemonstrated. τ² not re-collected; interaction not re-tested.

#### Models with similar capability scores differ in safety across organizations (Fig. 4).

41 models, 12 organizations (misalignment n=40; jailbreak 41 after one engagement-gate exclusion); six organizations ≥4 models (within-org), seven ≥3 (group test; Table 5). Within-org correlations vary in sign, averaging −0.14 (misalignment), +0.26 (jailbreak). After capability adjustment, groups differ (permutation p=0.004, p<0.001; Holm-significant; ε²=0.34, 0.57). Not causal: static composite may miss agentic ability, few models per org, restricted ranges. Supported: similar MMLU/GPQA ≠ similar safety.

#### Against jailbreak safety, AgentHarm shows the strongest held-out association.

StrongREJECT-scored. One template (AIM) ceilings (11/22 refuse all); three templates leave none at ceiling. Capability ρ=+0.08 (n=21, p=0.73); expanded +0.34 (n=41, p=0.04), change n.s. AgentHarm partial +0.72 (p<0.001) vs. +0.50 single-template; stable to LOO and independent grader. AgentHarm jailbreak>misalignment: Δ=+0.59, p=0.027 (model bootstrap); p=0.051 resampling eight organizations. Paired-panel misalignment partial +0.13 (n=19; Table 2 uses n=18). Shared harmful-compliance target makes +0.72 partly convergent validity. R-Judge/InjecAgent misalignment>jailbreak: marginal (Δ=+0.52, p=0.068).

#### Two alternative explanations, checked.

Seven blind LLM coders placed AgentHarm and jailbreak in harm blocking (7/7 each; α=0.61, three categories); misalignment less cleanly (A.1). AgentHarm is uncorrelated with XSTest benign over-refusal after capability control (ρ=−0.01, n=17), against blanket refusal.

## 5 Discussion

These benchmarks are not interchangeable measures of one property. AgentHarm–jailbreak is convergent validity with specificity unresolved (p=0.051); R-Judge/InjecAgent–misalignment is exploratory. Capability-adjusted organization-group differences are observational, identifying no organizational cause. A safety claim is defensible only with its benchmark, metric, target behavior, and model population attached.

## 6 Limitations and Threats to Validity

#### Panel size.

Cross-benchmark n=18 (first pass n=7); needed: a larger, capability-spanning agent-loop panel.

#### Capability anchors.

Re-measurement fixed the positive control (0.42→0.74); no RQ2/RQ3 sign changed. Anchors are static-knowledge tests, though the composite predicts τ² at +0.60. A six-model BFCL pilot failed as agentic anchor: syntax scoring ranked GPT-4o-mini highest (0.72), the two most capable lowest (0.42–0.48). Such error attenuates or suppresses partials rather than biasing them in a signable direction.

#### Multiple comparisons.

No family-wise correction; the crossover and AgentHarm→jailbreak (p<0.001) survive Bonferroni; R-Judge/InjecAgent–misalignment (p≈0.05–0.09) do not.

#### Judges and coverage.

Grader agreement ρ=0.97 (misalignment), 0.98 (jailbreak); AgentHarm refusal rescoring 0.69, partial +0.66 vs. +0.72. AgentDojo has only five models.

#### Criterion validity (RQ3).

τ² (40 tasks, one run/model) has unbounded user-simulator variance; misalignment is a fictional probe (halves Spearman–Brown 0.70). No criterion gives deployment-harm probabilities. Worst-case within-model SEs are 12–25% of between-model SD, supporting panel spread, not every pairwise difference. The larger three-scenario misalignment result is post-hoc.

## 7 Conclusion

Report the full confusion structure, drop standalone F1 wherever true negatives matter, and tie every claim to an explicit target behavior and model population.

## Appendix A Technical Appendix

### A.1 Ruling out two artifacts (detail for §4.4)

#### Blinded construct coding.

Coders: GPT-4o, Claude-3-Haiku, Gemini-2.5-Flash, Llama-3.3-70B, Qwen-2.5-72B, Mistral-Large, DeepSeek-V3, shown neutral descriptions with no criteria or RQ3 results, plus two definitions ("harm blocking" vs. "risk recognition and correct treatment of benign inputs"); options and order shuffled per coder.

| Benchmark | blind | author | B / R |
|---|---|---|---|
| R-Judge | recognition | recognition | 0/7 |
| InjecAgent | recognition | recognition | 1/6 |
| AgentHarm | blocking | blocking | 7/0 |
| AgentDojo | recognition | recognition | 0/7 |

Agreement 27/28, Krippendorff α=0.825 (nominal); chance of one coder matching all four: 0.5⁴=6.2%. Sole dissent: Llama-3.3-70B on InjecAgent. Controls coder bias, not rubric-designer bias (definitions written after seeing results); the RQ1 factor separates AgentHarm without reference to any criterion.

#### Blinded criterion coding, with a distractor.

Six targets plus a task-success distractor: α=0.61 (chance ≈1/3); distractor chosen 5/42. AgentHarm and jailbreak → harm blocking 7/7; R-Judge → recognition 7/7; InjecAgent 5/7. AgentDojo reads as task success to 4/7 (utility-scored). Misalignment: recognition 6/7 in a separate two-way round, harm blocking 6/7 here.

#### XSTest over-refusal (blanket-refusal check).

Published CPI scorer, gpt-4o-mini judge. Unsafe subset ceilinged (83–100% refusal, sd 4.5); safe-subset over-refusal ranges 0–60% (Claude-3-Haiku 60%, o3-mini 30%, others ≤15%). Capability-controlled partials:

| Predictor | unsafe refusal | safe compliance |
|---|---|---|
| AgentHarm safety | +0.24 (n=17) | +0.01 (n=17) |
| R-Judge specificity | +0.06 (n=17) | −0.15 (n=18) |
| Misalignment safety | +0.47 (n=18) | −0.11 (n=19) |

Capability: +0.21 (n.s.) benign compliance, −0.32 (n.s.) harmful refusal. Misalignment vs. over-refusal +0.11 (n=19, p=0.66); unsafe refusal p=0.052. Low power: blanket refusal neither supported nor ruled out.

### A.2 Panel coverage and power

| Analysis | n | note |
|---|---|---|
| R-Judge | 21 | Qwen3-32B dropped (21% unparseable) |
| InjecAgent | 22 | |
| AgentHarm | 19 | malformed tool calls / unsupported routing |
| AgentDojo | 5 | convergent control inconclusive |
| τ² | 20 | |
| misalignment | 21 | |
| jailbreak | 22 | 3-template average |
| expanded misalign./jailbreak | 40/41 | 12 organizations |
| XSTest | 17–20 | AgentHarm ∩ anchor overlap |
| cross-benchmark/partials | 18 | AgentHarm-limited |
| capability interaction | 20 | paired |

Evidence base (Fig. 3): up to 46 models × 9 instruments, 262 cells; rows beyond 41 excluded by telemetry gates; the 2026 expansion has anchors and the two chat criteria only.

#### Within-model score uncertainty.

Median/max SE: MMLU 1.6/2.2, GPQA 3.3/3.6, misalignment 4.0/5.0, R-Judge F1 1.8/3.0, jailbreak (~150 prompts) 2.3/3.7; between-model SDs 8.7, 18.2, 26.0, 14.9, 28.5 (expanded panel); worst-case SE 13–25% of SD, spread 4–8× worst-case SE. τ² uncoverable.

#### Internal consistency, not just precision.

| Instrument | models | items | split-half (SB) | rank stab. | quartile |
|---|---|---|---|---|---|
| R-Judge spec. | 21 | 570 | +0.98 (+0.99) | +0.99 | 0.90 |
| R-Judge F1 | 21 | 570 | +0.93 (+0.96) | +0.97 | 0.87 |
| InjecAgent | 22 | 300 | +0.95 (+0.98) | +0.98 | 0.88 |
| AgentHarm | 19 | 44 | +0.93 (+0.96) | +0.97 | 0.79 |
| Jailbreak | 42 | 20 | +0.94 (+0.97) | +0.98 | 0.96 |
| MMLU | 46 | 356 | +0.93 (+0.96) | +0.98 | 0.92 |
| GPQA | 45 | 164 | +0.95 (+0.97) | +0.98 | 0.92 |
| Misalignment (halves) | 22 | 2 | +0.54 (+0.70) | — | — |

Table 3: 50 random half-splits; item-bootstrap rank Spearman; top-quartile persistence. Items = complete-coverage intersection. Jailbreak resamples by base prompt (templates dependent); 20 of 50 prompts shared; 42 models incl. DeepSeek-R1 (no anchor pair). Misalignment: config-level blackmail-vs-leaking halves only. Item-sampling error is an unlikely cause of Table 1 disagreement.

Power for ρ=0.4 at α=0.05, n=18–20: ≈0.35 (simulated); 80%-power MDE ρ=0.62 at n=18; detecting +0.41 needs n≈40. Bonferroni over nine RQ3 partials: α=0.006.

### A.3 Cross-benchmark structure (detail for §4.3)

In Fig. 5, top models on either axis span the other's full range. PC sign is arbitrary; only relative directions matter.

#### Correlation-convention sensitivity.

Spearman PCA: anchors −0.72/−0.78, AgentHarm +0.38; positive control passes; PC1 48.7%→40.8%; ≥40% criterion holds under both. Negative control: 200 permuted matrices, median first eigenvalue 1.71 vs. 95th-percentile threshold 2.18 (exceedance 2%).

### A.4 R-Judge metric sensitivity (detail for §4.3–§4.4)

| R-Judge metric | RQ1 reversal | PC1 | MMLU | GPQA | τ² | misalign. | jailbr. |
|---|---|---|---|---|---|---|---|
| specificity | +0.02 | −0.71 | +0.16 | +0.07 | −0.09 | +0.41 | −0.11 |
| recall | −0.05 | −0.14 | +0.40 | +0.32 | +0.15 | −0.46 | −0.05 |
| balanced acc | −0.09 | −0.86 | +0.71 | +0.49 | +0.19 | +0.19 | −0.20 |
| F1 (official) | −0.29 | −0.75 | +0.76 | +0.62 | +0.32 | −0.29 | −0.17 |

Table 4: RQ1 n=18, RQ2 n=20; RQ3 partials (τ² n=19, misalignment 18, jailbreak 20). Only the misalignment cell is metric-sensitive (both extremes n.s.).

### A.5 Robustness of the capability crossover (detail for §4.4)

#### Partial-correlation convention.

Midranks residualized on the composite, residuals re-ranked. Pearson on rank residuals: R-Judge +0.49, InjecAgent +0.41 (within 0.08), AgentHarm–jailbreak +0.72.

Crossover: leave-one-out Δ∈[−1.16,−0.90]; leave-one-organization-out: Amazon −0.96, Anthropic −1.01, Cohere −0.90, DeepSeek −1.13, Google −0.98, Meta −0.96, Mistral −0.93, OpenAI −1.16, Qwen −0.95. Organization-clustered bootstrap Δ∈[−1.49,−0.59], P(Δ≥0)<10⁻⁴. Subsamples (median/% negative/% ≤−0.5/% flips): size-10 −0.95/100/97/0; size-12 −0.96/100/99/0; size-14 −0.97/100/100/0. Graders: misalignment Claude-3-Haiku vs. GPT-4o; jailbreak Gemini vs. gpt-4o; AgentHarm Gemini refusal judge on cached completions (n=19), redoing only the refusal component. Per-template AgentHarm→jailbreak partials: AIM +0.44, refusal-suppression +0.70, prefix-injection +0.62.

#### Misalignment 3-scenario battery (exploratory).

Adding murder (22 models) moves the interaction to Δ=+0.53 (p=0.048): tighter CI, same effect; a post-hoc forking path. AgentHarm→battery +0.43 dilutes specificity; blackmail+leaking stays primary.

### A.6 The 2026 model-panel expansion (n ≈ 41) and organization-group differences (detail for §4.4)

Same chat/scenario harnesses; behavior-only exclusions: Kimi-K2.6 and both Nemotron-3 (<50% usable anchor answers or unbounded reasoning); MiniMax-M3 below misalignment engagement threshold (truncation), kept for jailbreak; Gemini-2.5-Pro no valid anchor; GLM-4.5 MMLU only (routing), partial row.

#### Primary expansion result: the capability–misalignment correlation weakens.

Original bootstrap p=0.11, CI [−0.85,+0.09]; expanded leave-one-out [−0.25,−0.12]. Confirmation threshold (ρ≤−0.35, p<0.05) not met. Nested bootstrap p=0.29 (jailbreak Δρ +0.26, [−0.14,+0.69], p=0.22); 2026-only models ρ=+0.07 vs. −0.44 (Fisher z p=0.12); organization-matched size-21 IQR [−0.27,−0.08]. At composite ≈0.8: o3-mini 0.99, Grok-4.3 0.70, DeepSeek-V3 0.19, so ceilings do not explain all variation.

#### Organization-group analysis.

All five secondary tests survive Holm (max adjusted p=0.043). Within-org n 4–8; Anthropic ceilings near 1.0; the permutation test, not slopes, carries the claim. Calibration (2,000 synthetic null panels each): reshuffled or Gaussian safety gives 0.045–0.052 at α=0.05 (nominal at 0.01/0.005); clustered-variance null inflates jailbreak only (0.067; none at α=0.005); doubled p-values stay Holm-significant. Without Anthropic: misalignment ε²=0.27, p=0.018; jailbreak ε²=0.54, p<0.001.

| | n | misalign. ρ | jailbreak ρ |
|---|---|---|---|
| Anthropic | 5 | +0.71 | +0.67 |
| OpenAI | 8 | +0.59 | +0.64 |
| Qwen | 5 | −0.90 | +0.80 |
| Meta | 4 | −0.40 | −0.40 |
| Google | 4 | −0.60 | −0.40 |
| Mistral | 4 | −1.00 | −0.40 |
| n-weighted mean | | −0.14 | +0.26 |
| group test (n≥3; total 33): between-group share | | 0.508 | 0.674 |
| df-adjusted ε² | | 0.34 | 0.57 |
| permutation p (10⁴) | | 0.004 | <0.001 |
| Kruskal–Wallis p | | 0.02 | 0.002 |

Table 5: Within-organization slopes (top) and capability-residualized group differences (bottom); ε² is an effect size, not a causal share.

#### Expanded-panel jailbreak result.

Templates re-scored for all new models. Bootstrap p=0.035, CI [+0.03,+0.60], leave-one-out [+0.30,+0.39]. High-capability jailbreak-safety: Mistral-Large 0.09, DeepSeek-V3 0.37, Anthropic/OpenAI 0.98–0.99. Neither sign nor magnitude of capability→safety is stable across criteria or panels.

A pre-specified R-Judge–misalignment continuation fell from +0.41 (n=18) to +0.23 (n=25), below its 0.30 threshold; stopped, kept as hypothesis.