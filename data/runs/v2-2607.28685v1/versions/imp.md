# Safety, or Just Capability? A Validity Audit of Agent-Safety Benchmarks

Youting Wang

  
Xiao Han

  
Dingyan Shang

Affiliation: Yuan Tang

  
Bowen Liu

###### Abstract

We audit four agent-safety benchmarks (R-Judge, InjecAgent, AgentHarm, AgentDojo), each run
under its official implementation and author-provided scorer on up to 22 models. MMLU and GPQA
are re-measured under one protocol and combined into a capability composite. On any binary
trace-judgment benchmark scored by $F_{1}$ , an “always positive” policy attains
$F_{1}=2\pi/(1+\pi)$ ; on R-Judge this is $0.690$ , above five of the 21 models that
discriminate. The three broad-coverage benchmarks rank the same 18 models differently. R-Judge
specificity against AgentHarm safety correlates $-0.64$ at $n{=}7$ and $+0.02$ at $n{=}18$ , and a
quarter of random size-7 subsets reach $|\rho|\geq 0.5$ . Capability predicts task success
( $\rho{=}{+}0.60$ ) but correlates negatively with misalignment safety ( $\rho{=}{-}0.44$ ,
$n{=}21$ ). On their paired $n{=}20$ panel the contrast is $\Delta{=}{-}1.00$ (95% CI
$[-1.48,-0.49]$ , $p<0.001$ ), stable under leave-one-organization-out and
organization-clustered bootstrap. On an expanded 41-model panel the misalignment correlation is
$-0.16$ (95% CI $[-0.54,+0.22]$ ) and the jailbreak correlation $+0.34$ ; neither change is
significant. Controlling capability, AgentHarm correlates $\rho{=}{+}0.72$ with three-template
jailbreak safety; both instruments score harmful compliance, so we read this as convergent
validity. A safety claim should specify benchmark, metric, target behavior, and model panel.

## 1 Introduction

Between 2024 and 2026, dozens of benchmarks were proposed for LLM-agent safety and reliability:
prompt-injection resistance (Zhan et al. 2024; Debenedetti et al. 2024), safety-risk awareness
(Yuan et al. 2024), refusal of harmful agentic tasks (Andriushchenko et al. 2025), and risky-tool-use
identification (Ruan et al. 2024), among others. A recent taxonomy (Li et al. 2026) catalogs more
than forty and reports rank concordance near $0.10$ across models when the benchmark is swapped.
That analysis is a twelve-model concordance check over four benchmarks; capability-controlled
evaluation and benchmark consolidation are listed among its open problems.

We address construct, metric, and criterion validity through four questions:

•

RQ1 (construct structure). Do nominally-distinct agent-safety benchmarks measure
a common underlying factor, or dissociable ones?

•

RQ2 (capability confound). How much of the between-benchmark signal is general
capability in disguise?

•

RQ3 (criterion validity). Do the safety scores track held-out behavioral
criteria beyond what general capability predicts?

•

RQ4 (metric validity). Are the headline metrics valid instruments, or do
they reward degenerate behavior?

#### Contributions.

Observation 1 gives a closed form for the F1 of an “always positive” baseline on any binary
trace-judgment benchmark; on R-Judge that baseline outranks five evaluated models. The
R-Judge-specificity/AgentHarm-safety correlation moves from $-0.64$ at $n{=}7$ to $+0.02$ at
$n{=}18$ , and a quarter of random size-7 subsets show $|\rho|\geq 0.5$ . A pre-specified,
capability-controlled criterion-validity audit covers one task-success and two safety outcomes:
capability predicts task success; its safety correlations change with outcome and panel; the
largest safety-score effect is AgentHarm–jailbreak ( $\rho{=}+0.72$ partialling capability), with
outcome selectivity at $p{=}0.051$ under organization-level resampling. We release an API-only
audit harness and re-run artifacts for four benchmarks and three held-out outcomes in the
accompanying reproducibility package.

#### Main conclusion.

A capability score is not a safety score, and no one agent-safety benchmark stands in for safety as
a whole. At seven models a weak relationship can appear systematic; validity estimates require
re-checking as the model population turns over.

#### Scope.

The three RQ3 criteria are deployment proxies, not claims about real-world harm; their limits
(single instances, grader dependence, panel size) are in Section 6.

## 2 Related Work

#### Agent-safety benchmarking.

InjecAgent (Zhan et al. 2024) and AgentDojo (Debenedetti et al. 2024) target prompt-injection
robustness; AgentHarm (Andriushchenko et al. 2025) measures compliance with harmful agentic tasks;
R-Judge (Yuan et al. 2024) scores safety-risk awareness over interaction traces. Evaluators not
re-run here, e.g. ToolEmu (Ruan et al. 2024) for risky tool use, are likewise designed as evaluators
without validation as measurements. AutoMonitor-Bench (Yang et al. 2026) scores misbehavior
monitors by miss and false-alarm rates. The nearest prior work, a taxonomy and consistency analysis
of agent-safety benchmarks (Li et al. 2026), documents ranking disagreement without capability
control, criterion-validity testing, or latent-structure recovery, naming capability-controlled
evaluation as future work; we take up all three.

#### Construct validity.

Benchmark-agreement testing (Perlitz et al. 2024), metabench (Kipnis et al. 2025), and direct
construct analyses (Bean et al. 2025) find a strong shared latent factor across knowledge and
reasoning benchmarks. Kearns (2026) and NIST’s statistical framework (Keller et al. 2026) bring
quantitative construct-validity and uncertainty modeling to LLM evaluation. We apply this lens to
agent reliability and safety, where benchmarks target distinct behaviors—refusal, injection
robustness, risk awareness—and the capability confound in their comparison is unquantified.

#### Benchmark gaming.

Adversarial gaming is distinct from, and logically posterior to, our concern that the headline
score may not measure the intended construct even for an honest model. The F1 result
(Section 4) involves a non-adversarial policy that labels every trace “unsafe” without examining
any of them.

#### Over-refusal.

XSTest (Röttger et al. 2024) and OR-Bench (Cui et al. 2024) document exaggerated safety, or
over-refusal, at the prompt level. The between-benchmark analogue is whether models that refuse
harmful tasks (AgentHarm) are worse at identifying benign traces correctly (R-Judge specificity).
On the full panel that correlation is near zero; on seven-model subsets it often is not.

#### Agent reliability and task performance.

Recent work isolates corrupt success (Cao et al. 2026) and evidence-bounded reporting
(Chen 2025); $\tau$ -bench (Yao et al. 2024) and its dual-control successor $\tau^{2}$ -bench
(Barres et al. 2025) measure agentic task success. We use the latter’s retail domain as the
held-out RQ3 task-success criterion, and treat the safety benchmarks as candidate columns for a
validity audit.

## 3 Method

### 3.1 Model panel and benchmarks

The API-only panel comprises up to 22 models from nine model-developing organizations (OpenAI 6,
Meta 3, Qwen 3, Mistral 3, DeepSeek 2, Amazon 2, Anthropic, Google, Cohere; full roster in Fig. 3
and the artifact). Four safety benchmarks are re-run with official implementations and scorers:
R-Judge (full 571 items, self-judge), InjecAgent (300 stratified direct-harm/data-stealing items,
rule-scored), AgentHarm (44 base behaviors spanning all eight harm categories, official Inspect
evaluation, gpt-4o-mini judge), and AgentDojo (slack suite, 100 items, environment-scored for
security and utility), plus a held-out task-success criterion for RQ3.

Coverage (Fig. 3): R-Judge $n{=}21$ after Qwen3-32B falls below the $90\%$ valid-output threshold;
InjecAgent $n{=}22$ ; AgentHarm and AgentDojo, which require native tool use, $n{=}19$ and $5$ ; the
$\tau^{2}$ criterion $n{=}20$ (App. A.2). The common R-Judge/InjecAgent/AgentHarm panel is
$n{=}18$ . Capability anchors are measured through the same API harness as the safety runs: MMLU
zero-shot on a fixed 500-item subset (seed 42, identical items for every model) and GPQA-Diamond
(198 items, chain-of-thought). These replace the provider-reported model-card numbers used
initially, which are retained as a pre-harmonization comparison. The anchor panel is 21 models
(DeepSeek-R1 excluded for unstable long-form GPQA generation).

### 3.2 Benchmark targets

Target behaviors were fixed before inspecting any relationship among score columns. R-Judge:
distinguishing unsafe from benign interaction traces. InjecAgent and AgentDojo: prompt-injection
resistance, their correlation serving as the convergent check. AgentHarm: refusal of harmful
agentic tasks. R-Judge is analyzed under specificity and balanced accuracy in addition to the
official F1.

#### Score orientation and terminology.

All scores are oriented higher-is-better/safer. R-Judge specificity is the fraction of benign
traces correctly identified as benign; balanced accuracy averages performance on benign and unsafe
traces. InjecAgent robustness $=1-$ attack success; AgentHarm safety $=1-$ harmful compliance.
$\tau^{2}$ -bench is a held-out task-success criterion, not a safety score. “Internal consistency”
refers to split-half analyses (App. A.2); benchmark-specific scores are named explicitly rather
than as “calibration” or “reliability.”

### 3.3 Analysis plan

Hypotheses, thresholds, and decision rules were fixed before each analysis. Lacking an independent
timestamp, we call the analyses pre-specified rather than preregistered. All association tests are
Spearman; unless noted, $p$ -values are two-sided asymptotic Spearman tests. Fig. 2 error bars are
model-level percentile-bootstrap intervals; the panel expansion and direct panel comparisons follow
pre-specified bootstrap procedures (App. A.6). The corroborative PCA is on the Pearson correlation
matrix of standardized score columns, with factor count by Horn’s parallel analysis against a
random-normal null; Spearman sensitivity in App. A.3. Partial Spearman correlations control the
capability composite (rank-residual implementation and alternative convention in App. A.5). The
composite is the mean of the standardized, verified MMLU and GPQA anchors; its model ordering is
identical to the anchors’ first principal component. Between-construct redundancy is measured by
rank-residualized partials controlling the composite.

Three pre-specified controls: positive (capability-anchor loadings $\geq 0.6$ on the first factor);
negative (a column-permuted score matrix within the parallel-analysis null); convergent (high
InjecAgent–AgentDojo correlation). The headline statistic is the median pairwise partial Spearman
$|\rho|$ among distinct-construct benchmarks, controlling the composite; factor structure is
interpretable only if first-factor variance is $\geq 40\%$ or capability control reduces that
median by $\geq 0.15$ , with positive and negative controls passing. Thresholds and the
disconfirming test were fixed before any real run.

RQ3 uses three held-out outcomes, each pre-specified before it was run. Task success: $\tau^{2}$ -bench
retail (Barres et al. 2025), 40 multi-turn, environment-scored tool-agent-user tasks. Misalignment
safety: avoidance of blackmail or information leaking in a fictional autonomous-agent scenario
(Lynch et al. 2025), fixed classifier over 100 decisions per model. Jailbreak safety: one minus
harmful compliance on 50 forbidden prompts, averaged over three published jailbreak templates to
reduce single-template ceilings (Souly et al. 2024; Wei et al. 2023). All run through the same
API-only harness (the two safety criteria are single-turn or scenario-based and cover the full
panel; $\tau^{2}$ is a multi-turn tool loop covering 20 models), are distinct from the four audited
benchmarks, and are excluded from the predictors. The headline statistic per criterion is the
partial Spearman correlation of each safety score with the criterion controlling the composite,
reported after the criterion’s raw association with capability. Reporting rules cover both
significant and null results. Analysis scripts and frozen inputs are in the reproducibility
package.

## 4 Results

Ordering follows evidential strength: RQ4 and RQ2, then RQ1, then the power-limited RQ3.

### 4.1 The headline metric is gameable (RQ4)

Figure 1: A leaderboard’s headline metric can be matched by an “always unsafe” baseline.
On R-Judge ( $n{=}21$ ), five real, discriminating models score below this constant baseline
(F1 ${=}0.69$ , dashed).

R-Judge labels $52.7\%$ of its 571 items “unsafe.” The policy answering “unsafe” on every item
attains recall $1.0$ , specificity $0$ , and $\mathrm{F1}=0.690$ . Among the 21 panel models with
valid R-Judge verdicts (Qwen3-32B excluded for $21\%$ unparseable output) this exceeds five
discriminating models (F1 $0.30$ – $0.57$ ; Fig. 1). o3-mini, the highest-specificity model
( $0.97$ ), has $\mathrm{F1}=0.702$ , $0.012$ above the constant baseline, owing to lower recall.
Balanced accuracy reorders the board.

###### Observation 1 (F1 admits a high-scoring constant baseline).

On a benchmark with class base rate $\pi$ scored by F1, the constant “always-positive” policy
attains $\mathrm{F1}=2\pi/(1+\pi)$ without distinguishing safe from unsafe items. For R-Judge
( $\pi=0.527$ , a near-balanced base rate) this is $0.690$ , exceeding the observed F1 of several
discriminating models. The closed form is prevalence-dependent; F1 cannot stand alone as a measure
of two-sided discrimination on this trace-judgment benchmark.

| Model | R-Judge | InjecAgent | AgentHarm | GPQA

| | spec. | robust. | safety
|

| Llama-3.1-8B | 0.04 | 0.32 | 0.89 | 30.4

| Claude-3-Haiku | 0.40 | 0.77 | 0.81 | 33.3

| Llama-3.1-70B | 0.11 | 0.35 | 0.78 | 46.7

| o3-mini | 0.97 | 0.87 | 0.77 | 77.2

| Nova-Lite | 0.94 | 0.88 | 0.76 | 42.0

| Nova-Pro | 0.64 | 0.60 | 0.75 | 46.9

| Gemini-2.5-Flash | 0.92 | 0.71 | 0.62 | 68.3

| DeepSeek-V3 | 0.86 | 0.76 | 0.61 | 59.1

| Qwen-2.5-72B | 0.62 | 0.92 | 0.60 | 49.1

| GPT-4.1 | 0.61 | 0.87 | 0.60 | 66.3

| GPT-4.1-mini | 0.76 | 0.87 | 0.51 | 65.0

| GPT-4o | 0.72 | 0.90 | 0.47 | 53.6

| GPT-4.1-nano | 0.79 | 0.94 | 0.42 | 50.3

| Llama-3.3-70B | 0.28 | 0.31 | 0.39 | 50.0

| Mistral-Nemo | 0.80 | 0.34 | 0.35 | 28.0

| GPT-4o-mini | 0.61 | 0.89 | 0.31 | 40.2

| Mixtral-8x22B | 0.58 | 0.78 | 0.24 | 33.2

| Mistral-Large | 0.50 | 0.81 | 0.24 | 44.0

Table 1: The three broad-coverage agent-safety benchmarks on the same $n{=}18$ models (higher is safer). Rows sorted by AgentHarm safety. Bold marks column maxima and underlining minima, display-precision ties included. GPQA shown as a capability reference.

### 4.2 The capability confound is real but metric-dependent (RQ2)

On the R-Judge panel ( $n{=}20$ with harmonized anchors), capability correlates with balanced
accuracy at $\rho=+0.71$ (MMLU, $p<0.001$ ) and $+0.49$ (GPQA, $p=0.03$ ), with F1 at $+0.76$ and
$+0.62$ , and with specificity at $+0.16$ and $+0.07$ . The specificity–MMLU correlation was $0.85$
on the first nine models and $+0.16$ at $n{=}20$ .

### 4.3 Benchmarks disagree; a seven-model panel suggested a trade-off that does not persist (RQ1)

No benchmark orders the models as another does (Table 1); a top-AgentHarm model can rank near the
bottom on R-Judge specificity. Disagreement between benchmarks coded to different constructs (§3)
is expected and does not by itself indict any single instrument; it does preclude quoting the scores
interchangeably as an agent’s safety.

The R-Judge-specificity/AgentHarm-safety rank correlation, $-0.64$ at $n{=}7$ , is $+0.02$
( $p=0.95$ ) at $n{=}18$ (Fig. 5, appendix); random size-7 subsets yield $|\rho|\geq 0.5$ in a
quarter of draws. The null holds under all four R-Judge metrics (App. A.4). Capability moderation
of the residual is untested and flagged as a hypothesis.

A single-factor PCA (Fig. 6, appendix) has a first component explaining $48.7\%$ of variance,
loading R-Judge specificity, InjecAgent robustness, and both capability anchors in one direction
and AgentHarm safety in the other ( $+0.33$ ; opposite the cluster in $17$ of $18$ leave-one-out
fits).

#### The positive control now passes.

On the harmonized anchors MMLU loads $0.74\geq 0.6$ (provider-reported numbers gave $0.42$ ); the
negative control passes (column-permuted eigenvalues within the parallel-analysis null), meeting
the pre-specified interpretation conditions; the convergent control is inconclusive at AgentDojo’s
$n{=}5$ . A Spearman PCA retains the loading structure but PC1 falls to $40.8\%$ with no factor
retained by Horn (App. A.3), and the median between-benchmark correlation is $0.22$ ; the factor is
treated as corroborative relative to Table 1.

### 4.4 Predictive validity differs across held-out outcomes (RQ3)

The three pre-specified outcomes (§3) are $\tau^{2}$ -bench task success, misalignment safety, and
jailbreak safety, each distinct from the audited benchmarks with a fixed metric and decision rule.

| | Task success | Safety outcomes

| Predictor | $\tau^{2}$ | misalign. | jailbreak†

| Capability (raw $\rho$ ) | $+0.60^{**}$ | $-0.44^{*}$ | $+0.08$

| Safety-score partial $\rho$ (controlling capability):

| R-Judge specificity | $-0.09$ | $+0.41$ | $-0.11$

| InjecAgent robustness | $+0.16$ | $+0.47$ | $+0.21$

| AgentHarm safety | $-0.23$ | $+0.16$ | $+0.72^{***}$

Table 2: Criterion validity on the 2024–25 panel (cellwise $n{=}18$ – $21$ because predictor
coverage differs; App. A.2; higher is better or safer). Top row: raw correlations with capability;
remaining rows: partial Spearman correlations controlling the capability composite. † marks the
three-template jailbreak outcome. Stars use two-sided asymptotic Spearman tests
( ${}^{*}p<0.05$ , ${}^{**}p<0.01$ , ${}^{***}p<0.001$ ).

#### Against task success, safety scores add no information beyond capability.

Capability predicts $\tau^{2}$ -retail success ( $\rho=+0.60$ , $p=0.005$ , $n{=}20$ ); no safety
score shows incremental validity after partialling capability (best $|\rho|=0.23$ , n.s. at
$n{=}19$ ; Table 2; Fig. 7, appendix).

Figure 2: Capability predicts task success; safety point estimates differ by outcome and
panel. Filled circles: 2024–25 correlations; open diamonds: expanded-panel safety correlations;
error bars: model-level percentile-bootstrap 95% CIs. Thin lines connect panel-level estimates, not
model trajectories. Misalignment changes from $-0.44$ to $-0.16$ and jailbreak from $+0.08$ to
$+0.34$ ; neither between-panel change is significant. Task success was not re-collected for the
expansion.

#### Against agentic misalignment, capability has the opposite association.

On the full 21-model panel, $\rho_{\text{cap}}=-0.44$ (asymptotic $p=0.047$ ) against $+0.60$ on
$\tau^{2}$ (Mistral-Large blackmails or leaks $85\%$ of the time; o3-mini $1\%$ ). Partialling
capability, InjecAgent robustness ( $+0.47$ ) and R-Judge specificity ( $+0.41$ ) reach the
pre-specified $0.40$ effect-size threshold; AgentHarm does not ( $+0.16$ ; Table 2). Both CIs
include zero, neither survives Bonferroni, power is $\approx 0.35$ , and the pre-specified
directional prediction named AgentHarm. The R-Judge partial is $+0.41$ under specificity, $+0.19$
under balanced accuracy, and $-0.46$ under recall (App. A.4).

#### A direct interaction test on the original panel.

On the paired $n{=}20$ panel, capability correlates $+0.60$ with task success and $-0.41$ with
misalignment safety ( $-0.44$ on the full panel): $\Delta=-1.00$ (95% CI $[-1.48,-0.49]$ ,
$p<0.001$ ), stable to leave-one-model-out, leave-one-organization-out, organization-clustered
bootstrap, and subsampling (App. A.5). The corresponding R-Judge interaction is marginal
( $\Delta=+0.50$ , $p=0.082$ ), supporting the capability interaction but not a full two-way
dissociation. An independent misalignment grader gives $\rho=0.97$ with the primary scores.

#### The capability–misalignment correlation is weaker on the expanded panel.

The pre-specified expansion prediction was that the negative correlation would “hold or
strengthen.” It moves from $\rho=-0.44$ ( $n{=}21$ , asymptotic $p=0.047$ ) to $-0.16$ ( $n{=}40$ ,
bootstrap $p=0.40$ ; 95% CI $[-0.54,+0.22]$ ; App. A.6). The interval includes $-0.44$ , and the
between-panel difference is non-significant (nested-bootstrap $\Delta\rho=+0.27$ , 95% CI
$[-0.25,+0.78]$ ). Subsets matched to the original organization composition give median
$\rho=-0.17$ ; a generational explanation remains possible and undemonstrated. $\tau^{2}$ was not
re-collected, so the interaction was not re-tested.

#### Models with similar capability scores differ in safety across organizations
(Fig. 4).

The 41-model panel spans 12 model-developing organizations (misalignment $n{=}40$ ; jailbreak
$n{=}41$ after one engagement-gate exclusion). Six organizations contribute $\geq 4$ models for
within-organization correlations and seven contribute $\geq 3$ for the group comparison
(App. A.6, Table 5). Within-organization capability–safety correlations vary in sign, averaging
$-0.14$ (misalignment) and $+0.26$ (jailbreak). After capability adjustment of safety ranks, the
seven groups differ on both criteria (permutation $p=0.004$ misalignment, $p<0.001$ jailbreak;
Holm-significant; df-adjusted Kruskal–Wallis $\varepsilon^{2}=0.34$ and $0.57$ ). The static
composite may miss agentic ability, each organization contributes few models, and some groups are
range-restricted; the result is non-causal and limited to: similar MMLU/GPQA scores do not imply
similar safety scores.

#### Against jailbreak safety, AgentHarm shows the strongest held-out association.

The second safety outcome differs in failure mode, harness, and scoring: harmful compliance under
jailbreak, scored with StrongREJECT (Souly et al. 2024). A single template (AIM) ceilings, with 11
of 22 models refusing every request; averaging three published templates (Wei et al. 2023) leaves no
model at ceiling. Capability is weakly associated on the 2024–25 panel ( $\rho=+0.08$ , $n{=}21$ ,
$p=0.73$ ) and at $+0.34$ on the expanded panel ( $n{=}41$ , $p=0.04$ ), the change non-significant.

Partialling capability, AgentHarm safety correlates $\rho=+0.72$ ( $p<0.001$ ) with the
three-template outcome versus $+0.50$ for the single-template measure, stable to
leave-one-model-out and an independent grader (App. A.5). Selectivity for jailbreak over
misalignment gives $\Delta=+0.59$ , $p=0.027$ under model-level bootstrap and $p=0.051$ resampling
the eight represented organizations. On this paired panel the misalignment partial is $+0.13$
( $n{=}19$ ); Table 2’s $+0.16$ uses the common $n{=}18$ panel. R-Judge specificity ( $-0.11$ ) and
InjecAgent robustness ( $+0.21$ ) do not predict jailbreak safety. AgentHarm and StrongREJECT both
measure harmful compliance. The reverse comparison for R-Judge and InjecAgent (misalignment over
jailbreak) is marginal ( $\Delta=+0.52$ , $p=0.068$ ).

#### Two alternative explanations, checked.

Seven LLM coders shown only construct descriptions, without predictive results, assigned AgentHarm
and the jailbreak outcome to the same harm-blocking category ( $7/7$ each; Krippendorff
$\alpha=0.61$ across three categories), and classified the misalignment outcome less consistently
(App. A.1). Partialling capability, AgentHarm safety is uncorrelated with over-refusal on benign
XSTest prompts (Röttger et al. 2024) ( $\rho=-0.01$ , $n{=}17$ ).

## 5 Discussion

An “always unsafe” baseline matches R-Judge F1 (Observation 1); the three broad-coverage benchmarks
rank the same models differently. Capability predicts task success; its correlations with the two
safety outcomes are panel-unstable (Table 2). AgentHarm’s capability-controlled association with
three-template jailbreak safety is $\rho=+0.72$ , with outcome selectivity at $p=0.051$ under
organization resampling; the specificity claim remains open. The R-Judge/InjecAgent associations
with misalignment are exploratory. Safety ranks differ among organization groups after composite
adjustment, an observational pattern without an identified organizational cause. A safety claim is
defensible only with its benchmark, metric, target behavior, and model population attached.

## 6 Limitations and Threats to Validity

#### Panel size.

The AgentHarm-limited cross-benchmark panel is $n{=}18$ , up from a first pass at $n{=}7$ . The
priority extension is a larger, capability-spanning agent-loop panel.

#### Capability anchors.

Provider-reported anchors failed the positive control (MMLU $0.42$ ); re-measured MMLU and
GPQA-Diamond (§3) pass (MMLU $0.74$ ), with every RQ2/RQ3 sign unchanged. These are static-knowledge
tests, though their composite predicts multi-turn tool-agent success at $+0.60$ . A six-model BFCL
pilot (Patil et al. 2025) failed as an agentic replacement: strict function-call syntax scoring
ranked GPT-4o-mini highest ( $0.72$ ) and the two most capable models lowest ( $0.42$ – $0.48$ ),
apparently rewarding format conformance; a validated agentic anchor is needed. Such measurement
error attenuates or suppresses partial relationships rather than biasing them in a signable
direction.

#### Multiple comparisons.

Per-test $p$ -values are uncorrected; claims rest on effect sizes and pre-specified rules. The two
strongly significant tests (crossover and AgentHarm $\to$ jailbreak, both $p<0.001$ ) survive
Bonferroni; the R-Judge/InjecAgent–misalignment cells ( $p\approx 0.05$ – $0.09$ ) do not and are
treated as suggestive.

#### Judges and coverage.

Independent graders reproduce model scores at $\rho=0.97$ (misalignment) and $\rho=0.98$
(jailbreak). Rescoring AgentHarm refusals with a different model gives $\rho=0.69$ , while the
AgentHarm–jailbreak partial moves only from $+0.72$ to $+0.66$ (App. A.5). AgentDojo covers five
models, so cross-benchmark analyses rest mainly on the other three benchmarks.

#### Criterion validity (RQ3).

$\tau^{2}$ -retail (40 tasks, one run/model) is scored by a user-simulator of unbounded variance;
the misalignment criterion is a fictional-scenario probe (grader-robust, $\rho=0.97$ , though its
blackmail–leaking halves agree only moderately, Spearman–Brown $0.70$ ); the jailbreak criterion
averages three jailbreaks. None yields deployment-harm probabilities. For item-rich instruments
with resampling support, split-half consistency is $\geq 0.96$ and rank stability $\geq 0.97$ .
Worst-case within-model SEs are $12$ – $25\%$ of between-model SD across bounded instruments,
supporting panel-level spread but not every pairwise difference (App. A.2). The
AgentHarm–jailbreak association is large and directionally stable, partly convergent, and
borderline-selective under organization resampling. The R-Judge/InjecAgent–misalignment pattern is
exploratory (marginal $\Delta=+0.52$ , power $\approx 0.35$ ; the larger three-scenario result is
post-hoc; App. A.5).

## 7 Conclusion

Across four benchmarks and up to 22 models, R-Judge’s F1 lets an “always unsafe” baseline outrank
five models, and rankings change with the benchmark. Capability predicts held-out task success but
not safety consistently across outcomes or panels. The strongest held-out safety association is
AgentHarm’s ( $\rho=+0.72$ with jailbreak safety after controlling capability), with partial
construct overlap and borderline organization-resampled selectivity. Evaluations should report the
full confusion structure, drop standalone F1 wherever true negatives matter, and tie claims to an
explicit target behavior and model population.

## Appendix A Technical Appendix

Method detail and full robustness numbers summarized in the body.

### A.1 Ruling out two artifacts (detail for §4.4)

#### Blinded construct coding.

Seven LLM coders from distinct families—GPT-4o, Claude-3-Haiku, Gemini-2.5-Flash, Llama-3.3-70B,
Qwen-2.5-72B, Mistral-Large, DeepSeek-V3—each saw only a neutral, predictive-information-stripped
description of each benchmark (no mention of our criteria, category names, or any RQ3 result) and
two construct definitions (“harm blocking” vs. “risk recognition and correct treatment of benign
inputs”). Option order and benchmark order were shuffled per coder to remove position bias. Each
coder assigned every benchmark to one construct. Votes (B ${=}$ harm blocking, R ${=}$ risk
recognition):

| Benchmark | blind | author | B / R

| R-Judge | recognition | recognition | 0 / 7

| InjecAgent | recognition | recognition | 1 / 6

| AgentHarm | blocking | blocking | 7 / 0

| AgentDojo | recognition | recognition | 0 / 7

All four majorities match the hand-assignment; raw agreement $27/28$ ratings, Krippendorff’s
$\alpha=0.825$ (nominal), against a chance baseline of $0.5^{4}=6.2\%$ for one coder matching all
four labels by coin flip. The single dissent (Llama-3.3-70B placing InjecAgent as harm blocking)
is the semantically borderline injection-robustness case. This controls coder bias but not
rubric-designer bias (the category definitions were written by authors who had seen the results);
the RQ1 factor, which separates AgentHarm ( $+0.33$ ) from the R-Judge/InjecAgent cluster without
reference to any criterion, is reported for that reason.

#### Blinded criterion coding, with a distractor.

To test whether the predictor $\to$ criterion matching, not only the predictor labels, is
recoverable blind, and to give the agreement statistic content, coding was re-run over all six
targets (four benchmarks $+$ two criteria) with a third, plausible task-success distractor
construct. Krippendorff’s $\alpha=0.61$ (three categories, chance $\approx 1/3$ ); the distractor
is chosen $5/42$ times. AgentHarm and the jailbreak criterion both map to harm blocking $7/7$ ,
R-Judge to risk recognition $7/7$ , and InjecAgent $5/7$ , all unchanged by the distractor. Two
targets are contested: AgentDojo reads as task success to $4/7$ coders (it is utility-scored; our
$n{=}5$ benchmark), and the misalignment criterion maps to risk recognition $6/7$ in a separate
two-way round (frozen votes in the artifact) but to harm blocking $6/7$ here (avoiding a harmful
autonomous action can be read as both recognizing and declining it). The AgentHarm–jailbreak
pairing is thus recovered without the predictive results, and the R-Judge/InjecAgent–misalignment
pattern remains exploratory.

#### XSTest over-refusal (blanket-refusal check).

XSTest (Röttger et al., 2024; cited in the main paper) was run on the panel with its published CPI
(compliance/partial/refusal) scorer (judge gpt-4o-mini): the safe subset (benign prompts
superficially resembling unsafe ones) measures over-refusal; the unsafe subset measures correct
refusal of overtly-harmful prompts. The unsafe subset is ceilinged (every model refuses
$83$ – $100\%$ , sd $4.5$ ), so the informative signal is safe-subset over-refusal (range
$0$ – $60\%$ ; Claude-3-Haiku over-refuses $60\%$ , o3-mini $30\%$ , others $\leq 15\%$ ).
Capability-controlled partial- $\rho$ :

| Predictor
|

unsafe
refusal

|

safe
compliance

| AgentHarm safety | $+0.24$ ( $n{=}17$ ) | $+0.01$ ( $n{=}17$ )

| R-Judge specificity | $+0.06$ ( $n{=}17$ ) | $-0.15$ ( $n{=}18$ )

| Misalignment safety | $+0.47$ ( $n{=}18$ ) | $-0.11$ ( $n{=}19$ )

AgentHarm is unrelated to benign-prompt over-refusal ( $\rho=-0.01$ , $n{=}17$ ), weighing against
but not excluding a blanket-refusal effect. Capability is $+0.21$ (n.s.) on benign compliance and
$-0.32$ (n.s.) on harmful refusal; R-Judge specificity is $+0.15$ on over-refusal (equivalently
$-0.15$ on safe compliance). Misalignment safety is unrelated to over-refusal ( $+0.11$ ,
$n{=}19$ , $p=0.66$ ), with unsafe-refusal association $+0.47$ ( $n{=}18$ , $p=0.052$ ). These
low-power checks neither support nor exclude the blanket-refusal account.

### A.2 Panel coverage and power

|

Analysis

| $n$
|

note

|

R-Judge (specificity, F1)

| 21
|

Qwen3-32B dropped ( $21\%$ unparseable)

|

InjecAgent (robustness)

| 22
|

|

AgentHarm (safety)

| 19
|

malformed tool calls or unsupported routing

|

AgentDojo

| 5
|

convergent control inconclusive

|

$\tau^{2}$ criterion

| 20
|

|

misalignment criterion

| 21
|

|

jailbreak criterion (StrongREJECT)

| 22
|

3-template average

|

expanded misalignment / jailbreak

| 40 / 41
|

12 organizations (App. A.6)

|

XSTest over-refusal

| 17–20
|

AgentHarm $\cap$ anchor overlap

|

cross-benchmark / partials

| 18
|

AgentHarm-limited

|

capability interaction (paired)

| 20
|

Figure 3: Evidence base: up to 46 models $\times$ 9 instruments (two capability anchors, four
audited benchmarks, three held-out criteria; $262$ model $\times$ instrument cells). Cells are
per-instrument min–max-normalised scores; grey is not-run. Rows beyond the $41$ -model anchored
panel are models excluded by pre-specified telemetry gates (App. A.6). Rows sorted by capability:
the 2026 expansion (top) is scored on the anchors and the two chat-based criteria, which need no
tool-calling harness; the four audited benchmarks run on the original panel. The capability anchors
and the R-Judge/InjecAgent scores covary; the safety criteria do not track them consistently.

#### Within-model score uncertainty.

From existing item-level data (no new runs), per-model standard errors: MMLU binomial SE over 500
fixed items (median $1.6$ , max $2.2$ points), GPQA over 198 (median $3.3$ , max $3.6$ ), the
misalignment criterion over 100 decisions (median $4.0$ , max $5.0$ ), R-Judge F1 bootstrapped over
571 records (median $1.8$ , max $3.0$ ), and the jailbreak criterion bootstrapped over ${\sim}150$
judged prompts (median $2.3$ , max $3.7$ ). Against between-model SDs ( $8.7$ , $18.2$ , $26.0$ ,
$14.9$ , $28.5$ respectively, expanded panel), the worst-case within-model SE is $13$ – $25\%$ of
SD; panel-level spread is $4$ – $8\times$ the worst-case SE, though individual pairs can be closer.
$\tau^{2}$ -retail is a single run with unbounded user-simulator variance and is not covered.

#### Internal consistency, not just precision.

Per instrument, from item-level data: split-half consistency (Spearman–Brown), item-bootstrap rank
stability, and top-quartile persistence (Table 3). Every item-rich instrument with item-resampling
support has corrected split-half $\geq 0.96$ , rank stability $\geq 0.97$ , and top-quartile
persistence $0.79$ – $0.96$ across item resamples, making item-sampling error an unlikely source of
the ranking disagreement in Table 1. Misalignment has two scenario halves (Spearman–Brown $0.70$ );
$\tau^{2}$ has no resampling bound. Jailbreak resampling is by base prompt, respecting
template dependence.

| Instrument | models | items | split-half (SB) | rank stab. | quartile

| R-Judge specificity | 21 | 570 | $+0.98$ ( $+0.99$ ) | $+0.99$ | $0.90$

| R-Judge F1 | 21 | 570 | $+0.93$ ( $+0.96$ ) | $+0.97$ | $0.87$

| InjecAgent robustness | 22 | 300 | $+0.95$ ( $+0.98$ ) | $+0.98$ | $0.88$

| AgentHarm safety (behaviors) | 19 | 44 | $+0.93$ ( $+0.96$ ) | $+0.97$ | $0.79$

| Jailbreak crit. (by base prompt) | 42 | 20 | $+0.94$ ( $+0.97$ ) | $+0.98$ | $0.96$

| MMLU (anchor) | 46 | 356 | $+0.93$ ( $+0.96$ ) | $+0.98$ | $0.92$

| GPQA (anchor) | 45 | 164 | $+0.95$ ( $+0.97$ ) | $+0.98$ | $0.92$

| Misalignment crit. (scenario halves) | 22 | 2 | $+0.54$ ( $+0.70$ ) | — | —

Table 3: Internal consistency from available item- or scenario-level data: mean split-half correlation of model scores over 50 random half-splits (Spearman–Brown corrected), mean Spearman of item-bootstrapped vs. observed rankings, and probability that an observed top-quartile model stays top-quartile under resampling. Item counts are the complete-coverage intersection across listed models (e.g. 356 of the 500 administered MMLU items, 164 of 198 GPQA items). Splits/draws are shared across models over items common to every model’s logs; jailbreak resamples by base prompt (the three templates are dependent), with a shared set of 20 of the 50 suite prompts because each model leaves a few different prompts unjudged. Jailbreak uses 42 models: the 41 anchored models plus DeepSeek-R1, which lacks the required anchor pair; misalignment has config-level data only, so its row is the blackmail-vs-leaking halves convergence. Item-rich instruments with item-resampling support have Spearman–Brown $\geq 0.96$ and rank stability $\geq 0.97$ . Misalignment’s halves agree moderately (Spearman–Brown $0.70$ ). $\tau^{2}$ is a single run and the one unbounded instrument.

The exploratory R-Judge/InjecAgent–misalignment pattern is estimated at $n{=}18$ – $20$ with
simulated power $\approx 0.35$ to detect $\rho{=}0.4$ at $\alpha{=}0.05$ ; the minimum detectable
effect ( $80\%$ power) is $\rho{=}0.62$ at $n{=}18$ , above the observed $+0.41$ , so the null
cannot exclude an effect of practical interest. Detecting $+0.41$ with $80\%$ power would require
$n{\approx}40$ . Under Bonferroni over the nine RQ3 partial tests ( $\alpha{=}0.006$ ), the
capability crossover and AgentHarm $\to$ jailbreak (both $p<0.001$ ) survive; the
R-Judge/InjecAgent cells ( $p{=}0.05$ – $0.09$ ) do not.

Figure 4: Models with similar measured capability can differ widely in safety across
model-developing organizations. Each point is a model; colour and shape identify organization;
error bars are within-model SEs (§A.2). Lines show within-organization fits for the six
organizations with $\geq 4$ anchored models; the capability $\to$ safety slope is large but
sign-heterogeneous across organizations ( $n$ -weighted mean $-0.14$ misalignment, $+0.26$
jailbreak). Among higher-capability models (shaded, composite $>0.4$ ), safety spans nearly the full
range (arrows). After rank-residualising safety on capability, organization groups differ under
permutation tests (df-adjusted Kruskal–Wallis $\varepsilon^{2}{=}0.34$ / $0.57$ , $p\leq 0.004$ ).
The static composite does not capture all capability; this is an association, not evidence that
organizational identity causes the difference.

### A.3 Cross-benchmark structure (detail for §4.3)

Figures 5 and 6 show the $n{=}18$ R-Judge–AgentHarm relationship and the single-factor
structure.

Figure 5: AgentHarm safety against R-Judge specificity on the $n{=}18$ cross-benchmark panel,
coloured by the harmonized GPQA anchor. The 7-model correlation ( $\rho=-0.64$ ) is $\rho=+0.02$
( $p=0.95$ ; grey trend) at $n{=}18$ : models near the top of either axis span the full range of the
other, the pairwise view of Table 1.

Figure 6: PC1 loadings on the $n{=}18$ panel with harmonized anchors ( $48.7\%$ of variance):
R-Judge, InjecAgent, and capability load together, AgentHarm opposite, in $17$ of $18$
leave-one-model-out fits. Component sign is arbitrary; only relative directions matter. The median
between-benchmark correlation is $0.22$ (§4.3).

#### Correlation-convention sensitivity.

The PCA above uses the Pearson correlation matrix of standardized scores. A Spearman PCA keeps the
loading structure (anchors $-0.72$ / $-0.78$ , AgentHarm opposite at $+0.38$ ) and the positive
control still passes, but PC1 falls from $48.7\%$ to $40.8\%$ and Horn’s parallel analysis retains
no factor. The interpretation criterion (first-factor variance $\geq 40\%$ ) holds under both
conventions; the negative control passes as executed: $200$ column-permuted matrices give median
first eigenvalue $1.71$ against the parallel-analysis $95$ th-percentile threshold of $2.18$
(exceedance $2\%$ ). The factor is supporting rather than primary relative to the correlation-free
ranking disagreement of Table 1.

### A.4 R-Judge metric sensitivity (detail for §4.3–§4.4)

Specificity ignores true positives as F1 ignores true negatives; every R-Judge-involving result is
recomputed under four metrics on the paper’s panels (Table 4). The one metric-sensitive cell is the
already-exploratory R-Judge–misalignment partial.

| | RQ1 ( $n{=}18$ ) | RQ2 ( $n{=}20$ ) | RQ3 partials

| R-Judge metric | reversal | PC1 | MMLU | GPQA | $\tau^{2}$ | misalign. | jailbr.

| specificity | $+0.02$ | $-0.71$ | $+0.16$ | $+0.07$ | $-0.09$ | $+0.41$ | $-0.11$

| recall | $-0.05$ | $-0.14$ | $+0.40$ | $+0.32$ | $+0.15$ | $-0.46$ | $-0.05$

| balanced acc | $-0.09$ | $-0.86$ | $+0.71$ | $+0.49$ | $+0.19$ | $+0.19$ | $-0.20$

| F1 (official) | $-0.29$ | $-0.75$ | $+0.76$ | $+0.62$ | $+0.32$ | $-0.29$ | $-0.17$

Table 4: R-Judge metric sensitivity under specificity, recall, balanced accuracy, and F1. RQ3 entries are partial Spearman correlations controlling the capability composite ( $\tau^{2}$ $n{=}19$ , misalignment $n{=}18$ , jailbreak $n{=}20$ ). The RQ1 small-panel reversal does not return under any metric; the task-success and jailbreak nulls are stable; the misalignment estimate is $+0.41$ under specificity and $-0.46$ under recall (both n.s.).

### A.5 Robustness of the capability crossover (detail for §4.4)

#### Partial-correlation convention.

Partials residualize midranks on the harmonized capability composite and re-rank the residuals.
Under the classic alternative (Pearson on rank residuals) the exploratory misalignment estimates are
$+0.49$ (R-Judge) and $+0.41$ (InjecAgent), within $0.08$ of Table 2, and AgentHarm–jailbreak stays
$+0.72$ ; no conclusion changes.

Figure 7: $\tau^{2}$ -retail success against the harmonized capability composite ( $\rho=+0.60$ ,
$p=0.005$ , $n{=}20$ ; grey trend). Colour is AgentHarm safety (open circle: no AgentHarm score);
safe and unsafe models fall on both sides of the trend with no residual safety gradient, matching
the null incremental validity in Table 2.

Interaction $\Delta{=}{-}1.00$ (95% CI $[-1.48,-0.49]$ , bootstrap $p<0.001$ ). Leave-one-out
$\Delta\in[-1.16,-0.90]$ . Leave-one-organization-out, per omitted organization: Amazon $-0.96$ ,
Anthropic $-1.01$ , Cohere $-0.90$ , DeepSeek $-1.13$ , Google $-0.98$ , Meta $-0.96$ , Mistral
$-0.93$ , OpenAI $-1.16$ , Qwen $-0.95$ (the range coincides with leave-one-out because Cohere
contributes one model and OpenAI, the largest group, gives the widest swing).
Organization-clustered bootstrap: $\Delta\in[-1.49,-0.59]$ , $P(\Delta{\geq}0)<10^{-4}$ .
Subsampling (median / % negative / % $\leq-0.5$ / % sign-flips): size-10 $-0.95$ / $100\%$ /
$97\%$ / $0\%$ ; size-12 $-0.96$ / $100\%$ / $99\%$ / $0\%$ ; size-14 $-0.97$ / $100\%$ / $100\%$ /
$0\%$ (cf. the RQ1 R-Judge–AgentHarm correlation, $|\rho|{\geq}0.5$ in $\sim 25\%$ of size-7
draws). Independent graders: misalignment classifier $\rho=0.97$ (Claude-3-Haiku vs. GPT-4o),
jailbreak judge $\rho=0.98$ (Gemini vs. gpt-4o), AgentHarm refusal predictor $\rho=0.69$ (a Gemini
refusal judge re-scoring cached completions, $n{=}19$ ; AgentHarm–jailbreak $+0.66$ vs $+0.72$ ; the
swap redoes only the refusal component, not the full grading pipeline). Per-jailbreak AgentHarm
$\to$ jailbreak partials within the three-template suite: AIM $+0.44$ , refusal-suppression
$+0.70$ , prefix-injection $+0.62$ ; the original single-AIM criterion gave $+0.50$ .

#### Misalignment 3-scenario battery (exploratory).

Adding a third scenario (murder; our classifier, 22 models) to blackmail $+$ leaking moves the
R-Judge/InjecAgent–misalignment interaction from $\Delta=+0.52$ ( $p=0.068$ ) to $\Delta=+0.53$
( $p=0.048$ ); the gain in significance reflects a tighter CI from a larger battery, not a larger
effect, and a post-hoc extension crossing the threshold is a forking path. Murder is also a severe
harm forecast by harm-compliance (AgentHarm $\to$ battery $+0.43$ ), diluting construct
specificity; blackmail $+$ leaking remains primary.

### A.6 The 2026 model-panel expansion (n $\approx$ 41) and organization-group differences (detail for
§4.4)

The targeted expansion, specified before new models were evaluated on either criterion, predicted
that the crossover would “hold or strengthen,” and probed whether any panel change follows
capability or organization grouping. The panel grew to $41$ anchored models spanning $12$
model-developing organizations, using the original chat/scenario harnesses; the criteria need no
tool loop, and new models skip the tool-calling audit benchmarks. Exclusions used harness behavior
only, never outcome scores: Kimi-K2.6 and both Nemotron-3 models returned usable anchor answers on
fewer than $50\%$ of items or produced unbounded reasoning traces; MiniMax-M3 fell below the
misalignment engagement threshold through truncated outputs, not low harm, and was retained on the
single-turn jailbreak criterion (misalignment $n{=}40$ , jailbreak $n{=}41$ ). Gemini-2.5-Pro
yielded no valid anchor; GLM-4.5 yielded MMLU but not GPQA. Both encountered routing
incompatibilities; only GLM-4.5 appears as a partial row in Fig. 3. These exclusions illustrate
harness staleness.

#### Primary expansion result: the capability–misalignment correlation weakens.

Capability $\to$ misalignment-safety moves from $-0.44$ ( $n{=}21$ , asymptotic $p=0.047$ ;
bootstrap $p=0.11$ , $95\%$ CI $[-0.85,+0.09]$ ) to $-0.16$ ( $n{=}40$ , bootstrap $p=0.40$ ;
$95\%$ CI $[-0.54,+0.22]$ , leave-one-out $[-0.25,-0.12]$ ). The pre-specified confirmation
threshold ( $\rho\leq-0.35$ at $p<0.05$ ) is not met; the CI still includes $-0.44$ . Three tests of
the change itself (accounting for nesting of the original panel in the expanded one): a stratified
nested bootstrap gives $\Delta\rho=+0.27$ , $95\%$ CI $[-0.25,+0.78]$ , two-sided $p=0.29$
(jailbreak $+0.26$ , CI $[-0.14,+0.69]$ , $p=0.22$ ); the 2026-only models give $\rho=+0.07$ versus
the original $-0.44$ (independent Fisher $z$ , $p=0.12$ ); size-21 subsets matched to the original
organization histogram give median $\rho=-0.17$ (IQR $[-0.27,-0.08]$ ). Organization mix does not
reproduce the original value; generational change is compatible but not established, and the change
is non-significant. The estimate cannot be assumed to generalize across model populations, and the
“hold or strengthen” prediction was not met. Among models of similar measured capability, safety
varies widely across organizations (composite ${\approx}0.8$ : o3-mini $0.99$ , Grok-4.3 $0.70$ ,
DeepSeek-V3 $0.19$ ), so a ceiling does not account for the residual variation. With $\tau^{2}$ not
re-collected, the correlation is updated but not the interaction $\Delta$ .

#### Organization-group analysis.

Six organizations with $\geq 4$ anchored models permit partial separation of the collinear
capability and organization factors. Within-organization capability $\to$ safety slopes are
sign-heterogeneous and near-zero in aggregate; after rank-residualizing safety on capability,
organization groups differ on both criteria (permutation $p{=}0.004$ misalignment, $<0.001$
jailbreak; df-adjusted Kruskal–Wallis $\varepsilon^{2}{=}0.34$ and $0.57$ ; Table 5, Fig. 4). All
five secondary tests survive Holm (max adjusted $p=0.043$ ). Capability is matched only on a
static-knowledge composite (unmeasured agentic capability can appear as an organization
difference), within-organization $n$ is $4$ – $8$ , and some organizations are range-restricted
(Anthropic ceilings near safety $1.0$ ); the claim rests on the between-organization permutation
test, not the slopes. Because that test permutes labels over only seven organization clusters
(within-organization $n$ $4$ – $8$ ), its Type-I calibration was checked on $2{,}000$ synthetic
panels per null with no between-organization mean difference, holding models, organizations, and
capability fixed: with real safety values reshuffled across models, and with i.i.d. Gaussian safety,
rejection matches nominal $\alpha$ at $0.05/0.01/0.005$ on both criteria ( $0.045$ – $0.052$ at
$\alpha{=}0.05$ ). A worst-case null with organization-clustered variances but no mean difference
mildly inflates rejection on the jailbreak structure only ( $0.067$ at $\alpha{=}0.05$ ; none
detectable at $\alpha{=}0.005$ ); doubling both $p$ -values leaves the organization-group tests
Holm-significant. Dropping near-ceiling Anthropic gives misalignment $\varepsilon^{2}=0.27$ ,
$p=0.018$ and jailbreak $\varepsilon^{2}=0.54$ , $p<0.001$ .

| | $n$ | misalign. $\rho$ | jailbreak $\rho$

| within-organization slopes:

| Anthropic | 5 | $+0.71$ | $+0.67$

| OpenAI | 8 | $+0.59$ | $+0.64$

| Qwen | 5 | $-0.90$ | $+0.80$

| Meta | 4 | $-0.40$ | $-0.40$

| Google | 4 | $-0.60$ | $-0.40$

| Mistral | 4 | $-1.00$ | $-0.40$

| $n$ -weighted mean
| | $-0.14$ | $+0.26$

| organization-group test (group $n{\geq}3$ ; total $n{=}33$ ):

| between-group share
| | $0.508$ | $0.674$

|    df-adjusted $\varepsilon^{2}$
| | $0.34$ | $0.57$

| permutation $p$ ( $10^{4}$ )
| | $0.004$ | $<0.001$

| Kruskal–Wallis $p$
| | $0.02$ | $0.002$

Table 5: Organization-group differences on the $n{\approx}41$ expansion. Top: within-organization capability slopes on safety. Bottom: after rank-residualizing safety on the static capability composite, organization-group tests on both criteria; $\varepsilon^{2}$ is the df-adjusted Kruskal–Wallis effect size, not a causal share.

#### Expanded-panel jailbreak result.

The jailbreak criterion was rebuilt on the expanded panel (three published templates re-scored for
every new model). Capability $\to$ jailbreak-safety rises from $+0.08$ ( $n{=}21$ ) to $+0.34$
( $n{=}41$ ; bootstrap $p=0.035$ , $95\%$ CI $[+0.03,+0.60]$ , leave-one-out $[+0.30,+0.39]$ ),
the panel-to-panel change non-significant. Within-organization slopes remain heterogeneous
(weighted mean $+0.26$ ); models of similar measured capability vary across organizations
(permutation $p<0.001$ ; $\varepsilon^{2}{=}0.57$ ). Among high-capability models jailbreak-safety
spans $9$ – $99$ (Mistral-Large $0.09$ , DeepSeek-V3 $0.37$ , Anthropic/OpenAI $0.98$ – $0.99$ ).
Neither the sign nor the magnitude of capability $\to$ safety is stable across criteria or panels.

An earlier pre-specified R-Judge–misalignment continuation fell from $+0.41$ ( $n{=}18$ ) to
$+0.23$ ( $n{=}25$ ), below its $0.30$ threshold; it was stopped and is retained as a hypothesis.