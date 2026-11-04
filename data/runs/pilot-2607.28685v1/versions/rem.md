# Safety, or Just Capability? A Validity Audit of Agent-Safety Benchmarks

Youting Wang

  
Xiao Han

  
Dingyan Shang

Affiliation: Yuan Tang

  
Bowen Liu

###### Abstract

Agent-safety benchmarks measure different behaviors, and their scores get quoted
interchangeably as an agent’s safety. We treat four of them (R-Judge, InjecAgent,
AgentHarm, AgentDojo) as measurements to be validated, running each under its
official implementation and author-provided scorer on up to 22 models, with MMLU
and GPQA measured by us under one protocol as a capability composite. The metric
is the first problem. On any binary trace-judgment benchmark scored by $F_{1}$ , an
“always positive” policy attains $F_{1}=2\pi/(1+\pi)$ ; on R-Judge that is
$0.690$ , above five of the 21 models that actually discriminate. The three
broad-coverage benchmarks then rank the same 18 models differently, and the
trade-off behind that disagreement is a small-panel artifact: R-Judge specificity
against AgentHarm safety correlates $-0.64$ at $n{=}7$ and $+0.02$ at $n{=}18$ ,
and a quarter of random size-7 subsets reach $|\rho|\geq 0.5$ around that
near-zero value. Held-out validity turns on which outcome you pick. Capability
predicts task success ( $\rho{=}{+}0.60$ ) but correlates negatively with
misalignment safety ( $\rho{=}{-}0.44$ , $n{=}21$ ). On their paired $n{=}20$
panel, the corresponding contrast is $\Delta{=}{-}1.00$ (95% CI
$[-1.48,-0.49]$ , $p<0.001$ ), and it survives leave-one-organization-out and
organization-clustered bootstrap analyses. On an expanded 41-model panel, the
misalignment correlation weakens to $-0.16$ (95% CI $[-0.54,+0.22]$ ) and
jailbreak strengthens to $+0.34$ , though neither change is significant.
AgentHarm shows the strongest held-out association, $\rho{=}{+}0.72$
with three-template jailbreak safety after controlling capability. But both
instruments score harmful compliance, so this is evidence of convergent validity
rather than general safety. Naming the benchmark, metric, target behavior, and
model panel is the minimum a safety claim needs.

## 1 Introduction

Between 2024 and 2026, dozens of benchmarks appeared claiming to measure whether LLM agents are
safe or reliable: resistance to prompt injection
(Zhan et al. 2024; Debenedetti et al. 2024), safety-risk awareness (Yuan et al. 2024), refusal of
harmful agentic tasks (Andriushchenko et al. 2025), and risky-tool-use identification
(Ruan et al. 2024), among others. A recent taxonomy (Li et al. 2026) catalogs
more than forty of them and reports a symptom worth taking seriously: swap the benchmark and the
safety ranking contradicts itself, with rank concordance near $0.10$ across models. That work
stops at a twelve-model concordance check over four benchmarks, and lists
capability-controlled evaluation and benchmark consolidation among its open problems.

This paper picks up there. If a safety score is a measurement, it can fail in the ways
measurements fail, so we separate construct validity (what a score measures),
metric validity (whether its metric measures that target), and criterion validity
(whether it tracks held-out behavior). Four questions follow:

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

We identify a formal metric-validity failure (Observation 1): for any binary
trace-judgment benchmark scored by F1, the score of an “always positive” baseline has a closed
form, and on R-Judge that baseline outranks five evaluated models. We then quantify a small-panel
failure mode we walked into: the R-Judge-specificity/AgentHarm-safety correlation moves from
$-0.64$ at $n{=}7$ to $+0.02$ at $n{=}18$ , and a quarter of random size-7 subsets show
$|\rho|\geq 0.5$ despite the near-zero full-panel value. The core is a pre-specified,
capability-controlled criterion-validity audit over one task-success outcome and two safety
outcomes: capability predicts task success, its safety correlations move with the outcome and the
panel, and the strongest safety-score result is the AgentHarm–jailbreak association
( $\rho{=}+0.72$ after controlling capability), with borderline evidence of outcome selectivity
under organization-level resampling ( $p{=}0.051$ ). We release an API-only audit harness and re-run
artifacts for four benchmarks and three held-out outcomes, included with the accompanying reproducibility package.

#### Main conclusion.

A capability score is not a safety score, and no one agent-safety
benchmark stands in for safety as a whole. What a score licenses you to say depends on how it was
produced. Small panels are the sharpest edge: at seven models a weak relationship can look
systematic, so validity needs re-checking as the model population turns over.

#### Scope.

The three RQ3 criteria are stand-ins for deployment, not claims about
real-world harm; their limits (single instances, grader dependence, panel size) are in
Section 6.

## 2 Related Work

#### Agent-safety benchmarking.

Each of the four benchmarks we audit targets one construct.
InjecAgent (Zhan et al. 2024) and AgentDojo (Debenedetti et al. 2024) target prompt-injection
robustness; AgentHarm (Andriushchenko et al. 2025) measures compliance with harmful agentic tasks; R-Judge
(Yuan et al. 2024) scores safety-risk awareness over interaction traces. Evaluators we do not
re-run, such as ToolEmu (Ruan et al. 2024) for risky tool use, share the same status: designed as
evaluators, never validated as measurements. AutoMonitor-Bench (Yang et al. 2026)
scores misbehavior monitors by miss and false-alarm rates, the two-sided reporting our metric
analysis argues for. Closest to this paper is a taxonomy and consistency analysis of agent-safety
benchmarks (Li et al. 2026), which documents ranking disagreement but does not
control for capability, test criterion validity, or recover a latent structure; capability-controlled
evaluation is named there as future work, and we take up all three.

#### Construct validity.

Prior work asks whether benchmark scores agree and what latent
constructs they capture. Benchmark-agreement testing (Perlitz et al. 2024), metabench
(Kipnis et al. 2025), and direct construct analyses (Bean et al. 2025) find that
knowledge and reasoning benchmarks share a strong latent factor. Kearns (2026) and
NIST’s statistical framework (Keller et al. 2026) bring quantitative construct-validity and
uncertainty modeling to LLM evaluation. We point the same psychometric lens at agent reliability
and safety, where benchmarks target distinct behaviors—refusal, injection robustness, risk
awareness—and nobody has yet quantified how much capability confounds their comparison.

#### Benchmark gaming.

A separate concern is that a benchmark can be gamed by an
adversary. Our claim is different and logically prior: even for an honest model, the headline score
may not validly measure the intended construct. The F1 gameability we document
(Section 4) is a concrete instance. The policy that scores mid-leaderboard is not
adversarial; it labels every trace “unsafe” without examining any of them.

#### Over-refusal.

XSTest (Röttger et al. 2024) and OR-Bench (Cui et al. 2024) document
exaggerated safety, or over-refusal, at the level of individual prompts. We ask the
analogous question one level up, between benchmarks: are models that refuse harmful tasks
(AgentHarm) worse at identifying benign traces correctly (R-Judge specificity), and does that
produce the ranking disagreement? On the full panel that correlation sits near zero. On
seven-model subsets it often does not.

#### Agent reliability and task performance.

Recent work isolates specific agent-reliability failures
such as corrupt success (Cao et al. 2026) and evidence-bounded reporting
(Chen 2025); $\tau$ -bench (Yao et al. 2024) and its dual-control successor
$\tau^{2}$ -bench (Barres et al. 2025) measure agentic task success. We borrow the latter’s retail
domain as our held-out RQ3 task-success criterion, and treat the safety benchmarks as candidate
columns for a validity audit rather than as settled measurements.

## 3 Method

### 3.1 Model panel and benchmarks

We assemble a compute-light, API-only panel of up to 22 models from nine model-developing organizations
(OpenAI 6, Meta 3, Qwen 3, Mistral 3, DeepSeek 2, Amazon 2, Anthropic, Google, Cohere; full
roster in Fig. 3 and the artifact). We re-run four safety benchmarks with their official
implementations and scorers: R-Judge (full 571 items, self-judge), InjecAgent (300 stratified
direct-harm/data-stealing items, rule-scored), AgentHarm (44 base behaviors spanning all eight
harm categories, via the official Inspect evaluation with a gpt-4o-mini judge), and AgentDojo
(the slack suite, 100 items, environment-scored for security and utility), plus a
held-out task-success criterion for RQ3 (below).

Coverage differs by benchmark (Fig. 3 maps the full panel): R-Judge has $n{=}21$ after
Qwen3-32B falls below the $90\%$ valid-output threshold; InjecAgent has $n{=}22$ ; AgentHarm and AgentDojo,
which require native tool use, have $n{=}19$ and $5$ ; and the $\tau^{2}$ criterion has $n{=}20$
(App. A.2). The common R-Judge/InjecAgent/AgentHarm panel therefore has $n{=}18$ .
Capability anchors (MMLU and
GPQA-Diamond) are measured by us under one uniform protocol: MMLU zero-shot on a fixed 500-item
subset (seed 42, identical items for every model) and GPQA-Diamond (198 items, chain-of-thought),
through the same API harness as the safety runs. We began with provider-reported model-card
numbers; the replacements hold the protocol fixed across models, and the composite passes its
positive control (§4). The anchor panel is 21 models (DeepSeek-R1 is excluded
for unstable long-form GPQA generation); we keep the provider-reported anchors as a
pre-harmonization comparison.

### 3.2 Benchmark targets

Before looking at any relationship among the score columns, we wrote down what behavior each
benchmark is meant to measure. R-Judge tests whether a model distinguishes unsafe from benign
interaction traces. InjecAgent and AgentDojo both test resistance to prompt injection, so their
correlation serves as a convergent-validity check. AgentHarm tests whether a model refuses harmful agentic
tasks. For R-Judge, we analyze specificity and balanced accuracy in addition to the official F1
because the three metrics reward different error patterns.

#### Score orientation and terminology.

All reported scores are oriented so that higher is
better or safer. R-Judge specificity is the fraction of benign traces correctly identified
as benign; its balanced accuracy averages performance on benign and unsafe traces.
InjecAgent robustness is one minus attack success, and AgentHarm safety is one minus harmful
compliance. For RQ3, $\tau^{2}$ -bench is a held-out task-success criterion, not a safety score.
We use internal consistency for split-half analyses (App. A.2) and name each
benchmark-specific score explicitly, instead of stretching “calibration” or “reliability”
past their usual meanings.

### 3.3 Analysis plan

Before running each analysis, we fixed its hypotheses, thresholds, and decision rules. Because
these choices lack an independent timestamp, we describe the analyses as pre-specified rather than
preregistered.
We use Spearman correlations for all association tests. Unless otherwise noted, correlation
$p$ -values use two-sided asymptotic Spearman tests. In Fig. 2, error bars are
model-level percentile-bootstrap intervals. The panel expansion and direct panel comparisons
follow their pre-specified bootstrap procedures (App. A.6). The corroborative PCA operates on the
Pearson correlation matrix of the standardized score columns; factor count is chosen with Horn’s
parallel analysis against a random-normal null, with the rank-based (Spearman) sensitivity in
App. A.3. We report partial Spearman correlations controlling the capability composite;
App. A.5 gives the rank-residual implementation and an alternative convention.
The capability composite is the mean of the standardized, verified
MMLU and GPQA anchors; its model ordering is identical to the anchors’ first principal component.
Between-construct redundancy is measured by rank-residualized partial correlations that control
for the capability composite.

We pre-specified three controls: positive (capability-anchor loadings $\geq 0.6$ on the first
factor); negative (a column-permuted score matrix within the parallel-analysis null); and
convergent (a high correlation between the same-construct InjecAgent and AgentDojo scores).
The headline metric is the median pairwise partial
Spearman $|\rho|$ among distinct-construct benchmarks, controlling the capability composite; the decision
rule permits interpretation of the factor structure only if first-factor variance is $\geq 40\%$
or controlling capability reduces that median by $\geq 0.15$ , with the positive and negative
controls passing. These thresholds and the disconfirming test were fixed before any real run.

For criterion validity (RQ3) we pre-specified three held-out outcomes, each before running it.
Task success is $\tau^{2}$ -bench retail (Barres et al. 2025): 40 multi-turn,
environment-scored tool-agent-user tasks. Misalignment safety records whether a model avoids
blackmail or information leaking in a fictional autonomous-agent scenario
(Lynch et al. 2025), scored by a fixed classifier over 100 decisions per model.
Jailbreak safety is one minus harmful-compliance on 50 forbidden prompts, averaged over three published
jailbreak templates to reduce the ceiling created by any single template
(Souly et al. 2024; Wei et al. 2023). All three run through
the same API-only harness (the two safety criteria are single-turn or scenario-based and cover the
full panel; the $\tau^{2}$ criterion is a multi-turn tool loop covering 20 models), are distinct from
the four audited benchmarks, and are not among the predictors. For each, the headline metric is the partial
Spearman correlation of every safety score with the criterion, controlling the capability
composite. We first report the criterion’s raw association with capability, then test whether each
safety score adds predictive information. The pre-specified analysis gives reporting rules for both significant and null results. The
accompanying reproducibility package contains the analysis scripts and frozen inputs needed to
reproduce the reported results.

## 4 Results

Results run in order of evidential strength: the metric-validity failure (RQ4) and capability
correlations (RQ2) are firmest, cross-benchmark structure (RQ1) is weaker, and the power-limited
held-out validity tests (RQ3) come last.

### 4.1 The headline metric is gameable (RQ4)

Figure 1: A leaderboard’s headline metric can be matched by an “always unsafe” baseline.
On R-Judge ( $n{=}21$ ), five real, discriminating models score below this constant baseline
(F1 ${=}0.69$ , dashed): F1 gives no credit for correctly rejected benign traces, so the degenerate
policy can outrank models that discriminate.

R-Judge labels $52.7\%$ of its 571 items “unsafe.” A degenerate policy that answers
“unsafe” on every item, exercising no safety reasoning, therefore attains recall $1.0$ ,
specificity $0$ , and $\mathrm{F1}=0.690$ . On the 21 panel models that produce valid R-Judge
verdicts (Qwen3-32B is excluded for $21\%$ unparseable output; see Method) this constant-baseline score
outranks five real models (F1 $0.30$ – $0.57$ ; Fig. 1), all of which discriminate. The panel’s highest-specificity model, o3-mini
( $0.97$ ), receives $\mathrm{F1}=0.702$ , only $0.012$ above the constant baseline, because its recall is lower.
Balanced accuracy reorders the board, since it credits correct decisions on both classes.

###### Observation 1 (F1 admits a high-scoring constant baseline).

On a benchmark with class base rate $\pi$ scored by F1, the constant “always-positive” policy
attains $\mathrm{F1}=2\pi/(1+\pi)$ without distinguishing safe from unsafe items. For R-Judge ( $\pi=0.527$ , a
near-balanced base rate) this is $0.690$ , which exceeds the observed F1 of several models that do
discriminate. The closed form depends on prevalence, but the measurement problem is F1’s blindness
to true negatives: correctly identifying a benign trace earns nothing, so F1 cannot stand alone as
a measure of two-sided discrimination on this trace-judgment benchmark.

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

Table 1: The three broad-coverage agent-safety benchmarks rank the same $n{=}18$ models differently (higher is safer). Rows are sorted by AgentHarm safety. Bold marks column maxima and underlining marks minima, with display-precision ties included. GPQA is shown as a capability reference. A model high on AgentHarm can rank low on R-Judge specificity, so the scores are not interchangeable.

### 4.2 The capability confound is real but metric-dependent (RQ2)

Across the R-Judge panel ( $n{=}20$ with harmonized anchors), capability correlates strongly with
balanced accuracy ( $\rho=+0.71$ with MMLU, $p<0.001$ ; $+0.49$ with GPQA, $p=0.03$ ) and F1
( $+0.76$ and $+0.62$ ), but only weakly with specificity ( $+0.16$ and $+0.07$ ). This difference
follows the metric definitions: balanced accuracy and F1 include recall on unsafe traces, whereas
specificity measures only correct treatment of benign traces and can be inflated by rarely flagging
anything. So the capability confound is a property of the metric you report, not of the benchmark
as a whole—and it moves with the panel too: the specificity–MMLU correlation fell from $0.85$ on
our first nine models to $+0.16$ at $n{=}20$ .

### 4.3 Benchmarks disagree; a seven-model panel suggested a trade-off that does not persist (RQ1)

The firmest RQ1 result is descriptive and correlation-free: on this panel no single benchmark orders
the models the way another does (Table 1); a top-AgentHarm model can rank near the bottom on R-Judge specificity.
This makes “safety rank” benchmark-dependent without resting on any correlation estimate.
Disagreement between benchmarks coded to different constructs (§3) is expected
and does not by itself indict any single instrument; what it defeats is the practice of quoting
these scores interchangeably as the safety of an agent.

We read a trade-off into that disagreement early on; it does not survive the larger
cross-benchmark panel. The R-Judge-specificity/AgentHarm-safety rank correlation was $-0.64$
at $n{=}7$ but $+0.02$ ( $p=0.95$ ) at $n{=}18$ (Fig. 5, appendix), and random
size-7 subsets yield $|\rho|\geq 0.5$ in a quarter of draws despite the near-zero full-panel value.
Sampling variability alone is enough to manufacture a “clean reversal” at that panel size, and
the dissolution holds under all four R-Judge metrics (App. A.4). Whether capability
moderates what is left we do not test; we flag it as a hypothesis.

A single-factor PCA provides weaker, corroborative evidence (Fig. 6, appendix):
its first component ( $48.7\%$ variance) places R-Judge specificity, InjecAgent robustness, and both
capability anchors in one direction and AgentHarm safety in the other ( $+0.33$ , opposite the cluster
in $17$ of $18$ leave-one-out fits).

#### The positive control now passes.

On the harmonized anchors the capability composite
passes its pre-specified positive control (MMLU loads $0.74\geq 0.6$ ; on provider-reported numbers it had
failed at $0.42$ , our original reason for caution) and the negative control also passes (a
column-permuted score matrix yields eigenvalues within the parallel-analysis null), so the
pre-specified interpretation conditions are met (the convergent control is inconclusive at AgentDojo’s
$n{=}5$ ). We still treat the factor as corroboration only: a rank-based (Spearman) PCA keeps the loading
structure but drops PC1 to $40.8\%$ with no factor retained by Horn (App. A.3), and the
median between-benchmark correlation is just $0.22$ ; the firm, factor-independent RQ1 result is
the raw-score disagreement (Table 1).

### 4.4 Predictive validity differs across held-out outcomes (RQ3)

Everything above is internal to the benchmarks. RQ3 steps outside them: do these scores predict
held-out behavior that a capability test does not already predict? Which outcome you choose
decides the answer. We pre-specified three (§3):
$\tau^{2}$ -bench task success, misalignment safety, and jailbreak safety. All are distinct from the
four audited benchmarks, and each has a fixed metric and decision rule.

| | Task success | Safety outcomes

| Predictor | $\tau^{2}$ | misalign. | jailbreak†

| Capability (raw $\rho$ ) | $+0.60^{**}$ | $-0.44^{*}$ | $+0.08$

| Safety-score partial $\rho$ (controlling capability):

| R-Judge specificity | $-0.09$ | $+0.41$ | $-0.11$

| InjecAgent robustness | $+0.16$ | $+0.47$ | $+0.21$

| AgentHarm safety | $-0.23$ | $+0.16$ | $+0.72^{***}$

Table 2: Criterion validity on the 2024–25 panel (cellwise $n{=}18$ – $21$ because predictor
coverage differs; App. A.2; higher is better or safer).
The top row reports raw correlations with capability; the remaining rows report partial Spearman
correlations controlling the capability composite. Capability predicts task success but not either
safety outcome consistently. AgentHarm is strongly associated only with the three-template
jailbreak outcome, marked by †. Stars use two-sided asymptotic Spearman tests
( ${}^{*}p<0.05$ , ${}^{**}p<0.01$ , ${}^{***}p<0.001$ ).

#### Against task success, safety scores add no information beyond capability.

Capability predicts $\tau^{2}$ -retail success ( $\rho=+0.60$ ,
$p=0.005$ , $n{=}20$ ), while no safety score shows incremental validity once capability is partialled
out (best $|\rho|=0.23$ , n.s. at $n{=}19$ ; Table 2;
Fig. 7, appendix). Because capability already predicts $\tau^{2}$ , these partials ask the
stricter question of whether a safety score predicts what capability leaves unexplained; none does.
The null is not a verdict on the safety scores. Task success is simply an outcome that capability
already explains well, which is why we test two safety outcomes as well.

Figure 2: Capability predicts task success; safety point estimates differ by outcome and
panel. Filled circles show the 2024–25 correlations; open diamonds show the expanded-panel
safety correlations; error bars are model-level percentile-bootstrap 95% CIs. Thin lines connect
panel-level estimates, not model trajectories. Misalignment changes from $-0.44$ to $-0.16$ and
jailbreak from $+0.08$ to $+0.34$ ; neither between-panel change is significant. Task success was
not re-collected for the expansion.

#### Against agentic misalignment, capability has the opposite association.

On the full 21-model
panel, capability predicts misalignment-safety negatively ( $\rho_{\text{cap}}=-0.44$ , asymptotic
$p=0.047$ ),
the opposite sign to its $+0.60$ on $\tau^{2}$ : higher capability scores go, if anything, with lower
safety here (Mistral-Large blackmails or leaks $85\%$ of the time; o3-mini is a capable exception at $1\%$ ).
After controlling capability, InjecAgent robustness ( $+0.47$ ) and R-Judge specificity ( $+0.41$ )
reach the pre-specified $0.40$ effect-size threshold, while AgentHarm safety does not ( $+0.16$ ;
Table 2). The two associations remain exploratory: both confidence intervals include
zero, neither survives Bonferroni correction, power is only $\approx 0.35$ , and the pre-specified
directional prediction named AgentHarm, not the two scores that came up. The R-Judge result is
also metric-sensitive: its partial correlation falls from $+0.41$ under specificity to $+0.19$
under balanced accuracy and reverses to $-0.46$ under recall (App. A.4).

#### A direct interaction test on the original panel.

We compare capability’s correlations
with task success and misalignment on their paired $n{=}20$ panel. The task-success correlation is
$+0.60$ and the misalignment-safety correlation is $-0.41$ ( $-0.44$ on the full panel), a
difference of $\Delta=-1.00$ (95% CI $[-1.48,-0.49]$ , $p<0.001$ ). This difference is stable to
leave-one-model-out, leave-one-organization-out, organization-clustered bootstrap, and subsampling
(App. A.5). The corresponding R-Judge interaction is only marginal
( $\Delta=+0.50$ , $p=0.082$ ), so the evidence supports the capability interaction but not a full
two-way dissociation. An independent misalignment grader produces nearly identical model scores
( $\rho=0.97$ ).

#### The capability–misalignment correlation is weaker on the expanded panel.

Before running the expansion, we specified that the original negative correlation should “hold or
strengthen.” Instead, it changes from $\rho=-0.44$ ( $n{=}21$ , asymptotic $p=0.047$ ) to $-0.16$
( $n{=}40$ , bootstrap $p=0.40$ ; 95% CI $[-0.54,+0.22]$ ; App. A.6). The point estimate is less
than half as large and no longer significant, but the interval still includes $-0.44$ and the
between-panel difference is not significant (nested-bootstrap $\Delta\rho=+0.27$ , 95% CI
$[-0.25,+0.78]$ ). So the original estimate should not be assumed to carry over to the expanded
model population, though the data do not establish a temporal shift either. Matching subsets to the
original organization composition gives a similar estimate (median $\rho=-0.17$ ), which suggests
composition alone does not explain the difference; a generational explanation stays possible and
undemonstrated. Because $\tau^{2}$ was not re-collected, we could not re-test the original
interaction.

#### Models with similar capability scores differ in safety across organizations
(Fig. 4).

The 41-model panel spans 12 model-developing organizations
(misalignment $n{=}40$ ; jailbreak $n{=}41$ after one engagement-gate exclusion). Six
organizations contribute at least four models for within-organization correlations, and seven
contribute at least three models for the group comparison (App. A.6,
Table 5). Within organizations, capability–safety correlations vary in sign and
average near zero ( $-0.14$ misalignment, $+0.26$ jailbreak). After adjusting safety ranks for
capability, the seven organization groups differ on both criteria (permutation $p=0.004$
misalignment, $p<0.001$ jailbreak; Holm-significant), with df-adjusted Kruskal–Wallis effect
sizes $\varepsilon^{2}=0.34$ and $0.57$ . None of this is causal: the static capability composite may
miss agentic ability, each organization contributes few models, and some groups have restricted
safety ranges. What the panel supports is narrower—similar MMLU/GPQA scores do not imply similar
safety scores.

#### Against jailbreak safety, AgentHarm shows the strongest held-out association.

We pre-specified a second safety outcome with a different failure mode, harness, and scoring
procedure: harmful compliance under jailbreak, scored with StrongREJECT
(Souly et al. 2024). A single jailbreak template (AIM) produces a ceiling because 11 of 22
models refuse every request. Averaging scores over three published templates
(Wei et al. 2023) leaves no model at the ceiling. Capability is weakly associated with this
outcome on the 2024–25 panel ( $\rho=+0.08$ , $n{=}21$ , $p=0.73$ ). The expanded-panel estimate is
$+0.34$ ( $n{=}41$ , $p=0.04$ ), but the between-panel change is not significant.

After controlling capability, AgentHarm safety correlates with the three-template jailbreak outcome
at $\rho=+0.72$ ( $p<0.001$ ), compared with $+0.50$ for the ceiling-limited single-template
measure. The estimate is stable to leave-one-model-out analysis and an independent grader
(App. A.5). AgentHarm is more strongly associated with jailbreak than with
misalignment in a bootstrap that treats models as independent ( $\Delta=+0.59$ , $p=0.027$ ); a
bootstrap that instead resamples the eight represented organizations gives $p=0.051$ . On this
paired panel, the misalignment partial is $+0.13$ ( $n{=}19$ ); Table 2’s $+0.16$ uses
the common $n{=}18$ panel. R-Judge
specificity ( $-0.11$ ) and InjecAgent robustness
( $+0.21$ ) do not predict jailbreak safety. AgentHarm and StrongREJECT both measure harmful
compliance, so part of what the $+0.72$ shows is convergent validity. The stronger claim—that
AgentHarm is selective for jailbreak over misalignment—rests on a borderline
organization-resampled test, and the matching comparison for R-Judge and InjecAgent, whether they
predict misalignment better than jailbreak, is marginal as well ( $\Delta=+0.52$ , $p=0.068$ ).

#### Two alternative explanations, checked.

Seven LLM coders who saw only construct
descriptions, and none of the predictive results, independently assigned AgentHarm and the
jailbreak outcome to the same harm-blocking category ( $7/7$ for each; Krippendorff
$\alpha=0.61$ across three categories). They classified the misalignment outcome less cleanly,
one more reason to keep its R-Judge/InjecAgent associations exploratory
(App. A.1). The second check: after controlling capability, AgentHarm safety is
uncorrelated with over-refusal on benign XSTest prompts (Röttger et al. 2024) ( $\rho=-0.01$ ,
$n{=}17$ ), which weighs against the story that models score well on AgentHarm and the jailbreak
outcome simply by refusing everything.

## 5 Discussion

What the results support is narrow: these agent-safety benchmarks are not interchangeable measures
of a single property. An “always unsafe” baseline matches R-Judge F1
(Observation 1), and the three broad-coverage benchmarks rank the same models
differently. Held-out validity shifts with the outcome and the model panel
(Table 2): capability predicts task success, while its correlations with the two safety
outcomes are unstable across panels. AgentHarm’s capability-controlled association with
three-template jailbreak safety is strong ( $\rho=+0.72$ ), but both measures concern harmful
compliance and the outcome-selectivity test gives $p=0.051$ under organization resampling, which
makes this convergent validity with the specificity claim still open. The R-Judge/InjecAgent
associations with misalignment are exploratory. Safety ranks also differ among organization groups
after adjustment for the capability composite—an observational pattern that identifies no
organizational cause. A safety claim is defensible only with its benchmark, metric, target
behavior, and model population attached.

## 6 Limitations and Threats to Validity

[This section has been omitted.]

## 7 Conclusion

Across four benchmarks and up to 22 models, R-Judge’s F1 lets an “always unsafe” baseline outrank
five models, and switching benchmarks switches the rankings. Capability predicts held-out task
success; it does not predict safety consistently across outcomes or panels. The strongest held-out
safety association is AgentHarm’s ( $\rho=+0.72$ with jailbreak safety after controlling
capability), though the two measures partly overlap and organization-resampled evidence for
outcome selectivity is borderline. Safety evaluations should report the full confusion structure,
drop standalone F1 wherever true negatives matter, and tie every claim to an explicit target
behavior and model population.

## Appendix A Technical Appendix

This appendix holds method detail and the full robustness numbers summarized in the body; it is
supplementary and not required to follow the main argument.

### A.1 Ruling out two artifacts (detail for §4.4)

#### Blinded construct coding.

Seven LLM coders from distinct families—GPT-4o,
Claude-3-Haiku, Gemini-2.5-Flash, Llama-3.3-70B, Qwen-2.5-72B, Mistral-Large, DeepSeek-V3—were each
shown only a neutral, predictive-information-stripped description of what each benchmark
measures (no mention of our criteria, category names, or any RQ3 result) plus two construct
definitions (“harm blocking” vs. “risk recognition and correct treatment of benign inputs”). The two options
and the benchmark order were shuffled per coder to remove position bias. Each coder assigned every
benchmark to one construct. Votes (B ${=}$ harm blocking, R ${=}$ risk recognition):

| Benchmark | blind | author | B / R

| R-Judge | recognition | recognition | 0 / 7

| InjecAgent | recognition | recognition | 1 / 6

| AgentHarm | blocking | blocking | 7 / 0

| AgentDojo | recognition | recognition | 0 / 7

All four majorities match the hand-assignment; raw agreement is $27/28$ ratings, Krippendorff’s
$\alpha=0.825$ (nominal), against a chance baseline of $0.5^{4}=6.2\%$ for one coder matching all
four labels by coin flip. The single dissent (Llama-3.3-70B placing InjecAgent as harm blocking) is the
one semantically borderline case (injection-robustness). This controls coder bias but not
rubric-designer bias (the category definitions were written by authors who had seen the results); we
therefore also report the RQ1 factor, which separates AgentHarm ( $+0.33$ ) from the
R-Judge/InjecAgent cluster with no reference to any criterion.

#### Blinded criterion coding, with a distractor.

To check that the predictor $\to$ criterion
matching (not just the predictor labels) is recoverable blind, and to make the agreement statistic
meaningful, we re-ran the coding over all six targets (four benchmarks $+$ two criteria) with a third,
plausible task-success distractor construct. Krippendorff’s $\alpha=0.61$ (three categories,
chance $\approx 1/3$ ), and the distractor is chosen only $5/42$ times. The harm-blocking
classification is robust: AgentHarm and the jailbreak criterion both map to harm blocking $7/7$ ,
R-Judge to risk recognition $7/7$ ,
and InjecAgent $5/7$ , all unchanged by the distractor. Two targets are honestly contested: AgentDojo
reads as task success to $4/7$ coders (it is utility-scored; our $n{=}5$ benchmark), and the
misalignment criterion is contested: it maps to risk recognition $6/7$ in a separate two-way round
(frozen votes in the artifact), but to harm blocking $6/7$ here (avoiding a harmful autonomous action
can be read as both recognizing and declining it). This recovers the AgentHarm–jailbreak pairing
without using the predictive results, and is an independent reason we keep the
R-Judge/InjecAgent–misalignment pattern exploratory.

#### XSTest over-refusal (blanket-refusal check).

We ran XSTest (Röttger et al., 2024;
cited in the main paper) on the
panel via its published CPI (compliance/partial/refusal) scorer (judge gpt-4o-mini): the
safe subset (benign prompts that superficially resemble unsafe ones) measures over-refusal,
the unsafe subset measures correct refusal of overtly-harmful prompts. The unsafe subset is
ceilinged (every model refuses $83$ – $100\%$ , sd $4.5$ , limited discriminating variance), so the
informative signal is over-refusal on the safe subset (range $0$ – $60\%$ ; Claude-3-Haiku over-refuses
$60\%$ , o3-mini $30\%$ , others $\leq 15\%$ ). Partial- $\rho$ (capability-controlled) two-by-two:

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
but not ruling out a blanket-refusal effect. Capability is $+0.21$ (n.s.) on benign compliance and
$-0.32$ (n.s.) on harmful refusal; R-Judge specificity is $+0.15$ on over-refusal (equivalently
$-0.15$ on safe compliance). Misalignment safety is likewise unrelated to over-refusal
( $+0.11$ , $n{=}19$ , $p=0.66$ ), while its unsafe-refusal association is $+0.47$
( $n{=}18$ , $p=0.052$ ). These low-power checks neither support nor rule out that explanation.

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

12 organizations (App. A.6)

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

Figure 3: The audit’s evidence base: up to 46 models $\times$ 9 instruments (two capability
anchors, four audited benchmarks, three held-out criteria; $262$ model $\times$ instrument cells). Cells are
per-instrument min–max-normalised scores; grey is not-run. The rows beyond the $41$ -model anchored
panel are models excluded from the analyses by pre-specified telemetry gates
(App. A.6). Rows are sorted by capability: the
2026 expansion (top) is scored on the anchors and the two chat-based criteria, which need
no tool-calling harness, while the four audited benchmarks run on the original panel. The capability anchors
and the R-Judge/InjecAgent scores vary together; the safety criteria do not track them consistently.

#### Within-model score uncertainty.

From the existing item-level data (no new runs), per-model
standard errors for every instrument: MMLU binomial SE over its 500 fixed items (median $1.6$ , max $2.2$
points), GPQA over 198 (median $3.3$ , max $3.6$ ), the misalignment criterion over its 100 decisions
(median $4.0$ , max $5.0$ ), R-Judge F1 bootstrapped over its 571 records (median $1.8$ , max $3.0$ ), and
the jailbreak criterion bootstrapped over its ${\sim}150$ judged prompts (median $2.3$ , max $3.7$ ). Against each
instrument’s between-model SD ( $8.7$ , $18.2$ , $26.0$ , $14.9$ , $28.5$ respectively, on the expanded
panel), the worst-case within-model SE is $13$ – $25\%$ of that SD. Thus the observed
panel-level spread is $4$ – $8\times$ the worst-case SE, although individual model pairs can be
closer. ( $\tau^{2}$ -retail remains a
single run whose user-simulator variance we cannot bound; it is the one instrument this analysis
cannot cover.)

#### Internal consistency, not just precision.

Small SEs do not by themselves make rankings stable,
so we also compute, per instrument from its item-level data, split-half consistency
(Spearman–Brown), item-bootstrap rank stability, and top-quartile persistence
(Table 3). Every item-rich instrument with item-resampling support is internally consistent (split-half
$\geq 0.96$ corrected; rank stability $\geq 0.97$ ; a top-quartile model stays top-quartile in
$0.79$ – $0.96$ of item resamples), making item-sampling error an unlikely explanation for the
panel-level ranking disagreement in Table 1. Misalignment has only two scenario
halves (Spearman–Brown $0.70$ ), and $\tau^{2}$ has no resampling bound. The jailbreak
computation resamples by base prompt, so the three templates’ dependence is respected.

| Instrument | models | items | split-half (SB) | rank stab. | quartile

| R-Judge specificity | 21 | 570 | $+0.98$ ( $+0.99$ ) | $+0.99$ | $0.90$

| R-Judge F1 | 21 | 570 | $+0.93$ ( $+0.96$ ) | $+0.97$ | $0.87$

| InjecAgent robustness | 22 | 300 | $+0.95$ ( $+0.98$ ) | $+0.98$ | $0.88$

| AgentHarm safety (behaviors) | 19 | 44 | $+0.93$ ( $+0.96$ ) | $+0.97$ | $0.79$

| Jailbreak crit. (by base prompt) | 42 | 20 | $+0.94$ ( $+0.97$ ) | $+0.98$ | $0.96$

| MMLU (anchor) | 46 | 356 | $+0.93$ ( $+0.96$ ) | $+0.98$ | $0.92$

| GPQA (anchor) | 45 | 164 | $+0.95$ ( $+0.97$ ) | $+0.98$ | $0.92$

| Misalignment crit. (scenario halves) | 22 | 2 | $+0.54$ ( $+0.70$ ) | — | —

Table 3: Internal consistency from available item- or scenario-level data: mean split-half correlation of model scores over 50 random half-splits (Spearman–Brown corrected), mean Spearman of item-bootstrapped vs. observed rankings, and the probability that an observed top-quartile model stays top-quartile under resampling. Item counts are the complete-coverage intersection across all listed models (e.g. 356 of the 500 administered MMLU items, 164 of 198 GPQA items). Splits/draws are shared across models and use the items common to every model’s logs; the jailbreak criterion resamples by base prompt (the three templates are dependent), and its shared set is 20 of the 50 suite prompts because each model leaves a few different prompts unjudged. Jailbreak uses 42 models: the 41 anchored models plus DeepSeek-R1, which lacks the required anchor pair; misalignment has config-level data only, so its row is the blackmail-vs-leaking halves convergence. Every item-rich instrument with item-resampling support has Spearman–Brown $\geq 0.96$ and rank stability $\geq 0.97$ , making item-sampling error an unlikely explanation for the panel-level disagreement of Table 1. Misalignment’s two scenario halves agree only moderately (Spearman–Brown $0.70$ ). $\tau^{2}$ is a single run and remains the one unbounded instrument.

The exploratory R-Judge/InjecAgent–misalignment pattern is estimated at $n{=}18$ – $20$ with
power $\approx 0.35$ (simulated) to detect
$\rho{=}0.4$ at $\alpha{=}0.05$ ; the minimum detectable effect ( $80\%$ power) is $\rho{=}0.62$ at
$n{=}18$ , above the observed $+0.41$ . Thus the non-significant result cannot exclude an effect of
practical interest, and we retain it only as a hypothesis. Detecting an effect of $+0.41$ with
$80\%$ power would require $n{\approx}40$ . Under a Bonferroni correction over the nine RQ3 partial
tests ( $\alpha{=}0.006$ ), the two firm results (the capability crossover and AgentHarm $\to$ jailbreak,
both $p<0.001$ ) survive, while the R-Judge/InjecAgent cells ( $p{=}0.05$ – $0.09$ ) do not.

Figure 4: Models with similar measured capability can differ widely in safety across
model-developing organizations. Each point is a model; colour and shape identify its organization;
error bars are within-model SEs
(§A.2). Within an organization (lines, the six organizations with $\geq 4$ anchored models) the
capability $\to$ safety slope is large but sign-heterogeneous across organizations ( $n$ -weighted mean $-0.14$
misalignment, $+0.26$ jailbreak). Among higher-capability models (shaded, composite $>0.4$ ), safety
spans nearly the full range (arrows). After rank-residualising safety on capability, organization groups
differ under permutation tests (df-adjusted Kruskal–Wallis $\varepsilon^{2}{=}0.34$ / $0.57$ ,
$p\leq 0.004$ ). The static composite does not capture all capability, so this is an association,
not evidence that organizational identity causes the difference.

### A.3 Cross-benchmark structure (detail for §4.3)

Figures 5 and 6 visualise the two RQ1 results: the dissolution
of the small-panel R-Judge–AgentHarm trade-off, and the single-factor structure that provides a
weaker description of how the scores covary.

Figure 5: R-Judge and AgentHarm do not rank the full panel in opposite order. AgentHarm
safety against R-Judge specificity on the $n{=}18$ cross-benchmark panel, coloured by the harmonized
GPQA anchor. The trade-off a 7-model panel suggested ( $\rho=-0.64$ ) dissolves at $n{=}18$
( $\rho=+0.02$ , $p=0.95$ ; grey trend): models near the top of either axis span the full range of
the other, the pairwise view of Table 1’s rank disagreement.

Figure 6: The first principal component groups R-Judge, InjecAgent, and capability, with
AgentHarm in the opposite direction. PC1 loadings on the $n{=}18$ panel with harmonized anchors
( $48.7\%$ of variance). AgentHarm has the opposite sign from the other four measures in $17$ of
$18$ leave-one-model-out fits. The sign of a principal component is arbitrary; only the relative
directions matter. We treat this structure as suggestive because the median between-benchmark
correlation is only $0.22$ (§4.3).

#### Correlation-convention sensitivity.

The PCA above operates on the Pearson correlation
matrix of the standardized scores. A rank-based (Spearman) PCA keeps the loading structure (anchors
$-0.72$ / $-0.78$ , AgentHarm opposite at $+0.38$ ) and the pre-specified positive control still passes,
but PC1 falls from $48.7\%$ to $40.8\%$ of variance and Horn’s parallel analysis retains no factor.
The pre-specified criterion for interpreting PC1 (first-factor variance $\geq 40\%$ ) holds under both
conventions, and the pre-specified negative control passes as executed: $200$ column-permuted
matrices give median first eigenvalue $1.71$ against the parallel-analysis $95$ th-percentile
threshold of $2.18$ (exceedance $2\%$ ). The convention sensitivity is one
more reason we treat the factor as supporting rather than primary evidence: the firm RQ1 result is
the correlation-free ranking disagreement of Table 1.

### A.4 R-Judge metric sensitivity (detail for §4.3–§4.4)

Because the paper rejects R-Judge’s headline F1 for ignoring true negatives yet represents
R-Judge by specificity (which ignores true positives) in the construct and criterion analyses, we
recompute every R-Judge-involving result under four metric choices on the paper’s own panels
(Table 4). The conclusions the paper leans on are metric-robust; the one
metric-sensitive cell is the R-Judge–misalignment partial, which the paper already treats
as exploratory.

| | RQ1 ( $n{=}18$ ) | RQ2 ( $n{=}20$ ) | RQ3 partials

| R-Judge metric | reversal | PC1 | MMLU | GPQA | $\tau^{2}$ | misalign. | jailbr.

| specificity | $+0.02$ | $-0.71$ | $+0.16$ | $+0.07$ | $-0.09$ | $+0.41$ | $-0.11$

| recall | $-0.05$ | $-0.14$ | $+0.40$ | $+0.32$ | $+0.15$ | $-0.46$ | $-0.05$

| balanced acc | $-0.09$ | $-0.86$ | $+0.71$ | $+0.49$ | $+0.19$ | $+0.19$ | $-0.20$

| F1 (official) | $-0.29$ | $-0.75$ | $+0.76$ | $+0.62$ | $+0.32$ | $-0.29$ | $-0.17$

Table 4: R-Judge metric sensitivity under specificity, recall, balanced accuracy, and F1. RQ3 entries are partial Spearman correlations controlling the capability composite ( $\tau^{2}$ $n{=}19$ , misalignment $n{=}18$ , jailbreak $n{=}20$ ). The RQ1 small-panel reversal does not return under any metric, and the task-success and jailbreak nulls are stable. The exploratory misalignment estimate is not stable: $+0.41$ under specificity and $-0.46$ under recall (both n.s.).

### A.5 Robustness of the capability crossover (detail for §4.4)

#### Partial-correlation convention.

Partials residualize midranks on the harmonized
capability composite and re-rank the residuals. Under the classic alternative (Pearson on the rank
residuals) the two exploratory misalignment estimates move to $+0.49$ (R-Judge) and $+0.41$
(InjecAgent), within $0.08$ of Table 2, and the AgentHarm–jailbreak estimate stays
$+0.72$ : no
conclusion changes under either convention.

Figure 7: Against task success, capability is the predictor and AgentHarm adds nothing.
$\tau^{2}$ -retail success against the harmonized capability composite ( $\rho=+0.60$ , $p=0.005$ ,
$n{=}20$ ; grey trend). Colour is AgentHarm safety (open circle: no AgentHarm score): safe and
unsafe models sit on both sides of the trend with no safety gradient after controlling capability, the
pre-specified result that AgentHarm adds no incremental validity in Table 2.

Interaction $\Delta{=}{-}1.00$ (95% CI $[-1.48,-0.49]$ , bootstrap $p<0.001$ ). Leave-one-out
$\Delta\in[-1.16,-0.90]$ . Leave-one-organization-out, per omitted organization: Amazon $-0.96$ , Anthropic
$-1.01$ , Cohere $-0.90$ , DeepSeek $-1.13$ , Google $-0.98$ , Meta $-0.96$ , Mistral $-0.93$ , OpenAI
$-1.16$ , Qwen $-0.95$ (the range coincides with leave-one-out because Cohere contributes one model
and OpenAI, the largest group, gives the widest swing). Organization-clustered bootstrap:
$\Delta\in[-1.49,-0.59]$ , $P(\Delta{\geq}0)<10^{-4}$ . Subsampling stability (draws of each size):
size-10 median $-0.95$ , $100\%$ negative, $97\%\leq-0.5$ , $0\%$ sign-flips; size-12
$-0.96$ / $100\%$ / $99\%$ / $0\%$ ; size-14 $-0.97$ / $100\%$ / $100\%$ / $0\%$ (contrast: the RQ1
R-Judge–AgentHarm correlation reaches $|\rho|{\geq}0.5$ in $\sim 25\%$ of size-7 draws).
Independent-grader checks:
the misalignment classifier reproduces at $\rho=0.97$ (Claude-3-Haiku vs. GPT-4o), the jailbreak judge
at $\rho=0.98$ (Gemini vs. gpt-4o), and AgentHarm’s refusal predictor at $\rho=0.69$ (a Gemini refusal
judge re-scoring the cached completions, $n{=}19$ ; the AgentHarm–jailbreak estimate remains similar,
$+0.66$ vs $+0.72$ ; the moderate agreement reflects that the swap redoes only the refusal component,
not the full grading pipeline). Per-jailbreak AgentHarm $\to$ jailbreak partials (within the three-template
suite): AIM $+0.44$ , refusal-suppression $+0.70$ , prefix-injection $+0.62$ ; the original
single-AIM criterion gave $+0.50$ .

#### Misalignment 3-scenario battery (exploratory).

Adding a third scenario (murder; our
classifier, 22 models) to the blackmail $+$ leaking criterion lifts the
R-Judge/InjecAgent–misalignment
interaction from $\Delta=+0.52$ ( $p=0.068$ ) to $\Delta=+0.53$ ( $p=0.048$ ). The point estimate is
essentially unchanged, so the significance comes from the tighter CI of a larger battery rather
than from a stronger effect; a post-hoc extension crossing the threshold is a forking path, and we
do not lean on it. Murder is also a severe harm that harm-compliance forecasts
(AgentHarm $\to$ battery $+0.43$ ), diluting construct specificity, so we keep blackmail $+$ leaking
primary.

### A.6 The 2026 model-panel expansion (n $\approx$ 41) and organization-group differences (detail for
§4.4)

Before evaluating the new models on either criterion, we specified a targeted expansion predicting that
the crossover would “hold or strengthen,” both to test whether
the capability crossover holds on current models and to probe whether any panel change follows
capability or organization grouping. We grew the panel to $41$ anchored models spanning $12$
model-developing organizations,
using the same chat/scenario harnesses as the original panel. The two criteria require no tool loop;
new models simply skip the tool-calling audit benchmarks. Exclusions were based only on harness
behavior, never on outcome scores: Kimi-K2.6 and both Nemotron-3 models returned usable anchor
answers on fewer than $50\%$ of items or produced unbounded reasoning traces;
MiniMax-M3 fell below the misalignment engagement threshold because of truncated outputs, not low
harm, but was retained on the
single-turn jailbreak criterion, so the misalignment analysis uses $n{=}40$ whereas the jailbreak
analysis uses $n{=}41$ .
Gemini-2.5-Pro yielded no valid anchor; GLM-4.5 yielded MMLU but not GPQA. Both encountered routing
incompatibilities, so only GLM-4.5 appears as a partial row in Fig. 3. These exclusions are
a small instance of evaluation harnesses going stale.

#### Primary expansion result: the capability–misalignment correlation weakens.

Capability $\to$ misalignment-safety moves from
$-0.44$ ( $n{=}21$ , asymptotic $p=0.047$ ; bootstrap $p=0.11$ , $95\%$ CI $[-0.85,+0.09]$ ) to
$-0.16$ ( $n{=}40$ , bootstrap $p=0.40$ ; $95\%$ CI $[-0.54,+0.22]$ ,
leave-one-out $[-0.25,-0.12]$ ) and is no longer significant. The pre-specified confirmation
threshold ( $\rho\leq-0.35$ at $p<0.05$ ) is not met, so the original negative correlation is not
reproduced. However, the CI still includes $-0.44$ , so we report a weaker estimate rather than
evidence that the correlation has disappeared. Three direct
tests of the change itself (accounting for the original panel being nested in the expanded one):
a stratified nested bootstrap gives $\Delta\rho=+0.27$ , $95\%$ CI $[-0.25,+0.78]$ , two-sided
$p=0.29$ (jailbreak $+0.26$ , CI $[-0.14,+0.69]$ , $p=0.22$ ); the 2026-only models alone give
$\rho=+0.07$ versus the original $-0.44$ (independent Fisher $z$ , $p=0.12$ ); and size-21 subsets
of the expanded pool matched to the original panel’s organization histogram give median $\rho=-0.17$
(IQR $[-0.27,-0.08]$ ), i.e. organization composition alone does not reproduce the original value. So
organization mix alone does not explain the movement; generational change is compatible with the
result but is not established, and the change itself is not statistically significant. Our bounded
conclusion is that the estimate cannot be assumed to generalize across model populations and that
the “hold or strengthen” prediction was not met. A ceiling does not explain all remaining
variation: among
models with similar measured capability, safety varies widely across organizations (composite ${\approx}0.8$ :
o3-mini $0.99$ , Grok-4.3 $0.70$ , DeepSeek-V3 $0.19$ ). Because $\tau^{2}$ task success was not
re-collected, we update this correlation, not the interaction $\Delta$ .

#### Organization-group analysis.

With $\geq 4$ anchored
models each from six organizations we can begin to separate the collinear capability and organization factors.
Within-organization capability $\to$ safety slopes are sign-heterogeneous and near-zero in aggregate;
across organizations, organization groups differ after rank-residualizing safety on capability on both criteria
(permutation $p{=}0.004$ misalignment, $<0.001$ jailbreak; df-adjusted Kruskal–Wallis
$\varepsilon^{2}{=}0.34$ and $0.57$ ;
Table 5, Fig. 4). All five secondary tests survive Holm (max adjusted
$p=0.043$ ). As in the body, we stop short of calling this a clean identification: capability is
matched only on a static-knowledge composite (so unmeasured agentic capability can still appear as
an organization difference), within-organization $n$ is $4$ – $8$ , and some organizations are range-restricted
(Anthropic ceilings near safety $1.0$ ). The between-organization permutation test, not the
within-organization slopes, carries the claim. Because that test permutes labels over only seven
organization clusters (within-organization $n$ $4$ – $8$ ), we checked its Type-I calibration on
$2{,}000$ synthetic panels per null with no between-organization mean
difference, holding
models, organizations, and capability fixed: with the real safety values
reshuffled across models, and with i.i.d. Gaussian safety, rejection matches nominal $\alpha$ at
$0.05/0.01/0.005$ on both criteria ( $0.045$ – $0.052$ at $\alpha{=}0.05$ ). A worst-case null with
organization-clustered variances but no mean difference inflates rejection mildly on the jailbreak structure only
( $0.067$ at $\alpha{=}0.05$ ; no detectable inflation at $\alpha{=}0.005$ ) — too small to change any
conclusion: even doubling both $p$ -values leaves the organization-group tests Holm-significant.
Dropping near-ceiling Anthropic gives misalignment $\varepsilon^{2}=0.27$ , $p=0.018$ and jailbreak
$\varepsilon^{2}=0.54$ , $p<0.001$ ; neither group result is carried solely by Anthropic.

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

|    df-adjusted $\varepsilon^{2}$
| | $0.34$ | $0.57$

| permutation $p$ ( $10^{4}$ )
| | $0.004$ | $<0.001$

| Kruskal–Wallis $p$
| | $0.02$ | $0.002$

Table 5: Organization-group differences on the $n{\approx}41$ expansion. Holding the organization
fixed, capability’s slope on safety is heterogeneous and near-zero (top). After rank-residualizing
safety on the static capability composite, organization groups differ under permutation tests on both
criteria (bottom); $\varepsilon^{2}$ is the df-adjusted Kruskal–Wallis effect size, not a causal share.

#### Expanded-panel jailbreak result.

We rebuilt the jailbreak criterion on the
expanded panel (three published templates re-scored for every new model). Capability $\to$ jailbreak-safety
rises from $+0.08$ ( $n{=}21$ ) to $+0.34$ ( $n{=}41$ ; bootstrap $p=0.035$ , $95\%$ CI $[+0.03,+0.60]$ ,
leave-one-out $[+0.30,+0.39]$ ), although the panel-to-panel change is not significant. Within-organization
slopes remain heterogeneous (weighted mean $+0.26$ ), while models with similar measured capability
vary across organizations (permutation $p<0.001$ ; $\varepsilon^{2}{=}0.57$ ). Among high-capability
models, jailbreak-safety spans $9$ – $99$ (Mistral-Large $0.09$ , DeepSeek-V3 $0.37$ ,
Anthropic/OpenAI $0.98$ – $0.99$ ). Together with the misalignment result, this shows that neither
the sign nor the magnitude of capability $\to$ safety is stable across criteria or panels.

An earlier pre-specified R-Judge–misalignment continuation fell from $+0.41$ ( $n{=}18$ ) to
$+0.23$ ( $n{=}25$ ), below its $0.30$ threshold; we stopped it and retain it as a hypothesis.
