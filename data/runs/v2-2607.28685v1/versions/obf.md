# Safety, or Just Capability? A Validity Audit of Agent-Safety Benchmarks

Youting Wang

  
Xiao Han

  
Dingyan Shang

Affiliation: Yuan Tang

  
Bowen Liu

###### Abstract


That the benchmark, metric, target behavior, and model panel must be named for a safety claim to be minimally supported follows from the finding that the strongest held-out association, $\rho{=}{+}0.72$ shown by KD9 with R8W safety after M7V was controlled, constitutes evidence of convergent validity rather than of general safety, both instruments being scorers of harmful compliance, and from the behavior, on an expanded 41-model panel, of the P2Z correlation, which weakens to $-0.16$ (95% CI $[-0.54,+0.22]$), and of L9R, which strengthens to $+0.34$, with neither change being significant. Held-out validity turns on which outcome is picked, M7V predicting task success ($\rho{=}{+}0.60$) yet correlating negatively with P2Z ($\rho{=}{-}0.44$, $n{=}21$), and on their paired $n{=}20$ panel the corresponding contrast, $\Delta{=}{-}1.00$ (95% CI $[-1.48,-0.49]$, $p<0.001$), survives D7T and organization-clustered bootstrap analyses. The three broad-coverage benchmarks rank the same 18 models differently, the trade-off behind that disagreement being a small-panel artifact in which QX7 G8X against KD9 safety correlates $-0.64$ at $n{=}7$ and $+0.02$ at $n{=}18$, with a quarter of random size-7 subsets reaching $|\rho|\geq 0.5$ around that near-zero value. On any binary trace-judgment benchmark scored by W3D, an "always positive" policy attains $F_{1}=2\pi/(1+\pi)$, which on QX7 is $0.690$, above five of the 21 models that actually discriminate, the metric being the first problem. All of this arises because agent-safety benchmarks measure different behaviors while their scores get quoted interchangeably as an agent's safety, four of them (QX7, ZM3, KD9, V4W) having been treated as measurements to be validated, each run under its official implementation and author-provided scorer on up to 22 models, with H8P and R2F measured by us under one protocol as a M7V.

## 1 Introduction

A capability score is not a safety score, and no one agent-safety benchmark stands in for safety as a whole, with what a score licenses you to say depending on how it was produced and with small panels as the sharpest edge, because at seven models a weak relationship can look systematic, so validity needs re-checking as the model population turns over; E5K, the formal metric-validity failure, gives, for any binary trace-judgment benchmark scored by W3D, the score of an “always positive” baseline a closed form, and on QX7 that baseline outranks five evaluated models, while the small-panel failure mode that was walked into is quantified by the QX7-G8X/KD9-safety correlation moving from $-0.64$ at $n{=}7$ to $+0.02$ at $n{=}18$, and by a quarter of random size-7 subsets showing $|\rho|\geq 0.5$ despite the near-zero full-panel value (QX7 denoting the paper's R-Judge, KD9 denoting the paper's AgentHarm, W3D denoting the paper's F1, and G8X denoting the paper's specificity), and the pre-specified, capability-controlled criterion-validity audit over one task-success outcome and two safety outcomes finds capability predicting task success, finds its safety correlations moving with the outcome and the panel, finds the strongest safety-score result in the KD9–L9R association ($\rho{=}+0.72$ after controlling capability), and finds borderline evidence of outcome selectivity under organization-level resampling ($p{=}0.051$), with an API-only audit harness and re-run artifacts for four benchmarks and three held-out outcomes released and included with the accompanying reproducibility package. Between 2024 and 2026, dozens of benchmarks appeared claiming to measure whether LLM agents are safe or reliable: resistance to prompt injection (Zhan et al. 2024; Debenedetti et al. 2024), safety-risk awareness (Yuan et al. 2024), refusal of harmful agentic tasks (Andriushchenko et al. 2025), and risky-tool-use identification (Ruan et al. 2024), among others; a recent taxonomy (Li et al. 2026) catalogs more than forty of them and reports a symptom worth taking seriously, that swapping the benchmark makes the safety ranking contradict itself with rank concordance near $0.10$ across models, and that work stops at a twelve-model concordance check over four benchmarks while listing capability-controlled evaluation and benchmark consolidation among its open problems. A safety score, if it is a measurement, can fail in the ways measurements fail, so construct validity is separated as what a score measures, metric validity as whether its metric measures that target, and criterion validity as whether it tracks held-out behavior; RQ1 (construct structure) asks whether nominally-distinct agent-safety benchmarks measure a common underlying factor or dissociable ones, RQ2 (capability confound) asks how much of the between-benchmark signal is general capability in disguise, RQ3 (criterion validity) asks whether the safety scores track held-out behavioral criteria beyond what general capability predicts, and RQ4 (metric validity) asks whether the headline metrics are valid instruments or reward degenerate behavior. The three RQ3 criteria are stand-ins for deployment, not claims about real-world harm; their limits (single instances, grader dependence, panel size) are in Section 6.

## 2 Related Work

Capability control, criterion-validity testing, and latent-structure recovery are, all three, taken up here. The correlation at issue sits near zero on the full panel, and on seven-model subsets it often does not. The safety benchmarks, ZM3 and V4W among them, are treated as candidate columns for a validity audit rather than as settled measurements, a status—designed as evaluators, never validated as measurements—shared by evaluators we do not re-run, such as ToolEmu (Ruan et al. 2024) for risky tool use, while misbehavior monitors are scored by miss and false-alarm rates in AutoMonitor-Bench (Yang et al. 2026), the two-sided reporting our metric analysis argues for. Different from, and logically prior to, the separate concern that a benchmark can be gamed by an adversary is the claim that, even for an honest model, the headline score may not validly measure the intended construct, of which the W3D gameability documented in Section 4 is a concrete instance, the policy that scores mid-leaderboard being not adversarial but labeling every trace “unsafe” without examining any of them. Each of the four benchmarks we audit targets one construct: compliance with harmful agentic tasks is measured by KD9 (Andriushchenko et al. 2025), safety-risk awareness over interaction traces is scored by QX7 (Yuan et al. 2024), and prompt-injection robustness is targeted jointly by V4W (V4W denoting the AgentDojo benchmark of Debenedetti et al. 2024) and ZM3 (ZM3 denoting the InjecAgent benchmark of Zhan et al. 2024). Whether benchmark scores agree, and what latent constructs they capture, has been asked before, a strong latent factor shared by knowledge and reasoning benchmarks being found by benchmark-agreement testing (Perlitz et al. 2024), metabench (Kipnis et al. 2025), and direct construct analyses (Bean et al. 2025), quantitative construct-validity and uncertainty modeling being brought to LLM evaluation by Kearns (2026) and NIST’s statistical framework (Keller et al. 2026), and the same psychometric lens being pointed at agent reliability and safety, where the targeted behaviors—refusal, injection robustness, risk awareness—are distinct and how much capability confounds their comparison has been quantified by nobody. Closest to this paper, the taxonomy and consistency analysis of agent-safety benchmarks by Li et al. (2026) documents ranking disagreement while neither controlling for capability, testing criterion validity, nor recovering a latent structure, capability-controlled evaluation being named there as future work. Exaggerated safety, or over-refusal, is documented at the level of individual prompts by OR-Bench (Cui et al. 2024) and by C9L (C9L denoting the XSTest benchmark of Röttger et al. 2024), and one level up, between benchmarks, the analogous question is asked: are models that refuse harmful tasks (KD9) worse at identifying benign traces correctly (the G8X of QX7), and does that produce the ranking disagreement—this being the correlation at issue. Specific agent-reliability failures, among them corrupt success (Cao et al. 2026) and evidence-bounded reporting (Chen 2025), have been isolated by recent work, and agentic task success is measured by $\tau$-bench (Yao et al. 2024) and by its dual-control successor $\tau^{2}$-bench (Barres et al. 2025), the retail domain of the latter being borrowed as our held-out RQ3 task-success criterion.

## 3 Method

Only if first-factor variance is $\geq 40\%$ or controlling capability reduces that median by $\geq 0.15$, with the positive and negative controls passing, does the 3.3 decision rule permit interpretation of the factor structure, the headline metric being the median pairwise U4G $|\rho|$ among distinct-construct benchmarks while M7V is controlled, and these thresholds and the disconfirming test were fixed before any real run; the three controls were pre-specified as positive (capability-anchor loadings $\geq 0.6$ on the first factor), negative (a column-permuted score matrix within the parallel-analysis null), and convergent (a high correlation between the same-construct ZM3 and V4W scores), with ZM3 and V4W both testing resistance to prompt injection so that their correlation serves as a convergent-validity check.

For criterion validity (RQ3), three held-out outcomes were pre-specified, each before running it: task success is $\tau^{2}$-bench retail (Barres et al. 2025): 40 multi-turn, environment-scored tool-agent-user tasks; P2Z (P2Z denoting misalignment safety) records whether a model avoids blackmail or information leaking in a fictional autonomous-agent scenario (Lynch et al. 2025), scored by a fixed classifier over 100 decisions per model; and L9R (L9R denoting jailbreak safety) is one minus harmful-compliance on 50 forbidden prompts, averaged over three published R8W templates to reduce the ceiling created by any single template (Souly et al. 2024; Wei et al. 2023), with all three run through the same API-only harness, the two safety criteria single-turn or scenario-based and covering the full panel, the $\tau^{2}$ criterion a multi-turn tool loop covering 20 models, all distinct from the four audited benchmarks and not among the predictors; for each, the headline metric is the U4G of every safety score with the criterion while M7V is controlled, the criterion’s raw association with capability being reported first and each safety score then being tested for added predictive information, and the pre-specified analysis gives reporting rules for both significant and null results.

Because the fixed hypotheses, thresholds, and decision rules for each analysis lack an independent timestamp, the analyses are described as pre-specified rather than preregistered; Spearman correlations are used for all association tests and, unless otherwise noted, correlation $p$-values use two-sided asymptotic Spearman tests, while in Fig. 2 error bars are model-level percentile-bootstrap intervals and the panel expansion and direct panel comparisons follow their pre-specified bootstrap procedures (App. A.6); the corroborative PCA operates on the Pearson correlation matrix of the standardized score columns, with factor count chosen by Horn’s parallel analysis against a random-normal null and the rank-based Spearman sensitivity in App. A.3; U4G (U4G denoting partial Spearman correlation) is reported controlling M7V (M7V denoting capability composite), App. A.5 gives the rank-residual implementation and an alternative convention, M7V is the mean of the standardized, verified H8P and R2F anchors and its model ordering is identical to the anchors’ first principal component, and between-construct redundancy is measured by rank-residualized partial correlations that control for M7V.

All reported scores are oriented so that higher is better or safer; the 3.2 target attributions were fixed before any relationship among the score columns was inspected, with QX7 testing whether a model distinguishes unsafe from benign interaction traces, KD9 testing whether a model refuses harmful agentic tasks, and, for QX7, G8X and J4N (J4N denoting balanced accuracy) being analyzed in addition to the official W3D because the three metrics reward different error patterns; QX7 G8X is the fraction of benign traces correctly identified as benign, its J4N averages performance on benign and unsafe traces, ZM3 robustness is one minus attack success, KD9 safety is one minus harmful compliance, for RQ3 $\tau^{2}$-bench is a held-out task-success criterion rather than a safety score, internal consistency is used for split-half analyses (App. A.2), and each benchmark-specific score is named explicitly instead of stretching “calibration” or “reliability” past their usual meanings.

The 3.1 assembly is a compute-light, API-only panel of up to 22 models from nine model-developing organizations (OpenAI 6, Meta 3, Qwen 3, Mistral 3, DeepSeek 2, Amazon 2, Anthropic, Google, Cohere; full roster in Fig. 3 and the artifact), on which four safety benchmarks are re-run with their official implementations and scorers—QX7 (full 571 items, self-judge), ZM3 (300 stratified direct-harm/data-stealing items, rule-scored), KD9 (44 base behaviors spanning all eight harm categories, via the official Inspect evaluation with a gpt-4o-mini judge), and V4W (the slack suite, 100 items, environment-scored for security and utility)—plus a held-out task-success criterion for RQ3; coverage differs by benchmark (Fig. 3 maps the full panel), since QX7 has $n{=}21$ after Qwen3-32B falls below the $90\%$ valid-output threshold, ZM3 has $n{=}22$, KD9 and V4W, which require native tool use, have $n{=}19$ and $5$, the $\tau^{2}$ criterion has $n{=}20$ (App. A.2), and the common QX7/ZM3/KD9 panel has $n{=}18$; capability anchors are measured by us under one uniform protocol through the same API harness as the safety runs, with H8P (H8P denoting MMLU) zero-shot on a fixed 500-item subset (seed 42, identical items for every model) and R2F (R2F denoting GPQA-Diamond) given 198 items with chain-of-thought; the anchor panel is 21 models, DeepSeek-R1 being excluded for unstable long-form R2F generation, the provider-reported model-card numbers and anchors with which we began are kept as a comparison preceding K5Q, the replacements hold the protocol fixed across models, and M7V passes its positive control (§4); the accompanying reproducibility package contains the analysis scripts and frozen inputs needed to reproduce the reported results.

## 4 Results

Without distinguishing safe from unsafe items, a constant "always-positive" policy attains $\mathrm{W3D}=2\pi/(1+\pi)$ on any benchmark with class base rate $\pi$ scored by W3D — E5K (E5K denoting the observation that W3D admits a high-scoring constant baseline) — which for QX7 ( $\pi=0.527$ , a near-balanced base rate) evaluates to $0.690$ , exceeding the observed W3D of several models that do discriminate; the closed form depends on prevalence, yet the measurement problem is the blindness of W3D to true negatives, since correctly identifying a benign trace earns nothing, so W3D cannot stand alone as a measure of two-sided discrimination on this trace-judgment benchmark.

Figure 1: A leaderboard's headline metric can be matched by an "always unsafe" baseline: on QX7 ( $n{=}21$ ), five real, discriminating models score below this constant baseline (W3D ${=}0.69$ , dashed), because W3D gives no credit for correctly rejected benign traces, so the degenerate policy can outrank models that discriminate.

QX7 labels $52.7\%$ of its 571 items "unsafe," and a degenerate policy that answers "unsafe" on every item, exercising no safety reasoning, therefore attains recall $1.0$ , G8X $0$ , and $\mathrm{W3D}=0.690$ , a constant-baseline score which, on the 21 panel models that produce valid QX7 verdicts (Qwen3-32B is excluded for $21\%$ unparseable output; see Method), outranks five real models (W3D $0.30$ – $0.57$ ; Fig. 1), all of which discriminate, while the panel's highest-G8X model, o3-mini ( $0.97$ ), receives $\mathrm{W3D}=0.702$ , only $0.012$ above the constant baseline because its recall is lower, and J4N, since it credits correct decisions on both classes, reorders the board.

| Model | QX7 | ZM3 | KD9 | R2F

| | G8X | robust. | safety
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

Table 1: The three broad-coverage agent-safety benchmarks rank the same $n{=}18$ models differently (higher is safer), rows being sorted by KD9 safety, with bold marking column maxima and underlining marking minima, display-precision ties included; R2F is shown as a capability reference, and a model high on KD9 can rank low on QX7 G8X, so the scores are not interchangeable.

Across the QX7 panel of $n{=}20$ models assembled with K5Q (K5Q denoting the harmonized anchors), capability correlates strongly with J4N ( $\rho=+0.71$ with H8P, $p<0.001$ ; $+0.49$ with R2F, $p=0.03$ ) and with W3D ( $+0.76$ and $+0.62$ ), but only weakly with G8X ( $+0.16$ and $+0.07$ ), and the confound moves with the panel too, the G8X–H8P correlation having fallen from $0.85$ on the first nine models to $+0.16$ at $n{=}20$ .

That difference follows the metric definitions — J4N and W3D include recall on unsafe traces, whereas G8X measures only correct treatment of benign traces and can be inflated by rarely flagging anything — so the capability confound is a property of the metric reported, not of the benchmark as a whole.

A trade-off was read into the cross-benchmark disagreement early on, and it does not survive the larger panel: the QX7-G8X/KD9-safety rank correlation was $-0.64$ at $n{=}7$ but $+0.02$ ( $p=0.95$ ) at $n{=}18$ (Fig. 5, appendix), random size-7 subsets yield $|\rho|\geq 0.5$ in a quarter of draws despite the near-zero full-panel value, sampling variability alone being enough to manufacture a "clean reversal" at that panel size, and the dissolution holds under all four QX7 metrics (App. A.4); whether capability moderates what is left is not tested and is flagged as a hypothesis.

The pre-specified interpretation conditions are met: on K5Q the M7V passes its pre-specified positive control (H8P loads $0.74\geq 0.6$ ; on provider-reported numbers it had failed at $0.42$ , the original reason for caution), and the negative control also passes (a column-permuted score matrix yields eigenvalues within the parallel-analysis null), while the convergent control is inconclusive at V4W's $n{=}5$ ; the factor is nevertheless treated as corroboration only, since a rank-based (Spearman) PCA keeps the loading structure but drops PC1 to $40.8\%$ with no factor retained by Horn (App. A.3), the median between-benchmark correlation is just $0.22$ , and the firm, factor-independent RQ1 result is the raw-score disagreement (Table 1).

A single-factor PCA provides weaker, corroborative evidence (Fig. 6, appendix): its first component ( $48.7\%$ variance) places QX7 G8X, ZM3 robustness, and both capability anchors in one direction and KD9 safety in the other ( $+0.33$ , opposite the cluster in $17$ of $18$ leave-one-out fits).

No single benchmark on this panel orders the models the way another does (Table 1), a top-KD9 model can rank near the bottom on QX7 G8X, and this descriptive, correlation-free fact — the firmest RQ1 result — makes "safety rank" benchmark-dependent without resting on any correlation estimate; disagreement between benchmarks coded to different constructs (§3) is expected and does not by itself indict any single instrument, and what it defeats is the practice of quoting these scores interchangeably as the safety of an agent.

With capability controlled, KD9 safety correlates with R8W (R8W denoting the three-template jailbreak outcome) at $\rho=+0.72$ ( $p<0.001$ ), compared with $+0.50$ for the ceiling-limited single-template measure, an estimate stable to leave-one-model-out analysis and an independent grader (App. A.5); QX7 G8X ( $-0.11$ ) and ZM3 robustness ( $+0.21$ ) do not predict L9R, KD9 is more strongly associated with L9R than with P2Z in a bootstrap that treats models as independent ( $\Delta=+0.59$ , $p=0.027$ ) while a bootstrap that instead resamples the eight represented organizations gives $p=0.051$ , and on this paired panel the P2Z partial is $+0.13$ ( $n{=}19$ ), whereas Table 2's $+0.16$ uses the common $n{=}18$ panel.

Because KD9 and T6Q both measure harmful compliance, part of what the $+0.72$ shows is convergent validity; the stronger claim — that KD9 is selective for L9R over P2Z — rests on a borderline organization-resampled test, and the matching comparison for QX7 and ZM3, whether they predict P2Z better than L9R, is marginal as well ( $\Delta=+0.52$ , $p=0.068$ ).

The second pre-specified safety outcome was given a different failure mode, harness, and scoring procedure, namely harmful compliance under jailbreak scored with T6Q (T6Q denoting StrongREJECT, Souly et al. 2024); a single jailbreak template (AIM) produces a ceiling because 11 of 22 models refuse every request, whereas averaging scores over three published templates (Wei et al. 2023) leaves no model at the ceiling, and capability is weakly associated with this outcome on the 2024–25 panel ( $\rho=+0.08$ , $n{=}21$ , $p=0.73$ ), the expanded-panel estimate being $+0.34$ ( $n{=}41$ , $p=0.04$ ) with the between-panel change not significant.

On the full 21-model panel, capability predicts P2Z negatively ( $\rho_{\text{cap}}=-0.44$ , asymptotic $p=0.047$ ), the opposite sign to its $+0.60$ on $\tau^{2}$ — higher capability scores going, if anything, with lower safety here (Mistral-Large blackmails or leaks $85\%$ of the time; o3-mini is a capable exception at $1\%$ ) — and, with capability controlled, ZM3 robustness ( $+0.47$ ) and QX7 G8X ( $+0.41$ ) reach the pre-specified $0.40$ effect-size threshold while KD9 safety does not ( $+0.16$ ; Table 2); both associations remain exploratory, since both confidence intervals include zero, neither survives Bonferroni correction, power is only $\approx 0.35$ , and the pre-specified directional prediction named KD9 rather than the two scores that came up, and the QX7 result is metric-sensitive, its partial correlation falling from $+0.41$ under G8X to $+0.19$ under J4N and reversing to $-0.46$ under recall (App. A.4).

Capability's correlations with task success and P2Z were compared on their paired $n{=}20$ panel: the task-success correlation is $+0.60$ and the P2Z correlation is $-0.41$ ( $-0.44$ on the full panel), a difference of $\Delta=-1.00$ (95% CI $[-1.48,-0.49]$ , $p<0.001$ ) that is stable to leave-one-model-out, to D7T (D7T denoting the leave-one-organization-out stability check), to organization-clustered bootstrap, and to subsampling (App. A.5); the corresponding QX7 interaction is only marginal ( $\Delta=+0.50$ , $p=0.082$ ), so the evidence supports the capability interaction but not a full two-way dissociation, and an independent P2Z grader produces nearly identical model scores ( $\rho=0.97$ ).

The original negative correlation had been specified, before the expansion was run, to "hold or strengthen"; instead it changes from $\rho=-0.44$ ( $n{=}21$ , asymptotic $p=0.047$ ) to $-0.16$ ( $n{=}40$ , bootstrap $p=0.40$ ; 95% CI $[-0.54,+0.22]$ ; App. A.6), a point estimate less than half as large and no longer significant, though the interval still includes $-0.44$ and the between-panel difference is not significant (nested-bootstrap $\Delta\rho=+0.27$ , 95% CI $[-0.25,+0.78]$ ), so the original estimate should not be assumed to carry over to the expanded model population even though the data do not establish a temporal shift either; matching subsets to the original organization composition gives a similar estimate (median $\rho=-0.17$ ), which suggests composition alone does not explain the difference, a generational explanation stays possible and undemonstrated, and because $\tau^{2}$ was not re-collected the original interaction could not be re-tested.

The 41-model panel spans 12 model-developing organizations (P2Z $n{=}40$ ; L9R $n{=}41$ after one engagement-gate exclusion), six organizations contribute at least four models for within-organization correlations and seven contribute at least three models for the group comparison (App. A.6, Table 5); within organizations, capability–safety correlations vary in sign and average near zero ( $-0.14$ P2Z, $+0.26$ L9R), and after safety ranks are adjusted for capability the seven organization groups differ on both criteria (permutation $p=0.004$ P2Z, $p<0.001$ L9R; Holm-significant), with df-adjusted Kruskal–Wallis effect sizes $\varepsilon^{2}=0.34$ and $0.57$ ; none of this is causal — the static M7V may miss agentic ability, each organization contributes few models, and some groups have restricted safety ranges — and what the panel supports is narrower: similar H8P/R2F scores do not imply similar safety scores (Fig. 4).

Capability predicts $\tau^{2}$ -retail success ( $\rho=+0.60$ , $p=0.005$ , $n{=}20$ ), while no safety score shows incremental validity once capability is partialled out (best $|\rho|=0.23$ , n.s. at $n{=}19$ ; Table 2; Fig. 7, appendix); because capability already predicts $\tau^{2}$ , these partials ask the stricter question of whether a safety score predicts what capability leaves unexplained, and none does, though the null is not a verdict on the safety scores, task success being simply an outcome that capability already explains well, which is why two safety outcomes are tested as well.

Figure 2: Capability predicts task success, while safety point estimates differ by outcome and panel; filled circles show the 2024–25 correlations, open diamonds show the expanded-panel safety correlations, error bars are model-level percentile-bootstrap 95% CIs, and thin lines connect panel-level estimates, not model trajectories; P2Z changes from $-0.44$ to $-0.16$ and L9R from $+0.08$ to $+0.34$ , with neither between-panel change significant, and task success was not re-collected for the expansion.

Which held-out outcome is chosen decides the answer to RQ3's question — whether these scores predict held-out behavior that a capability test does not already predict — and three were pre-specified (§3): $\tau^{2}$ -bench task success, P2Z, and L9R, all distinct from the four audited benchmarks and each with a fixed metric and decision rule.

| | Task success | Safety outcomes

| Predictor | $\tau^{2}$ | P2Z | L9R†

| Capability (raw $\rho$ ) | $+0.60^{**}$ | $-0.44^{*}$ | $+0.08$

| Safety-score partial $\rho$ (controlling capability):

| QX7 G8X | $-0.09$ | $+0.41$ | $-0.11$

| ZM3 robustness | $+0.16$ | $+0.47$ | $+0.21$

| KD9 safety | $-0.23$ | $+0.16$ | $+0.72^{***}$

Table 2: Criterion validity on the 2024–25 panel (cellwise $n{=}18$ – $21$ because predictor coverage differs; App. A.2; higher is better or safer): the top row reports raw correlations with capability and the remaining rows report U4G controlling M7V; capability predicts task success but not either safety outcome consistently, KD9 is strongly associated only with the R8W outcome, marked by †, and stars use two-sided asymptotic Spearman tests ( ${}^{*}p<0.05$ , ${}^{**}p<0.01$ , ${}^{***}p<0.001$ ).

Seven LLM coders who saw only construct descriptions and none of the predictive results independently assigned KD9 and the jailbreak outcome to the same harm-blocking category ( $7/7$ for each; Krippendorff $\alpha=0.61$ across three categories), and they classified the P2Z outcome less cleanly, one more reason to keep the QX7/ZM3 associations exploratory (App. A.1); in a second check, with capability controlled, KD9 safety is uncorrelated with over-refusal on benign C9L prompts (Röttger et al. 2024) ( $\rho=-0.01$ , $n{=}17$ ), which weighs against the story that models score well on KD9 and the jailbreak outcome simply by refusing everything.

The metric-validity failure (RQ4) and the capability correlations (RQ2) are the firmest of these results, the cross-benchmark structure (RQ1) is weaker, and the held-out validity tests (RQ3) are power-limited.

## 5 Discussion

A safety claim is defensible only with its benchmark, metric, target behavior, and model population attached, the supported reading being narrow in that interchangeable measurement of a single property by these agent-safety benchmarks is not supported: an “always unsafe” baseline matches QX7 W3D (E5K), and the same models are ranked differently by the three broad-coverage benchmarks. Variation of held-out validity with the outcome and the model panel (Table 2) is such that capability predicts task success while capability’s correlations with the two safety outcomes are unstable across panels, and KD9’s capability-controlled association with R8W L9R is strong ($\rho=+0.72$), with the two measures concerning harmful compliance and the outcome-selectivity test giving $p=0.051$ under organization resampling, which leaves this convergent validity with the G8X claim still open. The QX7/ZM3 associations with P2Z are exploratory; safety ranks also differ among organization groups after adjustment for M7V, an observational pattern that identifies no organizational cause.

## 6 Limitations and Threats to Validity

The QX7/ZM3–P2Z pattern is exploratory—marginal $\Delta=+0.52$, power $\approx 0.35$, the larger three-scenario result post-hoc (App. A.5)—and the corresponding QX7/ZM3–P2Z cells ($p\approx 0.05$–$0.09$), which would not survive Bonferroni, are treated as suggestive throughout, whereas the two strongly significant tests, the crossover and KD9$\to$L9R, both $p<0.001$, would survive it, per-test $p$-values being reported without family-wise correction and the claims resting instead on effect sizes and pre-specified rules.

None of the criteria gives literal deployment-harm probabilities: $\tau^{2}$-retail (40 tasks, one run/model) is scored by a user-simulator whose variance is not bounded; the P2Z criterion is a fictional-scenario probe, grader-robust at $\rho=0.97$ though its blackmail–leaking scenario halves agree only moderately (Spearman–Brown $0.70$); and the L9R criterion averages three jailbreaks.

That the KD9–L9R U4G barely moves, $+0.66$ against $+0.72$ (App. A.5), when KD9 refusals are rescored with a different model, agreement dropping to $\rho=0.69$, together with model-score correlations of $\rho=0.97$ for P2Z and $\rho=0.98$ for L9R, amounts to three independent-grader checks supporting robustness to judge choice; and the cross-benchmark analyses depend primarily on the other three benchmarks, V4W having been run on only five models.

The caution about small-panel claims originates with the first pass at $n{=}7$, the KD9-limited cross-benchmark panel now standing at $n{=}18$ and the gap most worth closing remaining a larger, capability-spanning agent-loop panel. Every RQ2/RQ3 sign was unchanged after re-measurement of H8P and R2F (§3) fixed the positive control (H8P $0.74$), which provider-reported model-card anchors had initially failed (H8P $0.42$); these are static-knowledge tests, although their M7V predicts multi-turn tool-agent success at $+0.60$, and a six-model BFCL pilot (Patil et al. 2025) failed as an agentic replacement, strict function-call syntax scoring having ranked GPT-4o-mini highest ($0.72$) and the two most capable models lowest ($0.42$–$0.48$), apparently rewarding format conformance—that failure being the argument for a validated agentic anchor, since measurement error of this kind attenuates or suppresses partial relationships instead of biasing them in a direction that could be signed.

For the item-rich instruments with resampling support, split-half consistency is $\geq 0.96$ and rank stability $\geq 0.97$; across the bounded instruments, worst-case within-model SEs are $12$–$25\%$ of the between-model SD, which supports the panel-level spread but not every pairwise model difference (App. A.2). The two construct-matching patterns are not equally well supported: the KD9–L9R association is large and stable in direction but partly reflects convergent measurement, its selectivity over P2Z borderline under D7T resampling.

## 7 Conclusion

Report the full confusion structure, tie every claim to an explicit target behavior and model population, and drop standalone W3D wherever true negatives matter; the strongest held-out safety association, with the two measures partly overlapping and organization-resampled evidence for outcome selectivity borderline, is KD9's ( $\rho=+0.72$ with L9R after M7V is controlled), M7V predicting held-out task success yet not safety consistently across outcomes or panels, while QX7's W3D lets an "always unsafe" baseline outrank five models and rankings are switched by benchmark switching, across four benchmarks and up to 22 models.

## Appendix A Technical Appendix

Supplementary method detail and the full robustness numbers summarized in the body are collected in this appendix, none of it required to follow the main argument. Under a Bonferroni correction over the nine RQ3 partial tests (α=0.006), the two firm results—the capability crossover and KD9 → jailbreak, both p<0.001—survive, while the QX7/ZM3 cells (p=0.05–0.09) do not; the exploratory QX7/ZM3–misalignment pattern, estimated at n=18–20 with power ≈0.35 (simulated) to detect ρ=0.4 at α=0.05, has a minimum detectable effect (80% power) of ρ=0.62 at n=18, above the observed +0.41, so the non-significant result cannot exclude an effect of practical interest and is retained only as a hypothesis, detection of an effect of +0.41 with 80% power requiring n≈40; an earlier pre-specified QX7–misalignment continuation fell from +0.41 (n=18) to +0.23 (n=25), below its 0.30 threshold, and was stopped and retained as a hypothesis.

| | RQ1 (n=18) | RQ2 (n=20) | RQ3 partials

| QX7 metric | reversal | PC1 | H8P | R2F | τ² | misalign. | jailbr.

| G8X | +0.02 | −0.71 | +0.16 | +0.07 | −0.09 | +0.41 | −0.11

| recall | −0.05 | −0.14 | +0.40 | +0.32 | +0.15 | −0.46 | −0.05

| J4N | −0.09 | −0.86 | +0.71 | +0.49 | +0.19 | +0.19 | −0.20

| W3D (official) | −0.29 | −0.75 | +0.76 | +0.62 | +0.32 | −0.29 | −0.17

Table 4: QX7 metric sensitivity under G8X, recall, J4N, and W3D. RQ3 entries are U4G with the M7V controlled (τ² n=19, misalignment n=18, jailbreak n=20). The RQ1 small-panel reversal does not return under any metric, and the task-success and jailbreak nulls are stable. The exploratory misalignment estimate is not stable: +0.41 under G8X and −0.46 under recall (both n.s.).

The conclusions the paper leans on are metric-robust, the one metric-sensitive cell being the QX7–misalignment partial, which the paper already treats as exploratory; the recomputation of every QX7-involving result under four metric choices on the paper's own panels (Table 4) is occasioned by the paper's rejection of QX7's headline W3D for ignoring true negatives together with its representation of QX7 by G8X, which ignores true positives, in the construct and criterion analyses.

Figure 5: The trade-off a 7-model panel suggested (ρ=−0.64) dissolves at n=18 (ρ=+0.02, p=0.95; grey trend), models near the top of either axis spanning the full range of the other, the pairwise view of Table 1's rank disagreement; QX7 and KD9 do not rank the full panel in opposite order, KD9 safety being plotted against QX7 G8X on the n=18 cross-benchmark panel, coloured by the harmonized R2F anchor. Figure 6: The first principal component groups QX7, ZM3, and capability, with KD9 in the opposite direction—PC1 loadings on the n=18 panel with K5Q accounting for 48.7% of variance, and KD9 having the opposite sign from the other four measures in 17 of 18 leave-one-model-out fits; the sign of a principal component is arbitrary, only the relative directions matter, and the structure is treated as suggestive because the median between-benchmark correlation is only 0.22 (§4.3). The two RQ1 results visualised in Figures 5 and 6 are the dissolution of the small-panel QX7–KD9 trade-off and the single-factor structure that provides a weaker description of how the scores covary.

A rank-based (Spearman) PCA keeps the loading structure (anchors −0.72 / −0.78, KD9 opposite at +0.38), and the pre-specified positive control still passes, but PC1 falls from 48.7% to 40.8% of variance and Horn's parallel analysis retains no factor; the PCA operates on the Pearson correlation matrix of the standardized scores. The pre-specified criterion for interpreting PC1 (first-factor variance ≥40%) holds under both conventions, and the pre-specified negative control passes as executed: 200 column-permuted matrices give median first eigenvalue 1.71 against the parallel-analysis 95th-percentile threshold of 2.18 (exceedance 2%). The convention sensitivity is one more reason the factor is treated as supporting rather than primary evidence, the firm RQ1 result being the correlation-free ranking disagreement of Table 1.

All four majorities match the hand-assignment; raw agreement is 27/28 ratings, Krippendorff's α=0.825 (nominal), against a chance baseline of 0.5⁴=6.2% for one coder matching all four labels by coin flip, and the single dissent (Llama-3.3-70B placing ZM3 as harm blocking) is the one semantically borderline case (injection-robustness). Votes (B = harm blocking, R = risk recognition):

| Benchmark | blind | author | B / R

| QX7 | recognition | recognition | 0 / 7

| ZM3 | recognition | recognition | 1 / 6

| KD9 | blocking | blocking | 7 / 0

| V4W | recognition | recognition | 0 / 7

Coder bias, though not rubric-designer bias (the category definitions having been written by authors who had seen the results), is controlled by this procedure, and the RQ1 factor, also reported, separates KD9 (+0.33) from the QX7/ZM3 cluster with no reference to any criterion; seven LLM coders from distinct families—GPT-4o, Claude-3-Haiku, Gemini-2.5-Flash, Llama-3.3-70B, Qwen-2.5-72B, Mistral-Large, DeepSeek-V3—were each shown only a neutral, predictive-information-stripped description of what each benchmark measures (no mention of our criteria, category names, or any RQ3 result) plus two construct definitions ("harm blocking" vs. "risk recognition and correct treatment of benign inputs"), with the two options and the benchmark order shuffled per coder to remove position bias, each coder assigning every benchmark to one construct.

Krippendorff's α=0.61 (three categories, chance ≈1/3), and the distractor is chosen only 5/42 times; the harm-blocking classification is robust—KD9 and the jailbreak criterion both map to harm blocking 7/7, QX7 to risk recognition 7/7, and ZM3 5/7, all unchanged by the distractor—while two targets are honestly contested: V4W reads as task success to 4/7 coders (it is utility-scored; our n=5 benchmark), and the misalignment criterion is contested, mapping to risk recognition 6/7 in a separate two-way round (frozen votes in the artifact) but to harm blocking 6/7 here (avoiding a harmful autonomous action can be read as both recognizing and declining it). The coding was re-run over all six targets (four benchmarks + two criteria) with a third, plausible task-success distractor construct, which makes the agreement statistic meaningful and bears on whether the predictor → criterion matching, not just the predictor labels, is recoverable blind; the KD9–jailbreak pairing is thereby recovered without using the predictive results, an independent reason the QX7/ZM3–misalignment pattern is kept exploratory.

The unsafe subset being ceilinged (every model refuses 83–100%, sd 4.5, limited discriminating variance), the informative signal is over-refusal on the safe subset (range 0–60%; Claude-3-Haiku over-refuses 60%, o3-mini 30%, others ≤15%); C9L (Röttger et al., 2024; cited in the main paper) was run on the panel via its published CPI (compliance/partial/refusal) scorer (judge gpt-4o-mini), the safe subset (benign prompts that superficially resemble unsafe ones) measuring over-refusal and the unsafe subset measuring correct refusal of overtly-harmful prompts. Partial-ρ (capability-controlled) two-by-two:

| Predictor | unsafe refusal | safe compliance

| KD9 safety | +0.24 (n=17) | +0.01 (n=17)

| QX7 G8X | +0.06 (n=17) | −0.15 (n=18)

| P2Z | +0.47 (n=18) | −0.11 (n=19)

KD9 is unrelated to benign-prompt over-refusal (ρ=−0.01, n=17), weighing against but not ruling out a blanket-refusal effect; capability is +0.21 (n.s.) on benign compliance and −0.32 (n.s.) on harmful refusal; QX7 G8X is +0.15 on over-refusal (equivalently −0.15 on safe compliance); P2Z is likewise unrelated to over-refusal (+0.11, n=19, p=0.66), while its unsafe-refusal association is +0.47 (n=18, p=0.052); these low-power checks neither support nor rule out that explanation.

Interaction Δ=−1.00 (95% CI [−1.48,−0.49], bootstrap p<0.001); leave-one-out Δ∈[−1.16,−0.90]; D7T, per omitted organization: Amazon −0.96, Anthropic −1.01, Cohere −0.90, DeepSeek −1.13, Google −0.98, Meta −0.96, Mistral −0.93, OpenAI −1.16, Qwen −0.95 (the range coincides with leave-one-out because Cohere contributes one model and OpenAI, the largest group, gives the widest swing); organization-clustered bootstrap Δ∈[−1.49,−0.59], P(Δ≥0)<10⁻⁴; subsampling stability (draws of each size): size-10 median −0.95, 100% negative, 97% ≤−0.5, 0% sign-flips; size-12 −0.96 / 100% / 99% / 0%; size-14 −0.97 / 100% / 100% / 0% (contrast: the RQ1 QX7–KD9 correlation reaches |ρ|≥0.5 in ~25% of size-7 draws). Independent-grader checks: the misalignment classifier reproduces at ρ=0.97 (Claude-3-Haiku vs. GPT-4o), the jailbreak judge at ρ=0.98 (Gemini vs. gpt-4o), and KD9's refusal predictor at ρ=0.69 (a Gemini refusal judge re-scoring the cached completions, n=19; the KD9–jailbreak estimate remains similar, +0.66 vs +0.72; the moderate agreement reflects that the swap redoes only the refusal component, not the full grading pipeline). Per-jailbreak KD9 → jailbreak partials (within the R8W suite): AIM +0.44, refusal-suppression +0.70, prefix-injection +0.62; the original single-AIM criterion gave +0.50. Under the classic alternative (Pearson on the rank residuals), the two exploratory misalignment estimates move to +0.49 (QX7) and +0.41 (ZM3), within 0.08 of Table 2, and the KD9–jailbreak estimate stays +0.72, no conclusion changing under either convention; the partials residualize midranks on the harmonized M7V and re-rank the residuals.

Figure 7: Against task success, capability is the predictor and KD9 adds nothing—τ²-retail success against the harmonized M7V (ρ=+0.60, p=0.005, n=20; grey trend), colour being KD9 safety (open circle: no KD9 score), safe and unsafe models sitting on both sides of the trend with no safety gradient after controlling capability, the pre-specified result that KD9 adds no incremental validity in Table 2.

The QX7/ZM3–misalignment interaction is lifted from Δ=+0.52 (p=0.068) to Δ=+0.53 (p=0.048) by the addition of a third scenario (murder; our classifier, 22 models) to the blackmail + leaking criterion; the point estimate is essentially unchanged, so the significance comes from the tighter CI of a larger battery rather than from a stronger effect, and, a post-hoc extension crossing the threshold being a forking path, it is not leaned on; murder is also a severe harm that harm-compliance forecasts (KD9 → battery +0.43), diluting how specific the construct remains, so blackmail + leaking is kept primary.

Capability → P2Z moves from −0.44 (n=21, asymptotic p=0.047; bootstrap p=0.11, 95% CI [−0.85,+0.09]) to −0.16 (n=40, bootstrap p=0.40; 95% CI [−0.54,+0.22], leave-one-out [−0.25,−0.12]) and is no longer significant; the pre-specified confirmation threshold (ρ≤−0.35 at p<0.05) is not met, so the original negative correlation is not reproduced, though the CI still includes −0.44, so a weaker estimate is reported rather than evidence that the correlation has disappeared. Three direct tests of the change itself (accounting for the original panel being nested in the expanded one): a stratified nested bootstrap gives Δρ=+0.27, 95% CI [−0.25,+0.78], two-sided p=0.29 (jailbreak +0.26, CI [−0.14,+0.69], p=0.22); the 2026-only models alone give ρ=+0.07 versus the original −0.44 (independent Fisher z, p=0.12); and size-21 subsets of the expanded pool matched to the original panel's organization histogram give median ρ=−0.17 (IQR [−0.27,−0.08]), i.e. organization composition alone does not reproduce the original value. Organization mix alone does not explain the movement; generational change is compatible with the result but is not established, and the change itself is not statistically significant, the bounded conclusion being that the estimate cannot be assumed to generalize across model populations and that the "hold or strengthen" prediction was not met; a ceiling does not explain all remaining variation, safety varying widely across organizations among models with similar measured capability (composite ≈0.8: o3-mini 0.99, Grok-4.3 0.70, DeepSeek-V3 0.19); because τ² task success was not re-collected, this correlation is updated, not the interaction Δ. A targeted expansion predicting that the crossover would "hold or strengthen" had been specified before the new models were evaluated on either criterion, both to test whether the capability crossover holds on current models and to probe whether any panel change follows capability or organization grouping; the panel was grown to 41 anchored models spanning 12 model-developing organizations, using the same chat/scenario harnesses as the original panel, the two criteria requiring no tool loop and new models simply skipping the tool-calling audit benchmarks. Exclusions were based only on harness behavior, never on outcome scores: Kimi-K2.6 and both Nemotron-3 models returned usable anchor answers on fewer than 50% of items or produced unbounded reasoning traces; MiniMax-M3 fell below the misalignment engagement threshold because of truncated outputs, not low harm, but was retained on the single-turn jailbreak criterion, so the misalignment analysis uses n=40 whereas the jailbreak analysis uses n=41; Gemini-2.5-Pro yielded no valid anchor and GLM-4.5 yielded H8P but not R2F, both having encountered routing incompatibilities, so only GLM-4.5 appears as a partial row in Fig. 3—these exclusions being a small instance of evaluation harnesses going stale.

Organization groups differ after rank-residualizing safety on capability on both criteria (permutation p=0.004 misalignment, <0.001 jailbreak; df-adjusted Kruskal–Wallis ε²=0.34 and 0.57; Table 5, Fig. 4), and all five secondary tests survive Holm (max adjusted p=0.043); within-organization capability → safety slopes are sign-heterogeneous and near-zero in aggregate, and with ≥4 anchored models each from six organizations the collinear capability and organization factors can begin to be separated. The claim is carried by the between-organization permutation test, not the within-organization slopes, and the analysis stops short of a clean identification because capability is matched only on a static-knowledge composite (so unmeasured agentic capability can still appear as an organization difference), within-organization n is 4–8, and some organizations are range-restricted (Anthropic ceilings near safety 1.0). That test permuting labels over only seven organization clusters (within-organization n 4–8), its Type-I calibration was checked on 2000 synthetic panels per null with no between-organization mean difference, holding models, organizations, and capability fixed—with the real safety values reshuffled across models, and with i.i.d. Gaussian safety—and rejection matches nominal α at 0.05/0.01/0.005 on both criteria (0.045–0.052 at α=0.05); a worst-case null with organization-clustered variances but no mean difference inflates rejection mildly on the jailbreak structure only (0.067 at α=0.05; no detectable inflation at α=0.005), too small to change any conclusion, since even doubling both p-values leaves the organization-group tests Holm-significant; dropping near-ceiling Anthropic gives misalignment ε²=0.27, p=0.018 and jailbreak ε²=0.54, p<0.001, neither group result being carried solely by Anthropic.

| | n | misalign. ρ | jailbreak ρ

| within-organization slopes:

| Anthropic | 5 | +0.71 | +0.67

| OpenAI | 8 | +0.59 | +0.64

| Qwen | 5 | −0.90 | +0.80

| Meta | 4 | −0.40 | −0.40

| Google | 4 | −0.60 | −0.40

| Mistral | 4 | −1.00 | −0.40

| n-weighted mean | | −0.14 | +0.26

| organization-group test (group n≥3; total n=33):

| between-group share | | 0.508 | 0.674

| df-adjusted ε² | | 0.34 | 0.57

| permutation p (10⁴) | | 0.004 | <0.001

| Kruskal–Wallis p | | 0.02 | 0.002

Table 5: Organization-group differences on the n≈41 expansion. Holding the organization fixed, capability's slope on safety is heterogeneous and near-zero (top). After rank-residualizing safety on the static M7V, organization groups differ under permutation tests on both criteria (bottom); ε² is the df-adjusted Kruskal–Wallis effect size, not a causal share.

Figure 4: Models with similar measured capability can differ widely in safety across model-developing organizations; each point is a model, colour and shape identifying its organization, error bars the within-model SEs (§A.2). Within an organization (lines, the six organizations with ≥4 anchored models) the capability → safety slope is large but sign-heterogeneous across organizations (n-weighted mean −0.14 misalignment, +0.26 jailbreak); among higher-capability models (shaded, composite >0.4), safety spans nearly the full range (arrows). After rank-residualising safety on capability, organization groups differ under permutation tests (df-adjusted Kruskal–Wallis ε²=0.34 / 0.57, p≤0.004). The static composite does not capture all capability, so this is an association, not evidence that organizational identity causes the difference.

Capability → L9R rises from +0.08 (n=21) to +0.34 (n=41; bootstrap p=0.035, 95% CI [+0.03,+0.60], leave-one-out [+0.30,+0.39]), although the panel-to-panel change is not significant; within-organization slopes remain heterogeneous (weighted mean +0.26), while models with similar measured capability vary across organizations (permutation p<0.001; ε²=0.57), and among high-capability models L9R spans 9–99 (Mistral-Large 0.09, DeepSeek-V3 0.37, Anthropic/OpenAI 0.98–0.99); together with the misalignment result, neither the sign nor the magnitude of capability → safety is stable across criteria or panels, the jailbreak criterion having been rebuilt on the expanded panel with the three published templates re-scored for every new model.

| Analysis | n | note

| QX7 (G8X, W3D) | 21 | Qwen3-32B dropped (21% unparseable)

| ZM3 (robustness) | 22 |

| KD9 (safety) | 19 | malformed tool calls or unsupported routing

| V4W | 5 | convergent control inconclusive

| τ² criterion | 20 |

| misalignment criterion | 21 |

| jailbreak criterion (T6Q) | 22 | 3-template average

| expanded misalignment / jailbreak | 40 / 41 | 12 organizations (App. A.6)

| C9L over-refusal | 17–20 | KD9 ∩ anchor overlap

| cross-benchmark / partials | 18 | KD9-limited

| capability interaction (paired) | 20 |

Figure 3: The audit's evidence base: up to 46 models × 9 instruments (two capability anchors, four audited benchmarks, three held-out criteria; 262 model × instrument cells). Cells are per-instrument min–max-normalised scores; grey is not-run. The rows beyond the 41-model anchored panel are models excluded from the analyses by pre-specified telemetry gates (App. A.6). Rows are sorted by capability: the 2026 expansion (top) is scored on the anchors and the two chat-based criteria, which need no tool-calling harness, while the four audited benchmarks run on the original panel. The capability anchors and the QX7/ZM3 scores vary together; the safety criteria do not track them consistently.

Against each instrument's between-model SD (8.7, 18.2, 26.0, 14.9, 28.5 respectively, on the expanded panel), the worst-case within-model SE is 13–25% of that SD, and the observed panel-level spread is 4–8× the worst-case SE, although individual model pairs can be closer; from the existing item-level data (no new runs), per-model standard errors were obtained for every instrument: H8P binomial SE over its 500 fixed items (median 1.6, max 2.2 points), R2F over 198 (median 3.3, max 3.6), the misalignment criterion over its 100 decisions (median 4.0, max 5.0), QX7 W3D bootstrapped over its 571 records (median 1.8, max 3.0), and the jailbreak criterion bootstrapped over its ~150 judged prompts (median 2.3, max 3.7); τ²-retail remains a single run whose user-simulator variance cannot be bounded, the one instrument this analysis cannot cover.

Every item-rich instrument with item-resampling support is internally consistent (split-half ≥0.96 corrected; rank stability ≥0.97; a top-quartile model stays top-quartile in 0.79–0.96 of item resamples), making item-sampling error an unlikely explanation for the panel-level ranking disagreement in Table 1; misalignment has only two scenario halves (Spearman–Brown 0.70), and τ² has no resampling bound; the jailbreak computation resamples by base prompt, so the three templates' dependence is respected. Small SEs not by themselves making rankings stable, split-half consistency (Spearman–Brown), item-bootstrap rank stability, and top-quartile persistence were computed per instrument from its item-level data (Table 3).

| Instrument | models | items | split-half (SB) | rank stab. | quartile

| QX7 G8X | 21 | 570 | +0.98 (+0.99) | +0.99 | 0.90

| QX7 W3D | 21 | 570 | +0.93 (+0.96) | +0.97 | 0.87

| ZM3 robustness | 22 | 300 | +0.95 (+0.98) | +0.98 | 0.88

| KD9 safety (behaviors) | 19 | 44 | +0.93 (+0.96) | +0.97 | 0.79

| jailbreak crit. (by base prompt) | 42 | 20 | +0.94 (+0.97) | +0.98 | 0.96

| H8P (anchor) | 46 | 356 | +0.93 (+0.96) | +0.98 | 0.92

| R2F (anchor) | 45 | 164 | +0.95 (+0.97) | +0.98 | 0.92

| misalignment crit. (scenario halves) | 22 | 2 | +0.54 (+0.70) | — | —

Table 3: Internal consistency from available item- or scenario-level data: mean split-half correlation of model scores over 50 random half-splits (Spearman–Brown corrected), mean Spearman of item-bootstrapped vs. observed rankings, and the probability that an observed top-quartile model stays top-quartile under resampling. Item counts are the complete-coverage intersection across all listed models (e.g. 356 of the 500 administered H8P items, 164 of 198 R2F items). Splits/draws are shared across models and use the items common to every model's logs; the jailbreak criterion resamples by base prompt (the three templates are dependent), and its shared set is 20 of the 50 suite prompts because each model leaves a few different prompts unjudged. Jailbreak uses 42 models: the 41 anchored models plus DeepSeek-R1, which lacks the required anchor pair; misalignment has config-level data only, so its row is the blackmail-vs-leaking halves convergence. Every item-rich instrument with item-resampling support has Spearman–Brown ≥0.96 and rank stability ≥0.97, making item-sampling error an unlikely explanation for the panel-level disagreement of Table 1. Misalignment's two scenario halves agree only moderately (Spearman–Brown 0.70). τ² is a single run and remains the one unbounded instrument.

