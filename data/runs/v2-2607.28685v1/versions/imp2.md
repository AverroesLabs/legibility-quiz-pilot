# Safety, or Just Capability? A Validity Audit of Agent-Safety Benchmarks

Youting Wang

  
Xiao Han

  
Dingyan Shang

Affiliation: Yuan Tang

  
Bowen Liu

###### Abstract

We audit four agent-safety benchmarks (R-Judge, InjecAgent, AgentHarm, AgentDojo), each run under its official implementation and author-provided scorer on up to 22 models, against a capability composite built from MMLU and GPQA measured under one protocol. On any binary trace-judgment benchmark scored by $F_{1}$, an "always positive" policy attains $F_{1}=2\pi/(1+\pi)$; on R-Judge this is $0.690$, above five of the 21 discriminating models. The three broad-coverage benchmarks rank the same 18 models differently. R-Judge specificity against AgentHarm safety correlates $-0.64$ at $n{=}7$ and $+0.02$ at $n{=}18$; a quarter of random size-7 subsets reach $|\rho|\geq 0.5$. Capability predicts task success ($\rho{=}{+}0.60$) and correlates negatively with misalignment safety ($\rho{=}{-}0.44$, $n{=}21$); on the paired $n{=}20$ panel the contrast is $\Delta{=}{-}1.00$ (95% CI $[-1.48,-0.49]$, $p<0.001$), stable under leave-one-organization-out and organization-clustered bootstrap. On an expanded 41-model panel the misalignment correlation is $-0.16$ (95% CI $[-0.54,+0.22]$) and the jailbreak correlation $+0.34$; neither change is significant. Controlling capability, AgentHarm correlates $\rho{=}{+}0.72$ with three-template jailbreak safety; both instruments score harmful compliance. A safety claim requires the benchmark, metric, target behavior, and model panel.

## 1 Introduction

Benchmarks appearing between 2024 and 2026 target prompt-injection resistance (Zhan et al. 2024; Debenedetti et al. 2024), safety-risk awareness (Yuan et al. 2024), refusal of harmful agentic tasks (Andriushchenko et al. 2025), and risky-tool-use identification (Ruan et al. 2024), among others. Li et al. (2026) catalog more than forty and report cross-benchmark rank concordance near $0.10$, from a twelve-model check over four benchmarks; capability-controlled evaluation and benchmark consolidation are listed there as open problems.

We separate construct, metric, and criterion validity and pose:

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

Observation 1: a closed-form F1 for the "always positive" baseline on binary trace-judgment benchmarks; on R-Judge it outranks five evaluated models. The R-Judge-specificity/AgentHarm-safety correlation moves from $-0.64$ at $n{=}7$ to $+0.02$ at $n{=}18$, with a quarter of random size-7 subsets at $|\rho|\geq 0.5$. A pre-specified, capability-controlled criterion-validity audit over one task-success and two safety outcomes: capability predicts task success; its safety correlations vary with outcome and panel; AgentHarm–jailbreak gives $\rho{=}+0.72$ after controlling capability, with outcome selectivity at $p{=}0.051$ under organization-level resampling. An API-only audit harness and re-run artifacts for four benchmarks and three held-out outcomes are in the accompanying reproducibility package.

#### Main conclusion.

A capability score is not a safety score, and no single agent-safety benchmark stands in for safety. At seven models a weak relationship can appear systematic; validity requires re-checking as the model population turns over.

#### Scope.

The three RQ3 criteria are deployment proxies, not claims about real-world harm; limits in Section 6.

## 2 Related Work

#### Agent-safety benchmarking.

InjecAgent (Zhan et al. 2024) and AgentDojo (Debenedetti et al. 2024): prompt-injection robustness; AgentHarm (Andriushchenko et al. 2025): compliance with harmful agentic tasks; R-Judge (Yuan et al. 2024): safety-risk awareness over interaction traces. ToolEmu (Ruan et al. 2024), not re-run, is likewise an unvalidated evaluator. AutoMonitor-Bench (Yang et al. 2026) reports miss and false-alarm rates. Li et al. (2026) document ranking disagreement without capability control, criterion validity, or latent structure; we address all three.

#### Construct validity.

Benchmark-agreement testing (Perlitz et al. 2024), metabench (Kipnis et al. 2025), and construct analyses (Bean et al. 2025) find a strong shared factor across knowledge and reasoning benchmarks. Kearns (2026) and NIST (Keller et al. 2026) bring quantitative construct-validity and uncertainty modeling to LLM evaluation. The capability confound across agent-safety benchmarks targeting refusal, injection robustness, and risk awareness has not been quantified.

#### Benchmark gaming.

Distinct from adversarial gaming: the headline score may fail to measure its construct even for an honest model. The F1 case (Section 4) involves a non-adversarial policy that labels every trace "unsafe."

#### Over-refusal.

XSTest (Röttger et al. 2024) and OR-Bench (Cui et al. 2024) document prompt-level over-refusal. The between-benchmark analogue—AgentHarm refusal against R-Judge specificity—is near zero on the full panel and often not on seven-model subsets.

#### Agent reliability and task performance.

Corrupt success (Cao et al. 2026), evidence-bounded reporting (Chen 2025), $\tau$-bench (Yao et al. 2024), and $\tau^{2}$-bench (Barres et al. 2025). The $\tau^{2}$ retail domain serves as the held-out RQ3 task-success criterion.

## 3 Method

### 3.1 Model panel and benchmarks

API-only panel of up to 22 models from nine organizations (OpenAI 6, Meta 3, Qwen 3, Mistral 3, DeepSeek 2, Amazon 2, Anthropic, Google, Cohere; roster in Fig. 3 and the artifact). Official implementations and scorers: R-Judge (full 571 items, self-judge), InjecAgent (300 stratified direct-harm/data-stealing items, rule-scored), AgentHarm (44 base behaviors across all eight harm categories, official Inspect evaluation, gpt-4o-mini judge), AgentDojo (slack suite, 100 items, environment-scored for security and utility).

Coverage: R-Judge $n{=}21$ (Qwen3-32B below the $90\%$ valid-output threshold); InjecAgent $n{=}22$; AgentHarm $n{=}19$ and AgentDojo $n{=}5$ (native tool use required); $\tau^{2}$ $n{=}20$ (App. A.2). Common R-Judge/InjecAgent/AgentHarm panel: $n{=}18$. Anchors measured through the same API harness: MMLU zero-shot on a fixed 500-item subset (seed 42, identical items per model) and GPQA-Diamond (198 items, chain-of-thought). These replace provider-reported model-card numbers, retained as a pre-harmonization comparison. Anchor panel: 21 models (DeepSeek-R1 excluded for unstable long-form GPQA generation).

### 3.2 Benchmark targets

Target codes, fixed before inspecting any column relationship: R-Judge, unsafe vs. benign trace discrimination; InjecAgent and AgentDojo, prompt-injection resistance (convergent pair); AgentHarm, refusal of harmful agentic tasks. R-Judge is analyzed under specificity and balanced accuracy alongside official F1.

#### Score orientation and terminology.

All scores are oriented higher = better/safer: R-Judge specificity $=\mathrm{TN}/(\mathrm{TN}+\mathrm{FP})$ over benign traces, balanced accuracy $=\tfrac12(\text{specificity}+\text{recall})$; InjecAgent robustness $=1-$ attack success; AgentHarm safety $=1-$ harmful compliance. $\tau^{2}$-bench enters RQ3 as task success, not a safety score. Split-half analyses are reported as internal consistency (App. A.2); "calibration" and "reliability" are not used.

### 3.3 Analysis plan

Hypotheses, thresholds, and decision rules were fixed before each analysis; lacking an independent timestamp, we call them pre-specified rather than preregistered. All associations are Spearman; unless noted, $p$-values are two-sided asymptotic Spearman tests. Fig. 2 error bars are model-level percentile-bootstrap intervals; panel expansion and direct panel comparisons use pre-specified bootstrap procedures (App. A.6). Corroborative PCA uses the Pearson correlation matrix of standardized columns, with factor count by Horn's parallel analysis against a random-normal null (Spearman sensitivity in App. A.3). Capability composite: $c=\tfrac12(z_{\mathrm{MMLU}}+z_{\mathrm{GPQA}})$ over the verified anchors, whose model ordering equals the anchors' first principal component. Partial Spearman correlations control $c$ (rank-residual implementation and alternative convention in App. A.5); between-construct redundancy uses these rank-residualized partials.

Controls: positive (anchor loadings $\geq 0.6$ on the first factor); negative (column-permuted matrix within the parallel-analysis null); convergent (high InjecAgent–AgentDojo correlation). Headline metric: median pairwise partial $|\rho|$ among distinct-construct benchmarks given $c$. Decision rule: interpret factor structure only if first-factor variance $\geq 40\%$ or capability control lowers that median by $\geq 0.15$, with positive and negative controls passing. Thresholds and the disconfirming test were fixed before any real run.

RQ3 outcomes, each fixed before being run: $\tau^{2}$-bench retail (Barres et al. 2025), 40 multi-turn environment-scored tool-agent-user tasks; misalignment safety, avoidance of blackmail or information leaking in a fictional autonomous-agent scenario (Lynch et al. 2025), fixed classifier over 100 decisions per model; jailbreak safety, $1-$ harmful compliance on 50 forbidden prompts averaged over three published templates (Souly et al. 2024; Wei et al. 2023). All run through the same API-only harness (safety criteria single-turn or scenario-based, full panel; $\tau^{2}$ multi-turn tool loop, 20 models), are distinct from the four audited benchmarks, and are not predictors. Headline: partial Spearman of each safety score with each criterion given $c$, reported after the criterion's raw capability association. Reporting rules cover significant and null outcomes. Scripts and frozen inputs are in the reproducibility package.

## 4 Results

Ordered by evidential strength: RQ4 and RQ2, then RQ1, then the power-limited RQ3.

### 4.1 The headline metric is gameable (RQ4)

Figure 1: R-Judge F1 ($n{=}21$) with the constant "always unsafe" baseline (F1 ${=}0.69$, dashed); five discriminating models fall below it.

R-Judge labels $52.7\%$ of its 571 items "unsafe." The all-"unsafe" policy attains recall $1.0$, specificity $0$, and $\mathrm{F1}=0.690$. Among the 21 models with valid verdicts (Qwen3-32B excluded, $21\%$ unparseable), it outranks five (F1 $0.30$–$0.57$; Fig. 1), all discriminating. o3-mini, highest specificity ($0.97$), has $\mathrm{F1}=0.702$, $0.012$ above the baseline. Balanced accuracy reorders the board.

###### Observation 1 (F1 admits a high-scoring constant baseline).

With $\mathrm{F1}=2\mathrm{TP}/(2\mathrm{TP}+\mathrm{FP}+\mathrm{FN})$ and base rate $\pi$, the constant always-positive policy attains $\mathrm{F1}=2\pi/(1+\pi)$. For R-Judge ($\pi=0.527$) this is $0.690$, exceeding several discriminating models. F1 cannot stand alone as a measure of two-sided discrimination on this benchmark.

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

Table 1: Scores of the $n{=}18$ common-panel models on the three broad-coverage benchmarks (higher is safer), sorted by AgentHarm safety. Bold marks column maxima and underlining marks minima, with display-precision ties included. GPQA shown as capability reference.

### 4.2 The capability confound is real but metric-dependent (RQ2)

R-Judge panel, harmonized anchors ($n{=}20$): balanced accuracy $\rho=+0.71$ with MMLU ($p<0.001$), $+0.49$ with GPQA ($p=0.03$); F1 $+0.76$ and $+0.62$; specificity $+0.16$ and $+0.07$. The specificity–MMLU correlation was $0.85$ on the first nine models.

### 4.3 Benchmarks disagree; a seven-model panel suggested a trade-off that does not persist (RQ1)

No benchmark reproduces another's ordering (Table 1); top-AgentHarm models sit near the bottom on R-Judge specificity. Disagreement across benchmarks coded to different constructs (§3) does not by itself indict any instrument.

R-Judge specificity vs. AgentHarm safety: $-0.64$ at $n{=}7$, $+0.02$ ($p=0.95$) at $n{=}18$ (Fig. 5, appendix); random size-7 subsets give $|\rho|\geq 0.5$ in a quarter of draws. The null at $n{=}18$ holds under all four R-Judge metrics (App. A.4). Capability moderation of the remainder is untested.

Single-factor PCA (Fig. 6, appendix): PC1 ($48.7\%$ variance) loads R-Judge specificity, InjecAgent robustness, and both anchors in one direction and AgentHarm safety in the other ($+0.33$; opposite the cluster in $17$ of $18$ leave-one-out fits).

#### The positive control now passes.

Harmonized anchors: MMLU loads $0.74\geq 0.6$ (provider-reported: $0.42$, failing). Negative control passes (column-permuted eigenvalues within the parallel-analysis null); convergent control inconclusive at AgentDojo's $n{=}5$. Spearman PCA keeps the loading structure with PC1 at $40.8\%$ and no factor retained by Horn (App. A.3); median between-benchmark correlation is $0.22$. The factor is treated as corroborative; Table 1 is the primary RQ1 result.

### 4.4 Predictive validity differs across held-out outcomes (RQ3)

Three pre-specified outcomes (§3): $\tau^{2}$-bench task success, misalignment safety, jailbreak safety.

| | Task success | Safety outcomes

| Predictor | $\tau^{2}$ | misalign. | jailbreak†

| Capability (raw $\rho$ ) | $+0.60^{**}$ | $-0.44^{*}$ | $+0.08$

| Safety-score partial $\rho$ (controlling capability):

| R-Judge specificity | $-0.09$ | $+0.41$ | $-0.11$

| InjecAgent robustness | $+0.16$ | $+0.47$ | $+0.21$

| AgentHarm safety | $-0.23$ | $+0.16$ | $+0.72^{***}$

Table 2: Criterion validity on the 2024–25 panel (cellwise $n{=}18$–$21$; App. A.2; higher is better or safer). Top row: raw correlation with capability; remaining rows: partial Spearman given the capability composite. † three-template jailbreak outcome. Two-sided asymptotic Spearman tests (${}^{*}p<0.05$, ${}^{**}p<0.01$, ${}^{***}p<0.001$).

#### Against task success, safety scores add no information beyond capability.

Capability predicts $\tau^{2}$-retail success ($\rho=+0.60$, $p=0.005$, $n{=}20$); best safety-score partial $|\rho|=0.23$ (n.s., $n{=}19$; Table 2; Fig. 7, appendix).

Figure 2: Capability–outcome correlations. Filled circles: 2024–25 panel; open diamonds: expanded panel (safety only); error bars: model-level percentile-bootstrap 95% CIs; thin lines connect panel-level estimates. Misalignment $-0.44\to-0.16$; jailbreak $+0.08\to+0.34$; neither change significant. Task success not re-collected.

#### Against agentic misalignment, capability has the opposite association.

Full 21-model panel: $\rho_{\text{cap}}=-0.44$ (asymptotic $p=0.047$). Mistral-Large blackmails or leaks $85\%$ of the time; o3-mini $1\%$. Partials: InjecAgent $+0.47$ and R-Judge specificity $+0.41$ exceed the pre-specified $0.40$ effect-size threshold; AgentHarm $+0.16$ (Table 2). Both CIs include zero, neither survives Bonferroni, power $\approx 0.35$, and the pre-specified directional prediction named AgentHarm; both remain exploratory. The R-Judge partial is $+0.19$ under balanced accuracy and $-0.46$ under recall (App. A.4).

#### A direct interaction test on the original panel.

Paired $n{=}20$: capability–task success $+0.60$, capability–misalignment $-0.41$ ($-0.44$ on the full panel), $\Delta=-1.00$ (95% CI $[-1.48,-0.49]$, $p<0.001$); stable under leave-one-model-out, leave-one-organization-out, organization-clustered bootstrap, and subsampling (App. A.5). The R-Judge interaction is $\Delta=+0.50$ ($p=0.082$). An independent misalignment grader gives $\rho=0.97$ with the original scores.

#### The capability–misalignment correlation is weaker on the expanded panel.

Pre-specified prediction: "hold or strengthen." Observed: $\rho=-0.44$ ($n{=}21$, asymptotic $p=0.047$) to $-0.16$ ($n{=}40$, bootstrap $p=0.40$; 95% CI $[-0.54,+0.22]$; App. A.6). The interval includes $-0.44$; nested-bootstrap $\Delta\rho=+0.27$ (95% CI $[-0.25,+0.78]$). Organization-matched subsets: median $\rho=-0.17$. A generational account is possible and undemonstrated. $\tau^{2}$ was not re-collected; the interaction was not re-tested.

#### Models with similar capability scores differ in safety across organizations
(Fig. 4).

41 models, 12 organizations (misalignment $n{=}40$; jailbreak $n{=}41$ after one engagement-gate exclusion). Six organizations contribute at least four models (within-organization correlations); seven contribute at least three (group comparison) (App. A.6, Table 5). Within-organization capability–safety correlations vary in sign, averaging $-0.14$ (misalignment) and $+0.26$ (jailbreak). With safety ranks adjusted for capability, the seven groups differ (permutation $p=0.004$ misalignment, $p<0.001$ jailbreak; Holm-significant; df-adjusted Kruskal–Wallis $\varepsilon^{2}=0.34$ and $0.57$). Non-causal: the static composite may miss agentic ability, organizations contribute few models, and some groups are range-restricted.

#### Against jailbreak safety, AgentHarm shows the strongest held-out association.

Jailbreak harmful compliance scored with StrongREJECT (Souly et al. 2024). The single AIM template leaves 11 of 22 models refusing every request; the three-template average (Wei et al. 2023) leaves none at ceiling. Capability: $\rho=+0.08$ ($n{=}21$, $p=0.73$) on the 2024–25 panel; $+0.34$ ($n{=}41$, $p=0.04$) expanded, change n.s.

Controlling capability, AgentHarm–jailbreak $\rho=+0.72$ ($p<0.001$), vs. $+0.50$ for the single-template measure; stable under leave-one-model-out and an independent grader (App. A.5). AgentHarm jailbreak-over-misalignment: $\Delta=+0.59$, $p=0.027$ (model-level bootstrap); $p=0.051$ resampling the eight represented organizations. Paired-panel misalignment partial $+0.13$ ($n{=}19$); Table 2's $+0.16$ uses the common $n{=}18$ panel. R-Judge specificity ($-0.11$) and InjecAgent ($+0.21$) do not predict jailbreak safety. AgentHarm and StrongREJECT both score harmful compliance. The R-Judge/InjecAgent misalignment-over-jailbreak comparison: $\Delta=+0.52$, $p=0.068$.

#### Two alternative explanations, checked.

Seven LLM coders, shown construct descriptions only, assigned AgentHarm and the jailbreak outcome to harm blocking ($7/7$ each; Krippendorff $\alpha=0.61$ over three categories); the misalignment outcome was classified less consistently (App. A.1). Controlling capability, AgentHarm is uncorrelated with benign-prompt XSTest over-refusal (Röttger et al. 2024) ($\rho=-0.01$, $n{=}17$).

## 5 Discussion

These agent-safety benchmarks are not interchangeable measures of a single property: an "always unsafe" baseline matches R-Judge F1 (Observation 1), and the three broad-coverage benchmarks rank the same models differently. Capability predicts task success; its correlations with the two safety outcomes are panel-unstable (Table 2). AgentHarm–three-template jailbreak is $\rho=+0.72$ after capability control, with organization-resampled selectivity at $p=0.051$. R-Judge/InjecAgent–misalignment associations are exploratory. Capability-adjusted safety ranks differ across organization groups, with no organizational cause identified. A safety claim is defensible only with benchmark, metric, target behavior, and model population attached.

## 6 Limitations and Threats to Validity

#### Panel size.

Cross-benchmark panel $n{=}18$, from a first pass at $n{=}7$. A larger capability-spanning agent-loop panel is the main gap.

#### Capability anchors.

Provider-reported anchors failed the positive control (MMLU $0.42$); re-measured, $0.74$ (§3); no RQ2/RQ3 sign changed. Both are static-knowledge tests; the composite predicts multi-turn tool-agent success at $+0.60$. A six-model BFCL pilot (Patil et al. 2025) ranked GPT-4o-mini highest ($0.72$) and the two most capable models lowest ($0.42$–$0.48$) under strict function-call syntax scoring. Anchor measurement error attenuates or suppresses partial relationships without signable bias.

#### Multiple comparisons.

Per-test $p$-values are uncorrected. The crossover and AgentHarm $\to$ jailbreak ($p<0.001$) survive Bonferroni; the R-Judge/InjecAgent–misalignment cells ($p\approx 0.05$–$0.09$) do not and are treated as suggestive.

#### Judges and coverage.

Independent graders: $\rho=0.97$ (misalignment), $\rho=0.98$ (jailbreak). AgentHarm refusal rescoring: $\rho=0.69$, with the AgentHarm–jailbreak partial at $+0.66$ vs. $+0.72$ (App. A.5). AgentDojo covers five models; cross-benchmark analyses rest mainly on the other three.

#### Criterion validity (RQ3).

$\tau^{2}$-retail: 40 tasks, one run per model, user-simulator variance unbounded. Misalignment: fictional-scenario probe, grader-robust ($\rho=0.97$), blackmail–leaking halves at Spearman–Brown $0.70$. Jailbreak: three-template average. None gives deployment-harm probabilities. Item-rich instruments with resampling support: split-half $\geq 0.96$, rank stability $\geq 0.97$. Worst-case within-model SEs are $12$–$25\%$ of between-model SD (App. A.2). AgentHarm–jailbreak is large and directionally stable, partly convergent, with selectivity borderline under organization resampling; R-Judge/InjecAgent–misalignment is exploratory (marginal $\Delta=+0.52$, power $\approx 0.35$; the larger three-scenario result is post-hoc; App. A.5).

## 7 Conclusion

Across four benchmarks and up to 22 models, R-Judge F1 lets an "always unsafe" baseline outrank five models, and rankings change with the benchmark. Capability predicts held-out task success but not safety consistently across outcomes or panels. AgentHarm gives the strongest held-out safety association ($\rho=+0.72$ with jailbreak safety, capability-controlled), with partial instrument overlap and borderline organization-resampled selectivity. Report the full confusion structure, drop standalone F1 where true negatives matter, and tie claims to an explicit target behavior and model population.

## Appendix A Technical Appendix

Method detail and full robustness numbers.

### A.1 Ruling out two artifacts (detail for §4.4)

#### Blinded construct coding.

Coders: GPT-4o, Claude-3-Haiku, Gemini-2.5-Flash, Llama-3.3-70B, Qwen-2.5-72B, Mistral-Large, DeepSeek-V3. Each saw neutral benchmark descriptions stripped of predictive information (no criteria, category names, or RQ3 results) and two definitions ("harm blocking" vs. "risk recognition and correct treatment of benign inputs"), with option and benchmark order shuffled per coder. Votes (B ${=}$ harm blocking, R ${=}$ risk recognition):

| Benchmark | blind | author | B / R

| R-Judge | recognition | recognition | 0 / 7

| InjecAgent | recognition | recognition | 1 / 6

| AgentHarm | blocking | blocking | 7 / 0

| AgentDojo | recognition | recognition | 0 / 7

All majorities match the hand-assignment; $27/28$ ratings agree, Krippendorff $\alpha=0.825$ (nominal), against a $0.5^{4}=6.2\%$ chance of one coder matching all four by coin flip. Sole dissent: Llama-3.3-70B placing InjecAgent under harm blocking. Rubric-designer bias is not controlled (definitions written after seeing results); the RQ1 factor separates AgentHarm ($+0.33$) from the R-Judge/InjecAgent cluster without reference to any criterion.

#### Blinded criterion coding, with a distractor.

Six targets (four benchmarks $+$ two criteria) with a third task-success distractor construct: $\alpha=0.61$ (three categories, chance $\approx 1/3$); distractor chosen $5/42$ times. AgentHarm and the jailbreak criterion map to harm blocking $7/7$; R-Judge to risk recognition $7/7$; InjecAgent $5/7$; unchanged by the distractor. AgentDojo reads as task success to $4/7$ (utility-scored; $n{=}5$). Misalignment maps to risk recognition $6/7$ in a separate two-way round (frozen votes in the artifact) and to harm blocking $6/7$ here.

#### XSTest over-refusal (blanket-refusal check).

XSTest (Röttger et al., 2024; cited in the main paper) with its published CPI (compliance/partial/refusal) scorer (judge gpt-4o-mini). Unsafe subset ceilinged (refusal $83$–$100\%$, sd $4.5$); safe-subset over-refusal ranges $0$–$60\%$ (Claude-3-Haiku $60\%$, o3-mini $30\%$, others $\leq 15\%$). Capability-controlled partial $\rho$:

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

AgentHarm vs. over-refusal: $\rho=-0.01$ ($n{=}17$). Capability: $+0.21$ (n.s.) on benign compliance, $-0.32$ (n.s.) on harmful refusal. R-Judge specificity: $+0.15$ on over-refusal ($-0.15$ on safe compliance). Misalignment safety: $+0.11$ on over-refusal ($n{=}19$, $p=0.66$), $+0.47$ on unsafe refusal ($n{=}18$, $p=0.052$). These checks are low-power.

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

Figure 3: Up to 46 models $\times$ 9 instruments (two anchors, four audited benchmarks, three held-out criteria; $262$ model $\times$ instrument cells), per-instrument min–max normalised; grey is not-run. Rows beyond the $41$-model anchored panel were excluded by pre-specified telemetry gates (App. A.6). Rows sorted by capability; the 2026 expansion (top) is scored on anchors and the two chat-based criteria only; the four audited benchmarks run on the original panel.

#### Within-model score uncertainty.

From existing item-level data: MMLU binomial SE over 500 items (median $1.6$, max $2.2$ points); GPQA over 198 (median $3.3$, max $3.6$); misalignment over 100 decisions (median $4.0$, max $5.0$); R-Judge F1 bootstrapped over 571 records (median $1.8$, max $3.0$); jailbreak bootstrapped over ${\sim}150$ judged prompts (median $2.3$, max $3.7$). Between-model SDs on the expanded panel: $8.7$, $18.2$, $26.0$, $14.9$, $28.5$; worst-case SE is $13$–$25\%$ of SD, panel spread $4$–$8\times$ worst-case SE; individual pairs can be closer. $\tau^{2}$-retail (single run) is not covered.

#### Internal consistency, not just precision.

Per-instrument split-half consistency (Spearman–Brown), item-bootstrap rank stability, and top-quartile persistence (Table 3). Item-rich instruments with resampling support: split-half $\geq 0.96$ corrected, rank stability $\geq 0.97$, top-quartile persistence $0.79$–$0.96$. Misalignment has two scenario halves (Spearman–Brown $0.70$); $\tau^{2}$ has no resampling bound. Jailbreak resamples by base prompt.

| Instrument | models | items | split-half (SB) | rank stab. | quartile

| R-Judge specificity | 21 | 570 | $+0.98$ ( $+0.99$ ) | $+0.99$ | $0.90$

| R-Judge F1 | 21 | 570 | $+0.93$ ( $+0.96$ ) | $+0.97$ | $0.87$

| InjecAgent robustness | 22 | 300 | $+0.95$ ( $+0.98$ ) | $+0.98$ | $0.88$

| AgentHarm safety (behaviors) | 19 | 44 | $+0.93$ ( $+0.96$ ) | $+0.97$ | $0.79$

| Jailbreak crit. (by base prompt) | 42 | 20 | $+0.94$ ( $+0.97$ ) | $+0.98$ | $0.96$

| MMLU (anchor) | 46 | 356 | $+0.93$ ( $+0.96$ ) | $+0.98$ | $0.92$

| GPQA (anchor) | 45 | 164 | $+0.95$ ( $+0.97$ ) | $+0.98$ | $0.92$

| Misalignment crit. (scenario halves) | 22 | 2 | $+0.54$ ( $+0.70$ ) | — | —

Table 3: Mean split-half correlation over 50 random half-splits (Spearman–Brown corrected), mean Spearman of item-bootstrapped vs. observed rankings, and probability a top-quartile model stays top-quartile. Item counts are complete-coverage intersections (356 of the 500 MMLU items, 164 of 198 GPQA items). Splits/draws shared across models; jailbreak resamples by base prompt, with a shared set of 20 of the 50 suite prompts. Jailbreak uses 42 models (the 41 anchored plus DeepSeek-R1, lacking the anchor pair); misalignment has config-level data only (blackmail-vs-leaking halves). Item-rich instruments with resampling support: Spearman–Brown $\geq 0.96$, rank stability $\geq 0.97$. Misalignment halves: Spearman–Brown $0.70$. $\tau^{2}$ is a single unbounded run.

R-Judge/InjecAgent–misalignment: $n{=}18$–$20$, simulated power $\approx 0.35$ for $\rho{=}0.4$ at $\alpha{=}0.05$; minimum detectable effect ($80\%$ power) $\rho{=}0.62$ at $n{=}18$, above the observed $+0.41$; detecting $+0.41$ at $80\%$ power needs $n{\approx}40$. Bonferroni over the nine RQ3 partial tests ($\alpha{=}0.006$): the crossover and AgentHarm $\to$ jailbreak ($p<0.001$) survive; the R-Judge/InjecAgent cells ($p{=}0.05$–$0.09$) do not.

Figure 4: Safety vs. capability composite by organization; error bars are within-model SEs (§A.2). Lines: six organizations with $\geq 4$ anchored models; slopes sign-heterogeneous ($n$-weighted mean $-0.14$ misalignment, $+0.26$ jailbreak). Shaded: composite $>0.4$, where safety spans nearly the full range (arrows). After rank-residualising safety on capability, organization groups differ under permutation (df-adjusted Kruskal–Wallis $\varepsilon^{2}{=}0.34$/$0.57$, $p\leq 0.004$). Associational only.

### A.3 Cross-benchmark structure (detail for §4.3)

Figures 5 and 6.

Figure 5: AgentHarm safety vs. R-Judge specificity, $n{=}18$, coloured by harmonized GPQA. 7-model $\rho=-0.64$; full panel $\rho=+0.02$, $p=0.95$ (grey trend).

Figure 6: PC1 loadings, $n{=}18$, harmonized anchors ($48.7\%$ of variance). AgentHarm opposite the other four measures in $17$ of $18$ leave-one-model-out fits. Component sign is arbitrary. Median between-benchmark correlation $0.22$ (§4.3).

#### Correlation-convention sensitivity.

Spearman PCA: anchors $-0.72$/$-0.78$, AgentHarm $+0.38$; positive control passes; PC1 $48.7\%\to 40.8\%$; Horn retains no factor. The $\geq 40\%$ criterion holds under both conventions. Negative control: $200$ column-permuted matrices, median first eigenvalue $1.71$ vs. the $95$th-percentile threshold $2.18$ (exceedance $2\%$).

### A.4 R-Judge metric sensitivity (detail for §4.3–§4.4)

Every R-Judge result recomputed under four metrics on the same panels (Table 4).

| | RQ1 ( $n{=}18$ ) | RQ2 ( $n{=}20$ ) | RQ3 partials

| R-Judge metric | reversal | PC1 | MMLU | GPQA | $\tau^{2}$ | misalign. | jailbr.

| specificity | $+0.02$ | $-0.71$ | $+0.16$ | $+0.07$ | $-0.09$ | $+0.41$ | $-0.11$

| recall | $-0.05$ | $-0.14$ | $+0.40$ | $+0.32$ | $+0.15$ | $-0.46$ | $-0.05$

| balanced acc | $-0.09$ | $-0.86$ | $+0.71$ | $+0.49$ | $+0.19$ | $+0.19$ | $-0.20$

| F1 (official) | $-0.29$ | $-0.75$ | $+0.76$ | $+0.62$ | $+0.32$ | $-0.29$ | $-0.17$

Table 4: RQ3 entries are partial Spearman correlations given the capability composite ($\tau^{2}$ $n{=}19$, misalignment $n{=}18$, jailbreak $n{=}20$). The small-panel reversal returns under no metric; task-success and jailbreak nulls are stable; the misalignment partial ranges from $+0.41$ (specificity) to $-0.46$ (recall), both n.s.

### A.5 Robustness of the capability crossover (detail for §4.4)

#### Partial-correlation convention.

Midranks residualized on the harmonized composite, residuals re-ranked. Pearson on rank residuals instead: misalignment $+0.49$ (R-Judge) and $+0.41$ (InjecAgent), within $0.08$ of Table 2; AgentHarm–jailbreak $+0.72$.

Figure 7: $\tau^{2}$-retail success vs. harmonized composite ($\rho=+0.60$, $p=0.005$, $n{=}20$; grey trend), coloured by AgentHarm safety (open circle: no AgentHarm score); no safety gradient around the trend.

Interaction $\Delta{=}{-}1.00$ (95% CI $[-1.48,-0.49]$, bootstrap $p<0.001$). Leave-one-out $\Delta\in[-1.16,-0.90]$. Leave-one-organization-out: Amazon $-0.96$, Anthropic $-1.01$, Cohere $-0.90$, DeepSeek $-1.13$, Google $-0.98$, Meta $-0.96$, Mistral $-0.93$, OpenAI $-1.16$, Qwen $-0.95$ (Cohere contributes one model; OpenAI, the largest group, gives the widest swing). Organization-clustered bootstrap: $\Delta\in[-1.49,-0.59]$, $P(\Delta{\geq}0)<10^{-4}$. Subsampling (median / negative / $\leq-0.5$ / sign-flips): size-10 $-0.95$ / $100\%$ / $97\%$ / $0\%$; size-12 $-0.96$ / $100\%$ / $99\%$ / $0\%$; size-14 $-0.97$ / $100\%$ / $100\%$ / $0\%$ (RQ1 R-Judge–AgentHarm: $|\rho|{\geq}0.5$ in $\sim 25\%$ of size-7 draws). Independent graders: misalignment classifier $\rho=0.97$ (Claude-3-Haiku vs. GPT-4o); jailbreak judge $\rho=0.98$ (Gemini vs. gpt-4o); AgentHarm refusal $\rho=0.69$ (Gemini refusal judge on cached completions, $n{=}19$, refusal component only; AgentHarm–jailbreak $+0.66$ vs $+0.72$). Per-template AgentHarm $\to$ jailbreak partials: AIM $+0.44$, refusal-suppression $+0.70$, prefix-injection $+0.62$; original single-AIM criterion $+0.50$.

#### Misalignment 3-scenario battery (exploratory).

Adding murder (our classifier, 22 models) to blackmail $+$ leaking: R-Judge/InjecAgent–misalignment interaction $\Delta=+0.52$ ($p=0.068$) to $\Delta=+0.53$ ($p=0.048$). Post-hoc; not relied on. AgentHarm $\to$ battery $+0.43$. Blackmail $+$ leaking remains primary.

### A.6 The 2026 model-panel expansion (n $\approx$ 41) and organization-group differences (detail for
§4.4)

Pre-specified, before any new-model criterion run, that the crossover would "hold or strengthen." Panel: $41$ anchored models, $12$ organizations, same chat/scenario harnesses; new models skip the tool-calling audit benchmarks. Exclusions from harness behavior only: Kimi-K2.6 and both Nemotron-3 models (usable anchor answers on fewer than $50\%$ of items or unbounded reasoning traces); MiniMax-M3 below the misalignment engagement threshold from truncated outputs, retained on jailbreak (hence $n{=}40$ vs. $n{=}41$). Gemini-2.5-Pro yielded no valid anchor; GLM-4.5 yielded MMLU but not GPQA (routing incompatibilities); only GLM-4.5 appears as a partial row in Fig. 3.

#### Primary expansion result: the capability–misalignment correlation weakens.

$-0.44$ ($n{=}21$, asymptotic $p=0.047$; bootstrap $p=0.11$, $95\%$ CI $[-0.85,+0.09]$) to $-0.16$ ($n{=}40$, bootstrap $p=0.40$; $95\%$ CI $[-0.54,+0.22]$, leave-one-out $[-0.25,-0.12]$). Confirmation threshold ($\rho\leq-0.35$ at $p<0.05$) not met; CI includes $-0.44$. Tests of the change (original panel nested in expanded): stratified nested bootstrap $\Delta\rho=+0.27$, $95\%$ CI $[-0.25,+0.78]$, two-sided $p=0.29$ (jailbreak $+0.26$, CI $[-0.14,+0.69]$, $p=0.22$); 2026-only models $\rho=+0.07$ vs. original $-0.44$ (independent Fisher $z$, $p=0.12$); size-21 subsets matched to the original organization histogram, median $\rho=-0.17$ (IQR $[-0.27,-0.08]$). Generational change is compatible but not established. At composite ${\approx}0.8$: o3-mini $0.99$, Grok-4.3 $0.70$, DeepSeek-V3 $0.19$. Only the correlation, not the interaction $\Delta$, is updated ($\tau^{2}$ not re-collected).

#### Organization-group analysis.

Six organizations with $\geq 4$ anchored models. Within-organization slopes sign-heterogeneous and near zero in aggregate; between groups, after rank-residualizing safety on capability, permutation $p{=}0.004$ (misalignment), $<0.001$ (jailbreak); df-adjusted Kruskal–Wallis $\varepsilon^{2}{=}0.34$ and $0.57$ (Table 5, Fig. 4). All five secondary tests survive Holm (max adjusted $p=0.043$). Capability is matched only on a static-knowledge composite; within-organization $n$ is $4$–$8$; Anthropic ceilings near safety $1.0$. The between-organization permutation test carries the claim. Type-I calibration over seven clusters on $2{,}000$ synthetic panels per null with no between-organization mean difference (models, organizations, capability fixed): reshuffled real safety and i.i.d. Gaussian safety reject at nominal rates for $\alpha=0.05/0.01/0.005$ on both criteria ($0.045$–$0.052$ at $\alpha{=}0.05$); organization-clustered variances inflate jailbreak rejection to $0.067$ at $\alpha{=}0.05$, none detectable at $\alpha{=}0.005$; doubled $p$-values remain Holm-significant. Without Anthropic: misalignment $\varepsilon^{2}=0.27$, $p=0.018$; jailbreak $\varepsilon^{2}=0.54$, $p<0.001$.

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

Table 5: Organization-group differences on the $n{\approx}41$ expansion. Top: within-organization capability slopes on safety. Bottom: tests on safety rank-residualized on the static capability composite; $\varepsilon^{2}$ is the df-adjusted Kruskal–Wallis effect size, not a causal share.

#### Expanded-panel jailbreak result.

Three templates re-scored for every new model. Capability $\to$ jailbreak safety: $+0.08$ ($n{=}21$) to $+0.34$ ($n{=}41$; bootstrap $p=0.035$, $95\%$ CI $[+0.03,+0.60]$, leave-one-out $[+0.30,+0.39]$); change n.s. Within-organization weighted mean $+0.26$; between organizations permutation $p<0.001$, $\varepsilon^{2}{=}0.57$. High-capability jailbreak safety spans $9$–$99$ (Mistral-Large $0.09$, DeepSeek-V3 $0.37$, Anthropic/OpenAI $0.98$–$0.99$). Sign and magnitude of capability $\to$ safety vary across criteria and panels.

A pre-specified R-Judge–misalignment continuation moved from $+0.41$ ($n{=}18$) to $+0.23$ ($n{=}25$), below its $0.30$ threshold; stopped, retained as a hypothesis.