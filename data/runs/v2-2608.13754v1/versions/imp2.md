# Explanation Multiplicity: Circuit-Level Interpretability

Evidence Does Not Survive Defensible Analytic Variation

Ajay Pravin Mahale

12 August 2026

###### Abstract

Annex IV of the EU AI Act requires technical documentation of how high-risk systems reach decisions. We test circuit-discovery evidence under variation of defensible analytic settings, holding system and tool fixed.

A pre-registered crossed grid over seven analytic axes, every level from a published implementation, with each circuit passed through a deterministic claim map to a structured Annex IV statement: 15,840 pre-registered specifications on GPT-2 small / indirect object identification, 7,561 yielding a claim. The statement flips across 73.2% of specification pairs (95% CI 0.725 to 0.738); modal claim 41.1%. Filability fails at every plausible tolerance.

Metric standardised: 59.4%. Size removed from the claim and held fixed: 27.1% (95% CI 0.255 to 0.286), above the pre-registered threshold. Median pairwise Jaccard 4%; Cohen's kappa 0.015.

The filability criterion is given as a standalone protocol. One of the seven documented discovery objectives does not execute on the library's own canonical task. One model, one task; scale untested.

## 1 Introduction

Regulation (EU) 2024/1689: Annex IV point 2(b), "the general logic of the AI system and of the algorithms"; point 2(e), "an assessment of the technical measures needed to facilitate the interpretation of the outputs"; Article 86(1), "clear and meaningful explanations of the role of the AI system in the decision-making procedure and the main elements of the decision taken". Consolidated text as amended by Regulation (EU) 2026/1744, CELEX 02024R1689-20260727, read 11 August 2026.

We quantify movement of a regulatory claim across defensible analytic specifications of circuit discovery and attach a decision rule. No new discovery method is proposed. No claim is made that interpretability is worthless.

#### Contributions.

(i) A pre-registered multiverse over seven analytic axes of circuit discovery, all levels from published implementations, with a deterministic circuit-to-Annex IV map.
(ii) The criterion $\pi^{*}\geq 1-\alpha$ and its value on a canonical task.
(iii) Persistence under standardisation of any single axis, removal of size from the claim, and size held fixed.
(iv) Three pooled statistics that invert within size.

## 2 Related work

#### Non-identifiability.

Méloux et al. [6]: under mechanistic interpretability's own criteria a unique explanation need not exist. We take this as premise and measure its cost for filed rather than published explanations.

#### Faithfulness metric sensitivity.

Miller et al. [8]: ablation-based faithfulness is highly sensitive to seemingly insignificant design choices. One axis there; seven crossed here, propagated to a regulatory statement. Their library is our instrument.

#### Structure versus function.

Bayat Makou et al. [1]: structurally distinct circuits implement the same computation under varied input statistics (phantom specialisation). Addressed functionally in Section 7.6.

#### Variance as estimation.

Méloux et al. [7] analyse the variance of mechanistic interpretability as an estimator; seed is one of our axes.

#### Auditability.

Lan et al. [4] document conflicting conclusions about one behaviour, a third paper finding both partly correct and incomparable, and call for auditing standards. Sharkey et al. [10] list open problems including socio-technical ones; Mueller et al. [9] argue for standardised cross-method evaluation.

#### Multiverse method.

Steegen et al. [13]; specification curve after Simonsohn et al. [12]; undisclosed analytic flexibility, Simmons et al. [11].

#### Explanation metrics under optimisation pressure.

Hsia et al. [3]: sufficiency and comprehensiveness inflate without change in predictions or explanations. Both are among our four metrics; discard rates are reported per metric.

## 3 What this paper is not claiming

Not that discovered circuits are no better than random: at fixed size, 0.2746 against a size-matched null at 0.4230.

Not that mechanisms coincide: functional instability 0.3972 pooled, 0.4777 at the smallest circuits.

Not one mechanism under different descriptions: Cohen's kappa 0.0146.

Not seeds: the seed selects sampled prompts.

Nothing about scale. One model, one task.

## 4 Formalism

$s=(o,a,d,m,\tau,P,r,g)$ over objective, ablation operator, corruption distribution, evaluation metric, size threshold, prompt variant, seed, granularity; no level was invented for this study.

$C(s)\subseteq E$. GPT-2 small, factorised graph with separate query, key and value inputs: $|E|=32{,}491$, confirmed from the instrument and by closed-form count (block $b$ attention receives from $1+13b$ sources at three input slots for each of 12 heads; MLP $b$ from those sources plus its own block's heads; output from all 157 non-terminal sources). Components: 144 attention heads and 12 MLPs, 156 in total.

#### Circuit instability.

| | $D(s_{i},s_{j})=1-\frac{|C_{i}\cap C_{j}|}{|C_{i}\cup C_{j}|},$
| | (1)

with $\bar{J}$ the mean over unordered pairs.

#### Claim instability.

| | $F=1-\frac{\sum_{c}n_{c}(n_{c}-1)}{N(N-1)},$
| | (2)

$n_{c}$ the count of specifications whose $\varphi$-image is class $c$, $N$ the number of specifications; unit-tested against brute-force pairwise computation over randomised cases. $\pi^{*}=\max_{c}n_{c}/N$.

| | $F_{\text{within}}(G)=1-\frac{\sum_{g}\sum_{c}n_{g,c}(n_{g,c}-1)}{\sum_{g}n_{g}(n_{g}-1)},$
| | (3)

pair-weighted, equal to Equation 2 for empty $G$.

#### Filability.

Tolerance $\alpha$, criterion $\pi^{*}\geq 1-\alpha$.

$F\leq 1-1/k$; for the nine observed classes, 0.8889, of which 0.7316 is 82.3%.

## 5 The claim map

$\varphi$ is pre-registered, deterministic, published code, not language-model generated.

$\varphi_{\text{overseer}}$: Annex IV 2(e) and 3 with Article 14(4)(c); keys dominant layer band, size class, ranked input segment. $\varphi_{\text{affected}}$: Article 86(1); input segment first. Three nested granularities.

Size bins are the output of a committed calibration rule on a measured node-count curve (Table 1).

Table 1: Flip rate under alternative size binnings. Four alternatives exceed the committed value, including halving every threshold. Across a factor of four, 0.70 to 0.77; no size term, 0.5733.

| Binning | $F$

| committed bins | 0.7316

| all thresholds $\times$ 0.5 | 0.7448

| all thresholds $\times$ 0.8 | 0.7435

| all thresholds $\times$ 0.9 | 0.7394

| all thresholds $\times$ 1.1 | 0.7289

| all thresholds $\times$ 1.25 | 0.7314

| all thresholds $\times$ 1.5 | 0.7308

| all thresholds $\times$ 2 | 0.7232

| equal thirds | 0.7018

| deciles | 0.7719

| no size term (COARSE) | 0.5733

## 6 Method

#### Model and task.

GPT-2 small, indirect object identification [15], the scale of the literature extended here.

#### Grid.

Seven objectives, seven ablation operators, four corruption distributions nested within the five operators that read them (two operators never read the corrupt distribution), four metrics, three thresholds, two prompt variants, five seeds: 1,540 discovery cells, 18,480 specifications pre-registered. Edge attribution patching family [14] and integrated edge gradients, library of Miller et al. [8]; lineage Conmy et al. [2]; corruption levels from Wang et al. [15].

#### Pre-registration.

Plan committed and tagged before the confirmatory sweep, both abstracts drafted in advance. Deviations in an append-only file with dates and whether decided before or after the affected result was seen.

#### Statistics.

No $p$-values across specifications [13, 12]. Nonparametric bootstrap over specifications, never pairs, $B=10{,}000$; $F$ and $\bar{J}$ are U-statistics over pairs. Intervals are conditional on the grid.

#### Two exclusions, criteria fixed before any circuit was seen.

One objective: all 220 cells raise a runtime error (mean squared error against an integral target, inside library code); digest over sorted failing cell identifiers confirms the failure set is exactly that arm. Not repaired; library reproduced verbatim. Realised grid: six objectives, 1,320 cells, 15,840 specifications.

Metric-relative threshold rule: no circuit for 8,279 of 15,840, discard 52.3%, uneven (Table 2), disclosed before the lock. Surviving pool 52% comprehensiveness against 25% by design.

Table 2: Discard rate by evaluation metric under the pre-registered metric-relative threshold rule, disclosed before lock from a six-cell validation slice.

| Metric | Kept | Discarded | Rate

| comprehensiveness | 3,947 | 13 | 0.003

| KL divergence | 1,467 | 2,493 | 0.630

| logit difference | 1,270 | 2,690 | 0.679

| sufficiency | 877 | 3,083 | 0.779

## 7 Results

Table 3: Pre-registered outcomes. Thresholds fixed 6 August 2026, plan tagged before the sweep, none revisited. Within-size qualifications in Sections 7.5 and 7.6.

| | Decision rule | Result | Outcome

| P0 | $\bar{J}$ reported, not tested | 0.1396 [0.1377, 0.1418] | premise holds

| H2 | $F>0.20$ , CI lower bound above | 0.7316 [0.7247, 0.7380] | confirmed

| H3 | not separated from random null | separated, intervals disjoint | rejected

| H4 | $F-(1-\text{agreement})>0.10$ | 0.3344 [0.3297, 0.3394] | supported

### 7.1 Circuits are almost disjoint

$\bar{J}=0.1396$ (95% CI 0.1377 to 0.1418), exact over all 28,580,580 pairs; 7,561 specifications, 3,218 distinct circuits.

Quantiles of $D$: 1% 0.196, 25% 0.788, median 0.960, 75% 0.998. Median shared edges 4%; 1% of pairs above Jaccard 0.8. Median size 200 edges; random line $k/(2|E|-k)$ gives 0.0031, observed about 45 times higher. At 500 edges, expected random overlap 0.008.

### 7.2 The claim flips on three quarters of pairs

$F=0.7316$ (95% CI 0.7247 to 0.7380), over three times the 0.20 threshold. $\pi^{*}=0.4109$: 0.411 against 0.95, 0.90 and 0.80.

Table 4: Classes of $\varphi_{\text{overseer}}$, medium granularity. Marginal over size: 47.9% early, 43.6% late, 8.5% middle; the first equals the independently computed COARSE modal share 0.4788.

| Rank | Layer band | Size class | Count | Share

| 1 | early layers | distributed | 3,107 | 0.4109

| 2 | late layers | sparse | 2,153 | 0.2848

| 3 | late layers | moderate | 657 | 0.0869

| 4 | late layers | distributed | 489 | 0.0647

| 5 | early layers | sparse | 477 | 0.0631

| 6 | middle layers | moderate | 283 | 0.0374

| 7 | middle layers | sparse | 269 | 0.0356

| 8 | middle layers | distributed | 90 | 0.0119

| 9 | early layers | moderate | 36 | 0.0048

Ranks 1 and 2, 41.1% and 28.5%: early versus late layers.

Table 5; affected-person COARSE, two classes, flips on 39.5%. Figure 1: modal step width is $\pi^{*}$.

Table 5: Both addressees, three granularities, at $\alpha=0.20$ ($\pi^{*}\geq 0.80$).

| Addressee | Granularity | Classes | $F$ | $\pi^{*}$ | Filable

| affected | COARSE | 2 | 0.3950 | 0.7291 | no

| affected | MEDIUM | 4 | 0.6597 | 0.4875 | no

| affected | FINE | 8 | 0.6744 | 0.4786 | no

| overseer | COARSE | 3 | 0.5733 | 0.4788 | no

| overseer | MEDIUM | 9 | 0.7316 | 0.4109 | no

| overseer | FINE | 12 | 0.7681 | 0.4109 | no

Figure 1: Specification curve, $\varphi_{\text{overseer}}$, medium granularity. Upper: class against rank, ordered by frequency; dashed line, size-matched null modal share (pooled comparison size-confounded, Section 7.5). Lower: active level per specification; solid comprehensiveness band per Table 2; seed rows unstructured, cf. 0.0009 in Table 6.

### 7.3 No analytic axis rescues the filing

Table 6: Residual flip rate per standardised axis, Equation 3. Group sizes 1,512 to 2,520; bootstrap bias below 0.0006 on every axis; every observed value inside its interval.

| Axis standardised | Residual $F$ | 95% CI | Removes

| Evaluation metric | 0.5939 | 0.5803 to 0.6063 | 0.1377

| Ablation operator | 0.7014 | 0.6939 to 0.7075 | 0.0302

| Discovery objective | 0.7018 | 0.6938 to 0.7084 | 0.0298

| Threshold $\tau$ | 0.7070 | 0.6981 to 0.7149 | 0.0246

| Corruption | 0.7092 | 0.7008 to 0.7166 | 0.0224

| Prompt variant | 0.7232 | 0.7157 to 0.7300 | 0.0084

| Seed | 0.7307 | 0.7234 to 0.7368 | 0.0009

Metric standardised: $F=0.5939$, lower bound 0.5803.

REML on $\log_{10}$ selected circuit size: ablation-corruption interaction 78.8%, residual 11.5%, ablation 4.4%, threshold 4.1%, metric 0.7%.

### 7.4 The result under maximum pressure

Claim reduced to dominant layer band, size held fixed via Equation 3: $F=0.2706$ (95% CI 0.2550 to 0.2859); 27% of pairs.

### 7.5 The random baseline, and why its pooled direction misleads

Pooled, against a size-matched null from the full 32,491-edge population, $R=1{,}000$ null multiverses: 0.7316 against 0.6774, intervals disjoint. H3 rejected; direction not reported as a finding.

The null touches all 156 components above 1,000 edges (one claim). Within size: discovered 0.2746 against null 0.4230 at medium, 0.2706 against 0.3822 at coarse.

Table 7: Within-size flip rate, medium granularity. Null single-class at and above 1,000 edges; discovered single-class at 5,000 and 10,000, so 2,692 of 7,561 specifications contribute no flips and all instability is below 2,000 edges.

| Size | $n$ | Discovered $F$ | Classes | Null $F$ | Null classes

| 10 | 2,492 | 0.4543 | 3 | 0.7331 | 6

| 20 | 489 | 0.3380 | 4 | 0.6622 | 3

| 50 | 519 | 0.4639 | 3 | 0.7310 | 6

| 100 | 242 | 0.5792 | 3 | 0.6584 | 3

| 200 | 199 | 0.5122 | 4 | 0.6366 | 3

| 500 | 236 | 0.4258 | 3 | 0.1421 | 3

| 1,000 | 380 | 0.5133 | 3 | 0.0000 | 1

| 2,000 | 312 | 0.5714 | 3 | 0.0000 | 1

| 5,000 | 625 | 0.0000 | 1 | 0.0000 | 1

| 10,000 | 2,067 | 0.0000 | 1 | 0.0000 | 1

### 7.6 Filings differ more than mechanisms, over part of the range

Per-example agreement 0.6028, functional instability 0.3972, gap 0.3344 (95% CI 0.3297 to 0.3394) against 0.10; measures of Bayat Makou et al. [1], verbatim.

Cohen's kappa 0.0146 against 60% raw agreement. Across sizes kappa 0.016 to 0.047, raw agreement 0.52 to 0.92, accuracy 0.51 to 0.95. Phantom specialisation: rejected.

H4 supported on the pooled quantity. Within size: fails 0.10 at 6 of 10 sizes, negative at 5, negative at both ends; pooled figure carried by the 500 to 2,000 edge band.

## 8 The filability protocol

Filability of an interpretability claim. For a filed claim $c$ derived from a circuit:

1.

Declare $\mathcal{S}$: defensible analytic axes and levels, each cited.

2.

Declare $\varphi$ as deterministic, published code. Not a language model.

3.

Compute $\pi^{*}=\max_{c}n_{c}/N$ over $\mathcal{S}$, and $F$ by Equation 2.

4.

Filable at $\alpha$: $\pi^{*}\geq 1-\alpha$.

5.

Report $\pi^{*}$, $F$, the discard rate per axis level, and the within-group decomposition of any pooled statistic.

No adjudication of the correct analytic choice is required of the regulator.

## 9 Limitations

#### One model, one task.

Larger models: more components, more room in layer band and size class; sign of the effect on $F$ unknown.

Pre-registered, not run: a 924-cell Pythia-160m grid, different architecture family, same task, identical except three seed levels rather than five; plan, grid and replication/refutation criteria committed and timestamped before any cell ran; to be reported in revision either way. A second task designed, not run.

#### Circuit size is an outcome, not an axis.

Selected by the threshold rule; distorts pooled $F$, the null comparison direction, and the sign of the functional gap. Fixing it: 0.7316 to 0.2746, more than any pre-registered axis.

#### The multiverse is as wide as the library, not the literature.

Optimal ablation [5] is not implemented; $F$ is a lower bound.

#### The discard rate is high and uneven.

52.3%, metric-imbalanced; rule fixed in advance, nothing repaired, composition stated not adjusted.

#### The bootstrap is invalid for one reported family.

Groups of two to five specifications: retained self-pairs dominate; point estimates with diagnostic. No post hoc bias-corrected estimator.

## 10 Discussion

#### What standardising one axis buys.

Metric removes 0.1377 of 0.7316.

#### On the instrument.

One of seven documented, publicly exported objectives does not run on the library's canonical task; reported as a discard with its rate. The library's authors are among those calling for this to be addressed [10].

#### What a regulator can take from this.

Evidentiary weight is measurable, and cheap relative to the sweep producing the circuit.

## 11 Reproducibility

Pre-registration: timestamped commit tagged prereg-p1-confirmatory, public, before the sweep. Every number traces to config, seed and environment fingerprint. Single environment across all 1,320 cells, verified from the results archive; code commit recovered by content-addressed comparison against the git tree, 22 of 22 files identical. Analysis runs without a GPU.

Verification script: 36 checks, 0 failures, $F$ by three routes including brute-force pairwise. A second script matches every numeric token in the manuscript to stored results. Tables emitted from results. Test suite: 343 tests.