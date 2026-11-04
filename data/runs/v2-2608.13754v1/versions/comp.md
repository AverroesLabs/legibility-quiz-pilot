# Explanation Multiplicity: Circuit-Level Interpretability

The EU AI Act requires high-risk providers to file technical documentation on how the system decides. We test whether circuit-discovery evidence survives two competent analysts using the same system and tool with different defensible settings. A pre-registered crossed grid of seven analytic axes (every level from a published implementation) feeds a deterministic map from circuits to Annex IV statements. Over 15,840 specifications on GPT-2 small/IOI (7,561 yield a claim), the statement flips on 73.2% of pairs (95% CI 0.725–0.738) and the modal claim holds 41.1%. Standardising the metric leaves 59.4%; removing size from the claim and fixing it leaves 27.1% (CI 0.255–0.286). Median pairwise Jaccard is 4% and κ=0.015.

## 1 Introduction

An audit means two auditors reach the same conclusion. Reg. (EU) 2024/1689 (consolidated, amended by Reg. (EU) 2026/1744, CELEX 02024R1689-20260727, read 11 Aug 2026):
- Annex IV 2(b): "the general logic of the AI system and of the algorithms";
- 2(e): "an assessment of the technical measures needed to facilitate the interpretation of the outputs";
- Art. 86(1): "clear and meaningful explanations of the role of the AI system in the decision-making procedure and the main elements of the decision taken".

We propose no new method and do not call interpretability worthless.

Contributions:
- (i) pre-registered seven-axis multiverse plus a deterministic circuit→Annex IV map;
- (ii) filability criterion π*≥1−α, measured;
- (iii) instability survives standardising any axis, removing size from the claim, and fixing size;
- (iv) when specifications select objects of different sizes, pooled statistics can invert; three of ours do.

## 2 Related work

- **Non-identifiability.** A unique explanation need not exist. We measure its cost when an explanation is filed.
- **Faithfulness metric sensitivity.** Ablation-based faithfulness is highly sensitive to minor choices, so scores reflect researcher choices. That is one axis; we cross seven. Their library is our instrument, so this is a stress test by its own definitions.
- **Structure versus function.** Structurally distinct circuits can implement the same computation ("phantom specialisation"). If so, flip rate would measure which equivalence-class member was sampled. §7.6 answers this functionally: our circuits are not interchangeable.
- **Variance as estimation.** MI as statistical estimation; seed variance is one of our axes.
- **Auditability.** Prior calls for auditing standards to certify findings, and for standardised evaluation across methods. We supply a measurement and criterion; certification needs standardised specification.
- **Multiverse method.** We use the multiverse and the specification curve; undisclosed analytic flexibility can manufacture stability claims.
- **Explanation metrics under optimisation pressure.** Sufficiency and comprehensiveness can be inflated without changing predictions. Both are among our four metrics, so we report per-metric discard rates.

## 3 What this paper is not claiming

- That circuits are no better than random: at fixed size they beat a size-matched null, 0.2746 vs 0.4230.
- That mechanisms agree: functional instability is 0.3972 pooled, 0.4777 at smallest circuits.
- That one mechanism is redescribed: κ=0.0146.
- That seeds matter: the seed selects prompts (evaluation-set variability).
- Anything about scale.

## 4 Formalism

Specification s=(o,a,d,m,τ,P,r,g) has components: objective, ablation, corruption distribution, metric, size threshold, prompt variant, seed, granularity. No level was invented. C(s)⊆E with $|E|=32{,}491$ for GPT-2 small (factorised q/k/v). This is confirmed from the instrument and in closed form:
- block-b attention takes 1+13b sources × 3 slots × 12 heads;
- MLP b takes those sources plus own-block heads;
- the output takes all 157 non-terminal sources.

Circuits draw on 156 components (144 heads, 12 MLPs).

D(s_i,s_j)=1−|C_i∩C_j|/|C_i∪C_j| (1); J̄ = mean Jaccard over unordered pairs.

With φ: circuit→Annex IV statement, F=1−Σ_c n_c(n_c−1)/(N(N−1)) (2). This is unbiased Gini–Simpson, unit-tested against brute force. π*=max_c n_c/N.

F_within(G)=1−Σ_gΣ_c n_{g,c}(n_{g,c}−1)/Σ_g n_g(n_g−1) (3). It weights pairs equally and equals (2) for empty G.

Filable at α iff π*≥1−α. F≤1−1/k; with 9 classes the ceiling is 0.8889, and 0.7316 is 82.3% of it.

## 5 The claim map

φ is pre-registered, deterministic, published code, not an LM (that would confound model variance).
- φ_overseer: Annex IV 2(e), 3, Art. 14(4)(c); keyed on dominant layer band, then size class, then ranked input segment.
- φ_affected: Art. 86(1); leads with input segment.

Three nested granularities: finer refines, never contradicts. Size bins come from a committed calibration rule on a measured node-count curve.

Table 1: F under alternative binnings (range 0.70–0.77 across 4× factors).

| Binning | F |
|---|---|
| committed | 0.7316 |
| ×0.5 | 0.7448 |
| ×0.8 | 0.7435 |
| ×0.9 | 0.7394 |
| ×1.1 | 0.7289 |
| ×1.25 | 0.7314 |
| ×1.5 | 0.7308 |
| ×2 | 0.7232 |
| equal thirds | 0.7018 |
| deciles | 0.7719 |
| no size term (COARSE) | 0.5733 |

## 6 Method

**Model and task.** GPT-2 small, IOI [15]. Matches the literature's scale; the edge count makes a size-matched random baseline meaningful.

**Grid.** 7 objectives × 7 ablations; 4 corruptions nested in the 5 ablations that read them (crossing would duplicate); 4 metrics; 3 thresholds; 2 prompts; 5 seeds. That gives 1,540 cells, 18,480 specifications. Discovery: EAP family and integrated edge gradients from the faithfulness-sensitivity library; corruption levels from the IOI paper.

**Pre-registration.** Plan tagged before the sweep, both abstracts pre-drafted, deviations in a dated append-only log.

**Statistics.** No cross-specification p-values. Bootstrap resamples specifications, not pairs ($B=10{,}000$), since F and J̄ are pair U-statistics. Intervals hold only given the grid.

**Two exclusions, criteria fixed before any circuit was seen.** One objective fails on all 220 cells (MSE against an integral target, inside library code). A digest confirms the failure set is exactly that arm; it was left unrepaired. Realised grid: 6 objectives, 1,320 cells, 15,840 specifications. The metric-relative threshold rule admits no circuit for 8,279 (52.3%), disclosed pre-lock from a six-cell slice. The pool is 52% comprehensiveness against a 25% design.

Table 2: Discards by metric.

| Metric | Kept | Discarded | Rate |
|---|---|---|---|
| comprehensiveness | 3,947 | 13 | 0.003 |
| KL divergence | 1,467 | 2,493 | 0.630 |
| logit difference | 1,270 | 2,690 | 0.679 |
| sufficiency | 877 | 3,083 | 0.779 |

## 7 Results

Table 3: Pre-registered outcomes (thresholds fixed 6 Aug 2026, none revisited).

| | Rule | Result | Outcome |
|---|---|---|---|
| P0 | J̄ reported | 0.1396 [0.1377, 0.1418] | premise holds |
| H2 | F>0.20, CI lower bound above | 0.7316 [0.7247, 0.7380] | confirmed |
| H3 | not separated from null | separated, disjoint | rejected |
| H4 | F−(1−agreement)>0.10 | 0.3344 [0.3297, 0.3394] | supported |

### 7.1 Circuits are almost disjoint

J̄ is exact over 28,580,580 pairs (3,218 distinct circuits). D quantiles: 1%=0.196, 25%=0.788, median 0.960, 75%=0.998. Only 1% of pairs exceed Jaccard 0.8. The random line k/(2|E|−k) gives 0.0031 at the median 200 edges, so observed overlap is ~45× chance (500 edges: 0.008).

### 7.2 The claim flips on three quarters of pairs

F clears 0.20 by >3×. π*=0.411, against required 0.95/0.90/0.80.

Table 4: φ_overseer MEDIUM classes. Layer totals: early 47.9% (= COARSE modal 0.4788), late 43.6%, middle 8.5%.

| Band | Size | Count | Share |
|---|---|---|---|
| early | distributed | 3,107 | 0.4109 |
| late | sparse | 2,153 | 0.2848 |
| late | moderate | 657 | 0.0869 |
| late | distributed | 489 | 0.0647 |
| early | sparse | 477 | 0.0631 |
| middle | moderate | 283 | 0.0374 |
| middle | sparse | 269 | 0.0356 |
| middle | distributed | 90 | 0.0119 |
| early | moderate | 36 | 0.0048 |

The top two claims (41.1%, 28.5%) contradict: early vs late layers. The affected-person COARSE two-class claim still flips on 39.5%.

Table 5: At α=0.20 (π*≥0.80) no cell is filable.

| Addressee | Gran. | Classes | F | π* |
|---|---|---|---|---|
| affected | COARSE | 2 | 0.3950 | 0.7291 |
| affected | MEDIUM | 4 | 0.6597 | 0.4875 |
| affected | FINE | 8 | 0.6744 | 0.4786 |
| overseer | COARSE | 3 | 0.5733 | 0.4788 |
| overseer | MEDIUM | 9 | 0.7316 | 0.4109 |
| overseer | FINE | 12 | 0.7681 | 0.4109 |

### 7.3 No analytic axis rescues the filing

Table 6: Residual F per standardised axis via (3). Groups have 1,512–2,520 specifications; bootstrap bias <0.0006.

| Axis | F | 95% CI | Removes |
|---|---|---|---|
| Metric | 0.5939 | 0.5803–0.6063 | 0.1377 |
| Ablation | 0.7014 | 0.6939–0.7075 | 0.0302 |
| Objective | 0.7018 | 0.6938–0.7084 | 0.0298 |
| τ | 0.7070 | 0.6981–0.7149 | 0.0246 |
| Corruption | 0.7092 | 0.7008–0.7166 | 0.0224 |
| Prompt | 0.7232 | 0.7157–0.7300 | 0.0084 |
| Seed | 0.7307 | 0.7234–0.7368 | 0.0009 |

REML on log₁₀ circuit size: ablation×corruption 78.8%, residual 11.5%, ablation 4.4%, threshold 4.1%, metric 0.7%. Structure and claim follow different axes: standardising ablation would barely fix filings.

### 7.4 The result under maximum pressure

Layer-band-only claim with size fixed: F=0.2706 (0.2550–0.2859). Attribution flips on 27% of pairs.

### 7.5 The random baseline, and why its pooled direction misleads

Size-matched null over all 32,491 edges, R=1,000: pooled 0.7316 vs 0.6774. The direction is a size artefact: above 1,000 edges a uniform draw touches all 156 components. Fixed-size: 0.2746 vs 0.4230 (MEDIUM), 0.2706 vs 0.3822 (COARSE).

Table 7: Within-size F, MEDIUM. 2,692 specifications contribute no flips; all instability is below 2,000 edges.

| Size | n | Disc. F | Cl. | Null F | Cl. |
|---|---|---|---|---|---|
| 10 | 2,492 | 0.4543 | 3 | 0.7331 | 6 |
| 20 | 489 | 0.3380 | 4 | 0.6622 | 3 |
| 50 | 519 | 0.4639 | 3 | 0.7310 | 6 |
| 100 | 242 | 0.5792 | 3 | 0.6584 | 3 |
| 200 | 199 | 0.5122 | 4 | 0.6366 | 3 |
| 500 | 236 | 0.4258 | 3 | 0.1421 | 3 |
| 1,000 | 380 | 0.5133 | 3 | 0.0000 | 1 |
| 2,000 | 312 | 0.5714 | 3 | 0.0000 | 1 |
| 5,000 | 625 | 0.0000 | 1 | 0.0000 | 1 |
| 10,000 | 2,067 | 0.0000 | 1 | 0.0000 | 1 |

### 7.6 Filings differ more than mechanisms, over part of the range

Per-example agreement is 0.6028, using the phantom-specialisation measures verbatim. The 60% raw agreement reflects equal accuracy, not shared examples. Across sizes κ stays 0.016–0.047 while raw agreement rises 0.52→0.92 (accuracy 0.51→0.95). The circuits are not interchangeable. The H4 gap fails 0.10 at 6/10 sizes and is negative at 5; at both ends, mechanisms differ more than filings. The pooled gap comes from 500–2,000 edges alone.

## 8 The filability protocol

1. Declare space 𝒮: axes and defensible levels, each cited.
2. Declare φ as deterministic published code, not an LM.
3. Compute π* and F (Eq. 2).
4. Filable at α if π*≥1−α.
5. Report π*, F, per-level discard rates, within-group decompositions.

The regulator need not judge which choice is correct.

## 9 Limitations

- **One model, one task.** Larger models give band and size more room; the direction of the effect on F is unclear. Pre-registered, unrun: a 924-cell Pythia-160m replication, identical except 3 seeds, with replication criteria timestamped; reported either way. A second task is designed, not run.
- **Circuit size is an outcome, not an axis.** It distorted pooled F, null direction and functional-gap sign. Fixing it drops F 0.7316→0.2746, more than any axis.
- **The multiverse is as wide as the library, not the literature.** Optimal ablation is unimplemented, so F is a lower bound.
- **The discard rate is high and uneven.** 52.3%, metric-imbalanced, stated rather than adjusted.
- **The bootstrap is invalid for one reported family.** With 2–5-specification groups, self-pairs dominate; point estimates only. No post hoc estimator was built.

## 10 Discussion

#### What standardising one axis buys.

No single knob fixes it: the metric removes only 0.1377 of 0.7316.

#### On the instrument.

One of seven exported objectives fails on the library's canonical task. This is a tooling-maturity observation, not the headline.

#### What a regulator can take from this.

Evidentiary weight is measurable, cheaply relative to the sweep.

## 11 Reproducibility

Pre-registration tag: prereg-p1-confirmatory, committed before the sweep. Every number traces to config, seed and environment fingerprint. A single environment across 1,320 cells was verified from the archive; the code commit was recovered by content-addressing (22/22 files). Analysis runs without a GPU. Verification: 36 checks, 0 failures; F by three routes incl. brute force. Every manuscript number and table derives from stored results. 343 tests.