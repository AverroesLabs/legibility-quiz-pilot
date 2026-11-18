# Explanation Multiplicity: Circuit-Level Interpretability

Evidence Does Not Survive Defensible Analytic Variation

Ajay Pravin Mahale

12 August 2026

**Summary.** The paper asks whether circuit-derived regulatory claims remain compatible across defensible analytic settings. A pre-registered crossed design uses published implementations to discover circuits for GPT-2 small on indirect object identification. Deterministic code converts circuits into structured Annex IV statements. Among 7,561 claim-producing specifications, pairwise claim disagreement is 0.7316. No single-axis standardisation permits filing under the registered tolerances. Removing size from statements and conditioning comparisons on size leaves disagreement at 0.2706. Discovered circuits outperform size-matched randomness at fixed size, reversing the pooled comparison. Functional measurements do not support treating the circuits as interchangeable mechanisms. Size conditioning also qualifies the pooled functional-gap result. The study covers one model and one task; a committed second-model study remains unrun.

**Glossary**

- **AI** — The type of system addressed by the cited regulation.
- **Audit; auditability** — The commissioning requirement that two auditors reach the same conclusion.
- **Conformity assessment** — The setting in which the paper considers acceptable evidentiary tolerances.
- **Regulation (EU) 2024/1689** — The regulation requiring high-risk AI providers to file technical documentation.
- **Regulation (EU) 2026/1744** — The amendment used in the paper’s consolidated regulatory text.
- **CELEX 02024R1689-20260727** — The identifier of that consolidated text.
- **Annex IV 2(b)** — The provision requesting the system’s and algorithms’ general logic.
- **Annex IV 2(e)** — The provision requesting assessment of measures facilitating interpretation of outputs.
- **Annex IV 3; Article 14(4)(c)** — Additional provisions grounding the overseer mapping.
- **Article 86(1)** — The provision granting certain decision subjects clear and meaningful explanations.
- **Mechanistic interpretability** — The source of prospective technical content examined here through circuit discovery.
- **Circuit discovery** — An instrument returning a subgraph said to explain behaviour.
- **Circuit; subgraph; \(C(s)\)** — The discovery output selected under specification \(s\), contained in edge set \(E\).
- **Specification; \(s\)** — A combination of objective, operator, corruption, metric, threshold, prompt, seed and granularity.
- **Analytic axis; level** — A specification choice and one of its available settings.
- **Multiverse** — The set of results across defensible analytic choices.
- **Crossed design; grid** — The design combining the declared analytic levels, except where crossing would duplicate specifications.
- **Discovery cell** — A discovery-grid unit counted before metric, threshold and statement reporting.
- **Discovery objective; \(o\)** — The objective axis of a specification.
- **Ablation operator; \(a\)** — The operator axis, with five operators reading corruption and two not reading it.
- **Corruption distribution; \(d\)** — The distribution axis nested within operators that read corruption.
- **Evaluation metric; \(m\)** — The metric axis used in the threshold rule.
- **Size threshold; \(\tau\)** — The threshold axis used to select circuits.
- **Prompt variant; \(P\)** — The prompt-variant axis.
- **Seed; \(r\)** — The axis selecting sampled prompts, measuring evaluation-set variability rather than nondeterminism.
- **Granularity; \(g\)** — The statement-resolution axis.
- **G7U; claim map; \(\varphi\)** — Published, pre-registered deterministic code transforming circuits into structured Annex IV statements.
- **Claim; class; \(c\)** — A structured statement, with specifications grouped by the statement they produce.
- **Y3W; filability** — The agreement criterion for circuit-derived filed claims.
- **D9P; overseer** — The mapping ordering dominant layer band, size class and ranked input segment.
- **L3X; affected** — The mapping leading with input segment for the decision’s subject.
- **E7R; COARSE** — The coarsest statement resolution, also reported without a size term.
- **K8Z; MEDIUM** — The intermediate statement resolution.
- **V2H; FINE** — The finest statement resolution.
- **Dominant layer band** — The early, middle or late layer attribution used in statements.
- **Size class** — The sparse, moderate or distributed category used in statements.
- **Ranked input segment** — An input-based element of the claim mappings.
- **Q7V; GPT-2 small** — The tested model.
- **Z4K; indirect object identification** — The tested task.
- **M9B; Pythia-160m** — The model in the committed but unrun second-model plan.
- **Factorised graph** — Q7V’s graph with separate query, key and value inputs.
- **\(E\); \(|E|\)** — The graph’s edge set and its size, 32,491.
- **Query; key; value** — The three separate attention-input slots in the factorised graph.
- **Attention head** — A graph component, with 12 heads per block and 144 in total.
- **MLP** — A block component, with 12 in the graph.
- **\(b\)** — The block index in the graph-counting description.
- **\(C_i,C_j\)** — Circuits compared under specifications \(s_i,s_j\).
- **\(\cap;\cup;|\cdot|\)** — Intersection, union and set size in the circuit-overlap formulas.
- **\(D(s_i,s_j)\)** — One minus the circuits’ intersection-to-union ratio.
- **Jaccard similarity; \(\bar J\)** — Circuit intersection divided by union; \(\bar J\) is its mean over unordered pairs.
- **\(F\); claim instability; pair disagreement** — The probability that distinct specification draws without replacement produce different claims.
- **\(N;n_c\)** — The number of specifications and the number producing claim class \(c\).
- **Gini–Simpson form** — The unbiased disagreement formula used for \(F\).
- **\(G;F_{\mathrm{within}}(G)\)** — Conditioning axes and the resulting pair-weighted within-group disagreement.
- **\(g;n_g;n_{g,c}\)** — A conditioning group, its specification count and its count for class \(c\).
- **\(k\)** — The available claim-class count in the ceiling formula; circuit size in the analytic random-overlap formula.
- **\(\pi^*\)** — The largest claim-class share, \(\max_c n_c/N\).
- **\(\alpha\)** — The filing tolerance; Y3W requires \(\pi^*\geq1-\alpha\).
- **\(\mathcal S\)** — The declared space of defensible analytic axes and levels.
- **A6N; size-matched random null** — The random-circuit baseline sampling the full edge population.
- **\(R\)** — The number of null multiverses, 1,000.
- **Null degeneracy; collapse** — The condition in which sampled circuits produce only one claim.
- **Pooled statistic** — A statistic reported before within-size decomposition.
- **Fixed-size; within-size; size conditioning** — Comparisons conditioned on selected circuit size.
- **Standardisation** — Holding an analytic axis fixed and reporting remaining disagreement.
- **Residual \(F\); removes** — Remaining claim instability after standardisation and the amount removed.
- **Structural disagreement** — Disagreement in circuit edges.
- **Functional agreement** — Agreement on per-example correct/incorrect verdicts.
- **Functional instability** — One minus functional agreement.
- **Functional gap** — \(F-(1-\mathrm{agreement})\).
- **Cohen’s kappa** — The reported functional diagnostic accompanying raw agreement.
- **Accuracy** — The functional-performance quantity tracked alongside agreement.
- **W5T; phantom specialisation** — Structurally distinct circuits implementing the same computation.
- **F9A; specification curve** — The visualisation adapted to show claim classes and active analytic levels.
- **Faithfulness** — The ablation-based assessment whose sensitivity motivates the metric choices.
- **R8X; edge attribution patching** — The discovery family used in the study.
- **B3Q; integrated edge gradients** — The additional discovery method used through Miller et al.’s implementation.
- **T9V; comprehensiveness** — One of the four evaluation metrics.
- **H2W; KL divergence** — One of the four evaluation metrics.
- **N6C; logit difference** — One of the four evaluation metrics.
- **U4J; sufficiency** — One of the four evaluation metrics.
- **C4Y; optimal ablation** — A defensible operator unsupported by the tested instrument.
- **J6D; P0** — The descriptive premise assessed by reporting \(\bar J\).
- **X4R; H2** — The claim-instability hypothesis requiring \(F>0.20\), with the CI lower bound above it.
- **Q2F; H3** — The hypothesis that discovery is not separated from the random null.
- **Z8N; H4** — The functional-gap hypothesis requiring a gap greater than 0.10.
- **CI** — The reported confidence interval.
- **Bootstrap; \(B\)** — Resampling specifications rather than pairs; \(B=10{,}000\).
- **Pairwise U-statistics** — The paired statistics \(F,\bar J\), whose pairs share specifications.
- **\(p\)-values** — Specification-level tests not reported because the designed grid is not an independent sample.
- **REML** — The label used for the selected-circuit-size structural-variance decomposition.
- **\(\log_{10}\)** — The circuit-size scale used in that decomposition.
- **Quantile; median** — Distributional summaries reported for circuit disagreement and size.
- **Pre-registration; confirmatory sweep** — The committed plan and the subsequent sweep testing it.
- **Metric-relative threshold rule** — The pre-registered rule producing circuit exclusions.
- **Retention; discard rate** — The kept and excluded specification counts and rates.
- **Mean squared error; integral target** — The objective/target combination causing the library runtime error.
- **Digest; failing identifier** — The verification record establishing that exactly one objective arm failed.
- **Deviations file** — The append-only record of decisions, dates and whether results had been viewed.
- **Validation slice** — The six-cell check disclosing retention unevenness before lock.
- **Environment fingerprint** — The record linking reported results to the execution environment.
- **Content-addressed comparison; git tree** — The archive-to-code comparison used to recover and verify the commit.
- **`prereg-p1-confirmatory`** — The public timestamped commit tag.
- **GPU** — Hardware not required for the reported analysis.

###### Abstract

*The abstract reports failure of filing agreement, the controls applied and the untested scope.*

Circuit-derived regulatory evidence fails Y3W, the filability criterion for agreement among filed claims, throughout tolerances plausibly acceptable to a conformity-assessment body. Competent analysts sharing a system and tool obtain incompatible statements under different defensible settings. Standardisation, which holds an analytic axis fixed, and size controls do not remove the failure. Structural disagreement in circuit edges and functional disagreement in correct/incorrect verdicts exclude merely reworded mechanisms. Applicability beyond one model and task remains untested.

## 1 Introduction

*This section connects regulatory explanation requirements to agreement between analysts.*

An audit requires two auditors to reach the same conclusion. This is a commissioning requirement, not a machine-learning definition.

Regulation (EU) 2024/1689 requires high-risk AI providers to file technical documentation. Annex IV 2(b) requests “the general logic of the AI system and of the algorithms”; 2(e), “an assessment of the technical measures needed to facilitate the interpretation of the outputs”. Article 86(1) grants persons subject to certain automated decisions “clear and meaningful explanations of the role of the AI system in the decision-making procedure and the main elements of the decision taken”. Quotations follow the consolidated text amended by Regulation (EU) 2026/1744, CELEX 02024R1689-20260727, read 11 August 2026.

Mechanistic interpretability supplies prospective technical content. Its mature instrument, circuit discovery, returns a circuit: a subgraph said to explain behaviour. Filing this account presumes compatibility with another analyst’s account. This paper measures movement in regulatory claims across defensible specifications—combinations of analytic settings—using a standards-body decision rule.

#### Contributions.

*The study measures filing instability and tests whether standardisation or size explains it.*

The pre-registered crossed design combines declared analytic settings using published implementations exclusively. Deterministic code converts circuits into structured Annex IV statements. Controls standardise individual analytic axes, remove claim dependence on size, and condition comparisons on size. Three pooled statistics, calculated before size conditioning, invert when selected objects differ in size. No improved discovery method is proposed. The objection concerns the standard presumed by filing currently produced evidence, not interpretability’s worth.

## 2 Related work

*This section identifies prior work on explanation non-uniqueness, measurement choices, auditability and analytic flexibility.*

#### Non-identifiability.

*An explanation need not be unique under mechanistic interpretability’s own criteria.*

Méloux et al. [6] find that a behaviour need not have a unique explanation under mechanistic interpretability’s own criteria. This paper retains that premise but measures its downstream cost for filing rather than publication.

#### Faithfulness metric sensitivity.

*Prior work shows that ablation-based faithfulness scores depend on design choices.*

Miller et al. [8] find that ablation-based faithfulness is highly sensitive to seemingly insignificant design changes. Scores reflect researcher choices as well as circuits. Their single-axis finding becomes a seven-axis regulatory propagation here. Their released library supplies the instrument, tested under its own definitions.

#### Structure versus function.

*Structural differences could reflect the same computation, so the study measures function.*

Bayat Makou et al. [1] hold the task fixed, vary input statistics, and find structurally distinct circuits implementing the same computation. This pattern, W5T or phantom specialisation, supplies the strongest design objection: claim differences might identify sampled members of a behaviourally identical equivalence class. Functional measurement addresses that objection.

#### Variance as estimation.

*Statistical-estimation work supplies a companion framing for specification variability.*

Méloux et al. [7] analyse mechanistic interpretability as statistical estimation, including estimator variance. Seed variance is one axis here; their framing is the statistical companion to whole-specification analysis.

#### Auditability.

*Prior auditability work motivates a measurement and criterion, not a standards proposal.*

Lan et al. [4] describe two papers reaching conflicting conclusions about one behaviour and a third finding both partly correct and incomparable. Their request for safety-critical certification standards receives a measurement and criterion here, not a proposal. Sharkey et al. [10] include relevant socio-technical open problems. Mueller et al. [9] require standardised evaluation for method comparison, adjacent to standardised specification for certification.

#### Multiverse method.

*The study transfers reporting across defensible analytic choices to regulatory evidence.*

Steegen et al. [13] prescribe reporting a multiverse: results across defensible data-processing choices. Simonsohn et al.’s [12] F9A, or specification curve, visualises those choices and is adapted here. Stability claims are susceptible to the undisclosed analytic flexibility identified by Simmons et al. [11]. The psychology-methods transfer is substantive: flexibility can manufacture explanation stability.

#### Explanation metrics under optimisation pressure.

*Two metrics used here can be inflated without changing predictions or explanations.*

Hsia et al. [3] inflate U4J, or sufficiency, and T9V, or comprehensiveness, without changing predictions or explanations. Both occur among this study’s four metrics. Metric-specific discard reporting avoids silent pooling.

**Claims and evidence**

- Explanation non-uniqueness is a retained premise: Méloux et al. [6]; no numerical result is supplied here.
- Faithfulness depends strongly on design choices: Miller et al. [8]; no numerical result is supplied here.
- Structurally different circuits can implement the same computation: Bayat Makou et al. [1]; no numerical result is supplied here.
- Published conclusions can conflict and be incomparable: Lan et al. [4], two papers and a third assessment.
- U4J and T9V can be inflated without changing predictions or explanations: Hsia et al. [3]; no numerical result is supplied here.

## 3 What this paper is not claiming

*Circuits retain information, but neither their mechanisms nor the study’s scope should be overstated.*

Two unsupported readings were our own before decomposition. Discovered circuits carry information and, at fixed size, outperform A6N, the size-matched random null baseline. Their mechanisms nevertheless differ substantially and are functionally uncorrelated, rather than being one mechanism differently dressed.

The seed selects sampled prompts. It measures evaluation-set variability, not nondeterminism. No scale claim is made.

**Claims and evidence**

- Discovery outperforms the random baseline at fixed size: Section 7.5, “Comparison” table, “Fixed size, K8Z” row, 0.2746 versus 0.4230; “Fixed size, E7R” row, 0.2706 versus 0.3822.
- Mechanisms differ substantially and are functionally uncorrelated: Section 7.1, “Structural diagnostic” table, median \(D=0.960\); Section 7.6, “Functional diagnostic” table, Cohen’s kappa 0.0146.

## 4 Formalism

*This section defines specifications, circuit and claim disagreement, and the claim-class ceiling.*

A specification is
\[
s=(o,a,d,m,\tau,P,r,g),
\]
where \(o\) is discovery objective; \(a\), ablation operator; \(d\), corruption distribution; \(m\), evaluation metric; \(\tau\), size threshold; \(P\), prompt variant; \(r\), seed; and \(g\), granularity or statement resolution. Each axis is an analytic choice; each level is an available setting. Every level comes from a published implementation. None was invented here.

G7U, the claim map, transforms discovery output \(C(s)\subseteq E\) into statements. Here \(C(s)\) is the circuit and \(E\) is the graph’s edge set. Y3W concerns agreement among resulting claims, not adjudication of a correct specification.

Q7V, or GPT-2 small, has a factorised graph with separate query, key and value inputs. Its edge count is \(|E|=32{,}491\). This was confirmed from the instrument and independently by computation-order counting. For block index \(b\), attention receives \(1+13b\) sources at three input slots per each of 12 heads. MLP \(b\) receives those sources plus its block’s heads. Output receives all 157 non-terminal sources. Components comprise 144 attention heads and 12 MLPs, 156 total.

#### Circuit instability.

*Circuit instability is one minus Jaccard similarity.*

For circuits \(C_i,C_j\) from specifications \(s_i,s_j\), intersection \(\cap\), union \(\cup\), and set size \(|\cdot|\),
\[
D(s_i,s_j)=1-\frac{|C_i\cap C_j|}{|C_i\cup C_j|}. \tag{1}
\]

Jaccard similarity is the intersection-to-union ratio. \(\bar J\) denotes its mean over unordered pairs.

#### Claim instability.

*Claim instability is the probability that two distinct specifications produce different statements.*

Let \(\varphi(C)\) be the structured Annex IV statement produced by the claim map. Let \(n_c\) be the number of specifications producing claim class \(c\), and \(N\) the number of specifications. Then
\[
F=1-\frac{\sum_c n_c(n_c-1)}{N(N-1)}. \tag{2}
\]
This unbiased Gini–Simpson form gives disagreement probability for distinct draws without replacement. Randomised unit tests compare it with brute-force pairs.

For conditioning axes \(G\), group \(g\), group size \(n_g\), and class count \(n_{g,c}\),
\[
F_{\mathrm{within}}(G)=
1-\frac{\sum_g\sum_c n_{g,c}(n_{g,c}-1)}
{\sum_g n_g(n_g-1)}. \tag{3}
\]
Pairs, not groups, receive equal weight. Empty \(G\) reduces exactly to (2).

#### Y3W.

*The available number of claim classes limits the maximum disagreement rate.*

For \(k\) available claim classes, the ceiling is \(F\leq1-1/k\). The observed rate is not near one by construction.

| Ceiling diagnostic | Value |
|---|---:|
| Observed classes | 9 |
| Ceiling | 0.8889 |
| Observed fraction of ceiling | 82.3% |

**Claims and evidence**

- The observed rate is not near one by construction: “Ceiling diagnostic” table, “Observed classes” 9, “Ceiling” 0.8889, “Observed fraction of ceiling” 82.3%.
- The graph contains 32,491 edges and 156 components: computation-order count above; 144 attention heads and 12 MLPs. No table is supplied for this count.

## 5 G7U

*This section defines the statement mappings and reports sensitivity to size binning.*

Size bins follow a committed calibration rule applied to a measured node-count curve. They were not selected, as a reviewer suspected, for the reported \(F\). Four alternatives exceed the committed result, including halving thresholds. Factor-four binnings remain within the reported range.

Table 1: Alternative size binnings.

| Binning | \(F\) |
|---|---:|
| committed bins | 0.7316 |
| all thresholds \(\times\) 0.5 | 0.7448 |
| all thresholds \(\times\) 0.8 | 0.7435 |
| all thresholds \(\times\) 0.9 | 0.7394 |
| all thresholds \(\times\) 1.1 | 0.7289 |
| all thresholds \(\times\) 1.25 | 0.7314 |
| all thresholds \(\times\) 1.5 | 0.7308 |
| all thresholds \(\times\) 2 | 0.7232 |
| equal thirds | 0.7018 |
| deciles | 0.7719 |
| no size term (E7R) | 0.5733 |

| Sensitivity range | Value |
|---|---|
| Across factor-four binnings | 0.70 to 0.77 |

Regulatory addressees determine two mappings:

- **D9P, overseer:** grounded in Annex IV 2(e), 3 and Article 14(4)(c). It orders dominant layer band, size class and ranked input segment. Layer bands are early, middle and late; size classes are sparse, moderate and distributed.
- **L3X, affected:** grounded in Article 86(1). It leads with input segment because the right belongs to the decision’s subject.

E7R, or COARSE, is the coarsest resolution; K8Z, or MEDIUM, is intermediate; V2H, or FINE, is finest. These nested resolutions ensure that refinement cannot contradict the coarser statement.

The claim map \(\varphi\) is the paper’s most attackable choice. It is pre-registered, published deterministic code, not language-model output. Language-model output would inseparably add model variance to circuit variance.

**Claims and evidence**

- Four alternatives exceed the committed result: Table 1, “committed bins” 0.7316; “all thresholds \(\times\) 0.5” 0.7448; “\(\times\) 0.8” 0.7435; “\(\times\) 0.9” 0.7394; “deciles” 0.7719.
- Factor-four binnings remain within the reported range: “Sensitivity range” table, “Across factor-four binnings” 0.70 to 0.77.
- Removing the size term does not remove disagreement: Table 1, “no size term (E7R)” 0.5733.

## 6 Method

*This section describes the model, grid, registration, statistics and exclusions.*

#### Model and task.

*The study uses one model and task for comparability with prior work.*

Comparability with the extended non-identifiability and faithfulness literature motivates Q7V and Z4K, or indirect object identification [15]. The edge count makes size-matched randomness meaningful; scale does not serve as an apology.

#### Grid.

*The grid crosses published analytic choices without duplicating operators that ignore corruption.*

The grid contains seven objectives, seven operators, four corruption distributions nested within the five operators reading them, four metrics, three thresholds, two prompt variants and five seeds: 1,540 discovery cells and 18,480 pre-registered specifications. A discovery cell is a discovery-grid unit used to produce specifications. Two operators never read corruption; crossing them with corruption would duplicate specifications.

Discovery uses R8X, or edge attribution patching [14], and B3Q, or integrated edge gradients, as implemented by Miller et al. [8]. Automated discovery lineage comes from Conmy et al. [2]; corruption levels come from Wang et al. [15].

The metrics are T9V, H2W or KL divergence, N6C or logit difference, and U4J. Unequal retention across metrics materially changes the realised pool.

#### Pre-registration.

*The plan and both directional abstracts preceded the confirmatory sweep.*

The plan was committed and tagged before the confirmatory sweep. Both directional abstracts were drafted in advance, preventing either direction from being presented as a surprise. An append-only deviations file records dates and whether decisions preceded or followed viewing affected results.

#### Statistics.

*Uncertainty estimates resample specifications and apply only to the designed grid.*

No specification-level \(p\)-values are reported: the designed grid is not an independent sample [13, 12]. A nonparametric bootstrap resamples specifications, never pairs, with \(B=10{,}000\) resamples. \(F,\bar J\) are pairwise U-statistics: their pairs depend on shared specifications. Confidence intervals (CIs) condition on the grid and license no extrapolation.

#### Two exclusions, criteria fixed before any circuit was seen.

*One objective fails in the library, and the registered threshold rule discards specifications unevenly.*

Every cell of one objective fails inside the verbatim library. Mean squared error against an integral target raises a runtime error. A digest of sorted failing identifiers verifies exactly that arm and nothing else. The tested instrument was not repaired.

| Grid consequence | Value |
|---|---:|
| Failed objective cells | 220 |
| Realised objectives | 6 |
| Realised discovery cells | 1,320 |
| Realised specifications | 15,840 |
| Specifications admitting no circuit | 8,279 |
| Discard rate | 52.3% |
| Surviving T9V composition | 52% |
| Designed T9V composition | 25% |

The pre-registered metric-relative threshold rule produces the exclusions. A six-cell validation slice disclosed the unevenness before lock. Composition is reported, not smoothed.

Table 2: Metric-specific retention.

| Metric | Kept | Discarded | Rate |
|---|---:|---:|---:|
| T9V | 3,947 | 13 | 0.003 |
| H2W | 1,467 | 2,493 | 0.630 |
| N6C | 1,270 | 2,690 | 0.679 |
| U4J | 877 | 3,083 | 0.779 |

**Claims and evidence**

- One objective fails throughout its arm: “Grid consequence” table, “Failed objective cells” 220 and “Realised objectives” 6; the sorted-identifier digest verifies that arm.
- The realised grid has fewer cells and specifications: “Grid consequence” table, “Realised discovery cells” 1,320 and “Realised specifications” 15,840.
- Discards are high and uneven: “Grid consequence” table, “Specifications admitting no circuit” 8,279 and “Discard rate” 52.3%; Table 2, rates T9V 0.003, H2W 0.630, N6C 0.679 and U4J 0.779.
- Retention changes metric composition: “Grid consequence” table, “Surviving T9V composition” 52% versus “Designed T9V composition” 25%.

## 7 Results

*The registered outcomes show high claim instability, separation from randomness and a pooled functional gap requiring size qualification.*

J6D, or P0, is the descriptive premise. X4R, or H2, is the claim-instability hypothesis. Q2F, or H3, is the random-comparison hypothesis. Z8N, or H4, is the functional-gap hypothesis, where the gap is \(F-(1-\mathrm{agreement})\).

Their decision rules were fixed on 6 August 2026 and tagged before the sweep, without revision. Two outcomes require within-size qualifications here rather than deferral to an appendix.

Table 3: Registered outcomes.

| | Decision rule | Result | Outcome |
|---|---|---|---|
| J6D | \(\bar J\) reported, not tested | 0.1396 [0.1377, 0.1418] | premise holds |
| X4R | \(F>0.20\), CI lower bound above | 0.7316 [0.7247, 0.7380] | confirmed |
| Q2F | not separated from random null | separated, intervals disjoint | rejected |
| Z8N | \(F-(1-\mathrm{agreement})>0.10\) | 0.3344 [0.3297, 0.3394] | supported |

**Claims and evidence**

- The descriptive premise holds: Table 3, J6D row, 0.1396 [0.1377, 0.1418].
- Claim instability exceeds the registered threshold: Table 3, X4R row, 0.7316 [0.7247, 0.7380] against 0.20.
- Discovery is separated from the random null: Table 3, Q2F row, “separated, intervals disjoint”; numerical interval endpoints are not supplied.
- The registered pooled functional gap is supported: Table 3, Z8N row, 0.3344 [0.3297, 0.3394] against 0.10.

### 7.1 Circuits are almost disjoint

*Circuits overlap more than chance but share little structure.*

Circuit duplication makes exact, unsampled pair computation tractable. Distributional separation exceeds what the mean alone conveys. Overlap beats chance without implying much common structure.

| Structural diagnostic | Value |
|---|---|
| Claim-producing specifications | 7,561 |
| Distinct circuits | 3,218 |
| All specification pairs | 28,580,580 |
| \(D\), 1% quantile | 0.196 |
| \(D\), 25% quantile | 0.788 |
| \(D\), median | 0.960 |
| \(D\), 75% quantile | 0.998 |
| Median shared-edge fraction | 4% |
| Pairs exceeding Jaccard 0.8 | only 1% |
| Median circuit size | 200 edges |
| Analytic random line | \(k/(2|E|-k)\) |
| Random line at median size | 0.0031 |
| Observed/chance overlap | about 45 times |
| Expected random overlap, 500-edge circuit | 0.008 |

Here \(k\) in the analytic random line denotes circuit size.

**Claims and evidence**

- Exact pair computation covers the claim-producing pool: “Structural diagnostic” table, 7,561 specifications, 3,218 distinct circuits and 28,580,580 pairs.
- Circuits share little structure: the same table, median \(D\) 0.960, median shared-edge fraction 4%, and only 1% of pairs exceeding Jaccard 0.8.
- Overlap exceeds chance: the same table, median size 200 edges, random line 0.0031 and observed/chance overlap about 45 times.

### 7.2 The claim flips on three quarters of pairs

*Claims disagree frequently under every registered addressee-resolution combination.*

The lower interval bound clears the registered threshold by more than three times. None of the registered tolerances permits filing. The most frequent pair attributes the same model/task behaviour to early versus late layers: contradiction, not emphasis.

Table 4: \(\varphi_{\mathrm{D9P}}\), K8Z, frequency order.

| Rank | Layer band | Size class | Count | Share |
|---|---|---|---:|---:|
| 1 | early layers | distributed | 3,107 | 0.4109 |
| 2 | late layers | sparse | 2,153 | 0.2848 |
| 3 | late layers | moderate | 657 | 0.0869 |
| 4 | late layers | distributed | 489 | 0.0647 |
| 5 | early layers | sparse | 477 | 0.0631 |
| 6 | middle layers | moderate | 283 | 0.0374 |
| 7 | middle layers | sparse | 269 | 0.0356 |
| 8 | middle layers | distributed | 90 | 0.0119 |
| 9 | early layers | moderate | 36 | 0.0048 |

| Additional reporting form | Value |
|---|---|
| Pair disagreement, percentage | 73.2% |
| Its 95% CI, abstract precision | 0.725 to 0.738 |
| Modal percentage | 41.1% |
| Second-class percentage | 28.5% |
| Modal share, comparison precision | 0.411 |
| Required shares | 0.95, 0.90, 0.80 |
| Size-aggregated early / late / middle | 47.9% / 43.6% / 8.5% |

The early aggregate independently reproduces the E7R modal share. Even the coarsest L3X statement rejects Y3W. Fine-grained language is not required.

In Table 5, tolerance \(\alpha\) is the allowed filing disagreement, and \(\pi^*=\max_c n_c/N\) is the largest claim-class share.

Table 5: All addressee-resolution combinations; loosest registered tolerance \(\alpha=0.20\), required \(\pi^*\geq0.80\).

| Addressee | Granularity | Classes | \(F\) | \(\pi^*\) | Y3W |
|---|---|---:|---:|---:|---|
| L3X | E7R | 2 | 0.3950 | 0.7291 | no |
| L3X | K8Z | 4 | 0.6597 | 0.4875 | no |
| L3X | V2H | 8 | 0.6744 | 0.4786 | no |
| D9P | E7R | 3 | 0.5733 | 0.4788 | no |
| D9P | K8Z | 9 | 0.7316 | 0.4109 | no |
| D9P | V2H | 12 | 0.7681 | 0.4109 | no |

| Coarsest L3X percentage disagreement | 39.5% |
|---|---:|

Figure 1: F9A for \(\varphi_{\mathrm{D9P}}\), K8Z. Upper panel: frequency-ordered claim class against specification rank; step widths equal class shares, modal width \(\pi^*\). Dashed line: A6N modal share, its pooled comparison size-confounded. Lower panel: active analytic levels per specification; solid T9V band reflects discard-induced composition, unstructured seed rows the negligible seed contribution.

**Claims and evidence**

- Disagreement exceeds the registered threshold: Table 3, X4R row, lower bound 0.7247 against 0.20; “Additional reporting form” table, disagreement 73.2%.
- The two most frequent claims attribute behaviour to different layer bands: Table 4, rank 1, early/distributed, 3,107 and 0.4109; rank 2, late/sparse, 2,153 and 0.2848.
- The early aggregate reproduces the coarsest overseer modal share: “Additional reporting form” table, early 47.9%; Table 5, D9P/E7R row, \(\pi^*=0.4788\).
- No registered combination permits filing: Table 5, all six Y3W entries “no” at required \(\pi^*\geq0.80\); modal shares are 0.7291, 0.4875, 0.4786, 0.4788, 0.4109 and 0.4109.
- Coarse affected-person language still fails: Table 5, L3X/E7R row, \(F=0.3950\), \(\pi^*=0.7291\); accompanying percentage table, 39.5%.

### 7.3 No analytic axis rescues the filing

*Metric standardisation removes the most claim instability, but no single axis suffices.*

The metric dominates single-axis claim control but remains insufficient. Structural and claim variance have different drivers. Standardising the operator selected by structural decomposition would fix almost none of the filing instability.

Table 6: Pair-weighted residuals after standardisation. Valid intervals: group sizes 1,512 to 2,520; bootstrap bias below 0.0006 on every axis; every observation inside its interval.

| Axis standardised | Residual \(F\) | 95% CI | Removes |
|---|---:|---|---:|
| Evaluation metric | 0.5939 | 0.5803 to 0.6063 | 0.1377 |
| Ablation operator | 0.7014 | 0.6939 to 0.7075 | 0.0302 |
| Discovery objective | 0.7018 | 0.6938 to 0.7084 | 0.0298 |
| Threshold \(\tau\) | 0.7070 | 0.6981 to 0.7149 | 0.0246 |
| Corruption | 0.7092 | 0.7008 to 0.7166 | 0.0224 |
| Prompt variant | 0.7232 | 0.7157 to 0.7300 | 0.0084 |
| Seed | 0.7307 | 0.7234 to 0.7368 | 0.0009 |

The following REML table reports structural-variance components on the \(\log_{10}\) selected-circuit-size scale.

| REML component, \(\log_{10}\) selected circuit size | Structural variance |
|---|---:|
| Ablation-corruption interaction | 78.8% |
| Residual | 11.5% |
| Ablation | 4.4% |
| Threshold | 4.1% |
| Metric | 0.7% |

| Metric-standardised percentage disagreement | 59.4% |
|---|---:|

**Claims and evidence**

- Metric standardisation removes the most claim instability but leaves substantial disagreement: Table 6, “Evaluation metric” row, removes 0.1377, residual 0.5939, CI 0.5803 to 0.6063; percentage table, 59.4%.
- No single axis rescues filing: Table 6, residuals 0.5939, 0.7014, 0.7018, 0.7070, 0.7092, 0.7232 and 0.7307.
- Structural and claim drivers differ: REML table, ablation-corruption interaction 78.8% and metric 0.7%; Table 6, operator removes 0.0302 versus metric 0.1377.
- Seed contributes little claim control: Table 6, “Seed” row, removes 0.0009.

### 7.4 The result under maximum pressure

*Dominant-layer disagreement remains above threshold after removing size from statements and comparisons.*

Size is entirely absent from these statements and conditioned out of comparisons. Dominant-layer attribution still clears the registered instability threshold at the interval’s lower bound.

| Size-free claim, fixed size | Value |
|---|---|
| \(F\) | 0.2706 |
| 95% CI | 0.2550 to 0.2859 |
| Percentage, abstract precision | 27.1% |
| 95% CI, abstract precision | 0.255 to 0.286 |
| Percentage, discussion precision | 27% |

**Claims and evidence**

- Size removal and conditioning do not remove instability: “Size-free claim, fixed size” table, \(F=0.2706\), 95% CI 0.2550 to 0.2859; Table 3, X4R threshold 0.20.

### 7.5 The random baseline, and why its pooled direction misleads

*Fixed-size comparison reverses the pooled ordering, while discovered circuits still fail Y3W.*

Pooled discovery is less stable than A6N, with disjoint intervals. Rejection of Q2F concerns separation, not its size-confounded direction. Uniform null draws above the degeneracy threshold touch every component and produce only one claim. At fixed size, discovered circuits outperform randomness but still fail Y3W.

| Comparison | Discovered \(F\) | A6N \(F\) |
|---|---:|---:|
| Pooled | 0.7316 | 0.6774 |
| Fixed size, K8Z | 0.2746 | 0.4230 |
| Fixed size, E7R | 0.2706 | 0.3822 |

A6N samples the full edge population with \(R=1{,}000\) null multiverses, where \(R\) is the number of null repetitions. Its degeneracy occurs above 1,000 edges.

Table 7: K8Z within-size results. A6N collapses at and above 1,000 edges; discovery also collapses at 5,000 and 10,000. Thus 2,692 of 7,561 specifications contribute no flips; instability lives entirely below 2,000 edges.

| Size | \(n\) | Discovered \(F\) | Classes | A6N \(F\) | A6N classes |
|---|---:|---:|---:|---:|---:|
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

Here \(n\) is the number of specifications at each size.

**Claims and evidence**

- Pooled discovery is less stable than A6N: “Comparison” table, “Pooled” row, 0.7316 versus 0.6774; Table 3, Q2F row, intervals disjoint.
- Fixed-size comparison reverses the ordering: “Comparison” table, K8Z 0.2746 versus 0.4230; E7R 0.2706 versus 0.3822.
- The null collapses at and above 1,000 edges: Table 7, sizes 1,000, 2,000, 5,000 and 10,000, A6N \(F=0.0000\) and one class.
- Discovery collapses at 5,000 and 10,000 edges: Table 7, those rows, \(F=0.0000\), one class, \(n=625\) and \(n=2,067\); the caption reports 2,692 of 7,561 specifications with no flips.
- Instability lives entirely below 2,000 edges: Table 7 caption. The 2,000-edge row reports discovered \(F=0.5714\), three classes and \(n=312\).

### 7.6 Filings differ more than mechanisms, over part of the range

*The registered functional gap holds when pooled, but not across all fixed sizes.*

Agreement uses per-example correct/incorrect verdicts and Bayat Makou et al.’s [1] measures reproduced verbatim. Functional instability is one minus agreement. Raw agreement mostly reflects circuits being right on similar fractions, not identical examples. The functional interchangeability required by the W5T objection is absent.

Z8N holds for its registered pooled quantity, not under size conditioning. Mechanisms differ more than filings at both range ends. Only the intermediate band carries the pooled gap. Registration requires reporting the pooled result; decomposition prevents its misleading interpretation.

| Functional diagnostic | Value |
|---|---|
| Pooled per-example agreement | 0.6028 |
| Pooled functional instability | 0.3972 |
| Smallest-circuit functional instability | 0.4777 |
| Cohen’s kappa | 0.0146 |
| Kappa, abstract precision | 0.015 |
| Raw agreement, rounded description | 60% |
| Kappa range at every size | 0.016 to 0.047 |
| Raw-agreement climb | 0.52 to 0.92 |
| Accuracy tracked | 0.51 to 0.95 |
| Sizes failing gap threshold | 6 of 10 |
| Sizes with negative gap | 5 |
| Band carrying pooled gap | 500 to 2,000 edges |

Cohen’s kappa is the reported functional diagnostic accompanying raw agreement.

**Claims and evidence**

- Raw agreement does not establish functional interchangeability: “Functional diagnostic” table, agreement 0.6028 and Cohen’s kappa 0.0146; within-size kappa 0.016 to 0.047.
- Agreement tracks accuracy: the same table, raw agreement 0.52 to 0.92 and accuracy 0.51 to 0.95.
- The pooled gap holds but is not general across sizes: Table 3, Z8N row, 0.3344 [0.3297, 0.3394]; “Functional diagnostic” table, 6 of 10 sizes fail the gap threshold and 5 have negative gaps.
- Only the intermediate band carries the pooled gap: “Functional diagnostic” table, “Band carrying pooled gap” 500 to 2,000 edges.

## 8 The Y3W protocol

*This section states the declarations, agreement criterion and reporting required for filing.*

For circuit-derived filed claim \(c\):

1. Declare \(\mathcal S\), the space of defensible analytic axes and levels, each cited.
2. Declare \(\varphi\): published deterministic code, not a language model.
3. Compute \(\pi^*=\max_c n_c/N\) over \(\mathcal S\), and \(F\).
4. Evidence passes Y3W at tolerance \(\alpha\) if and only if \(\pi^*\geq1-\alpha\).
5. Report \(\pi^*\), \(F\), discard rate per axis level, and within-group decomposition of every pooled statistic.

A filing unable to state its own \(\pi^*\) is asserted, not audited. Implementability requires only a declared space and measured agreement, not regulatory adjudication of analytic correctness.

## 9 Limitations

*The study’s scope, instrument, exclusions and some bootstrap intervals limit its conclusions.*

#### One model, one task.

*Generalisation remains untested, including the committed second-model study.*

Larger models permit more layer-band and size-class variation. The direction for \(F\) is unclear. This limitation is most likely to change its value.

The committed, timestamped second-model plan uses M9B, or Pythia-160m: a different architecture family performing the same task. It specifies a 924-cell grid identical on every axis except three rather than five seed levels. Replication-versus-refutation criteria were fixed before any cell ran. The study remains unrun and will appear in a revision regardless of direction. A second task is designed but unrun.

#### Circuit size is an outcome, not an axis.

*Selected object size changes three pooled conclusions.*

Threshold-selected size distorts \(F\), the null-comparison direction and the functional-gap sign. Conditioning on size reduces claim instability more than standardising any registered axis. This is a general multiverse hazard when specifications select differently sized objects. Every pooled statistic is reported with within-size decomposition.

#### The multiverse is as wide as the library, not the literature.

*Unsupported defensible choices lie outside the measured space.*

C4Y, or optimal ablation [5], is defensible but unsupported and never enters the instrument’s space. This makes measured \(F\) a lower bound relative to the field’s full defensible choice set.

#### The discard rate is high and uneven.

*The retained metric imbalance remains part of the reported set.*

The retained metric imbalance is a reported-set property, not adjusted away. The threshold was fixed in advance, and no failed item was repaired.

#### The bootstrap is invalid for one reported family.

*Very small conditioning groups receive point estimates rather than unusable intervals.*

Conditioning groups containing only two to five specifications retain dominating self-pairs. Unusable intervals are replaced by point estimates with diagnostics. No bias-corrected estimator was constructed after naive intervals disagreed: result-responsive estimator selection is the behaviour under study.

**Claims and evidence**

- Size conditioning reduces instability more than any single-axis standardisation: Section 7.4, “Size-free claim, fixed size” table, \(F=0.2706\); Table 6, smallest residual \(F=0.5939\), “Evaluation metric” row.
- Size changes the null-comparison direction: Section 7.5, “Comparison” table, pooled 0.7316 versus 0.6774; fixed-size K8Z 0.2746 versus 0.4230; fixed-size E7R 0.2706 versus 0.3822.
- Size qualifies the functional-gap sign: Section 7.6, “Functional diagnostic” table, 5 sizes with negative gap and 6 of 10 failing the threshold.
- Discards are high and uneven: “Grid consequence” table, 52.3%; Table 2, rates 0.003, 0.630, 0.679 and 0.779.
- One bootstrap family is invalid: groups contain two to five specifications and retain dominating self-pairs; no numerical diagnostic table is supplied.

## 10 Discussion

*The discussion argues for measured evidentiary weight rather than presumed filing reliability.*

#### What standardising one axis buys.

*Even complete metric standardisation does not resolve the filing problem.*

Complete metric standardisation exceeds what a conformity procedure could demand. No single knob suffices. Declaring resolution after standardising one axis would be substantially mistaken.

#### On the instrument.

*Execution failure and discards document tooling maturity without becoming the headline.*

The failed objective is documented and publicly exported but cannot execute on the library’s own canonical task. The instrument’s discard rate also records tooling maturity beneath prospective regulatory evidence. This is neither the headline nor criticism of authors who are among those requesting these improvements [10].

#### What a regulator can take from this.

*Interpretability evidence can receive measured rather than presumed weight.*

Interpretability evidence is not worthless. Evidentiary weight is measurable rather than presumed. Measurement is cheap relative to the sweep producing the circuit.

**Claims and evidence**

- Single-axis standardisation does not resolve instability: Table 6, best residual \(F=0.5939\), 95% CI 0.5803 to 0.6063, “Evaluation metric” row; accompanying percentage table, 59.4%.
- The instrument has execution and retention problems: “Grid consequence” table, 220 failed objective cells, 8,279 specifications admitting no circuit and 52.3% discard rate.
- Discovered circuits retain information: Section 7.5, “Comparison” table, fixed-size K8Z 0.2746 versus random 0.4230; fixed-size E7R 0.2706 versus random 0.3822.
- Measurement is cheap relative to discovery: asserted here; no numerical cost comparison is supplied.

## 11 Reproducibility

*This section reports traceability, archive verification and automated numerical checks.*

The public timestamped commit is tagged `prereg-p1-confirmatory`. Every reported number traces to a configuration, seed and environment fingerprint. Archive verification establishes one environment across all 1,320 confirmatory cells. Content-addressed comparison recovers the code commit from the uploaded archive against the git tree: 22 of 22 files are identical. Analysis needs no GPU.

One script recomputes headline quantities from raw results and asserts reported values: 36 checks, 0 failures, with \(F\) checked through three routes including brute-force pairs. Another checks every manuscript numeric token against stored-result derivations. Tables are emitted directly, never transcribed. The suite contains 343 tests.

**Claims and evidence**

- Execution used one verified environment: archive verification across all 1,320 confirmatory cells; no table is supplied.
- Archived code matches the commit: content-addressed comparison, 22 of 22 files identical; no table is supplied.
- Headline recomputation passes: 36 checks, 0 failures, with \(F\) checked through three routes including brute-force pairs; no table is supplied.
- The test suite contains 343 tests: reproducibility report above; no table is supplied.
