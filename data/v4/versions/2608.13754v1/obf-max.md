# Explanation Multiplicity: Circuit-Level Interpretability

Evidence Does Not Survive Defensible Analytic Variation

Ajay Pravin Mahale

12 August 2026

###### Abstract

Circuit-derived regulatory evidence fails Y3W throughout tolerances plausibly acceptable to a conformity-assessment body: competent analysts sharing system and tool obtain incompatible statements under different defensible settings. Standardisation and size controls do not remove the failure; structural and functional disagreement exclude merely reworded mechanisms. Applicability beyond one model and task remains untested.

## 1 Introduction

An audit requires two auditors to reach the same conclusion; this commissioning requirement is not a machine-learning definition.

Regulation (EU) 2024/1689 requires high-risk AI providers to file technical documentation. Annex IV 2(b) requests “the general logic of the AI system and of the algorithms”; 2(e), “an assessment of the technical measures needed to facilitate the interpretation of the outputs”. Article 86(1) grants persons subject to certain automated decisions “clear and meaningful explanations of the role of the AI system in the decision-making procedure and the main elements of the decision taken”. Quotations follow the consolidated text amended by Regulation (EU) 2026/1744, CELEX 02024R1689-20260727, read 11 August 2026.

Mechanistic interpretability supplies prospective technical content; its mature instrument, circuit discovery, returns a subgraph said to explain behaviour. Filing such an account presumes compatibility with another analyst’s account. Regulatory-claim movement across defensible specifications is measured here with a standards-body decision rule.

#### Contributions.

The pre-registered crossed design uses published implementations exclusively; circuits become structured Annex IV statements deterministically. Controls standardise individual axes, remove claim size dependence, and condition on size. Three pooled statistics invert when selected objects differ in size. No improved discovery method is proposed; the objection concerns the standard presumed by filing currently produced evidence, not interpretability’s worth.

## 2 Related work

#### Non-identifiability.

Méloux et al. [6] find that a behaviour need not have a unique explanation under mechanistic interpretability’s own criteria. Their premise is retained; the downstream cost measured here concerns filing rather than publication.

#### Faithfulness metric sensitivity.

Miller et al. [8] find ablation-based faithfulness highly sensitive to seemingly insignificant design changes: scores reflect researcher choices as well as circuits. Their single-axis finding becomes a seven-axis regulatory propagation here; their released library supplies the instrument, tested under its own definitions.

#### Structure versus function.

Bayat Makou et al. [1] hold task fixed, vary input statistics, and find structurally distinct circuits implementing the same computation, a pattern (W5T: phantom specialisation) that supplies the strongest design objection: claim differences might identify sampled members of a behaviourally identical equivalence class. Functional measurement, rather than argument, addresses that objection.

#### Variance as estimation.

Méloux et al. [7] analyse mechanistic interpretability as statistical estimation, including estimator variance; seed variance is one axis here, their framing the statistical companion to whole-specification analysis.

#### Auditability.

Lan et al. [4] describe two papers’ conflicting conclusions about one behaviour and a third finding both partly correct and incomparable; their request for safety-critical certification standards receives a measurement and criterion here, not a proposal. Sharkey et al. [10] include relevant socio-technical open problems; Mueller et al. [9] require standardised evaluation for method comparison, adjacent to standardised specification for certification.

#### Multiverse method.

Steegen et al. [13] prescribe reporting across defensible data-processing choices, while Simonsohn et al.’s [12] visualisation (F9A: specification curve) is adapted to this evidentiary setting, where stability claims are susceptible to the undisclosed analytic flexibility identified by Simmons et al. [11]. The psychology-methods transfer is substantive: flexibility can manufacture explanation stability.

#### Explanation metrics under optimisation pressure.

Hsia et al. [3] inflate U4J and T9V without changing predictions or explanations. Both occur among our four metrics; metric-specific discard reporting avoids silent pooling.

## 3 What this paper is not claiming

Two unsupported readings were our own before decomposition. Discovered circuits carry information and, at fixed size, outperform the baseline (A6N: size-matched random null), whereas mechanisms themselves differ substantially and are functionally uncorrelated rather than one mechanism differently dressed.

The seed selects sampled prompts: evaluation-set variability, not nondeterminism. No scale claim is made.

## 4 Formalism

A specification is
\[
s=(o,a,d,m,\tau,P,r,g),
\]
respectively discovery objective, ablation operator, corruption distribution, evaluation metric, size threshold, prompt variant, seed, granularity. Every axis level comes from a published implementation; none was invented here.

A circuit-to-statement transformation (G7U: claim map) operates on discovery output \(C(s)\subseteq E\), while the admissibility property (Y3W: filability) concerns agreement among the resulting claims rather than adjudication of a correct specification.

For Q7V’s factorised graph with separate query, key and value inputs, \(|E|=32{,}491\), confirmed from the instrument and independently by computation-order counting: block \(b\) attention receives \(1+13b\) sources at three input slots per each of 12 heads; MLP \(b\) receives those sources plus its block’s heads; output receives all 157 non-terminal sources. Components comprise 144 attention heads and 12 MLPs, 156 total.

#### Circuit instability.

\[
D(s_i,s_j)=1-\frac{|C_i\cap C_j|}{|C_i\cup C_j|}. \tag{1}
\]

\(\bar J\) denotes mean Jaccard similarity over unordered pairs.

#### Claim instability.

For structured Annex IV statements \(\varphi(C)\), with \(n_c\) specifications producing class \(c\) and \(N\) specifications,
\[
F=1-\frac{\sum_c n_c(n_c-1)}{N(N-1)}. \tag{2}
\]
This unbiased Gini–Simpson form gives disagreement probability for distinct draws without replacement; randomised unit tests compare it with brute-force pairs.

Conditioning on axes \(G\),
\[
F_{\mathrm{within}}(G)=
1-\frac{\sum_g\sum_c n_{g,c}(n_{g,c}-1)}
{\sum_g n_g(n_g-1)}. \tag{3}
\]
Pairs, not groups, receive equal weight; empty \(G\) reduces exactly to (2).

#### Y3W.

The ceiling \(F\leq1-1/k\) depends on available claim classes; the observed rate is not near one by construction.

| Ceiling diagnostic | Value |
|---|---:|
| Observed classes | 9 |
| Ceiling | 0.8889 |
| Observed fraction of ceiling | 82.3% |

## 5 G7U

Size bins follow a committed calibration rule applied to a measured node-count curve, not reviewer-suspected selection for the reported \(F\). Four alternatives exceed the committed result, including halving thresholds; factor-four binnings remain within the reported range.

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

Regulatory addressees determine two mappings: the first (D9P: overseer), grounded in Annex IV 2(e), 3 and Article 14(4)(c), orders dominant layer band, size class, ranked input segment; the second (L3X: affected), grounded in Article 86(1), leads with input segment, the right belonging to the decision’s subject.

The nested resolutions (E7R: COARSE), intermediate resolution (K8Z: MEDIUM), and finest resolution (V2H: FINE) are constructed so refinement cannot contradict the coarser statement, while \(\varphi\), the paper’s most attackable choice, is pre-registered, published deterministic code rather than language-model output, which would inseparably add model variance to circuit variance.

## 6 Method

#### Model and task.

Comparability with the extended non-identifiability and faithfulness literature motivates the model (Q7V: GPT-2 small) and task (Z4K: indirect object identification) [15], with the edge count making size-matched randomness meaningful rather than scale serving as an apology.

#### Grid.

Seven objectives, seven operators, four corruption distributions nested within the five operators reading them, four metrics, three thresholds, two prompt variants, five seeds: 1,540 discovery cells, 18,480 pre-registered specifications. Two operators never read corruption; crossing would duplicate specifications.

Discovery uses the family (R8X: edge attribution patching) [14] and the additional method (B3Q: integrated edge gradients) as implemented by Miller et al. [8], with automated discovery lineage in Conmy et al. [2] and corruption levels from Wang et al. [15].

The four metrics comprise one measure (T9V: comprehensiveness), another (H2W: KL divergence), a third (N6C: logit difference), and the remaining measure (U4J: sufficiency), whose unequal retention materially changes the realised pool.

#### Pre-registration.

The plan was committed and tagged before the confirmatory sweep; both directional abstracts were drafted in advance, preventing either direction being presented as surprise. An append-only deviations file records dates and whether decisions preceded or followed viewing affected results.

#### Statistics.

No specification-level \(p\)-values: the designed grid is not an independent sample [13, 12]. Nonparametric bootstrap resamples specifications, never pairs, at \(B=10{,}000\); \(F,\bar J\) are pairwise U-statistics with dependent shared-specification pairs. Intervals condition on the grid and license no extrapolation.

#### Two exclusions, criteria fixed before any circuit was seen.

Every cell of one objective fails inside the verbatim library: mean squared error against an integral target raises a runtime error. A digest of sorted failing identifiers verifies exactly that arm, nothing else. No repair was made to the tested instrument.

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

The pre-registered metric-relative threshold rule produces the exclusions; unevenness was disclosed before lock from a six-cell validation slice, and composition is reported rather than smoothed.

Table 2: Metric-specific retention.

| Metric | Kept | Discarded | Rate |
|---|---:|---:|---:|
| T9V | 3,947 | 13 | 0.003 |
| H2W | 1,467 | 2,493 | 0.630 |
| N6C | 1,270 | 2,690 | 0.679 |
| U4J | 877 | 3,083 | 0.779 |

## 7 Results

The descriptive premise (J6D: P0), claim-instability hypothesis (X4R: H2), random-comparison hypothesis (Q2F: H3), and functional-gap hypothesis (Z8N: H4) retain decision rules fixed on 6 August 2026 and tagged before the sweep, without revision; two outcomes require within-size qualifications rather than appendix deferral.

Table 3: Registered outcomes.

| | Decision rule | Result | Outcome |
|---|---|---|---|
| J6D | \(\bar J\) reported, not tested | 0.1396 [0.1377, 0.1418] | premise holds |
| X4R | \(F>0.20\), CI lower bound above | 0.7316 [0.7247, 0.7380] | confirmed |
| Q2F | not separated from random null | separated, intervals disjoint | rejected |
| Z8N | \(F-(1-\mathrm{agreement})>0.10\) | 0.3344 [0.3297, 0.3394] | supported |

### 7.1 Circuits are almost disjoint

Exact, unsampled pair computation is tractable through circuit duplication. Distributional separation exceeds what the mean alone conveys; overlap beats chance without implying much common structure.

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

### 7.2 The claim flips on three quarters of pairs

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

The early aggregate independently reproduces the E7R modal share. Even the coarsest L3X statement rejects Y3W; fine-grained language is not required.

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

### 7.3 No analytic axis rescues the filing

The metric dominates single-axis claim control but remains insufficient. Structural and claim variance have different drivers; standardising the operator selected by structural decomposition would fix almost none of the filing instability.

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

| REML component, \(\log_{10}\) selected circuit size | Structural variance |
|---|---:|
| Ablation-corruption interaction | 78.8% |
| Residual | 11.5% |
| Ablation | 4.4% |
| Threshold | 4.1% |
| Metric | 0.7% |

| Metric-standardised percentage disagreement | 59.4% |
|---|---:|

### 7.4 The result under maximum pressure

With size entirely absent from statements and conditioned out of comparisons, dominant-layer attribution still clears the registered instability threshold at the interval’s lower bound.

| Size-free claim, fixed size | Value |
|---|---|
| \(F\) | 0.2706 |
| 95% CI | 0.2550 to 0.2859 |
| Percentage, abstract precision | 27.1% |
| 95% CI, abstract precision | 0.255 to 0.286 |
| Percentage, discussion precision | 27% |

### 7.5 The random baseline, and why its pooled direction misleads

Pooled discovery is less stable than A6N, with disjoint intervals; rejection of Q2F concerns separation, not its size-confounded direction. Uniform null draws above the degeneracy threshold touch every component and produce only one claim. Fixed-size comparison reverses the ordering: discovered circuits outperform randomness but still fail Y3W.

| Comparison | Discovered \(F\) | A6N \(F\) |
|---|---:|---:|
| Pooled | 0.7316 | 0.6774 |
| Fixed size, K8Z | 0.2746 | 0.4230 |
| Fixed size, E7R | 0.2706 | 0.3822 |

A6N samples the full edge population with \(R=1{,}000\) null multiverses; its degeneracy occurs above 1,000 edges.

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

### 7.6 Filings differ more than mechanisms, over part of the range

Agreement uses per-example correct/incorrect verdicts and Bayat Makou et al.’s [1] measures reproduced verbatim. Raw agreement mostly reflects circuits being right on similar fractions, not identical examples; functional interchangeability required by the W5T objection is absent.

Z8N holds on its registered pooled quantity, not under size conditioning. Mechanisms differ more than filings at both range ends; only the intermediate band carries the pooled gap. Registration requires reporting the pooled result; decomposition prevents its misleading interpretation.

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

## 8 The Y3W protocol

For circuit-derived filed claim \(c\):

1. Declare \(\mathcal S\): defensible analytic axes and levels, each cited.
2. Declare \(\varphi\): published deterministic code, not a language model.
3. Compute \(\pi^*=\max_c n_c/N\) over \(\mathcal S\), and \(F\).
4. Evidence passes Y3W at tolerance \(\alpha\) if and only if \(\pi^*\geq1-\alpha\).
5. Report \(\pi^*\), \(F\), discard rate per axis level, and within-group decomposition of every pooled statistic.

A filing unable to state its own \(\pi^*\) is asserted, not audited. Implementability requires only declared space and measured agreement, not regulatory adjudication of analytic correctness.

## 9 Limitations

#### One model, one task.

Larger models permit more layer-band and size-class variation; the direction for \(F\) is unclear, and this limitation is most likely to change its value.

The committed, timestamped second-model plan (M9B: Pythia-160m) specifies a different architecture family performing the same task, a 924-cell grid identical on every axis except three rather than five seed levels, and replication-versus-refutation criteria fixed before any cell ran. It remains unrun and will appear in a revision regardless of direction; a second task is designed but unrun.

#### Circuit size is an outcome, not an axis.

Threshold-selected size distorts \(F\), null-comparison direction, and functional-gap sign. Its conditioning reduces claim instability more than standardising any registered axis: a general multiverse hazard when specifications select differently sized objects. Every pooled statistic is reported with within-size decomposition.

#### The multiverse is as wide as the library, not the literature.

The defensible but unsupported operator (C4Y: optimal ablation) [5] never enters the instrument’s space, making measured \(F\) a lower bound relative to the field’s full defensible choice set.

#### The discard rate is high and uneven.

The retained metric imbalance is a reported-set property, not adjusted away; the threshold was fixed in advance and no failed item repaired.

#### The bootstrap is invalid for one reported family.

Conditioning groups containing only two to five specifications retain dominating self-pairs; unusable intervals are replaced by point estimates with diagnostics. No bias-corrected estimator was constructed after naive intervals disagreed: result-responsive estimator selection is the behaviour under study.

## 10 Discussion

#### What standardising one axis buys.

Complete metric standardisation exceeds what a conformity procedure could demand. No single knob suffices; declaring resolution after standardising one axis would be substantially mistaken.

#### On the instrument.

The failed objective is documented and publicly exported yet cannot execute on the library’s own canonical task. Its discard rate also records tooling maturity beneath prospective regulatory evidence; it is neither the headline nor criticism of authors among those requesting these improvements [10].

#### What a regulator can take from this.

Interpretability evidence is not worthless. Evidentiary weight is measurable rather than presumed; measurement is cheap relative to the sweep producing the circuit.

## 11 Reproducibility

The public timestamped commit is tagged `prereg-p1-confirmatory`. Every reported number traces to configuration, seed and environment fingerprint. Archive verification establishes one environment across all 1,320 confirmatory cells; content-addressed comparison recovers the code commit from the uploaded archive against the git tree, 22 of 22 files identical. Analysis needs no GPU.

One script recomputes headline quantities from raw results and asserts reported values: 36 checks, 0 failures, \(F\) through three routes including brute-force pairs. Another checks every manuscript numeric token against stored-result derivations. Tables are emitted directly, never transcribed; the suite contains 343 tests.
