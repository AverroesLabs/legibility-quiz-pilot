# Explanation Multiplicity: Circuit-Level Interpretability

Evidence Does Not Survive Defensible Analytic Variation

Ajay Pravin Mahale

12 August 2026

###### Abstract

Circuit-derived regulatory evidence consists of claims based on a selected subgraph of a model’s computation. This evidence fails Y3W, or **filability**: the requirement that the most common claim cover a sufficiently large share of defensible analytic settings. It fails throughout tolerances plausibly acceptable to a conformity-assessment body, which assesses compliance. Competent analysts sharing a system and tool obtain incompatible statements under different defensible settings.

Standardising individual analytic choices and controlling for circuit size do not remove the failure. Metric standardisation leaves claim disagreement at 0.5939 (Table 6). Removing size from statements and comparing only within size leaves disagreement at 0.2706 (§7.4). Structural and functional disagreement exclude merely reworded mechanisms: median structural distance is 0.960 (§7.1), and chance-adjusted functional agreement is 0.0146 (§7.6). Applicability beyond one model and task remains untested.

## 1 Introduction

An audit requires two auditors to reach the same conclusion. This is a commissioning requirement, not a machine-learning definition.

Regulation (EU) 2024/1689 requires high-risk artificial intelligence (AI) providers to file technical documentation. EU means European Union. Annex IV 2(b) requests “the general logic of the AI system and of the algorithms”; 2(e), “an assessment of the technical measures needed to facilitate the interpretation of the outputs”. Article 86(1) grants persons subject to certain automated decisions “clear and meaningful explanations of the role of the AI system in the decision-making procedure and the main elements of the decision taken”. Quotations follow the consolidated text amended by Regulation (EU) 2026/1744. Its legal-text identifier is CELEX 02024R1689-20260727; it was read 11 August 2026.

Mechanistic interpretability seeks to explain behaviour through a model’s internal computation. Its mature instrument, circuit discovery, returns a **circuit**: a subgraph said to explain behaviour. A graph represents computational components as nodes and their connections as edges. Filing a circuit-based account presumes compatibility with another analyst’s account. This paper measures how regulatory claims change across defensible analytic specifications, using a standards-body decision rule.

#### Contributions.

The pre-registered crossed design combines levels of analytic choices and uses published implementations exclusively. Deterministic code turns circuits into structured Annex IV statements. Controls standardise individual axes, remove claim dependence on size, and condition comparisons on size.

Three pooled statistics invert when selected objects differ in size. The relevant decompositions concern claim instability (§7.4), the direction of comparison with random circuits (§7.5), and the sign of the gap between claim and functional instability (§7.6).

No improved discovery method is proposed. The objection concerns the standard presumed when filing currently produced evidence, not interpretability’s worth.

## 2 Related work

#### Non-identifiability.

Non-identifiability means that the criteria used to explain a behaviour need not identify one unique explanation. Méloux et al. [6] find this under mechanistic interpretability’s own criteria. Their premise is retained. The downstream cost measured here concerns filing rather than publication.

#### Faithfulness metric sensitivity.

A faithfulness metric measures how well a proposed circuit accounts for the behaviour being explained. Ablation removes or replaces selected model activity to measure its effect. Miller et al. [8] find ablation-based faithfulness highly sensitive to seemingly insignificant design changes. Scores reflect researcher choices as well as circuits.

Their single-axis finding becomes a seven-axis regulatory propagation here: changes in analytic choices are followed through to changes in regulatory statements. Their released library supplies the instrument, tested under its own definitions.

#### Structure versus function.

Bayat Makou et al. [1] hold the task fixed, vary input statistics, and find structurally distinct circuits implementing the same computation. This pattern is W5T, or **phantom specialisation**. It supplies the strongest design objection: claim differences might identify sampled members of a behaviourally identical equivalence class—a set of different circuits performing the same computation. Functional measurement, rather than argument, addresses that objection (§7.6).

#### Variance as estimation.

Méloux et al. [7] analyse mechanistic interpretability as statistical estimation, including estimator variance: variation in the output of an estimation procedure. Seed variance is one axis here. A seed selects sampled evaluation prompts, so it changes the evaluation set rather than introducing computational nondeterminism. Their framing is the statistical companion to whole-specification analysis.

#### Auditability.

Lan et al. [4] describe two papers’ conflicting conclusions about one behaviour and a third finding both partly correct and incomparable. Their request for safety-critical certification standards receives a measurement and criterion here, not a proposal.

Sharkey et al. [10] include relevant socio-technical open problems. Mueller et al. [9] require standardised evaluation for method comparison. That requirement is adjacent to standardised specification for certification.

#### Multiverse method.

A **multiverse** examines results across defensible analytic choices rather than reporting only one choice. Steegen et al. [13] prescribe reporting across defensible data-processing choices. Simonsohn et al.’s [12] visualisation, F9A or **specification curve**, displays results across specifications. It is adapted here to regulatory evidence.

Stability claims are susceptible to the undisclosed analytic flexibility identified by Simmons et al. [11]. The psychology-methods transfer is substantive: flexibility can manufacture explanation stability.

#### Explanation metrics under optimisation pressure.

Hsia et al. [3] inflate U4J, or **sufficiency**, and T9V, or **comprehensiveness**, without changing predictions or explanations. Sufficiency concerns how much behaviour the retained explanation preserves; comprehensiveness concerns how much is lost when explanatory content is removed. Both occur among our four metrics. Metric-specific discard reporting avoids silent pooling: Table 2 reports discard rates of 0.779 for U4J and 0.003 for T9V.

## 3 What this paper is not claiming

Two unsupported readings were our own before decomposition. Discovered circuits carry information and, at fixed size, outperform A6N, the **size-matched random null**: a baseline of randomly selected edges matched to circuit size. The aggregate fixed-size comparisons are 0.2746 versus 0.4230 at intermediate claim resolution, and 0.2706 versus 0.3822 at coarse resolution (§7.5).

Mechanisms themselves differ substantially and are functionally uncorrelated, rather than one mechanism differently dressed. Median structural distance is 0.960 (§7.1), and pooled chance-adjusted functional agreement is 0.0146 (§7.6).

The seed selects sampled prompts: evaluation-set variability, not nondeterminism. No scale claim is made.

## 4 Formalism

A **specification** is a complete set of analytic settings:
\[
s=(o,a,d,m,\tau,P,r,g),
\]
where:

- \(o\) is the discovery objective: what the discovery procedure seeks to optimise.
- \(a\) is the ablation operator: how selected model activity is removed or replaced.
- \(d\) is the corruption distribution: the distribution supplying altered inputs or replacement activity.
- \(m\) is the evaluation metric: the measure used to assess a circuit.
- \(\tau\) is the size threshold used in circuit selection.
- \(P\) is the prompt variant: the form of the input prompt.
- \(r\) is the seed selecting sampled prompts.
- \(g\) is granularity: the resolution of the resulting statement.

Every axis level comes from a published implementation; none was invented here.

G7U, the **claim map**, transforms a circuit into a statement. It operates on discovery output \(C(s)\subseteq E\). Here \(E\) is the full set of graph edges, \(C(s)\) is the circuit selected under specification \(s\), and \(\subseteq\) means that its edges come from \(E\). The resulting structured Annex IV statement is written \(\varphi(C)\), where \(\varphi\) denotes the claim map.

Y3W, or filability, concerns agreement among these claims. It does not adjudicate which specification is correct.

The model is Q7V, or **GPT-2 small**, the small version of Generative Pre-trained Transformer 2. Its factorised graph represents query, key and value inputs separately. These are the three attention inputs used to score and combine information. The graph has \(|E|=32{,}491\) edges, where bars around a set denote its number of elements.

This count was confirmed from the instrument and independently by computation-order counting. For block index \(b\), attention receives \(1+13b\) sources at three input slots per each of 12 heads. An attention head is an individual attention computation. MLP \(b\)—the block’s multilayer perceptron, or feed-forward component—receives those sources plus its block’s heads. The output receives all 157 non-terminal sources, meaning sources before the final output. Components comprise 144 attention heads and 12 MLPs, 156 total.

#### Circuit instability.

For specifications \(s_i\) and \(s_j\), write their circuits as \(C_i=C(s_i)\) and \(C_j=C(s_j)\). Circuit distance is
\[
D(s_i,s_j)=1-\frac{|C_i\cap C_j|}{|C_i\cup C_j|}. \tag{1}
\]

The intersection, \(C_i\cap C_j\), contains shared edges. The union, \(C_i\cup C_j\), contains edges in either circuit. Their size ratio is **Jaccard similarity**; \(D\) is one minus that similarity. Thus larger \(D\) means less overlap.

\(\bar J\) denotes mean Jaccard similarity over unordered pairs, counting each pair once.

#### Claim instability.

A **claim class** \(c\) contains specifications producing the same structured statement \(\varphi(C)\). Let \(n_c\) be the number producing class \(c\), and \(N\) the total number of specifications. Claim instability is
\[
F=1-\frac{\sum_c n_c(n_c-1)}{N(N-1)}. \tag{2}
\]
The sum, \(\sum_c\), runs over claim classes.

This unbiased Gini–Simpson form gives the probability that two distinct specifications drawn without replacement produce different claims. Randomised unit tests compare it with direct enumeration of all pairs.

To condition on axes \(G\), compare only specifications sharing those settings:
\[
F_{\mathrm{within}}(G)=
1-\frac{\sum_g\sum_c n_{g,c}(n_{g,c}-1)}
{\sum_g n_g(n_g-1)}. \tag{3}
\]
In this equation, \(g\) indexes conditioning groups, not the granularity component of \(s\). The count \(n_g\) is the number of specifications in group \(g\); \(n_{g,c}\) is the number in that group producing claim class \(c\). The sums run over groups and classes.

Pairs, not groups, receive equal weight. Empty \(G\), meaning no conditioning axes, reduces exactly to (2).

#### Y3W.

Let \(\pi^*=\max_c n_c/N\) be the **modal claim share**: the largest share producing any one claim. The maximum, \(\max_c\), is taken over claim classes. At tolerance \(\alpha\), the allowed share outside that modal claim, Y3W passes if and only if \(\pi^*\geq1-\alpha\). Section 8 gives the full protocol.

For \(k\) available claim classes, the ceiling \(F\leq1-1/k\) depends on that number of classes. The observed rate is not near one by construction.

| Ceiling diagnostic | Value |
|---|---:|
| Observed classes | 9 |
| Ceiling | 0.8889 |
| Observed fraction of ceiling | 82.3% |

## 5 G7U

Regulatory addressees determine two claim mappings. D9P, or **overseer**, addresses oversight and is grounded in Annex IV 2(e), 3 and Article 14(4)(c). It orders dominant layer band, size class, and ranked input segment. A layer band groups model layers; a size class groups circuits by size; an input segment is a portion of the prompt. The reported layer bands are early, middle and late. The reported size classes are sparse, moderate and distributed.

L3X, or **affected**, addresses the person affected by a decision. Grounded in Article 86(1), it leads with input segment because the right belongs to the decision’s subject.

The nested resolutions are E7R, or **COARSE**; K8Z, or **MEDIUM**, the intermediate resolution; and V2H, or **FINE**, the finest resolution. E7R has no size term. The resolutions are constructed so refinement cannot contradict the coarser statement.

The claim map \(\varphi\) is the paper’s most attackable choice. It is pre-registered, published deterministic code rather than language-model output. Language-model output would inseparably add model variance to circuit variance.

Size bins—categories separated by thresholds—follow a committed calibration rule applied to a measured node-count curve. They were not selected for the reported \(F\), as a reviewer suspected. Table 1 reports the sensitivity analysis. Four alternatives exceed the committed result, including halving thresholds; factor-four binnings remain within the reported range. “Equal thirds” divides sizes into three equally populated bins; “deciles” uses ten.

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

## 6 Method

#### Model and task.

Comparability with the extended non-identifiability and faithfulness literature motivates Q7V, GPT-2 small, and Z4K, **indirect object identification** [15]. The task concerns identifying the indirect object in a prompt. The edge count makes size-matched randomness meaningful, rather than scale serving as an apology.

#### Grid.

The pre-registered grid contains seven objectives, seven operators, four corruption distributions nested within the five operators that read them, four metrics, three thresholds, two prompt variants, and five seeds. This gives 1,540 discovery cells and 18,480 pre-registered specifications. A discovery cell fixes the discovery-side settings; metrics and thresholds supply further specifications. Two operators never read corruption, so crossing them with corruption would duplicate specifications.

Discovery uses R8X, **edge attribution patching** [14], an edge-scoring discovery family, and B3Q, **integrated edge gradients**, an additional gradient-based edge-scoring method. Both use Miller et al.’s [8] implementations. Automated discovery lineage comes from Conmy et al. [2], and corruption levels from Wang et al. [15].

The four metrics are:

- T9V: comprehensiveness.
- H2W: **Kullback–Leibler (KL) divergence**, a measure of difference between probability distributions.
- N6C: **logit difference**, the difference between model output scores before conversion to probabilities.
- U4J: sufficiency.

Their unequal retention materially changes the realised pool. Table 2 reports the metric-specific counts.

#### Pre-registration.

The plan was committed and tagged before the confirmatory sweep, the full run under that plan. Both directional abstracts were drafted in advance, preventing either direction being presented as surprise. An append-only deviations file records dates and whether decisions preceded or followed viewing affected results.

#### Statistics.

No specification-level \(p\)-values are reported: the designed grid is not an independent sample [13, 12]. A \(p\)-value is a probability calculated under a statistical null model; that treatment is not used for individual specifications here.

A nonparametric bootstrap repeatedly resamples the observed specifications with replacement. It resamples specifications, never pairs, with \(B=10{,}000\) resamples. Both \(F\) and \(\bar J\) are pairwise U-statistics: averages formed from pairs of observations. Pairs sharing a specification are dependent.

Confidence intervals (CIs) condition on the grid and license no extrapolation. Bracketed ranges in the results give interval bounds.

#### Two exclusions, criteria fixed before any circuit was seen.

First, every cell of one objective fails inside the verbatim library. Mean squared error—the average squared difference between predictions and targets—against an integral, or integer-valued, target raises a runtime error. A digest of sorted failing identifiers, a compact check of that identifier list, verifies exactly that arm and nothing else. No repair was made to the tested instrument.

Second, the pre-registered metric-relative threshold rule excludes specifications admitting no circuit.

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

The threshold rule produces these exclusions. Unevenness was disclosed before lock from a six-cell validation slice. Composition is reported rather than smoothed: T9V accounts for 52% of survivors, against 25% in the design. In Table 2, “Rate” is the discarded fraction.

Table 2: Metric-specific retention.

| Metric | Kept | Discarded | Rate |
|---|---:|---:|---:|
| T9V | 3,947 | 13 | 0.003 |
| H2W | 1,467 | 2,493 | 0.630 |
| N6C | 1,270 | 2,690 | 0.679 |
| U4J | 877 | 3,083 | 0.779 |

## 7 Results

The registered labels are J6D, or **P0**, the descriptive premise about circuit overlap; X4R, or **H2**, the claim-instability hypothesis; Q2F, or **H3**, the random-comparison hypothesis; and Z8N, or **H4**, the functional-gap hypothesis.

Their decision rules were fixed on 6 August 2026 and tagged before the sweep. They remain unrevised. Two outcomes require within-size qualifications here rather than deferral to an appendix.

For Z8N, “agreement” is the fraction of examples on which two circuits give the same correct/incorrect verdict. Thus \(1-\mathrm{agreement}\) is functional instability, and the registered gap subtracts it from claim instability \(F\).

Table 3: Registered outcomes.

| | Decision rule | Result | Outcome |
|---|---|---|---|
| J6D | \(\bar J\) reported, not tested | 0.1396 [0.1377, 0.1418] | premise holds |
| X4R | \(F>0.20\), CI lower bound above | 0.7316 [0.7247, 0.7380] | confirmed |
| Q2F | not separated from random null | separated, intervals disjoint | rejected |
| Z8N | \(F-(1-\mathrm{agreement})>0.10\) | 0.3344 [0.3297, 0.3394] | supported |

### 7.1 Circuits are almost disjoint

Exact, unsampled pair computation is tractable because some specifications produce duplicate circuits: 7,561 claim-producing specifications yield 3,218 distinct circuits.

Distributional separation exceeds what the mean alone conveys. Mean Jaccard similarity is 0.1396 [0.1377, 0.1418] (Table 3, J6D), while median distance \(D\) is 0.960. Overlap beats chance without implying much common structure: observed/chance overlap is about 45 times, but the median shared-edge fraction is 4%.

In the analytic random line below, \(k\) means circuit size in edges, not the claim-class count used in §4. A quantile marks the value below which the stated percentage of distances falls.

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

Primary claim disagreement is 0.7316 [0.7247, 0.7380] (Table 3, X4R). The lower interval bound clears the registered threshold of 0.20 by more than three times. None of the registered tolerances permits filing: the required modal shares are 0.95, 0.90 and 0.80, whereas the observed primary modal share is 0.4109.

The two most frequent classes attribute the same model/task behaviour to early versus late layers. This is contradiction, not emphasis: Table 4 ranks early/distributed first and late/sparse second. The notation \(\varphi_{\mathrm{D9P}}\) means the overseer claim map.

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

The early aggregate independently reproduces the E7R modal share: 47.9% above, compared with 0.4788 in Table 5’s D9P/E7R row.

Even the coarsest L3X statement rejects Y3W. Its modal share is 0.7291, below the loosest required share of 0.80. Fine-grained language is not required for failure.

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

Standardising an axis means comparing only specifications with the same level on that axis, then combining those comparisons with pair weights as in (3).

The metric dominates single-axis claim control but remains insufficient. It removes 0.1377 of disagreement and leaves \(F=0.5939\) (Table 6). Structural and claim variance have different drivers. Ablation-corruption interaction accounts for 78.8% of the reported structural variance, but standardising the ablation operator removes only 0.0302 of claim disagreement. Standardising the operator selected by structural decomposition would therefore fix almost none of the filing instability.

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

The following variance components use restricted maximum likelihood (REML), a method for estimating contributions to variance. The measured outcome is \(\log_{10}\), the base-ten logarithm, of selected circuit size. “Interaction” is the joint contribution of ablation and corruption settings; “residual” is variation not assigned to the listed fitted components.

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

Here size is entirely absent from statements, and comparisons are conditioned on circuit size: only equal-sized circuits are compared. Dominant-layer attribution still clears the registered instability threshold. The interval’s lower bound, 0.2550, exceeds 0.20.

| Size-free claim, fixed size | Value |
|---|---|
| \(F\) | 0.2706 |
| 95% CI | 0.2550 to 0.2859 |
| Percentage, abstract precision | 27.1% |
| 95% CI, abstract precision | 0.255 to 0.286 |
| Percentage, discussion precision | 27% |

### 7.5 The random baseline, and why its pooled direction misleads

Pooled discovery is less stable than A6N: discovered \(F\) is 0.7316, against 0.6774 for A6N, with disjoint intervals. Rejection of Q2F concerns separation, not its size-confounded direction. The pooled comparison mixes different circuit sizes.

Uniform null draws above the degeneracy threshold touch every component and produce only one claim. Here **degeneracy** means collapse to a single claim class. Fixed-size comparison reverses the ordering: discovered circuits outperform randomness but still fail Y3W.

| Comparison | Discovered \(F\) | A6N \(F\) |
|---|---:|---:|
| Pooled | 0.7316 | 0.6774 |
| Fixed size, K8Z | 0.2746 | 0.4230 |
| Fixed size, E7R | 0.2706 | 0.3822 |

A6N samples the full edge population with \(R=1{,}000\) null multiverses, where \(R\) is the number of random baseline multiverses. Its degeneracy occurs above 1,000 edges. In Table 7, size is the number of edges, \(n\) is the number of specifications at that size, and class counts refer to distinct resulting claims.

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

Agreement uses per-example correct/incorrect verdicts and Bayat Makou et al.’s [1] measures reproduced verbatim. Raw agreement is the fraction of matching verdicts. Cohen’s kappa adjusts that agreement for agreement expected from the verdict frequencies.

Raw agreement mostly reflects circuits being right on similar fractions, not identical examples. Pooled raw agreement is 0.6028, but kappa is 0.0146. Within size, raw agreement climbs from 0.52 to 0.92 as accuracy tracks from 0.51 to 0.95; kappa remains between 0.016 and 0.047. The functional interchangeability required by the W5T objection is absent.

Z8N holds on its registered pooled quantity, not under size conditioning. The pooled gap is 0.3344 [0.3297, 0.3394] (Table 3), but 6 of 10 sizes fail the gap threshold and 5 have a negative gap. A negative gap means functional instability exceeds claim instability. Mechanisms differ more than filings at both range ends; only the intermediate band, 500 to 2,000 edges, carries the pooled gap.

Registration requires reporting the pooled result. Decomposition prevents its misleading interpretation.

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

1. Declare \(\mathcal S\), the specification space: defensible analytic axes and levels, each cited.
2. Declare \(\varphi\), the claim map: published deterministic code, not a language model.
3. Compute \(\pi^*=\max_c n_c/N\) over \(\mathcal S\), and \(F\). The modal share \(\pi^*\) is the largest claim-class count divided by the specification count; \(F\) is pair disagreement.
4. Evidence passes Y3W at tolerance \(\alpha\) if and only if \(\pi^*\geq1-\alpha\).
5. Report \(\pi^*\), \(F\), discard rate per axis level, and within-group decomposition of every pooled statistic.

A filing unable to state its own \(\pi^*\) is asserted, not audited. Implementability requires only a declared space and measured agreement. It does not require regulatory adjudication of analytic correctness.

## 9 Limitations

#### One model, one task.

Larger models permit more layer-band and size-class variation. The direction for \(F\) is unclear. This limitation is most likely to change its value.

The committed, timestamped second-model plan uses M9B, **Pythia-160m**, a different architecture family performing the same task. It specifies a 924-cell grid, identical on every axis except three rather than five seed levels. Replication-versus-refutation criteria—conditions for treating the run as supporting or opposing the result—were fixed before any cell ran.

The plan remains unrun and will appear in a revision regardless of direction. A second task is designed but unrun.

#### Circuit size is an outcome, not an axis.

Threshold-selected size distorts \(F\), null-comparison direction, and functional-gap sign. Conditioning on size reduces claim instability more than standardising any registered axis: fixed-size K8Z gives 0.2746 (§7.5), while the lowest single-axis residual is 0.5939 (Table 6). Removing size from statements as well gives 0.2706 (§7.4).

This is a general multiverse hazard when specifications select differently sized objects. Every pooled statistic is reported with within-size decomposition.

#### The multiverse is as wide as the library, not the literature.

C4Y, **optimal ablation** [5], is a defensible but unsupported ablation operator. It never enters the instrument’s space. This makes measured \(F\) a lower bound relative to the field’s full defensible choice set.

#### The discard rate is high and uneven.

The retained metric imbalance is a reported-set property, not adjusted away. Overall discard is 52.3%; Table 2 gives metric-specific rates from 0.003 to 0.779. The threshold was fixed in advance, and no failed item was repaired.

#### The bootstrap is invalid for one reported family.

Conditioning groups containing only two to five specifications retain dominating self-pairs: bootstrap copies of the same original specification can pair with one another. Unusable intervals are replaced by point estimates with diagnostics.

No bias-corrected estimator was constructed after naive intervals disagreed. Result-responsive estimator selection is the behaviour under study.

## 10 Discussion

#### What standardising one axis buys.

Complete metric standardisation exceeds what a conformity procedure could demand. Yet it leaves 59.4% disagreement, or \(F=0.5939\) (Table 6, evaluation metric). No single knob suffices. Declaring resolution after standardising one axis would be substantially mistaken.

#### On the instrument.

The failed objective is documented and publicly exported yet cannot execute on the library’s own canonical task. All 220 cells of that objective fail (§6). Its discard rate also records tooling maturity beneath prospective regulatory evidence.

This is neither the headline nor criticism of authors among those requesting these improvements [10].

#### What a regulator can take from this.

Interpretability evidence is not worthless. Evidentiary weight is measurable rather than presumed. The Y3W protocol measures agreement over a declared analytic space; measurement is cheap relative to the sweep producing the circuit.

## 11 Reproducibility

The public timestamped commit—a recorded code version—is tagged `prereg-p1-confirmatory`. This literal tag identifies the pre-registered confirmatory version. Every reported number traces to its configuration, seed and environment fingerprint, a record identifying the execution environment.

Archive verification establishes one environment across all 1,320 confirmatory cells. Content-addressed comparison, which identifies file contents by their computed digests, recovers the code commit from the uploaded archive against the git tree, the version-controlled source-file tree: 22 of 22 files identical. Analysis needs no graphics processing unit (GPU).

One script recomputes headline quantities from raw results and asserts reported values: 36 checks, 0 failures. It checks \(F\) through three routes, including brute-force pairs. Another checks every manuscript numeric token against stored-result derivations. Tables are emitted directly, never transcribed. The suite contains 343 tests.
