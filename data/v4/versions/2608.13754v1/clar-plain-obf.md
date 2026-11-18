# Explanation Multiplicity: Circuit-Level Interpretability

Evidence Does Not Survive Defensible Analytic Variation

Ajay Pravin Mahale

12 August 2026

###### Abstract

Regulatory evidence derived from circuits—subgraphs said to explain model behaviour—fails Y3W, this paper’s filing-agreement test, at all tolerances plausibly acceptable to a conformity-assessment body. Competent analysts who use the same system and tool reach incompatible statements under different defensible settings. Standardising settings and controlling circuit size do not remove this failure. Differences in structure and function rule out the possibility that the accounts merely reword the same mechanism. Whether these findings apply beyond one model and task remains untested.

## 1 Introduction

An audit requires two auditors to reach the same conclusion. This is a commissioning requirement, not a machine-learning definition.

Regulation (EU) 2024/1689 requires high-risk AI providers to file technical documentation. Annex IV 2(b) requests “the general logic of the AI system and of the algorithms”. Annex IV 2(e) requests “an assessment of the technical measures needed to facilitate the interpretation of the outputs”. Article 86(1) grants persons subject to certain automated decisions “clear and meaningful explanations of the role of the AI system in the decision-making procedure and the main elements of the decision taken”. These quotations follow the consolidated text amended by Regulation (EU) 2026/1744, CELEX 02024R1689-20260727, read 11 August 2026.

Mechanistic interpretability seeks to explain behaviour through a model’s internal mechanisms. It offers possible technical content for these documents. Its mature tool, circuit discovery, returns a subgraph: a subset of the model’s components and connections said to explain behaviour. Filing such an account assumes that it is compatible with another analyst’s account. This paper measures how regulatory claims change across defensible specifications, using a standards-body decision rule. A specification is a particular combination of analysis settings.

#### Contributions.

The pre-registered crossed design commits the plan before the confirmatory study and combines settings across analysis dimensions. It uses only published implementations. A deterministic procedure converts circuits into structured Annex IV statements: the same circuit always produces the same statement. Controls standardise individual dimensions, remove claims’ dependence on size, and compare results within size groups. Three pooled statistics—statistics combined across specifications—reverse direction when the selected objects differ in size. This paper proposes no improved discovery method. Its objection concerns the standard assumed when filing evidence that current methods produce, not the value of interpretability.

## 2 Related work

#### Non-identifiability.

Non-identifiability means that the criteria do not determine a unique explanation. Méloux et al. [6] find that a behaviour need not have one unique explanation under mechanistic interpretability’s own criteria. This paper retains that premise. It measures the downstream cost for filing rather than publication.

#### Faithfulness metric sensitivity.

A faithfulness metric measures how well a circuit accounts for model behaviour. Ablation tests this by removing or replacing internal activity. Miller et al. [8] find that ablation-based faithfulness is highly sensitive to seemingly insignificant design changes. Scores reflect researchers’ choices as well as circuits. This paper extends their single-axis finding to regulatory effects across seven axes, or dimensions of analytic choice. Their released library supplies the measurement tool, which this paper tests under its own definitions.

#### Structure versus function.

Bayat Makou et al. [1] hold the task fixed and vary input statistics. They find structurally distinct circuits that implement the same computation. This pattern, W5T: phantom specialisation, supplies the strongest objection to the present design. Different claims might describe sampled members of a behaviourally identical equivalence class—a set of circuits with the same behaviour. This paper addresses that objection through functional measurement rather than argument.

#### Variance as estimation.

Méloux et al. [7] analyse mechanistic interpretability as statistical estimation. Their analysis includes estimator variance: how much an estimate changes across samples. Seed variance is one axis here. Their framework provides the statistical counterpart to analysis across whole specifications.

#### Auditability.

Lan et al. [4] describe two papers that reach conflicting conclusions about one behaviour. A third finds both partly correct and incomparable. This paper responds to their request for safety-critical certification standards with a measurement and criterion, not a proposal. Sharkey et al. [10] include relevant open problems involving both technical and social factors. Mueller et al. [9] require standardised evaluation for method comparison. That requirement is related to standardising specifications for certification.

#### Multiverse method.

A multiverse analysis reports results across multiple defensible analysis choices. Steegen et al. [13] prescribe reporting across defensible data-processing choices. This paper adapts Simonsohn et al.’s [12] visualisation, F9A: specification curve, to regulatory evidence. In this setting, undisclosed analytic flexibility can affect stability claims, as identified by Simmons et al. [11]. This transfer from psychology methods is substantive: flexibility can manufacture explanation stability.

#### Explanation metrics under optimisation pressure.

Hsia et al. [3] inflate U4J and T9V without changing predictions or explanations. U4J is sufficiency, which measures how much the parts selected by an explanation retain. T9V is comprehensiveness, which measures what is lost when those parts are removed. Both appear among our four metrics. Reporting discards separately for each metric avoids silently pooling them.

## 3 What this paper is not claiming

Before we separated the results into their parts, we ourselves held two unsupported readings. Discovered circuits carry information and, at fixed size, outperform A6N: the size-matched random null, a baseline of randomly selected circuits with matching size. Yet the mechanisms themselves differ substantially and are functionally uncorrelated. They are not one mechanism presented in different forms.

The seed selects sampled prompts. It measures evaluation-set variability, not nondeterminism. This paper makes no claim about scale.

## 4 Formalism

A specification is
\[
s=(o,a,d,m,\tau,P,r,g),
\]
where the entries respectively denote the discovery objective, ablation operator, corruption distribution, evaluation metric, size threshold, prompt variant, seed, and granularity.

The discovery objective is the quantity that discovery targets. The ablation operator specifies how internal activity is removed or replaced. The corruption distribution specifies how altered inputs or activity are sampled. The evaluation metric scores the result. The size threshold controls circuit selection. The prompt variant changes the prompt form. Granularity sets the level of detail in the statement. Every level of every axis comes from a published implementation; none was invented here.

G7U, the claim map, transforms a circuit into a statement. It operates on discovery output \(C(s)\subseteq E\), where \(C(s)\) is the circuit found under specification \(s\), and \(E\) is the full set of edges, or connections. Y3W, called filability, concerns agreement among the resulting claims. It does not judge which specification is correct.

For Q7V, the GPT-2 small model, the factorised graph represents query, key, and value as separate attention inputs. These are the three input types used by the attention computation. The graph has \(|E|=32{,}491\) edges. We confirmed this count from the tool and independently by counting in computation order. Attention in block \(b\) receives \(1+13b\) sources at three input slots for each of 12 heads. An attention head is a component that selects and combines information. The multilayer perceptron (MLP), a neural-network block, in block \(b\) receives those sources plus its block’s heads. The output receives all 157 non-terminal sources. The components comprise 144 attention heads and 12 MLPs, 156 total.

#### Circuit instability.

For circuits \(C_i\) and \(C_j\) from specifications \(s_i\) and \(s_j\), define
\[
D(s_i,s_j)=1-\frac{|C_i\cap C_j|}{|C_i\cup C_j|}. \tag{1}
\]
Here, the intersection contains shared edges, the union contains edges in either circuit, and vertical bars denote counts. The fraction is Jaccard similarity, a measure of overlap. Thus \(D\) measures lack of overlap.

\(\bar J\) denotes mean Jaccard similarity over unordered pairs, with each pair counted once.

#### Claim instability.

Let \(\varphi(C)\) be a structured Annex IV statement derived from circuit \(C\). Let \(n_c\) count specifications that produce claim class \(c\), and let \(N\) count all specifications. A claim class groups identical structured statements. Define
\[
F=1-\frac{\sum_c n_c(n_c-1)}{N(N-1)}. \tag{2}
\]
This unbiased Gini–Simpson form gives the probability that two distinct draws without replacement disagree. Randomised unit tests compare it with direct enumeration of all pairs.

When comparisons condition on axes \(G\), define
\[
F_{\mathrm{within}}(G)=
1-\frac{\sum_g\sum_c n_{g,c}(n_{g,c}-1)}
{\sum_g n_g(n_g-1)}. \tag{3}
\]
Here, \(g\) denotes a group with the same settings on axes \(G\), \(n_g\) is its size, and \(n_{g,c}\) counts its members in claim class \(c\). Pairs, not groups, receive equal weight. Empty \(G\) reduces exactly to (2).

#### Y3W.

The ceiling \(F\leq1-1/k\) depends on the number \(k\) of available claim classes. The construction therefore does not force the observed rate to be near one.

| Ceiling diagnostic | Value |
|---|---:|
| Observed classes | 9 |
| Ceiling | 0.8889 |
| Observed fraction of ceiling | 82.3% |

## 5 G7U

Size bins follow a committed calibration rule applied to a measured node-count curve, which tracks the number of graph components. They do not reflect the reviewer-suspected selection for the reported \(F\). Four alternatives exceed the committed result, including halving the thresholds. Binnings that span a factor of four remain within the reported range.

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

The intended regulatory audience determines two mappings. D9P: overseer draws on Annex IV 2(e), 3 and Article 14(4)(c). It orders the statement by dominant layer band, size class, and ranked input segment. These describe the dominant range of model layers, the circuit-size category, and the ordered input portions. L3X: affected draws on Article 86(1). It leads with the input segment because the right belongs to the person subject to the decision.

The resolutions are nested: E7R: COARSE, K8Z: MEDIUM, and V2H: FINE. A more detailed statement cannot contradict its coarser version. The claim map \(\varphi\) is the paper’s most open-to-challenge choice. It is pre-registered, published deterministic code, not language-model output. Language-model output would inseparably add model variance to circuit variance.

## 6 Method

#### Model and task.

We chose Q7V: GPT-2 small and Z4K: indirect object identification [15] for comparison with the extended literature on non-identifiability and faithfulness. Indirect object identification asks the model to identify the indirect object in a prompt. The edge count makes size-matched randomness meaningful; scale does not serve as an excuse.

#### Grid.

The grid contains seven objectives, seven operators, four corruption distributions nested within the five operators that read them, four metrics, three thresholds, two prompt variants, and five seeds. This gives 1,540 discovery cells and 18,480 pre-registered specifications. Two operators never read corruption. Crossing those operators with corruption would duplicate specifications.

Discovery uses R8X: edge attribution patching [14] and the additional method B3Q: integrated edge gradients, as implemented by Miller et al. [8]. Edge attribution patching estimates the effects of replacing activity along edges. Integrated edge gradients accumulates gradient-based edge effects along an integration path. The automated discovery lineage comes from Conmy et al. [2], and the corruption levels come from Wang et al. [15].

The four metrics are T9V: comprehensiveness, H2W: KL divergence, N6C: logit difference, and U4J: sufficiency. KL means Kullback–Leibler divergence, a measure of difference between probability distributions. Logit difference compares model output scores before their conversion to probabilities. The metrics’ unequal retention materially changes the pool that remains.

#### Pre-registration.

We committed and tagged the plan before the confirmatory sweep, the full run of the planned grid. We drafted both directional abstracts in advance, preventing either direction from being presented as a surprise. An append-only deviations file records dates and whether decisions came before or after viewing the affected results.

#### Statistics.

We report no specification-level \(p\)-values, or statistical significance probabilities. The designed grid is not an independent sample [13, 12]. A nonparametric bootstrap, which resamples observed units to estimate uncertainty, resamples specifications, never pairs, with \(B=10{,}000\) resamples. Both \(F\) and \(\bar J\) are pairwise U-statistics: averages based on pairs. Pairs that share a specification are dependent. The intervals condition on the grid and permit no extrapolation. Reported confidence intervals use the abbreviation CI.

#### Two exclusions, criteria fixed before any circuit was seen.

Every cell of one objective fails inside the unchanged library. Computing mean squared error against an integer-valued target raises a runtime error. A digest—a summary used to check identity—of sorted failing identifiers verifies that exactly this arm failed, and nothing else. We did not repair the tested tool.

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

The pre-registered threshold rule, defined relative to each metric, produces the exclusions. We disclosed the unevenness before locking the analysis, based on a six-cell validation slice. We report the resulting composition rather than smooth it.

Table 2: Metric-specific retention.

| Metric | Kept | Discarded | Rate |
|---|---:|---:|---:|
| T9V | 3,947 | 13 | 0.003 |
| H2W | 1,467 | 2,493 | 0.630 |
| N6C | 1,270 | 2,690 | 0.679 |
| U4J | 877 | 3,083 | 0.779 |

## 7 Results

The descriptive premise J6D: P0, claim-instability hypothesis X4R: H2, random-comparison hypothesis Q2F: H3, and functional-gap hypothesis Z8N: H4 retain their original decision rules. We fixed those rules on 6 August 2026 and tagged them before the sweep, without revision. Two outcomes require qualifications within size groups rather than deferral to an appendix.

Table 3: Registered outcomes.

| | Decision rule | Result | Outcome |
|---|---|---|---|
| J6D | \(\bar J\) reported, not tested | 0.1396 [0.1377, 0.1418] | premise holds |
| X4R | \(F>0.20\), CI lower bound above | 0.7316 [0.7247, 0.7380] | confirmed |
| Q2F | not separated from random null | separated, intervals disjoint | rejected |
| Z8N | \(F-(1-\mathrm{agreement})>0.10\) | 0.3344 [0.3297, 0.3394] | supported |

### 7.1 Circuits are almost disjoint

Repeated circuits make exact computation over all pairs feasible without sampling. The distribution shows more separation than the mean alone conveys. Overlap exceeds chance, but this does not imply much shared structure.

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

In the analytic random line, \(k\) denotes circuit size.

### 7.2 The claim flips on three quarters of pairs

The lower interval bound exceeds the registered threshold by more than three times. None of the registered tolerances permits filing. The two most frequent claims attribute the same model/task behaviour to early versus late layers. This is contradiction, not emphasis.

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

The modal share is the share of the most frequent claim. Aggregating the early-layer claims independently reproduces the E7R modal share. Even the coarsest L3X statement rejects Y3W. Fine-grained language is not required.

Table 5: All addressee-resolution combinations; loosest registered tolerance \(\alpha=0.20\), required \(\pi^*\geq0.80\). Here, \(\alpha\) is the allowed disagreement tolerance, and \(\pi^*\) is the modal share.

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

Figure 1: F9A for \(\varphi_{\mathrm{D9P}}\), K8Z. Upper panel: claim class, ordered by frequency, against specification rank. Step widths equal class shares; the modal width is \(\pi^*\). The dashed line shows the A6N modal share, whose pooled comparison mixes the effect of size with the comparison of interest. Lower panel: active analytic levels for each specification. The solid T9V band reflects the composition caused by discards. The seed rows lack structure, reflecting the negligible seed contribution.

### 7.3 No analytic axis rescues the filing

Standardising the metric reduces claim disagreement more than standardising any other single axis, but it remains insufficient. Different choices drive structural variance and claim variance. Structural decomposition would select the operator for standardisation, yet doing so would remove almost none of the filing instability.

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

Restricted maximum likelihood (REML) estimates the variance components below. The size measure uses the base-10 logarithm, written \(\log_{10}\), of selected circuit size. The ablation-corruption interaction captures their joint contribution beyond their separate contributions.

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

We removed size entirely from the statements and conditioned comparisons on size. Dominant-layer attribution still exceeds the registered instability threshold at the interval’s lower bound.

| Size-free claim, fixed size | Value |
|---|---|
| \(F\) | 0.2706 |
| 95% CI | 0.2550 to 0.2859 |
| Percentage, abstract precision | 27.1% |
| 95% CI, abstract precision | 0.255 to 0.286 |
| Percentage, discussion precision | 27% |

### 7.5 The random baseline, and why its pooled direction misleads

In the pooled comparison, discovery is less stable than A6N, with disjoint intervals. Rejecting Q2F concerns separation, not the direction of that difference, which mixes in the effect of size. Uniform random draws above the degeneracy threshold touch every component and produce only one claim. Comparing at fixed size reverses the ordering. Discovered circuits outperform randomness but still fail Y3W.

| Comparison | Discovered \(F\) | A6N \(F\) |
|---|---:|---:|
| Pooled | 0.7316 | 0.6774 |
| Fixed size, K8Z | 0.2746 | 0.4230 |
| Fixed size, E7R | 0.2706 | 0.3822 |

A6N samples the full edge population with \(R=1{,}000\) null multiverses, or randomly generated comparison grids. Its degeneracy—collapse to one claim—occurs above 1,000 edges.

Table 7: K8Z within-size results. A6N collapses at and above 1,000 edges; discovery also collapses at 5,000 and 10,000. Thus 2,692 of 7,561 specifications contribute no flips; instability lives entirely below 2,000 edges. Here, \(n\) counts specifications.

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

Agreement uses correct/incorrect verdicts for each example and Bayat Makou et al.’s [1] measures, reproduced unchanged. Raw agreement mostly reflects circuits being right on similar fractions of examples, not on identical examples. The functional interchangeability required by the W5T objection is absent.

Z8N holds for its registered pooled quantity, but not after conditioning on size. Mechanisms differ more than filings at both ends of the size range. Only the intermediate band carries the pooled gap. Registration requires reporting the pooled result. Separating the results by size prevents a misleading interpretation.

Cohen’s kappa measures agreement after accounting for chance agreement.

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

For a filed claim \(c\) derived from a circuit:

1. Declare \(\mathcal S\), the set of defensible analytic axes and levels. Cite each one.
2. Declare the claim map \(\varphi\) as published deterministic code, not a language model.
3. Compute \(\pi^*=\max_c n_c/N\), the largest claim share over \(\mathcal S\), and compute \(F\).
4. Evidence passes Y3W at tolerance \(\alpha\) if and only if \(\pi^*\geq1-\alpha\).
5. Report \(\pi^*\), \(F\), the discard rate for each axis level, and a within-group decomposition of every pooled statistic.

A filing that cannot state its own \(\pi^*\) is asserted, not audited. Applying the protocol requires only a declared space and measured agreement. It does not require regulators to judge which analytic choice is correct.

## 9 Limitations

#### One model, one task.

Larger models permit more variation in layer bands and size classes. The direction of the effect on \(F\) is unclear. This limitation is the one most likely to change its value.

The committed, timestamped second-model plan uses M9B: Pythia-160m. It specifies a different architecture family performing the same task. Its 924-cell grid matches every axis except that it uses three rather than five seed levels. The plan fixes criteria for replication versus refutation before any cell runs. It remains unrun and will appear in a revision regardless of direction. A second task is designed but unrun.

#### Circuit size is an outcome, not an axis.

Size selected by the threshold distorts \(F\), the direction of the null comparison, and the sign of the functional gap. Conditioning on size reduces claim instability more than standardising any registered axis. This is a general multiverse hazard when specifications select objects of different sizes. Every pooled statistic is reported with a within-size decomposition.

#### The multiverse is as wide as the library, not the literature.

C4Y: optimal ablation [5] is a defensible operator that optimises the ablation choice, but the library does not support it. It never enters the tool’s choice space. Measured \(F\) is therefore a lower bound relative to the field’s full defensible choice set.

#### The discard rate is high and uneven.

The imbalance among retained metrics is a property of the reported set. We do not adjust it away. We fixed the threshold in advance and repaired no failed item.

#### The bootstrap is invalid for one reported family.

Conditioning groups with only two to five specifications retain self-pairs that dominate the result. These pairs match a specification with itself. We replace unusable intervals with point estimates and diagnostics. We did not construct a bias-corrected estimator after the naive intervals disagreed. Selecting an estimator in response to results is the behaviour under study.

## 10 Discussion

#### What standardising one axis buys.

Complete metric standardisation exceeds what a conformity procedure could demand. No single setting suffices. Declaring the issue resolved after standardising one axis would be substantially mistaken.

#### On the instrument.

The failed objective is documented and publicly exported, yet it cannot run on the library’s own canonical task. The tool’s discard rate also records its level of maturity beneath prospective regulatory evidence. This is neither the headline nor criticism of authors who are among those requesting these improvements [10].

#### What a regulator can take from this.

Interpretability evidence is not worthless. Its evidentiary weight can be measured rather than presumed. That measurement is cheap relative to the sweep that produces the circuit.

## 11 Reproducibility

The public timestamped commit is tagged `prereg-p1-confirmatory`. Every reported number traces to a configuration, seed, and environment fingerprint, which identifies the software environment. Archive verification establishes one environment across all 1,320 confirmatory cells. Comparison by file content recovers the code commit by checking the uploaded archive against the git tree. All 22 of 22 files are identical. Analysis needs no graphics processing unit (GPU).

One script recomputes the headline quantities from raw results and checks them against reported values. It performs 36 checks with 0 failures. It computes \(F\) through three routes, including direct enumeration of all pairs. Another script checks every numeric token in the manuscript against derivations from stored results. The code emits tables directly; no one transcribes them. The test suite contains 343 tests.
