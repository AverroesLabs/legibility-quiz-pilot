# Explanation Multiplicity: Circuit-Level Interpretability

Evidence Does Not Survive Defensible Analytic Variation

Ajay Pravin Mahale

12 August 2026

###### Abstract

Circuit-derived regulatory evidence fails filability (Y3W)—the requirement that a sufficiently large share of defensible specifications produce the same filed claim—throughout tolerances plausibly acceptable to a conformity-assessment body. In other words, competent analysts sharing a system and tool obtain incompatible statements under different defensible settings. All addressee-resolution combinations fail even the loosest registered tolerance (Table 5). Standardisation and size controls do not remove the failure: metric-standardised disagreement remains 59.4% (Table 6), and size-free, fixed-size disagreement remains 27.1% (Section 7.4). Moreover, structural and functional disagreement exclude merely reworded mechanisms: median circuit distance is 0.960 (Section 7.1), and pooled Cohen’s kappa is 0.0146 (Section 7.6). Applicability beyond one model and task remains untested.

## 1 Introduction

An audit requires two auditors to reach the same conclusion. This requirement is a commissioning requirement, not a machine-learning definition.

Regulation (EU) 2024/1689 requires high-risk artificial intelligence (AI) providers to file technical documentation. Annex IV 2(b) requests “the general logic of the AI system and of the algorithms”; Annex IV 2(e) requests “an assessment of the technical measures needed to facilitate the interpretation of the outputs”. Article 86(1) grants persons subject to certain automated decisions “clear and meaningful explanations of the role of the AI system in the decision-making procedure and the main elements of the decision taken”. Quotations follow the consolidated text amended by Regulation (EU) 2026/1744, CELEX 02024R1689-20260727, read 11 August 2026.

Mechanistic interpretability supplies prospective technical content for such documentation. Its mature instrument, circuit discovery, returns a subgraph—a circuit—said to explain behaviour. Filing a circuit-derived account presumes compatibility with another analyst’s account. Accordingly, this paper measures movement in regulatory claims across defensible specifications and assesses that movement with a standards-body decision rule.

#### Contributions.

The pre-registered crossed design uses published implementations exclusively, and a deterministic transformation converts discovered circuits into structured Annex IV statements. Controls standardise individual analytic axes, remove claim size dependence, and condition comparisons on size. These controls matter because three pooled statistics invert when selected objects differ in size. No improved discovery method is proposed. Rather, the objection concerns the standard presumed by filing currently produced evidence, not interpretability’s worth.

## 2 Related work

#### Non-identifiability.

Méloux et al. [6] find that a behaviour need not have a unique explanation under mechanistic interpretability’s own criteria. This paper retains that premise but measures its downstream cost for filing rather than publication.

#### Faithfulness metric sensitivity.

Miller et al. [8] find ablation-based faithfulness highly sensitive to seemingly insignificant design changes: faithfulness scores reflect researcher choices as well as circuits. This paper extends their single-axis finding into a seven-axis analysis of regulatory propagation. Their released library supplies the instrument, which is tested under the library’s own definitions.

#### Structure versus function.

Bayat Makou et al. [1] hold the task fixed, vary input statistics, and find structurally distinct circuits implementing the same computation. This pattern, labelled phantom specialisation (W5T), supplies the strongest design objection to the present study: claim differences might identify sampled members of a behaviourally identical equivalence class. Functional measurement, rather than argument, addresses that objection here.

#### Variance as estimation.

Méloux et al. [7] analyse mechanistic interpretability as statistical estimation, including estimator variance. Seed variance is one axis in the present study; their estimation framing is therefore the statistical companion to this paper’s whole-specification analysis.

#### Auditability.

Lan et al. [4] describe two papers’ conflicting conclusions about one behaviour and a third paper finding both conclusions partly correct and incomparable. Their request for safety-critical certification standards receives a measurement and criterion here, not a proposal. Sharkey et al. [10] include relevant socio-technical open problems. Mueller et al. [9] require standardised evaluation for method comparison, a concern adjacent to standardised specification for certification.

#### Multiverse method.

Steegen et al. [13] prescribe reporting across defensible data-processing choices. Simonsohn et al.’s [12] visualisation, the specification curve (F9A), is adapted here to an evidentiary setting in which stability claims are susceptible to the undisclosed analytic flexibility identified by Simmons et al. [11]. The transfer from psychology methods is substantive because analytic flexibility can manufacture explanation stability.

#### Explanation metrics under optimisation pressure.

Hsia et al. [3] inflate sufficiency (U4J) and comprehensiveness (T9V) without changing predictions or explanations. Sufficiency and comprehensiveness both occur among this paper’s four metrics. Because metric-specific retention differs, this paper reports discards separately by metric rather than silently pooling retained specifications.

## 3 What this paper is not claiming

Before decomposition, two unsupported readings were our own. First, discovered circuits do carry information: at fixed size, discovered circuits outperform the size-matched random null (A6N), as shown in Section 7.5. Second, the mechanisms themselves differ substantially and are functionally uncorrelated, rather than being one mechanism differently dressed; Section 7.6 reports the functional evidence.

The seed selects sampled prompts. Seed variation therefore measures evaluation-set variability, not nondeterminism. No scale claim is made.

## 4 Formalism

A specification is
\[
s=(o,a,d,m,\tau,P,r,g),
\]
where \(o\) is the discovery objective, \(a\) the ablation operator, \(d\) the corruption distribution, \(m\) the evaluation metric, \(\tau\) the size threshold, \(P\) the prompt variant, \(r\) the seed, and \(g\) the granularity. Every level of every axis comes from a published implementation; no axis level was invented here.

Let \(E\) be the graph’s edge set, and let \(C(s)\subseteq E\) be the circuit returned by discovery under specification \(s\). A circuit-to-statement transformation, called the claim map (G7U), operates on \(C(s)\). The admissibility property, filability (Y3W), concerns agreement among the resulting claims rather than adjudication of a correct specification. Section 8 gives the filability protocol.

The model is GPT-2 small (Q7V). For GPT-2 small’s factorised graph, which has separate query, key and value inputs, \(|E|=32{,}491\). This edge count was confirmed both from the instrument and independently by computation-order counting. With \(b\) indexing a block, attention in block \(b\) receives \(1+13b\) sources at three input slots per each of 12 heads. The multilayer perceptron (MLP) in block \(b\) receives those sources plus its block’s heads. The output receives all 157 non-terminal sources. The components comprise 144 attention heads and 12 MLPs, for 156 components in total.

#### Circuit instability.

For two specifications \(s_i\) and \(s_j\), write \(C_i=C(s_i)\) and \(C_j=C(s_j)\). Their circuit distance is one minus their Jaccard similarity:
\[
D(s_i,s_j)=1-\frac{|C_i\cap C_j|}{|C_i\cup C_j|}. \tag{1}
\]

Here, \(|C_i\cap C_j|\) counts shared edges, and \(|C_i\cup C_j|\) counts edges in either circuit. The symbol \(\bar J\) denotes mean Jaccard similarity over unordered pairs of specifications.

#### Claim instability.

Let \(\varphi(C)\) be the structured Annex IV statement produced from circuit \(C\). Let \(n_c\) be the number of specifications producing claim class \(c\), and let \(N\) be the total number of specifications. Claim instability is
\[
F=1-\frac{\sum_c n_c(n_c-1)}{N(N-1)}. \tag{2}
\]
This unbiased Gini–Simpson form gives the probability of claim disagreement for distinct draws without replacement. Randomised unit tests compare equation (2) with brute-force enumeration of pairs.

To condition on a set of axes \(G\), group specifications by their levels on those axes. Let \(n_g\) be the number of specifications in group \(g\), and let \(n_{g,c}\) be the number in group \(g\) producing claim class \(c\). The within-group disagreement is
\[
F_{\mathrm{within}}(G)=
1-\frac{\sum_g\sum_c n_{g,c}(n_{g,c}-1)}
{\sum_g n_g(n_g-1)}. \tag{3}
\]
Thus pairs, not groups, receive equal weight. When \(G\) is empty, equation (3) reduces exactly to equation (2).

#### Y3W.

For \(k\) available claim classes, the ceiling \(F\leq1-1/k\) depends on the number of available classes. Therefore, the observed disagreement rate is not near one by construction. The diagnostic below reports 9 observed classes, a ceiling of 0.8889, and an observed rate equal to 82.3% of that ceiling.

| Ceiling diagnostic | Value |
|---|---:|
| Observed classes | 9 |
| Ceiling | 0.8889 |
| Observed fraction of ceiling | 82.3% |

For the filability decision, \(\pi^*=\max_c n_c/N\) denotes the largest claim-class share. At tolerance \(\alpha\), evidence passes filability if and only if \(\pi^*\geq1-\alpha\). This decision rule concerns the concentration of claims, while \(F\) reports pairwise disagreement.

## 5 G7U

The claim map uses three nested resolutions: COARSE (E7R), the intermediate resolution MEDIUM (K8Z), and the finest resolution FINE (V2H).

The size bins follow a committed calibration rule applied to a measured node-count curve. The bins were not selected, as a reviewer suspected, for the reported \(F\). Table 1 reports the sensitivity evidence: four alternatives exceed the committed result of 0.7316, including halving the thresholds, which gives 0.7448. Factor-four binnings remain within the reported range.

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

Regulatory addressees determine two claim mappings. The overseer mapping (D9P), grounded in Annex IV 2(e), 3 and Article 14(4)(c), orders the statement by dominant layer band, size class, and ranked input segment. In contrast, the affected-person mapping (L3X), grounded in Article 86(1), leads with input segment because the right belongs to the subject of the decision.

The COARSE, MEDIUM and FINE resolutions are constructed so that refinement cannot contradict the coarser statement. The claim map \(\varphi\) is the paper’s most attackable choice. However, the claim map is pre-registered, published deterministic code rather than language-model output, because language-model output would inseparably add model variance to circuit variance.

## 6 Method

#### Model and task.

Comparability with the extended non-identifiability and faithfulness literature motivates the choice of GPT-2 small (Q7V) and the task of indirect object identification (Z4K) [15]. The graph’s 32,491-edge count makes size-matched randomness meaningful; scale is not offered as an apology.

#### Grid.

The designed grid contains seven discovery objectives, seven ablation operators, four corruption distributions nested within the five operators that read corruption, four evaluation metrics, three thresholds, two prompt variants, and five seeds. These choices yield 1,540 discovery cells and 18,480 pre-registered specifications. The other two operators never read corruption, so crossing those operators with corruption distributions would duplicate specifications.

Discovery uses edge attribution patching (R8X) [14] and the additional method integrated edge gradients (B3Q), as implemented by Miller et al. [8]. The automated discovery lineage is in Conmy et al. [2], and the corruption levels come from Wang et al. [15].

The four evaluation metrics are comprehensiveness (T9V), Kullback–Leibler (KL) divergence (H2W), logit difference (N6C), and sufficiency (U4J). Unequal retention across these metrics materially changes the realised pool, as Table 2 shows.

#### Pre-registration.

The plan was committed and tagged before the confirmatory sweep. Both directional abstracts were drafted in advance, preventing either direction from being presented as a surprise. An append-only deviations file records dates and whether decisions preceded or followed viewing the affected results.

#### Statistics.

No specification-level \(p\)-values are reported because the designed grid is not an independent sample [13, 12]. The nonparametric bootstrap uses \(B=10{,}000\) resamples of specifications, never pairs. This resampling unit matters because \(F\) and \(\bar J\) are pairwise U-statistics: pairs sharing a specification are dependent. The intervals condition on the grid and license no extrapolation. Below, CI abbreviates confidence interval.

#### Two exclusions, criteria fixed before any circuit was seen.

First, every cell of one discovery objective fails inside the verbatim library: mean squared error against an integral target raises a runtime error. A digest of sorted failing identifiers verifies that exactly this objective arm failed and that no other arm failed. No repair was made to the tested instrument.

Second, the pre-registered metric-relative threshold rule excludes specifications admitting no circuit. The grid consequences of these exclusions are reported below.

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

The threshold rule produces uneven exclusions across metrics. This unevenness was disclosed before lock using a six-cell validation slice, and the resulting composition is reported rather than smoothed. In particular, comprehensiveness accounts for 52% of surviving specifications rather than its designed 25%. Table 2 reports the metric-specific kept and discarded counts; “Rate” is the discard rate.

Table 2: Metric-specific retention.

| Metric | Kept | Discarded | Rate |
|---|---:|---:|---:|
| T9V | 3,947 | 13 | 0.003 |
| H2W | 1,467 | 2,493 | 0.630 |
| N6C | 1,270 | 2,690 | 0.679 |
| U4J | 877 | 3,083 | 0.779 |

## 7 Results

The registered quantities are the descriptive premise P0 (J6D), the claim-instability hypothesis H2 (X4R), the random-comparison hypothesis H3 (Q2F), and the functional-gap hypothesis H4 (Z8N). Their decision rules were fixed on 6 August 2026 and tagged before the sweep, and none was revised.

For the functional-gap hypothesis, “agreement” means agreement between circuits’ per-example correct/incorrect verdicts. Functional instability is therefore \(1-\mathrm{agreement}\), and the registered gap is claim instability minus functional instability, \(F-(1-\mathrm{agreement})\). Two outcomes require within-size qualifications, which are given in Sections 7.5 and 7.6 rather than deferred to an appendix.

Table 3: Registered outcomes.

| | Decision rule | Result | Outcome |
|---|---|---|---|
| J6D | \(\bar J\) reported, not tested | 0.1396 [0.1377, 0.1418] | premise holds |
| X4R | \(F>0.20\), CI lower bound above | 0.7316 [0.7247, 0.7380] | confirmed |
| Q2F | not separated from random null | separated, intervals disjoint | rejected |
| Z8N | \(F-(1-\mathrm{agreement})>0.10\) | 0.3344 [0.3297, 0.3394] | supported |

### 7.1 Circuits are almost disjoint

Circuit duplication makes exact, unsampled pair computation tractable: 7,561 claim-producing specifications yield 3,218 distinct circuits and 28,580,580 specification pairs. The distance distribution shows separation beyond what mean Jaccard similarity alone conveys. Median distance is 0.960, the median shared-edge fraction is 4%, and only 1% of pairs exceed Jaccard similarity 0.8. Thus overlap beats chance—by about 45 times—without implying much common structure.

In the analytic random line below, \(k\) denotes circuit size and \(|E|\) denotes the total number of graph edges.

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

Claim disagreement is \(F=0.7316\), with interval [0.7247, 0.7380] (Table 3, X4R row). The lower interval bound, 0.7247, clears the registered threshold of 0.20 by more than three times. None of the registered tolerances permits filing.

The most frequent pair of claim classes attributes the same model/task behaviour to early versus late layers: the leading class is “early layers, distributed”, with 3,107 specifications and share 0.4109, while the second class is “late layers, sparse”, with 2,153 specifications and share 0.2848 (Table 4, ranks 1 and 2). This is contradiction, not emphasis.

Table 4: \(\varphi_{\mathrm{D9P}}\), K8Z, frequency order. Here \(\varphi_{\mathrm{D9P}}\) is the overseer claim map, and K8Z is MEDIUM resolution.

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

The early-layer aggregate, 47.9%, independently reproduces the overseer COARSE modal share of 0.4788 in Table 5. Moreover, even the coarsest affected-person statement rejects filability: its modal share is 0.7291, below the loosest required share of 0.80. Fine-grained language is therefore not required for failure.

Table 5: All addressee-resolution combinations; loosest registered tolerance \(\alpha=0.20\), required \(\pi^*\geq0.80\). L3X is the affected-person mapping, D9P the overseer mapping; E7R, K8Z and V2H are COARSE, MEDIUM and FINE resolution, respectively.

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

Figure 1: F9A for \(\varphi_{\mathrm{D9P}}\), K8Z. This specification curve uses the overseer claim map at MEDIUM resolution. Upper panel: frequency-ordered claim class against specification rank; step widths equal class shares, and the modal width is \(\pi^*\). Dashed line: A6N modal share, whose pooled comparison is size-confounded. Lower panel: active analytic levels per specification; the solid T9V band reflects discard-induced comprehensiveness composition, while the unstructured seed rows reflect the negligible seed contribution.

### 7.3 No analytic axis rescues the filing

The evaluation metric dominates single-axis claim control: standardising the metric removes 0.1377 of disagreement, but leaves residual \(F=0.5939\) (Table 6). Thus even the strongest single-axis control remains insufficient.

Structural and claim variance have different drivers. The ablation-corruption interaction accounts for 78.8% of structural variance in \(\log_{10}\) selected circuit size, while standardising the ablation operator removes only 0.0302 of claim disagreement. Consequently, standardising the operator selected by structural decomposition would fix almost none of the filing instability.

Table 6: Pair-weighted residuals after standardisation. Valid intervals: group sizes 1,512 to 2,520; bootstrap bias below 0.0006 on every axis; every observation inside its interval. “Removes” is the reduction from pooled claim disagreement.

| Axis standardised | Residual \(F\) | 95% CI | Removes |
|---|---:|---|---:|
| Evaluation metric | 0.5939 | 0.5803 to 0.6063 | 0.1377 |
| Ablation operator | 0.7014 | 0.6939 to 0.7075 | 0.0302 |
| Discovery objective | 0.7018 | 0.6938 to 0.7084 | 0.0298 |
| Threshold \(\tau\) | 0.7070 | 0.6981 to 0.7149 | 0.0246 |
| Corruption | 0.7092 | 0.7008 to 0.7166 | 0.0224 |
| Prompt variant | 0.7232 | 0.7157 to 0.7300 | 0.0084 |
| Seed | 0.7307 | 0.7234 to 0.7368 | 0.0009 |

The following structural-variance components use restricted maximum likelihood (REML). The quantity \(\log_{10}\) selected circuit size is the base-10 logarithm of selected circuit size.

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

Size is now entirely absent from statements and conditioned out of comparisons. Even with both controls, dominant-layer attribution still clears the registered instability threshold of 0.20 at the interval’s lower bound: \(F=0.2706\), with a 95% CI of 0.2550 to 0.2859.

| Size-free claim, fixed size | Value |
|---|---|
| \(F\) | 0.2706 |
| 95% CI | 0.2550 to 0.2859 |
| Percentage, abstract precision | 27.1% |
| 95% CI, abstract precision | 0.255 to 0.286 |
| Percentage, discussion precision | 27% |

### 7.5 The random baseline, and why its pooled direction misleads

The size-matched random null (A6N) samples the full edge population using \(R=1{,}000\) null multiverses. Its degeneracy occurs above 1,000 edges: uniform null draws above the degeneracy threshold touch every component and produce only one claim.

In the pooled comparison, discovery is less stable than the random null: discovered \(F=0.7316\), compared with random-null \(F=0.6774\), with disjoint intervals. The rejection of the random-comparison hypothesis Q2F concerns separation, not the direction of this size-confounded comparison.

In contrast, fixed-size comparison reverses the ordering. At MEDIUM resolution, discovered \(F=0.2746\) is below random-null \(F=0.4230\); at COARSE resolution, discovered \(F=0.2706\) is below random-null \(F=0.3822\). Thus discovered circuits outperform randomness at fixed size but still fail filability.

| Comparison | Discovered \(F\) | A6N \(F\) |
|---|---:|---:|
| Pooled | 0.7316 | 0.6774 |
| Fixed size, K8Z | 0.2746 | 0.4230 |
| Fixed size, E7R | 0.2706 | 0.3822 |

Table 7: K8Z within-size results. A6N collapses at and above 1,000 edges; discovery also collapses at 5,000 and 10,000. Thus 2,692 of 7,561 specifications contribute no flips; instability lives entirely below 2,000 edges. K8Z is MEDIUM resolution, \(n\) is the number of specifications at each size, and A6N is the size-matched random null.

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

Functional agreement uses per-example correct/incorrect verdicts and Bayat Makou et al.’s [1] measures reproduced verbatim. Cohen’s kappa measures chance-adjusted agreement. Pooled raw agreement is 0.6028, but pooled kappa is 0.0146. Raw agreement therefore mostly reflects circuits being right on similar fractions of examples, not being right on identical examples. The functional interchangeability required by the phantom-specialisation objection (W5T) is absent.

The functional-gap hypothesis Z8N holds on its registered pooled quantity: \(F-(1-\mathrm{agreement})=0.3344\), with interval [0.3297, 0.3394] (Table 3). However, the hypothesis does not hold under size conditioning. Mechanisms differ more than filings at both ends of the size range; only the intermediate band, 500 to 2,000 edges, carries the pooled gap. Six of 10 sizes fail the gap threshold, and 5 sizes have a negative gap. Registration requires reporting the pooled result, while decomposition prevents that result from receiving a misleading interpretation.

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

For a circuit-derived filed claim \(c\), the filability protocol is:

1. Declare \(\mathcal S\), the space of defensible analytic axes and levels, citing every axis level.
2. Declare \(\varphi\), the circuit-to-statement claim map, as published deterministic code, not a language model.
3. Compute the modal claim share \(\pi^*=\max_c n_c/N\) over \(\mathcal S\), and compute claim disagreement \(F\).
4. Evidence passes Y3W at tolerance \(\alpha\) if and only if \(\pi^*\geq1-\alpha\).
5. Report \(\pi^*\), \(F\), the discard rate per axis level, and the within-group decomposition of every pooled statistic.

A filing unable to state its own modal claim share \(\pi^*\) is asserted, not audited. Implementability requires only a declared specification space and measured agreement; implementation does not require regulatory adjudication of analytic correctness.

## 9 Limitations

#### One model, one task.

Larger models permit more layer-band and size-class variation. However, the direction of the resulting change in \(F\) is unclear, and the one-model limitation is the limitation most likely to change \(F\)’s value.

The committed, timestamped second-model plan uses Pythia-160m (M9B), a different architecture family performing the same task. The plan specifies a 924-cell grid identical on every axis except that the second-model grid has three rather than five seed levels. Replication-versus-refutation criteria were fixed before any cell ran. The second-model plan remains unrun and will appear in a revision regardless of direction; a second task is also designed but unrun.

#### Circuit size is an outcome, not an axis.

Threshold-selected circuit size distorts claim disagreement \(F\), the direction of the null comparison, and the sign of the functional gap. Conditioning on size reduces claim instability more than standardising any registered axis: size-free, fixed-size \(F\) is 0.2706 (Section 7.4), whereas the lowest single-axis residual is 0.5939 after metric standardisation (Table 6). This is a general multiverse hazard when specifications select differently sized objects. Accordingly, every pooled statistic is reported with within-size decomposition.

#### The multiverse is as wide as the library, not the literature.

Optimal ablation (C4Y) [5] is a defensible but unsupported operator that never enters the instrument’s specification space. Its absence makes measured \(F\) a lower bound relative to the field’s full defensible choice set.

#### The discard rate is high and uneven.

The retained metric imbalance is a property of the reported set, not a property adjusted away: comprehensiveness contributes 52% of survivors rather than its designed 25%, and metric-specific discard rates range from 0.003 to 0.779 (Table 2). The threshold was fixed in advance, and no failed item was repaired.

#### The bootstrap is invalid for one reported family.

Conditioning groups containing only two to five specifications retain dominating self-pairs. For that reported family, unusable intervals are therefore replaced by point estimates with diagnostics. No bias-corrected estimator was constructed after naive intervals disagreed, because result-responsive estimator selection is the behaviour under study.

## 10 Discussion

#### What standardising one axis buys.

Complete metric standardisation exceeds what a conformity procedure could demand. Yet metric standardisation leaves 59.4% disagreement (Section 7.3), and no single knob suffices. Declaring the problem resolved after standardising one axis would therefore be substantially mistaken.

#### On the instrument.

The failed objective is documented and publicly exported, yet cannot execute on the library’s own canonical task. The instrument’s discard rate also records tooling maturity beneath prospective regulatory evidence. This tooling observation is neither the headline nor criticism of authors who are themselves among those requesting these improvements [10].

#### What a regulator can take from this.

Interpretability evidence is not worthless. Rather, evidentiary weight is measurable instead of presumed, and measuring that weight is cheap relative to the sweep producing the circuit.

## 11 Reproducibility

The public timestamped commit is tagged `prereg-p1-confirmatory`. Every reported number traces to a configuration, seed and environment fingerprint. Archive verification establishes one environment across all 1,320 confirmatory cells. Content-addressed comparison recovers the code commit from the uploaded archive against the git tree, with 22 of 22 files identical. Analysis needs no graphics processing unit (GPU).

One script recomputes headline quantities from raw results and asserts the reported values: 36 checks, 0 failures. The script computes \(F\) through three routes, including brute-force pairs. Another script checks every manuscript numeric token against stored-result derivations. Tables are emitted directly, never transcribed; the suite contains 343 tests.
