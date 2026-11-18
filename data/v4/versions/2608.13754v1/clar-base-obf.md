# Explanation Multiplicity: Circuit-Level Interpretability

Evidence Does Not Survive Defensible Analytic Variation

Ajay Pravin Mahale

12 August 2026

###### Abstract

Circuit-derived regulatory evidence consists of claims based on selected parts of a model’s computation. This evidence fails Y3W, the paper’s filability rule, throughout tolerances plausibly acceptable to a conformity-assessment body. Y3W requires a sufficiently large share of defensible analyses to produce the same claim. Competent analysts using the same system and tool obtain incompatible statements under different defensible settings (Table 5). Standardising individual settings and controlling circuit size do not remove the failure (Table 6; Section 7.4). Structural and functional disagreement rule out merely reworded mechanisms (Sections 7.1 and 7.6). Applicability beyond one model and task remains untested.

## 1 Introduction

An audit requires two auditors to reach the same conclusion. This is a commissioning requirement, not a machine-learning definition.

The European Union’s (EU) Regulation (EU) 2024/1689 requires providers of high-risk artificial intelligence (AI) systems to file technical documentation. Annex IV 2(b) requests “the general logic of the AI system and of the algorithms”; 2(e), “an assessment of the technical measures needed to facilitate the interpretation of the outputs”. Article 86(1) grants persons subject to certain automated decisions “clear and meaningful explanations of the role of the AI system in the decision-making procedure and the main elements of the decision taken”. These quotations follow the consolidated text amended by Regulation (EU) 2026/1744, legal-document identifier CELEX 02024R1689-20260727, read 11 August 2026.

Mechanistic interpretability seeks to explain behaviour through a model’s internal computations. It supplies prospective technical content for these requirements. Its mature instrument, circuit discovery, returns a subgraph: selected computational components and connections said to explain behaviour. Filing such an account presumes compatibility with another analyst’s account. This paper measures how regulatory claims change across defensible analytic specifications. It evaluates that change with a standards-body decision rule.

#### Contributions.

The pre-registered crossed design combines analytic choices using published implementations exclusively. A deterministic transformation converts circuits into structured Annex IV statements. Controls standardise individual axes of choice, remove claim dependence on circuit size, and compare circuits within size groups. Three pooled statistics invert when selected objects differ in size (Sections 7.4–7.6). No improved discovery method is proposed. The objection concerns the standard presumed when currently produced evidence is filed, not interpretability’s worth.

## 2 Related work

#### Non-identifiability.

Non-identifiability means that the criteria used to assess explanations do not select a unique explanation. Méloux et al. [6] find that a behaviour need not have a unique explanation under mechanistic interpretability’s own criteria. This paper retains that premise. It measures the downstream cost for filing rather than publication.

#### Faithfulness metric sensitivity.

A faithfulness metric measures how well a circuit preserves or accounts for the behaviour it is meant to explain. Ablation tests this by intervening on model computations. Miller et al. [8] find that ablation-based faithfulness is highly sensitive to seemingly insignificant design changes. Scores reflect researcher choices as well as circuits. Their single-axis finding becomes a seven-axis analysis of regulatory consequences here. Their released library supplies the instrument, which is tested under its own definitions.

#### Structure versus function.

Bayat Makou et al. [1] hold the task fixed, vary input statistics, and find structurally distinct circuits implementing the same computation. This pattern, labelled W5T or *phantom specialisation*, supplies the strongest design objection: different claims might describe sampled members of a behaviourally identical equivalence class. Here, an equivalence class means circuits treated as the same because they implement the same behaviour. Functional measurement, rather than argument, addresses that objection.

#### Variance as estimation.

Méloux et al. [7] analyse mechanistic interpretability as statistical estimation, including variation in the estimator’s output. Seed variance is one axis here. Their framing is the statistical companion to analysis across whole specifications.

#### Auditability.

Lan et al. [4] describe two papers that reach conflicting conclusions about one behaviour, and a third that finds both partly correct and incomparable. Their request for safety-critical certification standards receives a measurement and criterion here, not a proposal. Sharkey et al. [10] include relevant socio-technical open problems. Mueller et al. [9] require standardised evaluation for method comparison. That is adjacent to standardised specification for certification.

#### Multiverse method.

A multiverse analysis reports results across multiple defensible analytic choices. Steegen et al. [13] prescribe reporting across defensible data-processing choices. Simonsohn et al.’s [12] visualisation, labelled F9A or *specification curve*, displays outcomes across specifications. It is adapted here to an evidentiary setting. Stability claims in this setting are susceptible to the undisclosed analytic flexibility identified by Simmons et al. [11]. The transfer from psychology methods is substantive: flexibility can manufacture explanation stability.

#### Explanation metrics under optimisation pressure.

Hsia et al. [3] inflate U4J, or *sufficiency*, and T9V, or *comprehensiveness*, without changing predictions or explanations. Sufficiency measures whether the selected explanation is enough to preserve behaviour. Comprehensiveness measures how fully it accounts for that behaviour. Both are among this paper’s four metrics. Reporting discards separately for each metric avoids silently pooling differently retained sets.

## 3 What this paper is not claiming

Before decomposition, we ourselves held two unsupported readings. Discovered circuits carry information and, at fixed size, outperform A6N, the *size-matched random null*: randomly selected circuits with the same number of edges (Section 7.5). Yet the mechanisms themselves differ substantially. They are functionally uncorrelated, rather than one mechanism differently dressed (Sections 7.1 and 7.6).

The seed selects sampled prompts. Seed variation therefore measures evaluation-set variability, not nondeterminism. No claim about larger scale is made.

## 4 Formalism

A specification is a complete set of analytic settings:
\[
s=(o,a,d,m,\tau,P,r,g).
\]

The entries are:

- \(o\): the discovery objective, or quantity the discovery procedure seeks to optimise.
- \(a\): the ablation operator, which specifies how selected computations are intervened on.
- \(d\): the corruption distribution, or distribution of altered inputs used by operators that require them.
- \(m\): the evaluation metric used to score the circuit.
- \(\tau\): the size threshold, applied through the metric-relative selection rule.
- \(P\): the prompt variant.
- \(r\): the seed used to select sampled prompts.
- \(g\): the granularity, or resolution, of the resulting statement.

An axis is one of these choices; its levels are the available settings. Every axis level comes from a published implementation. None was invented here.

Let \(E\) be the graph’s full set of edges, or connections. Discovery under specification \(s\) returns circuit \(C(s)\subseteq E\). G7U, the *claim map*, transforms that circuit into a structured statement. Y3W, the *filability* property, concerns agreement among the resulting claims. It does not adjudicate which specification is correct.

Q7V denotes the model GPT-2 small. Its factorised graph represents the attention mechanism’s query, key and value inputs separately. These are the inputs used to match attention requests to sources and carry their content. For this graph, \(|E|=32{,}491\), where vertical bars denote the number of elements in a set. This count is confirmed both from the instrument and independently by counting in computation order.

For block index \(b\), attention receives \(1+13b\) sources at three input slots for each of 12 attention heads. An attention head is an individual attention component. The MLP in block \(b\) receives those sources plus its block’s heads. MLP means multilayer perceptron, the feed-forward component. The output receives all 157 non-terminal sources. The components comprise 144 attention heads and 12 MLPs, 156 total.

#### Circuit instability.

For specifications \(s_i\) and \(s_j\), write \(C_i=C(s_i)\) and \(C_j=C(s_j)\). Their circuit distance is
\[
D(s_i,s_j)=1-\frac{|C_i\cap C_j|}{|C_i\cup C_j|}. \tag{1}
\]

The intersection \(C_i\cap C_j\) contains shared edges. The union \(C_i\cup C_j\) contains edges in either circuit. Their size ratio is Jaccard similarity. Thus \(D\) measures structural disagreement. \(\bar J\) denotes mean Jaccard similarity over unordered pairs, counting each pair once.

#### Claim instability.

Let \(\varphi(C)\) be the structured Annex IV statement produced from circuit \(C\). A claim class \(c\) contains specifications producing the same statement. Let \(n_c\) be the number in class \(c\), and let \(N\) be the total number of specifications producing claims. Then
\[
F=1-\frac{\sum_c n_c(n_c-1)}{N(N-1)}. \tag{2}
\]

Here \(\sum_c\) means summing over claim classes. \(F\) is the probability that two distinct specifications, drawn without replacement, produce different claims. This is the unbiased Gini–Simpson form. Randomised unit tests compare it with direct enumeration of all pairs.

To condition on axes \(G\), compare only specifications sharing the same settings on those axes. Let \(g\) now index these conditioning groups, \(n_g\) be the group size, and \(n_{g,c}\) be the number in group \(g\) producing claim \(c\):
\[
F_{\mathrm{within}}(G)=
1-\frac{\sum_g\sum_c n_{g,c}(n_{g,c}-1)}
{\sum_g n_g(n_g-1)}. \tag{3}
\]

Pairs, not groups, receive equal weight. When \(G\) is empty, equation (3) reduces exactly to equation (2).

#### Y3W.

The Y3W protocol uses the modal share: the fraction producing the most frequent claim. Denote this share by \(\pi^*\). At tolerance \(\alpha\), the permitted fraction outside that claim, Y3W requires \(\pi^*\geq1-\alpha\). Section 8 gives the protocol.

For \(k\) available claim classes, the ceiling \(F\leq1-1/k\) depends on the number of classes. The observed rate is not near one by construction.

| Ceiling diagnostic | Value |
|---|---:|
| Observed classes | 9 |
| Ceiling | 0.8889 |
| Observed fraction of ceiling | 82.3% |

## 5 G7U

Size bins are categories of circuit size. They follow a committed calibration rule applied to a measured curve of node counts. They were not selected to obtain the reported \(F\), as reviewers suspected. Four alternatives exceed the committed result, including halving the thresholds (Table 1). Binnings spanning a factor of four remain within the reported range.

In Table 1, E7R denotes COARSE, the coarsest statement resolution. Its version used here omits the size term.

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

The intended regulatory reader determines which of two mappings is used:

- D9P, or *overseer*, is grounded in Annex IV 2(e), 3 and Article 14(4)(c). It orders the statement by dominant layer band, size class, and ranked input segment. A layer band groups model layers by position. An input segment is a part of the prompt.
- L3X, or *affected*, is grounded in Article 86(1). It leads with the input segment because the right belongs to the decision’s subject.

The resolutions are nested: E7R is COARSE, K8Z is the intermediate MEDIUM resolution, and V2H is the finest FINE resolution. They are constructed so that a refinement cannot contradict the coarser statement.

The claim map \(\varphi\) is the paper’s most attackable choice. It is nevertheless pre-registered, published deterministic code. It is not language-model output, which would inseparably add model variance to circuit variance.

## 6 Method

#### Model and task.

Comparability with the extended non-identifiability and faithfulness literature motivates Q7V, GPT-2 small, and Z4K, *indirect object identification* [15]. The task concerns identifying the indirect object in a prompt. The edge count makes size-matched randomness meaningful. Model scale is not offered as an apology.

#### Grid.

The design includes seven objectives, seven operators, four corruption distributions nested within the five operators that read them, four metrics, three thresholds, two prompt variants, and five seeds. It yields 1,540 discovery cells and 18,480 pre-registered specifications. A discovery cell fixes the discovery choices; the metrics and thresholds then produce the specifications evaluated from it. Two operators never read corruption. Crossing those operators with corruption distributions would duplicate specifications.

Discovery uses R8X, the *edge attribution patching* family [14], and B3Q, *integrated edge gradients*, as implemented by Miller et al. [8]. These methods score graph connections for circuit discovery; integrated edge gradients uses integrated gradient information. Automated discovery follows the lineage in Conmy et al. [2]. Corruption levels come from Wang et al. [15].

The four metrics are:

- T9V: comprehensiveness.
- H2W: KL divergence, meaning Kullback–Leibler divergence, a measure of difference between distributions.
- N6C: logit difference, the difference between model scores before conversion to probabilities.
- U4J: sufficiency.

Unequal retention across these metrics materially changes the realised pool (Table 2).

#### Pre-registration.

The plan was committed and tagged before the confirmatory sweep, the full run used to test the registered claims. Both directional abstracts were drafted in advance. This prevented either direction from being presented as a surprise. An append-only deviations file records dates and whether decisions preceded or followed viewing the affected results.

#### Statistics.

There are no specification-level \(p\)-values, or significance-test probabilities. The designed grid is not an independent sample [13, 12].

The nonparametric bootstrap resamples specifications with replacement, never pairs, using \(B=10{,}000\) resamples. Both \(F\) and \(\bar J\) are pairwise U-statistics: statistics formed from pairs of specifications. Pairs sharing a specification are dependent. Confidence intervals (CIs) condition on the grid and permit no extrapolation beyond it.

#### Two exclusions, criteria fixed before any circuit was seen.

The first exclusion affects an entire objective. Every cell of that objective fails inside the unchanged library. Mean squared error—the average squared difference from a target—raises a runtime error when used against an integral, or integer-valued, target. A digest, or identifying summary, of sorted failing identifiers verifies exactly that arm and nothing else. No repair was made to the tested instrument.

The second exclusion removes specifications that admit no circuit under the pre-registered metric-relative threshold rule.

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

The threshold rule sets acceptance relative to each metric and produces the exclusions. Uneven retention was disclosed before lock from a six-cell validation slice. The resulting composition is reported rather than smoothed. In Table 2, “Rate” is the discard rate.

Table 2: Metric-specific retention.

| Metric | Kept | Discarded | Rate |
|---|---:|---:|---:|
| T9V | 3,947 | 13 | 0.003 |
| H2W | 1,467 | 2,493 | 0.630 |
| N6C | 1,270 | 2,690 | 0.679 |
| U4J | 877 | 3,083 | 0.779 |

## 7 Results

The registered labels are J6D, or P0, for the descriptive premise; X4R, or H2, for claim instability; Q2F, or H3, for comparison with randomness; and Z8N, or H4, for the functional gap. Their decision rules were fixed on 6 August 2026 and tagged before the sweep. They were not revised. Two outcomes require within-size qualifications here, rather than deferral to an appendix.

For Z8N, *agreement* means the fraction of examples on which two circuits give the same correct/incorrect verdict. Functional instability is \(1-\mathrm{agreement}\). The functional gap is claim instability \(F\) minus functional instability. Bracketed intervals in Table 3 are confidence intervals.

Table 3: Registered outcomes.

| | Decision rule | Result | Outcome |
|---|---|---|---|
| J6D | \(\bar J\) reported, not tested | 0.1396 [0.1377, 0.1418] | premise holds |
| X4R | \(F>0.20\), CI lower bound above | 0.7316 [0.7247, 0.7380] | confirmed |
| Q2F | not separated from random null | separated, intervals disjoint | rejected |
| Z8N | \(F-(1-\mathrm{agreement})>0.10\) | 0.3344 [0.3297, 0.3394] | supported |

### 7.1 Circuits are almost disjoint

Repeated circuits make exact computation over all pairs tractable without sampling pairs. The distance distribution shows separation beyond what the mean alone conveys. Overlap exceeds chance, but this does not imply much common structure: median distance is 0.960 and median shared-edge fraction is 4% in the diagnostics below.

In the analytic random line, \(k\) now denotes circuit size in edges, rather than the number of claim classes used in Section 4.

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

The lower interval bound clears the registered threshold by more than three times: Table 3, X4R, reports a lower bound of 0.7247 against 0.20. None of the registered tolerances permits filing (Table 5).

The two most frequent claims attribute the same model/task behaviour to early versus late layers. This is contradiction, not emphasis (Table 4, ranks 1 and 2). Here \(\varphi_{\mathrm{D9P}}\) means the overseer version of the claim map.

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

The early-layer aggregate independently reproduces the E7R modal share (Table 5, D9P–E7R). Even the coarsest L3X statement rejects Y3W (Table 5, L3X–E7R). Fine-grained language is therefore not required.

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

Standardising an axis means comparing specifications only within groups that share its setting, using equation (3). Evaluation metric is the strongest single-axis control on claim instability, but remains insufficient: residual \(F=0.5939\) (Table 6, Evaluation metric).

Structural and claim variance have different drivers. The structural decomposition is dominated by the ablation-corruption interaction, at 78.8%. Yet standardising the operator removes only 0.0302 of filing instability (Table 6, Ablation operator). Selecting a standardisation target from the structural decomposition would therefore fix almost none of the filing instability.

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

The structural variance decomposition uses restricted maximum likelihood (REML), a method for estimating variance components. Its outcome is \(\log_{10}\) selected circuit size: the base-10 logarithm of that size. The interaction component represents variation associated with combinations of ablation and corruption settings. Residual variance is the variation left over.

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

Remove size entirely from statements and compare only circuits of the same size. Dominant-layer attribution still clears the registered instability threshold at the interval’s lower bound: 0.2550 below, against 0.20 in Table 3, X4R.

| Size-free claim, fixed size | Value |
|---|---|
| \(F\) | 0.2706 |
| 95% CI | 0.2550 to 0.2859 |
| Percentage, abstract precision | 27.1% |
| 95% CI, abstract precision | 0.255 to 0.286 |
| Percentage, discussion precision | 27% |

### 7.5 The random baseline, and why its pooled direction misleads

In the pooled comparison, discovery is less stable than A6N: \(F=0.7316\) versus 0.6774 below, with disjoint intervals. Rejection of Q2F concerns separation from the null, not the direction of this size-confounded comparison.

Uniform null draws above the degeneracy threshold touch every component and produce only one claim. Here degeneracy means that the null ceases to vary in its claims. Comparing at fixed size reverses the ordering: discovered circuits outperform randomness but still fail Y3W.

| Comparison | Discovered \(F\) | A6N \(F\) |
|---|---:|---:|
| Pooled | 0.7316 | 0.6774 |
| Fixed size, K8Z | 0.2746 | 0.4230 |
| Fixed size, E7R | 0.2706 | 0.3822 |

A6N samples uniformly from the full edge population. It uses \(R=1{,}000\) null multiverses, where \(R\) is the number of repeated random multiverse draws. Its degeneracy occurs above 1,000 edges. In Table 7, \(n\) is the number of specifications at the listed size.

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

Agreement uses per-example correct/incorrect verdicts and Bayat Makou et al.’s [1] measures, reproduced unchanged. Raw agreement mostly reflects circuits being correct on similar fractions of examples, not on identical examples. Cohen’s kappa measures agreement after accounting for agreement expected from the verdict frequencies. Its pooled value is 0.0146 below. The functional interchangeability required by the W5T objection is absent.

Z8N holds for its registered pooled quantity (Table 3), but not under size conditioning. Mechanisms differ more than filings at both ends of the size range. Only the intermediate band carries the pooled gap (functional diagnostics below). Registration requires reporting the pooled result. Decomposition prevents a misleading interpretation of it.

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

For a circuit-derived filed claim \(c\):

1. Declare \(\mathcal S\), the space of defensible analytic axes and levels. Cite each level.
2. Declare \(\varphi\), the circuit-to-claim map. It must be published deterministic code, not a language model.
3. Compute \(\pi^*=\max_c n_c/N\) over \(\mathcal S\), and compute \(F\). Here \(\max_c\) selects the largest claim-class share.
4. Evidence passes Y3W at tolerance \(\alpha\) if and only if \(\pi^*\geq1-\alpha\).
5. Report \(\pi^*\), \(F\), the discard rate for each axis level, and a within-group decomposition of every pooled statistic.

A filing unable to state its own \(\pi^*\) is asserted, not audited. Implementation requires only a declared space and measured agreement. It does not require a regulator to adjudicate analytic correctness.

## 9 Limitations

#### One model, one task.

Larger models permit more variation in layer bands and size classes. The direction of the effect on \(F\) is unclear. This limitation is the most likely to change its value.

The committed, timestamped second-model plan uses M9B, the model Pythia-160m. It specifies a different architecture family performing the same task. Its 924-cell grid is identical on every axis except that it has three rather than five seed levels. Criteria for replication versus refutation were fixed before any cell ran. The plan remains unrun and will appear in a revision regardless of direction. A second task is designed but unrun.

#### Circuit size is an outcome, not an axis.

The threshold selects circuit size; size is not itself a registered analytic axis. Threshold-selected size distorts \(F\), the direction of the null comparison, and the sign of the functional gap (Sections 7.4–7.6).

Conditioning on size reduces claim instability more than standardising any registered axis (Section 7.4; Table 6). This is a general multiverse hazard when specifications select differently sized objects. Every pooled statistic is reported with within-size decomposition.

#### The multiverse is as wide as the library, not the literature.

C4Y, or *optimal ablation*, is a defensible ablation operator [5]. The instrument does not support it, so it never enters the specification space. This makes measured \(F\) a lower bound relative to the field’s full defensible choice set.

#### The discard rate is high and uneven.

The retained metric imbalance is a property of the reported set, not something adjusted away (Table 2). The threshold was fixed in advance, and no failed item was repaired.

#### The bootstrap is invalid for one reported family.

Conditioning groups containing only two to five specifications retain dominating self-pairs: pairs formed from bootstrap copies of the same original specification. Unusable intervals are replaced by point estimates with diagnostics. No bias-corrected estimator was constructed after naive intervals disagreed. Selecting an estimator in response to the result is the behaviour under study.

## 10 Discussion

#### What standardising one axis buys.

Complete metric standardisation exceeds what a conformity procedure could demand. Yet no single choice suffices: even metric standardisation leaves \(F=0.5939\) (Table 6, Evaluation metric). Declaring the problem resolved after standardising one axis would be substantially mistaken.

#### On the instrument.

The failed objective is documented and publicly exported, yet cannot execute on the library’s own canonical task (Section 6, Two exclusions). The instrument’s discard rate also records the maturity of tooling beneath prospective regulatory evidence (Section 6, grid-consequence table; Table 2). It is neither the headline nor criticism of authors who are themselves among those requesting these improvements [10].

#### What a regulator can take from this.

Interpretability evidence is not worthless. Its evidentiary weight is measurable rather than presumed. Measuring that weight is cheap relative to the sweep that produces the circuit.

## 11 Reproducibility

The public timestamped commit, a recorded version of the code, is tagged `prereg-p1-confirmatory`. Every reported number traces to a configuration, seed and environment fingerprint, which identifies the execution environment.

Archive verification establishes one environment across all 1,320 confirmatory cells. Content-addressed comparison identifies the code commit by comparing the uploaded archive against the git version-control tree: 22 of 22 files are identical. Analysis needs no graphics processing unit (GPU).

One script recomputes the headline quantities from raw results and checks them against the reported values: 36 checks, 0 failures. It computes \(F\) through three routes, including direct enumeration of pairs. Another script checks every manuscript numeric token against derivations from stored results. Tables are emitted directly, never transcribed. The suite contains 343 tests.
