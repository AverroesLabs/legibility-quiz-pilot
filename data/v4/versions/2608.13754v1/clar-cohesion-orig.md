# Explanation Multiplicity: Circuit-Level Interpretability

Evidence Does Not Survive Defensible Analytic Variation

Ajay Pravin Mahale

12 August 2026

###### Abstract

The European Union Artificial Intelligence Act (EU AI Act) requires providers of high-risk systems to file technical documentation describing how the system reaches its decisions. Mechanistic interpretability is the obvious source of such evidence, and circuit discovery—producing a subgraph said to explain a behaviour—is its most developed instrument. We ask whether circuit-level evidence survives the condition under which the evidence would be relied upon: two competent analysts examine the same system with the same tool but use different defensible settings.

We pre-registered a crossed grid of seven analytic axes, with every level taken from a published implementation. Each specification is a combination of analytic choices. We mapped each discovered circuit through a deterministic claim map, a fixed rule that converts a circuit into a structured Annex IV statement. Across 15,840 pre-registered specifications on GPT-2 small and the indirect object identification task, 7,561 specifications produced a claim. Among those claim-producing specifications, the derived statement differs across 73.2% of specification pairs (95% confidence interval, or CI, 0.725 to 0.738), and the most frequent, or modal, claim commands 41.1% of the space. The evidence therefore fails a filability criterion—a requirement that a sufficiently large share of specifications yield the same claim—at every tolerance a conformity assessment body would plausibly accept.

Standardising the single most influential choice, the evaluation metric, leaves the claim flip rate at 59.4%. Removing circuit size from the claim entirely and holding circuit size fixed leaves a flip rate of 27.1% (95% CI 0.255 to 0.286), still above the pre-registered threshold. The circuits underlying these claims are also structurally near-disjoint: median pairwise Jaccard overlap, the share of edges common to two circuits relative to their combined edge set, is 4%. The circuits are functionally uncorrelated at Cohen’s kappa, a chance-corrected agreement measure, of 0.015. Thus, the instability is not one mechanism described in different words.

We give the filability criterion as a standalone protocol. We also report that one of the seven documented discovery objectives does not execute at all on the library’s own canonical task. The study covers one model and one task; whether the conclusion holds at scale is untested.

## 1 Introduction

An audit means two auditors reach the same conclusion. That is not a definition borrowed from machine learning; agreement between auditors is what makes an audit worth commissioning.

Regulation (EU) 2024/1689 requires providers of high-risk artificial intelligence (AI) systems to file technical documentation. Annex IV point 2(b) asks for “the general logic of the AI system and of the algorithms”. Point 2(e) asks for “an assessment of the technical measures needed to facilitate the interpretation of the outputs”. Article 86(1) gives a person subject to certain automated decisions the right to “clear and meaningful explanations of the role of the AI system in the decision-making procedure and the main elements of the decision taken”. These quotations are from the consolidated text as amended by Regulation (EU) 2026/1744, CELEX 02024R1689-20260727, read 11 August 2026.

Mechanistic interpretability offers to supply the technical content required by these provisions. Circuit discovery produces a subgraph said to explain a behaviour, and circuit discovery is the part of the field with mature tooling. However, if a provider files a circuit-level account of why its system behaved as it did, the filing is worth something only if a second analyst, given the same system and the same tool, would file something compatible.

We therefore measure how far a regulatory claim moves across the space of defensible analytic specifications, and we attach a decision rule that a standards body could use. We do not propose a better circuit discovery method, and we do not argue that interpretability is worthless. Instead, we argue that the evidence, as currently produced, does not meet the standard that filing the evidence presumes.

#### Contributions.

(i) A pre-registered multiverse—a study across combinations of defensible analytic choices—over seven analytic axes of circuit discovery, with every level drawn from a published implementation, and a deterministic map from circuits to Annex IV statements.

(ii) A filability criterion, $\pi^{*}\geq 1-\alpha$, that a conformity procedure can apply, together with the measured value on a canonical task. Here, $\pi^{*}$ is the share of specifications yielding the modal claim, and $\alpha$ is the permitted tolerance for specifications yielding other claims.

(iii) Evidence that the instability survives the most aggressive controls available: standardising any single axis, removing circuit size from the claim, and holding circuit size fixed.

(iv) A methodological result for multiverse designs generally: when specifications select objects of different sizes, pooled statistics can invert. We show three of our pooled statistics that do.

## 2 Related work

#### Non-identifiability.

Méloux et al. [6] ask whether, for a given behaviour and under mechanistic interpretability’s own criteria, a unique explanation exists. Their answer is that a unique explanation need not exist. Our premise is theirs. Our contribution is downstream: we measure what this non-identifiability costs when an explanation is filed rather than published.

#### Faithfulness metric sensitivity.

Miller et al. [8] survey design choices in ablation-based faithfulness measurement, which evaluates a circuit by removing or replacing parts of the computation. They find existing methods highly sensitive to seemingly insignificant changes, and conclude that faithfulness scores reflect the researcher’s methodological choices as well as the circuit. Thus, Miller et al. establish that one axis matters. We cross seven axes and propagate the resulting variation to a regulatory statement. Their released library is the instrument we use, so our study is a stress test of the tooling by the tooling’s own definitions.

#### Structure versus function.

Bayat Makou et al. [1] vary input statistics while holding the task fixed. They find that structurally distinct circuits implement the same computation, a pattern they term phantom specialisation. This finding supplies the strongest objection to our design: if discovery merely samples from an equivalence class of behaviourally identical subgraphs, a flip rate over claims measures which member of that class was sampled. We address this objection with a functional measurement rather than an argument in Section 7.6. The functional measurement shows that our circuits are not functionally interchangeable.

#### Variance as estimation.

Méloux et al. [7] treat mechanistic interpretability as statistical estimation and analyse the variance of the estimator. Seed variance is one axis of our specification space. Their framing is therefore the natural statistical companion to what we do at the level of the whole specification.

#### Auditability.

Lan et al. [4] observe that two papers reached conflicting conclusions about the same behaviour, with a third paper finding both partly correct and incomparable. They consequently call for auditing standards so that findings can be certified in safety-critical settings. We answer that call with a measurement and a criterion rather than a proposal. Sharkey et al. [10] list open problems in the field, including the socio-technical problems this paper touches. Mueller et al. [9] argue that comparison across methods requires standardised evaluation; their argument is adjacent to our argument that certification requires standardised specification.

#### Multiverse method.

The design follows Steegen et al. [13], who propose reporting results across the space of defensible data-processing choices, and Simonsohn et al. [12], whose specification curve is the visualisation we adapt. Simmons et al. [11] named the underlying problem: undisclosed flexibility in analysis lets an analyst present almost anything as significant. We are applying a psychology methods literature to a machine learning evidentiary question, and the transfer is not decorative. The claim that an explanation is stable is exactly the kind of claim that analytic flexibility can manufacture.

#### Explanation metrics under optimisation pressure.

Hsia et al. [3] show that sufficiency and comprehensiveness, two explanation-evaluation metrics, can be inflated without changing predictions or explanations. Two of our four evaluation metrics are sufficiency and comprehensiveness. This susceptibility is one reason we report the discard rate per metric rather than pooling silently.

## 3 What this paper is not claiming

We state these qualifications early because each addresses a reading the results do not support. Two of these readings were readings we held ourselves before the decompositions were run.

We are not claiming that discovered circuits are no better than random. At fixed circuit size, discovered circuits are more stable than a size-matched random null: the claim flip rate is 0.2746 for discovered circuits against 0.4230 for the random null. Thus, the discovered circuits carry information.

Nor are we claiming that filings differ where mechanisms do not. Mechanisms differ a great deal: functional instability is 0.3972 pooled and 0.4777 at the smallest circuits. Here, functional instability means disagreement between circuits’ per-example correct or incorrect verdicts.

Nor are we claiming that discovery samples one mechanism and dresses that mechanism differently. Cohen’s kappa of 0.0146 says the circuits are functionally uncorrelated.

Nor are we claiming that random seeds are the problem. The seed here selects which prompts are sampled, so seed variation is evaluation-set variability rather than nondeterminism.

And we claim nothing at all about scale. The study covers one model and one task.

## 4 Formalism

A specification is a tuple of analytic choices,

$$
s=(o,a,d,m,\tau,P,r,g),
$$

where $o$ is the discovery objective, $a$ is the ablation operator, $d$ is the corruption distribution, $m$ is the evaluation metric, $\tau$ is the size threshold, $P$ is the prompt variant, $r$ is the seed, and $g$ is the granularity of the claim. Every level of every axis is taken from a published implementation. No level was invented for this study.

Circuit discovery returns $C(s)\subseteq E$, where $C(s)$ is the circuit found under specification $s$ and $E$ is the model’s edge set. For GPT-2 small under the factorised graph with separate query, key and value inputs, the number of edges is $|E|=32{,}491$. We confirm this count from the instrument and independently from a closed-form count over the computation order. For block index $b$, block $b$ attention receives from $1+13b$ sources at three input slots for each of 12 heads. The multilayer perceptron (MLP) in block $b$ receives from those sources plus its own block’s heads, and the output receives from all 157 non-terminal sources. Circuits touch components drawn from 144 attention heads and 12 MLPs, so 156 components in total.

#### Circuit instability.

For two specifications $s_i$ and $s_j$, let $C_i=C(s_i)$ and $C_j=C(s_j)$. Circuit instability is their Jaccard distance:

$$
D(s_i,s_j)
=
1-\frac{|C_i\cap C_j|}{|C_i\cup C_j|}.
\tag{1}
$$

The intersection $C_i\cap C_j$ contains the edges shared by both circuits, and the union $C_i\cup C_j$ contains the edges present in either circuit. Their ratio is Jaccard similarity. We write $\bar{J}$ for the mean Jaccard similarity over unordered pairs of specifications.

#### Claim instability.

Let $\varphi$ be the claim map that converts a circuit to a structured Annex IV statement. The flip rate $F$ is the probability that two distinct specifications yield different statements:

$$
F
=
1-\frac{\sum_c n_c(n_c-1)}{N(N-1)}.
\tag{2}
$$

Here, $c$ indexes claim classes, $n_c$ counts specifications yielding claim class $c$, and $N$ is the number of specifications. Equation 2 is the unbiased Gini–Simpson form: the probability that two draws without replacement differ. The implementation is unit-tested against the brute-force pairwise definition over randomised cases. The modal share is

$$
\pi^{*}=\max_c n_c/N.
$$

We also use a within-group flip rate. Let $G$ be a set of conditioning axes, and let $g$ in the following equation index the groups formed by holding those axes fixed, rather than denote the claim-granularity entry in the specification tuple. Let $n_{g,c}$ count specifications in group $g$ yielding claim class $c$, and let $n_g$ be the number of specifications in that group. Then

$$
F_{\text{within}}(G)
=
1-
\frac{\sum_g\sum_c n_{g,c}(n_{g,c}-1)}
{\sum_g n_g(n_g-1)}.
\tag{3}
$$

Equation 3 weights every pair equally rather than every group equally. When the conditioning set $G$ is empty, Equation 3 reduces exactly to Equation 2.

#### Filability.

Evidence is filable at tolerance $\alpha$ if and only if

$$
\pi^{*}\geq 1-\alpha.
$$

The flip rate also has a ceiling that depends on the number of claim classes. For $k$ claim classes, $F\leq 1-1/k$, so a reported flip rate must be read against the number of classes the claim map can produce. With the nine classes observed here, the ceiling is 0.8889, and the observed flip rate of 0.7316 is 82.3% of that ceiling. Thus, $F$ is not near one by construction.

## 5 The claim map

The claim map $\varphi$ is the most attackable choice in the paper. For that reason, the claim map is fixed in the pre-registration, implemented as deterministic code, and published. The claim map is not generated by a language model: using a language model would measure model variance on top of circuit variance, with no way to separate the two sources of variance.

There are two addressees, following the regulation rather than convenience. The overseer claim map, $\varphi_{\text{overseer}}$, is built on Annex IV 2(e) and 3 with Article 14(4)(c). The overseer claim is keyed first on the dominant layer band, then on the size class, and then on the ranked input segment. The affected-person claim map, $\varphi_{\text{affected}}$, is built on Article 86(1). The affected-person claim leads with the input segment because the right is owed to the person the decision is about. Three granularities—COARSE, MEDIUM and FINE—are nested by construction, so a claim at a finer granularity refines rather than contradicts the coarser claim.

A reviewer will say the size bins were chosen to produce the reported flip rate. However, the size bins were returned by a committed calibration rule from a measured node-count curve, not chosen. Table 1 reports sensitivity to alternative size binnings.

Table 1: Flip rate under alternative size binnings. The committed bins are not a maximum: four alternatives give a higher $F$, including halving every threshold. Across binnings differing by a factor of four, the range is 0.70 to 0.77. Removing the size term from the claim entirely still leaves a flip rate of 0.5733.

| Binning | $F$ |
|---|---:|
| committed bins | 0.7316 |
| all thresholds $\times$ 0.5 | 0.7448 |
| all thresholds $\times$ 0.8 | 0.7435 |
| all thresholds $\times$ 0.9 | 0.7394 |
| all thresholds $\times$ 1.1 | 0.7289 |
| all thresholds $\times$ 1.25 | 0.7314 |
| all thresholds $\times$ 1.5 | 0.7308 |
| all thresholds $\times$ 2 | 0.7232 |
| equal thirds | 0.7018 |
| deciles | 0.7719 |
| no size term (COARSE) | 0.5733 |

## 6 Method

#### Model and task.

We use GPT-2 small on indirect object identification [15]. The choice is not an apology. The non-identifiability and faithfulness literature this paper extends works at this scale, so comparability is the point. In addition, the edge count is what makes a size-matched random baseline meaningful.

#### Grid.

The pre-registered grid contains seven discovery objectives, seven ablation operators, four corruption distributions nested within the five operators that read those distributions, four metrics, three thresholds, two prompt variants, and five seeds. The resulting pre-registered design contains 1,540 discovery cells and 18,480 specifications.

Corruption is nested rather than crossed because two ablation operators never read the corrupt distribution. Crossing those operators with corruption distributions would therefore emit duplicate specifications. Discovery uses the edge attribution patching family [14] and integrated edge gradients, as implemented in the library released with Miller et al. [8]. The automated circuit discovery lineage is Conmy et al. [2]. The four corruption levels are taken from Wang et al. [15].

#### Pre-registration.

The analysis plan was committed and tagged before the confirmatory sweep began. Both abstracts were also drafted in advance so that neither direction could be written up as a surprise. Deviations are recorded in an append-only file, with the date of each deviation and whether the deviation was decided before or after the affected result was seen.

#### Statistics.

No $p$-value is computed across specifications anywhere, because specifications are a designed grid rather than an independent sample [13, 12]. Uncertainty instead comes from a nonparametric bootstrap that resamples specifications, never pairs, with $B=10{,}000$ bootstrap resamples. We resample specifications because $F$ and $\bar{J}$ are U-statistics over pairs, and pairs sharing a specification are dependent. The intervals quantify uncertainty given the grid and license nothing beyond that grid.

#### Two exclusions, criteria fixed before any circuit was seen.

One discovery objective does not execute. All 220 of that objective’s cells raise a runtime error from a mean squared error computed against an integral target, inside the library’s own code. We verified, by a digest over the sorted failing cell identifiers, that the failure set is exactly that objective’s arm and contains nothing else. We did not repair the failure because the library is the instrument under test and is reproduced verbatim. After excluding this arm, the realised grid contains six objectives, 1,320 cells and 15,840 specifications.

The metric-relative threshold rule admits no circuit for 8,279 of the 15,840 specifications. This is a discard rate of 52.3%, distributed unevenly across metrics, as Table 2 shows. The unevenness was disclosed before the pre-registration lock. Consequently, the surviving pool is 52% comprehensiveness against 25% by design. We report this composition rather than smoothing the composition.

Table 2: Discard rate by evaluation metric under the pre-registered metric-relative threshold rule. The unevenness was disclosed before the pre-registration was locked, from a six-cell validation slice. KL divergence denotes Kullback–Leibler divergence.

| Metric | Kept | Discarded | Rate |
|---|---:|---:|---:|
| comprehensiveness | 3,947 | 13 | 0.003 |
| KL divergence | 1,467 | 2,493 | 0.630 |
| logit difference | 1,270 | 2,690 | 0.679 |
| sufficiency | 877 | 3,083 | 0.779 |

## 7 Results

Table 3 states each pre-registered outcome against its result. The labels P0, H2, H3 and H4 identify the pre-registered premise or hypotheses through the decision rules shown in the table. Those decision rules and thresholds were fixed and tagged before the confirmatory sweep ran, and none was revisited.

Table 3: Pre-registered outcomes and their results. Thresholds were fixed on 6 August 2026, and the plan was tagged before the sweep began. Two results carry qualifications that only the within-size decomposition reveals; those qualifications are reported in Sections 7.5 and 7.6 rather than deferred to an appendix. In the H4 rule, “agreement” means per-example functional agreement between circuits.

| | Decision rule | Result | Outcome |
|---|---|---|---|
| P0 | $\bar{J}$ reported, not tested | 0.1396 [0.1377, 0.1418] | premise holds |
| H2 | $F>0.20$, CI lower bound above | 0.7316 [0.7247, 0.7380] | confirmed |
| H3 | not separated from random null | separated, intervals disjoint | rejected |
| H4 | $F-(1-\text{agreement})>0.10$ | 0.3344 [0.3297, 0.3394] | supported |

### 7.1 Circuits are almost disjoint

Mean pairwise Jaccard similarity is $\bar{J}=0.1396$ (95% CI 0.1377 to 0.1418), computed exactly over all 28,580,580 specification pairs rather than sampled. The 7,561 specifications hold 3,218 distinct circuits, which makes the exact computation tractable.

The distribution of pairwise circuit instability says more than the mean similarity. For pairwise Jaccard distance $D$, the 1% quantile is 0.196, the 25% quantile is 0.788, the median is 0.960, and the 75% quantile is 0.998. Thus, the median pair of circuits shares 4% of its edges, and only 1% of pairs exceed a Jaccard similarity of 0.8.

For a circuit with $k$ edges, the analytic random line is $k/(2|E|-k)$. At the median circuit size of 200 edges, this line gives 0.0031, so the observed mean overlap is about 45 times chance. However, beating chance at this scale is not evidence of much: a 500-edge circuit has an expected random overlap of 0.008.

### 7.2 The claim flips on three quarters of pairs

The claim flip rate is $F=0.7316$ (95% CI 0.7247 to 0.7380), against a pre-registered threshold of 0.20 with the interval lower bound required to clear that threshold. The flip rate clears the threshold by more than three times.

The modal share is $\pi^{*}=0.4109$. Therefore, the evidence is not filable at any pre-registered tolerance: the modal share is 0.411 against required shares of 0.95, 0.90 and 0.80.

Table 4: The nine claim classes produced by $\varphi_{\text{overseer}}$ at medium granularity, ordered by frequency. Aggregating over size class, the layer attribution splits 47.9% early, 43.6% late and 8.5% middle. The first of these is the COARSE modal share of 0.4788, computed independently.

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

The two most common claims attribute the same model’s behaviour on the same task to early layers and to late layers. These claims occupy 41.1% and 28.5% of the space, respectively, as shown by the first two rows of Table 4. The claims are not different emphases. They contradict.

All six addressee-granularity combinations reject filability, as Table 5 shows. In particular, the coarsest statement available to an affected person is a two-class claim, yet that claim still flips on 39.5% of pairs. The result is therefore not an artefact of fine-grained claim language. Figure 1 shows the specification curve; the width of the modal step is the modal share $\pi^{*}$.

Table 5: Flip rate and modal share for both addressees at all three granularities. Filability is evaluated at the loosest pre-registered tolerance, $\alpha=0.20$, which requires $\pi^{*}\geq 0.80$. No cell passes.

| Addressee | Granularity | Classes | $F$ | $\pi^{*}$ | Filable |
|---|---|---:|---:|---:|---|
| affected | COARSE | 2 | 0.3950 | 0.7291 | no |
| affected | MEDIUM | 4 | 0.6597 | 0.4875 | no |
| affected | FINE | 8 | 0.6744 | 0.4786 | no |
| overseer | COARSE | 3 | 0.5733 | 0.4788 | no |
| overseer | MEDIUM | 9 | 0.7316 | 0.4109 | no |
| overseer | FINE | 12 | 0.7681 | 0.4109 | no |

Figure 1: Specification curve for $\varphi_{\text{overseer}}$ at medium granularity. Upper panel: claim class against specification rank, with classes ordered by frequency, so each step’s width is that class’s share of the space. The dashed line marks the size-matched random null’s modal share; the pooled comparison is confounded by circuit size, as discussed in Section 7.5. Lower panel: which analytic level is active for each specification. The solid band on the comprehensiveness row is the composition effect of the discard rule in Table 2. In contrast, the seed rows are unstructured, which is the same result as seed removing 0.0009 from $F$ in Table 6.

### 7.3 No analytic axis rescues the filing

Table 6: Residual flip rate once each axis is standardised, computed with Equation 3. “Removes” is the reduction from the pooled flip rate of 0.7316. All intervals are valid: group sizes are 1,512 to 2,520, bootstrap bias is below 0.0006 on every axis, and every observed value lies inside its own interval.

| Axis standardised | Residual $F$ | 95% CI | Removes |
|---|---:|---|---:|
| Evaluation metric | 0.5939 | 0.5803 to 0.6063 | 0.1377 |
| Ablation operator | 0.7014 | 0.6939 to 0.7075 | 0.0302 |
| Discovery objective | 0.7018 | 0.6938 to 0.7084 | 0.0298 |
| Threshold $\tau$ | 0.7070 | 0.6981 to 0.7149 | 0.0246 |
| Corruption | 0.7092 | 0.7008 to 0.7166 | 0.0224 |
| Prompt variant | 0.7232 | 0.7157 to 0.7300 | 0.0084 |
| Seed | 0.7307 | 0.7234 to 0.7368 | 0.0009 |

The evaluation metric is the largest single lever: standardising the metric removes 0.1377 from the pooled flip rate, more than any other row in Table 6. However, standardising the metric is not enough. Standardising the metric completely, which is more than a conformity procedure could demand, leaves $F=0.5939$ with an interval lower bound of 0.5803, still far above the pre-registered threshold.

Structure and claim are governed by different axes. A restricted maximum likelihood (REML) variance decomposition on $\log_{10}$, the base-10 logarithm, of selected circuit size attributes 78.8% of structural variance to the ablation-corruption interaction, 11.5% to residual, 4.4% to ablation, 4.1% to the threshold, and 0.7% to the metric. Thus, the metric accounts for only 0.7% in this structural decomposition even though the metric is the largest driver of claim variance. A standards body reading the structural decomposition would standardise the ablation operator and fix almost none of the filing instability.

### 7.4 The result under maximum pressure

We strip every trace of circuit size from the claim, leaving a statement of the dominant layer band alone, and then hold circuit size fixed with Equation 3. Under these controls, the flip rate is $F=0.2706$ (95% CI 0.2550 to 0.2859). The interval lower bound remains above the pre-registered threshold. Thus, where in the model the behaviour is attributed still flips on 27% of pairs when nothing about size can contribute.

### 7.5 The random baseline, and why its pooled direction misleads

For the random baseline, circuits are drawn from the full 32,491-edge population with sizes matched to the discovered circuits. We use $R=1{,}000$ null multiverses, where $R$ is the number of random-null repetitions. In the pooled comparison, the discovered multiverse is less stable than this size-matched random null: the discovered flip rate is 0.7316 against the null flip rate of 0.6774, with disjoint intervals.

H3, the pre-registered expectation of no separation from the random null, is therefore rejected: the distributions are separated. However, we do not report the direction of that separation as a finding, because the direction is an artefact of the size distribution. Table 7 shows why. The null is degenerate above 1,000 edges, where a uniform draw touches all 156 components and can produce only one claim.

When circuit size is held fixed, the ordering reverses. At medium granularity, the discovered flip rate is 0.2746 against a null flip rate of 0.4230. At coarse granularity, the discovered flip rate is 0.2706 against a null flip rate of 0.3822. Thus, discovered circuits are more stable than random circuits once the comparison is fair. However, the discovered circuits are still not stable enough to file.

Table 7: Flip rate within each circuit size, medium granularity. Here, $n$ is the number of specifications at that circuit size. The null collapses to a single claim class at and above 1,000 edges, which is what depresses the null’s pooled flip rate. The discovered multiverse also collapses at 5,000 and 10,000 edges, so 2,692 of 7,561 specifications contribute no flips at all and the instability lives entirely below 2,000 edges.

| Size | $n$ | Discovered $F$ | Classes | Null $F$ | Null classes |
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

We compute functional agreement from circuits’ per-example correct or incorrect verdicts, using the measures of Bayat Makou et al. [1], reproduced verbatim. Per-example functional agreement between circuits is 0.6028. Functional instability, defined as one minus that agreement, is therefore 0.3972. The gap between the claim flip rate and functional instability is 0.3344 (95% CI 0.3297 to 0.3394), against a pre-registered threshold of 0.10.

However, Cohen’s kappa is 0.0146. The 60% raw agreement is almost entirely explained by both circuits being right about the same fraction of examples, not the same examples. Kappa stays between 0.016 and 0.047 at every circuit size, while raw agreement climbs from 0.52 to 0.92, tracking accuracy from 0.51 to 0.95.

This is the answer to the phantom specialisation objection raised in Section 2. If discovery sampled from an equivalence class of behaviourally identical subgraphs, the circuits would be functionally interchangeable. At kappa 0.015, the circuits are not functionally interchangeable.

H4 is supported on the pre-registered quantity because the pooled gap of 0.3344 exceeds the threshold of 0.10. However, the gap does not survive holding size fixed: the gap fails the 0.10 threshold at 6 of 10 sizes and is negative at 5. At both ends of the size range, the mechanisms differ more than the filings do, and the pooled figure is carried by the 500 to 2,000 edge band alone. We report the pooled result because the pooled result was pre-registered, and we report the decomposition beside the pooled result because the pooled result alone would mislead.

## 8 The filability protocol

Filability of an interpretability claim. For a filed claim $c$ derived from a circuit:

1. Declare the specification space $\mathcal{S}$: the analytic axes and the levels of each that a competent analyst could defend, each with a citation.

2. Declare the claim map $\varphi$ as deterministic, published code. The claim map must not be a language model.

3. Over the specification space $\mathcal{S}$, compute the modal share $\pi^{*}=\max_c n_c/N$, where $n_c$ is the number of specifications yielding claim class $c$ and $N$ is the number of specifications. Also compute the flip rate $F$ by Equation 2.

4. The claim is filable at tolerance $\alpha$ if $\pi^{*}\geq 1-\alpha$.

5. Report the modal share $\pi^{*}$, the flip rate $F$, the discard rate per axis level, and the within-group decomposition of any pooled statistic.

A filing that cannot state its own modal share $\pi^{*}$ has not been audited. The filing has been asserted. The protocol does not require the regulator to adjudicate which analytic choice is correct. That absence of a requirement to adjudicate is the property that makes the protocol implementable: the protocol requires only that the filer declare the specification space and report how much of that space agrees.

## 9 Limitations

#### One model, one task.

Larger models have more components, so both the layer band and the size class have more room. However, the direction of the effect on the flip rate $F$ is not obvious. The restriction to one model and one task is the limitation most likely to change the number.

A second-model replication is pre-registered but not yet run: a 924-cell grid on Pythia-160m, a model of a different architecture family that performs the same task. The replication grid is identical on every axis except that the seed axis carries three levels rather than five. The analysis plan, the reduced grid, and the criteria for what would count as replication rather than refutation are committed and timestamped before any cell of the replication ran. The replication will be reported in a revision in whichever direction the result falls. A second task remains designed and not run.

#### Circuit size is an outcome, not an axis.

Circuit size is chosen by the threshold rule, and circuit size has distorted three separate pooled quantities: the flip rate $F$ itself, the direction of the null comparison, and the sign of the functional gap. Holding circuit size fixed drops $F$ from 0.7316 to 0.2746, a larger reduction than standardising any pre-registered axis. We take this to be a general hazard for multiverse designs whose specifications select objects of different sizes. For that reason, we report every pooled statistic with its within-size decomposition.

#### The multiverse is as wide as the library, not the literature.

Optimal ablation [5] is a defensible operator that the instrument does not implement, so optimal ablation never entered the specification space. The measured flip rate $F$ is therefore a lower bound with respect to the field’s full set of defensible choices.

#### The discard rate is high and uneven.

The discard rate is 52.3%, and the surviving pool is metric imbalanced. The threshold rule was fixed in advance and no failing item was repaired. Nevertheless, the composition is a property of the reported set, so we state the composition rather than adjust for the composition.

#### The bootstrap is invalid for one reported family.

Where the conditioning groups hold only two to five specifications, retained self-pairs dominate and the bootstrap intervals are unusable. Those quantities are therefore reported as point estimates with the diagnostic attached. We did not construct a bias-corrected estimator after seeing the naive intervals disagree, because choosing an estimator in response to a result is the behaviour this paper documents.

## 10 Discussion

#### What standardising one axis buys.

Standardising one axis buys less than appears. The evaluation metric is the biggest lever, but standardising the metric removes only 0.1377 of a 0.7316 flip rate. There is therefore no single knob. A procedure that standardises one axis and declares the problem solved would be mistaken by a wide margin.

#### On the instrument.

One of seven documented, publicly exported discovery objectives does not run on the library’s own canonical task. We report this failure as a discard with its rate, and as an observation about the maturity of the tooling that regulatory evidence is expected to rest on. The failure is not the paper’s headline, and the observation is not a criticism of a library whose authors are among those calling for the field to address exactly this [10].

#### What a regulator can take from this.

The conclusion is not that interpretability evidence is worthless. Rather, the evidentiary weight of interpretability evidence is a measurable property instead of an assumed property, and the measurement is cheap relative to the sweep that produced the circuit in the first place.

## 11 Reproducibility

The pre-registration is a timestamped commit tagged `prereg-p1-confirmatory` in the public repository, made before the confirmatory sweep began. Every reported number traces to a configuration, a seed and an environment fingerprint. The confirmatory sweep ran under a single environment across all 1,320 cells; the shared environment was verified from the results archive rather than asserted. The code commit was recovered by content-addressed comparison of the uploaded archive against the git tree, with 22 of 22 files identical. The analysis layer runs without a graphics processing unit (GPU).

A verification script recomputes every headline quantity from raw results and asserts against what is reported: 36 checks, 0 failures. The flip rate $F$ is derived by three routes, including a brute-force pairwise computation. A second script checks that every numeric token in this manuscript matches a value derived from the stored results. Every table in this paper is emitted directly from those results rather than transcribed. The test suite is 343 tests.
