# Explanation Multiplicity: Circuit-Level Interpretability

Evidence Does Not Survive Defensible Analytic Variation

Ajay Pravin Mahale

12 August 2026

###### Abstract

The European Union (EU) AI Act requires providers of high-risk artificial intelligence (AI) systems to file technical documentation that describes how their systems reach decisions. Mechanistic interpretability—the study of a model’s internal mechanisms to explain its behaviour—is the obvious source of this evidence. Circuit discovery, which finds a set of model connections said to explain a behaviour, is its most developed tool. We ask whether this evidence survives the conditions under which people would rely on it: two competent analysts use the same system and tool, but choose different defensible settings.

We pre-registered a grid that combines seven analytic axes, or dimensions of analytic choice. Every level came from a published implementation. We then used a deterministic claim map—a fixed rule that gives the same output for the same input—to turn each discovered circuit into a structured Annex IV statement. Across 15,840 pre-registered specifications on GPT-2 small and the indirect object identification task, 7,561 produced a claim. The resulting statement changes across 73.2% of specification pairs (95% confidence interval, or CI, 0.725 to 0.738). The modal claim, or most common claim, covers 41.1% of the space. The evidence fails a filability criterion—a rule for deciding whether evidence is stable enough to file—at every tolerance a conformity assessment body would plausibly accept.

Standardising the most influential single choice, the evaluation metric, leaves the flip rate at 59.4%. An evaluation metric is a rule for measuring performance. Removing circuit size from the claim and holding size fixed leaves 27.1% (95% CI 0.255 to 0.286), still above the pre-registered threshold. The circuits behind these claims share almost no structure. Their median pairwise Jaccard overlap—the shared edges divided by all edges in either circuit—is 4%. Their functional agreement beyond chance, measured by Cohen’s kappa, is 0.015. Thus, the instability does not reflect one mechanism described in different words.

We provide the filability criterion as a standalone protocol. We also report that one of the seven documented discovery objectives does not run on the library’s own canonical task. The study covers one model and one task. Whether the conclusion holds at scale remains untested.

## 1 Introduction

An audit means two auditors reach the same conclusion. This is not a definition borrowed from machine learning. It is what makes an audit worth commissioning.

Regulation (EU) 2024/1689 requires providers of high-risk AI systems to file technical documentation. Annex IV point 2(b) asks for “the general logic of the AI system and of the algorithms”. Point 2(e) asks for “an assessment of the technical measures needed to facilitate the interpretation of the outputs”. Article 86(1) gives a person subject to certain automated decisions the right to “clear and meaningful explanations of the role of the AI system in the decision-making procedure and the main elements of the decision taken”. These quotations come from the consolidated text as amended by Regulation (EU) 2026/1744, CELEX 02024R1689-20260727, read 11 August 2026.

Mechanistic interpretability offers to supply the technical content. Circuit discovery produces a subgraph—a selected part of the model’s network of connections—said to explain a behaviour. It is the part of the field with mature tools. A provider may file a circuit-level account of why its system behaved as it did. That filing has value only if a second analyst, given the same system and tool, would file a compatible account.

We measure how far a regulatory claim changes across defensible analytic specifications. We also provide a decision rule that a standards body could use. We do not propose a better circuit discovery method. Nor do we argue that interpretability is worthless. We argue that the evidence, as currently produced, does not meet the standard that filing it presumes.

#### Contributions.

(i) A pre-registered multiverse—a set of analyses covering defensible choices—over seven analytic axes of circuit discovery. Every level comes from a published implementation. A deterministic map turns circuits into Annex IV statements.

(ii) A filability criterion, $\pi^{*}\geq 1-\alpha$, that a conformity procedure can apply, together with its measured value on a canonical task. Here, $\pi^{*}$ is the modal claim’s share, and $\alpha$ is the allowed tolerance.

(iii) Evidence that the instability survives the strongest available controls: standardising any single axis, removing circuit size from the claim, and holding size fixed.

(iv) A methodological result for multiverse designs in general: when specifications select objects of different sizes, pooling them can reverse statistics. We show three such reversals.

## 2 Related work

#### Non-identifiability.

Non-identifiability means that the available criteria do not identify one unique explanation. Méloux et al. [6] ask whether a unique explanation exists for a given behaviour under mechanistic interpretability’s own criteria. They find that it need not. We share their premise. Our contribution concerns the next step: we measure the cost of non-identifiability when someone files an explanation rather than publishes it.

#### Faithfulness metric sensitivity.

Faithfulness measures how well a circuit accounts for the model behaviour it is meant to explain. Miller et al. [8] survey design choices in ablation-based faithfulness measurement, which tests a circuit by removing or replacing model signals. They find that existing methods are highly sensitive to seemingly minor changes. They conclude that faithfulness scores reflect the researcher’s methods as well as the circuit. They establish that one axis matters. We combine seven axes and carry the result through to a regulatory statement. We use their released library, so our study stress-tests the tools by their own definitions.

#### Structure versus function.

Bayat Makou et al. [1] vary input statistics while holding the task fixed. They find that structurally distinct circuits perform the same computation, a pattern they call phantom specialisation. This is the strongest objection to our design. If discovery merely samples from a set of behaviourally identical subgraphs, a claim flip rate measures which member it sampled. We address this objection with a functional measurement in Section 7.6, rather than with an argument. Our circuits are not functionally interchangeable.

#### Variance as estimation.

Méloux et al. [7] treat mechanistic interpretability as statistical estimation and analyse the estimator’s variance, or how much its estimates vary. Seed variance is one axis of our space. Their framework is the natural statistical companion to our analysis of the whole specification.

#### Auditability.

Lan et al. [4] note that two papers reached conflicting conclusions about the same behaviour. A third found both partly correct and incomparable. They call for auditing standards that would allow findings to be certified in safety-critical settings. We answer that call with a measurement and a criterion, rather than a proposal. Sharkey et al. [10] list open problems in the field, including the social and technical issues this paper touches. Mueller et al. [9] argue that comparing methods requires standardised evaluation. Their argument is close to ours: certification requires standardised specifications.

#### Multiverse method.

Our design follows Steegen et al. [13], who propose reporting results across defensible data-processing choices. It also follows Simonsohn et al. [12], whose specification curve—a plot of results across analytic specifications—we adapt. Simmons et al. [11] named the underlying problem: undisclosed flexibility lets an analyst present almost anything as significant. We apply methods from psychology to a question about machine learning evidence. This transfer is substantive. The claim that an explanation is stable is exactly the kind of claim that analytic flexibility can manufacture.

#### Explanation metrics under optimisation pressure.

Hsia et al. [3] show that sufficiency and comprehensiveness can be inflated without changing predictions or explanations. Sufficiency measures how much behaviour selected information preserves. Comprehensiveness measures how much behaviour changes when selected information is removed. These are two of our four evaluation metrics. This is one reason we report discard rates for each metric rather than silently pooling them.

## 3 What this paper is not claiming

We state these points early because the results do not support these readings. We held two of these readings ourselves before we ran the decompositions.

We do not claim that discovered circuits are no better than random. At fixed circuit size, they are more stable than a size-matched random null—a comparison based on randomly selected circuits of the same size: 0.2746 against 0.4230. They carry information.

Nor do we claim that filings differ where mechanisms do not. Mechanisms differ greatly. Functional instability is 0.3972 when pooled and 0.4777 at the smallest circuits.

Nor do we claim that discovery samples one mechanism and describes it in different ways. Cohen’s kappa of 0.0146 says the circuits are functionally uncorrelated.

Nor do we claim that random seeds are the problem. A seed sets a random procedure’s choices. Here, it selects which prompts are sampled, so it measures variation in the evaluation set rather than nondeterminism.

We claim nothing about scale. One model, one task.

## 4 Formalism

A specification is a tuple, or ordered set, of analytic choices:

$$
s=(o,a,d,m,\tau,P,r,g).
$$

Here, $o$ is the discovery objective, or quantity the method aims to optimise. The ablation operator $a$ specifies how signals are removed or replaced. The corruption distribution $d$ specifies the source of altered inputs. The evaluation metric is $m$, the size threshold is $\tau$, the prompt variant is $P$, the seed is $r$, and the granularity, or level of detail, is $g$. Every level of every axis comes from a published implementation. We invented none for this study.

Circuit discovery returns $C(s)\subseteq E$: a circuit whose edges form a subset of the model’s full edge set $E$. For GPT-2 small under the factorised graph, which separates query, key and value inputs, $|E|=32{,}491$. Query, key and value inputs let attention match and combine information.

We confirm this edge count from the instrument and independently from a closed-form count over the computation order. For block $b$, attention receives information from $1+13b$ sources at three input slots for each of 12 heads. An attention head is a component that weights and combines information. The multilayer perceptron (MLP), a feed-forward network in block $b$, receives information from those sources plus its own block’s heads. The output receives information from all 157 non-terminal sources. Circuits touch components drawn from 144 attention heads and 12 MLPs, for 156 components in total.

#### Circuit instability.

For two specifications, define the Jaccard distance as one minus their Jaccard similarity:

$$
D(s_i,s_j)=1-\frac{|C_i\cap C_j|}{|C_i\cup C_j|}.
\tag{1}
$$

The numerator counts shared edges. The denominator counts edges in either circuit. We use $\bar{J}$ for the mean Jaccard similarity over unordered pairs.

#### Claim instability.

Let $\varphi$ map a circuit to a structured Annex IV statement. The flip rate is the probability that two distinct specifications produce different statements:

$$
F=1-\frac{\sum_c n_c(n_c-1)}{N(N-1)}.
\tag{2}
$$

Here, $n_c$ counts specifications that produce claim class $c$, and $N$ is the number of specifications. Equation 2 uses the unbiased Gini–Simpson form. It gives the probability that two draws without replacement differ. Unit tests check it against the brute-force pairwise definition over randomised cases. The modal share is $\pi^{*}=\max_c n_c/N$.

We also use a within-group form. For a set of conditioning axes $G$:

$$
F_{\text{within}}(G)
=
1-\frac{\sum_g\sum_c n_{g,c}(n_{g,c}-1)}
{\sum_g n_g(n_g-1)}.
\tag{3}
$$

Here, $n_{g,c}$ counts specifications in group $g$ that produce claim class $c$, and $n_g$ counts all specifications in that group. This form weights every pair equally, rather than every group equally. It reduces exactly to Equation 2 when $G$ is empty.

#### Filability.

Evidence is filable at tolerance $\alpha$ if and only if $\pi^{*}\geq 1-\alpha$.

The flip rate has a ceiling: $F\leq 1-1/k$ for $k$ claim classes. A reported flip rate must therefore be read against the number of classes the map can produce. With the nine classes observed here, the ceiling is 0.8889. The observed 0.7316 is 82.3% of that ceiling. The construction does not force $F$ near one.

## 5 The claim map

The claim map $\varphi$ is the paper’s most open-to-challenge choice. We therefore fixed it in the pre-registration, implemented it as deterministic code, and published it. A language model does not generate it. Using one would add model variance to circuit variance, with no way to separate them.

The map has two addressees because the regulation does, not for convenience. We build $\varphi_{\text{overseer}}$ on Annex IV 2(e) and 3 with Article 14(4)(c). It uses the dominant layer band—the early, middle or late group of layers—then the size class, then the ranked input segment. We build $\varphi_{\text{affected}}$ on Article 86(1). It leads with the input segment because the right belongs to the person the decision concerns. The three granularities are nested by construction: a finer claim adds detail to a coarser claim rather than contradicting it.

A reviewer may argue that we chose size bins to produce the reported flip rate. We did not choose them directly. A committed calibration rule returned them from a measured node-count curve. Table 1 reports the sensitivity.

Table 1: Flip rate under alternative size binnings. The committed bins are not a maximum: four alternatives give a higher $F$, including halving every threshold. Across binnings differing by a factor of four, the range is 0.70 to 0.77. Removing the size term from the claim entirely still leaves 0.5733.

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

We study GPT-2 small on indirect object identification, the task of identifying the indirect object in a sentence [15]. We do not apologise for this choice. The non-identifiability and faithfulness literature we extend works at this scale. Comparability is the point. The edge count also makes a size-matched random baseline meaningful.

#### Grid.

The pre-registered grid includes seven discovery objectives, seven ablation operators, four corruption distributions nested within the five operators that read them, four metrics, three thresholds, two prompt variants, and five seeds. This gives 1,540 discovery cells and 18,480 specifications. A cell is a combination of discovery settings.

Corruption is nested rather than fully crossed because two ablation operators never read the corruption distribution. Crossing those operators with corruption would produce duplicate specifications. Discovery uses the edge attribution patching family [14] and integrated edge gradients, as implemented in the library released with Miller et al. [8]. These methods use changes in signals, or gradients measuring sensitivity to change, to assign importance to edges. The automated circuit discovery lineage is Conmy et al. [2]. The four corruption levels come from Wang et al. [15].

#### Pre-registration.

We committed and tagged the analysis plan before the confirmatory sweep—the planned full run—began. We drafted both abstracts in advance so that neither direction could be written up as a surprise. An append-only file records deviations, their dates, and whether we decided each one before or after seeing the affected result.

#### Statistics.

We compute no $p$-value, a statistical significance measure, across specifications anywhere. Specifications form a designed grid, not an independent sample [13, 12]. We quantify uncertainty with a nonparametric bootstrap: we repeatedly resample the specifications themselves, without assuming a distributional form. We use $B=10{,}000$ bootstrap repetitions. We resample specifications, never pairs, because $F$ and $\bar{J}$ are U-statistics, which combine measurements over pairs, and pairs that share a specification are dependent. The intervals quantify uncertainty given the grid. They support no inference beyond it.

#### Two exclusions, criteria fixed before any circuit was seen.

One discovery objective does not run. All 220 of its cells raise a runtime error inside the library’s own code. The error comes from computing mean squared error—the average squared difference between an output and a target—against an integer-valued target. We computed a digest, or identifying summary, over the sorted failing cell identifiers. This verified that the failures cover exactly that objective’s arm and nothing else. We did not repair the error. The library is the instrument under test, and we reproduce it verbatim. The realised grid contains six objectives, 1,320 cells and 15,840 specifications.

The metric-relative threshold rule admits no circuit for 8,279 of 15,840 specifications. The discard rate is 52.3%, and it differs across metrics (Table 2). We disclosed this before locking the pre-registration. Comprehensiveness therefore accounts for 52% of the surviving pool, against 25% by design. We report this composition rather than smoothing it.

Table 2: Discard rate by evaluation metric under the pre-registered metric-relative threshold rule. We disclosed the unevenness before locking the pre-registration, based on a six-cell validation slice. KL divergence means Kullback–Leibler divergence, a measure of difference between distributions. Logit difference measures a difference between model scores before their conversion to probabilities.

| Metric | Kept | Discarded | Rate |
|---|---:|---:|---:|
| comprehensiveness | 3,947 | 13 | 0.003 |
| KL divergence | 1,467 | 2,493 | 0.630 |
| logit difference | 1,270 | 2,690 | 0.679 |
| sufficiency | 877 | 3,083 | 0.779 |

## 7 Results

Table 3 gives each pre-registered outcome and its result. We fixed and tagged the decision rules and thresholds before the confirmatory sweep. We revisited none.

Table 3: Pre-registered outcomes and their results. We fixed the thresholds on 6 August 2026 and tagged the plan before the sweep began. Two results need qualifications that only the within-size decomposition reveals. We report these in Sections 7.5 and 7.6 rather than defer them to an appendix.

| | Decision rule | Result | Outcome |
|---|---|---|---|
| P0 | $\bar{J}$ reported, not tested | 0.1396 [0.1377, 0.1418] | premise holds |
| H2 | $F>0.20$, CI lower bound above | 0.7316 [0.7247, 0.7380] | confirmed |
| H3 | not separated from random null | separated, intervals disjoint | rejected |
| H4 | $F-(1-\text{agreement})>0.10$ | 0.3344 [0.3297, 0.3394] | supported |

### 7.1 Circuits are almost disjoint

We find $\bar{J}=0.1396$ (95% CI 0.1377 to 0.1418). We compute this exactly over all 28,580,580 specification pairs, rather than sampling pairs. The 7,561 specifications contain 3,218 distinct circuits, which makes exact computation feasible.

The distribution shows more than the mean. For pairwise $D$, the 1% quantile is 0.196, the 25% quantile is 0.788, the median is 0.960, and the 75% quantile is 0.998. These quantiles mark the values at the stated percentages of the distribution. The median, or middle, pair of circuits shares 4% of its edges. Only 1% of pairs exceed a Jaccard similarity of 0.8.

At the median circuit size of 200 edges, the analytic random baseline $k/(2|E|-k)$ gives 0.0031, where $k$ is circuit size. The observed overlap is therefore about 45 times chance. Beating chance at this scale is not evidence of much: a 500-edge circuit has an expected random overlap of 0.008.

### 7.2 The claim flips on three quarters of pairs

We find $F=0.7316$ (95% CI 0.7247 to 0.7380). The pre-registered threshold is 0.20, and the interval’s lower bound must exceed it. It clears the threshold by more than three times. The modal share is $\pi^{*}=0.4109$. The evidence is therefore not filable at any pre-registered tolerance: 0.411 against requirements of 0.95, 0.90 and 0.80.

Table 4: The nine claim classes produced by $\varphi_{\text{overseer}}$ at medium granularity, ordered by frequency. After aggregating over size class, layer attribution splits into 47.9% early, 43.6% late and 8.5% middle. The first is the independently computed COARSE modal share of 0.4788.

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

The two most common claims cover 41.1% and 28.5% of the space. They attribute the same model’s behaviour on the same task to early layers and to late layers (Table 4). These are not different emphases. They contradict.

All six addressee–granularity combinations reject filability (Table 5). The coarsest statement available to an affected person has two claim classes. It still flips on 39.5% of pairs. The result is therefore not an artefact of fine-grained claim language. Figure 1 shows the specification curve. The modal step’s width is $\pi^{*}$.

Table 5: Flip rate and modal share for both addressees at all three granularities. We evaluate filability at the loosest pre-registered tolerance, $\alpha=0.20$, which requires $\pi^{*}\geq 0.80$. No cell passes.

| Addressee | Granularity | Classes | $F$ | $\pi^{*}$ | Filable |
|---|---|---:|---:|---:|---|
| affected | COARSE | 2 | 0.3950 | 0.7291 | no |
| affected | MEDIUM | 4 | 0.6597 | 0.4875 | no |
| affected | FINE | 8 | 0.6744 | 0.4786 | no |
| overseer | COARSE | 3 | 0.5733 | 0.4788 | no |
| overseer | MEDIUM | 9 | 0.7316 | 0.4109 | no |
| overseer | FINE | 12 | 0.7681 | 0.4109 | no |

Figure 1: Specification curve for $\varphi_{\text{overseer}}$ at medium granularity. Upper panel: claim class against specification rank. Classes are ordered by frequency, so each step’s width gives that class’s share of the space. The dashed line marks the size-matched random null’s modal share. Circuit size confounds the pooled comparison, as discussed in Section 7.5. Lower panel: the active analytic level for each specification. The solid band on the comprehensiveness row reflects the discard rule’s effect on composition in Table 2. The seed rows show no structure, matching the result that seed removes 0.0009 from $F$ in Table 6.

### 7.3 No analytic axis rescues the filing

Table 6: Residual flip rate after standardising each axis, computed with Equation 3. All intervals are valid: group sizes range from 1,512 to 2,520, bootstrap bias is below 0.0006 on every axis, and every observed value lies inside its own interval.

| Axis standardised | Residual $F$ | 95% CI | Removes |
|---|---:|---|---:|
| Evaluation metric | 0.5939 | 0.5803 to 0.6063 | 0.1377 |
| Ablation operator | 0.7014 | 0.6939 to 0.7075 | 0.0302 |
| Discovery objective | 0.7018 | 0.6938 to 0.7084 | 0.0298 |
| Threshold $\tau$ | 0.7070 | 0.6981 to 0.7149 | 0.0246 |
| Corruption | 0.7092 | 0.7008 to 0.7166 | 0.0224 |
| Prompt variant | 0.7232 | 0.7157 to 0.7300 | 0.0084 |
| Seed | 0.7307 | 0.7234 to 0.7368 | 0.0009 |

The metric has the largest single effect, but it is not enough. Standardising it completely—more than a conformity procedure could demand—leaves $F=0.5939$. The interval’s lower bound is 0.5803, still far above the threshold.

Different axes govern structure and claims. We use restricted maximum likelihood (REML), a method for estimating variance components, to decompose variance in $\log_{10}$ of selected circuit size. Here, $\log_{10}$ means the base-ten logarithm. The decomposition attributes 78.8% of structural variance to the ablation–corruption interaction, or their combined effect. It attributes 11.5% to residual variance, 4.4% to ablation, 4.1% to the threshold, and 0.7% to the metric. Yet the metric is the largest driver of claim variance. A standards body that relied on the structural decomposition would standardise the ablation operator and fix almost none of the filing instability.

### 7.4 The result under maximum pressure

We remove every trace of circuit size from the claim, leaving only the dominant layer band. We also hold circuit size fixed using Equation 3. The result is $F=0.2706$ (95% CI 0.2550 to 0.2859). The interval’s lower bound exceeds the pre-registered threshold. Even when size cannot contribute, the claim about where in the model the behaviour arises still flips on 27% of pairs.

### 7.5 The random baseline, and why its pooled direction misleads

When pooled, the discovered multiverse is less stable than a size-matched random null drawn from the full population of 32,491 edges. Across $R=1{,}000$ null multiverses, the comparison is 0.7316 against 0.6774, with disjoint intervals.

We therefore reject H3: the distributions are separated. We do not report the direction of that separation as a finding because the size distribution creates it. Table 7 shows why. Above 1,000 edges, the null becomes degenerate: a uniform draw touches all 156 components and can produce only one claim.

Holding size fixed reverses the ordering. At medium granularity, the discovered flip rate is 0.2746 against a null rate of 0.4230. At coarse granularity, it is 0.2706 against 0.3822. Discovered circuits are more stable than random ones once the comparison is fair. They are still not stable enough to file.

Table 7: Flip rate within each circuit size, at medium granularity. The null collapses to a single claim class at and above 1,000 edges, which lowers its pooled flip rate. The discovered multiverse also collapses at 5,000 and 10,000 edges. Thus, 2,692 of 7,561 specifications contribute no flips at all, and the instability lies entirely below 2,000 edges. Here, $n$ is the number of specifications at each size.

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

Per-example functional agreement between circuits is 0.6028, so functional instability is 0.3972. The gap between functional instability and the claim flip rate is 0.3344 (95% CI 0.3297 to 0.3394), against a pre-registered threshold of 0.10. We compute agreement from each circuit’s correct or incorrect verdict on each example. We reproduce the measures of Bayat Makou et al. [1] verbatim.

Cohen’s kappa is 0.0146. The 60% raw agreement is almost entirely explained by both circuits getting the same fraction of examples right, rather than getting the same examples right. Kappa stays between 0.016 and 0.047 at every circuit size. Meanwhile, raw agreement climbs from 0.52 to 0.92, tracking accuracy from 0.51 to 0.95.

This answers the phantom specialisation objection. If discovery sampled from a set of behaviourally identical subgraphs, the circuits would be functionally interchangeable. At kappa 0.015, they are not.

The pre-registered quantity supports H4. But the gap does not survive holding size fixed. It fails the 0.10 threshold at 6 of 10 sizes and is negative at 5. At both ends of the range, mechanisms differ more than filings do. The pooled figure comes from the 500 to 2,000 edge band alone. We report the pooled result because we pre-registered it. We report the decomposition beside it because the pooled result alone would mislead.

## 8 The filability protocol

Filability of an interpretability claim. For a filed claim $c$ derived from a circuit:

1. Declare the specification space $\mathcal{S}$: the analytic axes and the levels of each that a competent analyst could defend. Give a citation for each level.

2. Declare the claim map $\varphi$ as deterministic, published code. Do not use a language model.

3. Compute $\pi^{*}=\max_c n_c/N$ over $\mathcal{S}$, and compute $F$ using Equation 2.

4. The claim is filable at tolerance $\alpha$ if $\pi^{*}\geq 1-\alpha$.

5. Report $\pi^{*}$, $F$, the discard rate for each axis level, and the within-group decomposition of any pooled statistic.

A filing that cannot state its own $\pi^{*}$ has not been audited. It has been asserted. The protocol does not require the regulator to decide which analytic choice is correct. This makes it practical to implement. It requires only that the filer declare the space and report how much of it agrees.

## 9 Limitations

#### One model, one task.

Larger models have more components, so both the layer band and the size class have more room. The direction of the effect on $F$ is not obvious. This limitation is the most likely to change the number.

We have pre-registered, but not yet run, a second-model replication: a 924-cell grid on Pythia-160m. This model belongs to a different architecture family and performs the same task. The grid is identical on every axis except seed, which has three levels rather than five. Before any cell ran, we committed and timestamped the analysis plan, the reduced grid, and the criteria for deciding whether the result would replicate or refute the finding. We will report it in a revision, whichever direction it takes. We have designed a second-task study but have not run it.

#### Circuit size is an outcome, not an axis.

The threshold rule chooses circuit size. Size has distorted three separate pooled quantities: $F$ itself, the direction of the null comparison, and the sign of the functional gap. Holding size fixed lowers $F$ from 0.7316 to 0.2746. This reduction exceeds the effect of standardising any pre-registered axis. We take this as a general hazard for multiverse designs whose specifications select objects of different sizes. We report every pooled statistic with its within-size decomposition.

#### The multiverse is as wide as the library, not the literature.

Optimal ablation [5] is a defensible operator that the instrument does not implement, so it never entered the space. The measured $F$ is therefore a lower bound relative to the field’s full set of defensible choices.

#### The discard rate is high and uneven.

The discard rate is 52.3%, and the surviving pool is imbalanced across metrics. We fixed the threshold rule in advance and repaired no failing item. The composition remains a property of the reported set, so we state it rather than adjust for it.

#### The bootstrap is invalid for one reported family.

When conditioning groups contain only two to five specifications, retained self-pairs—pairs that match a resampled specification with itself—dominate. The intervals are then unusable. We report those quantities as point estimates with the diagnostic attached. We did not construct a bias-corrected estimator after seeing the naive intervals disagree. Choosing an estimator in response to a result is the behaviour this paper documents.

## 10 Discussion

#### What standardising one axis buys.

It buys less than it appears to. The metric has the largest effect and removes 0.1377 from a flip rate of 0.7316. No single setting solves the problem. A procedure that standardised one axis and declared success would be mistaken by a wide margin.

#### On the instrument.

One of seven documented, publicly exported discovery objectives does not run on the library’s own canonical task. We report this as a discard with its rate. We also report it as an observation about the maturity of the tools on which regulatory evidence is expected to rest. It is not the paper’s headline. Nor is it a criticism of a library whose authors are among those calling for the field to address exactly this problem [10].

#### What a regulator can take from this.

The lesson is not that interpretability evidence is worthless. Its evidentiary weight is measurable, rather than something to assume. Measuring it is cheap relative to the sweep that produced the circuit in the first place.

## 11 Reproducibility

The pre-registration is a timestamped commit tagged `prereg-p1-confirmatory` in the public repository. We made it before the confirmatory sweep began. Every reported number traces to a configuration, a seed and an environment fingerprint—an identifying record of the computing environment.

The confirmatory sweep used one environment across all 1,320 cells. We verified this from the results archive rather than merely asserting it. We recovered the code commit through a content-addressed comparison of the uploaded archive against the git tree: 22 of 22 files were identical. The analysis layer runs without a graphics processing unit (GPU).

A verification script recomputes every headline quantity from raw results and checks it against the reported value: 36 checks, 0 failures. It derives $F$ by three routes, including a brute-force pairwise computation. A second script checks that every numeric token in this manuscript matches a value derived from the stored results. The results directly generate every table in this paper; we do not transcribe them. The test suite contains 343 tests.
