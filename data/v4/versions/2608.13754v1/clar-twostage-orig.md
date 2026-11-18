# Explanation Multiplicity: Circuit-Level Interpretability

Evidence Does Not Survive Defensible Analytic Variation

Ajay Pravin Mahale

12 August 2026

###### Abstract

The European Union (EU) Artificial Intelligence (AI) Act requires providers of high-risk systems to file technical documentation describing how the system reaches its decisions. Mechanistic interpretability—the study of a model’s internal computations—is the obvious source of such evidence. Its most developed instrument is circuit discovery: finding a subgraph of model components and connections said to explain a behaviour. We ask whether this evidence survives the conditions under which it would be used: two competent analysts, the same system, the same tool, and different defensible settings.

We pre-registered a crossed grid of seven analytic axes, or types of analytic choice. Every setting came from a published implementation. A deterministic claim map—a fixed rule implemented in code—converted each discovered circuit into a structured statement for Annex IV, the Act’s technical-documentation annex. Across 15,840 pre-registered specifications on the model named GPT-2 small and the indirect object identification task, 7,561 produced a claim. The statement differs across 73.2% of specification pairs (95% confidence interval, or CI, 0.725 to 0.738). The modal claim, meaning the most common claim, covers 41.1% of the space. The evidence fails our filability criterion, which requires sufficient agreement on a filing, at every tolerance a conformity assessment body—a body assessing compliance—would plausibly accept.

Standardising the most influential choice, the evaluation metric used to score circuits, leaves the flip rate at 59.4%. Removing circuit size—the number of selected edges—from the claim and also holding size fixed leaves 27.1% (95% CI 0.255 to 0.286), still above the pre-registered threshold. The underlying circuits are structurally near-disjoint. Their median pairwise Jaccard overlap—the number of shared edges divided by the number present in either circuit—is 4%. They are also functionally uncorrelated: Cohen’s kappa, an agreement measure adjusted for agreement expected from the circuits’ individual correct-answer rates, is 0.015. Thus, the instability is not one mechanism described in different words.

We provide the filability criterion as a standalone protocol. We also report that one of the seven documented discovery objectives—the quantities used to guide discovery—does not execute on the library’s own canonical task. The study covers one model and one task. Whether the conclusion holds at scale is untested.

## 1 Introduction

An audit means two auditors reach the same conclusion. That is not a definition borrowed from machine learning; it is what makes an audit worth commissioning.

Regulation (EU) 2024/1689 requires providers of high-risk AI systems to file technical documentation. Annex IV point 2(b) asks for “the general logic of the AI system and of the algorithms”. Point 2(e) asks for “an assessment of the technical measures needed to facilitate the interpretation of the outputs”. Article 86(1) gives a person subject to certain automated decisions the right to “clear and meaningful explanations of the role of the AI system in the decision-making procedure and the main elements of the decision taken”. These quotations come from the consolidated text as amended by Regulation (EU) 2026/1744. Its CELEX document identifier is 02024R1689-20260727. We read it on 11 August 2026.

Mechanistic interpretability offers to supply the technical content. Circuit discovery produces a subgraph said to explain a behaviour, and it is the part of the field with mature tooling. Suppose a provider files a circuit-level account of why its system behaved as it did. That filing is worth something only if a second analyst, given the same system and tool, would file something compatible.

We measure how far a regulatory claim moves across defensible analytic specifications: complete combinations of analytic settings. We attach a decision rule that a standards body could use. We do not propose a better circuit discovery method. We do not argue that interpretability is worthless. We argue that the evidence, as currently produced, does not meet the standard that filing it presumes. The main measured claim has a modal share of 0.4109, below the pre-registered filing requirements of 0.95, 0.90 and 0.80 (Section 7.2).

#### Contributions.

(i) A pre-registered multiverse—an analysis across defensible choices—over seven analytic axes of circuit discovery. Every setting comes from a published implementation. A deterministic map converts circuits into Annex IV statements.

(ii) A filability criterion, $\pi^{*}\geq 1-\alpha$, and its measured value on a canonical task. Here $\pi^{*}$ is the share of specifications producing the most common claim. The tolerance $\alpha$ is the allowed fraction outside that claim.

(iii) Evidence that instability survives the most aggressive available controls: standardising any single axis, removing circuit size from the claim, and holding size fixed. The residual flip rates appear in Table 6 and Section 7.4.

(iv) A methodological result for multiverse designs generally: when specifications select objects of different sizes, pooled statistics can invert. We show three such cases in our results and limitations.

## 2 Related work

#### Non-identifiability.

Non-identifiability means that the behaviour and the interpretability criteria do not determine a unique explanation. Méloux et al. [6] ask whether a unique explanation exists for a given behaviour under mechanistic interpretability’s own criteria. They answer that it need not. Our premise is theirs. Our contribution comes downstream: we measure what non-identifiability costs when an explanation is filed rather than published. Bracketed numbers identify cited sources.

#### Faithfulness metric sensitivity.

Faithfulness measures how well a circuit accounts for the model behaviour attributed to it. Ablation-based measurement assesses this by altering or removing model contributions. Miller et al. [8] survey design choices in such measurement. They find existing methods highly sensitive to seemingly insignificant changes. They conclude that faithfulness scores reflect methodological choices as well as the circuit.

They establish that one axis matters. We cross seven and propagate the result to a regulatory statement. We use their released library. Our study therefore stress-tests the tooling by its own definitions.

#### Structure versus function.

Bayat Makou et al. [1] vary input statistics while holding the task fixed. They find structurally distinct circuits implementing the same computation. They call this pattern *phantom specialisation*.

This is the strongest objection to our design. Discovery might merely sample from an equivalence class: a set of subgraphs that behave identically. If so, a claim flip rate would measure which member was sampled, not a change in mechanism. We address this with a functional measurement in Section 7.6. Cohen’s kappa is 0.0146, and our circuits are not functionally interchangeable.

#### Variance as estimation.

Méloux et al. [7] treat mechanistic interpretability as statistical estimation and analyse estimator variance: how much an estimated result changes. Seed variance is one axis of our space. Their framing is the natural statistical companion to our analysis of the whole specification.

#### Auditability.

Lan et al. [4] observe that two papers reached conflicting conclusions about the same behaviour. A third found both partly correct and incomparable. They call for auditing standards so that findings can be certified in safety-critical settings. We answer with a measurement and a criterion rather than a proposal.

Sharkey et al. [10] list open problems in the field, including the socio-technical problems touched on here. Mueller et al. [9] argue that comparing methods requires standardised evaluation. That is adjacent to our argument that certification requires standardised specification.

#### Multiverse method.

Our design follows Steegen et al. [13], who propose reporting results across defensible data-processing choices. We adapt the specification curve of Simonsohn et al. [12], a visualisation of results across specifications.

Simmons et al. [11] named the underlying problem: undisclosed analytic flexibility lets an analyst present almost anything as significant. We apply this psychology methods literature to a machine learning evidentiary question. The transfer is not decorative. A claim that an explanation is stable is exactly the kind of claim that flexibility can manufacture.

#### Explanation metrics under optimisation pressure.

Hsia et al. [3] show that sufficiency and comprehensiveness—two circuit evaluation metrics—can be inflated without changing predictions or explanations. Those are two of our four evaluation metrics. This is one reason we report the discard rate for each metric rather than silently pooling them. Table 2 shows the uneven rates.

## 3 What this paper is not claiming

We state these exclusions early because the results do not support them. We ourselves held two of these readings before running the decompositions.

We are not claiming that discovered circuits are no better than random. At fixed circuit size, they are more stable than a size-matched random null: a comparison made from randomly selected edges in circuits of matching sizes. The flip rates are 0.2746 for discovered circuits and 0.4230 for the null (Section 7.5). Discovered circuits carry information.

Nor are we claiming that filings differ where mechanisms do not. Mechanisms differ substantially. Functional instability—the fraction of per-example correct-or-incorrect verdicts on which circuits disagree—is 0.3972 pooled and 0.4777 at the smallest circuits.

Nor are we claiming that discovery samples one mechanism and dresses it differently. Cohen’s kappa of 0.0146 says the circuits are functionally uncorrelated (Section 7.6).

Nor are we claiming that random seeds are the problem. Here, a seed controls which prompts are sampled. It therefore represents evaluation-set variability, not nondeterminism. Standardising seed removes 0.0009 from the claim flip rate (Table 6).

And we claim nothing at all about scale. One model, one task.

## 4 Formalism

A specification is a tuple, or ordered collection, of analytic choices:

\[
s=(o,a,d,m,\tau,P,r,g).
\]

The entries are:

- $o$: the discovery objective, which guides the search for a circuit;
- $a$: the ablation operator, which specifies how model contributions are altered;
- $d$: the corruption distribution, which supplies the corrupted inputs used by operators that read them;
- $m$: the evaluation metric used to score a circuit;
- $\tau$: the size threshold used by the circuit-selection rule;
- $P$: the prompt variant;
- $r$: the seed selecting sampled prompts;
- $g$: the claim granularity, or level of detail.

Every level of every axis comes from a published implementation. None was invented for this study.

Circuit discovery returns $C(s)\subseteq E$. Here $C(s)$ is the circuit found under specification $s$, $E$ is the model’s set of edges, and $\subseteq$ means that the circuit contains only edges from that set. An edge is a connection in the model graph. Vertical bars around a set denote its number of members.

For GPT-2 small, the factorised graph represents attention’s query, key and value inputs separately. It has $|E|=32{,}491$ edges. We confirm this both from the instrument and from an independent closed-form count following the computation order. Let $b$ index a model block. Attention in block $b$ receives from $1+13b$ sources, at three input slots for each of 12 attention heads. An attention head is an attention component. The multilayer perceptron (MLP), a feed-forward component, in block $b$ receives from those sources plus its own block’s heads. The output receives from all 157 non-terminal sources, meaning sources before the final output. Circuits touch components drawn from 144 attention heads and 12 MLPs, or 156 components in total.

#### Circuit instability.

For specifications $s_i$ and $s_j$, write $C_i=C(s_i)$ and $C_j=C(s_j)$. Their circuit instability is

\[
D(s_i,s_j)
=
1-\frac{|C_i\cap C_j|}{|C_i\cup C_j|}.
\tag{1}
\]

Here $i$ and $j$ label specifications. The intersection $C_i\cap C_j$ contains edges shared by both circuits. The union $C_i\cup C_j$ contains edges present in either circuit. The fraction is their Jaccard similarity; $D$ is one minus that similarity. We write $\bar{J}$ for the mean Jaccard similarity over unordered pairs, counting each pair once.

#### Claim instability.

Let $\varphi$ be the claim map that converts a circuit into a structured Annex IV statement. The flip rate $F$ is the probability that two distinct specifications yield different statements:

\[
F
=
1-\frac{\sum_c n_c(n_c-1)}{N(N-1)}.
\tag{2}
\]

Here $c$ labels a claim class, meaning one distinct statement; $n_c$ counts specifications yielding that statement; $N$ is the number of specifications being compared; and $\sum_c$ means to sum over claim classes.

Equation 2 is the unbiased Gini–Simpson form: a count-based measure of the probability that two draws without replacement differ. “Without replacement” excludes drawing the same specification twice. We unit-test the equation against direct, brute-force pairwise calculations over randomised cases.

The modal share is

\[
\pi^{*}=\max_c n_c/N,
\]

where $\max_c$ selects the largest claim-class share.

We also use a within-group form. Let $G$ be a set of conditioning axes. Specifications are grouped by matching settings on those axes. Then

\[
F_{\text{within}}(G)
=
1-
\frac{\sum_g\sum_c n_{g,c}(n_{g,c}-1)}
{\sum_g n_g(n_g-1)}.
\tag{3}
\]

In this equation, $g$ indexes a group, rather than denoting the granularity entry in $s$. The count $n_{g,c}$ is the number of specifications in group $g$ yielding claim $c$, and $n_g$ is that group’s total count. Equation 3 compares only specifications within the same group. It weights every pair equally, not every group equally. With no conditioning axes, $G$ is empty and Equation 3 reduces exactly to Equation 2.

#### Filability.

Evidence is filable at tolerance $\alpha$ if and only if

\[
\pi^{*}\geq 1-\alpha.
\]

Thus, the most common claim must cover at least the required fraction $1-\alpha$ of specifications.

Note the ceiling: $F\leq 1-1/k$ for $k$ claim classes. A flip rate must therefore be read against the number of classes the map can produce. With the nine classes observed here, the ceiling is 0.8889. The observed 0.7316 is 82.3% of it. $F$ is not near one by construction.

## 5 The claim map

The map $\varphi$ is the most attackable choice in the paper. We therefore fixed it in the pre-registration, implemented it as deterministic code, and published it. It is not generated by a language model. That would add model variance to circuit variance without a way to separate them.

There are two addressees, following the regulation rather than convenience:

- $\varphi_{\text{overseer}}$, the map for an overseer, is built on Annex IV 2(e) and 3 with Article 14(4)(c). Its claims are keyed first on the dominant layer band—the group of model layers receiving the dominant attribution—then the size class, then the ranked input segment.
- $\varphi_{\text{affected}}$, the map for an affected person, is built on Article 86(1). It leads with the input segment, or part of the input, because the right is owed to the person the decision is about.

The three granularity codes are COARSE, MEDIUM and FINE, from less to more detail. They are nested by construction. A finer claim therefore refines, rather than contradicts, a coarser one. For the overseer, COARSE uses the dominant layer band alone; MEDIUM distinguishes layer band and size class, as shown in Table 4.

The layer-band labels are early layers, middle layers and late layers. The calibrated size-class labels are sparse, moderate and distributed.

A reviewer will say that the size bins—ranges assigned to those classes—were chosen to produce the reported flip rate. They were not chosen. A committed calibration rule returned them from a measured node-count curve, which records counts of graph components. Table 1 reports sensitivity to alternative binnings.

Table 1: Flip rate under alternative size binnings. The committed bins are not a maximum: four alternatives give a higher $F$, including halving every threshold. Across binnings differing by a factor of four the range is 0.70 to 0.77, and removing the size term from the claim entirely still leaves 0.5733.

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

Here $\times$ means multiplication. “Equal thirds” and “deciles” denote division into three and ten size groups, respectively.

## 6 Method

#### Model and task.

We use GPT-2 small on indirect object identification [15], the task of identifying the indirect object. The choice is not an apology. The non-identifiability and faithfulness literature we extend works at this scale. Comparability is the point. The edge count also makes a size-matched random baseline meaningful.

#### Grid.

The pre-registered grid contains seven discovery objectives, seven ablation operators, four corruption distributions nested within the five operators that read them, four metrics, three thresholds, two prompt variants and five seeds. This gives 1,540 discovery cells and 18,480 specifications as pre-registered. A discovery cell is one discovery-setting combination; specifications also apply the evaluation metric and threshold choices.

Corruption is nested rather than fully crossed. Two ablation operators never read the corruption distribution, so crossing it with those operators would emit duplicate specifications.

Discovery uses the edge attribution patching family [14] and integrated edge gradients, methods that attribute contributions to graph edges. We use their implementations in the library released with Miller et al. [8]. The automated circuit discovery lineage is Conmy et al. [2]. The four corruption levels come from Wang et al. [15].

#### Pre-registration.

The analysis plan was committed and tagged before the confirmatory sweep—the full planned run—began. Both abstracts were drafted in advance, so neither direction could be written up as a surprise. Deviations are recorded in an append-only file. Each entry records its date and whether the decision came before or after the affected result was seen.

#### Statistics.

We compute no $p$-value across specifications anywhere. A $p$-value assesses a result against a statistical null model. Here the specifications are a designed grid, not an independent sample [13, 12].

Uncertainty comes from a nonparametric bootstrap: repeated resampling of specifications with replacement, without imposing a distributional model. We use $B=10{,}000$ bootstrap resamples. We never resample pairs. The reason is that $F$ and $\bar{J}$ are U-statistics built from pairs, and pairs sharing a specification are dependent. The intervals quantify uncertainty given this grid. They license nothing beyond it.

#### Two exclusions, criteria fixed before any circuit was seen.

One discovery objective does not execute. All 220 of its cells raise a runtime error—an error during execution—from a mean squared error computed against an integral target inside the library’s code. Mean squared error averages squared prediction–target differences; an integral target is integer-valued. A digest, or identifying summary, of the sorted failing-cell identifiers verifies that the failures are exactly that objective’s arm and nothing else.

We did not repair the error. The library is the instrument under test, and we reproduce it verbatim. The realised grid contains six objectives, 1,320 cells and 15,840 specifications.

The metric-relative threshold rule selects circuits relative to their evaluation metric. It admits no circuit for 8,279 of 15,840 specifications: a discard rate of 52.3%. The rate differs sharply across metrics (Table 2). This was disclosed before the lock. The surviving pool is therefore 52% comprehensiveness, against 25% by design. We report that composition rather than smoothing it.

The remaining metric names are Kullback–Leibler (KL) divergence, a measure of difference between probability distributions, and logit difference, a difference between model output scores before their conversion to probabilities.

Table 2: Discard rate by evaluation metric under the pre-registered metric-relative threshold rule. The unevenness was disclosed before the pre-registration was locked, from a six-cell validation slice.

| Metric | Kept | Discarded | Rate |
|---|---:|---:|---:|
| comprehensiveness | 3,947 | 13 | 0.003 |
| KL divergence | 1,467 | 2,493 | 0.630 |
| logit difference | 1,270 | 2,690 | 0.679 |
| sufficiency | 877 | 3,083 | 0.779 |

## 7 Results

Table 3 pairs each pre-registered outcome with its result. The codes identify the reported premise, P0, and hypotheses H2, H3 and H4. Their meanings are the decision rules in the table. The rules and thresholds were fixed and tagged before the confirmatory sweep. None was revisited.

Table 3: Pre-registered outcomes and their results. Thresholds were fixed on 6 August 2026 and the plan was tagged before the sweep began. Two results carry qualifications that only the within-size decomposition reveals, and those are reported in Sections 7.5 and 7.6 rather than deferred to an appendix.

| | Decision rule | Result | Outcome |
|---|---|---|---|
| P0 | $\bar{J}$ reported, not tested | 0.1396 [0.1377, 0.1418] | premise holds |
| H2 | $F>0.20$, CI lower bound above | 0.7316 [0.7247, 0.7380] | confirmed |
| H3 | not separated from random null | separated, intervals disjoint | rejected |
| H4 | $F-(1-\text{agreement})>0.10$ | 0.3344 [0.3297, 0.3394] | supported |

For H4, “agreement” is per-example agreement between circuits’ correct-or-incorrect verdicts. Thus, $1-\text{agreement}$ is functional instability, and H4 tests the gap between claim instability and functional instability.

### 7.1 Circuits are almost disjoint

Mean Jaccard similarity is $\bar{J}=0.1396$ (95% CI 0.1377 to 0.1418). We compute it exactly over all 28,580,580 specification pairs, rather than sampling pairs. The 7,561 specifications contain 3,218 distinct circuits, which makes the exact computation tractable.

The distribution says more than the mean. The quantiles of pairwise instability $D$ are:

- 1% quantile: 0.196;
- 25% quantile: 0.788;
- median: 0.960;
- 75% quantile: 0.998.

The median pair shares 4% of its edges. Only 1% of pairs exceed a Jaccard similarity of 0.8.

At the median circuit size of 200 edges, the analytic random line $k/(2|E|-k)$ gives 0.0031. Here $k$ denotes circuit size, rather than the claim-class count used in Section 4. Observed overlap is about 45 times chance. Beating chance at this scale is not evidence of much: a 500-edge circuit has an expected random overlap of 0.008.

### 7.2 The claim flips on three quarters of pairs

The main claim flip rate is $F=0.7316$ (95% CI 0.7247 to 0.7380). The pre-registered threshold is 0.20, and the interval’s lower bound must exceed it. It clears by more than three times.

The modal share is $\pi^{*}=0.4109$. The evidence is therefore not filable at any pre-registered tolerance: 0.411 against required shares of 0.95, 0.90 and 0.80.

Table 4: The nine claim classes produced by $\varphi_{\text{overseer}}$ at medium granularity, ordered by frequency. Aggregating over size class, the layer attribution splits 47.9% early, 43.6% late and 8.5% middle; the first of these is the COARSE modal share of 0.4788, computed independently.

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

The two most common claims cover 41.1% and 28.5% of the space. They attribute the same model’s behaviour on the same task to early layers and to late layers, respectively (Table 4, ranks 1 and 2). They are not different emphases. They contradict.

All six addressee–granularity combinations reject filability (Table 5). Even the coarsest statement for an affected person, which has only two classes, flips on 39.5% of pairs. The result is therefore not an artefact of fine-grained claim language. Figure 1 shows the specification curve; the modal step’s width is $\pi^{*}$.

Table 5: Flip rate and modal share for both addressees at all three granularities. Filability is evaluated at the loosest pre-registered tolerance, $\alpha=0.20$, which requires $\pi^{*}\geq 0.80$. No cell passes.

| Addressee | Granularity | Classes | $F$ | $\pi^{*}$ | Filable |
|---|---|---:|---:|---:|---|
| affected | COARSE | 2 | 0.3950 | 0.7291 | no |
| affected | MEDIUM | 4 | 0.6597 | 0.4875 | no |
| affected | FINE | 8 | 0.6744 | 0.4786 | no |
| overseer | COARSE | 3 | 0.5733 | 0.4788 | no |
| overseer | MEDIUM | 9 | 0.7316 | 0.4109 | no |
| overseer | FINE | 12 | 0.7681 | 0.4109 | no |

Figure 1: Specification curve for $\varphi_{\text{overseer}}$ at medium granularity. Upper panel: claim class against specification rank, classes ordered by frequency, so each step’s width is that class’s share of the space. The dashed line marks the size-matched random null’s modal share; the pooled comparison is confounded by circuit size, discussed in Section 7.5. Lower panel: which analytic level is active for each specification. The solid band on the comprehensiveness row is the composition effect of the discard rule in Table 2; the seed rows are unstructured, which is the same result as seed removing 0.0009 from $F$ in Table 6.

### 7.3 No analytic axis rescues the filing

To standardise an axis, we compare specifications within groups that hold its setting fixed, using Equation 3. Table 6 reports the residual flip rate. “Removes” is the reduction from the pooled rate.

Table 6: Residual flip rate once each axis is standardised, computed with Equation 3. All intervals are valid: group sizes 1,512 to 2,520, bootstrap bias below 0.0006 on every axis, and every observed value lies inside its own interval.

| Axis standardised | Residual $F$ | 95% CI | Removes |
|---|---:|---|---:|
| Evaluation metric | 0.5939 | 0.5803 to 0.6063 | 0.1377 |
| Ablation operator | 0.7014 | 0.6939 to 0.7075 | 0.0302 |
| Discovery objective | 0.7018 | 0.6938 to 0.7084 | 0.0298 |
| Threshold $\tau$ | 0.7070 | 0.6981 to 0.7149 | 0.0246 |
| Corruption | 0.7092 | 0.7008 to 0.7166 | 0.0224 |
| Prompt variant | 0.7232 | 0.7157 to 0.7300 | 0.0084 |
| Seed | 0.7307 | 0.7234 to 0.7368 | 0.0009 |

Bootstrap bias here means the difference between the bootstrap mean and the observed estimate. The evaluation metric is the largest single lever, removing 0.1377, but it is not enough. Completely standardising it—more than a conformity procedure could demand—leaves $F=0.5939$. The lower confidence bound is 0.5803, still far above the threshold.

Different axes govern structure and claims. We use restricted maximum likelihood (REML), a method for estimating variance components, to decompose variation in $\log_{10}$ of selected circuit size. The notation $\log_{10}$ means the base-ten logarithm. The decomposition attributes:

- 78.8% of structural variance to the ablation–corruption interaction, their joint effect;
- 11.5% to the residual, the variation left unexplained;
- 4.4% to ablation;
- 4.1% to the threshold;
- 0.7% to the metric.

Yet the metric is the largest driver of claim variance. A standards body reading the structural decomposition would standardise the ablation operator and fix almost none of the filing instability: Table 6 shows a reduction of only 0.0302.

### 7.4 The result under maximum pressure

We remove every trace of circuit size from the claim. The statement then names only the dominant layer band. We also hold circuit size fixed, comparing only specifications with matching circuit sizes through Equation 3.

The resulting flip rate is $F=0.2706$ (95% CI 0.2550 to 0.2859). The interval’s lower bound remains above the pre-registered threshold. Thus, attribution of the behaviour to a location in the model still flips on 27% of pairs when size cannot contribute to the claim or differ within a comparison.

### 7.5 The random baseline, and why its pooled direction misleads

Pooled across sizes, the discovered multiverse is less stable than a size-matched random null. The flip rates are 0.7316 for discovery and 0.6774 for the null, with disjoint intervals. The null draws from the full 32,491-edge population. We generate $R=1{,}000$ null multiverses, where $R$ is the number of random comparison multiverses.

H3 is therefore rejected: the distributions are separated. We do not report the direction of separation as a finding. It is an artefact of the size distribution, as Table 7 shows.

The null is degenerate above 1,000 edges: it produces only one claim. At those sizes, a uniform random draw touches all 156 components. Holding size fixed reverses the ordering. At medium granularity, the flip rate is 0.2746 for discovered circuits and 0.4230 for the null. At coarse granularity, it is 0.2706 against 0.3822. Discovered circuits are more stable than random ones once the comparison is fair. They are still not stable enough to file.

Table 7: Flip rate within each circuit size, medium granularity. The null collapses to a single claim class at and above 1,000 edges, which is what depresses its pooled flip rate. The discovered multiverse also collapses at 5,000 and 10,000 edges, so 2,692 of 7,561 specifications contribute no flips at all and the instability lives entirely below 2,000 edges.

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

“Size” is the number of selected edges. The count $n$ is the number of specifications at that size. “Classes” counts discovered claim classes; “Null classes” counts those in the random comparison.

### 7.6 Filings differ more than mechanisms, over part of the range

Per-example functional agreement between circuits is 0.6028. Functional instability is therefore 0.3972. The gap between claim flip rate and functional instability is 0.3344 (95% CI 0.3297 to 0.3394), against a pre-registered threshold of 0.10. We compute agreement from per-example correct-or-incorrect verdicts using the measures of Bayat Makou et al. [1], reproduced verbatim.

Cohen’s kappa is 0.0146. The 60% raw agreement is almost entirely explained by both circuits being right about the same fraction of examples, not the same examples. At every circuit size, kappa stays between 0.016 and 0.047. Meanwhile, raw agreement rises from 0.52 to 0.92, tracking accuracy—the fraction of examples answered correctly—from 0.51 to 0.95.

This answers the phantom-specialisation objection. If discovery sampled an equivalence class of behaviourally identical subgraphs, the circuits would be functionally interchangeable. At kappa 0.015, they are not.

H4 is supported on the pre-registered pooled quantity. But the gap does not survive holding size fixed. It fails the 0.10 threshold at 6 of 10 sizes and is negative at 5. At both ends of the range, mechanisms differ more than filings. The pooled figure is carried by the 500 to 2,000 edge band alone.

We report the pooled result because it was pre-registered. We put the decomposition beside it because the pooled result alone would mislead.

## 8 The filability protocol

Filability of an interpretability claim. For a filed claim $c$ derived from a circuit:

1. Declare the specification space $\mathcal{S}$: the analytic axes and the levels of each that a competent analyst could defend. Give a citation for each level. The symbol $\mathcal{S}$ denotes the resulting set of specifications.

2. Declare the claim map $\varphi$ as deterministic, published code. Not a language model.

3. Compute $\pi^{*}=\max_c n_c/N$ over $\mathcal{S}$, and compute $F$ using Equation 2.

4. The claim is filable at tolerance $\alpha$ if $\pi^{*}\geq 1-\alpha$.

5. Report $\pi^{*}$, $F$, the discard rate per axis level, and the within-group decomposition of any pooled statistic.

A filing that cannot state its own $\pi^{*}$ has not been audited. It has been asserted. The protocol does not require the regulator to decide which analytic choice is correct. That makes it implementable: the filer need only declare the space and report how much of it agrees.

## 9 Limitations

#### One model, one task.

Larger models have more components. The layer band and size class therefore have more room, and the direction of the effect on $F$ is not obvious. This is the limitation most likely to change the number.

A second-model replication is pre-registered but not yet run. It uses a 924-cell grid on the model named Pythia-160m, described here as belonging to a different architecture family and performing the same task. Every axis is identical except seed, which has three levels rather than five.

The analysis plan, reduced grid, and criteria distinguishing replication from refutation were committed and timestamped before any cell ran. The result will be reported in a revision, whichever direction it falls. A second task remains designed and not run.

#### Circuit size is an outcome, not an axis.

The threshold rule chooses circuit size. Size has distorted three separate pooled quantities: $F$ itself, the direction of the null comparison, and the sign of the functional gap.

Fixing size drops $F$ from 0.7316 to 0.2746. That is a larger reduction than standardising any pre-registered axis (Table 6). We take this as a general hazard for multiverse designs whose specifications select objects of different sizes. We report every pooled statistic with its within-size decomposition.

#### The multiverse is as wide as the library, not the literature.

Optimal ablation [5], an alternative ablation operator, is defensible but not implemented by the instrument. It therefore never entered the space. The measured $F$ is consequently a lower bound with respect to the field’s full set of defensible choices.

#### The discard rate is high and uneven.

The discard rate is 52.3%, and the surviving pool is imbalanced across metrics (Table 2). The threshold rule was fixed in advance, and no failing item was repaired. Still, the composition is a property of the reported set. We state it rather than adjust for it.

#### The bootstrap is invalid for one reported family.

Where conditioning groups contain only two to five specifications, retained self-pairs dominate. These are pairs involving resampled copies of the same original specification. The intervals are unusable. We report those quantities as point estimates, without intervals, and attach the diagnostic.

We did not construct a bias-corrected estimator—one adjusted to reduce bias—after seeing the naive intervals disagree. Choosing an estimator in response to a result is the behaviour this paper documents.

## 10 Discussion

#### What standardising one axis buys.

Less than it appears. The metric is the largest lever, but it removes only 0.1377 from a flip rate of 0.7316 (Table 6). There is no single knob. A procedure that standardises one axis and declares the problem solved would be mistaken by a wide margin.

#### On the instrument.

One of seven documented, publicly exported discovery objectives does not run on the library’s own canonical task. All 220 cells in that arm fail (Section 6). We report this as a discard with its rate and as an observation about the maturity of tooling expected to support regulatory evidence.

It is not the paper’s headline. Nor is it a criticism of a library whose authors are among those calling for the field to address exactly this [10].

#### What a regulator can take from this.

The conclusion is not that interpretability evidence is worthless. Its evidentiary weight is a measurable property, not an assumed one. The measurement is cheap relative to the sweep that produced the circuit in the first place.

## 11 Reproducibility

The pre-registration is a timestamped commit—a saved code version—in the public repository. It is tagged `prereg-p1-confirmatory`, the identifier for the confirmatory pre-registration, and predates the confirmatory sweep.

Every reported number traces to a configuration, a seed and an environment fingerprint: a record identifying the software environment. The confirmatory sweep ran under one environment across all 1,320 cells. We verified this from the results archive rather than asserting it.

We recovered the code commit by content-addressed comparison, matching file contents in the uploaded archive against the git tree, the version-controlled file collection. The result was 22 of 22 files identical. The analysis layer runs without a graphics processing unit (GPU).

A verification script recomputes every headline quantity from raw results and checks it against the reported value: 36 checks, 0 failures. It derives $F$ by three routes, including brute-force pairwise computation.

A second script checks that every numeric token in this manuscript matches a value derived from the stored results. Every table is emitted directly from those results rather than transcribed. The test suite contains 343 tests.
