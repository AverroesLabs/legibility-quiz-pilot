# Explanation Multiplicity: Circuit-Level Interpretability

Evidence Does Not Survive Defensible Analytic Variation

Ajay Pravin Mahale

12 August 2026

###### Abstract

The European Union Artificial Intelligence Act (EU AI Act) requires providers of systems classified as high-risk to file technical documentation explaining how those systems reach decisions. Mechanistic interpretability studies how a model’s internal computations produce its behaviour. It is the obvious source of such evidence. Circuit discovery, which selects connections in a model’s computation to explain a behaviour, is its most developed instrument. We ask whether this evidence survives the condition under which it would be used: two competent analysts, the same system, the same tool, and different defensible settings.

We pre-registered a crossed grid of seven analytic axes: seven kinds of analytic choice, combined across their available settings. Every setting came from a published implementation. Pre-registration fixed the analysis plan before the confirmatory sweep. We mapped each discovered circuit through a deterministic claim map—a fixed rule that always gives the same output for the same input—to a structured statement for Annex IV, the Act’s technical-documentation annex.

The study covered 15,840 pre-registered specifications on the GPT-2 small language model and the indirect object identification task. A specification is one combination of analytic settings. Of these specifications, 7,561 produced a claim. The statement differs, or “flips,” across 73.2% of specification pairs. Its 95% confidence interval (CI), an interval quantifying statistical uncertainty, is 0.725 to 0.738. The modal claim—the most common statement—accounts for 41.1% of the space (Table 4, row 1). The evidence fails a filability criterion at every tolerance a conformity assessment body would plausibly accept. Here, filability means that a sufficiently large share of specifications supports the same claim.

Standardising the most influential single choice, the evaluation metric, leaves the flip rate at 59.4% (Table 6, “Evaluation metric”). Removing circuit size from the claim and also holding size fixed leaves 27.1% (95% CI 0.255 to 0.286), still above the pre-registered threshold (Section 7.4). The underlying circuits are structurally near-disjoint: their median pairwise Jaccard overlap, the fraction of edges shared out of all edges in either circuit, is 4% (Section 7.1). They are also functionally uncorrelated. Cohen’s kappa, an agreement measure adjusted for agreement expected from the circuits’ individual correctness rates, is 0.015 (Section 7.6). Thus, the instability is not one mechanism described in different words.

We give the filability criterion as a standalone protocol. We also report that one of the seven documented discovery objectives—the scoring targets used to discover circuits—does not execute on the library’s own canonical task. The study covers one model and one task. Whether the conclusion holds at scale is untested.

## 1 Introduction

An audit means two auditors reach the same conclusion. That is not a definition borrowed from machine learning; it is what makes an audit worth commissioning.

Regulation (EU) 2024/1689 requires providers of high-risk AI systems to file technical documentation. Annex IV point 2(b) asks for “the general logic of the AI system and of the algorithms”. Point 2(e) asks for “an assessment of the technical measures needed to facilitate the interpretation of the outputs”. Article 86(1) gives a person subject to certain automated decisions the right to “clear and meaningful explanations of the role of the AI system in the decision-making procedure and the main elements of the decision taken”. These quotations come from the consolidated text as amended by Regulation (EU) 2026/1744, CELEX 02024R1689-20260727, read 11 August 2026. CELEX is the legal text’s document identifier.

Mechanistic interpretability offers to supply this technical content. Circuit discovery produces a subgraph—a selected part of the model’s computational graph—said to explain a behaviour. A computational graph represents computational components and the connections between them. Circuit discovery is the part of the field with mature tooling. If a provider files a circuit-level account of why its system behaved as it did, that filing is worth something only if a second analyst, given the same system and tool, would file something compatible.

We measure how far a regulatory claim moves across defensible analytic specifications. We also attach a decision rule that a standards body could use. We do not propose a better circuit discovery method. We do not argue that interpretability is worthless. We argue that the evidence, as currently produced, does not meet the standard that filing it presumes.

#### Contributions.

(i) A pre-registered multiverse: an analysis across combinations of seven analytic axes of circuit discovery. Every level, or available setting, comes from a published implementation. A deterministic map converts circuits into Annex IV statements.

(ii) A filability criterion, $\pi^{*}\geq 1-\alpha$, and its measured value on a canonical task. Here, $\pi^{*}$ is the share of specifications supporting the most common claim. The tolerance $\alpha$ is the share allowed not to support that claim.

(iii) Evidence that instability survives the most aggressive available controls: standardising any single axis, removing circuit size from the claim, and holding size fixed.

(iv) A methodological result for multiverse designs generally. When specifications select objects of different sizes, pooled statistics—statistics calculated across those sizes together—can invert. We show three such cases.

## 2 Related work

#### Non-identifiability.

Non-identifiability means that the behaviour and explanatory criteria do not determine a unique explanation. Méloux et al. [6] ask whether a unique explanation exists for a given behaviour under mechanistic interpretability’s own criteria. They answer that it need not. Our premise is theirs. Our contribution comes afterward: we measure what non-identifiability costs when an explanation is filed rather than published.

#### Faithfulness metric sensitivity.

A faithfulness metric measures how well a selected circuit accounts for the model’s behaviour. Ablation tests this by removing or replacing parts of the computation. Miller et al. [8] survey choices in ablation-based faithfulness measurement. They find existing methods highly sensitive to seemingly insignificant changes. They conclude that faithfulness scores reflect the researcher’s methodological choices as well as the circuit.

They establish that one axis matters. We cross seven axes and carry the resulting variation through to a regulatory statement. Their released software library is our measurement instrument. Our study therefore stress-tests the tooling using its own definitions.

#### Structure versus function.

Bayat Makou et al. [1] vary input statistics while holding the task fixed. They find that structurally distinct circuits can implement the same computation. They call this pattern phantom specialisation.

This is the strongest objection to our design. Discovery might merely sample an equivalence class: a set of subgraphs that behave identically despite having different connections. If so, a claim flip rate would measure which member was sampled, not a functional difference. We address this objection with a functional measurement in Section 7.6. Our circuits are not functionally interchangeable.

#### Variance as estimation.

Méloux et al. [7] treat mechanistic interpretability as statistical estimation and analyse variation in the resulting estimator. A seed is a value that controls a sampling procedure. Variation across seeds is one axis of our space. Their framing is the natural statistical companion to our analysis of variation across whole specifications.

#### Auditability.

Lan et al. [4] observe that two papers reached conflicting conclusions about the same behaviour. A third found both partly correct and incomparable. They call for auditing standards so that findings can be certified in safety-critical settings.

We answer that call with a measurement and a criterion rather than a proposal. Sharkey et al. [10] list open problems, including the socio-technical problems—the problems involving both technical methods and their social use—that this paper touches. Mueller et al. [9] argue that comparison across methods requires standardised evaluation. That argument is adjacent to ours: certification requires standardised specification.

#### Multiverse method.

The design follows Steegen et al. [13], who propose reporting results across defensible data-processing choices. It also follows Simonsohn et al. [12], whose specification curve—a plot of results across specifications—we adapt.

Simmons et al. [11] named the underlying problem: undisclosed analytic flexibility lets an analyst present almost anything as significant. We apply this psychology methods literature to a machine learning evidentiary question. The transfer is not decorative. A claim that an explanation is stable is exactly the kind of claim that analytic flexibility can manufacture.

#### Explanation metrics under optimisation pressure.

Hsia et al. [3] show that sufficiency and comprehensiveness can be inflated without changing predictions or explanations. Sufficiency measures how well the retained explanation preserves behaviour. Comprehensiveness measures how much behaviour changes when the explanation is removed. These are two of our four evaluation metrics. This is one reason we report the discard rate separately for each metric rather than silently pooling them.

## 3 What this paper is not claiming

We state these limits early. Each rules out a reading that the results do not support. We held two of those readings ourselves before running the decompositions, which separate pooled results into their component groups.

We are not claiming that discovered circuits are no better than random. At fixed circuit size, their claim flip rate is lower than that of a size-matched random null—a comparison set of randomly selected circuits with the same sizes: 0.2746 against 0.4230 (Section 7.5). They carry information.

Nor are we claiming that filings differ where mechanisms do not. Mechanisms differ a great deal. Functional instability, the rate at which circuits disagree on whether examples are answered correctly, is 0.3972 pooled and 0.4777 at the smallest circuits (Section 7.6).

Nor are we claiming that discovery samples one mechanism and dresses it differently. Cohen’s kappa of 0.0146 says that the circuits are functionally uncorrelated (Section 7.6).

Nor are we claiming that random seeds are the problem. Here, the seed selects which prompts—the model’s input texts—are sampled. This is variation in the evaluation set, not nondeterminism in execution.

And we claim nothing at all about scale. One model, one task.

## 4 Formalism

A specification is a tuple, or ordered collection, of analytic choices:

\[
s=(o,a,d,m,\tau,P,r,g).
\]

The entries are:

- $o$: the discovery objective, which defines what discovery scores.
- $a$: the ablation operator, which specifies how parts of the computation are removed or replaced.
- $d$: the corruption distribution, which supplies altered inputs for operators that use them.
- $m$: the evaluation metric, which scores circuit behaviour.
- $\tau$: the size threshold used in selecting a circuit.
- $P$: the prompt variant.
- $r$: the seed that selects sampled prompts.
- $g$: the granularity, or level of detail, of the claim.

Every level of every axis comes from a published implementation. None was invented for this study.

Circuit discovery returns $C(s)\subseteq E$. Here, $C(s)$ is the set of edges selected under specification $s$, $E$ is the model’s full edge set, and $\subseteq$ means “is a subset of.” Edges are connections in the computational graph. Vertical bars around a set denote its number of members.

For GPT-2 small, the factorised graph represents separate query, key and value inputs to attention. These are the three input roles through which attention computes and uses relationships among input positions. This graph has $|E|=32{,}491$ edges.

We confirm that count from the instrument and independently from a closed-form count over the computation order. A closed-form count calculates the total directly rather than enumerating edges. Let $b$ index a model block. Attention in block $b$ receives from $1+13b$ sources at three input slots for each of 12 attention heads. An attention head is an attention-processing component. The multilayer perceptron (MLP), the block’s feed-forward component, receives from those sources plus its own block’s heads. The output receives from all 157 non-terminal sources, meaning sources before the final output. Circuits touch components drawn from 144 attention heads and 12 MLPs, or 156 components in total.

#### Circuit instability.

For two specifications $s_i$ and $s_j$, write their circuits as $C_i=C(s_i)$ and $C_j=C(s_j)$. Their Jaccard distance is

\[
D(s_i,s_j)=1-\frac{|C_i\cap C_j|}{|C_i\cup C_j|}.
\tag{1}
\]

Here, $\cap$ means the shared edges and $\cup$ means all edges present in either circuit. The fraction is Jaccard similarity. Distance $D$ is one minus that similarity. We use $\bar{J}$ for the mean Jaccard similarity over unordered pairs, counting each pair only once.

#### Claim instability.

Let $\varphi$ be the claim map: the rule that converts a circuit into a structured Annex IV statement. The flip rate $F$ is the probability that two distinct specifications yield different statements:

\[
F=1-\frac{\sum_c n_c(n_c-1)}{N(N-1)}.
\tag{2}
\]

Here, $c$ labels a claim class, meaning a distinct structured statement. The count $n_c$ is the number of specifications yielding class $c$. The total $N$ is the number of specifications being compared, and $\sum_c$ means to sum over claim classes. For claim calculations in the realised study, these are the specifications that produced claims.

Equation 2 is the unbiased Gini–Simpson form: it gives the probability that two draws without replacement differ. “Without replacement” means that the same specification cannot be drawn twice. We unit-test it against a brute-force calculation that directly compares every pair in randomised test cases. The modal share is

\[
\pi^{*}=\max_c n_c/N,
\]

where $\max_c$ selects the largest claim-class count.

We also use a within-group form. Let $G$ be a set of conditioning axes: choices whose levels must match within each comparison pair. Then

\[
F_{\text{within}}(G)
=
1-
\frac{\sum_g\sum_c n_{g,c}(n_{g,c}-1)}
{\sum_g n_g(n_g-1)}.
\tag{3}
\]

In this equation, $g$ labels a group, rather than claim granularity. The count $n_{g,c}$ is the number of specifications in group $g$ yielding claim $c$, and $n_g$ is the total number in that group. Equation 3 weights every eligible pair equally, not every group equally. With no conditioning axes—when $G$ is empty—it reduces exactly to Equation 2.

#### Filability.

Evidence is filable at tolerance $\alpha$ if and only if

\[
\pi^{*}\geq 1-\alpha.
\]

Thus, the most common claim must cover at least the required share $1-\alpha$ of specifications.

There is a ceiling: $F\leq 1-1/k$, where $k$ is the number of claim classes. A flip rate must therefore be read against the number of classes the map can produce. With the nine classes observed here, the ceiling is 0.8889. The observed 0.7316 is 82.3% of that ceiling. $F$ is not near one by construction.

## 5 The claim map

The claim map $\varphi$ is the paper’s most attackable choice. We therefore fixed it in the pre-registration, implemented it as deterministic code, and published it. A language model does not generate the claims. Using one would add model variation to circuit variation without a way to separate them.

There are two addressees, following the regulation rather than convenience:

- $\varphi_{\text{overseer}}$ addresses an overseer. It is built on Annex IV 2(e) and 3, with Article 14(4)(c). It keys the statement first on the dominant layer band—the early, middle or late part of the model most represented by the circuit—then on size class, then on the ranked input segment, or portion of the prompt.
- $\varphi_{\text{affected}}$ addresses the person affected by the decision. It is built on Article 86(1). It leads with the input segment because the right is owed to the person the decision is about.

Three granularities, labelled COARSE, MEDIUM and FINE, are nested by construction. A finer claim adds detail to the coarser claim rather than contradicting it.

A reviewer will say that we chose the size bins—ranges used to assign size classes—to produce the reported flip rate. We did not choose them directly. A committed calibration rule returned them from a measured node-count curve, which records how many computational components selected circuits touch. Table 1 reports sensitivity to other binnings.

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

“Deciles” divides the size distribution into ten equal-frequency groups.

## 6 Method

#### Model and task.

We use GPT-2 small on indirect object identification [15], a task requiring the model to identify the indirect object in a prompt. The choice is not an apology. The non-identifiability and faithfulness literature we extend works at this scale. Comparability is the point. The edge count also makes a size-matched random baseline meaningful.

#### Grid.

The pre-registered grid has seven discovery objectives, seven ablation operators, four corruption distributions nested within the five operators that read them, four metrics, three thresholds, two prompt variants and five seeds. This gives 1,540 discovery cells and 18,480 specifications. A discovery cell is a combination of settings used to discover and evaluate circuits under the remaining specification choices.

Corruption is nested rather than fully crossed. That means its four levels are varied only for operators that use them. Two ablation operators never read the corruption distribution. Crossing those operators with corruption levels would produce duplicate specifications.

Discovery uses the edge attribution patching family [14] and integrated edge gradients, as implemented in the library released with Miller et al. [8]. These methods score edges by their estimated contributions to the discovery objective; integrated edge gradients combine gradient information along an integration path. The automated circuit discovery lineage is Conmy et al. [2]. The four corruption levels come from Wang et al. [15].

#### Pre-registration.

We committed and tagged the analysis plan before the confirmatory sweep began. Both abstracts were drafted in advance, so neither result direction could be written up as a surprise. Deviations are recorded in an append-only file. Each entry gives its date and states whether the decision came before or after the affected result was seen.

#### Statistics.

We compute no $p$-value across specifications. A $p$-value is a significance-test quantity, and specifications here are a designed grid, not an independent sample [13, 12].

Uncertainty comes from a nonparametric bootstrap: repeated resampling of the observed specifications, without assuming a parametric distribution. We resample specifications, never pairs, using $B=10{,}000$ bootstrap resamples. The quantities $F$ and $\bar{J}$ are U-statistics, meaning statistics formed from comparisons over pairs. Pairs sharing a specification are dependent. Resampling pairs would ignore that dependence. The intervals quantify uncertainty given this grid. They license nothing beyond it.

#### Two exclusions, criteria fixed before any circuit was seen.

One discovery objective does not execute. All 220 of its cells raise a runtime error inside the library’s own code. The error comes from computing mean squared error—the average squared difference from a target—against an integral, or integer-valued, target.

We verified the failure set by computing a digest, a content-based identifier, over the sorted failing cell identifiers. The set is exactly that objective’s arm and nothing else. We did not repair the error. The library is the instrument under test and is reproduced verbatim. The realised grid therefore has six objectives, 1,320 cells and 15,840 specifications.

The metric-relative threshold rule, which selects a circuit according to a threshold tied to its evaluation metric, admits no circuit for 8,279 of 15,840 specifications. The discard rate is 52.3%, and it differs sharply by metric (Table 2). This was disclosed before the pre-registration was locked. The surviving pool is therefore 52% comprehensiveness, against 25% by design. We report this composition rather than smoothing it.

Table 2: Discard rate by evaluation metric under the pre-registered metric-relative threshold rule. The unevenness was disclosed before the pre-registration was locked, from a six-cell validation slice.

| Metric | Kept | Discarded | Rate |
|---|---:|---:|---:|
| comprehensiveness | 3,947 | 13 | 0.003 |
| KL divergence | 1,467 | 2,493 | 0.630 |
| logit difference | 1,270 | 2,690 | 0.679 |
| sufficiency | 877 | 3,083 | 0.779 |

Kullback–Leibler (KL) divergence measures a difference between output probability distributions. A logit is an output score before conversion to probabilities; logit difference compares the scores of the relevant outputs.

## 7 Results

Table 3 gives each pre-registered outcome and its result. The decision rules and thresholds were fixed and tagged before the confirmatory sweep. None was revisited.

Table 3: Pre-registered outcomes and their results. Thresholds were fixed on 6 August 2026 and the plan was tagged before the sweep began. Two results carry qualifications that only the within-size decomposition reveals, and those are reported in Sections 7.5 and 7.6 rather than deferred to an appendix.

| | Decision rule | Result | Outcome |
|---|---|---|---|
| P0 | $\bar{J}$ reported, not tested | 0.1396 [0.1377, 0.1418] | premise holds |
| H2 | $F>0.20$, CI lower bound above | 0.7316 [0.7247, 0.7380] | confirmed |
| H3 | not separated from random null | separated, intervals disjoint | rejected |
| H4 | $F-(1-\text{agreement})>0.10$ | 0.3344 [0.3297, 0.3394] | supported |

P0 labels the premise about circuit overlap. H2, H3 and H4 label the pre-registered hypotheses tested by the listed rules. Bracketed ranges are confidence intervals. In H4, “agreement” is the fraction of examples for which two circuits receive the same correct-or-incorrect verdict. Thus, $1-\text{agreement}$ is functional instability.

### 7.1 Circuits are almost disjoint

Mean Jaccard similarity is $\bar{J}=0.1396$ (95% CI 0.1377 to 0.1418). We compute it exactly over all 28,580,580 specification pairs, rather than sampling pairs. The 7,561 specifications contain 3,218 distinct circuits. Repeated circuits make the exact computation tractable.

The distribution says more than the mean. A quantile gives the value below which the stated fraction of observations falls. For pairwise distance $D$, the 1% quantile is 0.196, the 25% quantile is 0.788, the median is 0.960, and the 75% quantile is 0.998. The median is the middle value. Thus, the median pair shares 4% of its edges, and only 1% of pairs exceed a Jaccard similarity of 0.8.

At the median circuit size of 200 edges, the analytic random line

\[
\frac{k}{2|E|-k}
\]

gives 0.0031. Here, $k$ denotes circuit size, rather than the number of claim classes used in Section 4. Observed mean overlap is about 45 times chance. Beating chance at this scale is not evidence of much: a 500-edge circuit has expected random overlap of 0.008.

### 7.2 The claim flips on three quarters of pairs

The claim flip rate is $F=0.7316$ (95% CI 0.7247 to 0.7380). The pre-registered rule requires $F$ to exceed 0.20 and the interval’s lower bound to clear that threshold. It clears by more than three times (Table 3, H2).

The modal share is $\pi^{*}=0.4109$ (Table 4, row 1). The evidence is therefore not filable at any pre-registered tolerance: 0.411 against required modal shares of 0.95, 0.90 and 0.80.

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

“Sparse,” “moderate” and “distributed” are the claim map’s circuit-size classes.

The two most common claims account for 41.1% and 28.5% of the space. One attributes the same model’s behaviour on the same task to early layers; the other attributes it to late layers (Table 4, rows 1 and 2). These are not different emphases. They contradict.

All six addressee–granularity combinations reject filability (Table 5). The coarsest statement available to an affected person has only two classes, yet still flips on 39.5% of pairs (Table 5, “affected / COARSE”). The result is therefore not an artefact of fine-grained claim language. Figure 1 shows the specification curve. The width of its modal step is $\pi^{*}$.

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

Here, “confounded by circuit size” means that pooling different circuit sizes changes the comparison, as the within-size results in Section 7.5 show.

### 7.3 No analytic axis rescues the filing

For each axis, we compare only pairs whose specifications use the same level of that axis. We then pool those eligible pairs with Equation 3. Table 6 reports the remaining, or residual, flip rate.

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

“Removes” is the reduction from the pooled flip rate. Bootstrap bias is the difference between the bootstrap’s average estimate and the observed estimate.

The metric is the largest single lever, and it is not enough. Standardising it completely, which is more than a conformity procedure could demand, leaves $F=0.5939$. The interval’s lower bound is 0.5803, still far above the threshold (Table 6, “Evaluation metric”).

Different axes govern structure and claims. We use restricted maximum likelihood (REML), a method for estimating variance components, to decompose variation in $\log_{10}$ of selected circuit size. The expression $\log_{10}$ is the base-ten logarithm. This decomposition attributes:

- 78.8% of structural variance to the ablation–corruption interaction, their combined effect beyond separate contributions;
- 11.5% to residual variation, the variation left over;
- 4.4% to ablation;
- 4.1% to the threshold;
- 0.7% to the metric.

Yet the metric is the largest driver of claim variance (Table 6). A standards body reading only the structural decomposition would standardise the ablation operator and fix almost none of the filing instability.

### 7.4 The result under maximum pressure

We remove every trace of circuit size from the claim. The statement then identifies only the dominant layer band. We also hold circuit size fixed by comparing only same-size pairs with Equation 3.

The resulting flip rate is $F=0.2706$ (95% CI 0.2550 to 0.2859). The interval’s lower bound remains above the pre-registered threshold. Even when size cannot contribute, the model location to which behaviour is attributed still flips on 27% of pairs.

### 7.5 The random baseline, and why its pooled direction misleads

In the pooled comparison, the discovered multiverse is less stable than a size-matched random null drawn from the full 32,491-edge population. We use $R=1{,}000$ null multiverses, where $R$ is the number of repeated random comparison sets. The flip rates are 0.7316 for discovered circuits and 0.6774 for the null. Their intervals are disjoint.

H3 is therefore rejected: the distributions are separated (Table 3, H3). We do not report the direction of that separation as a finding. It is an artefact of the size distribution, as Table 7 shows.

The null is degenerate above 1,000 edges: its claims collapse to a single class. At those sizes, a uniform random draw touches all 156 components and can produce only one claim. With size held fixed, the ordering reverses. At medium granularity, the discovered flip rate is 0.2746 against the null’s 0.4230. At coarse granularity, it is 0.2706 against 0.3822. Discovered circuits are more stable than random circuits once the comparison is fair. They are still not stable enough to file.

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

“Size” is the number of selected edges. The count $n$ is the number of specifications at that size. “Classes” is the number of distinct discovered claim classes; “Null classes” is the corresponding number for the random comparison.

### 7.6 Filings differ more than mechanisms, over part of the range

Per-example functional agreement between circuits is 0.6028. Agreement means that both circuits receive the same correct-or-incorrect verdict on an example. Functional instability is therefore 0.3972. The gap between the claim flip rate and functional instability is 0.3344 (95% CI 0.3297 to 0.3394), against a pre-registered threshold of 0.10 (Table 3, H4). We use the per-example correctness measures of Bayat Makou et al. [1], reproduced verbatim.

Cohen’s kappa is 0.0146. The 60% raw agreement is almost entirely explained by both circuits being right about the same fraction of examples, rather than the same examples. At every circuit size, kappa stays between 0.016 and 0.047. Meanwhile, raw agreement climbs from 0.52 to 0.92, tracking accuracy—the fraction of examples answered correctly—from 0.51 to 0.95.

This answers the phantom specialisation objection. If discovery sampled behaviourally identical subgraphs, the circuits would be functionally interchangeable. At kappa 0.015, they are not.

H4 is supported on the pre-registered quantity. But the gap does not survive holding size fixed. It fails the 0.10 threshold at 6 of 10 sizes and is negative at 5. At both ends of the size range, mechanisms differ more than filings do. The pooled figure is carried by the 500 to 2,000 edge band alone.

We report the pooled result because it was pre-registered. We report the decomposition beside it because the pooled result alone would mislead.

## 8 The filability protocol

Filability of an interpretability claim. For a filed claim $c$ derived from a circuit:

1. Declare the specification space $\mathcal{S}$: the set of analytic axes and levels that a competent analyst could defend. Give a citation for each level.

2. Declare the claim map $\varphi$ as deterministic, published code. Not a language model.

3. Compute $\pi^{*}=\max_c n_c/N$ over $\mathcal{S}$, and compute $F$ using Equation 2. Here, $n_c$ counts specifications yielding claim $c$, and $N$ is the total number compared.

4. The claim is filable at tolerance $\alpha$ if $\pi^{*}\geq 1-\alpha$.

5. Report $\pi^{*}$, $F$, the discard rate for each axis level, and the within-group decomposition of every pooled statistic.

A filing that cannot state its own $\pi^{*}$ has not been audited. It has been asserted. The protocol does not require a regulator to decide which analytic choice is correct. That is what makes it implementable. It requires only that the filer declare the space and report how much of it agrees.

## 9 Limitations

#### One model, one task.

Larger models have more components. Both the layer band and the size class therefore have more room, and the direction of the effect on $F$ is not obvious. This limitation is the one most likely to change the number.

A second-model replication is pre-registered but not yet run. It uses a 924-cell grid on Pythia-160m, a model from a different architecture family that performs the same task. The grid is identical on every axis except the seed axis, which has three levels rather than five.

The analysis plan, reduced grid, and criteria for replication rather than refutation were committed and timestamped before any cell ran. The replication will be reported in a revision whichever direction it takes. A second task remains designed and not run.

#### Circuit size is an outcome, not an axis.

The threshold rule chooses circuit size. Size has distorted three separate pooled quantities: $F$ itself, the direction of the null comparison, and the sign of the functional gap.

Holding size fixed drops $F$ from 0.7316 to 0.2746 (Section 7.5). This reduction is larger than that from standardising any pre-registered axis (Table 6). We take this to be a general hazard for multiverse designs whose specifications select objects of different sizes. We report every pooled statistic with its within-size decomposition.

#### The multiverse is as wide as the library, not the literature.

Optimal ablation [5], an ablation method, is a defensible operator that the instrument does not implement. It therefore never entered the space. The measured $F$ is thus a lower bound with respect to the field’s full set of defensible choices.

#### The discard rate is high and uneven.

The discard rate is 52.3%, and the surviving pool is imbalanced across metrics (Table 2). The threshold rule was fixed in advance, and we repaired no failing item. Still, this composition is a property of the reported set. We state it rather than adjust for it.

#### The bootstrap is invalid for one reported family.

Where conditioning groups contain only two to five specifications, retained self-pairs dominate and the intervals are unusable. Self-pairs arise when resampling retains repeated copies of the same specification. We report those quantities as point estimates, without intervals, and attach the diagnostic explaining the problem.

We did not construct a bias-corrected estimator after seeing the naive intervals disagree. Choosing an estimator in response to a result is the behaviour this paper documents.

## 10 Discussion

#### What standardising one axis buys.

Less than it appears. The metric is the biggest lever. It removes 0.1377 from a flip rate of 0.7316 (Table 6, “Evaluation metric”). There is no single knob. A procedure that standardises one axis and declares the problem solved would be mistaken by a wide margin.

#### On the instrument.

One of seven documented, publicly exported discovery objectives does not run on the library’s own canonical task. “Publicly exported” means that the library exposes it for users to call. We report this as a discard with its rate, and as an observation about the maturity of the tooling on which regulatory evidence is expected to rest (Section 6, “Two exclusions, criteria fixed before any circuit was seen”).

It is not the paper’s headline. Nor is it a criticism of a library whose authors are among those calling for the field to address exactly this [10].

#### What a regulator can take from this.

The conclusion is not that interpretability evidence is worthless. Its evidentiary weight is a measurable property, not an assumed one. The measurement is cheap relative to the sweep that produced the circuit in the first place.

## 11 Reproducibility

The pre-registration is a timestamped code commit—a recorded repository version—tagged `prereg-p1-confirmatory` in the public repository. It was made before the confirmatory sweep began. Every reported number traces to a configuration, a seed and an environment fingerprint, which identifies the execution environment.

The confirmatory sweep ran under a single environment across all 1,320 cells. We verified this from the results archive rather than merely asserting it. We recovered the code commit by content-addressed comparison: matching the uploaded archive’s file contents against the git tree, the version-controlled set of files. All 22 of 22 files were identical. The analysis layer runs without a graphics processing unit (GPU).

A verification script recomputes every headline quantity from raw results and checks it against the reported value: 36 checks, 0 failures. It derives $F$ by three routes, including a brute-force pairwise computation.

A second script checks that every numeric token in this manuscript matches a value derived from stored results. Every table is emitted directly from those results rather than transcribed. The test suite contains 343 tests.
