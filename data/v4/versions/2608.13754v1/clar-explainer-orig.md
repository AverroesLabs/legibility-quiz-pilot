**Summary.** Can circuit discovery—finding a model subgraph said to explain a behaviour—produce stable regulatory evidence? The paper varies seven analytic choices and deterministically maps each circuit to a structured Annex IV statement. On GPT-2 small and indirect object identification, 7,561 of 15,840 specifications produce a claim. The claim flip rate—the probability that two distinct specifications produce different statements—is 73.2%. The most common claim covers 41.1% of specifications. Standardising the evaluation metric leaves a 59.4% flip rate. Removing circuit size from the claim and holding size fixed leaves 27.1%. Median circuit overlap is 4%, and functional agreement adjusted for chance is near zero. One documented discovery objective does not execute. The study covers one model and one task; whether its conclusion holds at scale is untested.

## Glossary

*This glossary defines the paper’s terms, abbreviations, symbols and codes by their use in the paper.*

- **Ablation; ablation operator** — An analytic choice used in circuit discovery and faithfulness measurement.
- **Ablation-corruption interaction** — The joint ablation and corruption term in the structural variance decomposition.
- **Accuracy** — The fraction of examples on which a circuit gives a correct verdict.
- **Addressee** — Either the overseer or the affected person for whom a claim is constructed.
- **Affected person; affected** — The person the decision is about, addressed by the Article 86(1) claim map.
- **Analytic axis; analytic level** — A choice in the specification space, and one available setting of that choice.
- **Annex IV statement; claim** — A structured regulatory statement obtained from a circuit through the claim map.
- **Audit; auditability** — An audit means two auditors reach the same conclusion.
- **Automated circuit discovery** — The discovery lineage attributed to Conmy et al. [2].
- **Behaviour** — The model behaviour that a discovered circuit is said to explain.
- **Bootstrap; nonparametric bootstrap** — Resampling specifications, not pairs, to quantify uncertainty given the grid.
- **Calibration rule** — The committed rule that returned size bins from a measured node-count curve.
- **Canonical task** — The library’s own task on which the study tests its discovery objectives.
- **Cell; discovery cell** — A discovery configuration in the pre-registered grid.
- **Chance; random line** — The analytic random-overlap comparison \(k/(2|E|-k)\).
- **Circuit** — A discovered subset of the model’s edges, said to explain a behaviour.
- **Circuit discovery** — The procedure that returns a circuit for a specification.
- **Circuit instability** — One minus the Jaccard similarity of two circuits.
- **Claim class** — A class of specifications yielding the same structured statement.
- **Claim flip rate; claim instability** — The probability that two distinct specifications yield different statements.
- **Claim map** — Deterministic, published code mapping a circuit to a structured Annex IV statement.
- **Closed-form count** — The independent edge count based on the model’s computation order.
- **COARSE; MEDIUM; FINE** — The three nested claim granularities.
- **Cohen’s kappa; kappa** — The functional-agreement measure used to distinguish raw agreement from agreement explained by correctness rates.
- **Component** — An attention head or MLP touched by a circuit.
- **Comprehensiveness** — One of the four evaluation metrics.
- **Config** — The configuration to which a reported number can be traced.
- **Conformity assessment body; conformity procedure** — The body or procedure applying a filability tolerance.
- **Corruption distribution; corruption** — A specification choice read by five of the seven ablation operators.
- **Decision rule; threshold** — A pre-registered condition for judging an outcome.
- **Deterministic** — Fixed code, rather than language-model generation, for mapping circuits to claims.
- **Discard rate** — The proportion of specifications for which the metric-relative threshold rule admits no circuit.
- **Discovery objective** — The objective used by circuit discovery; seven were documented and six executed.
- **Distributed; moderate; sparse** — The size-class labels used in the claim map.
- **Dominant layer band** — The early, middle or late layer attribution used by the claim map.
- **Edge attribution patching; integrated edge gradients** — The discovery methods used from the library released with Miller et al. [8].
- **Environment fingerprint** — The environment record associated with a reported result.
- **Equivalence class** — A set of behaviourally identical subgraphs in the phantom-specialisation objection.
- **EU AI Act; Regulation (EU) 2024/1689** — The regulation requiring technical documentation for high-risk AI systems.
- **Regulation (EU) 2026/1744** — The amendment identified for the consolidated regulatory text.
- **CELEX 02024R1689-20260727** — The identifier of the consolidated text read on 11 August 2026.
- **Annex IV 2(b)** — The provision asking for the system’s and algorithms’ “general logic”.
- **Annex IV 2(e)** — The provision asking for assessment of measures facilitating output interpretation.
- **Annex IV 3; Article 14(4)(c)** — Provisions used with Annex IV 2(e) for the overseer claim map.
- **Article 86(1)** — The provision giving certain affected persons a right to clear and meaningful explanations.
- **Evaluation metric; metric** — The criterion used to evaluate a circuit.
- **Evaluation-set variability** — Variation from the seed selecting which prompts are sampled.
- **Faithfulness** — The circuit property whose ablation-based measurement is reviewed by Miller et al. [8].
- **Factorised graph** — The GPT-2 small graph with separate query, key and value inputs.
- **Filability; filable** — Evidence is filable at tolerance \(\alpha\) exactly when \(\pi^{*}\geq1-\alpha\).
- **Functional agreement** — Agreement between circuits’ per-example correct or incorrect verdicts.
- **Functional instability** — One minus functional agreement.
- **Functional gap** — Claim flip rate minus functional instability.
- **GPT-2 small** — The model used in the confirmatory study.
- **GPU** — The computing resource not required by the analysis layer.
- **Granularity** — One of three nested levels at which the claim map produces a statement.
- **Grid; multiverse; specification space** — The designed set of defensible analytic specifications.
- **H2** — The pre-registered outcome requiring \(F>0.20\), with the confidence-interval lower bound above that threshold.
- **H3** — The pre-registered outcome that the result is not separated from the random null.
- **H4** — The pre-registered outcome requiring \(F-(1-\text{agreement})>0.10\).
- **Indirect object identification** — The task studied on GPT-2 small [15].
- **Input segment** — The ranked input portion used by the claim map and prioritised for the affected person.
- **Interval; CI; 95% CI** — The reported uncertainty interval, obtained by bootstrap and conditional on the grid.
- **Jaccard similarity; overlap** — Shared circuit edges divided by the edges in either circuit.
- **KL divergence; logit difference; sufficiency** — The other three evaluation metrics.
- **Layer band; early layers; middle layers; late layers** — The three layer-attribution categories used in claims.
- **Mechanistic interpretability** — The field offering circuit-level technical accounts of model behaviour.
- **MLP** — One of the model component types counted alongside attention heads.
- **Modal claim; modal share** — The most common claim and its share of specifications.
- **Node-count curve** — The measured curve from which the calibration rule returned size bins.
- **Non-identifiability** — The possibility that a behaviour has no unique explanation under mechanistic interpretability’s criteria.
- **Null; size-matched random null** — Random circuits drawn from the full edge population with circuit sizes matched.
- **Optimal ablation** — A defensible operator [5] not implemented by the instrument.
- **Overseer** — The addressee of the claim map built on Annex IV 2(e) and 3 with Article 14(4)(c).
- **P0** — The pre-registered premise for which mean Jaccard similarity is reported, not tested.
- **Phantom specialisation** — Structurally distinct circuits implementing the same computation [1].
- **Pooled statistic** — A statistic combining specifications across circuit sizes.
- **Pre-registration; confirmatory sweep** — The committed analysis plan and the subsequent grid execution.
- **prereg-p1-confirmatory** — The public-repository tag for the timestamped pre-registration commit.
- **Prompt variant** — One of the two prompt settings in the grid.
- **Pythia-160m** — The different-architecture model named for the unrun replication.
- **Quantile** — A reported position in the distribution of pairwise circuit instability.
- **REML variance decomposition** — The decomposition applied to \(\log_{10}\) selected circuit size.
- **Seed** — The choice selecting which prompts are sampled, not nondeterminism.
- **Self-pair** — A retained bootstrap pair that dominates in conditioning groups of two to five specifications.
- **Size binning; size class** — The thresholds and labels that classify selected circuit size for the claim map.
- **Specification** — A tuple of analytic choices.
- **Specification curve** — The visualisation adapted from Simonsohn et al. [12].
- **Structural variance** — Variation in \(\log_{10}\) selected circuit size.
- **Technical documentation** — The documentation required from providers of high-risk systems.
- **Threshold rule; metric-relative threshold rule** — The rule that selects circuit size or admits no circuit.
- **Tolerance** — The permitted disagreement level in the filability criterion.
- **U-statistic** — The pair-based statistical form of \(F\) and \(\bar J\), whose pairs share specifications.
- **Within-group; within-size decomposition** — A comparison restricted to matching conditioning groups or circuit sizes.
- **\(s; s_i,s_j\)** — A specification, or two indexed specifications.
- **\(o,a,d,m,\tau,P,r,g\)** — Respectively objective, ablation operator, corruption distribution, metric, size threshold, prompt variant, seed and granularity.
- **\(C(s); C_i,C_j\)** — The circuit returned for a specification, or two indexed circuits.
- **\(E; |E|\)** — The model’s edge set and its edge count.
- **\(b\)** — The block index in the independent edge count.
- **\(D(s_i,s_j)\)** — One minus the Jaccard similarity of two circuits.
- **\(\bar J\)** — Mean Jaccard similarity over unordered specification pairs.
- **\(F\)** — Claim flip rate.
- **\(c; n_c\)** — A claim class and the number of specifications yielding it.
- **\(N\)** — The number of specifications.
- **\(\pi^{*}\)** — Modal claim share, \(\max_c n_c/N\).
- **\(G\)** — The set of conditioning axes.
- **\(g; n_g; n_{g,c}\)** — In Equation 3, a conditioning group, its specification count, and its count for claim class \(c\).
- **\(F_{\text{within}}(G)\)** — Pair-weighted claim flip rate within conditioning groups.
- **\(\alpha\)** — Filability tolerance.
- **\(k\)** — The number of claim classes in the stated flip-rate ceiling; circuit size in the random-overlap formula.
- **\(\varphi\)** — The deterministic circuit-to-claim map.
- **\(\varphi_{\text{overseer}};\varphi_{\text{affected}}\)** — The maps for the overseer and the affected person.
- **\(\mathcal S\)** — The declared specification space in the filability protocol.
- **\(B=10{,}000\)** — The number of bootstrap resamples.
- **\(R=1{,}000\)** — The number of random-null multiverses.
- **\(p\)-value** — A quantity the paper does not compute across specifications.
- **\(\log_{10}\)** — The transformation of selected circuit size used in the variance decomposition.
- **\(\subseteq,\cap,\cup,|\cdot|\)** — Respectively subset, shared edges, edges in either set, and set count in the circuit formulas.
- **\(\sum,\max,\times\)** — Respectively summation, maximum and multiplication in the reported formulas and binnings.

# Explanation Multiplicity: Circuit-Level Interpretability

Evidence Does Not Survive Defensible Analytic Variation

Ajay Pravin Mahale

12 August 2026

###### Abstract

*The study tests whether circuit-derived regulatory statements survive defensible changes in analytic settings.*

The EU AI Act, the regulation requiring technical documentation from providers of high-risk systems, asks providers to describe how a system reaches decisions. Mechanistic interpretability, which offers technical accounts of model behaviour, is the obvious source of such evidence. Circuit discovery, which finds a subgraph said to explain a behaviour, is its most developed instrument. We ask whether this evidence survives its intended conditions of use: two competent analysts, the same system, the same tool, and different defensible settings.

We pre-registered a crossed grid of seven analytic axes, or choices, with every level taken from a published implementation. A deterministic claim map converts each discovered circuit into a structured Annex IV statement. Across 15,840 pre-registered specifications—tuples of analytic choices—on GPT-2 small and the indirect object identification task, 7,561 produced a claim. The claim flip rate, the probability that two distinct specifications yield different statements, is 73.2% (95% CI 0.725 to 0.738). Here CI denotes the reported uncertainty interval. The modal claim, the most common statement, covers 41.1% of the space. The evidence fails the filability criterion at every tolerance a conformity assessment body would plausibly accept.

Standardising the most influential choice, the evaluation metric used to assess a circuit, leaves a 59.4% flip rate. Removing circuit size from the claim and holding size fixed leaves 27.1% (95% CI 0.255 to 0.286), still above the pre-registered threshold. The circuits are structurally near-disjoint: median pairwise Jaccard overlap, shared edges divided by edges in either circuit, is 4%. They are functionally uncorrelated at Cohen’s kappa 0.015, the agreement measure used to distinguish raw agreement from agreement explained by correctness rates. The instability is therefore not one mechanism described in different words.

We provide a standalone filability protocol. We also report that one of seven documented discovery objectives does not execute on the library’s own canonical task. The study covers one model and one task. Whether the conclusion holds at scale is untested.

**Claims and evidence**
- Claims are unstable: 7,561 claims from 15,840 specifications; 73.2% flips, 95% CI 0.725 to 0.738; modal share 41.1% (detailed in Table 3, H2, and Table 4, rank 1).
- Controls do not remove instability: 59.4% after metric standardisation (Table 6, Evaluation metric); 27.1%, 95% CI 0.255 to 0.286, after removing size from the claim and fixing size (Section 7.4).
- Circuits differ structurally and functionally: median overlap 4%; kappa 0.015 (Sections 7.1 and 7.6).
- The instrument has a non-executing arm: one of seven objectives (Section 6).

## 1 Introduction

*The section explains why regulatory use of circuit evidence requires compatible conclusions across analysts.*

An audit means two auditors reach the same conclusion. This is not a definition borrowed from machine learning. It is what makes an audit worth commissioning.

Regulation (EU) 2024/1689 requires providers of high-risk AI systems to file technical documentation. Annex IV point 2(b) asks for “the general logic of the AI system and of the algorithms”. Point 2(e) asks for “an assessment of the technical measures needed to facilitate the interpretation of the outputs”. Article 86(1) gives a person subject to certain automated decisions the right to “clear and meaningful explanations of the role of the AI system in the decision-making procedure and the main elements of the decision taken”. These quotations come from the consolidated text as amended by Regulation (EU) 2026/1744, CELEX 02024R1689-20260727, read 11 August 2026.

Mechanistic interpretability offers the technical content. Circuit discovery has mature tooling and produces a subgraph said to explain a behaviour. A provider’s circuit-level filing is worth something only if a second analyst, given the same system and tool, would file something compatible.

We measure how far a regulatory claim moves across defensible analytic specifications. We also provide a decision rule for a standards body. We do not propose a better discovery method or argue that interpretability is worthless. We argue that evidence as currently produced does not meet the standard presumed by filing it.

#### Contributions.

*The paper contributes a multiverse, a filability criterion, controls and a warning about pooled statistics.*

(i) A pre-registered multiverse—a designed space of defensible analytic specifications—over seven circuit-discovery axes. Every level comes from a published implementation. A deterministic map converts circuits to Annex IV statements.

(ii) A filability criterion, \(\pi^{*}\geq1-\alpha\), for a conformity procedure, and its measured value on a canonical task. Here \(\pi^{*}\) is modal claim share and \(\alpha\) is the permitted disagreement tolerance.

(iii) Evidence that instability survives standardising any single axis, removing circuit size from the claim, and holding size fixed.

(iv) A general methodological result: when specifications select objects of different sizes, pooled statistics, which combine sizes, can invert. We show three such inversions.

## 2 Related work

*The section connects non-identifiability, metric sensitivity, functional comparison, auditing and multiverse analysis.*

#### Non-identifiability.

*Prior work shows that a behaviour need not have a unique mechanistic explanation.*

Méloux et al. [6] ask whether a behaviour has a unique explanation under mechanistic interpretability’s own criteria. They answer that it need not. This non-identifiability is our premise. Our contribution is downstream: measuring its cost when an explanation is filed rather than published.

#### Faithfulness metric sensitivity.

*Prior work finds that ablation-based faithfulness scores depend strongly on methodological choices.*

Miller et al. [8] survey choices in ablation-based faithfulness measurement, which evaluates circuits using ablation operators. They find high sensitivity to seemingly insignificant changes. They conclude that scores reflect the researcher’s choices as well as the circuit. They establish that one axis matters; we cross seven and propagate the result to a regulatory statement. We use their released library. Our study therefore stress-tests the tooling by its own definitions.

#### Structure versus function.

*Structurally different circuits can implement the same computation, so this study also measures function.*

Bayat Makou et al. [1] vary input statistics while holding the task fixed. They find structurally distinct circuits implementing the same computation, which they call phantom specialisation. This is the strongest objection to our design. If discovery samples an equivalence class of behaviourally identical subgraphs, claim flips may only identify which member was sampled. Section 7.6 answers with functional measurement: our circuits are not functionally interchangeable.

#### Variance as estimation.

*Estimator variance provides a statistical companion to variation across whole specifications.*

Méloux et al. [7] treat mechanistic interpretability as statistical estimation and analyse estimator variance. Seed variance is one axis in our space. Their framing complements our analysis of the whole specification.

#### Auditability.

*Prior work calls for auditing standards and standardised evaluation.*

Lan et al. [4] describe two papers reaching conflicting conclusions about the same behaviour. A third found both partly correct and incomparable. They call for auditing standards to certify findings in safety-critical settings. We answer with a measurement and criterion. Sharkey et al. [10] list open problems, including socio-technical ones touched here. Mueller et al. [9] argue that comparing methods requires standardised evaluation. This is adjacent to our argument that certification requires standardised specification.

#### Multiverse method.

*Multiverse analysis exposes conclusions that analytic flexibility can manufacture.*

Steegen et al. [13] propose reporting across defensible data-processing choices. Simonsohn et al. [12] provide the specification-curve visualisation we adapt. Simmons et al. [11] identify the underlying problem: undisclosed analytic flexibility lets an analyst present almost anything as significant. We apply this psychology methods literature to a machine learning evidentiary question. Explanation stability is exactly the kind of claim that flexibility can manufacture.

#### Explanation metrics under optimisation pressure.

*Two metrics used here can be inflated without changing predictions or explanations.*

Hsia et al. [3] show that sufficiency and comprehensiveness can be inflated without changing predictions or explanations. These are two of our four evaluation metrics. This is one reason we report discard rates by metric rather than silently pooling.

**Claims and evidence**
- Explanations need not be unique: Méloux et al. [6].
- Faithfulness scores are sensitive to analytic choices: Miller et al. [8].
- Structural difference need not imply functional difference: Bayat Makou et al. [1].
- Sufficiency and comprehensiveness can be inflated without changing predictions or explanations: Hsia et al. [3].
- This section gives no numerical results or supporting table for these cited findings.

## 3 What this paper is not claiming

*The results do not imply random circuits, equivalent mechanisms, seed nondeterminism or generalisation to scale.*

We state these limits early. The results do not support the following readings; we ourselves held two before running the decompositions.

We do not claim that discovered circuits are no better than random. At fixed circuit size, they are more stable than a size-matched random null: 0.2746 against 0.4230. They carry information.

We do not claim that filings differ where mechanisms do not. Functional instability—one minus agreement between circuits’ per-example verdicts—is 0.3972 pooled and 0.4777 at the smallest circuits.

We do not claim that discovery samples one mechanism and dresses it differently. Cohen’s kappa of 0.0146 says the circuits are functionally uncorrelated.

We do not claim that random seeds are the problem. Here the seed selects sampled prompts. It represents evaluation-set variability, not nondeterminism.

We claim nothing about scale: one model, one task.

**Claims and evidence**
- Discovered circuits carry information: fixed-size flip rates 0.2746 discovered versus 0.4230 random (Section 7.5).
- Mechanisms differ: functional instability 0.3972 pooled and 0.4777 at the smallest circuits (section text; pooled result also in Section 7.6).
- Circuits are functionally uncorrelated: kappa 0.0146 (Section 7.6).
- The seed and scale qualifications are design limits: prompt sampling; one model and one task (Sections 6 and 9).

## 4 Formalism

*The section defines specifications, circuit and claim instability, and the filability rule.*

A specification is a tuple of analytic choices:
\[
s=(o,a,d,m,\tau,P,r,g).
\]
Here \(o\) is the discovery objective; \(a\), the ablation operator; \(d\), the corruption distribution; \(m\), the evaluation metric; \(\tau\), the size threshold; \(P\), the prompt variant; \(r\), the seed; and \(g\), the granularity, or claim-detail level. Every level comes from a published implementation. None was invented for this study.

Discovery returns \(C(s)\subseteq E\): the circuit is a subset of the model’s edge set \(E\). For GPT-2 small’s factorised graph, which separates query, key and value inputs, \(|E|=32{,}491\).

We confirm the count from the instrument and independently from a closed-form count over computation order. Block \(b\) attention receives from \(1+13b\) sources at three input slots for each of 12 heads. MLP \(b\) receives from those sources plus its own block’s heads. The output receives from all 157 non-terminal sources. Circuits touch 144 attention heads and 12 MLPs, the two component types counted here: 156 components in total.

#### Circuit instability.

*Circuit instability is one minus edge-set Jaccard similarity.*

For two specifications \(s_i,s_j\), with circuits \(C_i,C_j\),
\[
D(s_i,s_j)=1-\frac{|C_i\cap C_j|}{|C_i\cup C_j|}.
\tag{1}
\]
Here \(\cap\) counts shared edges, \(\cup\) counts edges in either circuit, and \(|\cdot|\) denotes set count. The mean Jaccard similarity over unordered pairs is \(\bar J\).

#### Claim instability.

*Claim instability is the pairwise probability of different structured statements.*

Let \(\varphi\) be the deterministic map from circuits to structured Annex IV statements. The flip rate is
\[
F=1-\frac{\sum_c n_c(n_c-1)}{N(N-1)},
\tag{2}
\]
where \(c\) is a claim class, \(n_c\) counts specifications yielding it, \(N\) counts specifications, and \(\sum\) denotes summation.

Equation 2 is the unbiased Gini–Simpson form: the probability that two draws without replacement differ. Unit tests compare it with the brute-force pairwise definition over randomised cases. Modal share is
\[
\pi^{*}=\max_c n_c/N,
\]
where \(\max\) selects the largest class count.

For conditioning axes \(G\), we also use
\[
F_{\text{within}}(G)
=1-\frac{\sum_g\sum_c n_{g,c}(n_{g,c}-1)}
{\sum_g n_g(n_g-1)}.
\tag{3}
\]
Here \(g\) indexes conditioning groups, \(n_g\) is a group’s specification count, and \(n_{g,c}\) is its count for class \(c\). This within-group flip rate weights every pair equally, not every group equally. With \(G\) empty, it reduces exactly to Equation 2.

#### Filability.

*Evidence is filable when its modal share meets the declared tolerance.*

Evidence is filable at tolerance \(\alpha\) if and only if
\[
\pi^{*}\geq1-\alpha.
\]

The stated ceiling is \(F\leq1-1/k\), where \(k\) is the number of claim classes. A flip rate must therefore be read against the map’s class count. With nine observed classes, the ceiling is 0.8889. The observed 0.7316 is 82.3% of it. \(F\) is not near one by construction.

**Claims and evidence**
- The graph has 32,491 edges and 156 components: instrument count and independent count using \(1+13b\), three input slots, 12 heads, 157 non-terminal sources, 144 attention heads and 12 MLPs (section text).
- Equation 2 matches the brute-force definition in randomised unit tests; Equation 3 reduces to Equation 2 when \(G\) is empty (section text).
- Observed instability is 82.3% of the stated ceiling: nine classes, ceiling 0.8889, observed \(F=0.7316\) (Table 5, overseer/MEDIUM, for class count and \(F\)).

## 5 The claim map

*The section fixes the regulatory claim map and checks sensitivity to size binning.*

The map \(\varphi\) is the paper’s most attackable choice. It is therefore pre-registered, deterministic and published. It is not generated by a language model. That would add model variance to circuit variance without separating them.

There are two addressees, following the regulation. The overseer map, \(\varphi_{\text{overseer}}\), uses Annex IV 2(e) and 3 with Article 14(4)(c). It is keyed first on dominant layer band, then size class, then ranked input segment. The affected-person map, \(\varphi_{\text{affected}}\), uses Article 86(1). It leads with input segment because the right is owed to the person the decision is about. The three granularities are nested: a finer claim refines rather than contradicts a coarser one.

The size bins were not chosen to produce the reported flip rate. A committed calibration rule returned them from a measured node-count curve. Table 1 reports sensitivity.

Table 1: Flip rate under alternative size binnings. The committed bins are not a
maximum: four alternatives give a higher \(F\), including halving every threshold.
Across binnings differing by a factor of four the range is 0.70 to 0.77, and
removing the size term from the claim entirely still leaves 0.5733.

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
| no size term (COARSE) | 0.5733 |

Here \(\times\) means multiplication. COARSE is the coarsest claim granularity.

**Claims and evidence**
- Committed bins do not maximise \(F\): 0.7316 versus 0.7448, 0.7435, 0.7394 and 0.7719 (Table 1, committed bins; thresholds \(\times\) 0.5, 0.8 and 0.9; deciles).
- Binning sensitivity remains in the reported 0.70 to 0.77 range: Table 1, equal thirds 0.7018 through deciles 0.7719.
- Removing size does not remove instability: \(F=0.5733\) (Table 1, no size term (COARSE)).

## 6 Method

*The section specifies the model, grid, pre-registration, uncertainty calculation and two exclusions.*

#### Model and task.

*The study uses GPT-2 small on indirect object identification for comparability and a meaningful random baseline.*

We use GPT-2 small on indirect object identification [15]. This scale matches the non-identifiability and faithfulness literature we extend. Comparability is the point. The edge count also makes a size-matched random baseline meaningful.

#### Grid.

*The pre-registered grid nests corruption choices where operators use them.*

The grid has seven discovery objectives, seven ablation operators, four corruption distributions nested within the five operators that read them, four metrics, three thresholds, two prompt variants and five seeds. As pre-registered, this gives 1,540 discovery cells and 18,480 specifications.

Corruption is nested, not crossed, because two operators never read the corrupt distribution. Crossing them would duplicate specifications. Discovery uses the edge attribution patching family [14] and integrated edge gradients, the methods implemented in Miller et al.’s library [8]. The automated circuit discovery lineage is Conmy et al. [2]. The four corruption levels come from Wang et al. [15].

#### Pre-registration.

*The analysis plan and both possible abstracts were committed before the confirmatory sweep.*

The plan was committed and tagged before the confirmatory sweep began. Both abstracts were drafted in advance so neither direction could be presented as a surprise. An append-only file records deviations, dates, and whether each decision preceded or followed seeing the affected result.

#### Statistics.

*Uncertainty is conditional on the designed grid and is estimated by resampling specifications.*

We compute no \(p\)-value across specifications. The specifications form a designed grid, not an independent sample [13, 12].

A nonparametric bootstrap resamples specifications, never pairs, with \(B=10{,}000\) resamples. Both \(F\) and \(\bar J\) are U-statistics over pairs: pairs sharing a specification are dependent. The intervals quantify uncertainty given the grid and license nothing beyond it.

#### Two exclusions, criteria fixed before any circuit was seen.

*One objective fails to execute, and the threshold rule discards specifications unevenly across metrics.*

One objective does not execute. All 220 of its cells raise a runtime error inside the library: mean squared error is computed against an integral target. A digest over sorted failing cell identifiers verifies that the failures are exactly this arm and nothing else. We did not repair the library. It is the instrument under test and is reproduced verbatim. The realised grid has six objectives, 1,320 cells and 15,840 specifications.

The metric-relative threshold rule admits no circuit for 8,279 of 15,840 specifications. The discard rate is 52.3% and is uneven across metrics (Table 2). This was disclosed before the lock. The surviving pool is 52% comprehensiveness, against 25% by design. We report this composition rather than smoothing it.

Table 2: Discard rate by evaluation metric under the pre-registered
metric-relative threshold rule. The unevenness was disclosed before the
pre-registration was locked, from a six-cell validation slice.

| Metric | Kept | Discarded | Rate |
|---|---:|---:|---:|
| comprehensiveness | 3,947 | 13 | 0.003 |
| KL divergence | 1,467 | 2,493 | 0.630 |
| logit difference | 1,270 | 2,690 | 0.679 |
| sufficiency | 877 | 3,083 | 0.779 |

KL divergence, logit difference and sufficiency are the other three evaluation metrics.

**Claims and evidence**
- One objective fails completely: 220 cells; the realised grid is six objectives, 1,320 cells and 15,840 specifications (section text).
- Discards are high and uneven: 8,279 of 15,840, or 52.3%; surviving comprehensiveness share 52% versus 25% by design (section text).
- Metric-specific discards: 13 at 0.003; 2,493 at 0.630; 2,690 at 0.679; 3,083 at 0.779 (Table 2, comprehensiveness, KL divergence, logit difference and sufficiency).
- Unevenness was disclosed before lock using a six-cell validation slice (Table 2 caption).

## 7 Results

*The section reports pre-registered outcomes and decomposes structural, claim, random-baseline and functional results.*

Table 3 matches each pre-registered outcome to its result. Decision rules and thresholds were fixed and tagged before the confirmatory sweep. None was revisited.

Table 3: Pre-registered outcomes and their results. Thresholds were fixed on
6 August 2026 and the plan was tagged before the sweep began. Two results carry
qualifications that only the within-size decomposition reveals, and those are
reported in Sections 7.5 and 7.6 rather than deferred
to an appendix.

| | Decision rule | Result | Outcome |
|---|---|---|---|
| P0 | \(\bar{J}\) reported, not tested | 0.1396 [0.1377, 0.1418] | premise holds |
| H2 | \(F>0.20\), CI lower bound above | 0.7316 [0.7247, 0.7380] | confirmed |
| H3 | not separated from random null | separated, intervals disjoint | rejected |
| H4 | \(F-(1-\text{agreement})>0.10\) | 0.3344 [0.3297, 0.3394] | supported |

P0 is the reported, untested structural premise. H2 tests claim instability. H3 concerns separation from the random null. H4 tests the gap between claim instability and functional instability.

### 7.1 Circuits are almost disjoint

*Circuit pairs have little edge overlap, although their overlap exceeds chance.*

Mean Jaccard similarity is \(\bar J=0.1396\) (95% CI 0.1377 to 0.1418). We compute it exactly over all 28,580,580 specification pairs, not a sample. The 7,561 specifications contain 3,218 distinct circuits, making exact computation tractable.

The pairwise \(D\) distribution adds detail. Its 1% quantile is 0.196; its 25% quantile, 0.788; its median, 0.960; and its 75% quantile, 0.998. Thus, the median pair shares 4% of edges. Only 1% of pairs exceed Jaccard 0.8.

At the median size of 200 edges, the analytic random line \(k/(2|E|-k)\), where \(k\) here is circuit size, gives 0.0031. Observed overlap is about 45 times chance. Beating chance at this scale is not evidence of much: a 500-edge circuit has expected random overlap 0.008.

**Claims and evidence**
- Mean overlap is low: 0.1396, 95% CI 0.1377 to 0.1418 (Table 3, P0); exactly 28,580,580 pairs, 7,561 specifications and 3,218 distinct circuits (section text).
- Typical circuits are nearly disjoint: \(D\) quantiles 0.196, 0.788, 0.960 and 0.998 at 1%, 25%, median and 75%; median overlap 4%; only 1% exceed Jaccard 0.8 (section text).
- Overlap exceeds chance: median size 200, random line 0.0031, about 45 times chance; at 500 edges, expected random overlap 0.008 (section text).

### 7.2 The claim flips on three quarters of pairs

*Claims fail every pre-registered filability tolerance, including at the coarsest granularity.*

The flip rate is \(F=0.7316\) (95% CI 0.7247 to 0.7380). The pre-registered threshold is 0.20, with the interval’s lower bound required to exceed it. It clears by more than three times.

Modal share is \(\pi^{*}=0.4109\). The evidence is not filable at any pre-registered tolerance: 0.411 against required shares of 0.95, 0.90 and 0.80.

Table 4: The nine claim classes produced by \(\varphi_{\text{overseer}}\) at medium
granularity, ordered by frequency. Aggregating over size class, the layer
attribution splits 47.9% early, 43.6% late and 8.5% middle; the first of these
is the COARSE modal share of 0.4788, computed independently.

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

Early, middle and late are the layer-band labels. Sparse, moderate and distributed are the size-class labels.

The two most common claims cover 41.1% and 28.5% of the space. They attribute the same model’s behaviour on the same task to early and late layers (Table 4). They are not different emphases. They contradict.

All six addressee-granularity combinations reject filability (Table 5). For the affected person, the coarsest statement has two classes and still flips on 39.5% of pairs. The result is not an artefact of fine-grained claim language. Figure 1 shows the specification curve; the modal step’s width is \(\pi^{*}\).

Table 5: Flip rate and modal share for both addressees at all three
granularities. Filability is evaluated at the loosest pre-registered tolerance,
\(\alpha=0.20\), which requires \(\pi^{*}\geq0.80\). No cell passes.

| Addressee | Granularity | Classes | \(F\) | \(\pi^{*}\) | Filable |
|---|---|---:|---:|---:|---|
| affected | COARSE | 2 | 0.3950 | 0.7291 | no |
| affected | MEDIUM | 4 | 0.6597 | 0.4875 | no |
| affected | FINE | 8 | 0.6744 | 0.4786 | no |
| overseer | COARSE | 3 | 0.5733 | 0.4788 | no |
| overseer | MEDIUM | 9 | 0.7316 | 0.4109 | no |
| overseer | FINE | 12 | 0.7681 | 0.4109 | no |

MEDIUM and FINE denote the middle and finest nested granularities.

Figure 1: Specification curve for \(\varphi_{\text{overseer}}\) at medium
granularity. Upper panel: claim class against specification rank, classes ordered
by frequency, so each step’s width is that class’s share of the space. The dashed
line marks the size-matched random null’s modal share; the pooled comparison is
confounded by circuit size, discussed in Section 7.5. Lower panel:
which analytic level is active for each specification. The solid band on the
comprehensiveness row is the composition effect of the discard rule in
Table 2; the seed rows are unstructured, which is the same result
as seed removing 0.0009 from \(F\) in Table 6.

**Claims and evidence**
- H2 is confirmed: \(F=0.7316\), 95% CI 0.7247 to 0.7380, against 0.20 (Table 3, H2).
- The dominant claims contradict: early/distributed, 3,107 and 0.4109; late/sparse, 2,153 and 0.2848—reported as 41.1% and 28.5% (Table 4, ranks 1 and 2).
- Layer attribution splits 47.9% early, 43.6% late and 8.5% middle; early COARSE share is 0.4788 (Table 4 caption; Table 5, overseer/COARSE).
- Every addressee-granularity combination fails \(\pi^{*}\geq0.80\): shares 0.7291, 0.4875, 0.4786, 0.4788, 0.4109 and 0.4109 (Table 5, all six rows).
- Coarse wording does not remove instability: two classes and \(F=0.3950\), or 39.5% (Table 5, affected/COARSE).

### 7.3 No analytic axis rescues the filing

*Standardising any single axis leaves substantial claim instability, and structural variance has different drivers.*

Table 6: Residual flip rate once each axis is standardised, computed with
Equation 3. All intervals are valid: group sizes 1,512 to 2,520,
bootstrap bias below 0.0006 on every axis, and every observed value lies inside
its own interval.

| Axis standardised | Residual \(F\) | 95% CI | Removes |
|---|---:|---|---:|
| Evaluation metric | 0.5939 | 0.5803 to 0.6063 | 0.1377 |
| Ablation operator | 0.7014 | 0.6939 to 0.7075 | 0.0302 |
| Discovery objective | 0.7018 | 0.6938 to 0.7084 | 0.0298 |
| Threshold \(\tau\) | 0.7070 | 0.6981 to 0.7149 | 0.0246 |
| Corruption | 0.7092 | 0.7008 to 0.7166 | 0.0224 |
| Prompt variant | 0.7232 | 0.7157 to 0.7300 | 0.0084 |
| Seed | 0.7307 | 0.7234 to 0.7368 | 0.0009 |

The metric is the largest single lever, but it is not enough. Fully standardising it—more than a conformity procedure could demand—leaves \(F=0.5939\), with lower interval bound 0.5803, far above threshold.

Different axes govern structure and claims. A REML variance decomposition, applied here to \(\log_{10}\) selected circuit size, attributes 78.8% of structural variance to the ablation-corruption interaction, 11.5% to residual, 4.4% to ablation, 4.1% to threshold and 0.7% to metric. The metric is nevertheless the largest driver of claim variance. A standards body guided by the structural decomposition would standardise ablation and fix almost none of the filing instability.

**Claims and evidence**
- Metric standardisation is the largest single reduction but does not rescue filing: residual 0.5939, CI 0.5803 to 0.6063, removing 0.1377 (Table 6, Evaluation metric).
- No other axis rescues filing: residuals 0.7014, 0.7018, 0.7070, 0.7092, 0.7232 and 0.7307; reductions 0.0302, 0.0298, 0.0246, 0.0224, 0.0084 and 0.0009 (Table 6, Ablation operator through Seed).
- Structural drivers differ: 78.8% interaction, 11.5% residual, 4.4% ablation, 4.1% threshold and 0.7% metric (section text).
- Reported intervals pass diagnostics: groups 1,512 to 2,520; bootstrap bias below 0.0006; every estimate inside its interval (Table 6 caption).

### 7.4 The result under maximum pressure

*Layer-attribution claims remain unstable after size is removed from the claim and held fixed.*

Remove every trace of circuit size from the claim, leaving only dominant layer band. Then hold circuit size fixed using Equation 3. The result is \(F=0.2706\) (95% CI 0.2550 to 0.2859).

The lower bound remains above the pre-registered threshold. Attribution of where the behaviour occurs still flips on 27% of pairs when size cannot contribute.

**Claims and evidence**
- The strongest size control does not remove instability: \(F=0.2706\), 95% CI 0.2550 to 0.2859; reported as 27% (section text).
- The lower bound 0.2550 exceeds the pre-registered 0.20 threshold (section text; Table 3, H2, for the threshold).

### 7.5 The random baseline, and why its pooled direction misleads

*The pooled random comparison reverses when circuit size is held fixed.*

Pooled, the discovered multiverse is less stable than a size-matched random null: 0.7316 against 0.6774, with disjoint intervals. The null uses \(R=1{,}000\) random multiverses drawn from all 32,491 edges.

H3 is rejected because the distributions are separated. We do not report the separation’s direction as a finding. It is an artefact of the size distribution, as Table 7 shows. Above 1,000 edges, the null is degenerate: a uniform draw touches all 156 components and can produce only one claim.

Holding size fixed reverses the ordering. At medium granularity, discovered circuits have flip rate 0.2746 against null 0.4230. At coarse granularity, the values are 0.2706 against 0.3822. Discovered circuits are more stable than random ones in the fair comparison. They are still not stable enough to file.

Table 7: Flip rate within each circuit size, medium granularity. The null
collapses to a single claim class at and above 1,000 edges, which is what
depresses its pooled flip rate. The discovered multiverse also collapses at
5,000 and 10,000 edges, so 2,692 of 7,561 specifications contribute no
flips at all and the instability lives entirely below 2,000 edges.

| Size | \(n\) | Discovered \(F\) | Classes | Null \(F\) | Null classes |
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

Here \(n\) is the number of specifications at each circuit size.

**Claims and evidence**
- H3 is rejected: pooled discovered 0.7316 versus null 0.6774, disjoint intervals; \(R=1{,}000\) null multiverses from 32,491 edges (section text; Table 3, H3).
- The pooled direction reverses at fixed size: medium 0.2746 versus 0.4230; coarse 0.2706 versus 0.3822 (section text).
- The null collapses at large sizes: \(F=0.0000\), one class at 1,000, 2,000, 5,000 and 10,000 edges (Table 7, those four rows).
- Discovered circuits also collapse at 5,000 and 10,000: \(n=625\) and 2,067, \(F=0.0000\), one class; 2,692 of 7,561 specifications contribute no flips (Table 7, those two rows and caption).
- Smaller-size results vary: discovered/null \(F\) is 0.4543/0.7331, 0.3380/0.6622, 0.4639/0.7310, 0.5792/0.6584, 0.5122/0.6366 and 0.4258/0.1421 (Table 7, sizes 10, 20, 50, 100, 200 and 500).

### 7.6 Filings differ more than mechanisms, over part of the range

*The pooled functional gap passes its threshold, but the within-size results qualify it.*

Per-example functional agreement is 0.6028, so functional instability is 0.3972. The gap to claim flip rate is 0.3344 (95% CI 0.3297 to 0.3394), against the pre-registered 0.10 threshold. Agreement uses each circuit’s correct or incorrect verdict on each example, following Bayat Makou et al.’s measures [1] verbatim.

Cohen’s kappa is 0.0146. The 60% raw agreement is almost entirely explained by both circuits being right on the same fraction of examples, not the same examples. At every circuit size, kappa stays between 0.016 and 0.047. Raw agreement rises from 0.52 to 0.92, tracking accuracy—the fraction of correct verdicts—from 0.51 to 0.95.

This answers the phantom-specialisation objection. If discovery sampled behaviourally identical subgraphs, the circuits would be functionally interchangeable. At kappa 0.015, they are not.

H4 is supported on the pre-registered quantity. The gap does not survive fixing size: it fails the 0.10 threshold at 6 of 10 sizes and is negative at 5. At both ends, mechanisms differ more than filings. The pooled figure is carried by the 500 to 2,000 edge band alone. We report the pooled result because it was pre-registered and the decomposition because the pooled result alone would mislead.

**Claims and evidence**
- H4 passes pooled: agreement 0.6028, functional instability 0.3972, gap 0.3344, 95% CI 0.3297 to 0.3394, against 0.10 (section text; Table 3, H4).
- Circuits are functionally uncorrelated: kappa 0.0146, also stated as 0.015; raw agreement 60% (section text).
- Raw agreement rises without comparable kappa: size-specific kappa 0.016 to 0.047; agreement 0.52 to 0.92; accuracy 0.51 to 0.95 (section text).
- The pooled gap does not generalise across sizes: 6 of 10 fail 0.10; 5 are negative; the pooled result is carried by 500 to 2,000 edges (section text).

**Claims and evidence — Section 7 overview**
- P0 holds: \(\bar J=0.1396\) [0.1377, 0.1418] (Table 3, P0).
- H2 is confirmed: \(F=0.7316\) [0.7247, 0.7380] (Table 3, H2).
- H3 is rejected: separated distributions with disjoint intervals, qualified by the within-size reversal (Table 3, H3; Section 7.5).
- H4 is supported: gap 0.3344 [0.3297, 0.3394], qualified by failure at 6 of 10 sizes and negative gaps at 5 (Table 3, H4; Section 7.6).

## 8 The filability protocol

*The protocol requires a declared specification space, a deterministic map and measured agreement.*

Filability of an interpretability claim. For a filed claim \(c\) derived from a circuit:

1. Declare the specification space \(\mathcal S\): the analytic axes and defensible levels of each, with a citation for every level.
2. Declare the claim map \(\varphi\) as deterministic, published code. Not a language model.
3. Compute \(\pi^{*}=\max_c n_c/N\) over \(\mathcal S\), and compute \(F\) using Equation 2.
4. The claim is filable at tolerance \(\alpha\) if \(\pi^{*}\geq1-\alpha\).
5. Report \(\pi^{*}\), \(F\), discard rate by axis level, and the within-group decomposition of any pooled statistic.

A filing that cannot state its own \(\pi^{*}\) has been asserted, not audited. The regulator need not decide which analytic choice is correct. This makes the protocol implementable: the filer must declare the space and report how much of it agrees.

## 9 Limitations

*The section limits generalisation, identifies size and discard effects, and discloses invalid bootstrap intervals.*

#### One model, one task.

*Scale remains untested, and the planned model and task replications have not run.*

Larger models have more components, giving both layer band and size class more room. The direction of the effect on \(F\) is not obvious. This limitation is the one most likely to change the number.

A second-model replication is pre-registered but unrun: a 924-cell grid on Pythia-160m, a different-architecture model performing the same task. Every axis is identical except that seeds have three levels rather than five. The plan, reduced grid, and criteria distinguishing replication from refutation were committed and timestamped before any cell ran. A revision will report either direction. A second task is designed but unrun.

#### Circuit size is an outcome, not an axis.

*The threshold-selected size distorts three pooled quantities.*

The threshold rule chooses circuit size. Size has distorted \(F\), the direction of the null comparison, and the sign of the functional gap. Fixing size drops \(F\) from 0.7316 to 0.2746, more than standardising any pre-registered axis. We take this as a general hazard for multiverses whose specifications select differently sized objects. We report every pooled statistic with its within-size decomposition.

#### The multiverse is as wide as the library, not the literature.

*The instrument omits at least one defensible operator.*

Optimal ablation [5] is a defensible operator absent from the instrument. It therefore never entered the space. The measured \(F\) is a lower bound with respect to the field’s full defensible choices.

#### The discard rate is high and uneven.

*The retained set is metric-imbalanced despite the pre-fixed rule and absence of repairs.*

The discard rate is 52.3%. The surviving pool is metric-imbalanced. We fixed the threshold rule in advance and repaired no failing item. We state the resulting composition rather than adjust it.

#### The bootstrap is invalid for one reported family.

*Very small conditioning groups make the intervals unusable, so the paper reports point estimates.*

For conditioning groups containing only two to five specifications, retained self-pairs dominate the bootstrap. Those intervals are unusable. We report point estimates with the diagnostic attached. We did not construct a bias-corrected estimator after seeing naive intervals disagree. Choosing an estimator in response to a result is the behaviour this paper documents.

**Claims and evidence**
- Generalisation is untested: one model and one task; planned 924-cell Pythia-160m grid with three rather than five seed levels; second task designed but unrun (section text).
- Size changes the result more than any registered axis: 0.7316 to 0.2746 (section text; Table 6 for single-axis residuals).
- Discards remain a limitation: 52.3%; uneven metric rates 0.003, 0.630, 0.679 and 0.779 (Table 2, all four metric rows).
- One interval family is invalid: conditioning groups of two to five specifications, dominated by retained self-pairs; only point estimates are reported (section text).
- Coverage excludes optimal ablation [5] because the instrument does not implement it (section text).

## 10 Discussion

*The section argues for measured evidentiary weight rather than assuming that interpretability evidence is stable.*

#### What standardising one axis buys.

*Even the largest single-axis reduction leaves most filing instability.*

The metric is the biggest lever. It removes 0.1377 from a 0.7316 flip rate. There is no single knob. A procedure that standardises one axis and declares the problem solved would be mistaken by a wide margin.

#### On the instrument.

*One documented objective fails on the library’s canonical task.*

One of seven documented, publicly exported discovery objectives does not run on the library’s canonical task. We report this as a discard with its rate and as an observation about the maturity of tooling expected to support regulatory evidence. It is not the headline. Nor is it criticism of a library whose authors are among those calling for the field to address exactly this [10].

#### What a regulator can take from this.

*Interpretability evidence should receive measured, not assumed, evidentiary weight.*

The result is not that interpretability evidence is worthless. Its evidentiary weight is measurable rather than assumed. This measurement is cheap relative to the sweep that produced the circuit.

**Claims and evidence**
- One-axis standardisation is insufficient: metric removes 0.1377 from 0.7316, leaving 0.5939, CI 0.5803 to 0.6063 (Table 6, Evaluation metric).
- The instrument has a non-executing objective: one of seven; all 220 cells in that arm fail (Section 6).
- Evidentiary weight is measured by the protocol’s \(\pi^{*}\) and \(F\) (Section 8); this section gives no numerical cost comparison.

## 11 Reproducibility

*The section describes the committed plan, environment verification, result checks and test suite.*

The pre-registration is a timestamped public-repository commit tagged `prereg-p1-confirmatory`, made before the confirmatory sweep. Every reported number traces to a config, or configuration, a seed and an environment fingerprint.

The results archive verifies a single environment across all 1,320 cells. Content-addressed comparison of the uploaded archive with the git tree recovered the code commit: 22 of 22 files were identical. The analysis layer runs without a GPU, the computing resource not required for this layer.

A verification script recomputes every headline quantity from raw results and asserts that it matches the report: 36 checks, 0 failures. It derives \(F\) by three routes, including brute-force pairwise computation.

A second script checks every numeric token in the manuscript against a value derived from stored results. Every table is emitted directly from those results, not transcribed. The test suite contains 343 tests.

**Claims and evidence**
- The sweep used one environment across 1,320 cells, verified from the archive (section text).
- The recovered code matches the archive: 22 of 22 files identical (section text).
- Headline verification passes: 36 checks, 0 failures; \(F\) computed by three routes (section text).
- Numeric tokens and tables are checked or generated from stored results; the suite contains 343 tests (section text).
