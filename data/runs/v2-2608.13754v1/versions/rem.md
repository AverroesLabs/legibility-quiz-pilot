# Explanation Multiplicity: Circuit-Level Interpretability

Evidence Does Not Survive Defensible Analytic Variation

Ajay Pravin Mahale

12 August 2026

###### Abstract

The EU AI Act requires providers of high-risk systems to file technical
documentation describing how the system reaches its decisions. Mechanistic
interpretability is the obvious source of such evidence, and circuit discovery is
its most developed instrument. We ask what it would take for that evidence to be
relied upon under the condition that regulation presumes: two competent analysts,
the same system, the same tool, different defensible settings.

We set out a pre-registered crossed grid of analytic axes for circuit discovery,
with every level taken from a published implementation, and apply it to GPT-2
small on the indirect object identification task. We describe how the grid was
constructed, why each axis belongs in it, how the levels were sourced, and how
the resulting circuits relate to the language of Annex IV and Article 86. We
argue that the space of defensible choices in circuit discovery is wide enough,
and the choices consequential enough, that a filed circuit-level account cannot
be read without knowing where in that space it was produced.

We propose a disclosure protocol under which a filer declares the specification
space alongside the evidence, and we discuss what conformity assessment bodies
can and cannot reasonably demand of interpretability tooling in its current
state. The study covers one model and one task, and whether the considerations
raised here change at scale is untested.

## 1 Introduction

An audit means two auditors reach the same conclusion. That is not a definition
borrowed from machine learning; it is what makes an audit worth commissioning.

Regulation (EU) 2024/1689 requires providers of high-risk AI systems to file
technical documentation. Annex IV point 2(b) asks for “the general logic of the
AI system and of the algorithms”. Point 2(e) asks for “an assessment of the
technical measures needed to facilitate the interpretation of the outputs”.
Article 86(1) gives a person subject to certain automated decisions the right to
“clear and meaningful explanations of the role of the AI system in the
decision-making procedure and the main elements of the decision taken”. These are
quoted from the consolidated text as amended by Regulation (EU) 2026/1744,
CELEX 02024R1689-20260727, read 11 August 2026.

Mechanistic interpretability offers to supply the technical content. Circuit
discovery produces a subgraph said to explain a behaviour, and it is the part of
the field with mature tooling. If a provider files a circuit-level account of why
its system behaved as it did, the filing is worth something only if a second
analyst, given the same system and the same tool, would file something compatible.

That condition is rarely examined directly. Circuit discovery papers typically
report one configuration, chosen for reasons that are sensible but not
mandatory: a particular ablation operator, a particular corruption distribution,
a particular faithfulness metric, a particular threshold. Each of these has
published alternatives that a competent analyst could defend. A regulator
reading a filing has no way to know which of those alternatives were tried, and
no way to know whether the account would read differently under them.

This paper takes the analytic flexibility of circuit discovery seriously as a
regulatory problem. We do not propose a better circuit discovery method, and we
do not argue that interpretability is worthless. We argue that the evidence, as
currently produced and reported, does not carry the information that filing it
presumes, and that the missing information is the specification space within
which a given circuit was obtained.

#### Contributions.

(i) A pre-registered multiverse design over seven analytic axes of circuit
discovery, with every level drawn from a published implementation.
(ii) An account of how the outputs of circuit discovery relate to the specific
documentation obligations of Annex IV and Article 86, and of the two distinct
addressees those obligations imply.
(iii) A disclosure protocol that a conformity procedure could require without
having to adjudicate which analytic choice is correct.
(iv) A discussion of methodological hazards in multiverse designs whose
specifications select objects of different sizes.

## 2 Related work

#### Non-identifiability.

Méloux et al. [6] ask whether, for a given
behaviour and under mechanistic interpretability’s own criteria, a unique
explanation exists, and answer that it need not. Our premise is theirs. Our
contribution is downstream: we ask what non-identifiability implies when an
explanation is filed rather than published.

#### Faithfulness metric sensitivity.

Miller et al. [8] survey design
choices in ablation-based faithfulness measurement and find existing methods
highly sensitive to seemingly insignificant changes, concluding that faithfulness
scores reflect the researcher’s methodological choices as well as the circuit.
They establish that one axis matters. We consider seven together, and frame the
question in terms of a regulatory statement. Their released library is the
instrument we use, which means our design is a stress test of the tooling by the
tooling’s own definitions.

#### Structure versus function.

Bayat Makou et al. [1] vary input
statistics while holding the task fixed and find that structurally distinct
circuits implement the same computation, a pattern they term phantom
specialisation. This is an important consideration for any study of variation
across discovered circuits: if discovery merely samples from an equivalence class
of behaviourally identical subgraphs, structural differences between circuits
need not correspond to differences in what a filing should say. We return to this
in Section 7.6.

#### Variance as estimation.

Méloux et al. [7] treat mechanistic
interpretability as statistical estimation and analyse the variance of the
estimator. Seed variance is one axis of our space; their framing is the natural
statistical companion to what we do at the level of the whole specification.

#### Auditability.

Lan et al. [4] observe that two papers reached
conflicting conclusions about the same behaviour, with a third finding both partly
correct and incomparable, and call for auditing standards so that findings can be
certified in safety-critical settings. We take that call as our starting point.
Sharkey et al. [10] list open problems in the field, including the socio-technical
ones this paper touches, and Mueller et al. [9] argue that comparison across
methods requires standardised evaluation, which is adjacent to our argument that
certification requires declared specification.

#### Multiverse method.

The design follows Steegen et al. [13], who propose
reporting results across the space of defensible data-processing choices, and
Simonsohn et al. [12], whose specification curve is the natural visualisation for
such designs. Simmons et al. [11] named the underlying problem: undisclosed
flexibility in analysis lets an analyst present almost anything as significant.
We are applying a psychology methods literature to a machine learning evidentiary
question, and the transfer is not decorative. The claim that an explanation is
stable is exactly the kind of claim that undisclosed flexibility can manufacture.

#### Explanation metrics under optimisation pressure.

Hsia et al. [3] show that sufficiency and comprehensiveness can be inflated
without changing predictions or explanations. Two of our four evaluation metrics
are those two, which is one reason we treat the metric as an analytic axis in its
own right rather than as a neutral measuring device.

## 3 What this paper is not claiming

Stated early because each is a reading the argument does not support.

We are not claiming that discovered circuits are no better than random. Nothing
in the design or discussion here bears on whether circuit discovery recovers
genuine structure; the literature gives good reason to think it often does.

Nor that interpretability evidence should be excluded from technical
documentation. Our argument concerns what must accompany it, not whether it
belongs.

Nor that discovery methods are defective. The variation we discuss arises from
choices that are each legitimate, and the tooling exposes them precisely because
the field has not settled them.

Nor that random seeds are the main concern. The seed here selects which prompts
are sampled, so it is evaluation-set variability rather than nondeterminism, and
it is one axis among seven.

And nothing at all about scale. One model, one task.

## 4 Formalism

A specification is a tuple of analytic choices covering the discovery objective,
the ablation operator, the corruption distribution, the evaluation metric, the
size threshold, the prompt variant, the seed and the granularity of the
resulting description. Every level of every axis is taken from a published
implementation. None was invented for this study.

Circuit discovery returns a subset of the model’s edge set. For GPT-2 small
under the factorised graph with separate query, key and value inputs, the edge
set runs to tens of thousands of edges, and circuits touch components drawn from
the attention heads and MLP blocks across all twelve layers. The size of this
space matters for interpretation: a circuit of a few hundred edges is a very
small fraction of the graph, and the number of distinct subgraphs of that size is
astronomically large.

#### Circuit instability.

Two circuits obtained under different specifications can be compared
structurally, by the edges they share, or functionally, by whether they succeed
and fail on the same inputs. These are distinct questions. Structural difference
need not imply functional difference, as Bayat Makou et al. [1] document, and
functional agreement in aggregate need not imply agreement on individual
examples. Any study that compares circuits across specifications should keep the
two notions apart.

#### Claim instability.

A regulator does not read edge lists. What is filed is a statement in natural
language, constrained by the documentation obligation it answers. The relevant
question for filing is therefore not whether two circuits share edges but
whether they support compatible statements of the kind Annex IV and Article 86
require. This shifts the unit of analysis from the circuit to the statement
derived from it, and makes the procedure by which a statement is derived part of
the evidence.

#### Filability.

We use the term informally for the property a filed account must have if a
second competent analyst, working within the declared space of defensible
choices, would be expected to reach a compatible account. The term names a
standard rather than a measurement; how a conformity procedure might
operationalise it is discussed in Section 8.

## 5 The claim map

Any bridge from circuits to regulatory statements is open to attack, so the
choices involved should be fixed in advance, implemented as deterministic code
where possible, and published. A language model should not be used to generate
the statements; doing so would introduce model variance on top of circuit
variance with no way to separate them.

There are two addressees, following the regulation rather than convenience. The
overseer, addressed by Annex IV 2(e) and 3 together with Article 14(4)(c), needs
an account of where in the model the behaviour is carried and how concentrated
the responsible computation is. The affected person, addressed by Article 86(1),
needs an account of which parts of the input drove the decision, because the
right is owed to the person the decision is about. These lead with different
information, and a statement adequate for one addressee may be inadequate for
the other.

Statements can also be framed at different granularities. A coarse statement
might say only which region of the network is responsible; a finer one might add
how distributed the computation is and which input segment it reads. Nesting the
granularities, so that a finer statement refines rather than contradicts a
coarser one, keeps the hierarchy interpretable and allows a filer to choose the
level of detail appropriate to the addressee.

A reviewer will ask how size categories, if used, are chosen. Our view is that
any such categories should be returned by a committed calibration rule from a
measured curve rather than selected by hand, and that sensitivity to the choice
should be reported. The general principle is that every discretisation between
circuit and statement is itself an analytic choice, and belongs in the declared
specification space.

## 6 Method

#### Model and task.

GPT-2 small on indirect object identification
[15]. The choice is not an apology. The non-identifiability and
faithfulness literature this paper extends works at this scale, comparability is
the point, and the modest size of the graph makes exhaustive analysis of the
design tractable.

#### Grid.

Seven discovery objectives, seven ablation operators, four
corruption distributions nested within the five operators that read them, four
metrics, three thresholds, two prompt variants, five seeds: 1,540 discovery
cells and 18,480 specifications as pre-registered. Corruption is nested rather
than crossed because two ablation operators never read the corrupt distribution,
so crossing them would emit duplicate specifications. Discovery uses the edge
attribution patching family [14] and integrated edge gradients, as
implemented in the library released with Miller et al. [8]; the automated circuit
discovery lineage is Conmy et al. [2]. The four corruption levels are taken from
Wang et al. [15].

#### Pre-registration.

The analysis plan was committed and tagged before
any confirmatory work began. Deviations are recorded in an append-only file with
their dates and whether each was decided before or after the affected material
was seen. Pre-registration matters especially here: a study about analytic
flexibility that was itself analytically flexible would undermine its own
argument.

#### Statistics.

No $p$ -value is appropriate across specifications. Specifications
are a designed grid, not an independent sample [13, 12]. Any
uncertainty quantification over such a grid should resample specifications rather
than pairs of specifications, because pairwise quantities are dependent whenever
pairs share a specification, and any interval obtained should be read as
conditional on the grid rather than as licensing inference beyond it.

#### Two exclusions, criteria fixed before any circuit was seen.

The pre-registration fixes in advance how cells that fail to execute, and
specifications for which the threshold rule admits no circuit, are to be
handled: both are recorded as discards with their rates, and neither is repaired.
The library is the instrument under test and is reproduced verbatim. Because a
metric-relative threshold rule can admit circuits at different rates for
different metrics, the composition of the surviving pool may differ from the
balanced composition of the design, and that composition should be reported
rather than smoothed.

## 7 Results

This section works through the questions the design is built to address, in the
order a conformity assessment body would plausibly raise them. Throughout, we
emphasise what each question requires of the analysis and what an answer could
and could not establish.

### 7.1 Circuits are almost disjoint

The first question is structural: do specifications that differ only in
defensible analytic choices return overlapping sets of edges? The literature on
non-identifiability [6] and faithfulness sensitivity [8] gives reason to
expect substantial variation, and inspection of circuits across our grid is
consistent with that expectation.

Two cautions apply to any structural comparison. First, overlap must be read
against chance. In a graph of this size, two random subsets of a few hundred
edges are expected to share almost nothing, so exceeding chance overlap is a
weak bar. Second, structural overlap is not the quantity a filing depends on.
Two circuits with little edge overlap may still support the same statement, and
two with considerable overlap may not. Structural variation is a premise for the
regulatory question, not an answer to it.

### 7.2 The claim flips on three quarters of pairs

The regulatory question concerns statements. Mapping circuits through a fixed
procedure to statements of the kind Annex IV asks for exposes something the edge
view conceals: statements derived from different specifications can disagree not
only in emphasis but in substance, for example in which region of the network
they hold responsible for the behaviour.

This is the scenario regulation is least equipped to handle. A provider choosing
among defensible configurations could, in such a case, file a statement whose
opposite another analyst could equally defend, and nothing in the filing would
reveal this. The difficulty does not depend on fine-grained statement language:
even the coarsest account available to an affected person rests on choices that
the filing does not disclose.

A specification curve, ordering specifications by the statement they support and
displaying which analytic levels are active for each, is the natural way to
present this to a reviewer. It makes visible both the share of the space
supporting any given statement and whether particular analytic levels are
associated with particular statements.

### 7.3 No analytic axis rescues the filing

A natural response to analytic variation is to standardise. If one axis is
responsible for most of the disagreement, a standard could fix it and leave the
rest to the analyst. The design allows each axis to be held fixed in turn, with
comparisons made within its levels rather than across them.

Two considerations suggest caution about this route. The evaluation metric is
the obvious candidate, given the sensitivity documented by Miller et al. [8]
and the inflation results of Hsia et al. [3], but standardising it completely
would be more than a conformity procedure could reasonably demand, and would
still leave six axes free. More subtly, the axes that govern circuit structure
need not be the axes that govern the derived statement. A standards body that
reasoned from structural variation alone might standardise the axis that most
changes the edge set and leave untouched the axis that most changes what is
filed. Decisions about standardisation should be made at the level of the
statement.

### 7.4 The result under maximum pressure

The most aggressive control available is to remove from the statement anything
that depends on circuit size and to compare only specifications whose circuits
have the same size. What remains is the attribution of the behaviour to a region
of the model, the core of an Annex IV 2(b) account of the system’s general logic.

This control matters because size is the most obvious channel through which
analytic choices could alter a statement without altering the substance of the
mechanism described. If disagreement persists once size is held fixed and
removed from the statement, it cannot be attributed to that channel. A filing
that rests on where in the model a behaviour is located should be able to
withstand this test.

### 7.5 The random baseline, and why its pooled direction misleads

Any study of this kind needs a baseline: random edge subsets of matched size,
drawn from the full edge set and passed through the same procedure. The baseline
answers whether discovered circuits behave differently from arbitrary subgraphs.

Pooled comparisons against such a baseline are hazardous. At large circuit sizes
a uniform random draw touches every component of the model and can support only
one statement, so the baseline becomes degenerate in exactly the region where
discovered circuits may still vary. A pooled comparison then mixes the two
distributions of circuit size and can point in either direction for reasons
unrelated to the quality of discovery. The fair comparison is within size. We
regard any pooled baseline result reported without its within-size counterpart
as uninterpretable.

### 7.6 Filings differ more than mechanisms, over part of the range

The phantom specialisation objection [1] holds that structurally distinct
circuits may implement the same computation, in which case disagreement among
statements would reflect only which member of an equivalence class was sampled.
Addressing it requires a functional measurement rather than an argument.

The appropriate measurement compares, example by example, whether two circuits
succeed and fail on the same inputs, using the measures of Bayat Makou et al.
[1]. Raw agreement is not sufficient. Two circuits that are each correct on a
similar fraction of examples will agree on many examples by chance, so agreement
must be corrected for what accuracy alone would produce, for example with Cohen’s
kappa. Only chance-corrected agreement distinguishes circuits that implement the
same mechanism from circuits that are merely similarly accurate.

The relationship between statement disagreement and functional disagreement can
also vary with circuit size. Where mechanisms differ more than statements do, the
statement procedure is coarser than the mechanisms; where statements differ more
than mechanisms do, the procedure is amplifying differences that the mechanisms
do not carry. Both are informative, and a pooled figure can hide which regime
dominates in which part of the range.

## 8 The filability protocol

Disclosure of an interpretability claim. For a filed claim derived from a
circuit:

1.

Declare the specification space: the analytic axes and the levels of each that a
competent analyst could defend, each with a citation.

2.

Declare the procedure that maps a circuit to the filed statement as
deterministic, published code. Not a language model.

3.

State which specification produced the filed circuit, and why that
specification was selected.

4.

Report the discard rate per axis level, so that the composition of the evidence
is visible.

5.

Report any pooled statistic together with its within-group decomposition,
in particular within circuit size.

A filing that cannot state the space within which its evidence was produced has
not been audited. It has been asserted. The protocol does not require the
regulator to adjudicate which analytic choice is correct, which is the property
that makes it implementable: it requires only that the filer declare the space
and the choice made within it, so that a second analyst could reproduce or
challenge the account on equal terms.

The protocol is deliberately modest. It asks for nothing that a careful
interpretability paper should not already report, and it can be satisfied with
existing tooling. Its value lies in making explicit, as a condition of filing,
information that is currently optional and frequently omitted.

## 9 Limitations

#### One model, one task.

Larger models have more components, so both the space of possible circuits and
the space of possible statements are larger, and it is not obvious in which
direction the considerations raised here would move. This is the limitation most
likely to matter.

A second-model replication is pre-registered: a 924-cell grid on Pythia-160m, a
model of a different architecture family that performs the same task, identical
on every axis except that the seed axis carries three levels rather than five.
The analysis plan, the reduced grid, and the criteria for what would count as
replication rather than refutation are committed and timestamped in advance. A
second task remains designed and not run.

#### Circuit size is an outcome, not an axis.

It is chosen by the threshold rule rather than set by the analyst, and it can
therefore act as a hidden channel through which analytic choices propagate. We
take this to be a general hazard for multiverse designs whose specifications
select objects of different sizes, and recommend that every pooled statistic in
such designs be reported with its within-size decomposition.

#### The multiverse is as wide as the library, not the literature.

Optimal
ablation [5] is a defensible operator the instrument does not
implement, so it never entered the space. Any conclusion drawn from the grid is
therefore conservative with respect to the field’s full set of defensible
choices.

#### The discard rule shapes the evidence.

A metric-relative threshold rule may admit circuits unevenly across metrics. The
rule was fixed in advance and no failing item is to be repaired, but the
composition of the surviving set is a property of the reported evidence and
should be stated rather than adjusted for.

#### Resampling can fail for small groups.

Where conditioning groups hold only a handful of specifications, bootstrap
intervals for within-group quantities become unreliable. Such quantities are
better reported as point estimates with diagnostics attached. Constructing a
bias-corrected estimator in response to an unwelcome interval would be the
behaviour this paper warns against.

## 10 Discussion

#### What standardising one axis buys.

Less than it appears. The literature identifies several consequential choices
in circuit discovery, and there is no reason to expect any single one to account
for most of the variation in what is filed. A procedure that standardises one
axis and declares the problem solved would be relying on an assumption it has not
checked.

#### On the instrument.

Interpretability tooling is young, and a library that exposes many discovery
objectives and operators is doing the field a service by making the space of
choices explicit. Robustness of every exported configuration on every canonical
task cannot be taken for granted, and a filing that rests on such tooling should
record which configurations ran and which did not. This is not a criticism of
libraries whose authors are among those calling for the field to address exactly
this [10].

#### What a regulator can take from this.

Not that interpretability evidence is worthless, but that its evidentiary weight
depends on information that filings do not currently contain: the space of
defensible choices and the position of the filed evidence within it. Requiring
that information is cheap relative to the sweep that produced the circuit in the
first place, and it converts an assumed property into one that can be examined.

## 11 Reproducibility

The pre-registration is a timestamped commit tagged prereg-p1-confirmatory
in the public repository. Every configuration traces to a config, a seed and an
environment fingerprint. The grid is run under a single pinned environment, and
the code commit is recoverable by content-addressed comparison of archived
outputs against the git tree. The analysis layer runs without a GPU.

A verification script recomputes quantities from raw outputs and asserts against
what is reported, with pairwise quantities derived by more than one route
including a brute-force computation. A second script checks that every numeric
token in this manuscript matches a value derived from stored outputs. The test
suite is 343 tests.