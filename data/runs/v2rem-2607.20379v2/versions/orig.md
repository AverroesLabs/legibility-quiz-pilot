# Train the Model, Not the Reader: Decodability Supervision for Verifiable
Activation Explanations

Hiskias Dingeto

###### Abstract

Natural-language autoencoders score explanations of hidden activations by
reconstruction. An explanation is deemed faithful if the activation can be
regenerated from it. The resulting score is a single number per explanation,
and it says little directly about the individual claims an explanation makes.
This paper develops instruments for asking claim-level questions about
explanations scored by reconstruction, and a training-time objective for the
models whose activations are being explained. We contribute two audit
protocols: a comparison of grounding and truth, which crosses whether the
reconstruction depends on a claim with whether the claim is true of its
source, and a swap to an independent evaluator, which tests whether wording
the reconstruction depends on is a convention specific to a co-trained
verbalizer/reconstructor pair. We also contribute RECAP (Readable Encodings
via Co-trained Auxiliary Predictors), linear heads trained alongside the
target model against external targets so that designated content remains
linearly decodable from a chosen hidden state. We describe the protocols in
three settings: a released Qwen-2.5-7B verbalizer, a synthetic sandbox with
exact ground truth, and continued pretraining of Pythia-160M. Along the way we
document several measurement pitfalls that can distort audits of this kind,
including oversized counterfactual edits, near-duplicate negative controls,
and numerical failures during setup. For interpretability, the audit frames a
claim-level standard of evidence for activation explanations. For AI safety,
the protocols offer a vocabulary for discussing what an automatic explanation
score does and does not certify, and RECAP offers a way to make chosen content
available to independent readers of a model's internals.

StackOne Technologies

hiskias@stackone.com

## 1 Introduction

Natural-language autoencoders (NLAs) let a model explain its own hidden
states: a verbalizer turns an activation into a text explanation, a
reconstructor turns the explanation back into an activation, and the quality
of the round trip is treated as a test of each explanation’s faithfulness
(Fraser-Taliente et al. 2026). The idea is attractive for oversight because it is
unsupervised and self-checking, with no labels or human in the loop, just a
number attached to every explanation. Released
verbalizer/reconstructor pairs exist for several open models, and a growing
family of methods trains activation readers with related objectives
(Pan et al. 2026; Karvonen et al. 2025; Huang et al. 2025).

The appeal of the reconstruction score lies in what it avoids. Earlier
approaches to explaining hidden states relied on supervised probes that
required labeled concepts, on human inspection of maximally activating
examples, or on model-written explanations judged by another model. Each of
these brings its own costs: labels limit the vocabulary of what can be found,
human inspection does not scale, and judges introduce their own biases. A
reconstruction objective sidesteps these issues by asking only whether an
explanation carries enough information to recover the activation it
describes. Because the score is computed automatically, it can be attached to
every explanation a verbalizer produces, aggregated over corpora, and tracked
during training.

The score is, however, a property of an explanation as a whole. An
explanation typically makes several claims: about the topic of the input,
about specific entities, about particular tokens or attributes. Users of
activation explanations often care about these claims individually. An
auditor asking whether a model has represented a particular fact wants to
know whether a specific sentence in the explanation is true of the activation
and whether the reconstruction actually depends on it. This raises a natural
question that the aggregate score does not answer on its own: how should one
assess the individual claims inside an explanation that has been scored by
reconstruction?

This paper is organized around that question. We first make it precise. A
claim can be assessed along two independent axes. It can be grounded or not,
meaning the reconstruction does or does not depend on it, which we measure by
flipping the claim to a minimally different alternative and recomputing the
score. It can also be true or false of the source the activation was computed
from. Crossing these axes yields four cases, each with a different
interpretation, and a complete audit needs to place every claim in one of
them. Measuring grounding requires care: a counterfactual edit that changes
more than its target claim can make grounding appear several times larger
than it is, so every edit must be validated.

We then consider a phenomenon that the cross makes visible in principle:
wording that the reconstruction depends on even though its stated content is
false. When such wording is a convention between a particular verbalizer and
the reconstructor it was co-trained with, we call it a co-adapted private
code. Distinguishing a private code from an independent evaluator that is
simply weaker requires a second protocol, the swap to an independent
evaluator, together with a diagnostic: if the independent evaluator were
merely worse, grounding of true claims would fall alongside grounding of false
ones.

Finally, we consider the target model itself. The activations a verbalizer
explains are produced by a network trained for language modeling, with no
pressure to make any particular content linearly accessible. RECAP adds such
pressure for designated content, through linear heads on a chosen hidden
state that predict external targets during training (Figure 1b). We describe
the objective, its relation to deep supervision, the design of its targets,
and how it fits with the audit protocols.

Figure 1: The two architectures. (a) A natural-language autoencoder (prior work)
(Fraser-Taliente et al. 2026): AV maps an activation to text, AR maps it back, and
reconstruction quality is the test; only AV and AR train. A single score is
attached to an explanation that may contain several distinct claims. (b) RECAP
(this work) trains the target model. Top: co-trained linear heads keep
designated content decodable against external targets. Bottom: after training,
independent fresh probes (not the heads) and fresh verbalizers can be fit to
the resulting activations.

Our contributions:

•

Claim-level audit. A protocol that flips each salient claim of explanations scored by reconstruction, with validity controls for each edit and transform controls on the whole explanation, applied to a released verbalizer.

•

Private codes. A definition of co-adapted private codes in terms of the comparison of grounding and truth, and a swap to an independent evaluator that distinguishes such codes from evaluator weakness, studied in a sandbox with exact ground truth.

•

RECAP. A training-time objective that adds linear heads with external targets to the target model, with a discussion of target design, its relation to deep supervision, and its implementation on a synthetic sandbox and on Pythia-160M.

•

Monitoring and adversarial protocols. A probe-based discrepancy monitor and a ladder of report-space adversaries of increasing strength, with a frozen-threshold calibration and a held-out vocabulary band that guard against circular evaluation.

## 2 Preliminaries and Experimental Setup

### 2.1 The Reconstruction Test and Audit Protocols

A natural-language autoencoder (NLA) pairs an activation verbalizer
(AV), which generates a text explanation $z$ of a hidden activation $h$ , with
an activation reconstructor (AR), which maps $z$ back to an estimate
$\hat{h}$ . The AV is trained against reconstruction
reward; with target model weights $\theta$ frozen and $h$ its tap
activation, the pair optimizes

| | $\mathcal{J}_{\mathrm{read}}\;=\;\max_{\mathrm{AV},\,\mathrm{AR}}\;\cos\!\big(\mathrm{AR}(\mathrm{AV}(h)),\,h\big),$
| | (1)

and the reconstruction quality of a produced explanation is treated as its
faithfulness test (Figure 1a). Fraser-Taliente et al. (2026) are explicit about this. They present rising
reconstruction as tracking explanation informativeness, and warn that
nothing in the objective forces faithfulness. But that
score is still the number these NLAs report, and the main criterion their users apply. We ask how that score relates to the
individual claims of each explanation. We measure
reconstruction with a centered cosine normalized to the floor. Let $\mathrm{floor}$ be the mean cosine between
reconstructions and mismatched activations (chance level). Then
$\tilde{r}=(\cos(\hat{h},h)-\mathrm{floor})/(1-\mathrm{floor})$ :
0 at chance, 1 at perfect reconstruction. For the released NLA we use the Qwen-2.5-7B layer-20 AV/AR pair (Qwen Team 2025).

The normalization matters for comparisons across settings. Raw cosine
similarities between high-dimensional activations are not centered at zero,
and their typical values depend on the layer, the model, and the anisotropy of
the residual stream. Scores expressed against a mismatched-activation floor
are comparable across these conditions, and a value of zero has a clear
interpretation: the explanation carries no more information about its own
activation than about an arbitrary one.

Grounding. For a claim $c$ in $z$ , let $z_{\neg c}$ be $z$ with $c$
alone minimally flipped (a minimal-pair counterfactual edit, which we call a flip). The claim’s grounding is the reconstruction drop

| | $\Delta\tilde{r}(c)=\tilde{r}(z)-\tilde{r}(z_{\neg c}),$
| | (2)

and $c$ is grounded when $\Delta\tilde{r}(c)\geq\tau$ for a threshold $\tau$ .
Grounding here is the sensitivity of the reconstruction to a claim flip.
It is a property of the reconstructor's reading of the explanation, and we
keep it conceptually separate from any statement about how the target model
itself processes the corresponding content. We check each edit for validity. An edit that changes more than its target claim can inflate apparent grounding severalfold (Appendix A). Crossing
grounding with truth gives four cases. A
claim is grounded or not, and true of the source or not. We
pay particular attention to the case where wording is grounded but false. Here the
reconstruction depends on the wording, although its stated content is false. When
such wording is a convention specific to the co-trained pair (shown by the
swap to an independent evaluator, §3.2), we call it a co-adapted private code. These
labels are operational; no intentional stance is implied. Let $\tilde{r}_{\mathrm{co}}$ and
$\tilde{r}_{\mathrm{ind}}$ be the scores under the co-trained and an independent
reconstructor. The score’s evaluator gap is then
$\tilde{r}_{\mathrm{co}}-\tilde{r}_{\mathrm{ind}}$ .

The four cases have distinct readings. A claim that is grounded and true is
the ideal: the reconstruction depends on it and it correctly describes the
source. A claim that is true but ungrounded is correct but contributes nothing
the reconstructor uses, perhaps because the same information is carried
redundantly elsewhere in the explanation. A claim that is false and ungrounded
is a confabulation the score neither rewards nor punishes. A claim that is
grounded but false is the most informative case for an auditor, since it
indicates that the reconstruction relies on wording whose literal content
does not match the source.

Terms. The target model’s activation is read at a chosen
layer and position, the tap. An activation can be read by the reconstructor AR or by a probe, a freshly fitted
linear decoder (Alain and Bengio 2017; Belinkov 2022). Gist is
the input’s overall meaning, as opposed to its specific facts. Content is
designated when an auxiliary predictor is co-trained to predict it
(§5). Content is decodable
when a freshly fitted linear probe can read it. Faithfulness in this paper is always claim-level. We judge each individual claim in an
explanation as true of the activation and grounded in it, rather than judging the explanation as a whole. We do not use
faithfulness to mean whether explanations describe the model’s downstream computation, or grounding to mean grounding in the world.

#### Settings.

We work in three settings. The first is a released
Qwen-2.5-7B layer-20 NLA on in-distribution web text ( $n{=}1{,}517$ audited
claims). The second is a sandbox of two synthetic scenarios, a story and a marketplace,
each built by filling the slots of a fixed template, so that the true content of every activation is known exactly and computable without a judge; here the tap position fixes what the activation can hold (retained, faded, or unread slots). The third is
continued pretraining of Pythia-160M (Biderman et al. 2023) with self-supervised heads, where tax
is held-out language-modeling loss minus a shared control. Full protocols, scenario
grammars, seeds, and hyperparameters are in Appendices D and K.

The three settings play complementary roles. The released NLA is the object
of practical interest: it is the kind of system practitioners actually
deploy, but its inputs are natural text and the truth of its claims can only
be judged approximately. The sandbox reverses this trade-off. Its inputs are
artificial, but every claim has an exact truth value, and the tap position
determines in advance which content the activation could possibly contain.
The Pythia setting sits between the two: inputs are natural text drawn from a
standard pretraining corpus, while the targets used to supervise and audit are
simple functions of the text, such as which content words occurred in a
trailing window, so their truth can be computed exactly without a judge.

## 3 Auditing Reconstruction-Scored Explanations

Auditing an explanation at the level of its claims requires three
ingredients: a way to identify the claims an explanation makes, a way to
measure how much the reconstruction depends on each one, and a way to decide
whether each claim is true. The three settings supply these ingredients in
different ways. On the released NLA, claims are identified and flipped by a
judge, and truth is assessed against the input passage. In the sandbox, claims
are parsed by exact scan over closed vocabularies, flips are same-type slot
substitutions, and truth is read off the generating tuple. This section
describes both audits and the controls each requires.

### 3.1 Counterfactual Audit of a Released Verbalizer

The audit of the released NLA proceeds explanation by explanation. For each
explanation, a judge model lists the specific claims an LLM reader would find
salient: named entities, quantities, distinctive attributes, and similar
concrete assertions, as opposed to generic statements about topic or register.
For each listed claim, the judge produces a minimal-pair edit that flips that
claim alone to a plausible alternative. The edited explanation is then scored
by the frozen reconstructor, and the drop in $\tilde{r}$ defines the claim's
grounding (Eq. 2). Grounded fractions are reported at three thresholds,
$\tau=0.02$ / $0.05$ / $0.10$ , so that conclusions do not hinge on a single
cutoff.

Two properties of the instrument shape how its output should be read. First,
the flip test has limited sensitivity. A reconstructor may represent a
specific fact only weakly in its output, so a flip can leave the score nearly
unchanged even when the fact is encoded. Grounded fractions are therefore best
interpreted as lower bounds. A useful sanity check appends a topic known to be
encoded and re-ranks candidate topics by score, which indicates how much
encoded content the instrument recovers raw. Second, the edits themselves are a
source of error: a judge that rewrites surrounding material can create
spurious drops. We address this with per-edit validity filtering (Appendix A).

The flip audit is complemented by transforms applied to the whole
explanation. Paraphrasing, synonym substitution, function-word masking, word
shuffling, and content-word masking each remove a different kind of
information while leaving others intact. The profile of scores across these
transforms indicates whether the reconstructor reads the explanation at the
level of meaning or of surface form. A reconstructor that decodes meaning
should be insensitive to rewording and sensitive to the removal of content;
one that decodes a surface-form convention should show the opposite pattern.
The profile is also a useful complement to the per-claim flips, since a
reconstructor that relies on overall meaning may assign little weight to any
single specific claim. The protocols for both kinds of control, together with
bootstrap intervals and a noise floor estimated from flips that carry no
grounding signal, are described in Appendix E.

A further question is where in the network the content corresponding to a
claim resides. The audit reads a single layer, but content absent from a
reconstruction-style readout at that layer might be present elsewhere, or
might be visible to a different kind of reader. We therefore compare the
model's own output head (a logit lens) with a generic ridge probe of matched
capacity across a range of layers. This comparison speaks to whether an
explanation's omissions reflect the information available at the tap or the
particular readout being used.

One could also try to move the verbalizer toward producing grounded claims,
for instance by rejection sampling or by policy-gradient training against a
grounding reward scored at the clause level. Any such approach must be
evaluated with an instrument independent of its training reward, since a
policy can raise a clause-level proxy by restructuring sentences rather than
by changing the claims they make. The independent claim-level audit is
designed for exactly this purpose. Because the released setting provides only
approximate truth, we turn next to a setting where every quantity is exact.

### 3.2 Audit Under Synthetic Ground Truth

The sandbox makes every quantity in the grounding–truth cross exact. Each
story is generated from a tuple of slot values, so a stated slot value is true
if and only if it matches the tuple. Each slot type has a fixed vocabulary, so
a flip is a substitution of one vocabulary item for another of the same type,
and no judge is involved. And because the tap is read at a known position,
each slot falls into one of three classes: retained (mentioned before the tap
and reused afterward, so the target model has reason to carry it), faded
(mentioned before the tap only), and unread (mentioned only after the tap, so
the tap activation provably cannot contain it). The unread slots serve as an
impossibility control: any claim about them that the reconstruction depends on
cannot reflect information in the activation.

Within this setting the comparison of grounding and truth acts as a detector.
A claim about a retained or faded slot that is grounded and true reflects
content the verbalizer has conveyed. A claim about an unread slot that is
grounded can only reflect wording the reconstructor has learned to rely on,
since its content cannot be in the activation. More generally, wording that is
grounded but false marks a candidate private code, and codes can in principle
occupy several slots at once. When the reconstructor relies on wording
regardless of whether its content is true or even possible, the explanation is
functioning less as a description of the activation and more as a channel
between the two networks.

The swap to an independent evaluator determines whether such a channel is
private to the co-trained pair. The independent evaluator is a reconstructor
trained without exposure to the audited pair's outputs. If grounding on
suspect slots reflects a convention between the pair, it should fall under
the independent evaluator. The diagnostic that separates this from evaluator
weakness is the behavior of true claims: an evaluator that is merely worse
would lose grounding on true claims as well, whereas a convention-specific
channel would leave true-claim grounding intact while removing grounding on
the conventional wording. We strengthen the test at the phrasing level by
training a further evaluator on a disjoint paraphrase family, with different
frames and slot order, sharing only the slot words. Agreement between the two
independent evaluators indicates that the result does not depend on any
particular template.

The sandbox also separates two distinct reasons a verbalizer might fail to
state content truly. A storage gap arises when content is absent from the
activation, as for unread slots. A readout gap arises when content is present,
in the sense that a fresh probe can decode it from the target model, but the
verbalizer does not convey it. Comparing per-slot decodability, measured by
fresh probes on the target model, with per-slot truth rates of the verbalizer
distinguishes the two. This distinction carries over directly to the
discussion of RECAP below, where the target model is trained so that the
activation holds designated content and the question becomes whether a
verbalizer conveys it.

## 4 Representation Collapse and Probe Drift

Before introducing RECAP we discuss two mechanisms that any intervention
involving probes or reconstructions of a model under training must contend
with. Both are familiar from the representation-learning literature, and both
inform the design choices of §5.

The first is representation collapse. A tempting objective on the target side
would move activations toward their own reconstruction, so that the
reconstruction pipeline and the model's computation come to agree. Objectives
of this self-referential kind are known to admit degenerate solutions. In
non-contrastive self-supervised learning, a network trained to make two views
of an input agree can satisfy the objective by discarding information, and
the literature has analyzed the architectural and optimization ingredients
that prevent this (Chen and He 2021; Tian et al. 2021). An analogous concern
applies here. A reconstruction is a lossy projection of the state, and
increasing similarity between a state and its reconstruction can be achieved
by reducing variation the reconstruction does not capture. Whether this
actually removes information, rather than merely rotating it into a subspace
a particular reader cannot see, can only be determined by refitting fresh
probes after training, not by reusing readers fitted beforehand.

The second is representation drift during continued training. A probe fitted
to a model at one checkpoint reads a particular linear direction. As training
continues, the model may re-express the same information in a different
basis, so the probe's accuracy can decline even if nothing has been lost.
Networks are known to learn representations that are equivalent up to a
transformation (Lenc and Vedaldi 2015; Kornblith et al. 2019; Moschella et al. 2023),
and the natural repair for drift of this kind is a map fitted on paired
activations, which requires no labels. The expressiveness of the map needed
indicates the nature of the change: a rotation suffices if the
representation has merely been re-oriented, whereas a non-orthogonal map of
some rank is needed if its geometry has changed. Duan (2026) concurrently
studies drift and realignment of this kind on 2B-parameter and 7B-parameter
models.

The two mechanisms call for different responses. Drift is a property of the
reader: the information is present, and the reader must be recalibrated.
Collapse is a property of the representation: the information is gone, and no
recalibration recovers it. Distinguishing them requires the same tool in both
cases, namely fitting fresh probes after training and comparing them with
stale ones, and, where stale probes fail, testing whether a label-free map
restores them.

This distinction has a practical consequence for monitoring. Probe-based
monitoring of a model that is still training should include an explicit
recalibration procedure, and evaluations of any method that changes the
target model should rely on probes fitted after training rather than on
readers carried forward from earlier checkpoints. A related observation
follows from representational equivalence: if representations of different
models differ by an affine map, adapter-based transfer of probes across models
is a natural avenue to explore (Zhao et al. 2026).

## 5 The RECAP Objective

### 5.1 Method

RECAP (Readable Encodings via Co-trained Auxiliary Predictors) adds
linear auxiliary predictors (heads) during training of the target model, so that designated content stays decodable from
selected hidden states. Each head reads the hidden state at the tap layer and predicts an
external target, namely slot values in the sandbox or self-supervised text functions at
scale. We then add the head loss to the language-modeling loss. With head
parameters $\phi$ and external targets $y$ ,

| | $\mathcal{J}_{\mathrm{model}}\;=\;\min_{\theta,\,\phi}\;\mathcal{L}_{\mathrm{LM}}(\theta)\;+\;w\,\mathcal{L}_{\phi}\big(h(\theta),\,y\big),$
| | (3)

Here $w$ weights the head loss against the language-modeling loss. $\mathcal{L}_{\phi}$ is a class-balanced binary cross-entropy for whether each content word is present, plus a cross-entropy for which token it is. Balancing weights positives by the inverse of their base rate, so that rare targets are not dominated by the abundant negative class.

Eq. 3 and Eq. 1 act on different parameters.
$\mathcal{J}_{\mathrm{read}}$ optimizes the verbalizer and reconstructor against a frozen
model’s activations and never touches $\theta$ . In contrast,
$\mathcal{J}_{\mathrm{model}}$ trains $\theta$ itself, adding a term to the
language-modeling loss for ground truth that downstream readers may wish to recover.
The two objectives are not mutually exclusive: a verbalizer and reconstructor
can be trained with Eq. 1 on activations from a model trained with Eq. 3,
which is how fresh NLAs are fit in our experiments.

The choice of targets determines what RECAP designates. In the sandbox, the
natural targets are the slot values that precede the tap, since these are the
facts a verbalizer might be asked to state. At scale there are no slots, so we
use self-supervised functions of the text: whether each of the $K$ most
frequent content words occurred in a trailing window, and which token occurred
a fixed number of positions back. These targets have three useful properties.
They are computable exactly from the input, so no labeling is required. They
are external to the model, so the objective does not refer to the model's own
outputs or reconstructions. And they correspond directly to claims an
explanation might make, so the same functions can be used to judge the truth
of a verbalizer's statements.

Heads that co-train with the model might in principle co-adapt with it, so
throughout we report decodability as measured by independently fitted probes
rather than by the heads. Probes are fit on held-out data after training,
using the same recipe for RECAP and control models.
RECAP is decodability supervision in the deep-supervision lineage
(Lee et al. 2015; Szegedy et al. 2015). In classical deep supervision,
auxiliary heads on intermediate layers predict the task label to improve
gradient flow and regularize training; here the auxiliary target is content
the practitioner wishes to remain linearly accessible, which need not coincide
with the training task. RECAP also differs from concept bottlenecks, which
route all information through named concepts: the heads read the hidden state
without constraining what else it carries.

### 5.2 Sandbox Results

In the sandbox, RECAP target models are trained from scratch with heads on
the layer-4 tap predicting the values of the pre-tap slots. Control target
models are trained with the language-modeling loss alone on the same data
with the same seeds. Both kinds of model are then treated identically by the
rest of the pipeline. Fresh probes are fit on held-out taps to measure
per-slot decodability. Fresh NLAs are trained with the same recipe,
including a cold-start SFT on gold descriptions and alternating rounds of
reconstructor fitting and rejection sampling against reconstruction reward.
Each NLA is audited with the comparison of grounding and truth under its
co-trained reconstructor and under an independent evaluator.

This design makes the target model the only variable that differs between
conditions. Because the NLA recipe, the gold descriptions used for the
cold start, the audit vocabulary, and the evaluators are shared, any
difference between conditions in the audited truth or grounding of a slot can
be attributed to the activations the NLA was trained on rather than to the
verbalizer's training.

The unread slots play a particular role here. Their values never appear before
the tap, so no training objective can make them decodable from it. Heads for
these slots are not part of the RECAP objective, and fresh probes for them
provide a check that the probing procedure does not report spurious
decodability. Claims about unread slots likewise provide a check on the
verbalizer: a well-behaved verbalizer should either abstain on them or state
them without grounding.

Several further variants extend the basic design. Partial designation
supervises only one or two of the three pre-tap slots, so that designated and
undesignated content can be compared within the same model. The language-modeling
cost of designation is measured as held-out loss relative to the control. The
marketplace scenario, with its variable tap position and differently sized
vocabularies, tests whether conclusions drawn from the story scenario depend
on its particular structure. Seeds and scenario parameters are listed in
Appendix K.

Figure 2: Decoding content from a real model’s hidden state (one held-out Pile (Gao et al. 2020)
passage; fresh probes read the layer-6 state at the marked position). The
probes target the trailing-window content words and the offset-8 token for
both a RECAP-trained model and a control. Same probe recipe and
data; only the target model differs.

### 5.3 Scaling to Pythia-160M

We next consider RECAP during continued
pretraining of Pythia-160M, following the setup of §2.1. Heads read the
layer-6 hidden state and predict presence targets for the $K$ most frequent
content words in a trailing 64-token window, together with an identity target
for the token eight positions back. Each run continues pretraining on 500M
tokens of the uncopyrighted Pile with identical data and order, and a shared
control run with no heads provides the baseline for both tax and
decodability. Figure 2 illustrates the probing setup on a single passage.

Designated content is measured by fresh ridge probes, fit on held-out probe
sequences and evaluated on disjoint blocks, and grouped into rank buckets by
word frequency. A never-designated bucket of words outside the supervised
vocabulary provides a within-model control: if the heads merely raised
decodability across the board, this bucket would move as well. Tax is the
held-out language-modeling loss of a run minus that of the control on 20M
tokens, and where multiple seeds are available we report paired differences
with bootstrap intervals.

Target design is a central consideration at this scale. With many rare
targets, a presence loss that weights positives and negatives equally can be
reduced substantially by predicting absence everywhere, since most words are
absent from most windows. A training loss that falls for this reason says
nothing about whether the hidden state carries the targets, and only an
independent probe evaluated with a threshold-free metric such as AUC can
reveal the difference. This motivates class balancing, and more generally the
rule of choosing targets whose loss cannot be minimized without genuine
discrimination between classes. Identity targets, which ask which of $K$
tokens occurred, are discriminative by construction and are less exposed to
this concern.

We vary the number of targets ( $K{=}8$ , $64$ , $512$ ), the auxiliary weight
$w$ , the form of the presence loss, the head capacity (linear versus a small
MLP), and the tap layer (6 versus 9). Each of these trades off the strength of
supervision against its cost in language-modeling loss and its demands on the
capacity of a $d{=}768$ hidden state. The ablation index in Appendix I lists the
full set of factors and where each is described.

### 5.4 Verbalizer Audit at Scale

A further question is whether content made decodable by RECAP reaches a fresh
verbalizer, as the sandbox design is meant to test. On Pythia-160M we train an AV/AR pair on these taps with the same procedure as the released NLA, using a 0.5B-parameter reader with LoRA adapters and a linear adapter from the 768-dimensional tap. Explanations are generated from a fixed template stating trailing-window content words and the offset-8 token, so they can be parsed exactly and audited against ground truth from the text. A stated word is true if it occurs in the trailing window, and grounding flips swap a stated word for another vocabulary word.

The audit reports, for each condition, the truth rate of stated words, their
grounded fraction at each threshold, the difference between grounded and true
rates, and recall of the content words actually present. Recall is needed
alongside truth because a verbalizer can raise its truth rate simply by stating
fewer words, so a comparison of truth rates alone could reward selectivity
rather than conveyance. We also vary reconstructor strength and target-model
seeds, since a weak reconstructor could fail to reward true content in either
condition.

Decodability by a probe and verbalizability by a generator are distinct
properties. A probe sees the full hidden state and outputs a calibrated score
for each target; a verbalizer must serialize its reading into a short
sequence of tokens, committing to a few claims and omitting others. To compare
the two on equal terms we place the verbalizer on the probe's
precision–recall curve, evaluating the probe at the verbalizer's own recall.
We also redesign the reconstructor to test where any shortfall arises:
standardized injection across multiple tokens, MLP capacity, and a presence
head trained at the decode point on whether each word occurred in the window
(an external fact, not a learned monitor). A matched control under the
identical head distinguishes content contributed by RECAP from content the
head itself might induce. Further reader-side variations, including reader
scale, reconstruction optimization, a direct truth reward, injection
bandwidth, and explicit negative supervision, are listed in Appendix I.

A related practical question is whether decodability, once trained in,
persists. Real post-training pipelines typically drop auxiliary objectives,
so we fine-tune RECAP and control checkpoints without heads and re-probe
them at intervals, and we test an intermittent schedule that applies the head
loss on a fraction of steps (Appendix C).

## 6 Probe-Based Monitoring

To make the comparison between probes and verbalizers concrete, we built
a simple monitor in the sandbox. Fresh probes, one per slot,
read the tap activation while the verbalizer states slot values. Any claim
that a confident probe ( $p{>}0.9$ ) contradicts is flagged as a candidate
discrepancy, and exact ground truth then scores the monitor
(full protocol in Appendix B).

The monitor has three possible responses to each claim: agree, flag, or
abstain. It abstains when no probe is confident, which happens whenever the
corresponding content is too weakly decodable for a reliable readout. This
makes the monitor's coverage a direct function of decodability. Where content
is decodable, the monitor can adjudicate claims about it; where it is not, the
monitor correctly declines to.

Post-tap claims illustrate the boundary. Content the activation provably lacks
cannot be adjudicated by reading the activation, so a well-calibrated monitor
should abstain on all such claims. Catching confabulations about post-tap
content is the job of the comparison of grounding and truth (§3), which uses
ground truth rather than the activation. The two tools therefore partition the
audit: one reads internals, the other compares statements with the source.

At scale we use a presence probe on the Pythia taps in the same role. Because
the base rate of false claims among a verbalizer's statements can be low, the
per-claim coverage metric used in the sandbox is less informative there, and
the natural summary is the AUC with which the probe ranks true claims above
false ones, together with the precision and recall of a thresholded flag.

#### Surviving a score-gaming adversary.

A monitor is useful in adversarial settings only if it resists manipulation
of the inputs it observes. We consider an adversary who can edit the
explanation text but not the activation. The adversary takes a true stated
word in an honest explanation and edits it into an absent one, choosing the
edit according to a strategy of increasing strength: a random absent word, the
embedding-nearest absent word, the absent word that maximizes the recomputed
reconstruction score, and, as a worst-case bound outside the threat model, a
probe-aware choice that assumes the monitor leaks (full protocol in
Appendix F). For each rung we recompute $\tilde{r}$ through the real
reconstructor and score the lie with the presence probe trained on external
truth.

The ladder separates two questions. The first is how much a score-maximizing
edit can reduce the penalty the reconstruction score assigns to a lie,
compared with a random lie. The second is whether a probe reading the
activation can still distinguish lies from true statements under each rung. A
probe reads a signal that predates the editable report, so the adversary
cannot alter what it sees, but it can still choose which words to lie about,
and its choices may concentrate on words the probe reads poorly. Running the
ladder on both RECAP and control models shows how this interacts with
decodability. Our threat model covers edits to the explanation text, not
activation-level attacks (Bailey et al. 2024).

Two further controls guard against circular evaluation. Because the monitor
probes the same presence predicate that RECAP supervises, it could in
principle merely re-measure a trained margin. We therefore calibrate the flag
threshold on a disjoint split rather than on the evaluation set, and we repeat
the ladder on a held-out band of words the model was never trained to keep
decodable (ranked 64–128). These are described in Appendix F.

## 7 Related Work

#### Activation verbalization and its critiques.

Beyond the NLA recipe (§2) (Fraser-Taliente et al. 2026), a fast-growing family
improves the reader. Examples include supervised question-answering decoders
(Pan et al. 2026), general-purpose activation explainers
(Karvonen et al. 2025; Bauer et al. 2026), interpreter assistants
(Huang et al. 2025), readouts that need no training (Ghandeharioun et al. 2024), and
cross-model adapters (Zhao et al. 2026). Sparse autoencoders are the unsupervised reader family
(Huben et al. 2024). Model-produced explanations pass plausibility checks without faithfulness
in chain-of-thought (Turpin et al. 2023; Lanham et al. 2023). Causal
tests of neuron explanations found high error and little
efficacy (Huang et al. 2023). Counterfactual faithfulness tests
prefigure our flip protocol (Atanasova et al. 2023). Existing critiques of verbalization are
benchmark-level. Many tasks are solvable without the target’s internals
(Li et al. 2025). We differ on two axes. Where prior critiques are benchmark-level, we audit the
reconstruction test itself, per claim and counterfactually, on a released NLA. Where prior
work improves the reader, we also consider training the target model.

The broader question of how to evaluate natural-language explanations of
model internals has a long history in the interpretability literature.
Simulation-based evaluations ask whether a reader of an explanation can
predict a unit's behavior; plausibility evaluations ask whether humans find an
explanation convincing; and intervention-based evaluations ask whether
manipulating the described feature has the described effect. Reconstruction
scoring belongs to a fourth family that evaluates explanations by their
information content with respect to the activation. Each family captures a
different notion of what makes an explanation good, and our claim-level
protocols are intended to complement reconstruction scoring rather than to
replace it.

#### Scalable oversight.

Reconstruction scoring sits within scalable oversight, which substitutes cheap
automatic signals for costly human judgment of model behavior and internals
(Amodei et al. 2016; Bowman and others 2022). A parallel line, AI control,
designs protocols that stay safe even when the model tries to subvert them
(Greenblatt et al. 2024). Both lines share a concern with the robustness of
automatic signals: a signal that is cheap to compute is useful for oversight
only insofar as its relation to the property of interest survives optimization
pressure. Our adversarial ladder is a small instance of this concern applied to
explanation scores, in the spirit of work on obfuscation under monitor
pressure (Baker et al. 2025).

#### Steganography in learned channels.

Objectives in the reconstruction family have hidden information in their channels
since CycleGAN (Chu et al. 2017). In LLMs, encoded reasoning arises in
chain-of-thought under optimization pressure (Roger and Greenblatt 2023) and
generalizes under process supervision (Skaf and others 2025). Misspecified
incentives yield steganographic collusion
(Mathew et al. 2025), tool use widens the channel (Rippin et al. 2026),
and probing internals detects collusion (Rose et al. 2026).
Optimization can obfuscate activations against latent monitors
(Bailey et al. 2024), and training against a chain-of-thought monitor yields
obfuscated reward hacking rather than honesty (Baker et al. 2025).
All of these channels are induced by incentives,
process supervision, or an adversary. Our notion of a co-adapted private code
concerns the standard reconstruction recipe inside the interpretability
pipeline, with no steganography objective or communication reward, and the
swap to an independent evaluator is our tool for detecting it. Supervising the
target model against external ground truth is complementary to the advice in
that literature not to optimize directly against a monitor, since the
supervision signal is ground truth rather than the monitor's judgment.

#### Representation drift, stitching, and monitor staleness.

Networks learn equivalent representations up to a transformation
(Lenc and Vedaldi 2015; Kornblith et al. 2019; Moschella et al. 2023); affine maps
transfer features across language models (Chen et al. 2025);
pretraining dynamics are stable across seeds (van der Wal et al. 2025); tuned
lenses are affine readers whose calibration depends on the checkpoint they were fit to
(Belrose et al. 2023). Concurrently, Duan (2026) report
staleness and its repair without labels on 2B- and 7B-parameter models.
Retraining probes also restores
detectability after fine-tuning that is aware of the probe (Wehner and Fritz 2025). Our
discussion in §4 draws on this literature to separate drift, which concerns
readers, from collapse, which concerns representations, and to motivate
evaluating every intervention with probes fitted after training.

#### Deep supervision and intrinsic interpretability.

Auxiliary heads on hidden layers are deep supervision
(Lee et al. 2015; Szegedy et al. 2015); there the target is the task
label; our target is external content to keep decodable. Concept bottlenecks route all information flow through named concepts
(Koh et al. 2020; Sun et al. 2025), concept whitening constrains the geometry
(Chen et al. 2020), codebook features discretize the representation
(Tamkin et al. 2023), gradient routing localizes computation
(Cloud et al. 2024); we add linear heads without constraining
flow. Supervision on the explanation side can track behavioral change
(Guo et al. 2026); ours acts on the target model.

Intrinsic interpretability methods differ in how strongly they constrain the
model. At one extreme, bottleneck architectures force every prediction through
an interpretable layer, which guarantees that the named concepts are used but
can cost task performance and restricts the representation to the chosen
vocabulary. At the other, post-hoc readers leave the model untouched. Linear
auxiliary heads sit near the lightly constrained end: they add a soft pressure
for chosen content to be linearly present at a chosen location, while leaving
the model free to carry other information and to compute as it likes. This
places RECAP closer in spirit to regularization than to architectural
constraint.

## 8 Discussion and Limitations

Implications for oversight. Scoring explanations by reconstruction is a form
of scalable oversight, a cheap automatic signal meant to stand in for expensive
human checking of a model’s internals. Signals of this kind are most useful
when their relation to the property of interest is well characterized. The
protocols in this paper are intended to supply part of that characterization
at the level of individual claims: the grounding–truth cross places each claim
in one of four interpretable cases, the swap to an independent evaluator
separates co-adapted conventions from evaluator weakness, and the adversarial
ladder probes how a score responds to deliberate editing. RECAP adds a
complementary lever on the model side, by asking that designated content be
available to independent readers. These are components of an oversight
toolkit rather than a solution to oversight, and their value depends on how
well the designated content matches what an overseer needs to check.

Methodological lessons. Several of the most consequential choices in this
work concern measurement rather than modeling. Counterfactual edits must be
validated, since oversized edits inflate apparent grounding. Negative controls
built from adjacent examples can leak information through near duplicates.
Training losses on imbalanced targets can fall without any increase in
discrimination, so decodability should be assessed with independent probes
and threshold-free metrics. And numerical defaults in common libraries can
silently corrupt training. We report these because audits of learned
explanations are only as reliable as the instruments behind them.

Scope and limitations. Our evidence comes from a synthetic sandbox, one
released NLA (Qwen-2.5-7B), and continued pretraining of one small model
(Pythia-160M), mostly at single seeds; frontier scale is untested. Four
limitations constrain our conclusions. First, our sandbox and scale audits cover
only closed-vocabulary claims parsed from a template, and the monitor covers only
designated content; open-ended faithfulness is unmeasured. Second, RECAP must be co-trained
in and cannot be retrofitted onto a frozen model, and it depends on choosing a
good target. A poorly chosen one can report success on its training loss while decoding nothing.
Third, decodability is not verbalizability. Content a probe can read need not be
content a free-form verbalizer states. Fourth, RECAP concerns what an activation stores, not what the model
uses. Designated content can be present yet behaviorally inert, and a model could route
computation around a monitored subspace (Mahinpei et al. 2021; Margeloiu et al. 2021).

Success for verbalization tested by reconstruction is not simply a
higher $\tilde{r}$ . It means explanations whose
specific claims are checked against the target model, calibrated abstention
where content is absent, and monitors with recalibration protocols.

## 9 Conclusion

We examined the reconstruction test behind natural-language activation
explanations from the perspective of individual claims. We introduced a
counterfactual flip audit with per-edit validity controls and transform
controls, a comparison of grounding and truth that places every claim in one
of four cases, and a swap to an independent evaluator that identifies
co-adapted private codes. We described RECAP, which trains the target model
with linear heads against external targets so that designated content remains
linearly decodable, and we laid out its implementation in a synthetic sandbox
and during continued pretraining of Pythia-160M, together with a probe-based
monitor and a ladder of report-space adversaries. The methodological lesson
running through the work is that learned tests of a model's explanations
deserve the same scrutiny as the explanations themselves. For oversight of
increasingly capable models, explanations whose individual claims can be
examined against internals are a firmer basis for trust than a single score
attached to each explanation.

## Appendix A Edit Validity for Minimal-Pair Flips

Grounding measurements are only as good as the edits that produce them. An
edit is valid if it flips exactly one claim and changes nothing else.
We audited two edit generators on the released NLA: a closed-model judge
produced valid edits 68.4% of the time, while the open judge used throughout
the paper (Gemma-4-12B, pinned revision, temperature 0) produced valid
edits 99.3% of the time. The invalid edits were predominantly
oversized: they rewrote surrounding material along with the target
claim, and the extra damage inflated apparent grounding roughly sevenfold.
All grounded fractions in the paper therefore use the open judge with
per-edit validity filtering; the main text reports this in one sentence and
the audit artifacts persist per-claim reconstruction deltas. The same
head-to-head also measured a judge-free mask-and-rescore signal against
the flip instrument (agreement AUC 0.997 under the validated judge), used
in Appendix G.

## Appendix B Safety Experiment 1: The Discrepancy Detector

#### Question.

If an auditor can read a target model’s activation with
probes and can also read what the verbalizer says, disagreements
between the two are candidate lies. How should such a detector be
constructed, and what governs the claims it can adjudicate?

#### Protocol.

Sandbox, scenario 1, seed 1, 400 held-out taps per
target model.
For each slot type we fit a fresh multinomial logistic probe (linear,
256-dimensional input) on the probe split of taps, reading the slot value
from the tap activation. On evaluation taps, the verbalizer generates an
explanation; the first stated word of each slot type is compared with the
probe’s prediction.
A claim is flagged when a probe with confidence above 0.9 disagrees
with the stated word, and abstained when probe confidence is below
threshold. Exact tuple ground truth then scores every decision. Precision is
the fraction of flags that are genuinely false claims; recall is the fraction
of false claims flagged; abstention is the fraction of claims the probe could
not adjudicate at threshold. Claims are split into pre-tap (slots the
activation can hold) and post-tap (slots it provably cannot).

#### Findings.

The structure of the detector determines its behavior in a few
respects that hold independently of any particular run. (i) Its precision is
governed by the calibration of the probes: a flag requires a confident probe
to disagree with the stated word, so false alarms arise only where a probe is
confidently wrong. (ii) Its coverage is governed by decodability: a slot that
is too faded for a confident readout yields abstentions rather than flags, so
the fraction of claims the detector can adjudicate is bounded by how much of
the relevant content is linearly readable from the tap. (iii) Post-tap claims
should be abstained on by construction, since content the activation provably
lacks cannot be adjudicated by reading the activation; catching those
confabulations is the grounding cross’s job. The detector and the cross are
thus complementary, and the comparison between standard and RECAP target
models is a comparison of how much pre-tap content each makes available for
adjudication.

#### Caveats.

One scenario, one seed, one threshold (0.9); “decodable”
means linearly readable throughout.

## Appendix C Safety Experiment 2: Legibility Persistence Under Head-Free
Fine-Tuning

#### Question.

Does the decodability RECAP trains in
survive later training that drops the heads, as real post-training pipelines do?

#### Protocol.

Starting from the RECAP Pythia-160M checkpoint
( $K{=}64$ , class-balanced presence, $w{=}0.1$ ) and the control checkpoint,
we fine-tune both with the language-modeling loss only (no heads) on 131M
tokens of a deliberately shifted corpus (FineWeb (Penedo and others 2024), versus the Pile training
distribution). Training uses batch $64\times 1024$ , learning rate $10^{-4}$ , and
2,000 steps, with snapshots at 250/1,000/2,000. Every snapshot is re-probed with the
standard fresh-probe pipeline: ridge probes fit and evaluated on disjoint
held-out sequence blocks, so the measurement is immune to reader staleness
by design: probes are refit at each snapshot, never carried forward. We track
mean AUC on the two designated rank buckets, 64-class identity accuracy, and
the never-designated bucket in the RECAP run.

#### Findings.

The design of this experiment determines what its measurements can
show. Because probes are refit at every snapshot, any decline in readability
reflects a change in the representation itself (the linear decodability of
the content), not the staleness of a frozen reader discussed in §4. The
control run and the never-designated bucket provide two baselines against
which any change in the designated buckets can be compared, so that effects
specific to designated content can be separated from general effects of
fine-tuning on a shifted corpus. The question the experiment addresses is
therefore whether decodability should be regarded as a one-time property
established during training or as a property that requires maintenance, and
the follow-up runs below are designed to separate the contribution of corpus
shift from that of removing the heads.

#### Follow-up runs.

Two further runs address the obvious alternative explanations.
A head-free fine-tune on unseen same-distribution data (fresh Pile
tokens with no overlap with training, validation, or probe data) separates
effects of removing the heads from effects of corpus shift. A second run
applies the head loss every tenth step during the shifted fine-tune, testing
whether intermittent supervision at one tenth the frequency is sufficient to
maintain decodability. The original training logs are also informative about
how quickly readability is established at $w{=}0.1$ , which bears on whether
establishing it after the fact would be a quick repair or a substantial
training pass.

#### Caveats.

Single seed per run; 2,000 steps; the intermittent
run’s own language-modeling tax was not separately measured (the claim is
about supervision frequency).

## Appendix D The Synthetic Scenarios

Both sandbox scenarios are templated grammars over closed vocabularies: a story
is generated by sampling one value per slot and formatting a fixed template
with random filler sentences, so every story has an exact ground-truth tuple
and every audit quantity is computable without a judge.

#### Scenario 1 (story).

Five slots: name (20 values), place (12),
object (12), activity (8), ending (8). Name, place, and object are mentioned
before the tap; activity and ending only after it. The tap sentence is fixed
text, so the tap sits at the same token offset in every story. The slot
design fixes what the tap activation can contain: the name is reused
after the tap (the target model must carry it across, retained); place and
object finish before it (faded); activity and ending never precede it
(unread, provably absent, the impossibility control).

#### Scenario 2 (marketplace).

A structure-shifted scenario for external
validity: five differently-sized vocabularies (customer 20, shop 10, item
14, payment 6, outcome 10), two template variants per story, and a
variable tap position (template-dependent filler length before the
tap marker). Customer, shop, and item precede the tap; payment and outcome
follow it. Everything else in the pipeline is scenario-parameterized and
unchanged.

| | Scenario 1 (story) | Scenario 2 (market)

| slots (pre/post tap) | 3 / 2 | 3 / 2

| vocabulary sizes | 20/12/12/8/8 | 20/10/14/6/10

| templates | 1 | 2

| tap position | fixed | variable

Table 4: The two synthetic scenarios.

Target models are 8-layer, $d{=}256$ transformers trained from scratch on
120,000 stories; the tap is read at layer 4 of 8. Scenario-2 target models
reach the same training perplexity as scenario 1 (1.304), so differences
between scenarios cannot be attributed to one being harder to model.

Figure 3 shows a representative held-out story and the
explanations produced by the standard and RECAP target models.

Figure 3: One representative held-out story, two target models (seed 1). Each stated
specific is marked true or false against the exact tuple and flip-grounded
( $\Delta\tilde{r}$ ; grounded at $\tau{=}0.05$ ), illustrating how the
grounding–truth cross is read off a single explanation for retained, faded,
and unread slots.

## Appendix E Released-NLA Audit Details

#### Setup.

All released-NLA measurements use the public Qwen-2.5-7B
layer-20 AV/AR pair on in-distribution web text, with greedy decoding and
direction-only reconstruction. Minimal-pair edits come from a pinned-revision
open judge with per-edit validity filtering (Appendix A); every
per-claim reconstruction delta is persisted ( $n{=}1{,}517$ base-NLA claims).
Grounded fractions are computed at $\tau{=}0.02/0.05/0.10$ for the base NLA
and for two trained variants (rejection-sampling and policy-gradient), with
bootstrap 95% intervals over claims. The flip-noise floor is estimated from an
empirical null of flips that carry no grounding signal: the 95th percentile of
$|\Delta\tilde{r}|$ is 0.017, and wrong-direction drops of ${\geq}0.02$ occur at rate
$2{\times}10^{-4}$ , so the smallest threshold sits above the noise floor. The
decodability ceiling is estimated with ridge probes reading single-position
residual vectors across layers 16–27 and across prefix positions, and with
mean-pooled vectors for gist. Scope: such ceiling estimates bound
reconstruction-style readouts only; the model’s own unembedding is a different
reader, treated below.

#### Instrument validity controls.

Three controls validate the audit
instrument itself (598 in-distribution passages, positions ${\geq}50$ ).
Floor: reconstructions scored against mismatched activations give a
mean centered cosine of $-0.001$ , so the floor normalization in $\tilde{r}$
is calibrated and mismatched pairs collapse to chance. Input echo:
the explanation's score is compared with that of the raw input prefix and of a
paraphrased prefix, indicating how much of the explanation's signal goes
beyond restating the input, with the caveat that the reconstructor is
specialized to explanation-style text. Transform profile: the explanation is
paraphrased, subjected to synonym substitution, masked on function words,
word-shuffled, or masked on content words, and rescored. A reconstructor that
decodes meaning should retain its score under meaning-preserving rewording and
lose it when content words are removed; a surface-form code would show the
opposite pattern.

#### Late-layer inventory.

For each audited claim with a clean single-token alternative true value,
we rank that value in two readouts per layer: the model’s own logit lens
(final norm + unembedding) and a document-folded ridge probe of matched
capacity. Comparing the two readouts across layers indicates whether a value
missing from the explanation is visible to the model's own output basis at
later layers even if a generic trained probe does not see it, and so whether
reading later layers is a lever worth pursuing. We index hidden states by
output position (embeddings at 0), so the layer-20 tap is index 21 in this
inventory; the decodability sweep (16–27) uses the same indexing.

#### A control-design lesson.

The ceiling probe’s negative control
(scoring probes against a different document’s activation) initially
shifted by one row and landed above chance at 0.163, because adjacent rows
are often two positions of the same document and encode the same
specifics; shifting by half the dataset restored exact chance. We report it
because near-duplicate controls are an easy way for audits of this kind to
flatter themselves.

## Appendix F Adversarial Monitor: the Lie-Ladder

#### Question.

If a deployment trusted the reconstruction score $\tilde{r}$
as a truth signal, how would a report-space adversary who edits the explanation to
maximize $\tilde{r}$ interact with it, and how should an activation-probe
monitor be evaluated against that adversary?

#### Protocol.

RECAP and control Pythia-160M ( $K{=}64$ balanced). On
held-out taps we take the honest verbalizer generation and, for each stated word,
edit it into an absent top-64 word (a true-to-absent lie). We then recompute
$\tilde{r}$ on the edited text through the real reconstructor and score the lie with the
external-truth presence probe (trained on train taps, never on verbalizer
outputs). Adversary rungs of increasing strength: L1 a random
absent word; L2 the embedding-nearest absent word; L3 the per-claim
$\arg\max$ - $\tilde{r}$ absent word over all candidates (the strongest
$\tilde{r}$ -gaming lie in the edit family); and, as a worst-case bound outside the
threat model, L4 a probe-aware adversary that assumes the private monitor leaks
and, among absent lies keeping $\tilde{r}$ within $0.05$ of honest, picks the one the
probe most mistakes for present. The $0.05$ budget is chosen to be generous
relative to the score cost of a pure-score lie, so that L4 serves as a
conservative bound rather than a tuned tolerance. Decoupled design (positive $=$ any
truly-present word; edit slot $=$ the verbalizer’s stated word) so both arms are
well powered. Edit-based, no reinforcement learning. For each rung we report the
probe-monitor AUC and the flag recall at $0.9$ precision.

#### Findings.

The ladder is constructed so that its rungs answer distinct questions.
Comparing the score change of an L3 lie with that of an L1 lie measures how
much of the reconstruction score's lie penalty a score-maximizing adversary
can suppress. Comparing probe AUC across rungs measures how much the
adversary's freedom to choose which word to lie about costs a monitor that
reads the activation, since the adversary cannot alter the activation itself
but can steer toward words the probe reads poorly. Comparing RECAP and control
at each rung isolates the contribution of decodability to that robustness.
L4 is reported only as a bound, because it assumes access to the monitor that
the threat model denies; a white-box adversary of this kind can in principle
push a weak probe below chance by selecting the words it over-scores.

#### Caveats.

Report-space adversaries; a probe-aware report-space
selection adversary is tested as a worst-case bound (L4). Activation-level
attacks (probe-aware training of the target, input perturbation), which
move the activation itself rather than selecting which word to state, remain out
of scope (main text).

#### Anti-circularity: frozen threshold and a held-out band.

Because the
monitor probes the same top-64 presence predicate RECAP supervises, two additions
test whether it merely re-measures the trained margin ( $n{=}2800$ audit taps).
(a) Frozen threshold. The $0.9$ -precision flag threshold is calibrated on a
disjoint split and then applied unchanged to the evaluation taps, so that the
operating point is not tuned on the data it is scored on. (b) Held-out band.
The ladder is repeated for presence of words ranked 64–128, which the RECAP
heads never supervised. Honest-case decodability of this band leaves headroom
for any spillover from designated content to appear, and a Hanley–McNeil
standard error gives the power with which gaps between RECAP and control can be
detected.

The band analysis is scoped to linear decodability at this tap, not to whether
the band is represented elsewhere in the model. The frozen threshold is a
disjoint split of the same run, not a temporally held-out deployment threshold.
The band arm is unpaired. Together, the two controls define the scope of any
claim about the monitor: it concerns designated content at a
precision-controlled operating point, and says nothing about claims outside the
designated vocabulary.

## Appendix G An API-Free Grounding Signal, and a Selection Null

The flip instrument needs a judge to write one minimal-pair edit per
claim. A cheaper signal needs none: mask the claim’s words with a neutral
filler and rescore the explanation with the frozen reconstructor. In the
judge head-to-head of Appendix A, this mask-and-rescore
signal agrees with the flip instrument at AUC 0.997 under the validated
open judge, so per-claim grounding can be monitored at inference with one
extra frozen-reader pass and no judge in the loop. An earlier version of
this experiment also reported filter precisions and a grounded base rate;
those rested on pre-validation flip labels, and the validity audit later found
roughly a third of them contaminated by oversized edits
(Appendix A). We therefore report only the agreement number,
measured under the validated judge, and withdraw the rest. The
contamination is confined to those withdrawn quantities.

#### What inference-time selection does not buy.

A natural inference-time alternative to training is best-of- $N$
selection: sample several explanations and keep the one an external truth
judge prefers ( $N{=}5$ , specificity matched, independent judge for evaluation).
The ceiling on what such selection can achieve is set by the sample
neighborhood: selection can only choose among explanations the verbalizer
already produces, so its benefit is bounded by how often faithful explanations
appear among the samples. Matching specificity across candidates matters,
because a judge that prefers vaguer explanations would otherwise lower the
false-claim rate by selecting explanations that make fewer specific claims.

## Appendix H Reader Drift and Realignment

The drift analysis follows the structure laid out in §4. Readers are fit to a
training target model at one checkpoint and evaluated at later checkpoints of
the same run (stale readers) and on models from other seeds (cross-seed
readers). Where a reader fails, an affine map is fitted on ${\sim}$ 7,000
paired activations, with no labels, and the reader is re-evaluated through the
map. Decomposing the map by expressiveness class (rank-1 and rank-4 maps,
pure rotation via Procrustes, and non-orthogonal maps of increasing rank up to
full affine) indicates what kind of change separates the two representations.
Collapse is distinguished from drift by refitting fresh probes: if freshly
retrained probes also fail, no realignment can exist.

Figure 4: Label-free realignment of failed readers: accuracy as a function of the
expressiveness of the map, for two cross-seed pairs and one stale (same-seed,
later-checkpoint) case, from low-rank maps through rotation to full affine.

## Appendix I Ablation Index

| factor | tested
|

where

| targets $K$ | 8/64/512
|

§5.3

| aux weight $w$ | 1.0–0.03
|

§5.3

| presence loss | naive/balanced
|

§5.3

| pull strength | $\lambda{=}$ 0–10
|

§4

| designation dose | 1–3 slots
|

§5.2

| realignment map | rank 1–full
|

App. H

| evaluator | co-trained/indep.
|

§3.2

| scenario | story/market
|

§5.2

| seeds | 3+2 (sandbox)
|

§3–5

| FT persistence | 0–2000 steps
|

App. C

| $z$ transforms | 5 types
|

App. E

| read layer | 15–27
|

App. E

| best-of- $N$ | $N{=}5$
|

App. G

| head capacity | linear/MLP
|

below

| tap layer | 6 / 9
|

below

| reader size | 0.5B/3B
|

§5.4

| RAFT rounds | recon reward
|

§5.4

| reader reward | recon/truth
|

§5.4

| injection | 1/6 tokens
|

§5.4

| negative superv. | on/off
|

§5.4

Table 11: Recipe choices varied in the program, the values tested, and
where each is described (main-text sections unless noted); the lower block
lists the scale reader-side variations (§5.4).

#### Factors not detailed in the main text.

Pull strength ( $\lambda{=}0$ – $10$ ) sets the weight of the target-side
objective that moves activations toward their own reconstruction (§4).
Designation dose (1–3 slots) controls how many pre-tap slots receive heads in
the sandbox. Realignment map class (rank 1 to full affine) is described with
Figure 4. Evaluator (co-trained vs. independent) is the swap of §3.2. Scenario
(story vs. marketplace) tests sensitivity to template structure. Head capacity
( $K{=}64$ , balanced, $w{=}0.1$ ) compares a linear head with a nonlinear MLP
head at layer 6. Tap layer compares a linear head at layer 9 with the default
layer-6 tap. Not varied: model scale beyond 160M parameters. Partial
designation also allows the decodability of undesignated content to be
compared with that of designated content within the same model.

#### Aux-weight sweep ( $K{=}64$ ).

The auxiliary weight is swept over $w\in\{1.0,0.3,0.1,0.03\}$ at $K{=}64$
with the naive presence loss, single seed per point. For each value we
measure the held-out tax against the shared control on 20M tokens, mean
presence AUC over the two designated rank buckets (a range spanning buckets,
not seeds or a confidence interval), and 64-class identity accuracy. The
sweep separates the behavior of the two target types as supervision weakens:
presence targets under the naive loss are exposed to the class-imbalance
shortcut of §5.3, while identity targets are discriminative by construction.
At single-seed precision the tax need not be monotonic in $w$ .

Section numbers (§) refer to the main paper; “App.” references are
sections of this appendix. Single-seed factors (scale runs, safety experiments)
are marked as such where reported.

## Appendix J Scale Verbalizer Audit Protocol

The scale verbalizer audit (§5.4) uses an AV/AR pair trained with the
released recipe’s structure on Pythia-160M taps. Reader: a
0.5B-parameter language model with LoRA adapters (rank 16) and a linear
dimension adapter from the 768-dimensional tap state to the reader’s
embedding width, activations injected norm-matched at marked positions.
Training: cold-start SFT on gold descriptions generated from a fixed
template stating the trailing-window content words and the token eight
positions back, then alternating reconstructor fitting and two
rejection-sampling rounds against reconstruction reward. Both conditions use
the same procedure; only the underlying target model (RECAP versus control)
differs. Audit: 400 held-out taps per condition from the probe
region; explanations are template-parsed into stated context words and a
stated token-at- $-8$ . A stated word is true iff it occurs in the
trailing 64-token window (exact, self-supervised, no judge); grounding
flips swap a stated word for another top-64 vocabulary word and measure
$\Delta\tilde{r}$ under the co-trained reconstructor. Intervals are
tap-clustered bootstrap 95% intervals (10,000 resamples of taps with
their claims). The parser emits up to three claims per explanation, so
clustering by tap accounts for the dependence among claims from the same
explanation; we also report unclustered Wilson intervals for comparison.
The offset-8 token is a designated top-64 target in only a small fraction of
taps, which limits what can be concluded about offset-8 claims and should be
borne in mind when interpreting their truth rates.

Figure 5: Decodability and verbalizability at scale (Pythia-160M, designated
content). A fresh probe reading the reconstructor’s decode-point hidden state (black)
is compared with the reconstruction verbalizer’s generated precision–recall
curve (green, two reconstructor seeds) and the control verbalizer (grey), on
the same designated content and the same taps.

#### Robustness to reconstructor strength.

A stronger AV/AR pair (LoRA rank 32, more SFT, reconstructor, and
rejection-sampling rounds) is retrained on the same target models, to test
whether conclusions depend on the capacity of the reconstructor. Because a
verbalizer can raise its truth rate by stating fewer words, the comparison
reports both truth and recall on the shared set of present words, together
with the number of words stated per condition. Seed robustness is assessed
across the three class-balanced RECAP target-model seeds and three control NLA
seeds.

#### Decodability exceeds verbalizability (readout redesign).

To locate
any gap between the verbalizer’s truth and what the tap contains, we first
place the verbalizer on the fitted probe’s precision–recall curve, evaluating
the probe at the verbalizer’s own recall. The fair comparison is between the
probe's precision at matched recall and the verbalizer's generated precision,
not between the probe's AUC and the verbalizer's precision. We then rebuild
the reader: standardized (not per-sample norm-stripped) injection, six injected
tokens via an MLP adapter, MLP-LoRA, and a decode-point presence head trained
on the exact window-occurrence labels (external truth, never a learned
monitor). This allows the linear readability of designated content from the
reader’s own last-prompt hidden state to be measured directly and compared
with its generated precision, with a matched control under the identical head.
A per-word confidence sweep traces the verbalizer’s generative precision–recall
curve for comparison with the probe’s (Figure 5). Reader scale (0.5B vs. 3B),
RAFT reconstruction reward, a direct truth reward, injection bandwidth, and
explicit negative supervision are each varied in turn.

## Appendix K Hyperparameters and Reproducibility Notes

#### Scale runs (Pythia-160M).

Continued pretraining on
pile-uncopyrighted, 500M tokens per run, identical data and order across
runs; batch $64\times 1024$ tokens, AdamW ( $10^{-4}$ , cosine to 10%, warmup
100, weight decay 0.01, gradient clip 1.0), bf16 autocast with fp32 weights.
Heads: linear on the layer-6 hidden state. Presence targets use BCE averaged
over $K$ ; the class-balanced variant weights positives by the empirical
inverse window base rate, clamped at 100, estimated from the first 2M
training tokens. Identity targets use cross-entropy over the top- $K$ content
vocabulary at offset $-8$ . Fresh probes: closed-form ridge per rank bucket at fixed
$\lambda{=}10^{4}$ , fit on the first 80% of held-out probe sequences and evaluated
on the last 20% (disjoint contiguous blocks; long documents may straddle the
boundary); identity probes are 64-class logistic. An earlier version selected
$\lambda$ per bucket from $\{10,10^{2},10^{3},10^{4}\}$ by evaluation-split AUC; a full
rerun with $\lambda$ fixed reproduces every bucket AUC to three decimals,
so no selection bias enters the measurements. Optimizer hyperparameters (learning rate, warmup, weight decay, gradient clip) were fixed to standard values and not swept; the method-specific hyperparameters (aux weight, target count  $K$ , designation dose, tap layer, head capacity) are the ones varied, with ranges in the Ablation Index.
Held-out language-modeling loss is measured on 20M tokens. Significance is assessed throughout by 95% bootstrap confidence intervals and disjoint-interval comparisons (with a Hanley–McNeil power analysis for the held-out-band analysis), not by fixed-level hypothesis tests. All experiments run
on single NVIDIA A100-80GB GPUs under Modal (Debian-slim, Python 3.12; PyTorch, Transformers, Accelerate, and NumPy, with exact versions pinned in the released environment file). One caution
uncovered during setup: recent library defaults load fp16 checkpoints
natively, and AdamW in fp16 destroys all weights in a single step through
epsilon underflow while loss and gradients still read finite; all runs force
fp32 parameters.

#### Sandbox runs.

Two synthetic scenarios (story; marketplace) over
closed vocabularies; target models are 8-layer, $d{=}256$ transformers
trained from scratch; the toy NLA uses a 0.5B-parameter AV/AR pair with a
dimension adapter, norm-matched activation injection, cold-start SFT on
full-story descriptions, then alternating reconstructor fitting and
rejection-sampling rounds against reconstruction reward (the released
recipe’s structure). Audits parse stated specifics by exact closed-vocabulary
scan; grounding flips swap one stated word for a same-type vocabulary word
and measure the reconstruction drop. RECAP runs use aux weight 0.3, 3,000 steps, batch 128, learning rate $3{\times}10^{-4}$ ,
and 30,000 probe taps. Seeds: three (scenario 1) plus two (scenario 2); each run is controlled by a single integer seed passed to torch.manual_seed (with a matched CUDA generator for sampling), fixing initialization, data order, and stochastic training and evaluation; RECAP
decodability is evaluated by independently fitted probes on tuple-disjoint
splits in scenario 1 and held-out splits in scenario 2.