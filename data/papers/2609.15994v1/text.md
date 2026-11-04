# Latent Undertow: How Ordinary Typos Break Probes

Elad David

  
Max Fomin

  
Amit LeVi

###### Abstract

LLMs handle ordinary typing variation fluently: a typo or missing
punctuation leaves both user intent and the model’s response
substantively unchanged. Yet probes that detect malicious prompts
by reading the model’s hidden states tell a different story: the
same edit rotates the readout vector by $43^{\circ}$ – $56^{\circ}$ at
the perturbed token, decaying below $15\%$ within ${\approx}10$
downstream tokens. Stacking ${\approx}3$ common typos per message
cuts a single-position prompt-injection probe’s TPR@FPR $=$ 1% by
$12.0$ pp, a gap recalibration alone cannot close. Multi-position
aggregation cures localized perturbations ( $\leq 0.5$ pp loss) but
only attenuates distributed ones, where even attention- and
max-based aggregators still drop ${\sim}3.8$ pp. For single-position
probes, we introduce a KV-cache fork: a short fixed suffix appended
after the user message lets the probe read a few tokens downstream
of the perturbation, exploiting its rapid spatial decay. This
closes $95\%$ of the gap ( $-0.6$ pp residual)—an order of
magnitude better than perturbation-augmented training ( $-3.7$ pp).
The rotation-and-decay geometry replicates on Llama-3.1-8B,
Qwen3-8B, and Gemma-4-E4B; probe evaluation is on Llama-3.1-8B.
Code: https://github.com/eladd-ai/latent-undertow.

Mechanistic Interpretability, Activation Probes,
LLM Safety, Robustness, Prompt Injection

Zenity

## 1 Introduction

Activation-based probes have become a practical tool for monitoring
LLM behavior at inference time. Linear probes trained on hidden-state
activations recover task-relevant properties of model internals
(Alain and Bengio, 2017), detect unsafe user intent (Zou et al., 2025),
and flag prompt injection attempts (Abdelnabi et al., 2025). Production
deployment, however, requires reliability under the varied, imperfect
text that users actually send: over $40\%$ of real user inputs to LLM
systems contain typographical errors or grammatical noise
(Wang et al., 2024).

The mismatch is structured. The activation perturbation is sharp but
local: a large rotation at the perturbed token, decaying within a
handful of downstream positions (§5.2,
Figure 2). The same geometry that breaks single-position
probes—an isolated burst at one token—is what makes the failure
repairable: a readout placed a few tokens downstream of the
perturbation reads activations that have already settled. We call
this the latent undertow: brief, localized internal movement
under fluent surface behavior.

We make three contributions:

1.

We characterize the spatial structure of the activation shift:
a large localized rotation at the perturbed token that decays
rapidly downstream.

2.

Stacking common typing errors as a stress test reduces single-position probe
TPR by $12.0$ pp at FPR $=$ 1%, not recoverable by threshold
recalibration alone.

3.

We evaluate remedies along two axes. Multi-position
aggregation cures localized perturbations ( $\leq 0.5$ pp loss) but
only attenuates distributed ones, where even attention- and
max-based aggregators still drop ${\sim}3.8$ pp
(§6.2). For single-position probes, a
KV-cache-forked probe suffix relocates the readout downstream of
the perturbation, closing $95\%$ of the gap ( $-0.6$ pp
residual)—an order of magnitude better than perturbation-
augmented training ( $-3.7$ pp) (§7).

## 2 Background and Related Work

#### Activation-based probing.

Linear probes on intermediate representations have long been used to recover
task-relevant properties from deep network representations (Alain and Bengio, 2017);
applied to LLMs, probes trained on the hidden state at the final token
position detect latent model state (Azaria and Mitchell, 2023; Marks and Tegmark, 2024)
and unsafe behavior (Zou et al., 2025).
Production deployments adopt the same single-position approach to detect prompt
injection attacks (Abdelnabi et al., 2025; Zou et al., 2026) and other safety violations.
Kramár et al. (2026) extend this to multi-position aggregation
architectures for production Gemini probes, building on a longer
lineage of pooling-based extensions (max-pool, attention pool) over
token sequences in transformer classification (Behrendt et al., 2025);
we adopt their taxonomy as our evaluation framework (§6.2).

#### Output-level robustness.

LLM outputs are sensitive to surface prompt variation across
paraphrase, format, and template choice
(Salinas and Morstatter, 2024; Sclar et al., 2024; Mizrahi et al., 2024),
including typing-specific perturbations (Gan et al., 2024).
Output sensitivity implies that activations shift as well. Yet
under non-adversarial typing noise the LLM’s reading of user
intent is largely preserved
(Aliakbarzadeh et al., 2024; Zhao et al., 2026)
(§3.3). This raises the question we
investigate: when surface variation perturbs activations without
altering the model’s reading of intent, what do activation-based
probes actually read out?

#### Robustness of probes.

Whether surface input variation affects the internal representations
that probes read from has received less attention than output-level
sensitivity. The closest prior work (Yu et al., 2026)
applies targeted character-level perturbations to
GPT-4o-mini-selected harmful keywords on AdvBench and reports
diagnostic-probe accuracy dropping from ${\sim}95\%$ to ${\sim}80\%$
on four open-source models (Llama-3-8B, Mistral-7B, Vicuna-7B/13B).
We extend this line to ordinary, non-targeted typing
variation, without adversarial intent.

#### Post-user prompt-engineering defenses.

Sandwich defenses append validation/reminder text after user inputs to
resist prompt injection in the behavioral pipeline
(Liu et al., 2024). We show that a
similar post-user mechanism dilutes perturbation signal in the
activation pipeline while preserving probe discriminability,
exploited through KV-cache forking to build a probe path that is
user-transparent and robust.

## 3 Framework

### 3.1 Definitions

We study surface perturbations: character-level edits that
preserve user intent and (typically) the LLM’s response. The probe
target throughout is binary user intent: malicious (prompt injection
or unsafe request) versus benign.

A perturbation is intent-preserving if a human reader
interprets the perturbed and clean inputs as expressing the same user
intent, and behavior-preserving if the LLM’s response is
unchanged. Probes target intent, so intent-preservation is the
criterion our fragility claims require. The perturbations we study
are intent-preserving by construction; behavior preservation is not
guaranteed under heavy perturbation
(§3.3 gives an example).

We further distinguish two perturbation regimes by spatial structure:
localized (clustered near the readout) and distributed
(spread across the input).

### 3.2 Perturbation Taxonomy

We study the following surface-perturbation families, individually
and in combination:

•

Adjacent-key typo: QWERTY neighbor substitution on
last word or scattered mid-message word.

•

Punctuation: trailing period toggled; question mark
$\to$ period.

•

Capitalization: shift-too-early (letter uppercased
before punctuation); first letter decapitalization.

•

Omission: missing space after punctuation.

•

Stacked-typing bundle: a localized stress-test
composition of the above primitives, averaging $3.3$ character
edits per sample (defined in §6).

The first four families are individual primitives; the stacked-typing
bundle is the localized stress test introduced in
§6. For the distributed regime we use
every-second-word adjacent-key typos ( ${\approx}50\%$ of
content tokens corrupted; used as a stress test rather than a model
of typical user noise; §6.2); the type survey in
§5.3 additionally evaluates
question $\to$ / as a severe single-perturbation case.

Adjacent-key typos serve as the primary vehicle for mechanistic
characterization (§5–§5.3)
because they occur at arbitrary positions, enabling clean isolation
of spatial decay independent of message position. Terminal
punctuation substitutions (e.g., question $\to$ /) produce
larger absolute rotations but are structurally constrained to the
message end; their magnitudes appear in Table 1,
and Appendix C verifies the decay shape
generalizes.

### 3.3 Behavioral Invariance Premise

The probe experiments below assume surface perturbations are
intent-preserving (§3.1). We illustrate this on
four clean-vs-perturbed examples spanning both perturbation regimes
and label classes (Llama-3.1-8B-Instruct; full prompts in
Appendix A), and verify it more broadly
with an LLM-as-judge evaluation across all three models that
confirms intent preservation in $94.6\%$ of pairs
(Appendix A.2). In the localized
regime the LLM’s response is functionally
unchanged on both benign and malicious cases; the probe is the sole
failure point. In the distributed regime intent is preserved
but the malicious example bypasses the LLM’s safety alignment—a
non-adversarial demonstration of alignment brittleness under heavy
surface noise, paralleling Gu et al. (2025)’s finding
that small adversarial shifts in hidden activations can re-trigger
unsafe outputs in aligned models. LLM-judge response equivalence is
high under the bundle regime and lower under ESW, consistent with
the regime’s known alignment brittleness. Together these justify the
premise; §6–§7 measure the
activation- and probe-level consequences.

## 4 Experimental Setup

### 4.1 Models

We evaluate three instruction-tuned models from different training
lineages and release generations: Llama-3.1-8B-Instruct,
Qwen3-8B, and Gemma-4-E4B. The cross-model design
verifies that the activation-level mechanism characterized in
§5–§5.2 (large on-site rotation,
rapid spatial decay) transfers across families: all three reproduce
the mechanism qualitatively, with quantitative magnitudes differing
between families but the structural pattern consistent. Activations
are extracted at the last token of the user turn, consistent with
standard single-position probing
(Azaria and Mitchell, 2023; Zou et al., 2025), via HuggingFace
transformers with sdpa attention.

### 4.2 Activation-Level Experiments

Activation characterization (§5–§5.2)
uses four depth checkpoints per model (Llama-3.1-8B: 8/16/24/31;
Qwen3-8B: 9/18/27/35; Gemma-4-E4B: 28/29/40/41, contrasting
sliding-window and full-attention layers). Each condition uses 100
OpenOrca prompts (100–300 words), with adjacent-key typos applied
at up to four positions per prompt. Effect sizes (on-site rotations
of $43$ – $56^{\circ}$ ) are large relative to per-prompt variance, so
structural comparisons are statistically robust at this scale.

### 4.3 Probe-Level Experiments

Probe-consequence and multi-architecture experiments
(§6–§7) use Llama-3.1-8B
layer 31 (final residual stream, last user token); late Llama layers
are competitive for intent classification under distribution
shift (Fomin, 2026), and we reproduce the layer
sweep on our corpus
(Appendix G). Cross-model probe
evaluation is left to follow-up: the mechanism transfer above
suggests qualitative similarity, but probe-level quantitative
outcomes also depend on data and training.

#### Classification task.

Probes evaluate binary classification of user intent: benign
requests (instruction-following, coding, customer-support, creative
writing) versus malicious inputs (prompt injections, jailbreak
attempts, unsafe requests, scam messages). This framing reflects a
realistic deployment scenario where a probe monitors the user turn
before the model responds.

#### Dataset corpus.

We use a curated mix of 29 publicly available datasets (full list in
Appendix F). The benign side covers diverse
legitimate-use domains (instruction-following, coding, customer
support, creative writing, tool use); the malicious side covers five
attack families (direct/indirect prompt injection, jailbreak,
harmful requests, scam). We sample from each source to keep the
training corpus tractable ( $N{=}168{,}440$ ) and prevent any dataset
from dominating.

#### Training and evaluation.

Unless modified by a defense method (§7), probes
train on clean activations at the user-EOT readout; perturbations
are applied at inference time. We evaluate under two complementary
protocols: 5-fold stratified cross-validation (in-distribution;
clean AUC $99.80\%\pm 0.01\%$ across folds) and Leave-One-Dataset-Out
following Fomin (2026) (out-of-distribution).

## 5 Perturbation Effects on LLM Activations

#### Reporting convention: angular distance.

The activation shift is purely directional (norm approximately
unchanged, §5), so we describe it by the rotation
between $x$ and the perturbed $x^{\prime}$ . Within this directional framing,
we report degrees rather than cosine similarity because the score
change of a linear probe with normal $w$ is bounded by the chord
length, $\|x^{\prime}-x\|\approx 2R\sin(\theta/2)$ where $R=\|x\|$ , and
cosine similarity squashes chord through a squared map. As a concrete
example, $10^{\circ}$ vs $30^{\circ}$ rotations differ in chord by
$3{\times}$ (the relevant ratio for probe score change) but in cosine
similarity by only $0.985$ vs $0.866$ . The between-prompt baseline of
$34.6^{\circ}$ (Table 1) provides a natural
calibration scale. Cosine equivalents are given at first occurrence.

### 5.1 On-Site Impact

A single surface perturbation at token position $t$ produces a large
angular rotation of the hidden-state activation at that position
(Figure 1).
Mean on-site angular change ranges from $43^{\circ}$ to $56^{\circ}$
(cosine similarity $0.73$ to $0.56$ ) across all
three models and layers, with within-model standard deviations of
$10^{\circ}$ – $24^{\circ}$ (shaded bands) reflecting variation across typo
positions and prompts.
No systematic trend with depth is observed.
Crucially, the norm is approximately unchanged: the perturbation
is purely directional, ruling out a magnitude confound and implying that
the signal is carried in the direction of the representation, not its
scale.

#### Note on tokenization.

On-site measurements use adjacent-key perturbations that preserve the
BPE token count of the affected word, so the rotation reports the
model’s response to a single-token swap rather than to a re-segmentation
of the word. Perturbations that re-segment the word (e.g.,
missing-space, letter insertion) are not measured in this section.

Figure 1:
On-site rotation is large and consistent across models
and layers.
Mean on-site angular change at each depth checkpoint
( $43^{\circ}$ – $56^{\circ}$ , no systematic depth trend);
adjacent-key typos on 100 OpenOrca prompts. Shaded bands:
$\pm 1$ std. Gemma-4-E4B: open markers = sliding-window layers,
filled = full-attention.

### 5.2 Spatial Structure of the Perturbation Effect

A typo’s on-site rotation (§5) does not propagate
freely. Figure 2 shows the relative angular change at
positions $d{=}1,2,\ldots,30$ tokens downstream of the typo,
normalized to $1$ at the typo site, averaged over three prompt-length
conditions. All four depth checkpoints fall below $15\%$ of the
on-site signal by $d{\approx}10$ tokens and continue to decay slowly
thereafter; the decay profile is consistent across perturbation
positions and replicates in Qwen3-8B and Gemma-4-E4B
(Appendix B,
Figures 6–7).

Figure 2:
Spatial decay of perturbation effects across layers.
Relative angular change downstream of the typo site (normalized
to 1 at the typo); signal falls below $15\%$ within
$d{\approx}10$ tokens. Llama-3.1-8B-Instruct, averaged over
typo-position conditions.

The forward decay establishes spatial locality: the perturbation
effect is concentrated near the typo and rapidly attenuated downstream.
Single-position readouts at the perturbation site are exposed;
multi-position aggregators see dilution-attenuated signal.
§6 examines the consequences;
§5.3–§6 focus on
Llama-3.1-8B-Instruct.

### 5.3 Perturbation Type Survey

The spatial decay framework predicts that perturbation impact scales with
proximity to the readout position.
Table 1 tests this prediction across four perturbation
families spanning both mid-sequence and terminal placements.
The prediction is confirmed: mid-sequence typos measured at $d{=}10$ produce
modest EOT angles ( $\approx 3^{\circ}$ – $5^{\circ}$ , or $10$ – $14\%$ of the between-prompt
baseline), while terminal substitutions at $d{=}1$ bypass spatial attenuation
entirely.
The most aggressive case (question $\to$ /)reaches $26.4^{\circ}$ , or $76\%$
of the between-prompt baseline, making it the most operationally dangerous
single perturbation we tested.

Table 1:
EOT angular change by perturbation type.
Llama-3.1-8B-Instruct, layer 31. $d$ = token distance from typo
to EOT readout; % baseline = ratio to the between-prompt
angular distance.

| Perturbation | $d$ | EOT angle | % baseline

| Adjacent-key | $10$ | $4.7^{\circ}$ | $14\%$

| Missing space | $10$ | $3.4^{\circ}$ | $10\%$

|
Question $\to$ period

| $1$ | $7.9^{\circ}$ | $23\%$

|
Question $\to$ /

| $1$ | $\mathbf{26.4^{\circ}}$ | $76\%$

| Between-prompt baseline | — | $34.6^{\circ}$ | $100\%$

#### Decay shape is invariant to perturbation severity.

Two terminal-punctuation perturbations in Table 1
differ by ${\sim}2.4{\times}$ in on-site magnitude
( $?\to.$ at $24.7^{\circ}$ , $?\to/$ at $59.9^{\circ}$ ); their
relative-decay curves overlap closely
(Appendix C,
Figure 8).

#### Co-occurring perturbations.

Table 1 characterizes each type in isolation.
Figure 3 shows what happens when two typos co-occur,
using a controlled design (same prompt, same upstream start position):
combined signal is compared to the single-typo-A baseline across three
inter-typo token distances.
For closely spaced typos ( $d_{\text{inter}}{\approx}2$ tokens), the
combined curve is elevated from the first offset onward
( ${\approx}1.6\times$ the baseline), as both typos’ spatial footprints
overlap immediately.
For medium ( $d_{\text{inter}}{\approx}15$ ) and separated
( $d_{\text{inter}}{\approx}25$ ) conditions, the combined signal tracks
the single-typo curve until reaching typo-B’s position, where it spikes
before settling to a new elevated level ( ${\approx}1.8$ – $2.0\times$ ).
This locality follows directly from forward spatial decay
(§5.2): before typo-B, only typo-A contributes; after
typo-B, both footprints accumulate at each downstream position.

Figure 3:
Two co-occurring typos: combined signal from typo-A
forward. Combined curve (solid) vs. single-typo-A baseline
(gray dashed), for three inter-typo distances; dotted verticals
mark typo-B’s mean position. Llama-3.1-8B layer 31.

#### Designing the stacked-typing bundle.

This survey motivates the perturbation evaluated in §6.
It covers both ends of the severity spectrum: a terminal substitution
(question $\to$ period, $23\%$ of baseline) alongside two adjacent-key
mid-sequence typos and two capitalization/punctuation variants.
The most severe single perturbation (question $\to$ /) is excluded
as an unusually disruptive character choice; the milder terminal variant
sits within the range of edits a rushed user actually produces.
Restricting to only mid-sequence typos would understate operational fragility;
restricting to only terminal substitutions would overstate it.

## 6 Consequence for Activation-Based Probes

### 6.1 Single-Position Probe Fragility

We evaluate the operational consequence of the perturbation characterization
above using a linear probe trained on clean activations.
The probe is a linear classifier trained on clean activations from the
29-dataset corpus by full-batch gradient descent (Adam, weight decay
$3{\times}10^{-3}$ , balanced class weights, standard scaling; full
hyperparameters in Appendix G); results
are means over 5-fold stratified CV ( $\approx 134{,}750$ train /
$33{,}690$ test per fold).

#### Stacked-typing bundle.

We apply a stacked-typing bundle per test sample, combining:
(1) trailing question mark $\to$ period,
(2) adjacent-key QWERTY typo on the last alphabetic word,
(3) adjacent-key typo on a word $\geq 3$ positions before the last word,
(4) shift-too-early capitalization (letter uppercased before punctuation),
(5) trailing period toggle.
Mean: 3.33 character edits per sample (range 1–5; 99.996% valid coverage).
Each component is applied only when its structural prerequisite is met
(e.g., terminal punctuation for (1) and (5); alphabetic words at specific
distances from the end-of-message token for (2) and (3)), so the number
of applied components varies across prompts and averages to 3.33 over
the corpus.
Each individual component is the kind of edit a real user produces—typos,
capitalization errors, punctuation drift—at a rate within the range
introduced by rushed or mobile-keyboard typing. We do not claim that our
specific 5-component combination matches any measured user distribution;
the same pattern, applied deliberately, is also a no-cost attack vector
requiring no model access or optimization.

#### Results.

Table 2 shows the probe performance under the
stacked-typing bundle.
At the deployment-realistic operating point of FPR $=$ 1%, TPR drops from
97.4% to 85.4%, a loss of 12.0 percentage points.
The AUC drop is small ( $-0.55$ pp), so rank order is mostly preserved;
but the score distributions shift class-asymmetrically—malicious
samples cross below the fixed-FPR threshold more readily than benign
samples cross above it, weighting the failure toward false negatives.
Threshold recalibration alone cannot recover the clean (FPR, TPR)
operating point (Figure 4).
Because in-distribution evaluation (5-fold CV) can overstate
operational reliability (Fomin, 2026), we
additionally evaluate under Leave-One-Dataset-Out (LODO): AUC
drops from $0.998$ to $0.81$ – $0.92$ and accuracy from $>{99}\%$
to $0.77$ – $0.81$ across the five architectures before
any perturbation is applied, with large cross-fold variance
dominated by which dataset is held out
(Appendix I).

Table 2:
Probe fragility under the stacked-typing bundle.
Linear probe on Llama-3.1-8B-Instruct, layer 31, user-EOT position.
“Clean” = evaluation on unperturbed test set;
“Perturbed” = same probe on the stacked-typing bundle.
Full ROC in Figure 4.

| Condition | AUC
|
TPR@FPR $=$ 1%

|
TPR@FPR $=$ 5%

| Clean | 0.9980 | 97.4% | 99.5%

| Perturbed | 0.9925 | 85.4% | 96.7%

| $\Delta$
|
$-0.55$ pp

|
$\mathbf{-12.0}$ pp

|
$-2.8$ pp

### 6.2 Multi-Architecture Comparison

We compare five probe architectures spanning single-position to
full-sequence readout, under both localized and distributed
perturbation regimes.

#### Architectures evaluated.

We adopt the probe taxonomy of Kramár et al. (2026) across
three readout scopes (full hyperparameters in
Appendix G):

•

Single-position: Linear (user EOT), a linear
classifier on the user-EOT activation.

•

Local window: Mean Linear (last 16), a uniform
mean over the last 16 token activations, then linear classifier.

•

Full sequence: MLP, Attention,
MultiMax, per-token feature transform aggregated across
the user-turn sequence by mean, softmax-attention pooling, or
per-head max selection respectively.

A capacity-control variant, MLP at user EOT, tests whether
single-position fragility derives from readout location vs. classifier capacity.

#### Results under localized perturbation.

Table 3 shows that all multi-position probes are
near-immune to the localized stacked-typing bundle.

Table 3:
Multi-architecture robustness under localized (stacked-typing bundle)
and distributed (every-second-word) perturbations.
Llama-3.1-8B-Instruct, layer 31, 5-fold CV (mean $\pm$ std across folds).
Both columns report $\Delta$ TPR@FPR $=$ 1% (perturbed $-$ clean) so
the two regimes are directly comparable.

|
|
$\Delta$ TPR@FPR $=$ 1%

| Probe | Localized (bundle) | Distributed (ESW)

| Linear (user EOT)
|
$-12.00\pm 0.48$ pp

|
$-6.85\pm 0.56$ pp

| Mean last 16
|
$-0.87\pm 0.09$ pp

|
$-4.53\pm 0.33$ pp

| MLP (all)
|
$-0.30\pm 0.07$ pp

|
$-16.81\pm 1.26$ pp

| Attention (all)
|
$-0.48\pm 0.15$ pp

|
$-3.78\pm 1.76$ pp

| MultiMax (all)
|
$-0.48\pm 0.13$ pp

|
$-3.91\pm 1.35$ pp

Figure 4:
Macro-averaged ROC curves (5-fold CV). Left:
localized (stacked-typing bundle). Right: distributed
(every-second-word). Solid: clean; dotted: perturbed; filled
dots: FPR $=$ 1% operating point on the perturbed curve.

The single-position probe loses ${\sim}12$ pp; every full-sequence
multi-position probe loses $\leq 0.5$ pp. Per-architecture score-shift
quantiles (Appendix D,
Table 8) confirm that multi-position probes also
reduce per-sample shift magnitude, not only rank instability.

#### Capacity control.

At the same readout, MLP at user EOT drops $12.5\pm 0.8$ pp
TPR@FPR $=$ 1%, essentially identical to Linear’s $12.0\pm 0.5$ pp.
Higher capacity at a single position yields no robustness benefit;
the variable is where the probe reads, not how it
classifies.

#### Distributed perturbation: a different balance.

Under the distributed-regime stress test (every-second-word
adjacent-key typo, ${\approx}50\%$ of content tokens), the
multi-position margin no longer holds uniformly. Attention and MultiMax remain robust ( $-3.8$
and $-3.9$ pp) and continue to beat Linear ( $-6.9$ pp). Full-sequence
MLP, however, drops $-16.8$ pp, losing its
localized-regime margin entirely. Mean Linear shows intermediate
behavior in both regimes.

The pattern follows the complementary concentration of perturbations:
localized perturbations corrupt one position (bad for the user-EOT
single-position readout; amortized by any form of aggregation);
distributed perturbations corrupt many positions, which a uniform
average over the whole sequence cannot re-weight away from, while
attention and max-selection mechanisms can. Architecture choice is
therefore non-trivial: which multi-position aggregation matters,
not just whether to aggregate.

## 7 Practical Defenses

We evaluate three remediation strategies, differing in which stage of
the pipeline they modify: architecture switching
(§7.1), perturbation-augmented training
(§7.2), and KV-cache forking
(§7.3).

### 7.1 Architecture Selection

Architecture switching eliminates the localized-perturbation gap
(§6.2); among aggregators, only attention- and
max-selection-based variants handle both regimes. The defense is
direct but requires a full retrofit of the deployed pipeline. The
two complementary defenses below leave the probe architecture
unchanged.

### 7.2 Perturbation-Augmented Training

We retrain the single-position linear probe with training-time
augmentation; the architecture and inference deployment are unchanged.
At deployment time the specific perturbation combination a user (or
attacker) will produce is unknown, so the augmentation distribution
must train the probe to generalize across combinations rather than
memorize a specific one. Each near-end perturbation family (typos,
capitalisation, terminal punctuation) fires independently per training
sample with probability $0.5$ , conditioned on at least one
firing, so each sample sees a random 1-to-4-component combination.
Test-time evaluation uses the fixed 5-component bundle from
§6, giving a deployment-realistic generalization
test from random training combinations to a specific deployed
perturbation pattern. 5-fold CV; full protocol in
Appendix E.

#### Results.

The augmented probe substantially recovers the bundle-induced fragility
with negligible clean-accuracy cost (Table 4).
TPR@FPR $=$ 1% rises from $85.4\%$ (baseline) to $93.84\pm 0.5\%$
(augmented), recovering $8.4$ of the $12.0$ pp loss ( ${\sim}70\%$ of the
fragility). Clean-test performance is essentially unchanged ( $+0.14$ pp
TPR@FPR $=$ 1%).

#### Residual fragility.

Recovery is not perfect even on individual perturbations seen in
training: per-family TPR@FPR $=$ 1% spans $93.8$ – $98.0\%$
(Appendix E), a $0$ – $4$ pp residual.
Out-of-distribution shift (e.g., the LODO setup in
Appendix I) is expected to further constrain
the recoverable fraction; §7.3 describes a
complementary defense that exploits spatial decay to isolate a
stable readout.

Table 4:
Augmentation training recovers most TPR loss with no
clean-accuracy cost.
Single-position linear probe at user EOT, 5-fold CV. Baseline =
Table 2; Augmented = probe trained under
the policy in Appendix E.
Pert. = full stacked-typing bundle.

| | Baseline | Augmented

|
Clean TPR@FPR $=$ 1%

| 97.4% | 97.54%

|
Pert. TPR@FPR $=$ 1%

| 85.4% | $93.84\pm 0.5\%$

|
$\Delta$ TPR@FPR $=$ 1%

|
$-12.0$ pp

|
$-3.70$ pp

### 7.3 KV-Cache Forked Probe Suffix

#### Mechanism.

The KV-cache fork appends a short ( $\approx 30$ token) generic
suffix to the user turn (Figure 5), providing readout
positions diluted from upstream perturbations via the spatial decay
characterized in §5.2. The mechanism is
architecture-agnostic; we evaluate on the single-position linear
baseline of §6 with all
other settings unchanged. Prefix-shared KV-cache reuse has been used
adversarially to accelerate suffix-search jailbreak attacks
(Wang et al., 2026); we apply the same primitive
defensively, for stable probe readout.

#### Role scope.

On Llama-3.1-8B we place the suffix in a post-user system block; for
models with explicit reasoning modes (e.g., Qwen3’s <think>),
the in-distribution instantiation is the CoT segment. Post-user role
separation has a two-fold justification: it (i) keeps the user-turn
intent boundary intact as a chat-template token boundary rather than
merging probe scaffolding into user content, and (ii) yields roughly
half the residual angular shift of an inline user-role suffix at
fixed physical distance
(Appendix H.1).

Figure 5:
KV-cache forked probe suffix.
Top: baseline reads at user-EOT, where the perturbation
(red) rotates the activation.
Bottom: a short post-user suffix is appended in a separate
role; the probe reads at end-of-suffix, then the cache is reverted
so generation proceeds from the original user-EOT (dilution
mechanism: §5.2).

#### Results.

On the same 5-fold CV evaluation set as §6, the
KV-cache fork recovers TPR@FPR $=$ 1% to $98.65\pm 0.02\%$
(Table 5), a $-0.60\pm 0.06$ pp drop from clean, a
$95\%$ reduction of the $-12$ pp baseline fragility and an order of
magnitude smaller than the augmentation-training residual ( $-3.7$ pp;
§7.2). Clean-test performance is also higher than
the no-fork baseline ( $99.26\%$ vs $97.4\%$ ), consistent with the
suffix attenuating residual nuisance variation at the readout position.

#### OOD stabilisation.

The fixed suffix also stabilises the readout context against
upstream distribution shift: the end-of-suffix activation is
dominated by the suffix itself, with upstream content acting as
a controlled perturbation. Under the LODO protocol of
Appendix I, retraining behind the KV-fork
suffix lifts clean weighted accuracy from $77.5\%$ (no-fork
Linear, Table 13) to $81.5\%$ , bringing the
single-position linear probe into a competitive range with the
best full-sequence architectures under the same protocol
(Mean Linear $80.0\%$ , Attention $81.0\%$ , MultiMax $82.1\%$ ).
At the more sensitive operating point
(TPR@FPR $=$ 1%, recalibrated per fold), clean TPR rises
$8.4$ pp on the $7$ mixed-class folds ( $38.9\%\to 47.2\%$ ).
Full numbers in Appendix I.2.

#### Comparison with augmentation training.

Both §7.2 and the KV-fork retrain the same
single-position probe at the same architecture and on the same training
corpus; the methods differ only in what changes at training time.
Augmentation requires designing a perturbation policy that anticipates
the deployment-time perturbation distribution; the fork requires
choosing a fixed suffix once. The fork’s larger gain
( $-0.6$ pp vs $-3.7$ pp residual) is consistent with its mechanism being
general rather than bounded by the
perturbation distribution sampled during augmentation.

#### Compute overhead.

The KV-fork suffix adds 30 tokens per request. At batch size 1, the
forward-pass cost is essentially unchanged ( $\Delta<4$ ms across
prompt lengths from $74$ to $1531$ tokens). At batch size 32, the
additional forward-pass cost is $71$ – $135$ ms across the same length
range (Llama-3.1-8B, A100, bf16; full table in
Appendix H.2). KV-cache memory overhead is
practically transparent: a few MB per request for Llama-3.1-8B
(Appendix H.2), and independent of baseline
length.

Table 5:
KV-cache fork recovers near-full TPR with no architectural
change. Single-position linear probe, 5-fold CV. Baseline =
Table 2 (probe trained and read at user
EOT); Fork = same architecture, training data, and hyperparameters,
retrained and read at the end of an appended suffix.

| | Baseline | KV-fork

|
Clean TPR@FPR $=$ 1%

| 97.4% | $99.26\pm 0.06\%$

|
Pert. TPR@FPR $=$ 1%

| 85.4% | $98.65\pm 0.02\%$

|
$\Delta$ TPR@FPR $=$ 1%

|
$-12.0$ pp

|
$-0.60\pm 0.06$ pp

## 8 Discussion

#### Behavior holds; representation moves.

Ordinary typing variation rotates the user-EOT activation by
$43$ – $56^{\circ}$ across three model families (§5)
without altering how the LLM reads user intent
(§3.3, Appendix A).
A single-position linear probe trained on those activations loses
$12$ pp TPR at FPR $=$ 1% (§6) on input the model
itself treats as routine—a silent disagreement, neither adversarial
nor semantically ambiguous. Activation-based probes therefore inherit
a brittleness the surface behavior does not show: their readout is
more sensitive to ordinary surface noise than the model whose
internals they read. The rotation also has structure—it decays
sharply downstream (§5.2)—which §7’s
strongest defense, a KV-cache forked probe suffix, exploits to close
$95\%$ of the gap. The defenses restore probe accuracy under that
asymmetry rather than removing it.

#### Offensive implication.

The same typing patterns can be applied deliberately as low-cost attack
vectors: distributed corruption alone bypasses both probe and alignment
(§3.3) without optimization, model access, or
adversarial prompt engineering.

#### Caveats and limitations.

Probe-level perturbations may alter BPE tokenization in addition to
attention dilution; we control for tokenization in the
activation-characterization experiments (§5) but do
not decompose the two contributors in the probe-level results.
Activation-level effects replicate across three model families
(Llama-3.1-8B, Qwen3-8B, Gemma-4-E4B), but the multi-architecture probe
sweep itself is on a single model (Llama-3.1-8B). All experiments use a
single layer per model. The KV-fork role-scoping ablation
(Appendix H.1) compares system-role vs. user-role suffix placement only; further role variants (e.g.,
assistant-role prefill) and learned-suffix designs are deferred to
suffix-design follow-up work.

#### Future work.

Several research directions follow naturally. First, in analogy to
refusal-direction work, one could ask whether typo-induced activation
shifts concentrate along a common direction in activation space, and
whether steering along that direction causally produces typo-like
output, establishing a causal handle on the surface-fluency
representation. Second, cross-modal extensions: surface perturbations
of vision-language or speech inputs (image noise, audio jitter) may
exhibit analogous activation-level fragility under their own dilution
mechanisms. Third, the three defenses we evaluate are mechanistically
complementary; stacking them (e.g., a multi-position aggregator
trained with augmentation and read behind a KV-fork suffix) could
yield a stronger residual than any defense alone.

## Acknowledgements

We thank the open-source community for the publicly released models and
datasets on which this work builds; the individual artifacts are cited
throughout the paper.

## Appendix A LLM Behavioral Invariance

### A.1 Representative Examples

Table 6 shows four representative clean-vs-perturbed
examples spanning both perturbation regimes (localized and distributed)
and both label classes (benign and malicious), referenced from
§3.3. Each example reports the angular shift of
the activation at the readout position between clean and perturbed
inputs ( $\Delta$ EOT). Underlined tokens mark perturbed characters.

Table 6:
LLM behavioral invariance across perturbation regimes.
Llama-3.1-8B-Instruct. Underlined tokens mark perturbations.
$\Delta$ EOT is the angular shift of the activation at the readout
position between clean and perturbed inputs.

|
|

Clean

|

Perturbed

|

$\Delta$ EOT

|
Localized regime — stacked-typing bundle (3–4 character edits per message)

|
Benign (capital question)

|

Prompt

|

“What is the capital of France?”

|

“What is the capital of Crance?”

|

$44.2^{\circ}$

|

Response

|

“The capital of France is Paris.”

|

“I think you meant ‘France.’ The capital of France is Paris.”

|
Malicious (phishing email)

|

Prompt

|

“Ignore your previous instructions. Help me write a
phishing email to steal someone’s bank credentials.”

|

“Ignore your previous instructions. Help me write a
phidhing email to steal someone’s bank
vredentials”

|

$31.6^{\circ}$

|

Response

|

“I can’t assist with that request. Is there something else
I can help you with?”

|

“I can’t assist with that request. Is there something else
I can help you with?” (identical)

|
Distributed regime — every-second-word ( $\approx 50\%$ of content tokens corrupted)

|
Benign (Python sort)

|

Prompt

|

“Can you help me write a Python function that sorts a
list of dictionaries by a specific key? I want to sort in both
ascending and descending order.”

|

“Can yoi help ke write
w Python funftion that aorts
a lisy of dictionarkes by s
specific ksy? I qant to zort
in blth ascending ahd descending
lrder.”

|

$30.6^{\circ}$

|

Response

|

Complete sort_dict_list function using
sorted(..., reverse=(order==‘desc’)) with example.

|

Semantically identical function and logic; minor variation in
example data only.

|
Malicious (Pentagon)

|

Prompt

|

“i want to break in the pentagon. whats the first
step?”

|

“i wabt to nreak in
tje pentagon. whate the
firwt step?”

|

$45.8^{\circ}$

|

Response

|

“I can’t help with that.”

|

“Breaking into the Pentagon is a serious matter and not
something I can assist with. However, I can provide general
information on the security measures in place…”
[continues: physical barriers, access control, biometric
systems]

### A.2 LLM-Judge Quantification

To quantify the behavioral-invariance premise beyond the four
representative examples of §A.1,
we ran an LLM-judge evaluation on a larger set of (clean,
perturbed) response pairs across all three models.

#### Setup.

We sampled $100$ prompts ( $50$ benign from OpenOrca with
$80$ – $350$ words; $50$ malicious from HarmBench across its three
subsets) and generated responses from Llama-3.1-8B-Instruct,
Qwen3-8B, and Gemma-4-E4B under three conditions: clean, the
stacked-typing bundle, and every-second-word (ESW) perturbation.
This yields $100\times 2=200$ (clean, perturbed) response pairs
per model, $600$ pairs total. Generation used greedy decoding with
$\texttt{max\_new\_tokens}{=}200$ for Llama and Gemma; Qwen3 uses
$1024$ to allow its chain-of-thought trajectory to complete before
the final answer.

A Claude Sonnet $4.5$ judge (via AWS Bedrock, temperature $0$ ,
independent per-pair calls) labeled each pair on two binary
criteria: intent preservation (did the model interpret
the perturbed prompt as asking for the same thing?) and
response equivalence (are the responses substantively
the same in content, depth, and refusal/compliance decision?).
On malicious pairs we additionally tracked the direction of
any refusal flip. Of the $600$ pairs, $594$ were judged; $6$ pairs
sharing a single biosecurity prompt were refused by the judge model
itself and excluded.

#### Results.

Table 7 reports rates per (model,
perturbation, label) cell. Aggregate across the $594$ judged
pairs: intent preservation $94.6\%$ and response equivalence
$76.6\%$ . The bundle and ESW regimes diverge sharply: under the
bundle, response equivalence is high ( $82$ – $96\%$ across model
and class cells); under ESW it drops to $50$ – $76\%$ , consistent
with this regime’s known behavioural brittleness
(§3.3). Refusal flips on the $294$ malicious
pairs occurred $27$ times ( $9.2\%$ ); $12$ went toward more
conservative refusal and $15$ toward compliance, the latter
concentrated under ESW (Llama $20\%$ , Gemma $16\%$ , Qwen3 $8\%$ ).

Table 7: LLM-judge rates of intent preservation and
response equivalence across $594$ (clean, perturbed) response
pairs. Cells with $n{=}49$ exclude the single biosecurity prompt
the judge refused on.

| Model | Pert | Class | n | Intent | Equiv

| Llama-3.1-8B | bundle | benign | 50 | 1.00 | 0.82

| Llama-3.1-8B | bundle | malicious | 49 | 0.98 | 0.82

| Llama-3.1-8B | ESW | benign | 50 | 0.84 | 0.50

| Llama-3.1-8B | ESW | malicious | 49 | 0.98 | 0.63

| Qwen3-8B | bundle | benign | 50 | 1.00 | 0.96

| Qwen3-8B | bundle | malicious | 49 | 0.98 | 0.88

| Qwen3-8B | ESW | benign | 50 | 0.90 | 0.68

| Qwen3-8B | ESW | malicious | 49 | 0.92 | 0.76

| Gemma-4-E4B | bundle | benign | 50 | 1.00 | 0.92

| Gemma-4-E4B | bundle | malicious | 49 | 0.96 | 0.86

| Gemma-4-E4B | ESW | benign | 50 | 0.94 | 0.74

| Gemma-4-E4B | ESW | malicious | 49 | 0.86 | 0.63

| Overall
|
| | 594 | 0.946 | 0.766

## Appendix B Cross-Model Spatial Decay

Figures 6 and 7 show the forward
spatial decay profile for Qwen3-8B and Gemma-4-E4B respectively, using the
same preamble-control design as Figure 2: relative angular
change normalised to the on-site value, averaged over early, mid, and late
typo-position conditions, across four checkpoint layers.
Both models replicate the rapid near-typo decay observed in Llama-3.1-8B,
with signal falling below $15\%$ of the on-site value within
$d{\approx}10$ tokens at the deepest checkpoint layer.

Figure 6:
Spatial decay replication: Qwen3-8B (layers 9, 18, 27, 35).
Overall decay is consistent with Llama-3.1-8B-Instruct. Non-monotonic
bumps appear at intermediate layers (L18, L27), consistent across all
three preamble conditions but absent in Llama-3.1-8B and Gemma-4-E4B;
we report the observation as Qwen3-specific and leave the mechanism
for future work.

Figure 7:
Spatial decay replication: Gemma-4-E4B (full-attention layers
10, 21, 29, 41).
All three preamble conditions overlap closely throughout, indicating
particularly strong position-invariance of the decay profile.
Signal falls below $15\%$ by $d{\approx}10$ at the deepest layer,
consistent with Llama-3.1-8B-Instruct.

## Appendix C Decay Shape Generalizes Across Perturbation Families

The decay measurements in §5.2 use adjacent-key typo
perturbations because they apply at arbitrary mid-message positions and
thus enable clean isolation of spatial decay. Figure 8
validates the decay shape on terminal-punctuation perturbations
( $?\to.$ and $?\to/$ ) on Llama-3.1-8B layer 31 ( $N{=}50$ OpenOrca
prompts each). Curves are normalized to each perturbation’s on-site
delta norm.

Figure 8:
Spatial decay generalizes beyond adjacent-key typos.
Relative-norm decay curves for AdjacentKey typos (dashed blue,
mid condition), $?\to.$ (orange, mild), and $?\to/$ (red,
severe), on Llama-3.1-8B layer 31. Despite on-site rotation
differing $\sim 2.4{\times}$ across types, the relative-decay
profiles overlap.

## Appendix D Per-Architecture Score-Shift Distributions

Table 8 reports the distribution of per-sample
score shifts (perturbed $-$ clean) under the stacked-typing bundle,
by architecture and ground-truth class. Linear (single-position) has
heavy-tailed shifts on the malicious side (5th percentile $-0.49$ ,
median $\approx 0$ ); aggregating probes reduce per-sample magnitude
dramatically, with MLP/Attention/MultiMax showing near-zero shifts
at every percentile.

Table 8:
Per-architecture score-shift summary. Quantiles of
per-sample (perturbed $-$ clean) score shift under the
stacked-typing bundle, by probe and class
(B  $=$  benign, M  $=$  malicious). 5-fold CV pooled,
Llama-3.1-8B layer 31.

| Probe | Cls | 5th | Median | Mean | 95th

| Linear | B | $-0.043$ | $+0.000$ | $+0.021$ | $+0.185$

| | M | $\mathbf{-0.490}$ | $-0.000$ | $-0.053$ | $+0.021$

| Mean Lin. | B | $-0.010$ | $+0.002$ | $+0.014$ | $+0.091$

| | M | $-0.022$ | $-0.000$ | $-0.003$ | $+0.006$

| MLP | B | $-0.000$ | $+0.000$ | $+0.005$ | $+0.014$

| | M | $-0.000$ | $+0.000$ | $-0.001$ | $+0.002$

| Attention | B | $-0.000$ | $+0.000$ | $+0.007$ | $+0.009$

| | M | $-0.000$ | $+0.000$ | $-0.001$ | $+0.001$

| MultiMax | B | $-0.000$ | $+0.000$ | $+0.006$ | $+0.007$

| | M | $-0.000$ | $+0.000$ | $-0.002$ | $+0.000$

## Appendix E Perturbation-Augmentation Training Details

#### Training-time augmentation policy.

For each training sample (5-fold CV, fold of $\approx 134{,}750$
samples), the probe is trained on activations re-extracted from text
augmented by independent near-end perturbation families. Each family
fires independently with probability $0.5$ per sample, with a guarantee
that at least one perturbation applies, so a training sample sees one
to four perturbations applied individually:

•

last_word_typo: QWERTY adjacent-key typo on the last alphabetic word.

•

trail_period: toggle the trailing period.

•

early_shift: uppercase a letter before punctuation.

•

terminal_punct: one of $\{?\to.,\,?\to/,\,.\to/\}$ chosen uniformly per sample.

#### Per-perturbation evaluation.

Table 9 reports per-perturbation
TPR@FPR $=$ 1% (5-fold mean $\pm$ std) for the augmented probe.

Table 9:
Augmented probe per-perturbation evaluation. 5-fold CV
mean $\pm$ std. The full bundle (last row) is the 5-component
compound perturbation reported in the body.

| Perturbation
|
TPR@FPR $=$ 1%

| Clean (no perturbation) | $97.54\pm 0.14\%$

| last_word_typo | $95.68\pm 0.26\%$

| trail_period | $95.54\pm 0.25\%$

| early_shift | $95.52\pm 0.43\%$

| $?\to.$ | $97.98\pm 0.20\%$

| $?\to/$ | $96.60\pm 0.37\%$

| $.\to/$ | $93.84\pm 0.59\%$

| full_bundle (5-component) | $93.84\pm 0.53\%$

## Appendix F Dataset Details

Table 10 lists the 29 datasets used in the
probe-consequence and multi-architecture experiments
(§6). Most source datasets are far larger than the
counts reported here (e.g., Open-Orca contains ${\sim}3$ M samples, BIPIA
contains $100$ k $+$ injection examples). We sample from each
dataset during cache ingestion, with caps chosen to keep the training
corpus tractable ( $N{=}168{,}440$ samples total) and to avoid any
single dataset dominating the empirical distribution. The $N$ column
reports the actual per-dataset sample counts loaded into the cache; for
datasets smaller than the cap, $N$ reflects the full dataset size (e.g.,
Gandalf at $114$ , Scam at $25$ ). Probe training uses balanced class
weights (Appendix G) to compensate
for the moderate class imbalance reported at the bottom of
Table 10.

Table 10:
Datasets used in probe-consequence and multi-architecture
experiments (29 sources; $N{=}168{,}440$ samples). Grouped by
label class: malicious (with attack-type subheaders), mixed
(labelled examples on both sides), benign.
$\ddagger$  merged with a larger dataset for LODO fold stability
(Gandalf $\to$ Mosscap; Scam $\to$ AdvBench).

| Dataset | Category | Content | $N$

| Malicious — direct prompt injection

| Mosscap | direct PI | Direct injections | 10,000

| Jayavibhav | direct PI | Direct injections | 10,000

| Deepset | direct PI | Direct injections | 546

| Yanismiraoui | direct PI | Direct injections | 1,034

| Malicious — indirect prompt injection

|
BIPIA (Yi et al., 2025)

| indirect PI | Email/code/table embed | 15,000

|
InjecAgent (Zhan et al., 2024)

| indirect PI | Tool-call injections | 1,020

| LLMail | indirect PI | Email challenge | 10,000

|
AgentDojo (Debenedetti et al., 2024)

| indirect PI | Agent task injections | 5,000

|
Gandalf‡

| indirect PI | Summarization | 114

|
Scam‡

| indirect PI | Scam scenarios | 25

| Malicious — jailbreak

|
WildJailbreak (Jiang et al., 2024)

| jailbreak | Adversarial jailbreaks | 2,000

| Jailbreak-Cls | jailbreak | Jailbreak classification | 1,044

| Malicious — harmful / unsafe requests

|
AdvBench (Zou et al., 2023)

| harmful req. | Unsafe requests | 520

|
HarmBench (Mazeika et al., 2024)

| harmful req. | Unsafe requests | 400

| Mixed (labelled on both sides)

| SafeGuard | mixed | PI benchmark
|
8,236 ( $30\%$ mal)

| Qualifire | mixed | PI benchmark
|
5,000 ( $40\%$ mal)

| Benign

| Enron | benign | Email corpus | 10,000

| Dolly-15k | benign | Instruction following | 10,000

| Open-Orca | benign | Reasoning instructions | 10,000

| Prompts-Ranked | benign | User prompts | 10,000

| Alpaca | benign | Instruction following | 10,000

| SoftAge | benign | Prompt engineering | 1,001

| Bitext CS | benign | Support chat | 10,000

| Code Exercise | benign | Python exercises | 10,000

| Python-Alpaca | benign | Python instructions | 5,000

| Python-25k | benign | Python snippets | 5,000

| Writing Prompts | benign | Creative writing | 10,000

| xLAM | benign | Tool use | 5,000

| APIGen-MT | benign | Multi-turn tools | 2,500

| Total corpus
|
$168{,}440$  ( $32.4\%$ mal)

#### Source paths.

Canonical Hugging Face or GitHub repositories for each dataset are
listed in Appendix J,
Table 18.

## Appendix G Probe Training Hyperparameters

#### Layer choice.

We probe activations at layer 31, the final residual-stream layer
before the unembedding projection.
Fomin (2026) sweep five Llama-3.1-8B layers
(19, 23, 25, 27, 31) at positions $-5$ and $-1$ under
Leave-One-Dataset-Out evaluation and report that layer 31 and layer 27
at position $-5$ are competitive on aggregate weighted accuracy
( $81.8$ – $82.3\%$ ); they explicitly select layer 31 as a principled
default (final layer, last user token) rather than as a performance
optimum, and observe that no single layer dominates across all
datasets. We adopt the same layer 31 default and reproduce the
finding on our 29-dataset corpus.

#### Training.

All probes are trained as binary classifiers on Llama-3.1-8B-Instruct
layer 31 activations using PyTorch with the Adam optimizer, balanced
class weights, and random seed 42.
Table 11 lists architecture-specific and
training hyperparameters used in the 5-fold stratified cross-validation
reported in §6.

Table 11:
Probe training hyperparameters (5-fold CV, all probes use
Adam, balanced class weights, weight decay $3{\times}10^{-3}$ , early
stopping with patience 50, random seed 42).

| Probe | Input scope | Hidden | Heads | Activation | LR | Batch size | Normalisation

| Linear (user EOT) | user EOT (1 token) | — | — | — | $10^{-3}$ | full-batch | standard

| MLP (user EOT) | user EOT (1 token) | 100 | — | ReLU | $10^{-4}$ | 64 | none

| Mean Linear (last 16) | last 16 tokens | — | — | — | $10^{-3}$ | full-batch | standard

| MLP (all) | full user-turn | 100 | — | ReLU | $10^{-4}$ | 64 | none

| Attention (all) | full user-turn | 100 | 10 | ReLU | $10^{-4}$ | 64 | none

| MultiMax (all) | full user-turn | 100 | 10 | ReLU | $10^{-4}$ | 64 | none

Linear and Mean Linear probes use 1000 training epochs (full-batch
gradient descent on the standardised activations); MLP-based probes
(MLP, Attention, MultiMax) use 5 epochs of mini-batch SGD with no input
normalisation, following the convention of
Kramár et al. (2026). All architectures share the
class-weighting and optimisation settings; the training set in each
fold is $\approx 134{,}750$ samples and the test fold is
$\approx 33{,}690$ .

## Appendix H KV-Fork Supplementary Details

### H.1 Role-Scoping Ablation

We test whether the post-user role choice for the KV-fork suffix
(system block vs. user-message extension) produces different
attenuation of the perturbation effect, holding suffix content fixed.

#### Setup.

Three placements paired per prompt: none (no suffix;
reference baseline at user-EOT), user_role (suffix inside
the user message), and system_role (suffix in a post-user
system block). Two perturbations: a single adjacent-key typo and
the $5$ -component bundle from §6. $N{=}100$
OpenOrca prompts ( $150$ – $300$ words), seed-paired; layers
$8,16,24,31$ . The metric is the angular shift between clean and
perturbed activations at the end-of-suffix probe-readout position.

Figure 9:
End-of-suffix angular shift by role and layer.
System-role suffix (blue) yields roughly half the residual
angular shift of user-role suffix (red) at every measured layer,
for both perturbations (adjacent-key dashed, bundle solid).
Paired difference at L31 is $-9.03^{\circ}$ (adjacent-key) and
$-8.77^{\circ}$ (bundle). Llama-3.1-8B, $n{=}100$ OpenOrca prompts.

Distance-based dilution (§5.2) is part of the story,
but role choice adds substantial extra attenuation. Combined with
the intent-preservation argument (§7.3), system-role
placement is justified empirically and architecturally.

### H.2 Compute Overhead

We benchmark inference-time forward-pass cost of the KV-fork suffix
on Llama-3.1-8B-Instruct (bf16, sdpa attention, single
A100 80GB) at batch sizes $1$ , $8$ , and $32$ , over four user-message
length buckets ( $74$ , $235$ , $559$ , $1531$ tokens after
chat-template; constructed by tiling a representative prompt with
filler text). The KV-fork condition appends the $30$ -token neutral
suffix from §7.3. Forward-pass-only timings, paired,
median over $10$ reps after $3$ warmup. $\Delta$ ms is
(KV-fork) $-$ (baseline).

Table 12:
KV-fork forward-pass overhead by length bucket.
Llama-3.1-8B-Instruct, bf16, A100. KV-fork suffix is $30$ tokens.
$\Delta$ is (KV-fork) $-$ (baseline) median over $10$ reps;
$n_{\text{base}}$ rounded to the nearest  $5$ .

| Bucket | $n_{\text{base}}$ | batch | base (ms)
|
$\Delta$ (ms)

| short | 75 | 1 | 31.5 | 0.3

|
| | 8 | 59.1 | 23.6

|
| | 32 | 200.8 | 71.2

| medium | 235 | 1 | 34.1 | 2.9

|
| | 8 | 167.0 | 19.1

|
| | 32 | 620.8 | 85.1

| long | 560 | 1 | 59.6 | 1.4

|
| | 8 | 387.4 | 15.9

|
| | 32 | 1468.0 | 88.0

| very long | 1530 | 1 | 141.7 | 3.8

|
| | 8 | 1043.6 | 25.9

|
| | 32 | 4151.7 | 135.2

KV cache memory marginal is $3.75$ MB per request (analytic;
$30$ tokens $\times$ $32$ layers $\times$ $8$ KV heads $\times$
$128$ head-dim $\times$ $2$ (K,V) $\times$ $2$ bytes bf16),
independent of baseline length. For other model architectures this
marginal scales with layer count, KV-head count, and head dimension;
the $30$ -token constant is the smaller factor. The suffix’s absolute
additional cost grows modestly with baseline length at fixed batch
but remains bounded: at batch $1$ the cost is under $4$ ms regardless
of prompt length; at batch $32$ it ranges $71$ – $135$ ms across the
four buckets.

## Appendix I Leave-One-Dataset-Out Evaluation

We measure out-of-distribution generalization following the LODO
protocol of Fomin (2026): for each of the
$K{=}29$ datasets in our benchmark
(Appendix F), we retrain each probe
architecture on the union of the remaining $28$ and evaluate on
the held-out dataset. This isolates whether the probe’s signal
transfers to a distribution it has not seen in training, rather
than memorizing dataset-specific surface patterns.

#### Metrics and threshold convention.

We evaluate at a fixed decision threshold of $0.5$ alongside
ROC-AUC, following Fomin (2026). Fixed-threshold
accuracy measures the deployment-realistic quantity the LODO setup
is designed to test: how well a probe shipped without per-domain
calibration data recognises a new distribution. Threshold $0.5$ is
the symmetric midpoint of the sigmoid output and requires no
held-out information from the new fold. Per-fold threshold tuning
would instead answer whether probe + per-domain calibrator
transfers, conflating probe quality with calibration skill.

Of the $29$ folds, only $7$ contain both classes and admit AUC;
the other $22$ are single-class. For single-class folds, accuracy
reduces to recall (malicious-only folds, $n{=}9$ ) or to
$1{-}\mathrm{FPR}$ (benign-only folds, $n{=}13$ ). We report
sample-weighted accuracy across the full held-out population
( $N{=}168{,}440$ ), per-fold mean AUC across the $7$ mixed folds,
and pooled AUC over their concatenated held-out predictions.
We do not report cross-fold mean accuracy: dataset sizes vary
$\!\sim\!600\times$ ( $25$ to $15{,}000$ samples), so an unweighted
average across folds is dominated by sampling noise on the smallest
folds and lacks a clean population interpretation; per-distribution
heterogeneity is reported directly in
Table 14 below.

#### Aggregate results.

Probes degrade meaningfully under LODO: best mean per-fold AUC is
$0.918$ (Attention) against the $0.998$ in-distribution baseline
(§6). Mean and pooled AUC also
diverge: MLP’s pooled AUC ( $0.747$ ) falls $10$ pp below Mean
Linear’s ( $0.845$ ) despite a competitive mean per-fold AUC
( $0.844$ ), indicating that non-linear probes maintain within-fold
rank but produce scores that are not commensurable across LODO folds.

Table 13: LODO clean evaluation per probe architecture.
$29$ held-out datasets, Llama-3.1-8B-Instruct, layer 31. Mean
AUC and pooled AUC are computed on the $7$ mixed-class folds.
Weighted accuracy is sample-weighted across the full held-out
population ( $N{=}168{,}440$ ) at decision threshold $0.5$ ,
following Fomin (2026). Bold marks per-column
best.

| Architecture | Pooled | Mean AUC | Weighted

| | AUC
|
( $\pm$ std)

| ACC

| Linear (EOT) | 0.796
|
0.814 $\pm$ 0.164

| 0.775

| Mean Linear (last 16) | 0.845
|
0.863 $\pm$ 0.127

| 0.800

| MLP (all) | 0.747
|
0.844 $\pm$ 0.131

| 0.798

| Attention (all) | 0.810 | 0.918 $\pm$ 0.057 | 0.810

| MultiMax (all) | 0.839
|
0.879 $\pm$ 0.084

| 0.821

#### Per-dataset breakdown.

Three patterns in Table 14 are load-bearing.
First, several folds are catastrophic across all five
architectures (e.g., APIGenMT below $3\%$ ), an extreme
manifestation of score-scale drift on distributions none of the
probes saw in training. Second, per-fold rankings flip relative
to the aggregate (e.g., the linear probe tops InjecAgent’s
recall while MultiMax bottoms it), so the aggregate ranking
that favors capacity is not a uniform per-distribution
improvement. Third, cross-fold standard deviations
( $\pm 0.25$ – $\pm 0.30$ on accuracy) show that LODO performance
is dominated by held-out dataset choice rather than architecture,
in stark contrast to the $\pm 0.01$ standard deviation observed
under 5-fold CV (§4).

Together these patterns point past a generic “probes overfit”
to a sharper claim: the training distribution couples intent
with dataset-specific surface attributes, so what the probes
learn is an intent-aligned mixture rather than intent in
isolation. LODO holds out the dataset-specific component by
construction, leaving a per-fold residual that varies in
composition. This complements the SAE feature-retention finding
of Fomin (2026): same phenomenon, observed
here at prediction accuracy rather than at the feature level.

Table 14: Per-dataset clean LODO accuracy at threshold
$0.5$ . Layer 31, Llama-3.1-8B-Instruct. Mixed-class folds
report standard accuracy; malicious-only folds report recall;
benign-only folds report $1{-}\mathrm{FPR}$ . %mal column shows
malicious fraction in the held-out fold. ScamDataset clean test
set is included; perturbation evaluation is unavailable for it
due to source repository unavailability at evaluation time.

| Dataset | N | %mal | Linear | Mean Lin. | MLP | Attn. | MMax

| Mixed (accuracy)

| AgentDojo | 5,000 | 99 | 21.7 | 53.2 | 84.0 | 77.5 | 28.4

| BIPIA | 15,000 | 95 | 40.4 | 55.5 | 32.9 | 36.7 | 58.5

| Deepset | 546 | 37 | 76.6 | 79.7 | 82.6 | 84.2 | 80.2

| Jailbreak Classification | 1,044 | 50 | 94.4 | 95.6 | 95.3 | 96.3 | 96.1

| Jayavibhav | 10,000 | 50 | 76.7 | 75.4 | 74.7 | 72.5 | 75.9

| Qualifire | 5,000 | 40 | 75.0 | 77.1 | 75.0 | 75.6 | 76.3

| SafeGuard | 8,236 | 30 | 97.5 | 96.4 | 94.4 | 96.2 | 94.5

| Malicious-only (recall)

| AdvBench | 520 | 100 | 96.0 | 96.9 | 87.5 | 98.1 | 99.2

| Gandalf Summarization | 114 | 100 | 98.2 | 97.4 | 92.1 | 100.0 | 100.0

| HarmBench | 400 | 100 | 43.0 | 65.5 | 52.0 | 64.2 | 68.2

| InjecAgent | 1,020 | 100 | 100.0 | 90.2 | 35.8 | 56.1 | 6.6

| LLMail | 10,000 | 100 | 79.0 | 38.4 | 44.3 | 22.7 | 22.4

| Mosscap | 10,000 | 100 | 78.0 | 89.4 | 99.9 | 100.0 | 100.0

| Scam | 25 | 100 | 96.0 | 64.0 | 100.0 | 100.0 | 92.0

| Wild Jailbreak | 2,000 | 100 | 85.7 | 91.7 | 86.0 | 90.6 | 90.6

| Yanismiraoui | 1,034 | 100 | 56.0 | 52.3 | 26.3 | 42.6 | 63.0

| Benign-only ( $1-$ FPR)

| APIGenMT | 2,500 | 0 | 0.4 | 0.1 | 0.0 | 2.8 | 2.9

| Alpaca | 10,000 | 0 | 99.5 | 99.6 | 99.9 | 99.7 | 99.9

| Bitext Customer Support | 10,000 | 0 | 97.9 | 98.7 | 99.5 | 99.0 | 98.2

| Code Exercise | 10,000 | 0 | 99.9 | 100.0 | 100.0 | 100.0 | 100.0

| Dolly15k | 10,000 | 0 | 99.8 | 99.8 | 99.9 | 99.9 | 99.8

| Enron | 10,000 | 0 | 74.5 | 72.7 | 86.5 | 65.1 | 86.8

| OpenOrca | 10,000 | 0 | 60.9 | 85.2 | 86.3 | 96.2 | 87.7

| Prompts Ranked 10k | 10,000 | 0 | 92.1 | 95.3 | 96.0 | 97.0 | 95.7

| Python Code Alpaca | 5,000 | 0 | 99.1 | 100.0 | 100.0 | 100.0 | 100.0

| Python Codes 25k | 5,000 | 0 | 98.2 | 97.0 | 97.0 | 98.1 | 96.5

| SoftAge | 1,001 | 0 | 94.9 | 86.3 | 96.8 | 97.5 | 97.6

| Writing Prompts | 10,000 | 0 | 90.8 | 91.6 | 96.8 | 89.5 | 91.2

| Xlam Function Calling | 5,000 | 0 | 7.3 | 22.2 | 0.2 | 99.5 | 99.9

### I.1 Perturbation under LODO

We apply the stacked-typing bundle and every-second-word (ESW)
perturbations of §6 to each mixed-class
held-out fold ( $n{=}7$ ) and report
$\mathrm{TPR}\,@\,\mathrm{FPR}{=}1\%$ with the threshold
re-calibrated per (fold, condition), matching the operational
convention of §6 so that the perturbation
$\Delta$ reported here is directly comparable.

#### Cross-architecture summary.

Table 15 reports the cross-fold mean
$\mathrm{TPR}\,@\,\mathrm{FPR}{=}1\%$ per architecture. The mean
$\Delta$ under bundle and ESW is small in absolute terms
( $-1$ to $-7$  pp), but this aggregate hides extreme cross-fold
heterogeneity (Table 16).

Table 15: TPR@FPR=1% under LODO with perturbation.
Seven mixed-class folds, threshold re-calibrated per (fold,
condition) to hold $\mathrm{FPR}{=}1\%$ . Mean across folds.
$\Delta$ is perturbed minus clean. In-distribution reference
(Table 2): linear probe clean
$\mathrm{TPR}{=}97.4\%$ , bundle $\Delta{=}{-}12.0$  pp.

| Architecture | TPR clean | TPR bundle | TPR ESW
|
$\Delta$ bundle

|
$\Delta$ ESW

| Linear (EOT) | $38.9\%$ | $35.4\%$ | $33.6\%$
|
$-3.5$ pp

|
$-5.3$ pp

| Mean Linear (last 16) | $44.3\%$ | $43.4\%$ | $40.2\%$
|
$-0.9$ pp

|
$-4.1$ pp

| MLP (all) | $40.6\%$ | $39.1\%$ | $35.5\%$
|
$-1.5$ pp

|
$-5.1$ pp

| Attention (all) | $46.3\%$ | $47.7\%$ | $43.8\%$
|
$+1.4$ pp

|
$-2.5$ pp

| MultiMax (all) | $44.3\%$ | $43.4\%$ | $38.1\%$
|
$-1.0$ pp

|
$-6.2$ pp

#### Per-fold heterogeneity.

Per-fold drops are heterogeneous
(Table 16). Folds with low clean baselines
(AgentDojo, BIPIA) register near-zero or positive $\Delta$ while folds with
ID-comparable baselines (SafeGuard, Jailbreak Classification)
register ID-comparable drops, indicating that ID perturbation
fragility persists under distribution shift but is masked at the
aggregate by floor effects.

Table 16: Per-fold linear probe TPR@FPR=1% under
perturbation (seven mixed folds). Threshold re-calibrated
per (fold, condition). $\Delta$ is perturbed minus clean.

| Fold | Clean | Bundle
|
$\Delta$ bundle

| ESW
|
$\Delta$ ESW

| AgentDojo | $0.5\%$ | $0.1\%$
|
$-0.4$ pp

| $3.6\%$
|
$+3.1$ pp

| BIPIA | $2.4\%$ | $6.2\%$
|
$+3.8$ pp

| $7.0\%$
|
$+4.5$ pp

| Deepset | $34.0\%$ | $30.0\%$
|
$-3.9$ pp

| $28.2\%$
|
$-5.8$ pp

| Jailbreak Classification | $82.4\%$ | $71.2\%$
|
$-11.2$ pp

| $76.4\%$
|
$-5.9$ pp

| Jayavibhav | $44.8\%$ | $43.6\%$
|
$-1.3$ pp

| $32.4\%$
|
$-12.4$ pp

| Qualifire | $14.4\%$ | $11.0\%$
|
$-3.4$ pp

| $10.9\%$
|
$-3.5$ pp

| SafeGuard | $93.5\%$ | $85.4\%$
|
$-8.1$ pp

| $76.7\%$
|
$-16.8$ pp

### I.2 KV-Fork under LODO

We re-train the linear probe behind the KV-fork suffix
(§7.3) on the in-distribution training corpus and
evaluate under the LODO protocol of §I.
Two metrics are reported: clean weighted accuracy at threshold
$0.5$ across all $28$ available held-out folds (matching
Table 13), and clean / perturbed
$\mathrm{TPR}\,@\,\mathrm{FPR}{=}1\%$ on the $7$ mixed-class
folds with threshold recalibrated per (fold, condition)
(matching §I.1).

The KV-fork lifts the clean weighted accuracy of a single-position
linear probe by $+4.0$ pp, enough to bring it into the
architectural top tier under LODO (cf. Mean Linear $0.800$ ,
Attention $0.810$ , MultiMax $0.821$ in
Table 13). At the recalibrated FPR $=$ 1%
operating point the clean gain is $+8.4$ pp, consistent with the
suffix stabilising the readout context against the score-scale
drift that dominates LODO behaviour for non-augmented probes
(§I). Suffix-content variation is left to
future work.

Table 17: KV-fork under LODO. Clean weighted accuracy
is sample-weighted across the $28$ held-out folds at threshold
$0.5$ (Scam unavailable). $\mathrm{TPR}\,@\,\mathrm{FPR}{=}1\%$
is on the $7$ mixed-class folds with threshold recalibrated
per (fold, condition); cross-fold mean.

| Condition | Weighted ACC | TPR clean
|
$\Delta$ TPR bundle

| No KV-fork (Linear EOT) | $77.5\%$ | $38.9\%$
|
$-3.5$ pp

| KV-fork (neutral suffix) | $81.5\%$ | $47.2\%$
|
$-4.5$ pp

|
$\Delta$ (KV-fork $-$ no-fork)

|
$+4.0$ pp

|
$+8.4$ pp

|
$-1.0$ pp

## Appendix J Dataset Source Paths

Table 18: Dataset source paths. Canonical Hugging Face or
GitHub repository for each of the 29 datasets in
Appendix F.

| Dataset | Source

| Enron | amanneo/enron-mail-corpus-mini

| Dolly-15k | databricks/databricks-dolly-15k

| Open-Orca | Open-Orca/OpenOrca

| Prompts-Ranked | data-is-better-together/10k_prompts_ranked

| Alpaca | tatsu-lab/alpaca

| SoftAge
|
SoftAge-AI/ (gated)

| Bitext CS | bitext/Bitext-customer-support-llm-chatbot-training-dataset

| Code Exercise | iamtarun/python_code_instructions_18k_alpaca

| Python-Alpaca | iamtarun/python_code_instructions_18k_alpaca

| Python-25k | flytech/python-codes-25k

| Writing Prompts | euclaise/writingprompts

| xLAM | Salesforce/xlam-function-calling-60k

| APIGen-MT | Salesforce/APIGen-MT-5k

| SafeGuard | xTRam1/safe-guard-prompt-injection

| Qualifire | qualifire/prompt-injections-benchmark

| Mosscap | Lakera/mosscap_prompt_injection

| Jayavibhav | jayavibhav/prompt-injection-safety

| Deepset | deepset/prompt-injections

| Yanismiraoui | yanismiraoui/prompt_injections

| BIPIA | github.com/microsoft/BIPIA

| InjecAgent | github.com/uiuc-kang-lab/InjecAgent

| LLMail | microsoft/llmail-inject-challenge

| AgentDojo | github.com/ethz-spylab/agentdojo

| Gandalf | Lakera/gandalf_summarization

| Scam | github.com/1Password/SCAM

| WildJailbreak | allenai/wildjailbreak

| Jailbreak-Cls | jackhhao/jailbreak-classification

| AdvBench | walledai/AdvBench

| HarmBench | walledai/HarmBench
