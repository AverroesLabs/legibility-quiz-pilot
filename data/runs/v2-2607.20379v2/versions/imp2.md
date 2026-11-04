# Train the Model, Not the Reader: Decodability Supervision for Verifiable
Activation Explanations

Hiskias Dingeto

###### Abstract

Natural-language autoencoders score explanations of hidden activations by
reconstruction. On a released Qwen-2.5-7B verbalizer, explanations reconstruct
well above chance while as few as ${\sim}2\%$ of their specific claims are
grounded. Under exact synthetic ground truth, standard training consistently
yields co-adapted private codes. Reader-side fixes leave both unchanged. We
contribute two audit protocols, the grounding/truth cross and the
independent-evaluator swap, and RECAP (Readable Encodings via Co-trained
Auxiliary Predictors), linear heads co-trained with the target model on
designated content. On RECAP-trained sandbox models, fresh verbalizers state the
designated content truly and codes are absent, at $+0.001$ nats. The result
replicates on a pretrained Pythia-160M under a target-design rule. Content
becomes reliably probe-decodable, and a fresh verbalizer conveys it in part
(truth 0.44–0.46 against a near-zero control). On the pretrained model, an
independent probe ranks the verbalizer's true claims above its false ones at AUC
0.96, versus 0.82 without RECAP. A score-maximizing lying adversary suppresses
about 87% of its lie penalty. The RECAP probe still flags the lies at AUC 0.95,
while the control probe falls to 0.51.

StackOne Technologies

hiskias@stackone.com

## 1 Introduction

NLAs (Fraser-Taliente et al. 2026) attach a round-trip reconstruction score to
every explanation of a hidden state. The score needs no labels and no human
judge. Released verbalizer/reconstructor pairs exist for several open models,
and related reader objectives are proliferating (Pan et al. 2026; Karvonen et
al. 2025; Huang et al. 2025).

The test is passed in two ways, and neither is faithful.

- **Gist.** On a released Qwen-2.5-7B NLA, explanations reconstruct well above
  chance, yet a per-claim flip audit detects grounding in only ${\sim}2\%$ of
  salient specific claims. This figure is a lower bound given the audit's low
  sensitivity.
- **Codes.** In a synthetic sandbox with exact ground truth, co-trained pairs
  under standard training consistently develop private codes in two scenarios,
  with grounded-minus-true gaps of 0.30–0.80.

Two training fixes fail. Minimizing the distance between a hidden state and its
reconstruction removes probe-readable content. Rewarding the verbalizer for
grounding raises the score through which sentences the metric counts, while
claim-level grounding does not increase.

RECAP (Figure 1b) uses linear heads on external targets, co-trained with the
target model (§5). Fresh NLAs on RECAP models then state designated content
faithfully in the sandbox under both co-trained and independent evaluators, and
codes are absent. The self-reconstruction objective collapses decodable content.
Frozen probes go stale under drift, which a label-free realignment fully
reverses.

Figure 1: The two architectures. (a) A natural-language autoencoder (prior work)
(Fraser-Taliente et al. 2026): AV maps an activation to text, AR maps it back,
and reconstruction quality is the test; only AV and AR train. The score can stay
high while the text asserts an ungrounded specific (red “mirror”). (b) RECAP
(this work) trains the target model. Top: co-trained linear heads on external
targets. Bottom: after training, an independent fresh probe (not the heads)
recovers the content (AUC 0.95) and a fresh verbalizer states it truly.

Contributions:

•

Claim-level audit. A per-claim flip audit with per-edit validity controls on a
released verbalizer finds its specific claims largely ungrounded. Its score
survives paraphrase and collapses under content-word masking.

•

Private codes. In the sandbox, the grounding/truth cross and the
independent-evaluator swap detect and quantify private codes.

•

RECAP. Self-reconstruction collapses decodable content and frozen probes drift.
RECAP restores claim-level faithfulness in the sandbox, and on Pythia-160M it
yields probe-decodable designated content at a measured tax.

•

Adversarial monitor. A score-gaming text-editing adversary suppresses
${\sim}87\%$ of its lie penalty on both models. RECAP's activation probe still
flags the lies at AUC $0.95$, while the control probe is at $0.51$.

## 2 Preliminaries and Experimental Setup

### 2.1 The Reconstruction Test and Audit Protocols

The AV emits $z$ from tap activation $h$, and the AR returns $\hat{h}$. With
target weights $\theta$ frozen,

| | $\mathcal{J}_{\mathrm{read}}\;=\;\max_{\mathrm{AV},\,\mathrm{AR}}\;\cos\!\big(\mathrm{AR}(\mathrm{AV}(h)),\,h\big),$
| | (1)

and reconstruction quality serves as the faithfulness test (Figure 1a).
Fraser-Taliente et al. (2026) present rising reconstruction as tracking
informativeness, and they caution that nothing in the objective forces
faithfulness. This score is nonetheless what these NLAs report and what users
apply. We report
$\tilde{r}=(\cos(\hat{h},h)-\mathrm{floor})/(1-\mathrm{floor})$, using a
centered cosine with $\mathrm{floor}$ the mean cosine against mismatched
activations. The released pair is Qwen-2.5-7B layer 20 (Qwen Team 2025).

Grounding. With $z_{\neg c}$ the minimal-pair flip of claim $c$ alone,

| | $\Delta\tilde{r}(c)=\tilde{r}(z)-\tilde{r}(z_{\neg c}),$
| | (2)

and $c$ counts as grounded at $\Delta\tilde{r}(c)\geq\tau$. Edits are
validity-filtered. An edit that changes more than its target can inflate
apparent grounding severalfold (Appendix A). We cross grounded/ungrounded with
true/false and focus on grounded-and-false wording. Where such wording is
specific to the co-trained pair under the swap of §3.2, we call it a private
code. These labels are operational and carry no intentional stance. The
evaluator gap is $\tilde{r}_{\mathrm{co}}-\tilde{r}_{\mathrm{ind}}$, computed
under the co-trained and an independent reconstructor.

Probes are freshly fitted linear decoders (Alain and Bengio 2017; Belinkov
2022). Faithfulness is claim-level throughout. We do not use it for downstream
computation, and grounding is never meant as grounding in the world.

#### Settings.

We use three settings.

1. The released Qwen-2.5-7B layer-20 NLA on in-distribution web text
   ($n{=}1{,}517$ audited claims).
2. A sandbox of two templated scenarios, story and marketplace, with exact
   judge-free ground truth and retained, faded, and unread slots.
3. Continued pretraining of Pythia-160M (Biderman et al. 2023) with
   self-supervised heads, with tax measured against a shared control.

Protocols, grammars, seeds, and hyperparameters are given in Appendices D and K.

## 3 Auditing Reconstruction-Scored Explanations

On the released NLA, the missing specifics are largely absent from the tap. In
the sandbox the content is present, and the independent-evaluator swap exposes
the codes.

### 3.1 Counterfactual Audit of a Released Verbalizer

On in-distribution web text, the released NLA's explanations reconstruct at
$\tilde{r}=0.84$. Grounding appears in about $2\%$ of LLM-salient specific
claims: $4.2\%$ / $2.1\%$ / $1.6\%$ at $\tau=0.02$ / $0.05$ / $0.10$ (Appendix
E). These are lower bounds. An append-and-rerank control recovers a topic known
to be encoded at only $28\%$ raw.

The gist–specifics gap is the more robust statistic. Specific facts are probed
about $3\times$ less accurately than gist (${\sim}6\%$ versus ${\sim}18\%$
chance-corrected per position). Mean-pooled gist reaches ${\sim}50\%$ (Appendix
E). The score survives paraphrase and collapses under content-word masking. The
pattern holds across layers 16–27 and is specific to reconstruction-style
readouts. The model's own output head does see late-layer specifics in its
next-token predictions. Appendix E gives bootstrap intervals and a flip-noise
floor below the smallest $\tau$.

We tried two reader-side fixes, rejection sampling and GRPO, both against a
clause-level grounding reward. Each raised its proxy, while the independent
claim-level audit stayed null at every $\tau$, with CIs straddling zero. Text
statistics attribute the gains to sentence restructuring, that is, merging and
packaging of policy-controlled units.

### 3.2 Audit Under Synthetic Ground Truth

In every sandbox run the toy NLA asserts every slot and scores $\tilde{r}$
0.71–0.86 in both scenarios (example in Appendix D). Truth rates by slot are:

| slot | truth rate
|---|---
| fully decodable | 0.44–0.96 across seeds (grounded 0.76–0.88 in all)
| partially decodable | 0.15–0.26
| unread | at chance

Under standard training, 5/5 runs develop private codes, with grounded-minus-true
gaps of 0.30–0.80. Codes can occupy several slots at once. In the marketplace,
one code occupies an unread slot.

The independent evaluator is trained without exposure to the audited pair's
outputs. Under it, grounding on code-carrying slots collapses from 0.53–0.89 to
0.08–0.23, while grounding of true claims survives. The swap locates every claim
in the cross. The evaluator gap is 0.17–0.38 on standard runs versus
${\sim}0.003$ on RECAP runs (§5). An independent evaluator trained on a disjoint
paraphrase family (different frames and slot order, slot words shared)
reproduces the pattern within ${\sim}0.05$ across three seeds. Code grounding
stays collapsed while genuine grounding survives, with RECAP slots at 1.00.

Standard verbalizers also under-read content that is present. In the
marketplace, slots 55–72% decodable from the target are stated truly only
15–26% of the time. This readout gap sits alongside the storage gap, and the
intervention below closes both.

## 4 Representation Collapse and Probe Drift

Collapse. A target-side objective pulls activations toward their own
reconstruction. At every tested strength and initialization, it removes
decodable slot content with perplexity unchanged. Freshly retrained probes also
lose accuracy. This is consistent with collapse analyses in non-contrastive SSL
(Chen and He 2021; Tian et al. 2021).

Drift. Probes matched to a training model decay to near-chance within 1,500
continued steps and are at chance across seeds in all six pairs. An affine map
fitted on paired activations, without labels, restores native performance in
all nine cases. Rotation alone recovers about half; full recovery needs a
non-orthogonal change of rank ${\sim}$ 16–64 (Appendix H). Duan (2026)
concurrently reports the same drift and realignment at 2B and 7B parameters.

Probe-based monitoring of a model still in training therefore needs explicit
recalibration. Realignment corrects how a probe reads the representation. RECAP
changes what the representation contains. Affine equivalence also suggests
adapter-based cross-model probe transfer (Zhao et al. 2026).

## 5 The RECAP Objective

| configuration | tax (nats) | probe AUC

| control (baseline) | n/a | 0.77–0.80

| $K{=}8$ , naive, $w{=}1.0$ | $+0.047$ | 1.000

| $K{=}64$ , naive, $w{=}1.0$ | $+0.064$ | 0.95–0.97

| $K{=}64$ , balanced, $w{=}0.1$ | $+0.010$ | 0.97–0.99

| $K{=}512$ , balanced, $w{=}1.0$ | $+0.14$ – $0.20$ | 0.97–0.99

Table 1: RECAP at scale: held-out tax (LM loss minus control, nats) and
fresh-probe AUC on designated targets. The $K{=}64$ balanced tax is a 3-seed
paired difference (95% CI $[-0.015,+0.034]$ ); the other balanced row is a
3-seed range, the rest single-seed.

### 5.1 Method

Linear heads read the tap and predict external targets $y$. In the sandbox $y$
is the slot values; at scale it is self-supervised text functions. With head
parameters $\phi$,

| | $\mathcal{J}_{\mathrm{model}}\;=\;\min_{\theta,\,\phi}\;\mathcal{L}_{\mathrm{LM}}(\theta)\;+\;w\,\mathcal{L}_{\phi}\big(h(\theta),\,y\big),$
| | (3)

At scale, $\mathcal{L}_{\phi}$ is class-balanced presence BCE over content words
plus identity CE (cf. §5.3). Every reported decodability number comes from
independently fitted probes, never from the heads. The construction follows
deep supervision (Lee et al. 2015; Szegedy et al. 2015).

### 5.2 Sandbox Results

Designated content is 100% decodable by independent probes, at $+0.001$ nats,
in all five runs. On unread slots the heads decode nothing in any run. At scale
the tax is $+0.010$ nats and depends on target design (§5.3).

Fresh NLAs on RECAP targets state designated content at ${\sim}1.00$ truth and
grounding under both co-trained and independent evaluators. Remaining false
content sits at grounding ${\sim}0$, and no codes appear in any RECAP run
(Appendix D ledger). Both NLA conditions receive identical training, including a
cold-start SFT on gold descriptions; only the target model differs.

Figure 2: Decoding content from a real model’s hidden state (one held-out Pile
(Gao et al. 2020) passage; fresh probes read the layer-6 state at the marked
position). On the RECAP model the probes recover the trailing-window content
words and the offset-8 token; on the control the readout is scrambled. Same
probe recipe and data; only the target model differs.

### 5.3 Scaling to Pythia-160M

Results are in Table 1 and Figure 2.

- **Decodability.** Fresh probes reach 0.95–1.00 AUC for $K{=}8$ and $K{=}64$
  designated content words, against a 0.77–0.80 control.
- **Naive $K{=}512$.** The naive presence head minimizes its loss by always
  predicting “absent” and makes nothing decodable to a fresh probe (Appendix I).
  Class-frequency balancing restores 0.97–0.99 AUC across all 512 targets.
  Targets should be ones whose loss cannot be minimized without genuine
  discrimination.
- **Tax.** For 64 targets the tax is $+0.010$ nats, indistinguishable from
  zero. For 512 targets into $d{=}768$ it is $+0.14$ – $0.20$ nats.
- **Other targets.** A discriminative identity target stays decodable at
  $w{=}0.03$ with tax within noise of zero. The never-designated bucket stays
  at control level in all eight supervised runs.

An MLP head gives the same 0.98–0.99 designated AUC at layer 6. A linear head at
layer 9 lifts AUC from 0.77–0.79 to 0.95–0.98 at $+0.01$ nats. Appendix I
indexes the full sweep: weight, pull strength, dose, realignment map class,
evaluator, scenario, head capacity, and tap layer.

### 5.4 Verbalizer Audit at Scale

We train an AV/AR pair on Pythia-160M taps using the released recipe and audit
it against exact text ground truth.

- **Truth.** Stated context words are true at 0.44 on RECAP versus 0.06 on the
  control.
- **Grounding.** Control words are grounded at 0.48 but true at 0.06. RECAP
  moves the gap from $+0.42$ to $+0.33$.
- **Recall.** RECAP reaches 0.29 versus 0.07 while stating fewer words per
  explanation.
- **Stability.** Results are stable across reconstructor strength and three
  target seeds (truth 0.44–0.46 versus a near-zero control; Appendix J).

At the verbalizer's own recall, a fresh probe on the same tap reaches precision
${\sim}0.80$. The verbalizer reaches 0.44–0.63 across reconstructors and seeds
(Appendix J). We redesigned the reconstructor with:

- standardized multi-token injection;
- MLP capacity;
- a decode-point presence head trained on window occurrence, an external fact
  and not a learned monitor.

This makes designated content linearly decodable at the decode point
(0.79–0.81, two seeds). Generated precision stays at 0.56–0.63, and a matched
control reads 0.03 under the identical head. None of the following closes the
gap: reconstructor scale (a 3B reader matches the 0.5B), reconstruction
optimization, a direct truth reward, injection bandwidth, or negative
supervision. The monitor below checks claims against internals rather than the
prose.

Without heads, decodability erodes within a few hundred steps of fine-tuning.
Continuous maintenance costs $+0.010$ nats; intermittent maintenance costs about
a tenth the compute (Appendix C).

## 6 Probe-Based Monitoring

In the sandbox, fresh per-slot probes read the tap while the verbalizer states
slot values. A claim contradicted by a probe at $p{>}0.9$ is flagged, and exact
ground truth scores the monitor (Appendix B).

- **Standard target.** Every flag is a genuine false claim, but only 27 of 733
  false pre-tap claims are caught, since most fall in low-decodability slots.
- **RECAP target.** No false pre-tap claims remain, and abstention is zero.
  Coverage of designated pre-tap content rises from 37% to 100%. This is a
  coverage gain, not a precision gain, and it is confined to designated content.
- **Post-tap claims.** These are abstained on both targets and are left to the
  grounding/truth cross of §3 (one scenario, one seed, one threshold).

On Pythia-160M, a fresh presence probe ranks true over false claims at AUC
$0.965$ versus $0.823$ for the control, with disjoint $95\%$ intervals. It flags
false claims at $0.93$ recall and $0.87$ precision. At the $1.5\%$ base rate we
report AUC and flag precision/recall in place of coverage. The verbalizer
proposes; the probe verifies.

#### Surviving a score-gaming adversary.

The adversary edits a true stated word into an absent one, choosing the edit
that maximizes the recomputed $\tilde{r}$ (Appendix F). At the $\tilde{r}$-optimal
edit, ${\sim}87\%$ of the lie penalty is suppressed on both models. The
external-truth probe flags these lies at AUC $0.952$ on RECAP and $0.508$ on the
control, which reads the same activation the adversary cannot touch.

Candidate selection moves RECAP AUC from $0.993$ to $0.952$ as the chosen absent
words grow semantically nearer. The threat model is text edits only, excluding
activation-level attacks (Bailey et al. 2024). Within it, immunity is structural
for score manipulation and empirical for candidate selection. A probe-aware
adversary that assumes monitor leakage lies outside the threat model. Against
it, RECAP holds at AUC $0.909$, while the control drops to $0.242$; the
separation grows with adversary strength.

The operating point is not tuned on the evaluation set:

- Recalibrating the flag threshold on a disjoint split leaves recall essentially
  unchanged.
- No control threshold reaches $0.9$ precision even against random lies.
- On never-designated words (ranked 64–128), RECAP has no advantage
  (AUC $0.66$ versus $0.70$; Appendix F).

## 7 Related Work

#### Activation verbalization and its critiques.

Beyond NLAs (Fraser-Taliente et al. 2026), reader-side work includes:

- supervised QA decoders (Pan et al. 2026);
- general activation explainers (Karvonen et al. 2025; Bauer et al. 2026);
- interpreter assistants (Huang et al. 2025);
- training-free readouts (Ghandeharioun et al. 2024);
- cross-model adapters (Zhao et al. 2026);
- SAEs (Huben et al. 2024).

Related critiques:

- Plausible-but-unfaithful chain-of-thought (Turpin et al. 2023; Lanham et al. 2023).
- Causal tests of neuron explanations (Huang et al. 2023).
- Counterfactual faithfulness tests (Atanasova et al. 2023).
- Benchmark-level verbalization critiques, which find many tasks solvable
  without the target's internals (Li et al. 2025).

We instead audit the reconstruction test per claim and counterfactually on a
released NLA, and our repair trains the target model.

#### Scalable oversight.

Reconstruction scoring is a cheap automatic oversight signal (Amodei et al.
2016; Bowman and others 2022), and AI control seeks robustness to subversion
(Greenblatt et al. 2024). We contribute a monitor driven by an independent probe
that survives score gaming (Baker et al. 2025).

#### Steganography in learned channels.

Prior hidden channels:

- reconstruction-family objectives since CycleGAN (Chu et al. 2017);
- encoded CoT reasoning (Roger and Greenblatt 2023), which generalizes under
  process supervision (Skaf and others 2025);
- steganographic collusion (Mathew et al. 2025), widened by tool use (Rippin et
  al. 2026) and detected by probes (Rose et al. 2026);
- activation obfuscation against latent monitors (Bailey et al. 2024);
- obfuscated reward hacking under CoT monitoring (Baker et al. 2025).

All of these arise from incentives, process supervision, or an adversary. Our
codes arise from the standard reconstruction recipe alone, with no
steganography objective or communication reward. We supervise the target model
against external ground truth rather than against a monitor.

#### Representation drift, stitching, and monitor staleness.

Background:

- representation equivalence up to transformation (Lenc and Vedaldi 2015;
  Kornblith et al. 2019; Moschella et al. 2023);
- affine feature transfer across LMs (Chen et al. 2025);
- cross-seed stability of pretraining dynamics (van der Wal et al. 2025);
- tuned lenses (Belrose et al. 2023).

Duan (2026) concurrently reports the same staleness and label-free repair at 2B
and 7B. Probe retraining restores detectability after probe-aware fine-tuning
(Wehner and Fritz 2025). We add the recovery anatomy (§4), the cross-seed
failure, exact evidence that information persists when readers fail, and the
separation of drift from collapse.

#### Deep supervision and intrinsic interpretability.

In deep supervision (Lee et al. 2015; Szegedy et al. 2015) the auxiliary target
is the task label; ours is external content. Intrinsic alternatives:

- concept bottlenecks (Koh et al. 2020; Sun et al. 2025);
- concept whitening (Chen et al. 2020);
- codebook features (Tamkin et al. 2023);
- gradient routing (Cloud et al. 2024).

We add linear heads without constraining information flow.
Explanation-side supervision can track behavioral change (Guo et al. 2026); ours
acts on the target.

## 8 Discussion and Limitations

Implications for oversight. Reconstruction scoring stands in for human checking
of internals. The scored models pass it without faithfulness, and an adversary
games it. RECAP-trained designated content is checkable by an independent probe
that survives explanation editing (cf. Baker et al. 2025). The guarantee covers
only designated, decodable content, so RECAP is a component of oversight, not a
complete solution.

Scope and limitations. Evidence comes from a sandbox, one released NLA
(Qwen-2.5-7B), and one small model (Pythia-160M), mostly at single seeds;
frontier scale is untested.

1. Audits cover closed-vocabulary template-parsed claims, and the monitor covers
   only designated content. Open-ended faithfulness is unmeasured.
2. RECAP must be co-trained in and cannot be retrofitted to a frozen model. A
   poorly chosen target can report success while decoding nothing.
3. Decodability is not verbalizability.
4. RECAP certifies storage, not use. Patching the tap barely moves behavior
   (${\sim}0.003$), and computation could route around a monitored subspace
   (Mahinpei et al. 2021; Margeloiu et al. 2021).

Success for reconstruction-tested verbalization is not a higher $\tilde{r}$.
It is per-claim verification against the target, calibrated abstention, and
monitors with recalibration protocols.

## 9 Conclusion

Reconstruction-scored explanations pass via gist and via private codes, and
training against the test yields collapse or proxy gaming. RECAP produces
claim-level faithful verbalizers in the sandbox. On Pythia-160M it produces
probe-decodable designated content, which a fresh verbalizer states truly
against a near-zero control. Sandbox auditor coverage of designated content
rises from 37% to 100%. At scale, a probe ranks true over false claims, reading
the content more reliably than the verbalizer generates it.

## Appendix A Edit Validity for Minimal-Pair Flips

A valid edit flips exactly one claim and nothing else. We compared two judges on
the released NLA:

| judge | valid edits
|---|---
| closed-model | 68.4%
| Gemma-4-12B (pinned revision, temperature 0; used throughout) | 99.3%

Invalid edits were mostly oversized rewrites and inflated apparent grounding
roughly sevenfold. All grounded fractions use the open judge with per-edit
filtering, and per-claim deltas are persisted. In the same head-to-head, a
judge-free mask-and-rescore signal agrees with the flip instrument at AUC 0.997
under the validated judge (Appendix G).

## Appendix B Safety Experiment 1: The Discrepancy Detector

#### Question.

How useful is probe–verbalizer disagreement as a lie detector, and on what does
it depend?

#### Protocol.

Sandbox scenario 1, seed 1, with 400 held-out taps per target. For each slot
type, a fresh multinomial logistic probe (linear, 256-dimensional input) is fit
on the probe split. On evaluation taps, the first stated word of each slot type
is compared against the probe prediction. A claim is flagged when probe
confidence exceeds 0.9 and the probe disagrees, and abstained below threshold.
Exact tuple ground truth scores every decision.

| model | phase | claims | false | flagged | prec. | abstain

| control | pre-tap | 1200 | 733 | 27 | 1.000 | 0.632

| control | post-tap | 800 | 698 | 0 | — | 1.000

| RECAP | pre-tap | 1200 | 0 | 0 | — | 0.000

| RECAP | post-tap | 800 | 706 | 0 | — | 1.000

Table 2: Discrepancy detector, 2,000 claims per target model. Precision is the
fraction of flags that are genuinely false; recall the fraction of false claims
flagged; abstain the fraction not adjudicable at threshold.

#### Findings.

(i) All 27 flags are genuinely false claims, with zero false alarms.
(ii) On the control, 63% of pre-tap claims are unadjudicable, so recall on the
733 lies is 3.7%.
(iii) On RECAP there are 0 false pre-tap claims in 1,200, and abstention is 0%.
(iv) Post-tap claims are 100%-abstained on both targets.
Pre-tap coverage moves from 37% to 100%.

#### Caveats.

One scenario, one seed, one threshold (0.9); “decodable” means linearly
readable.

## Appendix C Safety Experiment 2: Legibility Persistence Under Head-Free
Fine-Tuning

#### Question.

Does RECAP decodability survive head-free later training?

#### Protocol.

We start from the RECAP Pythia-160M checkpoint ($K{=}64$, balanced presence,
$w{=}0.1$) and the control. Both are fine-tuned with LM loss only on 131M tokens
of FineWeb (Penedo and others 2024), shifted from the Pile training data. Setup:
batch $64\times 1024$, learning rate $10^{-4}$, 2,000 steps, snapshots at
250/1,000/2,000. Ridge probes are refit at every snapshot on disjoint held-out
sequence blocks.

| | RECAP run | control
|

| step | b1 / b2 | ident | b4 | b1 / b2 | ident
|

| 0 | .983 / .992 | .991 | .808 | .768 / .796 | .115
|

| 250 | .827 / .849 | .320 | .805 | .769 / .795 | .151
|

| 1000 | .806 / .823 | .180 | .810 | .771 / .795 | .118
|

| 2000 | .789 / .812 | .169 | .802 | .770 / .793 | .136
|

Table 3: Fresh-probe readability during head-free fine-tuning. b1/b2 = mean AUC
on the two designated rank buckets; ident = 64-class identity accuracy; b4 = the
never-designated bucket in the RECAP run.

#### Findings.

Most of the RECAP surplus is gone by 250 steps and nears control by 2,000, with
a small residual edge. The control and b4 stay flat. Since probes are refit,
this is loss of decodability, not frozen-reader staleness. Keeping the heads
costs $+0.010$ nats (paired 95% CI $[-0.015,+0.034]$, 3 seeds each) at $K{=}64$.

#### Follow-up runs.

- **Same-distribution data.** A head-free fine-tune on unseen Pile data gives
  b1/b2/ident $.790/.811/.151$ at step 2,000, versus $.789/.812/.169$ shifted.
- **Intermittent heads.** Applying the head loss every tenth step during the
  shifted fine-tune gives $.849/.900/.891$ at step 2,000, roughly stable from
  step 250.
- **Original logs.** At $w{=}0.1$, identity loss is near chance after 400 steps.

#### Caveats.

Single seed per run; 2,000 steps; the intermittent run's own tax was not
measured.

## Appendix D The Synthetic Scenarios

Both scenarios are templated grammars over closed vocabularies. One value is
sampled per slot, and a fixed template is filled with random filler sentences,
giving an exact ground-truth tuple per story.

#### Scenario 1 (story).

Slots: name (20), place (12), object (12), activity (8), ending (8). Name,
place, and object occur before the tap; activity and ending occur after it. The
tap sentence is fixed text at a fixed token offset. The name is reused after the
tap (retained), place and object finish before it (faded), and activity and
ending never precede it (unread).

#### Scenario 2 (marketplace).

Slots: customer (20), shop (10), item (14), payment (6), outcome (10). There are
two template variants, and the tap position varies with template-dependent
filler. Customer, shop, and item precede the tap; payment and outcome follow it.
The rest of the pipeline is unchanged.

| | Scenario 1 (story) | Scenario 2 (market)

| slots (pre/post tap) | 3 / 2 | 3 / 2

| vocabulary sizes | 20/12/12/8/8 | 20/10/14/6/10

| templates | 1 | 2

| tap position | fixed | variable

Table 4: The two synthetic scenarios.

Targets are 8-layer, $d{=}256$ transformers trained from scratch on 120,000
stories, with the tap at layer 4 of 8. Scenario-2 targets reach the same
training perplexity (1.304), and every ledger phenomenon transfers (Table 5).

| phenomenon | scenario 1 | scenario 2 | total

| codes, standard | 3/3 | 2/2 | 5/5

| codes, RECAP (5 seeds) | 0/3 | 0/2 | 0/5

| codes, RECAP dose (seed 1) | 0/3 | — | 0/3

| RECAP ceiling 1.0 | 3/3 | 2/2 | 5/5

| propagation | 3/3 | 2/2 | 5/5

Table 5: Cross-scenario emergence ledger. Dose variants are three
partial-designation levels on scenario-1 seed 1 (— = not run).

Figure 3: One held-out story, two target models (seed 1). Each stated specific
is marked true/false against the exact tuple and flip-grounded
($\Delta\tilde{r}$; grounded at $\tau{=}0.05$). The control explanation scores
$\tilde{r}{=}0.94$ (per-example; run-level means 0.71–0.86) while the
reconstruction depends on three false claims (codes). The RECAP explanation is
grounded-and-true on every pre-tap slot, and its false post-tap claim has
${\sim}0$ grounding.

## Appendix E Released-NLA Audit Details

#### Setup.

Public Qwen-2.5-7B layer-20 AV/AR pair; in-distribution web text; greedy
decoding; direction-only reconstruction; pinned open judge with per-edit
filtering (Appendix A). Per-claim deltas are persisted ($n{=}1{,}517$).

| variant | $\tau{=}0.02$ | $\tau{=}0.05$ | $\tau{=}0.10$

| base | 0.042 | 0.021 | 0.016

| RAFT | 0.046 | 0.025 | 0.017

| RL | 0.042 | 0.020 | 0.014

Table 6: Grounded fraction of LLM-salient specific claims by $\tau$ and variant.
Sensitivity-limited lower bounds.

Trained variants fall within the base bootstrap intervals at every threshold.

- **Base 95% intervals.** 3.2–5.2% / 1.5–2.9% / 1.0–2.2% at
  $\tau{=}0.02/0.05/0.10$.
- **Noise floor.** The 95th percentile of $|\Delta\tilde{r}|$ is 0.017, and
  wrong-direction drops ${\geq}0.02$ occur at rate $2{\times}10^{-4}$ under the
  empirical null.
- **Decodability ceiling.** Single-position specifics decode at ${\sim}6\%$
  chance-corrected, unchanged across layers 16–27 and all prefix positions.
  Mean-pooled gist reaches up to 50%. The AR itself does not exceed the ridge
  ceiling. These bounds apply to reconstruction-style readouts; the unembedding
  sees late-layer specifics in next-token form (below).

#### Instrument validity controls.

598 in-distribution passages, positions ${\geq}50$.

- **Floor.** Mismatched reconstructions give a mean centered cosine of $-0.001$.
- **Input echo.** $\tilde{r}{=}0.84$ for the explanation, $0.27$ for the raw
  prefix, and $0.24$ for a paraphrased prefix. Reconstructor specialization to
  explanation-style text accounts for part, not the bulk, of the gap.
- **Transform profile.** See Table 7.

| transform of $z$ | $\tilde{r}$ | fraction kept

| none (original) | 0.84 | 1.00

| paraphrase | 0.75 | 0.89

| synonym substitution | 0.70 | 0.83

| mask function words | 0.72 | 0.85

| word shuffle | 0.32 | 0.37

| mask content words | 0.04 | 0.04

Table 7: Reconstruction retained under transforms of the explanation (released
NLA).

#### Late-layer inventory.

For each false claim with a clean single-token true value ($n{=}316$), we rank
the true value under two readouts: the logit lens (final norm + unembedding) and
a document-folded ridge probe of matched capacity (Table 8).

- The logit lens surfaces the missed value in the top-50 seventeen times more
  often at layer 27 than at the read layer (0.6% to 10.1%).
- The ridge probe stays flat.
- Recovery peaks near 10%.

Hidden states are indexed by output position (embeddings at 0), so the layer-20
tap is index 21. The 16–27 sweep uses the same indexing.

| layer | logit-lens top-50 | ridge-probe top-50

| 15 | 0.000 | 0.054

| 21 (read layer) | 0.006 | 0.047

| 24 | 0.035 | 0.044

| 26 | 0.073 | 0.051

| 27 | 0.101 | 0.060

Table 8: Share of missed true values in the vocabulary top-50 by layer and
readout (released NLA, $n{=}316$ false claims).

#### A control-design lesson.

A one-row-shifted negative control landed above chance at 0.163, because
adjacent rows are often the same document. Shifting by half the dataset restored
exact chance.

## Appendix F Adversarial Monitor: the Lie-Ladder

#### Question.

Can a report-space adversary maximizing $\tilde{r}$ slip a false claim past the
score, and does an activation-probe monitor survive it?

#### Protocol.

RECAP and control Pythia-160M ($K{=}64$ balanced). On held-out taps, each word
stated in an honest generation is edited into an absent top-64 word. We
recompute $\tilde{r}$ through the real reconstructor and score the lie with the
external-truth presence probe, trained on train taps and never on verbalizer
outputs. Rungs:

- **L1:** a random absent word.
- **L2:** the embedding-nearest absent word.
- **L3:** the per-claim $\arg\max$-$\tilde{r}$ absent word.
- **L4** (outside the threat model): a probe-aware choice. Among absent lies
  within $0.05$ of honest $\tilde{r}$, it picks the one the probe most mistakes
  for present. The $0.05$ budget exceeds twice L3's $0.022$ cost.

The design is decoupled: a positive is any truly-present word, and the edit slot
is the stated word. RECAP $n{=}758$, control $n{=}200$; honest $\tilde{r}$
$0.304$ / $0.275$. Edit-based, no RL.

| adversary rung | RECAP AUC / recall | control AUC / recall

| L1 random | 0.993 / 0.99 | 0.729 / 0.06

| L2 embedding-near | 0.971 / 0.93 | 0.635 / 0.06

| L3 $\tilde{r}$ -optimal | 0.952 / 0.86 | 0.508 / 0.01

| L4 probe-aware∗ | 0.909 / 0.70 | 0.242 / 0.00

Table 9: Lie-ladder: probe-monitor AUC and flag recall (at $0.9$ precision). At
L3 the optimal lie lowers RECAP $\Delta\tilde{r}$ on a stated true word by only
$0.022$ (CI $[0.017,0.027]$), versus $0.173$ for a random lie. ∗L4 assumes
monitor leakage and is a worst-case bound outside the threat model.

#### Findings.

(i) At L3, ${\sim}$87% of the random-lie penalty is suppressed (RECAP
$\Delta\tilde{r}$ $0.022$, CI $[0.017,0.027]$).
(ii) The RECAP probe holds at AUC 0.952; the control is at 0.508.
(iii) Selection leakage is small: RECAP goes $0.993\to 0.952$ down the rungs.
(iv) L4 degrades RECAP to $0.909$ (recall $0.70$) and the control to $0.242$.
Sub-chance is expected of a white-box attack on a weak probe.
The separation grows from $0.27$ at L1 to $0.67$ at L4.

#### Caveats.

Report-space adversaries only, with L4 as a bound. Activation-level attacks
(probe-aware training of the target, input perturbation) are out of scope.
$\tilde{r}$ is nearly, not literally, claim-insensitive; the residual is an
aggregate effect with no usable per-instance signal.

#### Anti-circularity: frozen threshold and a held-out band.

$n{=}2800$ audit taps reproduce the headline at doubled $n$.

(a) Frozen threshold. A disjoint-split calibration of the $0.9$-precision
threshold gives RECAP recall $0.878$ (L3) / $0.732$ (L4), matching $0.86$ /
$0.70$. No control threshold reaches $0.9$ precision even at L1, so control
recall is undefined.

(b) Held-out band. For words ranked 64–128, never supervised, the RECAP
advantage vanishes (Table 10).

| adversary rung | RECAP AUC | control AUC | gap

| L1 random | 0.661 | 0.702 | $-0.041$

| L2 embedding-near | 0.633 | 0.662 | $-0.029$

| L3 $\tilde{r}$ -optimal | 0.546 | 0.585 | $-0.039$

| L4 probe-aware∗ | 0.290 | 0.226 | $+0.064$

Table 10: Held-out band (words ranked 64–128). Honest-case AUC $0.66$ – $0.70$
leaves headroom. The Hanley–McNeil SE gives power to detect gaps above
${\sim}0.07$, versus $0.23$ – $0.64$ supervised gaps, so this is a powered
null. The L1 control exceeds RECAP non-significantly ($p{\approx}0.10$). ∗L4 out
of threat model.

The band null is scoped to linear decodability at this tap. The frozen threshold
is a disjoint split of the same run, not a temporally held-out one. The band arm
is unpaired.

## Appendix G An API-Free Grounding Signal, and a Selection Null

Mask-and-rescore replaces the claim's words with neutral filler and rescores
with the frozen reconstructor. It agrees with the flip instrument at AUC 0.997
under the validated judge (Appendix A), using one extra frozen pass and no
judge. An earlier version also reported filter precisions and a grounded base
rate based on pre-validation labels. Roughly a third of those labels were
contaminated by oversized edits, so we withdraw those quantities. All retained
numbers postdate the filter.

#### What inference-time selection does not buy.

Best-of-$N$ by an external truth judge ($N{=}5$, specificity-matched,
independent evaluation judge) moves the false-claim rate from 0.747 to 0.688.

## Appendix H Reader Drift and Realignment

Matched readers decay to near-chance within 1,500 steps (3/3 seeds) and read
nothing across seeds (6/6 pairs). An affine map fitted on ${\sim}$ 7,000 paired
activations, without labels, restores native performance in 9/9 cases.

| map class (three reader pairs: two cross-seed, one stale) | recovery
|---|---
| rank-1, rank-4 | almost nothing
| pure rotation (Procrustes) | about half
| rank 16–64 non-orthogonal | nearly all

Under collapse, freshly retrained probes also fail.

Figure 4: Accuracy recovered by label-free maps of increasing expressiveness,
for two cross-seed pairs and one stale (same-seed, later-checkpoint) case.
Rotation recovers about half; rank 16–64 of non-orthogonal change restores
nearly all.

## Appendix I Ablation Index

| factor | tested
|

outcome

| where

| targets $K$ | 8/64/512
|

tax grows; naive 512 fails

| §5.3

| aux weight $w$ | 1.0–0.03
|

identity free at 0.03

| §5.3

| presence loss | naive/balanced
|

balancing repairs both

| §5.3

| pull strength | $\lambda{=}$ 0–10
|

collapse at all strengths

| §4

| designation dose | 1–3 slots
|

any dose prevents codes

| §5.2

| realignment map | rank 1–full
|

rotation ${\sim}$ half

| App. H

| evaluator | co-trained/indep.
|

code grounding collapses

| §3.2

| scenario | story/market
|

all phenomena transfer

| §5.2

| seeds | 3+2 (sandbox)
|

all claims replicate

| §3–5

| FT persistence | 0–2000 steps
|

erosion within 250

| App. C

| $z$ transforms | 5 types
|

gist carries the score

| App. E

| read layer | 15–27
|

top-50 recovery $\leq$ 0.10

| App. E

| best-of- $N$ | $N{=}5$
|

false rate 0.75 to 0.69

| App. G

| head capacity | linear/MLP
|

decodability unchanged

| below

| tap layer | 6 / 9
|

replicates at 9

| below

| reader size | 0.5B/3B
|

truth unchanged

| §5.4

| RAFT rounds | recon reward
|

score up; RECAP flat, control falls

| §5.4

| reader reward | recon/truth
|

neither raises truth

| §5.4

| injection | 1/6 tokens
|

decode-point info recovered

| §5.4

| negative superv. | on/off
|

precision unchanged

| §5.4

Table 11: Recipe choices varied, values tested, and where reported; the lower
block lists the scale reader-side ablations behind §5.4.

#### Factors not detailed in the main text.

- **Pull strength** ($\lambda{=}0$ – $10$): collapse throughout.
- **Dose** (1–3 slots): codes prevented at any dose.
- **Map class:** Figure 4.
- **Evaluator:** code grounding collapses under the independent one.
- **Scenario:** all phenomena transfer.
- **Head capacity** ($K{=}64$, balanced, $w{=}0.1$): an MLP head gives
  designated AUC 0.98–0.99 at layer 6 (tax $+0.09$).
- **Tap layer:** a linear head at layer 9 gives 0.77–0.79 (control) to
  0.95–0.98 at $+0.01$ nats.
- **Not varied:** scale beyond 160M parameters.

Codes were expected to relocate under partial designation. Instead any dose
prevented them (single scenario and seed), and no mechanism was isolated.
Partial designation suppressed undesignated decodability in the sandbox
(0.21–0.27 to 0.10–0.15); this did not replicate at scale.

#### Aux-weight sweep ( $K{=}64$ ).

Identity stays readable from $w{=}1.0$ down to $w{=}0.03$, with tax within noise
of zero. Naive presence loses readability below $w{=}1.0$. The single-seed tax
is non-monotonic in $w$.

| $w$ | tax (nats) | presence | identity

| 1.0 | $+0.064$ | 0.95–0.97 | 1.000

| 0.3 | $+0.027$ | 0.79–0.81 | 1.000

| 0.1 | $+0.052$ | 0.79–0.81 | 0.997

| 0.03 | $-0.003$ | 0.79–0.81 | 0.953

Table 12: Aux-weight sweep at $K{=}64$ (naive presence), single seed per point.
Tax vs. the shared control on 20M held-out tokens. Presence is mean AUC, with the
range spanning the two designated rank buckets; identity is 64-class accuracy.

§ refers to the main paper; “App.” to this appendix. Single-seed factors are
marked where reported.

## Appendix J Scale Verbalizer Audit Protocol

- **Reader.** 0.5B LM with LoRA (rank 16) and a linear adapter from the
  768-dimensional tap to embedding width. Activations are injected norm-matched
  at marked positions.
- **Training.** Cold-start SFT on templated gold descriptions (trailing-window
  content words and the token eight positions back), then alternating
  reconstructor fitting and two rejection-sampling rounds against
  reconstruction reward. Only the target model differs between conditions.
- **Audit.** 400 held-out taps per condition. A stated word is true iff it
  occurs in the trailing 64-token window. Flips swap in another top-64 word,
  scored under the co-trained reconstructor.
- **Intervals.** Tap-clustered bootstrap 95% (10,000 resamples): 0.385–0.506
  RECAP versus 0.040–0.085 control. The parser emits up to three claims per
  explanation, but at most two shared a tap. Unclustered Wilson intervals are
  0.384–0.506 / 0.042–0.089.

Figure 5: Decodability exceeds verbalizability at scale (Pythia-160M, designated
content). A fresh probe on the reconstructor's decode-point state (black)
recovers designated content across operating points. The verbalizer's
generative precision–recall curve (green, two reconstructor seeds) lies below
it, with its operating point near 0.60 precision. Control verbalizer (grey) at
the floor.

| metric (400 taps each) | control | RECAP

| stated word claims | 406 | 252

| word truth | 0.062 | 0.444

| word grounded @ $\tau{=}0.02$ | 0.48 | 0.78

| word grounded @ $\tau{=}0.05$ | 0.29 | 0.65

| word grounded @ $\tau{=}0.10$ | 0.15 | 0.47

| grounded-minus-true gap | $+0.42$ | $+0.33$

| token-at- $-8$ truth | 0.043 | 0.039

| token-at- $-8$ grounded @ $\tau{=}0.05$ | 0.32 | 0.43

| mean $\tilde{r}$ | 0.19 | 0.24

Table 13: Scale verbalizer audit, one seed per condition. The true offset-8
token is a designated top-64 target in only 2.2% of taps ($n{\approx}9$), and
the verbalizer emits a claim on every tap. Its truth is reported as unrecovered,
not as a code.

#### Robustness to reconstructor strength.

A stronger pair (LoRA rank 32, more SFT, reconstructor, and rejection-sampling
rounds) gives mean $\tilde{r}$ 0.208 (control) / 0.300 (RECAP).

- **Truth:** 0.574 versus 0.152.
- **Recall** on 381 shared present words: 0.470 versus 0.037.
- **Volume:** the control states 92 words versus 312 for RECAP.
- **Offset-8 truth:** 0.028/0.026, against the 2.2% base rate.
- **Seeds:** RECAP truth 0.444, 0.455, 0.443 across three target seeds; three
  control NLA seeds give 0.062, 0.060, 0.000.

#### Decodability exceeds verbalizability (readout redesign).

At the verbalizer's recall, the probe reaches ${\sim}0.80$ precision
(designated AUC 0.99), while the verbalizer reaches 0.44–0.63. The comparison is
AUC versus precision, not literally $0.99$ versus $0.57$.

The rebuilt reader uses standardized injection (not per-sample norm-stripped),
six tokens via an MLP adapter, MLP-LoRA, and a decode-point presence head on
exact window-occurrence labels. It reads designated content at the last-prompt
state at $0.79$ – $0.81$ (two seeds), while generated precision is $0.56$ –
$0.63$; a matched control reads $0.03$. A per-word confidence sweep places the
generative PR curve entirely below the probe's (Figure 5). Reader scale (0.5B
vs. 3B), RAFT reconstruction reward, direct truth reward, injection bandwidth,
and negative supervision leave generated precision unchanged. This is consistent
with a serialization cost for sparse superposed features.

## Appendix K Hyperparameters and Reproducibility Notes

#### Scale runs (Pythia-160M).

- **Data.** pile-uncopyrighted, 500M tokens per run, identical data and order.
- **Optimization.** Batch $64\times 1024$; AdamW ($10^{-4}$, cosine to 10%,
  warmup 100, weight decay 0.01, clip 1.0); bf16 autocast with fp32 weights.
  These were fixed, not swept. Method hyperparameters (aux weight, $K$, dose,
  tap layer, head capacity) are varied as listed in Appendix I.
- **Heads.** Linear on the layer-6 state. Presence uses BCE averaged over $K$;
  the balanced variant weights positives by inverse window base rate, clamped at
  100, estimated on the first 2M tokens. Identity uses CE over the top-$K$
  content vocabulary at offset $-8$.
- **Probes.** Closed-form ridge per rank bucket at $\lambda{=}10^{4}$, fit on
  the first 80% of held-out probe sequences and evaluated on the last 20%
  (contiguous blocks; long documents may straddle). Identity probes are
  64-class logistic. An earlier per-bucket selection of $\lambda$ from
  $\{10,10^{2},10^{3},10^{4}\}$ by eval AUC was rerun with fixed $\lambda$,
  reproducing every bucket AUC to three decimals.
- **Evaluation.** Held-out LM loss on 20M tokens. Significance via 95%
  bootstrap CIs and disjoint-interval comparisons, plus Hanley–McNeil power for
  the band null; no fixed-level tests.
- **Compute.** Single A100-80GB GPUs under Modal (Debian-slim, Python 3.12;
  PyTorch, Transformers, Accelerate, NumPy, pinned versions).
- **Precision caveat.** fp16 checkpoints load natively under recent defaults,
  and AdamW in fp16 destroys all weights in one step via epsilon underflow
  while loss and gradients read finite. All runs force fp32 parameters.

#### Sandbox runs.

- **Targets.** Two closed-vocabulary scenarios; 8-layer, $d{=}256$ targets
  trained from scratch.
- **NLA.** 0.5B AV/AR with a dimension adapter and norm-matched injection.
  Cold-start SFT on full-story descriptions, then alternating reconstructor
  fitting and rejection sampling against reconstruction reward.
- **Audits.** Exact closed-vocabulary scan; flips swap a same-type word.
- **RECAP.** Aux weight 0.3, 3,000 steps, batch 128, learning rate
  $3{\times}10^{-4}$, 30,000 probe taps.
- **Seeds.** Three (scenario 1) plus two (scenario 2), each a single integer
  passed to torch.manual_seed with a matched CUDA generator.
- **Ceilings.** Evaluated by independent probes on tuple-disjoint splits
  (scenario 1) and held-out splits (scenario 2).