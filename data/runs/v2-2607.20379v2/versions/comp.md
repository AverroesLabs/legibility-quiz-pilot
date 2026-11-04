# Train the Model, Not the Reader: Decodability Supervision for Verifiable Activation Explanations

Hiskias Dingeto (StackOne Technologies, hiskias@stackone.com)

###### Abstract

NLAs score activation explanations by reconstruction, a test structurally insensitive to individual false claims. On a released Qwen-2.5-7B verbalizer, explanations reconstruct well above chance while ~2% of specific claims are grounded, so the score tracks gist. Under exact synthetic ground truth, standard training develops co-adapted private codes, and fixes that leave the target model unchanged fail. We contribute two audits (grounding-vs-truth; independent-evaluator swap) and RECAP (Readable Encodings via Co-trained Auxiliary Predictors): linear heads co-trained with the target model to keep designated content decodable. In the sandbox, fresh verbalizers state it truly and codes vanish (+0.001 nats). On Pythia-160M it becomes probe-decodable (verbalizer truth 0.44–0.46 vs near-zero control). A probe ranks true over false verbalizer claims at AUC 0.96 (0.82 without RECAP). Under a score-maximizing lying adversary (~87% of lie penalty suppressed) it holds AUC 0.95; the control probe falls to 0.51.

## 1 Introduction

In an NLA, a verbalizer (AV) maps an activation to text and a reconstructor (AR) maps it back. Round-trip quality is treated as each explanation's faithfulness test: unsupervised, label-free, self-checking. Released AV/AR pairs exist for several open models. The test has a structural gap, a property of the objective at any model or scale. It checks only that the explanation suffices to regenerate the activation and is indifferent among claim values inducing the same reconstruction, so false additions go unpenalized.

The test is passed in two non-faithful ways:
- **Gist:** on a released Qwen-2.5-7B NLA, a per-claim flip audit finds grounding in only ~2% of salient specific claims. This is a lower bound, since flips have low sensitivity.
- **Codes:** in a sandbox with exact ground truth, co-trained pairs consistently develop private codes, i.e. false wording the reconstruction depends on (grounded–true gaps 0.30–0.80, two scenarios).

Two natural fixes fail. Minimizing hidden-state–reconstruction distance removes probe-readable content. Rewarding grounding raises the score by changing which sentences the metric counts. A frozen probe goes stale under drift, which label-free realignment fully reverses. What survives our controls trains the target model: RECAP.

Figure 1: (a) NLA: only AV and AR train; the score can stay high while the text asserts an ungrounded specific. (b) RECAP trains the target with co-trained linear heads on external targets. After training, an independent fresh probe (not the heads) recovers the content (AUC 0.95) and a fresh verbalizer states it truly.

Contributions:
- **Claim-level audit** with per-edit validity controls. The released verbalizer passes with largely ungrounded specifics; its score survives paraphrase and collapses under content-word masking.
- **Private codes**, detected and quantified by grounding-vs-truth and the evaluator swap.
- **RECAP** restores claim-level faithfulness in the sandbox and transfers to Pythia-160M at a measured cost.
- **Adversarial monitor:** a score-gaming adversary suppresses ~87% of the lie penalty on both models, yet the RECAP probe flags lies (AUC 0.95) while the control falls to chance (0.51). The reconstruction score must not be used as a truth signal.

## 2 Preliminaries and Experimental Setup

### 2.1 The Reconstruction Test and Audit Protocols

AV generates explanation $z$ of activation $h$; AR maps $z$ to $\hat h$. With target weights $\theta$ frozen and $h$ the tap activation:

| | $\mathcal{J}_{\mathrm{read}}=\max_{\mathrm{AV},\mathrm{AR}}\cos(\mathrm{AR}(\mathrm{AV}(h)),h)$ | (1)

The NLA authors warn nothing forces faithfulness, yet this is the reported score and users' main criterion. floor = mean cosine of reconstructions vs mismatched activations; $\tilde r=(\cos(\hat h,h)-\mathrm{floor})/(1-\mathrm{floor})$, which is 0 at chance and 1 when perfect. Released NLA: Qwen-2.5-7B layer-20 AV/AR.

**Grounding.** Let $z_{\neg c}$ be $z$ with claim $c$ alone minimally flipped (a flip).

| | $\Delta\tilde r(c)=\tilde r(z)-\tilde r(z_{\neg c})$ | (2)

$c$ is grounded iff $\Delta\tilde r(c)\ge\tau$. Thesis: reconstruction sensitivity to a flip ≠ the model's sensitivity, and the two diverge in practice. Edits changing more than their target inflate grounding severalfold (App. A). Grounded × true gives four cases; we focus on grounded-but-false. If that wording is pair-specific (shown by the evaluator swap, §3.2), it is a co-adapted private code. The label is operational, with no intentional stance. Evaluator gap $=\tilde r_{\mathrm{co}}-\tilde r_{\mathrm{ind}}$.

**Terms.**
- Tap: the layer/position read.
- Probe: a freshly fitted linear decoder.
- Gist: overall meaning, vs specific facts.
- Designated: an auxiliary predictor is co-trained to predict it.
- Decodable: a fresh linear probe reads it.
- Faithfulness: always claim-level (true of and grounded in the activation); not downstream computation, not world grounding.

#### Settings.
1. Released Qwen-2.5-7B layer-20 NLA on in-distribution web text ($n{=}1{,}517$ audited claims).
2. Sandbox: two templated synthetic scenarios (story, marketplace) with exact, judge-free truth. Tap position fixes what the activation holds (retained, faded, unread slots).
3. Continued pretraining of Pythia-160M with self-supervised heads. Tax = held-out LM loss minus a shared control.

Details: App. D, K.

## 3 Auditing Reconstruction-Scored Explanations

The mechanisms differ. In the released NLA, specifics are largely absent from the tap, so the score depends on gist. In the sandbox, content is present but stated in wording only the co-trained AR understands, which the evaluator swap exposes. Only the sandbox failure is a pure metric artifact.

### 3.1 Counterfactual Audit of a Released Verbalizer

Explanations reach $\tilde r=0.84$, yet ~2% of LLM-salient specific claims are grounded (4.2/2.1/1.6% at $\tau$=0.02/0.05/0.10; App. E).
- **Lower bound:** an append-and-re-rank control recovers a topic known to be encoded at only 28% raw.
- **Gist vs specifics (more robust):** probes recover specifics ~3× less accurately than gist (~6% vs ~18% chance-corrected per position; mean-pooled gist ~50%).
- **Layers and readouts:** holds across layers 16–27 and is specific to reconstruction-style readouts; the model's output head does see late-layer specifics.

Training the verbalizer toward grounding fails. Rejection sampling and GRPO with a clause-level grounding reward improved their proxy, but the independent claim-level audit stayed null at every $\tau$ (CIs straddle zero). Text statistics attribute the gains to sentence restructuring (merging/packaging units), not grounding.

### 3.2 Audit Under Synthetic Ground Truth

The toy NLA asserts every slot confidently at $\tilde r$ 0.71–0.86. Truth tracks decodability only loosely:
- fully decodable slot: true 0.44–0.96 across seeds, grounded 0.76–0.88 in all;
- partially decodable slots: true 0.15–0.26;
- unread slots: true at chance.

In 5/5 standard runs, private codes appear (grounded–true gaps 0.30–0.80). Codes can occupy several slots. In the marketplace one occupies an unread slot, so the sentence acts as a channel rather than a description.

Evaluator swap: under an independent evaluator never exposed to the pair's outputs, grounding on code slots collapses 0.53–0.89 → 0.08–0.23 while true-claim grounding survives. This is co-adaptation, not a weaker evaluator. Cross-evaluator $\tilde r$ gap: 0.17–0.38 on standard runs vs ~0.003 on RECAP runs. An evaluator trained on a disjoint paraphrase family (different frames and slot order, sharing only slot words) reproduces the pattern within ~0.05 over three seeds (RECAP slots at 1.00). The channel is private to the pair.

Readout gap: in the marketplace, slots 55–72% decodable are stated truly only 15–26% of the time. Content is present but not conveyed, a gap beyond the storage gap; RECAP closes it.

## 4 Representation Collapse and Probe Drift

**Collapse.** Pulling activations toward their own reconstruction removes decodable slot content at every tested strength and initialization, with perplexity unchanged. Retrained probes also fail, so the information is gone, not rotated. Reconstruction is a lossy projection, and raising similarity to it removes variation outside the probe subspace (cf. non-contrastive SSL collapse).

**Drift.** Matched probes decay to near-chance within 1,500 continued steps and are at chance across seeds (6/6 pairs). A label-free affine map on paired activations restores native performance in 9/9 cases. Rotation alone recovers ~half; full recovery needs non-orthogonal rank ~16–64 (App. H). Concurrent work reports the same at 2B/7B.

Monitoring a model still in training needs explicit recalibration. Realignment fixes how a probe reads; RECAP changes what is stored. Corollary: adapter-based cross-model probe transfer should work.

## 5 The RECAP Objective

| configuration | tax (nats) | probe AUC
|---|---|---|
| control | n/a | 0.77–0.80
| $K{=}8$, naive, $w{=}1.0$ | $+0.047$ | 1.000
| $K{=}64$, naive, $w{=}1.0$ | $+0.064$ | 0.95–0.97
| $K{=}64$, balanced, $w{=}0.1$ | $+0.010$ | 0.97–0.99
| $K{=}512$, balanced, $w{=}1.0$ | $+0.14$–$0.20$ | 0.97–0.99

Table 1: Held-out tax and fresh-probe AUC on designated targets. $K{=}64$ balanced tax is a 3-seed paired difference (95% CI $[-0.015,+0.034]$); the other balanced row is a 3-seed range; the rest are single-seed.

### 5.1 Method

Linear heads read the tap hidden state and predict external targets (sandbox slot values; self-supervised text functions at scale):

| | $\mathcal{J}_{\mathrm{model}}=\min_{\theta,\phi}\mathcal{L}_{\mathrm{LM}}(\theta)+w\,\mathcal{L}_\phi(h(\theta),y)$ | (3)

$\mathcal{L}_\phi$ = class-balanced BCE on each content word's presence + CE on token identity; balancing prevents the §5.3 shortcut. Eq. 1 never touches $\theta$; Eq. 3 trains it. RECAP addresses each failure at its source:
- no frozen probe, so no drift;
- external targets rather than self-reconstruction, so no collapse;
- heads read activations directly, so no text for codes to hide in.

All decodability numbers come from independently fitted probes, never the heads. RECAP is decodability supervision in the deep-supervision lineage.

### 5.2 Sandbox Results

Designated content is 100% decodable by fresh probes at +0.001 nats in 5/5 runs; heads decode nothing on unread slots. Fresh NLAs on RECAP models state designated content at ~1.00 truth and grounding under both evaluators, and give remaining false content grounding ~0. No codes appear in any RECAP run (App. D). NLA training is identical in both conditions (including cold-start SFT on gold descriptions), isolating target-model decodability.

Figure 2: On a held-out Pile passage, fresh probes on the layer-6 state recover trailing-window content words and the offset-8 token on RECAP; on the control the readout is scrambled (same recipe and data).

### 5.3 Scaling to Pythia-160M

- **Decodability:** fresh probes reach 0.95–1.00 AUC for $K$=8, 64, vs 0.77–0.80 control.
- **Shortcut:** at $K{=}512$ the naive presence head minimized training loss by always predicting "absent", leaving nothing decodable to a fresh probe (App. I). Class balancing restores 0.97–0.99 across all 512 targets. Design rule: use targets whose loss cannot be minimized without genuine discrimination.
- **Cost:** +0.010 nats at $K{=}64$ (indistinguishable from zero); +0.14–0.20 for 512 targets into $d{=}768$. An identity target stays decodable at $w{=}0.03$ with tax within noise of zero.
- **Controls:** the never-designated bucket stayed at control level in all eight supervised runs.
- **Robustness:** an MLP head gives 0.98–0.99 at layer 6; a linear head at layer 9 lifts AUC 0.77–0.79 → 0.95–0.98 at +0.01 nats.

### 5.4 Verbalizer Audit at Scale

An AV/AR pair trained on Pythia taps with the released recipe states context words truly at 0.44 (RECAP) vs 0.06 (control). The control shows grounded-but-false at scale (grounded 0.48, true 0.06). RECAP narrows the gap from +0.42 to +0.33 by more truthful statement, not code removal. RECAP's recall of present words is 0.29 vs 0.07 while it states fewer words. Truth is 0.44–0.46 across three seeds vs a near-zero control and is robust to reconstructor strength (App. J).

The shortfall lies in generation, not storage. At the verbalizer's recall, a fresh probe has precision ~0.80 vs the verbalizer's 0.44–0.63. A redesigned reconstructor (standardized multi-token injection, MLP capacity, decode-point presence head on external window-occurrence labels) makes designated content linearly decodable at its decode point (0.79–0.81, two seeds; matched control 0.03), yet generated precision stays 0.56–0.63. None of these closes the gap: reconstructor scale (3B matches 0.5B), reconstruction optimization, truth reward, injection bandwidth, negative supervision. RECAP guarantees decodability; the verbalizer renders it lossily, so the monitor checks claims against internals.

Without heads, decodability erodes within a few hundred fine-tuning steps. Maintenance is continuous (+0.010 nats) or intermittent at ~1/10 compute; it is better trained in from the start (App. C).

## 6 Probe-Based Monitoring

**Sandbox.** Fresh per-slot probes read the tap. A claim contradicted by a confident probe ($p{>}0.9$) is flagged and scored by exact truth (App. B).
- Standard model: sound (every flag is a real false claim) but low recall, catching 27 of 733 false pre-tap claims since most slots are too faded.
- RECAP: no false pre-tap claims and zero abstention. Coverage of designated pre-tap content rises 37%→100%; this is a coverage gain, not precision, and not the whole activation.
- Post-tap claims are correctly abstained on both models; grounding-vs-truth handles them (one scenario, seed, threshold).

**Pythia-160M.** A fresh presence probe ranks true over false claims at AUC 0.965 vs 0.823 control (disjoint 95% CIs). It flags false claims at 0.93 recall, 0.87 precision. At the 1.5% base rate sandbox coverage doesn't transfer, so AUC/PR are reported. The verbalizer proposes; the probe verifies.

#### Surviving a score-gaming adversary.
The adversary edits a true stated word to the absent word maximizing the recomputed score (App. F). This suppresses ~87% of the lie penalty on both models, yet the RECAP probe holds AUC 0.952 while the control falls to 0.508. Reading a signal that predates the report is necessary but not sufficient: content must be decodable.
- **Candidate selection:** costs little (0.993→0.952 as lies grow semantically nearer).
- **Threat model:** text edits, not activation attacks. Immunity is structural for score manipulation, empirical for selection.
- **Probe-aware worst case:** an adversary assuming monitor leakage (outside the threat model) reaches RECAP 0.909, control 0.242; separation grows with adversary strength.
- **Threshold:** recalibrating on a disjoint split leaves recall unchanged; no control threshold reaches 0.9 precision even vs random lies.
- **Scope:** on undesignated words (ranks 64–128), RECAP has no advantage (0.66 vs 0.70).

## 7 Related Work

#### Activation verbalization and its critiques.
Reader improvements include supervised QA decoders, general explainers, interpreter assistants, training-free readouts and cross-model adapters; SAEs are the unsupervised family. Explanations pass plausibility without faithfulness in CoT, and causal tests of neuron explanations found high error and little efficacy. Counterfactual faithfulness tests prefigure our flips. Prior critiques are benchmark-level (many tasks solvable without internals). We audit the reconstruction test per claim on a released NLA and repair by training the target model.

#### Scalable oversight.
Cheap automatic signals substitute for human judgment, and AI control designs protocols safe under subversion. We add a monitor whose signal is an independent probe of the target and survives score gaming.

#### Steganography in learned channels.
Information hides in reconstruction objectives (CycleGAN) and in CoT under optimization pressure, generalizing under process supervision. Misspecified incentives yield collusion; tool use widens channels; probing detects collusion. Optimization obfuscates activations against latent monitors, and training against CoT monitors yields obfuscated reward hacking. Those channels are induced by incentives, process supervision or adversaries; our codes arise from the standard reconstruction recipe with no steganography objective. Complementing "don't optimize on the monitor", supervising against external truth makes content decodable without pressure to fool the probe or reconstructor.

#### Representation drift, stitching, and monitor staleness.
Representations are equivalent up to transformation, and affine maps transfer features across LMs. Pretraining dynamics are stable across seeds. Tuned lenses are the affine readers whose staleness we characterize; concurrent work shows label-free repair at 2B/7B, and retraining restores detectability after probe-aware fine-tuning. We add recovery anatomy, cross-seed failure, exact evidence that information outlives its readers, and the drift/collapse separation.

#### Deep supervision and intrinsic interpretability.
Deep supervision targets task labels; ours targets external content. Concept bottlenecks route flow through concepts, concept whitening constrains geometry, codebooks discretize, and gradient routing localizes; we add heads without constraining flow. Explanation-side supervision can track behavior change; ours acts on the target model.

## 8 Discussion and Limitations

Reconstruction scoring is scalable oversight, and it is fragile: passable without faithfulness and gameable. RECAP moves oversight from trusting prose to checking an independent probe that survives explanation editing. It covers only designated, decodable content, so it is a component of oversight, not a solution.

Evidence comes from a sandbox, one released NLA and one small model, mostly at single seeds; frontier scale is untested. Limitations:
1. Only closed-vocabulary, template-parsed claims are audited and the monitor covers designated content; open-ended faithfulness is unmeasured.
2. RECAP must be co-trained, not retrofitted to a frozen model, and needs a good target; a poor one can report success while decoding nothing.
3. Decodability ≠ verbalizability.
4. RECAP certifies storage, not use. Content can be behaviorally inert (tap patching moves behavior ~0.003), and a model could route around a monitored subspace.

Success means claim-verified explanations, calibrated abstention where content is absent, and monitors with recalibration protocols, not higher $\tilde r$.

## 9 Conclusion

Models pass the reconstruction test via gist and private codes; training against it yields collapse or proxy gaming. RECAP trains the target model, yielding claim-level-faithful sandbox verbalizers and probe-decodable designated content on Pythia-160M, where probes read content more reliably than verbalizers generate it. A model can game a learned test of its own explanations, so training decodability in beats post-hoc auditing.

## Appendix A Edit Validity for Minimal-Pair Flips

A valid edit flips exactly one claim and nothing else. The closed-model judge was valid 68.4% of the time; the open judge used throughout (Gemma-4-12B, pinned, temp 0) 99.3%. Invalid edits were mostly oversized and inflated grounding ~7×. All grounded fractions use the open judge with per-edit filtering.

## Appendix B Safety Experiment 1: The Discrepancy Detector

Scenario 1, seed 1, 400 held-out taps per model. Per slot type, a fresh multinomial logistic probe (linear, 256-d) is compared with the first stated word. A claim is flagged if a probe with confidence >0.9 disagrees and abstained if confidence is below threshold.

| model | phase | claims | false | flagged | prec. | abstain
|---|---|---|---|---|---|---|
| control | pre-tap | 1200 | 733 | 27 | 1.000 | 0.632
| control | post-tap | 800 | 698 | 0 | — | 1.000
| RECAP | pre-tap | 1200 | 0 | 0 | — | 0.000
| RECAP | post-tap | 800 | 706 | 0 | — | 1.000

Table 2: Precision = flags that are false claims.

Findings:
- No false alarms.
- The bottleneck is decodability: 63% of control pre-tap claims are too faded, giving 3.7% recall on 733 lies.
- RECAP removes pre-tap lies at the source.

Caveats: one scenario, seed, threshold; "decodable" means linearly readable.

## Appendix C Safety Experiment 2: Legibility Persistence Under Head-Free Fine-Tuning

RECAP ($K{=}64$, balanced, $w{=}0.1$) and control Pythia-160M are fine-tuned LM-only on 131M tokens of shifted FineWeb (vs Pile): batch 64×1024, lr $10^{-4}$, 2,000 steps. Ridge probes are refit at each snapshot on disjoint held-out blocks.

| step | RECAP b1/b2 | ident | b4 | ctrl b1/b2 | ctrl ident
|---|---|---|---|---|---|
| 0 | .983/.992 | .991 | .808 | .768/.796 | .115
| 250 | .827/.849 | .320 | .805 | .769/.795 | .151
| 1000 | .806/.823 | .180 | .810 | .771/.795 | .118
| 2000 | .789/.812 | .169 | .802 | .770/.793 | .136

Table 3: b1/b2 = AUC on designated buckets; ident = 64-class accuracy; b4 = never-designated bucket.

Findings:
- A small residual edge remains at 2,000 steps.
- Because probes are refit, this is decodability loss, stronger than drift staleness.
- Unseen same-distribution Pile erodes identically (.790/.811/.151), so erosion is intrinsic to removing heads.
- Applying the head loss every 10th step retains .849/.900/.891 (stable from 250).
- Readability establishes slowly at $w{=}0.1$ (identity near chance after 400 steps), so maintenance beats restoration.

Caveats: single seed; 2,000 steps; intermittent tax unmeasured.

## Appendix D The Synthetic Scenarios

Templated grammars over closed vocabularies with one value per slot, a fixed template and random filler, giving exact tuple truth.
- **Story:** name, place, object precede the tap; activity, ending follow. The tap sentence is fixed. Name is reused after the tap (retained); place/object are faded; activity/ending are unread (impossibility control).
- **Marketplace:** customer, shop, item precede; payment, outcome follow.

| | Story | Market
|---|---|---|
| slots pre/post | 3/2 | 3/2
| vocab sizes | 20/12/12/8/8 | 20/10/14/6/10
| templates | 1 | 2
| tap position | fixed | variable

Table 4.

Targets are 8-layer, $d{=}256$ transformers trained on 120,000 stories, with the tap at layer 4; both scenarios reach perplexity 1.304.

| phenomenon | sc.1 | sc.2 | total
|---|---|---|---|
| codes, standard | 3/3 | 2/2 | 5/5
| codes, RECAP | 0/3 | 0/2 | 0/5
| codes, RECAP dose (seed 1) | 0/3 | — | 0/3
| RECAP ceiling 1.0 | 3/3 | 2/2 | 5/5
| propagation | 3/3 | 2/2 | 5/5

Table 5: Dose = three partial-designation levels.

Figure 3: On one seed-1 story, the control passes at $\tilde r{=}0.94$ while depending on three false claims; the RECAP explanation is grounded-and-true on all pre-tap slots, and its false post-tap claim has ~0 grounding ($\tau{=}0.05$).

## Appendix E Released-NLA Audit Details

Greedy decoding, direction-only reconstruction, validated edits; $n{=}1{,}517$ base claims.

| variant | $\tau$=0.02 | 0.05 | 0.10
|---|---|---|---|
| base | 0.042 | 0.021 | 0.016
| RAFT | 0.046 | 0.025 | 0.017
| RL | 0.042 | 0.020 | 0.014

Table 6: Grounded fraction (lower bounds).

- **Stability:** trained variants lie within base bootstrap CIs (3.2–5.2/1.5–2.9/1.0–2.2%).
- **Noise floor:** 95th pct $|\Delta\tilde r|$ = 0.017; wrong-direction drops ≥0.02 at rate $2{\times}10^{-4}$.
- **Ceiling:** the ~6% specifics figure is unchanged across prefix positions; the AR itself does not exceed the ridge ceiling. The ceiling bounds reconstruction-style readouts only.

Instrument controls (598 passages, positions ≥50):
- **Floor:** mismatched pairs give a centered cosine of −0.001.
- **Input echo:** $\tilde r$ 0.84 vs 0.27 for the raw prefix, 0.24 for a paraphrased prefix (reconstructor specialization explains part, not the bulk).
- **Transforms:** a surface-form code would show the opposite of Table 7.

| transform | $\tilde r$ | kept
|---|---|---|
| none | 0.84 | 1.00
| paraphrase | 0.75 | 0.89
| synonym sub. | 0.70 | 0.83
| mask function words | 0.72 | 0.85
| word shuffle | 0.32 | 0.37
| mask content words | 0.04 | 0.04

Table 7.

**Late-layer inventory.** False claims with a single-token true value ($n{=}316$) are ranked by the logit lens and by a document-folded ridge probe of matched capacity. Logit-lens top-50 rises 0.6%→10.1% (×17) by layer 27 while ridge stays flat. Recovery peaks near 10%, so most missed specifics are unrecoverable by these readouts. Indexing is by output position (embeddings = 0), so the layer-20 tap is index 21.

| layer | logit-lens | ridge
|---|---|---|
| 15 | 0.000 | 0.054
| 21 (read) | 0.006 | 0.047
| 24 | 0.035 | 0.044
| 26 | 0.073 | 0.051
| 27 | 0.101 | 0.060

Table 8.

**Control lesson.** A negative control shifted by one row scored 0.163 because adjacent rows share documents; shifting by half the dataset restored chance.

## Appendix F Adversarial Monitor: the Lie-Ladder

**Protocol.** RECAP and control Pythia-160M ($K{=}64$ balanced). Each honest stated word is edited into an absent top-64 word. $\tilde r$ is recomputed via the real AR, and the lie is scored by an external-truth presence probe (trained on train taps, never on verbalizer outputs).

Rungs:
- **L1:** random absent word.
- **L2:** embedding-nearest absent word.
- **L3:** per-claim argmax-$\tilde r$ over all candidates.
- **L4** (outside threat model): probe-aware, assumes the monitor leaks. Among lies keeping $\tilde r$ within 0.05 of honest, it picks what the probe most mistakes for present. The 0.05 budget is over twice L3's 0.022 cost, a deliberately generous bound.

Decoupled design (positive = any truly present word; edit slot = the stated word): RECAP $n{=}758$, control $n{=}200$; honest $\tilde r$ 0.304/0.275. Edit-based, no RL.

| rung | RECAP AUC/recall | control AUC/recall
|---|---|---|
| L1 random | 0.993/0.99 | 0.729/0.06
| L2 embedding-near | 0.971/0.93 | 0.635/0.06
| L3 $\tilde r$-optimal | 0.952/0.86 | 0.508/0.01
| L4 probe-aware∗ | 0.909/0.70 | 0.242/0.00

Table 9: Recall at 0.9 precision. ∗Outside threat model.

Findings:
- At L3, the optimal lie lowers RECAP $\Delta\tilde r$ by 0.022 (CI [0.017,0.027]) vs 0.173 for a random lie.
- The sub-chance L4 control is the expected signature of selecting words the weak probe over-scores.
- RECAP–control separation is 0.27 at L1 and 0.67 at L4.
- $\tilde r$ is nearly, not literally, claim-insensitive: the residual is an aggregate effect with no usable per-instance signal.
- Activation-level attacks (probe-aware target training, input perturbation) are out of scope.

**Anti-circularity** ($n{=}2800$ taps, reproducing the headline):
- (a) Frozen threshold on a disjoint split: RECAP recall 0.878 (L3) / 0.732 (L4) vs eval-tuned 0.86/0.70; control recall is undefined.
- (b) Held-out band (ranks 64–128, never supervised): see Table 10.

| rung | RECAP | control | gap
|---|---|---|---|
| L1 | 0.661 | 0.702 | −0.041
| L2 | 0.633 | 0.662 | −0.029
| L3 | 0.546 | 0.585 | −0.039
| L4∗ | 0.290 | 0.226 | +0.064

Table 10: Honest AUC 0.66–0.70 leaves headroom. Hanley–McNeil power detects gaps >~0.07, vs 0.23–0.64 supervised gaps, so this is a powered null; the L1 control edge is not significant ($p{\approx}0.10$).

Caveats: the band null covers linear decodability at this tap; the frozen threshold is a same-run split, not temporally held out; the band arm is unpaired. RECAP gives no protection outside the designated vocabulary.

## Appendix G An API-Free Grounding Signal, and a Selection Null

Mask-and-rescore: mask the claim's words with a neutral filler and rescore with the frozen AR. It agrees with flips at AUC 0.997 under the validated judge: one extra pass, no judge. Earlier filter precisions and a grounded base rate rested on pre-validation labels (~1/3 contaminated by oversized edits) and are withdrawn; all retained numbers postdate the filter.

#### What inference-time selection does not buy.
Best-of-5 by an external truth judge (specificity matched, independent evaluation judge) moves the false-claim rate only 0.747→0.688; faithful explanations are rare in the sample neighborhood.

## Appendix H Reader Drift and Realignment

Drift: 3/3 seeds decay; the affine map uses ~7,000 paired activations. By map class (two cross-seed pairs, one stale same-seed checkpoint):
- rank-1/4: almost nothing;
- Procrustes: ~half;
- rank 16–64 non-orthogonal: nearly all.

Under collapse, fresh probes also fail.

Figure 4: Recovery by map expressiveness.

## Appendix I Ablation Index

| factor | tested | outcome | where
|---|---|---|---|
| $K$ | 8/64/512 | tax grows; naive 512 fails | §5.3
| $w$ | 1.0–0.03 | identity free at 0.03 | §5.3
| presence loss | naive/bal. | balancing repairs | §5.3
| pull $\lambda$ | 0–10 | collapse at all | §4
| dose | 1–3 slots | prevents codes | §5.2
| realign map | rank 1–full | rotation ~half | H
| evaluator | co/indep. | code grounding collapses | §3.2
| scenario | story/market | all transfer | §5.2
| seeds | 3+2 | all replicate | §3–5
| FT persistence | 0–2000 | erosion by 250 | C
| $z$ transforms | 5 | gist carries score | E
| read layer | 15–27 | top-50 ≤0.10 | E
| best-of-N | 5 | false 0.75→0.69 | G
| head | linear/MLP | unchanged | §5.3
| tap layer | 6/9 | replicates at 9 | §5.3
| reader size | 0.5B/3B | truth unchanged | §5.4
| RAFT | recon reward | score up; RECAP flat, control falls | §5.4
| reader reward | recon/truth | no truth gain | §5.4
| injection | 1/6 tokens | decode-point info recovered | §5.4
| neg. superv. | on/off | precision same | §5.4

Table 11.

- MLP head tax: +0.09.
- Not varied: scale >160M.
- Partial designation (one scenario/seed): codes were expected to relocate but were prevented (no mechanism isolated). It suppressed undesignated decodability 0.21–0.27→0.10–0.15, which did not replicate at scale.

| $w$ | tax | presence | identity
|---|---|---|---|
| 1.0 | +0.064 | 0.95–0.97 | 1.000
| 0.3 | +0.027 | 0.79–0.81 | 1.000
| 0.1 | +0.052 | 0.79–0.81 | 0.997
| 0.03 | −0.003 | 0.79–0.81 | 0.953

Table 12: $K{=}64$, naive presence, single seed; tax on 20M tokens (non-monotonic); presence range spans the two buckets.

## Appendix J Scale Verbalizer Audit Protocol

**Reader.** 0.5B LM with LoRA r16 and a linear adapter from the 768-d tap; norm-matched injection.

**Training.** Cold-start SFT on template gold descriptions (window words, token 8 back), then alternating AR fitting and two rejection-sampling rounds on reconstruction reward. Identical across conditions.

**Audit.** 400 held-out probe-region taps per condition. A word is true iff it occurs in the trailing 64-token window. Flips swap in another top-64 word under the co-trained AR. Tap-clustered bootstrap CIs (10,000 resamples): 0.385–0.506 RECAP vs 0.040–0.085 control. Wilson CIs are 0.384–0.506 / 0.042–0.089; at most 2 claims shared a tap.

| metric | control | RECAP
|---|---|---|
| word claims | 406 | 252
| word truth | 0.062 | 0.444
| grounded τ=.02 | 0.48 | 0.78
| grounded τ=.05 | 0.29 | 0.65
| grounded τ=.10 | 0.15 | 0.47
| grounded−true | +0.42 | +0.33
| token−8 truth | 0.043 | 0.039
| token−8 grounded τ=.05 | 0.32 | 0.43
| mean $\tilde r$ | 0.19 | 0.24

Table 13: One seed each. The offset-8 token is a designated target in only 2.2% of taps ($n{\approx}9$); its near-zero truth is a base-rate artifact, reported as unrecovered, not as a code.

Figure 5: The fresh probe on the decode-point state lies above the verbalizer's PR curve (two AR seeds, operating point ~0.60 precision); the control is at the floor.

**Stronger reconstructor** (LoRA rank 32, more SFT/AR/RS rounds):
- mean $\tilde r$ 0.208 control / 0.300 RECAP;
- truth 0.574 vs 0.152;
- on the shared 381 present words, recall 0.470 vs 0.037;
- the control states fewer words (92 vs 312) yet trails on both, so its higher truth is selectivity;
- offset-8 truth 0.028/0.026.

**Seeds.** RECAP 0.444/0.455/0.443; control NLA seeds 0.062/0.060/0.000.

**Readout redesign.** Designated-content probe AUC is 0.99. The fair comparison is AUC vs precision mismatch, not 0.99 vs 0.57. Redesign details: standardized (not per-sample norm-stripped) injection, six tokens via an MLP adapter, MLP-LoRA, and a decode-point head on exact window labels (never a learned monitor). A per-word confidence sweep places the generative PR curve entirely below the probe's. The residual is consistent with the cost of serializing sparse superposed features by generation, not missing information.

## Appendix K Hyperparameters and Reproducibility Notes

#### Scale runs (Pythia-160M).
- **Data:** pile-uncopyrighted, 500M tokens/run, identical data and order.
- **Optimization:** batch 64×1024; AdamW $10^{-4}$, cosine to 10%, warmup 100, wd 0.01, clip 1.0; bf16 autocast, fp32 weights.
- **Heads:** linear on layer 6. Presence: BCE averaged over $K$; balanced positives weighted by inverse window base rate (clamped 100, estimated on first 2M tokens). Identity: CE over top-$K$ vocabulary at offset −8.
- **Probes:** ridge, $\lambda{=}10^4$ fixed, fit on first 80% / evaluated on last 20% of held-out sequences (contiguous blocks; long documents may straddle); identity probes 64-class logistic. Earlier per-bucket $\lambda$ selection from $\{10,…,10^4\}$ by eval AUC reproduces to three decimals with fixed $\lambda$.
- **Sweeps:** optimizer hyperparameters not swept.
- **Evaluation:** held-out LM loss on 20M tokens; 95% bootstrap CIs and disjoint-interval tests.
- **Compute:** single A100-80GB on Modal, Python 3.12, pinned versions.
- **Caution:** fp16-loaded checkpoints with AdamW in fp16 destroy all weights in one step via epsilon underflow while loss reads finite; all runs force fp32.

#### Sandbox runs.
- **NLA:** 0.5B AV/AR, dimension adapter, norm-matched injection, cold-start SFT on full-story descriptions, alternating AR fitting and RS rounds.
- **Audit:** exact vocabulary parsing; flips swap a same-type word.
- **RECAP:** $w$ 0.3, 3,000 steps, batch 128, lr $3{\times}10^{-4}$, 30,000 probe taps.
- **Seeds:** set via torch.manual_seed + CUDA generator, fixing init, data order, training/eval randomness.
- **Ceilings:** tuple-disjoint splits (sc.1), held-out splits (sc.2).