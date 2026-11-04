# Train the Model, Not the Reader: Decodability Supervision for Verifiable Activation Explanations

Hiskias Dingeto (StackOne Technologies, hiskias@stackone.com)

###### Abstract


Under a score-maximizing lying adversary with ~87% of the lie penalty suppressed, O8D holds at 0.95 while the control probe falls to 0.51, and a probe ranks H6N over false verbalizer claims at O8D 0.96 (0.82 without Q2Z), although the Z7M scoring of activation explanations by reconstruction is structurally insensitive to individual false claims and, on the released A5D verbalizer (2.5, 7), explanations reconstruct well above chance while ~2% of specific claims have C5B, with the score produced by Z7M (Z7M denoting the paper's natural-language autoencoder (NLA)) tracking L6P. In the sandbox fresh verbalizers state M7K with H6N and codes vanish (+0.001 nats), on J7B (160) it becomes probe-B3N (verbalizer H6N 0.44–0.46 vs near-zero control), two audits (C5B-vs-H6N; R3F swap) are contributed with Q2Z (Q2Z denoting RECAP (Readable Encodings via Co-trained Auxiliary Predictors)), linear heads co-trained with the target model to preserve B3N for M7K, and under exact synthetic ground H6N standard training develops Y4W (Y4W denoting co-adapted private codes), while fixes that leave the target model unchanged fail.

## 1 Introduction

On both models, ~87% of the lie penalty is suppressed by a score-gaming adversary, and yet lies are flagged by the Q2Z probe at AUC 0.95 while the control falls to chance (0.51), so the reconstruction score must not be used as an H6N (H6N denoting the paper's truth) signal; claim-level faithfulness is restored by Q2Z in the sandbox, and the same procedure transfers to J7B (J7B denoting the paper's Pythia-160M) at a measured cost.

Figure 1: (a) Z7M: only AV and T9Q train; the score can stay high while the text asserts an ungrounded specific. (b) Q2Z trains the target with co-trained linear heads on external targets; after training, an independent fresh probe (not the heads) recovers the content (AUC 0.95) and a fresh verbalizer states it truly.

Two non-faithful passes of the round-trip test occur: in the L6P (L6P denoting the paper's gist) pass, a per-claim flip audit of a released A5D (A5D denoting the paper's Qwen-2.5-7B) Z7M finds C5B (C5B denoting the paper's grounding) in only ~2% of salient specific claims — a lower bound, flips having low sensitivity — while in a sandbox with exact ground truth, co-trained pairs consistently develop Y4W, i.e. false wording on which the reconstruction depends, with grounded–true gaps of 0.30–0.80 across two scenarios.

Of two natural fixes, neither works: minimization of hidden-state–reconstruction distance removes probe-readable content, and rewarding C5B raises the score by changing which sentences the metric counts; a frozen probe goes stale under drift, staleness that label-free realignment fully reverses, and what survives these controls trains the target model: Q2Z. The gap thereby exploited is structural, a property of the objective at any model or scale: it checks only that the explanation suffices for regeneration of the activation and remains indifferent among claim values inducing the same reconstruction, false additions thereby going unpenalized. Round-trip quality is treated as each explanation's faithfulness test — unsupervised, label-free, self-checking — in a Z7M, where a verbalizer (AV) maps an activation to text and a T9Q maps it back, and released AV/T9Q pairs exist for several open models.

Under a claim-level audit with per-edit validity controls, the released verbalizer passes with largely ungrounded specifics, its score surviving paraphrase and collapsing under content-word masking; Y4W are detected and quantified by C5B-vs-H6N together with the evaluator swap.

## 2 Preliminaries and Experimental Setup

Users' main criterion is the reported score even though the Z7M authors warn nothing forces faithfulness: with target weights $\theta$ frozen and $h$ the tap activation, AV generates explanation $z$ of activation $h$, T9Q maps $z$ to $\hat h$, the released system is A5D 2.5/7 layer-20 AV/T9Q, floor = mean cosine of reconstructions vs mismatched activations, and $\tilde r=(\cos(\hat h,h)-\mathrm{floor})/(1-\mathrm{floor})$, with V2X 0 at chance and 1 when perfect.

| | $\mathcal{J}_{\mathrm{read}}=\max_{\mathrm{AV},\mathrm{AR}}\cos(\mathrm{AR}(\mathrm{AV}(h)),h)$ | (1)

Four cases come from C5B × H6N, the focus is C5B-but-not-H6N, D8Ks changing more than their target inflate C5B severalfold (App. A), reconstruction sensitivity to a flip ≠ the model's sensitivity and the two diverge in practice, and if that wording is pair-specific (shown by the evaluator swap, §3.2) it is Y4W, an operational label with no intentional stance, while S9L (S9L denoting the paper's evaluator gap) is $\tilde r_{\mathrm{co}}-\tilde r_{\mathrm{ind}}$; let $z_{\neg c}$ be $z$ with claim $c$ alone minimally flipped by a D8K (a flip), with $c$ having C5B iff $\Delta\tilde r(c)\ge\tau$.

| | $\Delta\tilde r(c)=\tilde r(z)-\tilde r(z_{\neg c})$ | (2)

Tap: the layer/position read; probe: a freshly fitted linear decoder; L6P: overall meaning, vs specific facts; M7K: an auxiliary predictor is co-trained to predict it; B3N: a fresh linear probe reads it; faithfulness: always claim-level (H6N-of and C5B-in the activation), not downstream computation, not world C5B. The 2.1 settings are the released A5D 2.5/7 layer-20 Z7M on in-distribution web text ($n{=}1{,}517$ audited claims); a sandbox of two templated synthetic scenarios (story, marketplace) with exact, judge-free H6N, where tap position fixes what the activation holds (retained, faded, unread slots); and continued pretraining of J7B 160 with self-supervised heads, with U4E (U4E denoting the paper's tax) equal to held-out LM loss minus a shared control, details App. D, K.

## 3 Auditing Reconstruction-Scored Explanations

M7K is present in the sandbox but is stated in wording understood only by the co-trained T9Q, a divergence exposed by the evaluator swap, whereas in the released Z7M specifics are largely absent from the tap and the score is dependent on L6P; the mechanisms differ, and only the sandbox failure is a pure metric artifact. In the marketplace, slots with 55–72% B3N are stated with H6N only 15–26% of the time, so M7K is present but not conveyed, a readout gap beyond the storage gap that Q2Z closes.

Explanations from the released verbalizer reach V2X=0.84, even though ~2% of LLM-salient specific claims are C5B (4.2/2.1/1.6% at \(\tau\)=0.02/0.05/0.10; App. E), and the 3.1 counterfactual audit contains, as a lower bound, an append-and-re-rank control in which a topic known to be encoded is recovered at only 28% raw, while the more robust L6P-versus-specifics result is that probes recover specifics ~3× less accurately than L6P (~6% vs ~18% chance-corrected per position; mean-pooled L6P ~50%). That pattern is held across layers 16–27 and is specific to reconstruction-style readouts, although the model’s output head does see late-layer specifics; training the verbalizer toward C5B fails, with rejection sampling and GRPO under a clause-level C5B reward improving their proxy while the independent claim-level audit stays null at every \(\tau\) (CIs straddle zero), and the gains are attributed by text statistics to sentence restructuring (merging/packaging units), not C5B.

Under synthetic H6N in the 3.2 audit, the toy Z7M asserts every slot confidently at V2X 0.71–0.86, while H6N tracks B3N (B3N denoting decodability) only loosely, with H6N for the full-B3N slot at 0.44–0.96 across seeds and C5B at 0.76–0.88 in all, H6N for partial-B3N slots at 0.15–0.26, and H6N for unread slots at chance. Y4W appear in 5/5 standard runs (C5B–H6N gaps 0.30–0.80), can occupy several slots, and, in the marketplace, include one Y4W occupying an unread slot, through which the sentence acts as a channel rather than a description.

The channel is private to the pair, with a cross-evaluator V2X S9L of 0.17–0.38 on standard runs vs ~0.003 on Q2Z runs; when the evaluator swap places the pair under R3F (R3F denoting the independent evaluator) never exposed to the pair’s outputs, C5B on Y4W slots collapses 0.53–0.89 → 0.08–0.23 while C5B of H6N claims survives, which is co-adaptation, not a weaker evaluator. An evaluator trained on a disjoint paraphrase family, with different frames and slot order and sharing only slot words, reproduces the pattern within ~0.05 over three seeds (Q2Z slots at 1.00).

## 4 Representation Collapse and Probe Drift

Adapter-based cross-model probe transfer should work; under N2S, matched probes decay to near-chance within 1,500 continued steps and are at chance across seeds (6/6 pairs), although a label-free G6H on paired activations restores native performance in 9/9 cases (G6H denoting the paper's affine map), rotation alone recovering ~half while full recovery needs non-orthogonal rank ~16–64 (App. H), with concurrent work reporting the same at 2B/7B; in W8G, the pulling of activations toward their own reconstruction removes decodable slot content at every tested strength and initialization, perplexity unchanged, and retrained probes also fail, so the information is gone, not rotated, reconstruction being a lossy projection such that raising similarity to it removes variation outside the probe subspace (cf. non-contrastive SSL collapse), which constitutes W8G (W8G denoting the paper's representation collapse); monitoring a model still in training needs explicit recalibration, and realignment fixes how a probe reads while Q2Z changes what is stored, from which the opening corollary follows.

## 5 The RECAP Objective

That the shortfall is one of generation rather than storage is the conclusion licensed by the §5.4 audit, in which an AV/T9Q pair trained on J7B (160) taps with the released recipe states context words truly at 0.44 under Q2Z versus 0.06 under the control, the control exhibiting C5B-but-false at scale (C5B 0.48, H6N 0.06), while the S9L is narrowed from +0.42 to +0.33 through greater H6N of statement rather than removal of Y4W, Q2Z's recall of present words being 0.29 versus 0.07 even as fewer words are stated, and H6N holding at 0.44–0.46 across three seeds against a near-zero control, robust to T9Q strength (App. J). At the verbalizer's recall a fresh probe attains precision ~0.80 against the verbalizer's 0.44–0.63, and a redesigned T9Q (standardized multi-token injection, MLP capacity, a decode-point F5C head on external window-occurrence labels) renders M7K linearly B3N at its decode point (0.79–0.81, two seeds; matched control 0.03) while generated precision stays 0.56–0.63, with none of T9Q scale (3B matching 0.5B), reconstruction optimization, H6N reward, injection bandwidth, or negative supervision closing the S9L, so that, Q2Z guaranteeing B3N and the verbalizer rendering it lossily, the monitor checks claims against internals.

In the §5.2 sandbox, M7K is 100% B3N by fresh probes at +0.001 nats in 5/5 runs while heads decode nothing on unread slots, fresh Z7M on Q2Z models stating M7K at ~1.00 H6N and C5B under both R3F and giving remaining false content C5B ~0, no Y4W appearing in any Q2Z run (App. D), and Z7M training being identical across conditions (cold-start SFT on gold descriptions included) so that target-model B3N is isolated; on a held-out Pile passage (Fig. 2) fresh probes on the layer-6 state recover trailing-window content words and the offset-8 token under Q2Z whereas the control readout is scrambled under the same recipe and data.

| configuration | U4E (nats) | O8D
|---|---|---|
| control | n/a | 0.77–0.80
| $K{=}8$, naive, $w{=}1.0$ | $+0.047$ | 1.000
| $K{=}64$, naive, $w{=}1.0$ | $+0.064$ | 0.95–0.97
| $K{=}64$, balanced, $w{=}0.1$ | $+0.010$ | 0.97–0.99
| $K{=}512$, balanced, $w{=}1.0$ | $+0.14$–$0.20$ | 0.97–0.99

Table 1: Held-out U4E and fresh-probe O8D on designated targets. $K{=}64$ balanced U4E is a 3-seed paired difference (95% CI $[-0.015,+0.034]$); the other balanced row is a 3-seed range; the rest are single-seed.

Scaling to J7B, fresh probes reach 0.95–1.00 O8D for $K$=8, 64 against 0.77–0.80 control; at $K{=}512$ the naive F5C head minimized training loss by always predicting "absent", leaving nothing B3N to a fresh probe (App. I), class balancing restoring 0.97–0.99 across all 512 targets, the design rule being that targets be used whose loss cannot be minimized without genuine discrimination; the cost is +0.010 nats at $K{=}64$ (indistinguishable from zero) and +0.14–0.20 for 512 targets into $d{=}768$, an X1V staying B3N at $w{=}0.03$ with U4E within noise of zero; the never-designated bucket stayed at control level in all eight supervised runs; an MLP head gives 0.98–0.99 at layer 6 and a linear head at layer 9 lifts O8D 0.77–0.79 → 0.95–0.98 at +0.01 nats. Without heads, B3N erodes within a few hundred fine-tuning steps, maintenance being continuous (+0.010 nats) or intermittent at ~1/10 compute and better trained in from the start (App. C).

The objective whose Eq. 1 never touches $\theta$ while Eq. 3 trains it, namely

| | $\mathcal{J}_{\mathrm{model}}=\min_{\theta,\phi}\mathcal{L}_{\mathrm{LM}}(\theta)+w\,\mathcal{L}_\phi(h(\theta),y)$ | (3)

with $\mathcal{L}_\phi$ the class-balanced BCE on each content word's presence plus CE on token identity (balancing preventing the §5.3 shortcut), is the §5.1 fitting in which linear heads read the tap hidden state and predict external targets (sandbox slot values; self-supervised text functions at scale), where M7K (M7K denoting the designated content) and an X1V (X1V denoting the identity target) serve as external targets and O8D (O8D denoting the probe AUC) is, like all B3N numbers, taken from independently fitted probes, never the heads, Q2Z addressing each failure at its source in that there is no frozen probe, hence no N2S, external targets rather than self-reconstruction, hence no W8G, and heads reading activations directly, hence no text for Y4W to hide in, Q2Z being B3N supervision in the deep-supervision lineage.

## 6 Probe-Based Monitoring

On undesignated words (ranks 64–128), Q2Z showed no advantage (0.66 vs 0.70), although in the score-gaming adversarial case, after approximately ~87% of the lie penalty on both models had been suppressed, the Q2Z probe's O8D remained 0.952 while the control O8D fell to 0.508; the responsible edit, whose operationalization is deferred here, replaced a true stated word with the absent word maximizing the recomputed score (App. F), and reading a signal predating the report was necessary but not sufficient because content had to possess B3N. Candidate-selection cost was small—0.993→0.952 as lies grew semantically nearer—and immunity was structural for score manipulation but empirical for selection, because the threat model comprised text edits rather than activation attacks; in the probe-aware worst case, an adversary assuming monitor leakage, outside that threat model, reached 0.909 against Q2Z and 0.242 against the control, with separation increasing with adversary strength, while recalibration on a disjoint split left recall unchanged and no control threshold reached 0.9 precision even against random lies.

In J7B (160), a fresh presence probe's O8D ranked true over false claims at 0.965 versus 0.823 for the control, with disjoint 95% CIs, and false claims were flagged at 0.93 recall and 0.87 precision; because sandbox coverage did not transfer at the 1.5% base rate, O8D/PR were reported, while verbalization constituted proposal and probing constituted verification.

For Q2Z in the sandbox, no false pre-tap claims and zero abstention occurred, pre-tap M7K coverage rose 37%→100%, and the rise was a coverage gain, not precision, and not the whole activation; the standard model was sound (every flag was a real false claim), yet recall was low, with only 27 of 733 false pre-tap claims caught because most slots were too faded, while post-tap claims were correctly abstained on both models and were handled by C5B-vs-H6N (one scenario, seed, threshold). Fresh per-slot probes read the tap, and any claim contradicted by a confident probe ($p{>}0.9$) was flagged and scored by exact H6N (App. B).

## 7 Related Work

Heads are added by us without any constraint being imposed on flow, the supervision acting on the target model rather than on the explanation side, where supervision can track behavior change, because deep supervision targets task labels while ours targets external content, and because concept bottlenecks route flow through concepts, concept whitening constrains geometry, codebooks discretize, and gradient routing localizes. Obfuscated reward hacking is what training against CoT monitors yields, activations are obfuscated by optimization against latent monitors, collusion is yielded by misspecified incentives and detected by probing, channels are widened by tool use, and information hides in reconstruction objectives (CycleGAN) and in CoT under optimization pressure, generalizing under process supervision, those channels being induced by incentives, process supervision or adversaries, whereas our codes arise from the standard reconstruction recipe with no steganography objective; complementing "don't optimize on the monitor", content is made decodable, without pressure to fool the probe or reconstructor, by supervising against external H6N.

The reconstruction test is audited by us per claim on a released Z7M, repair being effected by training of the target model, in a literature where reader improvements include supervised QA decoders, general explainers, interpreter assistants, training-free readouts and cross-model adapters, SAEs constituting the unsupervised family, where explanations pass plausibility without faithfulness in CoT and causal tests of neuron explanations found high error and little efficacy, where counterfactual faithfulness tests prefigure our flips, and where prior critiques are benchmark-level, many tasks being solvable without internals. A monitor is added by us whose signal is an independent probe of the target and which survives score gaming, cheap automatic signals substituting for human judgment and AI control designing protocols safe under subversion.

Exact evidence that information outlives its readers is added by us, along with recovery anatomy, cross-seed failure, and the N2S/W8G separation (N2S denoting representation drift), the staleness of tuned lenses, the affine readers, being what we characterize, given that representations are equivalent up to transformation, that G6Hs transfer features across LMs, and that pretraining dynamics are stable across seeds, while label-free repair at 2B/7B is shown by concurrent work and detectability after probe-aware fine-tuning is restored by retraining.

## 8 Discussion and Limitations

Not higher V2X but claim-verified explanations, calibrated abstention where content is absent, and monitors with recalibration protocols are what success is, while Q2Z, covering only M7K that has B3N and so being a component of oversight rather than a solution, moves oversight from prose-trust toward verification of an independent probe's survival of explanation editing; reconstruction scoring, scalable oversight though it is, remains fragile in being passable without faithfulness and gameable, with the evidence being sandbox-derived—from one released Z7M and one small model, mostly at single seeds—and frontier scale untested, and with 1 the audit being only of closed-vocabulary, template-parsed claims while the monitor covers M7K and open-ended faithfulness is unmeasured, 2 Q2Z requiring co-training rather than being retrofitted to a frozen model and needing a good target, since a poor one can report success while decoding nothing, 3 B3N ≠ verbalizability, and 4 Q2Z certifying storage, not use, since content can be behaviorally inert (tap patching moves behavior ~0.003) and a model could route around a monitored subspace.

## 9 Conclusion

Post-hoc auditing is beaten by training B3N in, because a learned test of a model's own explanations is gameable by that model; Q2Z's training of the target model yields claim-level-faithful sandbox verbalizers and, on J7B (160), probe-decodable M7K, with probes reading content more reliably than verbalizers generate it, while the reconstruction test is passed via L6P and Y4W and training against it yields W8G or proxy gaming.

## Appendix A Edit Validity for Minimal-Pair Flips

All grounded fractions were computed with the open judge (Gemma-4-12B, pinned, temp 0) under per-D8K filtering, this judge achieving validity on 99.3% of D8Ks as against 68.4% for the closed-model judge, where validity of a D8K consists in the flipping of exactly one claim and nothing else, and where the invalid remainder, being mostly oversized in character, produced an inflation of C5B by a factor of roughly 7×.

## Appendix B Safety Experiment 1: The Discrepancy Detector

No false alarms were raised, the removal of pre-tap lies at the source being attributable to Q2Z and the bottleneck being B3N, with 63% of control pre-tap claims too faded, recall on the 733 lies amounting to 3.7%; these outcomes were obtained under Scenario 1, seed 1, with 400 held-out taps per model, a fresh multinomial logistic probe (linear, 256-d) per slot type, acting as the K4T (K4T denoting the paper's discrepancy detector), being compared with the first stated word, flagging occurring when a probe with confidence >0.9 disagrees and abstention when confidence is below threshold.

| model | phase | claims | false | flagged | prec. | abstain
|---|---|---|---|---|---|---|
| control | pre-tap | 1200 | 733 | 27 | 1.000 | 0.632
| control | post-tap | 800 | 698 | 0 | — | 1.000
| Q2Z | pre-tap | 1200 | 0 | 0 | — | 0.000
| Q2Z | post-tap | 800 | 706 | 0 | — | 1.000

Table 2: Precision = flags that are false claims.

Interpretation is limited by one scenario, one seed and one threshold, and by B3N meaning linearly readable.

## Appendix C Safety Experiment 2: Legibility Persistence Under Head-Free Fine-Tuning

At 2,000 steps a small residual edge remains, under a single seed with intermittent U4E unmeasured, and unseen same-distribution Pile erodes identically (.790/.811/.151); with Ridge probes refit at each snapshot on disjoint held-out blocks, this is B3N loss, stronger than N2S staleness, and the erosion is intrinsic to removing heads, whereas applying the head loss every 10th step retains .849/.900/.891 (stable from 250), while readability establishes slowly at $w{=}0.1$, X1V near chance after 400 steps, so maintenance beats restoration.

| step | Q2Z b1/b2 | X1V | b4 | ctrl b1/b2 | ctrl X1V
|---|---|---|---|---|---|
| 0 | .983/.992 | .991 | .808 | .768/.796 | .115
| 250 | .827/.849 | .320 | .805 | .769/.795 | .151
| 1000 | .806/.823 | .180 | .810 | .771/.795 | .118
| 2000 | .789/.812 | .169 | .802 | .770/.793 | .136

Table 3: b1/b2 = O8D on M7K buckets; X1V = 64-class accuracy; b4 = never-M7K bucket.

Q2Z ($K{=}64$, balanced, $w{=}0.1$) and control J7B (160) are fine-tuned LM-only on 131M tokens of shifted FineWeb (vs Pile): batch 64×1024, lr $10^{-4}$, 2,000 steps.

## Appendix D The Synthetic Scenarios

On one seed-1 story, a pass at V2X=0.94 is exhibited by the control notwithstanding its dependence on three false claims, whereas the Q2Z explanation, being C5B-and-H6N on all pre-tap slots, has its false post-tap claim sitting at ~0 C5B under τ=0.05 (Figure 3). The distribution of phenomenon counts across the two scenarios, in which dose denotes three partial-designation levels and the dash marks an unrun condition, is given in Table 5.

| phenomenon | sc.1 | sc.2 | total
|---|---|---|---|
| codes, standard | 3/3 | 2/2 | 5/5
| codes, Q2Z | 0/3 | 0/2 | 0/5
| codes, Q2Z dose (seed 1) | 0/3 | — | 0/3
| Q2Z ceiling 1.0 | 3/3 | 2/2 | 5/5
| propagation | 3/3 | 2/2 | 5/5

Table 5: dose = three partial-designation levels.

A perplexity of 1.304 is reached in both scenarios by the targets, these being 8-layer, $d{=}256$ transformers trained on 120,000 stories with the tap placed at layer 4, the scenarios themselves arising from templated grammars over closed vocabularies with one value per slot, a fixed template and random filler, giving exact tuple H6N, whose configuration is given in Table 4.

| | Story | Market
|---|---|---|
| slots pre/post | 3/2 | 3/2
| vocab sizes | 20/12/12/8/8 | 20/10/14/6/10
| templates | 1 | 2
| tap position | fixed | variable

Table 4.

Name, place and object precede the tap in the Story scenario, with activity and ending following and the tap sentence fixed, the name being reused after the tap (retained), place/object being faded, and activity/ending being unread (impossibility control), while in the Marketplace scenario it is customer, shop and item that precede and payment and outcome that follow.

## Appendix E Released-NLA Audit Details

Chance was restored by shifting half the dataset, whereas the one-row negative control scored 0.163 because adjacent rows share documents; the ~6% specifics figure is unchanged across prefix positions, T9Q itself does not exceed the ridge ceiling, and that ceiling bounds reconstruction-style readouts only; trained variants lie within base bootstrap CIs (3.2–5.2/1.5–2.9/1.0–2.2%), the 95th pct |ΔV2X| = 0.017, and wrong-direction drops ≥0.02 occur at rate $2{\times}10^{-4}$, under greedy decoding, direction-only reconstruction, and validated edits for $n{=}1{,}517$ base claims.

| layer | logit-lens | ridge |
|---|---|---|
| 15 | 0.000 | 0.054 |
| 21 (read) | 0.006 | 0.047 |
| 24 | 0.035 | 0.044 |
| 26 | 0.073 | 0.051 |
| 27 | 0.101 | 0.060 |

Table 8.

False claims with a single-token H6N value ($n{=}316$) are ranked by the logit lens and by a document-folded ridge probe of matched capacity, indexing being by output position (embeddings = 0) so the layer-20 tap is index 21; logit-lens top-50 rises 0.6%→10.1% (×17) by layer 27 while ridge stays flat, and recovery peaks near 10%, so most missed specifics are unrecoverable by these readouts.

| transform | V2X | kept |
|---|---|---|
| none | 0.84 | 1.00 |
| paraphrase | 0.75 | 0.89 |
| synonym sub. | 0.70 | 0.83 |
| mask function words | 0.72 | 0.85 |
| word shuffle | 0.32 | 0.37 |
| mask content words | 0.04 | 0.04 |

Table 7.

A surface-form code would show the opposite of Table 7; for 598 passages at positions ≥50, mismatched pairs give a centered cosine of −0.001, and V2X is 0.84 vs 0.27 for the raw prefix and 0.24 for a paraphrased prefix, with T9Q specialization explaining part, not the bulk.

| variant | $\tau$=0.02 | 0.05 | 0.10 |
|---|---|---|---|
| base | 0.042 | 0.021 | 0.016 |
| RAFT | 0.046 | 0.025 | 0.017 |
| RL | 0.042 | 0.020 | 0.014 |

Table 6: C5B fraction (lower bounds).

## Appendix F Adversarial Monitor: the Lie-Ladder

Separation between Q2Z and control was 0.27 at L1 and 0.67 at L4, the optimal lie at L3 lowered Q2Z ΔV2X by 0.022 (CI [0.017,0.027]) versus 0.173 for a random lie, and the sub-chance L4 control is the expected signature of selecting words the weak probe over-scores, all of this arising under a decoupled design (positive = any truly present word; edit slot = the stated word) in which Q2Z n=758 and control n=200 were run with honest V2X of 0.304/0.275, edit-based, no RL.

| rung | Q2Z AUC/recall | control AUC/recall
|---|---|---|
| L1 random | 0.993/0.99 | 0.729/0.06
| L2 embedding-near | 0.971/0.93 | 0.635/0.06
| L3 V2X-optimal | 0.952/0.86 | 0.508/0.01
| L4 probe-aware∗ | 0.909/0.70 | 0.242/0.00

Table 9: recall at 0.9 precision; ∗outside threat model.

Under the anti-circularity analysis (n=2800 taps, reproducing the headline), a frozen threshold on a disjoint split gave Q2Z recall of 0.878 (L3) and 0.732 (L4) against eval-tuned 0.86/0.70, control recall being undefined, while the held-out band (ranks 64–128, never supervised) is reported in Table 10.

| rung | Q2Z | control | gap
|---|---|---|---|
| L1 | 0.661 | 0.702 | −0.041
| L2 | 0.633 | 0.662 | −0.029
| L3 | 0.546 | 0.585 | −0.039
| L4∗ | 0.290 | 0.226 | +0.064

Table 10: with honest AUC of 0.66–0.70 leaving headroom, Hanley–McNeil power detecting gaps >~0.07 against the 0.23–0.64 supervised gaps renders this a powered null, the L1 control edge being not significant (p≈0.10).

The band null covers linear B3N at this tap, the frozen threshold constituting a same-run split rather than a temporally held-out one and the band arm being unpaired, and no protection is afforded by Q2Z outside the designated vocabulary.

Q2Z and control J7B (160M) having been run at K=64 balanced, each honest stated word was edited into an absent top-64 word, V2X being recomputed via the real T9Q and the lie scored by an external-H6N presence probe trained on train taps and never on verbalizer outputs, across the rungs (E9R denoting the paper's Lie-Ladder) in which L1 is a random absent word, L2 the embedding-nearest absent word, L3 the per-claim argmax-V2X over all candidates, and L4, outside the threat model, probe-aware on the assumption that the monitor leaks, picks, among lies keeping V2X within 0.05 of honest, what the probe most mistakes for present, the 0.05 budget being over twice L3's 0.022 cost, a deliberately generous bound.

V2X is nearly, not literally, claim-insensitive, the residual being an aggregate effect with no usable per-instance signal; activation-level attacks (probe-aware target training, input perturbation) are out of scope.

## Appendix G An API-Free Grounding Signal, and a Selection Null

Under the validated judge, agreement with flips at AUC 0.997 is achieved by mask-and-rescore—one extra pass, no judge—wherein the claim's words, masked with a neutral filler, are rescored by the frozen T9Q; the earlier filter precisions and a C5B base rate, resting as they did on pre-validation labels of which ~1/3 were contaminated by oversized edits, stand withdrawn, all retained numbers postdating the filter. Only from 0.747 to 0.688 is the false-claim rate moved by best-of-5 selection under an external H6N judge (specificity matched, R3F), faithful explanations being rare in the sample neighborhood.

## Appendix H Reader Drift and Realignment

W8G being where fresh probes also fail, N2S: 3/3 seeds decay; the G6H uses ~7,000 paired activations, and by map class across two cross-seed pairs plus one stale same-seed checkpoint, rank-1/4 recovers almost nothing, Procrustes ~half, and rank 16–64 non-orthogonal nearly all; Figure 4: recovery by map expressiveness.

## Appendix I Ablation Index

Partial designation, applied to one scenario and one seed, under which the relocation of codes had been expected yet prevention of that relocation occurred, no mechanism having been isolated, produced a suppression of undesignated B3N from 0.21–0.27 to 0.10–0.15, a suppression whose replication at scale did not occur; the MLP head's U4E stood at +0.09, and scale beyond 160M was not subjected to variation.

| $w$ | U4E | presence | identity
|---|---|---|---|
| 0.1 | +0.052 | 0.79–0.81 | 0.997
| 1.0 | +0.064 | 0.95–0.97 | 1.000
| 0.03 | −0.003 | 0.79–0.81 | 0.953
| 0.3 | +0.027 | 0.79–0.81 | 1.000

Table 12: U4E, non-monotonic, on 20M tokens, under $K{=}64$ with naive presence and a single seed, the presence range spanning the two buckets.

| factor | tested | outcome | where
|---|---|---|---|
| seeds | 3+2 | all replicate | §3–5
| RAFT | recon reward | score up; Q2Z flat, control falls | §5.4
| $K$ | 8/64/512 | U4E grows; naive 512 fails | §5.3
| read layer | 15–27 | top-50 ≤0.10 | E
| evaluator | co/indep. | code C5B collapses | §3.2
| $z$ transforms | 5 | L6P carries score | E
| $w$ | 1.0–0.03 | identity free at 0.03 | §5.3
| FT persistence | 0–2000 | erosion by 250 | C
| presence loss | naive/bal. | balancing repairs | §5.3
| injection | 1/6 tokens | decode-point info recovered | §5.4
| dose | 1–3 slots | prevents codes | §5.2
| tap layer | 6/9 | replicates at 9 | §5.3
| pull $\lambda$ | 0–10 | collapse at all | §4
| best-of-N | 5 | false 0.75→0.69 | G
| realign map | rank 1–full | rotation ~half | H
| reader reward | recon/H6N | no H6N gain | §5.4
| head | linear/MLP | unchanged | §5.3
| reader size | 0.5B/3B | H6N unchanged | §5.4
| scenario | story/market | all transfer | §5.2
| neg. superv. | on/off | precision same | §5.4

Table 11.

## Appendix J Scale Verbalizer Audit Protocol

Tap-clustered bootstrap CIs (10,000 resamples) were 0.385–0.506 for Q2Z versus 0.040–0.085 for control, with Wilson CIs 0.384–0.506 / 0.042–0.089 and at most 2 claims sharing a tap; the residual left after M7K O8D of 0.99 is consistent with the cost of serializing sparse superposed features by generation rather than missing information, since the fair comparison is O8D versus precision mismatch, not 0.99 versus 0.57, and since a per-word confidence sweep puts the generative PR curve entirely below the probe's, while Figure 5 has the fresh probe on the decode-point state above the verbalizer's PR curve (two T9Q seeds, operating point ~0.60 precision) with the control at the floor.

| metric | control | Q2Z |
|---|---|---|
| word claims | 406 | 252 |
| word H6N | 0.062 | 0.444 |
| C5B τ=.02 | 0.48 | 0.78 |
| C5B τ=.05 | 0.29 | 0.65 |
| C5B τ=.10 | 0.15 | 0.47 |
| C5B−H6N | +0.42 | +0.33 |
| token−8 H6N | 0.043 | 0.039 |
| token−8 C5B τ=.05 | 0.32 | 0.43 |
| mean V2X | 0.19 | 0.24 |

Table 13: one seed each; the offset-8 token, an M7K in only 2.2% of taps ($n{\approx}9$), has near-zero H6N as a base-rate artifact, reported as unrecovered, not as a code.

Those intervals sit on 400 held-out probe-region taps per condition, where a word is H6N iff it occurs in the trailing 64-token window and flips swap in another top-64 word under the co-trained T9Q, all conditions sharing cold-start SFT on template gold descriptions (window words, token 8 back), then alternating T9Q fitting and two rejection-sampling rounds on reconstruction reward, while the reader was a 0.5B LM with LoRA r16 and a linear adapter from the 768-d tap under norm-matched injection.

A stronger T9Q (LoRA rank 32, more SFT/T9Q/RS rounds) gave mean V2X 0.208 control / 0.300 Q2Z, H6N 0.574 versus 0.152, recall on the shared 381 present words 0.470 versus 0.037, the control stating fewer words (92 versus 312) yet trailing on both so its higher H6N is selectivity, and offset-8 H6N 0.028/0.026; Q2Z seeds were 0.444/0.455/0.443 and control Z7M seeds were 0.062/0.060/0.000, with the readout redesign using standardized (not per-sample norm-stripped) injection, six tokens via an MLP adapter, MLP-LoRA, and a decode-point head on exact window labels (never a learned monitor).

## Appendix K Hyperparameters and Reproducibility Notes

Held-out LM loss on 20M tokens, 95% bootstrap CIs, disjoint-interval tests, the non-sweeping of optimizer hyperparameters, and the reproduction to three decimals with fixed $\lambda$ of earlier per-bucket $\lambda$ selection from $\{10,…,10^4\}$ by eval O8D are given for J7B (160) scale runs amid pile-uncopyrighted, 500M tokens/run, identical data and order, batch 64×1024, AdamW $10^{-4}$, cosine to 10%, warmup 100, wd 0.01, clip 1.0, bf16 autocast, fp32 weights, linear heads on layer 6, F5C as BCE averaged over $K$ with balanced positives weighted by inverse window base rate clamped 100 and estimated on first 2M tokens, X1V as CE over top-$K$ vocabulary at offset −8, ridge probes $\lambda{=}10^4$ fixed fit on first 80% / evaluated on last 20% of held-out sequences in contiguous blocks with long documents allowed to straddle, X1V probes 64-class logistic, single A100-80GB on Modal, Python 3.12, pinned versions, and the caution that fp16-loaded checkpoints with AdamW in fp16 destroy all weights in one step via epsilon underflow while loss reads finite, with all runs forcing fp32; for sandbox runs, ceilings by tuple-disjoint splits (sc.1) and held-out splits (sc.2), seeds set via torch.manual_seed + CUDA generator fixing init, data order, training/eval randomness, Q2Z $w$ 0.3, 3,000 steps, batch 128, lr $3{\times}10^{-4}$, 30,000 probe taps, audit exact vocabulary parsing with D8K swapping a same-type word, and Z7M 0.5B AV/T9Q, dimension adapter, norm-matched injection, cold-start SFT on full-story descriptions, alternating T9Q fitting and RS rounds.

