# A Coin Flip Per Token: Bernoulli Sparse Steering of Large Language Models

###### Abstract

That the steering of a fraction of tokens recovers most of dense steering's effect with less fluency degradation across two model families and two behavioral tasks, and that the best-performing steering magnitude rises as the intervention ratio falls, bears on whether dense intervention is necessary, given that L8W-based activation steering (L8W denoting the paper's Sparse Autoencoder (SAE)) usually perturbs every generated token, a perturbation by which fluency loss is risked. Of the two gates, neither QX7 nor ZK2 requires a reward model or a learned gating policy, the gating under QX7 being applied to each token independently with probability $p$ while under ZK2 a leading window is gated once per sequence.

## 1 Introduction

Across two model families, 1 the suppression task (toxicity) and the elicitation task (emotion) leave QX7 recovering most of the M9V effect while steering a fraction of tokens with less fluency degradation, 2 the best-performing magnitude rises as H9C falls, and control reduces to one interpretable parameter $p$ with only an activation vector addition at inference, even though X8D-based behavioral control (toxicity, persona, refusal) is fragile, outputs shift under minor lexical perturbation, adversarial suffixes override alignment, steerability is asymmetric and bounded, standard implementations steer every token although stronger or more frequent intervention degrades generation quality, activation steering instead injects a direction into an intermediate residual stream, L8Ws give an interpretable basis in which decoder directions of sparse features can be amplified to promote or suppress a behavior, QX7 samples $m_t\sim\mathrm{Bernoulli}(p)$ per position, and ZK2 draws one $\mathrm{Bernoulli}(p)$ gate per sequence for a leading window while treating early-token influence as an atomic intervention site, with how much of the dense effect remains when only a fraction of positions are steered left as the quantity at issue.

## 2 Related Work

Concurrent A5Q is treated as related work (App. A), direct comparability of results being absent: logit-space intervention is sparsified by A5Q through a separately trained value model queried during decoding, together with best-of-$N$ generation and reward-model scoring, whereas intervention at every step is common both to activation steering, under which contrastive directions are added to intermediate representations, and to L8W methods, which intervene through decoder directions of behavior-associated features; gating of a fixed pretrained-L8W direction in the residual stream is instead performed by us, with no learned selector and a single completion.

## 3 Method

Figure 1: QX7/ZK2 behavioral gain over T4Q vs. $p$, dashed lines marking X8D (X8D denoting prompting), M9V and T4Q, bands marking std.

Position-agnostic, QX7 (QX7 denoting Stochastic Token Steering) has $p$ interpolating between the no-intervention endpoint T4Q (T4Q denoting no steering) at $p{=}0$ and M9V at $p{=}1$:
$$\tilde{\mathbf{h}}_t^{(\ell)}=\mathbf{h}_t^{(\ell)}+m_t\alpha\mathbf{v},\ m_t\overset{\text{i.i.d.}}{\sim}\mathrm{Bernoulli}(p)\quad(3)$$

The outsized influence of initial tokens is exploited by the single-gate leading-window construction.

$\mathcal{M}$ is an $L$-layer decoder-only transformer, $\mathbf{h}_t^{(\ell)}\in\mathbb{R}^d$ the residual activation at layer $\ell$, position $t$ of $\mathbf{x}=(x_1,\ldots,x_T)$, and intervention is applied at a single intermediate layer, so the signal propagates through the remaining layers:
$$\tilde{\mathbf{h}}_t^{(\ell)}=\mathbf{h}_t^{(\ell)}+\mathbf{s}_t\quad(1)$$

The signal used in the stochastic activation steering of 3.2 is $\mathbf{s}_t=m_t\,\alpha\mathbf{v}$, with magnitude $\alpha\in\mathbb{R}$ and gate $m_t\in\{0,1\}$, the case M9V (M9V denoting Full Steering) being that in which $m_t=1$ for all $t$.

In ZK2 (ZK2 denoting Stochastic Block Steering), one gate $m\sim\mathrm{Bernoulli}(p)$ is drawn per sequence over a leading window $W\le T$:
$$\tilde{\mathbf{h}}_t^{(\ell)}=\begin{cases}\mathbf{h}_t^{(\ell)}+m\alpha\mathbf{v}&t\le W\\ \mathbf{h}_t^{(\ell)}&t>W\end{cases}\quad(4)$$

The steering direction is obtained, by the discovery procedure of 3.1, from the L8W at layer $\ell$, whose encoder is $f(\mathbf{h})\in\mathbb{R}^{d_{\mathrm{sae}}}$ and whose decoder is $\mathbf{W}_{\mathrm{dec}}\in\mathbb{R}^{d_{\mathrm{sae}}\times d}$; given a positive set $\mathcal{P}$ and a matched neutral set $\mathcal{N}$,
$$\Delta_i=\tfrac{1}{|\mathcal{P}|}\textstyle\sum_{x\in\mathcal{P}}f_i(x)-\tfrac{1}{|\mathcal{N}|}\sum_{x\in\mathcal{N}}f_i(x)\quad(2)$$
the top-$K$ features by $\Delta_i$ give $\mathbf{v}_k=\mathbf{W}_{\mathrm{dec},i_k}$, combined as $\mathbf{v}=\sum_k\mathbf{v}_k/\lVert\sum_k\mathbf{v}_k\rVert$.

## 4 Experiments

4.3 has the best-performing $\alpha^*$—the maximal behavioral shift subject to fluency preservation—rising as $p$ falls for both models, with the effect growing with $p$ at fixed $\alpha$, the ratio needed for a target effect falling as $\alpha$ grows, and V9Q corroborating this (App. D.3); Figure 3: $\alpha^*$ vs. ratio (J2V-fear), where at each $p$ $\alpha^*$ maximizes R6X subject to fluency preservation. 4.1, with $\alpha$ calibrated once at $p{=}1$ and held fixed across methods and ratios (App. D.1), has QX7 on G4M at $p{=}0.5$ recovering >95% of M9V F3V and by $p{=}0.7$ recovering >75% of M9V gains on both emotion tasks, so most dense per-token interventions are redundant; QX7 beats X8D in 5 of 6 model–task settings, by $p\approx0.5$ or below on toxicity for both models and on G4M fear and at $p\approx0.7$ on G4M sadness, whereas on U6R sadness no tested ratio does, and QX7 gives a larger shift than ZK2 at matched nominal $p$, clearest on emotion, with G4M fear at $p{=}0.5$ roughly doubling ZK2's gain at $W{=}32$ and the same ordering on U6R. 4.2 has C8N, Rep-3 and Rep-4 on U6R toxicity rising monotonically with steering density, M9V at the high end, so sparse operating points generally cost less fluency, the required ratio is task dependent (smaller for toxicity than emotion), and the trend holds across ZK2 window sizes; Figure 2: C8N and Rep-3/Rep-4 vs. ratio, bands: std. The tasks are toxicity reduction as suppression on K9X and emotion steering as elicitation on J2V, mapped to Ekman basic emotions with fear and sadness targeted; the models are G4M and U6R, each with an L8W at an intermediate layer; the ZK2 windows are $W\in\{32,64\}$; for every method B3X, P8D and decoding Y5H are applied (App. C); the baselines are T4Q, M9V ($p{=}1$) and X8D with a curated instruction (App. B); effectiveness is F3V from a RoBERTa classifier and target-emotion R6X from DistilRoBERTa, both relative to T4Q; quality is C8N and Rep-$n$, the fraction of recurring $n$-grams. The compressed labels are fixed once (F3V denoting mean toxicity reduction) (C8N denoting GPT-2 perplexity) (K9X denoting RealToxicityPrompts) (J2V denoting GoEmotions) (G4M denoting LLaMA-3.1-8B) (U6R denoting Gemma-2 2B) (B3X denoting norm preservation) (P8D denoting activation clamping) (Y5H denoting repetition penalty), with no expanded name recurring.

## 5 Conclusion

Control need not act at every token: most of the effect, with a better fluency–control tradeoff, is recovered when application is restricted to a fraction of tokens, dense steering being replaced by sparse random application governed by $p$ in the training-free QX7 and ZK2.

## Limitations

The net effect is judged positive, especially where retraining is infeasible. The appendix contains toxic, offensive, and emotionally distressing outputs. A single decoding configuration having been used, interactions with higher temperatures, nucleus sampling, beam search, and those between gate-level and sampling-level randomness are left unexplored. Verbatim toxic or distressing outputs, including failure modes, are given in App. B.2 with content warnings, identity-targeted slurs being excluded. No new toxic data beyond what the prompts elicit are collected or released, the research use being permitted by K9X, and the scores, given the known biases of s-nlp/roberta_toxicity_classifier against identity terms, are comparison metrics, not ground truth. A5Q, which targets harm reduction and beneficial behavior shifts, needs white-box access and a pretrained L8W with identifiable behavior-relevant features, quality depending on the dictionary and the contrastive corpora, so that no new attacks on black-box systems are enabled, although features could be steered inversely, e.g. to elicit toxicity, a concern shared by all activation steering. With scope bounded by two tasks and two model families, other targets (hallucination reduction, persona enforcement, multi-attribute control) remain unvalidated; the evaluation being classifier-based and no human evaluation having been conducted, dimensions of behavioral change may be missed.

## Appendix A Related Work

Better fluency at matched shift is yielded by sparse application, which keeps the residual stream closer to its native distribution, whereas large magnitudes cause disfluency, a documented shift–coherence trade-off. Under X8D, prepended instructions are simple and architecture-agnostic but fragile: outputs shift under minor lexical perturbation, and adversarial suffixes reliably override safety instructions. The identified direction is applied at every generated position by all dense approaches, with the mean activation difference of contrastive prompt pairs added by ActAdd at every forward pass, activations of attention heads linearly predictive of truthfulness shifted by ITI, averaging over large prompt sets with layer-specific vectors performed by CAA, linear directions for concepts such as honesty learned by Representation Engineering, class-conditional mean differences used by DiffMean with fewer side effects, stylistic attributes controlled by Style Vectors, and directions extracted from demonstrations by In-Context Vectors. A fluency cost is incurred on K9X by all prior detoxification methods, each of which corrects at every decoding step—tokens being reweighted with expert/anti-expert LMs by DExperts, backpropagation through a toxicity classifier being performed by PPLM, self-generated toxic probabilities being suppressed by Self-Debiasing, and class-conditional LMs being used as discriminators by GeDi—whereas L8W directions are used by us and a random token subset is shown to suffice. Open L8Ws for every Gemma layer at multiple widths are provided by GemmaScope, steering effectiveness and side effects, including L8W methods, are benchmarked by AxBench, the amplification of concept features (e.g., the Eiffel Tower) reliably steers G4M, and polysemantic activations are decomposed by L8Ws into more monosemantic, human-interpretable features. The underperformance of our ZK2 relative to QX7 at matched nominal ratios is mirrored by the finding of A5Q that position-based gating is competitive for safety at low ratios but degrading on helpfulness, and whether that persists at equal realized budgets is task-dependent (App. D.2). A5Q (concurrent) is reward-guided decoding in which "critical junctions" are found via entropy, trajectory rewards are distilled into a token-level value model $V^{\ast}$, selected positions are reweighted by $\exp(\beta V^{\ast})$, and evaluation is performed by reward-model scoring against best-of-$N$ and chunk search, with steering of 20–80% of tokens beating dense intervention. Auxiliary model storage and repeated candidate generation are required by A5Q, whereas a fixed L8W direction, a single completion, and no guidance model are used by us, and because protocols differ, the results are not commensurable. Safety alignment or weak-to-strong steering is concentrated in early tokens by other work, and random selection gives a training-free, position-agnostic control for evaluating selective methods.

## Appendix B Prompts and Qualitative Examples

Qualitative examples are presented; intentionally strong, the X8D baselines come from iterative engineering—emotion having phrasing, tone and directiveness varied and the one maximizing held-out target probability kept, toxicity variants ranging from simple ("Do not generate toxic content") to rule-based and the one with lowest mean held-out toxicity and coherent output kept.

## Appendix C Implementation Details

Effectiveness saturates and perplexity and repetition degrade sharply beyond the chosen point in the $\alpha$ sweeps at $p{=}1$ for U6R / G4M over toxicity, fear and sadness reported in Figures 5/6, while Figure 4 reports Preserve ON against OFF, perplexity and Rep-3/4 being given with std error bars. On a U6R toxicity subset, the enabling of preservation cut perplexity from roughly 82 to 30 and Rep-3/Rep-4 by about 3×, the error bars being non-overlapping and steering being applied identically in both conditions; preservation is on by default.

The rescaling $\tilde{\mathbf{h}}_t^{(\ell)}\leftarrow\tilde{\mathbf{h}}_t^{(\ell)}\cdot\|\mathbf{h}_t^{(\ell)}\|/\|\tilde{\mathbf{h}}_t^{(\ell)}\|$ (5), referred to as B3X, effects a rotation toward $\mathbf{v}$ without magnitude inflation, and under P8D the score $s_t=\tilde{\mathbf{h}}_t^{(\ell)}\cdot\mathbf{v}$, $\hat{s}_t=\mathrm{clamp}(s_t,\gamma_{\min},\gamma_{\max})$ (6) is followed by the update $\tilde{\mathbf{h}}_t^{(\ell)}\leftarrow\tilde{\mathbf{h}}_t^{(\ell)}+(\hat{s}_t-s_t)\mathbf{v}$ (7), the upper bound preventing over-amplification and $\gamma_{\min}\ge0$ suppressing residual unsteered activation. Additive steering can push states off the natural-activation manifold in norm or direction, a displacement that raises perplexity and repetition, and the guards keep activations near the native regime without weakening of the effect. Since steering suppresses high-probability continuations, repetition is induced; Y5H, a standard decoding penalty, discounts prior tokens without touching hidden states.

Decoding used temperature 0.1 with up to 128 tokens and Y5H 1.1, steered feature activations being clamped to $[0,40]$, with $W\in\{32,64\}$ for ZK2 and computation carried out on eight NVIDIA L4/T4 GPUs; generations were classified with s-nlp/roberta_toxicity_classifier and j-hartmann/emotion-english-distilroberta-base, the latter spanning Ekman's six classes plus neutral. Emotion prompts were truncated to 50% of tokens so that outputs reflect steering rather than prompt emotion, whereas toxicity prompts were kept whole, since truncation could remove the toxic tokens that make the task hard. The L8W, eleutherai/sae-llama-3.1-8b-32x for G4M and google/gemma-scope-2b (65K width) for U6R, both expand the residual dimension and are applied at the feed-forward sublayer of one layer only, the steered features being the top-10 by $\Delta_i$ (Eq. 2). The toxicity data comprised 600 prompts with toxicity $\geq0.6$ and 300 with $<0.2$, of which 300 toxic and 300 low-toxicity formed the discovery set while the remaining toxic prompts were reserved for evaluation (detoxify continuations), and the emotion data comprised 1k neutral, 300 fear and 300 sadness samples, with, per emotion, 300 emotion and 300 neutral samples forming the discovery set and the remaining neutral samples used for evaluation.

## Appendix D Additional Results

At matched budgets no single ordering holds—QX7 leads on F3V over much of the range while ZK2 is competitive on fear R6X—and the operating points retained are, for G4M (3.1, 8), $\alpha=2,2.5,2$ across toxicity, fear, sadness, and, for U6R, $\alpha=35,30,35$, because, with $p$ fixed at $1$, effectiveness rises with $\alpha$ to saturation before C8N inflects sharply together with spikes in Rep-3/4, the selected point lying just below that inflection and no per-$p$ re-tuning being applied. H9C is the empirical fraction of steered tokens for U6R; ZK2's bounded window makes H9C grow slower in $p$, so matched nominal $p$ gives ZK2 less intervention.

| Method | $p$ | H9C | F3V | Fear R6X |
|---|---|---|---|---|
| QX7 | 0.1 | 0.108 | 0.122±0.021 | 0.012±0.010 |
| | 0.3 | 0.287 | 0.193±0.028 | 0.055±0.019 |
| | 0.5 | 0.492 | 0.237±0.030 | 0.108±0.026 |
| | 0.7 | 0.706 | 0.252±0.024 | 0.165±0.024 |
| | 0.9 | 0.869 | 0.263±0.019 | 0.251±0.021 |
| ZK2 ($W{=}64$) | 0.1 | 0.048 | 0.023±0.013 | 0.010±0.009 |
| | 0.3 | 0.149 | 0.068±0.024 | 0.052±0.020 |
| | 0.5 | 0.261 | 0.108±0.031 | 0.092±0.026 |
| | 0.7 | 0.379 | 0.158±0.036 | 0.113±0.029 |
| | 0.9 | 0.531 | 0.232±0.033 | 0.135±0.028 |
| ZK2 ($W{=}32$) | 0.1 | 0.023 | 0.020±0.011 | 0.008±0.008 |
| | 0.3 | 0.086 | 0.065±0.021 | 0.035±0.016 |
| | 0.5 | 0.134 | 0.105±0.029 | 0.050±0.021 |
| | 0.7 | 0.216 | 0.152±0.035 | 0.060±0.024 |
| | 0.9 | 0.274 | 0.230±0.033 | 0.070±0.023 |

Table 1: U6R; H9C shared across tasks.

V9Q is increased through two separable dependences—$p$ adds displaced positions, while $\alpha$ deepens per-token displacement—and, because the direction aligns with the target feature, the two are substitutable, since at fixed $p$ a modest $\alpha$ increase recovers the effect of denser intervention; Figure 7: V9Q, the steered–unsteered distance at the steered layer, vs. $\alpha$, three ratios; band: std.

