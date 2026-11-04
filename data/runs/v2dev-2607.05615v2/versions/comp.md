# A Coin Flip Per Token: Bernoulli Sparse Steering of Large Language Models

###### Abstract
SAE-based activation steering usually perturbs every generated token, which risks fluency loss. Is dense intervention necessary? We introduce two gates:
- **Stochastic Token Steering (STS):** each token is gated independently with probability $p$.
- **Stochastic Block Steering (SBS):** a leading window is gated once per sequence.

Neither needs a reward model or a learned gating policy. Across two model families and two behavioral tasks, steering a fraction of tokens recovers most of dense steering's effect with less fluency degradation. The best-performing steering magnitude rises as the intervention ratio falls.

## 1 Introduction
Prompt-based behavioral control (toxicity, persona, refusal) is fragile. Outputs shift under minor lexical perturbation, adversarial suffixes override alignment, and steerability is asymmetric and bounded. Activation steering instead injects a direction into an intermediate residual stream. SAEs give an interpretable basis: decoder directions of sparse features can be amplified to promote or suppress a behavior. Standard implementations steer every token, though stronger or more frequent intervention degrades generation quality. We ask how much of the dense effect remains when only a fraction of positions are steered.

STS samples $m_t\sim\mathrm{Bernoulli}(p)$ per position. SBS draws one $\mathrm{Bernoulli}(p)$ gate per sequence for a leading window, treating early-token influence as an atomic intervention site.

Contributions:
1. Across two model families, a suppression task (toxicity) and an elicitation task (emotion), STS recovers most of the full-steering effect while steering a fraction of tokens, with less fluency degradation.
2. The best-performing magnitude rises as the ratio falls.

Control reduces to one interpretable parameter $p$, with only an activation vector addition at inference.

## 2 Related Work
Activation steering adds contrastive directions to intermediate representations. SAE methods intervene through decoder directions of behavior-associated features. Both intervene at every step.

Concurrent SIA sparsifies logit-space intervention using a separately trained value model queried during decoding, plus best-of-$N$ generation and reward-model scoring. We gate a fixed pretrained-SAE direction in the residual stream, with no learned selector and a single completion. Results are not directly comparable, so SIA is treated as related work (App. A).

## 3 Method
$\mathcal{M}$ is an $L$-layer decoder-only transformer, and $\mathbf{h}_t^{(\ell)}\in\mathbb{R}^d$ is the residual activation at layer $\ell$, position $t$ of $\mathbf{x}=(x_1,\ldots,x_T)$. We intervene at a single intermediate layer, so the signal propagates through the remaining layers:
$$\tilde{\mathbf{h}}_t^{(\ell)}=\mathbf{h}_t^{(\ell)}+\mathbf{s}_t\quad(1)$$

### 3.1 Steering Direction Discovery via SAE
The SAE at layer $\ell$ has encoder $f(\mathbf{h})\in\mathbb{R}^{d_{\mathrm{sae}}}$ and decoder $\mathbf{W}_{\mathrm{dec}}\in\mathbb{R}^{d_{\mathrm{sae}}\times d}$. Given a positive set $\mathcal{P}$ and a matched neutral set $\mathcal{N}$:
$$\Delta_i=\tfrac{1}{|\mathcal{P}|}\textstyle\sum_{x\in\mathcal{P}}f_i(x)-\tfrac{1}{|\mathcal{N}|}\sum_{x\in\mathcal{N}}f_i(x)\quad(2)$$
The top-$K$ features by $\Delta_i$ give $\mathbf{v}_k=\mathbf{W}_{\mathrm{dec},i_k}$, combined as $\mathbf{v}=\sum_k\mathbf{v}_k/\lVert\sum_k\mathbf{v}_k\rVert$.

Figure 1: STS/SBS behavioral gain over no steering vs. $p$. Dashed lines: prompting, full and no steering. Bands: std.

### 3.2 Stochastic Activation Steering
The signal is $\mathbf{s}_t=m_t\,\alpha\mathbf{v}$, with magnitude $\alpha\in\mathbb{R}$ and gate $m_t\in\{0,1\}$. Full Steering (FS) sets $m_t=1$ for all $t$.

#### Stochastic Token Steering (STS).
$$\tilde{\mathbf{h}}_t^{(\ell)}=\mathbf{h}_t^{(\ell)}+m_t\alpha\mathbf{v},\ m_t\overset{\text{i.i.d.}}{\sim}\mathrm{Bernoulli}(p)\quad(3)$$
STS is position-agnostic. $p$ interpolates between no intervention ($p{=}0$) and FS ($p{=}1$).

#### Stochastic Block Steering (SBS).
One gate $m\sim\mathrm{Bernoulli}(p)$ is drawn per sequence over a leading window $W\le T$:
$$\tilde{\mathbf{h}}_t^{(\ell)}=\begin{cases}\mathbf{h}_t^{(\ell)}+m\alpha\mathbf{v}&t\le W\\ \mathbf{h}_t^{(\ell)}&t>W\end{cases}\quad(4)$$
This exploits the outsized influence of initial tokens.

## 4 Experiments
#### Setup.
- **Tasks:** toxicity reduction (suppression, RealToxicityPrompts) and emotion steering (elicitation, GoEmotions mapped to Ekman basic emotions; targets fear and sadness).
- **Models:** LLaMA-3.1-8B and Gemma-2 2B, each with an SAE at an intermediate layer.
- **SBS windows:** $W\in\{32,64\}$.
- **Corrections (all methods):** norm preservation, activation clamping, decoding repetition penalty (App. C).

#### Baselines and Metrics.
- **Baselines:** No Steering; FS ($p{=}1$); Prompting with a curated instruction (App. B).
- **Effectiveness:** mean toxicity reduction (RoBERTa classifier) and mean target-emotion probability gain (DistilRoBERTa), both relative to no steering.
- **Quality:** GPT-2 perplexity; Rep-$n$, the fraction of recurring $n$-grams.

### 4.1 Sparse Steering Effectiveness
$\alpha$ is calibrated once at $p{=}1$ and held fixed across methods and ratios (App. D.1).

On LLaMA 3.1-8B, STS at $p{=}0.5$ recovers >95% of full-steering toxicity reduction, and by $p{=}0.7$ it recovers >75% of full-steering gains on both emotion tasks. Most dense per-token interventions are therefore redundant.

STS beats prompting in 5 of 6 model–task settings:
- by $p\approx0.5$ or below on toxicity (both models) and LLaMA fear;
- at $p\approx0.7$ on LLaMA sadness;
- not at any tested ratio on Gemma-2 2B sadness.

STS yields a larger shift than SBS at matched nominal $p$, clearest on emotion. On LLaMA fear, STS at $p{=}0.5$ roughly doubles the gain of SBS ($W{=}32$). The same ordering holds on Gemma-2 2B.

### 4.2 Fluency Under Sparse and Dense Steering
On Gemma-2 2B toxicity, perplexity, Rep-3 and Rep-4 rise monotonically with steering density, with FS at the high end. Sparse operating points therefore generally cost less fluency. The ratio required is task dependent (smaller for toxicity than emotion). The trend holds across SBS window sizes.

Figure 2: Perplexity and Rep-3/Rep-4 vs. ratio. Bands: std.

### 4.3 Steering Magnitude and Intervention Ratio
The best-performing $\alpha^*$ (maximal behavioral shift subject to fluency preservation) rises as $p$ falls, for both models. At fixed $\alpha$, the effect grows with $p$; the ratio needed for a target effect falls as $\alpha$ grows. Residual displacement corroborates this (App. D.3).

Figure 3: $\alpha^*$ vs. ratio (GoEmotions-fear); at each $p$, $\alpha^*$ maximizes probability gain subject to fluency preservation.

## 5 Conclusion
Training-free STS and SBS replace dense steering with sparse random application governed by $p$. A fraction of tokens recovers most of the effect with a better fluency–control tradeoff. Control need not act at every token.

## Limitations
- **Scope:** two tasks, two model families. Other targets (hallucination reduction, persona enforcement, multi-attribute control) remain unvalidated.
- **Evaluation:** classifier-based, so it may miss dimensions of behavioral change; no human evaluation.
- **SAE dependence:** requires a pretrained SAE with identifiable behavior-relevant features; quality depends on the dictionary and the contrastive corpora.
- **Decoding:** a single configuration. Interactions with higher temperatures, nucleus sampling, beam search, and between gate-level and sampling-level randomness are unexplored.

#### Intended use and dual-use considerations.
The method targets harm reduction and beneficial behavior shifts. Features could be steered inversely, e.g. to elicit toxicity, a concern shared by all activation steering. It needs white-box access and an SAE, so it enables no new attacks on black-box systems. We judge the net effect positive, especially where retraining is infeasible.

#### Use of sensitive data.
RealToxicityPrompts permits this research use. We collect and release no new toxic data beyond what the prompts elicit. s-nlp/roberta_toxicity_classifier has known biases against identity terms; scores are comparison metrics, not ground truth.

#### Qualitative examples.
App. B.2 gives verbatim toxic or distressing outputs, including failure modes, with content warnings. Identity-targeted slurs are excluded.

Content Warning. The appendix contains toxic, offensive, and emotionally distressing outputs.

## Appendix A Related Work
#### Prompt-Based Behavioral Control.
Prepended instructions are simple and architecture-agnostic but fragile. Outputs shift under minor lexical perturbation, and adversarial suffixes reliably override safety instructions.

#### Dense Activation Steering.
All of these apply the identified direction at every generated position:
- **ActAdd:** adds the mean activation difference of contrastive prompt pairs every forward pass.
- **ITI:** shifts activations of attention heads linearly predictive of truthfulness.
- **CAA:** averages over large prompt sets with layer-specific vectors.
- **Representation Engineering:** learns linear directions for concepts such as honesty.
- **DiffMean:** uses class-conditional mean differences, with fewer side effects.
- **Style Vectors:** control stylistic attributes.
- **In-Context Vectors:** extract directions from demonstrations.

#### Sparse Autoencoders for Interpretability and Steering.
SAEs decompose polysemantic activations into more monosemantic, human-interpretable features. GemmaScope provides open SAEs for every Gemma layer at multiple widths. Amplifying concept features (e.g., the Eiffel Tower) reliably steers LLaMA. AxBench benchmarks steering effectiveness and side effects, including SAE methods.

#### Detoxification.
All prior methods correct at every decoding step:
- **DExperts:** reweights tokens with expert/anti-expert LMs.
- **PPLM:** backpropagates through a toxicity classifier.
- **Self-Debiasing:** suppresses self-generated toxic probabilities.
- **GeDi:** uses class-conditional LMs as discriminators.

On RealToxicityPrompts all incur fluency cost. We use SAE directions and show a random token subset suffices.

#### Selective and Sparse Steering.
SIA (concurrent) is reward-guided decoding:
- finds "critical junctions" via entropy;
- distils trajectory rewards into a token-level value model $V^{\ast}$;
- reweights selected positions by $\exp(\beta V^{\ast})$;
- is evaluated by reward-model scoring against best-of-$N$ and chunk search.

Steering 20–80% of tokens beats dense intervention. SIA needs auxiliary model storage and repeated candidate generation; we use a fixed SAE direction, a single completion, and no guidance model. Protocols differ, so results are not commensurable.

Other work concentrates safety alignment or weak-to-strong steering in early tokens. SIA finds position-based gating competitive for safety at low ratios but degrading on helpfulness. This mirrors our SBS underperforming STS at matched nominal ratios; whether that persists at equal realized budgets is task-dependent (App. D.2). Random selection gives a training-free, position-agnostic control for evaluating selective methods.

#### Fluency–Alignment Trade-offs.
Large magnitudes cause disfluency, with a documented shift–coherence trade-off. Sparse application keeps the residual stream closer to its native distribution, yielding better fluency at matched shift.

## Appendix B Prompts and Qualitative Examples
### B.1 Prompts
Prompting baselines come from iterative engineering. For toxicity, variants ranged from simple ("Do not generate toxic content") to rule-based; we kept the one with lowest mean held-out toxicity and coherent output. For emotion, we varied phrasing, tone and directiveness, keeping the one maximizing held-out target probability. Baselines are intentionally strong.

### B.2 Qualitative Examples
Qualitative examples are presented.

## Appendix C Implementation Details
**SAEs:** eleutherai/sae-llama-3.1-8b-32x for LLaMA 3.1 8B; google/gemma-scope-2b (65K width) for Gemma-2 2B. Both expand the residual dimension and are applied at the feed-forward sublayer of one layer only.

**Features:** top-10 by $\Delta_i$ (Eq. 2).
- **Toxicity:** 600 prompts with toxicity $\geq0.6$ and 300 with $<0.2$. 300 toxic + 300 low-toxicity form the discovery set; the remaining toxic prompts are for evaluation (detoxify continuations).
- **Emotion:** 1k neutral, 300 fear and 300 sadness samples. Per emotion, 300 emotion + 300 neutral form the discovery set; the remaining neutral samples are for evaluation.

**Decoding:** temperature 0.1, up to 128 tokens, repetition penalty 1.1. Steered feature activations are clamped to $[0,40]$. SBS $W\in\{32,64\}$. Compute: eight NVIDIA L4/T4 GPUs.

**Inputs:** emotion prompts are truncated to 50% of tokens so outputs reflect steering, not prompt emotion. Toxicity prompts are kept whole, since truncation could remove the toxic tokens that make the task hard.

**Classifiers:** s-nlp/roberta_toxicity_classifier; j-hartmann/emotion-english-distilroberta-base (Ekman's six classes + neutral).

Figure 4: Preserve ON vs. OFF; perplexity and Rep-3/4 with std error bars.
Figure 5/6: $\alpha$ sweeps at $p{=}1$ for Gemma-2 2B / LLaMA 3.1-8B (toxicity, fear, sadness). Effectiveness saturates; perplexity and repetition degrade sharply beyond the chosen point.

### C.1 Fluency-Preserving Regularization Details
Additive steering can push states off the natural-activation manifold in norm or direction, raising perplexity and repetition. The guards keep activations near the native regime without weakening the effect.

#### Norm Preservation.
$$\tilde{\mathbf{h}}_t^{(\ell)}\leftarrow\tilde{\mathbf{h}}_t^{(\ell)}\cdot\|\mathbf{h}_t^{(\ell)}\|/\|\tilde{\mathbf{h}}_t^{(\ell)}\|\quad(5)$$
This rotates toward $\mathbf{v}$ without inflating magnitude.

#### Activation Clamping.
$$s_t=\tilde{\mathbf{h}}_t^{(\ell)}\cdot\mathbf{v},\ \hat{s}_t=\mathrm{clamp}(s_t,\gamma_{\min},\gamma_{\max})\quad(6)$$
$$\tilde{\mathbf{h}}_t^{(\ell)}\leftarrow\tilde{\mathbf{h}}_t^{(\ell)}+(\hat{s}_t-s_t)\mathbf{v}\quad(7)$$
The upper bound prevents over-amplification; $\gamma_{\min}\ge0$ suppresses residual unsteered activation.

#### Repetition Penalty.
Steering suppresses high-probability continuations, inducing repetition. A standard decoding penalty discounts prior tokens without touching hidden states.

#### Effect of Fluency Preservation.
On a Gemma-2-2B toxicity subset, enabling preservation cuts perplexity from roughly 82 to 30 and Rep-3/Rep-4 by about 3×, with non-overlapping error bars. Steering is applied identically in both conditions. Preservation is on by default.

## Appendix D Additional Results
### D.1 Steering Magnitude Calibration
At $p{=}1$, effectiveness rises with $\alpha$ to saturation, then perplexity inflects sharply with spikes in Rep-3/4. We pick the point just below the inflection:
- LLaMA 3.1-8B: $\alpha=2,2.5,2$ (toxicity, fear, sadness).
- Gemma-2 2B: $\alpha=35,30,35$.

There is no per-$p$ re-tuning.

### D.2 Comparison at Matched Intervention Budget
The realized ratio is the empirical fraction of steered tokens (Gemma-2 2B). SBS's bounded window makes its realized ratio grow slower in $p$, so matched nominal $p$ gives SBS less intervention. At matched budgets no single ordering holds: STS leads on toxicity over much of the range, while SBS is competitive on fear.

| Method | $p$ | Realized Ratio | Toxicity Reduction | Fear Prob. Gain |
|---|---|---|---|---|
| STS | 0.1 | 0.108 | 0.122±0.021 | 0.012±0.010 |
| | 0.3 | 0.287 | 0.193±0.028 | 0.055±0.019 |
| | 0.5 | 0.492 | 0.237±0.030 | 0.108±0.026 |
| | 0.7 | 0.706 | 0.252±0.024 | 0.165±0.024 |
| | 0.9 | 0.869 | 0.263±0.019 | 0.251±0.021 |
| SBS ($W{=}64$) | 0.1 | 0.048 | 0.023±0.013 | 0.010±0.009 |
| | 0.3 | 0.149 | 0.068±0.024 | 0.052±0.020 |
| | 0.5 | 0.261 | 0.108±0.031 | 0.092±0.026 |
| | 0.7 | 0.379 | 0.158±0.036 | 0.113±0.029 |
| | 0.9 | 0.531 | 0.232±0.033 | 0.135±0.028 |
| SBS ($W{=}32$) | 0.1 | 0.023 | 0.020±0.011 | 0.008±0.008 |
| | 0.3 | 0.086 | 0.065±0.021 | 0.035±0.016 |
| | 0.5 | 0.134 | 0.105±0.029 | 0.050±0.021 |
| | 0.7 | 0.216 | 0.152±0.035 | 0.060±0.024 |
| | 0.9 | 0.274 | 0.230±0.033 | 0.070±0.023 |

Table 1: Gemma-2 2B; realized ratios shared across tasks.

### D.3 Residual Stream Shift
Figure 7: normalized steered–unsteered distance at the steered layer vs. $\alpha$, three ratios; band: std.

$p$ adds displaced positions, while $\alpha$ deepens per-token displacement. Since the direction aligns with the target feature, the two are substitutable: at fixed $p$, a modest $\alpha$ increase recovers the effect of denser intervention.