# A Coin Flip Per Token: Bernoulli Sparse Steering of Large Language Models

###### Abstract

Activation steering via sparse autoencoders (SAEs) enables behavioral control of large language models without task-specific fine-tuning, but standard methods apply the steering signal at every generated token, incurring constant per-token perturbation that risks degrading fluency. We introduce Stochastic Token Steering (STS), which gates each token independently with probability $p$, and Stochastic Block Steering (SBS), which gates a leading window once per sequence; neither requires a reward model or learned gating policy. Across two model families and two behavioral tasks, steering only a fraction of tokens recovers most of the behavioral effect of dense steering while incurring less fluency degradation. Across our experimental settings, the best-performing steering magnitude consistently rises as the intervention ratio falls.

## 1 Introduction

Standard activation-steering implementations apply the selected direction at every generated token. We replace deterministic every-token intervention with stochastic gating and measure the behavioral and fluency effects across intervention ratios. Dense every-token intervention is largely unnecessary: across two model families and two task types spanning suppression (toxicity reduction) and elicitation (emotion steering), STS recovers most of the full-steering behavioral effect while steering only a fraction of tokens, and causes less degradation in fluency. The best-performing steering magnitude rises as the intervention ratio falls, revealing a consistent interaction between intervention frequency and per-token strength. The method reduces inference-time behavioral control to a single interpretable sparsity parameter $p$, requiring no reward model or learned gating policy, and adds only an activation vector addition during inference.

## 2 Related Work

Prior SAE- and activation-steering methods generally apply a fixed intervention at every generation step; we study its application to only a subset of token positions. The closest concurrent work, SIA, sparsifies logit-space intervention using a separately trained value model to select positions, requires auxiliary-model queries during decoding, and uses best-of-$N$ generation with reward-model scoring. Our method instead gates a fixed direction from a pretrained SAE at an intermediate residual stream, requires no learned selector, and generates a single completion with only a lightweight activation update when the gate fires. Given differences in intervention space, training, inference, and evaluation, results are not directly comparable (Appendix A).

## 3 Method

Let $\mathcal{M}$ be an $L$-layer decoder-only transformer. For a tokenized input $\mathbf{x}=(x_{1},\ldots,x_{T})$, let $\mathbf{h}_{t}^{(\ell)}\in\mathbb{R}^{d}$ denote the residual-stream activation at layer $\ell$ and position $t$. We inject a steering signal into the residual stream at a single intermediate layer $\ell$:

$$\tilde{\mathbf{h}}_{t}^{(\ell)}=\mathbf{h}_{t}^{(\ell)}+\mathbf{s}_{t}, \tag{1}$$

where $\mathbf{s}_{t}\in\mathbb{R}^{d}$ is defined in Sec. 3.2. Restricting the intervention to a single intermediate layer lets the signal propagate through the remaining layers.

### 3.1 Steering Direction Discovery via SAE

We use an SAE trained on residual-stream activations at layer $\ell$. Its encoder maps $\mathbf{h}$ to a sparse feature vector $f(\mathbf{h})\in\mathbb{R}^{d_{\mathrm{sae}}}$, and its decoder $\mathbf{W}_{\mathrm{dec}}\in\mathbb{R}^{d_{\mathrm{sae}}\times d}$ reconstructs $\mathbf{h}$. We encode a positive set $\mathcal{P}$ exhibiting the target behavior and a neutral set $\mathcal{N}$ of matched controls, and compute the mean activation differential of each feature:

$$\Delta_{i}=\frac{1}{|\mathcal{P}|}\sum_{x\in\mathcal{P}}f_{i}(x)-\frac{1}{|\mathcal{N}|}\sum_{x\in\mathcal{N}}f_{i}(x). \tag{2}$$

We select the top-$K$ features by $\Delta_{i}$, form directions $\mathbf{v}_{k}=\mathbf{W}_{\mathrm{dec},i_{k}}$, and aggregate them into a unit-norm steering direction $\mathbf{v}=\sum_{k=1}^{K}\mathbf{v}_{k}/\bigl\lVert\sum_{k=1}^{K}\mathbf{v}_{k}\bigr\rVert$.

Figure 1: Steering effectiveness of STS and SBS versus intervention ratio $p$, measured as behavioral gain over no steering. Dashed lines are static baselines (prompting, full- and no steering); shaded bands show standard deviation.

### 3.2 Stochastic Activation Steering

The steering signal at position $t$ is $\mathbf{s}_{t}=m_{t}\cdot\alpha\,\mathbf{v}$, where $\alpha\in\mathbb{R}$ is the intervention magnitude, $\mathbf{v}$ the unit-norm steering direction, and $m_{t}\in\{0,1\}$ a binary gate. Full Steering (FS) sets $m_{t}=1$ at every position; we instead treat the gate as random.

#### Stochastic Token Steering (STS).

At each position $t$, we sample an independent gate:

$$\tilde{\mathbf{h}}_{t}^{(\ell)}=\mathbf{h}_{t}^{(\ell)}+m_{t}\cdot\alpha\,\mathbf{v},\ m_{t}\stackrel{\text{i.i.d.}}{\sim}\mathrm{Bernoulli}(p). \tag{3}$$

In expectation, a fraction $p$ of positions receive the signal. The design is position-agnostic; $p$ interpolates between no intervention ($p{=}0$) and full steering ($p{=}1$).

#### Stochastic Block Steering (SBS).

We treat a leading prompt window of size $W\leq T$ as an atomic steering unit, sampling a single gate $m\sim\mathrm{Bernoulli}(p)$ once per sequence:

$$\tilde{\mathbf{h}}_{t}^{(\ell)}=\begin{cases}\mathbf{h}_{t}^{(\ell)}+m\cdot\alpha\,\mathbf{v}&t\leq W,\\[2.0pt]
\mathbf{h}_{t}^{(\ell)}&t>W,\end{cases} \tag{4}$$

With probability $p$ the entire window is steered coherently; with probability $1{-}p$ it is left intact.

## 4 Experiments

#### Setup.

We evaluate on toxicity reduction, a suppression task built on RealToxicityPrompts, and emotion steering, an elicitation task built on GoEmotions, mapping the fine-grained annotations onto Ekman's basic emotions and taking fear and sadness as targets. We use LLaMA-3.1-8B and Gemma-2 2B. For each model, we steer with an SAE at an intermediate transformer layer and evaluate SBS at window sizes $W\in\{32,64\}$. We pair every steering method with three corrections: norm preservation, activation clamping, and a repetition penalty at decoding (Appendix C).

#### Baselines and Metrics.

We compare STS and SBS against three baselines: (1) No Steering, (2) Full Steering ($p{=}1$), and (3) Prompting (a task-specific safety or emotion instruction prepended to the input, Appendix B). For effectiveness we report mean toxicity reduction (decrease in classifier-predicted toxicity relative to no steering) and mean probability gain (increase in target-emotion probability relative to unsteered generation); toxicity is scored by a RoBERTa toxicity classifier and emotion by a DistilRoBERTa emotion classifier. For generation quality we report GPT-2 perplexity and repetition scores, where Rep-$n$ is the fraction of $n$-grams in a generation that recur.

### 4.1 Sparse Steering Effectiveness

Figure 1 reports steering effectiveness across intervention ratios on three tasks (toxicity reduction, fear and sadness steering). We calibrate the steering magnitude $\alpha$ once under full steering ($p{=}1$) and hold it fixed across all methods and ratios (Appendix D.1). Most of the behavioral shift is recoverable at substantially reduced cost: for LLaMA 3.1-8B, STS at $p{=}0.5$ recovers over 95% of full-steering toxicity reduction and recovers over 75% of full-steering gains on both emotion tasks by $p{=}0.7$. STS overtakes the prompting baseline in five of the six model–task settings: on toxicity reduction (both models) and fear steering on LLaMA 3.1-8B by $p\!\approx\!0.5$ or below; on sadness steering for LLaMA 3.1-8B at $p\!\approx\!0.7$. The exception is sadness steering on Gemma-2 2B, where prompting remains ahead across the ratios tested. STS yields a larger behavioral shift than SBS at matched nominal ratios $p$, most clearly on emotion steering: for fear on LLaMA 3.1-8B, STS at $p{=}0.5$ roughly doubles the probability gain of SBS ($W{=}32$). This ordering holds for Gemma-2 2B as well.

### 4.2 Fluency Under Sparse and Dense Steering

Figure 2 reports perplexity and repetition (Rep-3, Rep-4) against intervention ratio for Gemma 2–2B on toxicity reduction, with full steering as reference. Degradation grows monotonically with density: both perplexity and repetition rise as more positions are steered, and full steering sits at the high end of both. Sparse operating points generally sit below the full-steering cost; the ratio required is task dependent (smaller for toxicity reduction than for emotion steering), and the trend holds across SBS window sizes.

Figure 2: Generation quality versus intervention ratio: perplexity and repetition scores (Rep-3, Rep-4), shaded bands show standard deviation. Both metrics rise with steering density.

### 4.3 Steering Magnitude and Intervention Ratio

Figure 3 plots the best-performing $\alpha^{*}$ (maximizing behavioral shift subject to fluency preservation) against the intervention ratio for both models. The best-performing steering magnitude tends to rise as the ratio falls. At a fixed magnitude the behavioral effect increases with the intervention ratio, and conversely the ratio required to reach a target effect decreases as the per-token strength grows. Appx. D.3 corroborates that normalized residual-stream displacement grows with both $\alpha$ and $p$.

Figure 3: Best-performing steering magnitude $\alpha^{*}$ versus intervention ratio (GoEmotions-fear). At each $p$, $\alpha^{*}$ maximizes target-emotion probability gain subject to fluency preservation.

## 5 Conclusion

We introduced STS and SBS, two training-free interventions that replace dense, every-token activation steering with sparse random application governed by a single parameter $p$. Across two model families and both suppression and elicitation tasks, steering a fraction of tokens recovers most of the full-steering behavioral shift while improving the fluency-control tradeoff, and the best-performing magnitude rises as $p$ falls.

## Limitations

Our evaluation covers two behavioral tasks on two model families; generalization to other steering targets (e.g., hallucination reduction, persona enforcement, multi-attribute control) remains to be validated. We use classifier-based evaluation, which may not capture all dimensions of behavioral change; human evaluation is beyond the scope of this paper. The method requires access to a pretrained SAE with identifiable behavior-relevant features, and steering quality depends on SAE dictionary quality and the choice of contrastive corpora. Our experiments use a single decoding configuration; interaction with other decoding settings (higher temperatures, nucleus sampling, beam search) and between gate-level and sampling-level randomness is unexplored.

#### Intended use and dual-use considerations.

The same mechanism could in principle be inverted to amplify harmful behaviors (e.g., steering toxicity features in the opposite direction), a dual-use concern shared by all activation-steering methods. However, the methods require white-box access to model activations and a trained SAE and do not enable new attacks on deployed black-box systems.

#### Use of sensitive data.

Toxicity experiments use RealToxicityPrompts; we do not generate fresh toxic content beyond what the benchmark prompts naturally elicit and do not collect or release new toxic data. Automated toxicity classifiers (s-nlp/roberta_toxicity_classifier) carry known biases against text containing identity-related terms, which may inflate or deflate reported scores for affected subsets; we report classifier scores as a standardized comparison metric, not ground truth.

#### Qualitative examples.

Appendix B.2 reproduces verbatim model outputs that include toxic, offensive, and emotionally distressing content, prefaced by content warnings. We have excluded examples containing identity-targeted slurs.

Content Warning. The appendix contains examples of toxic, offensive, and emotionally distressing model outputs. Reader discretion is advised.

## Appendix A Related Work

SIA trains a token-level value model $V^{\ast}$ by distilling trajectory-level rewards, reweights the token distribution by $\exp(\beta V^{\ast})$ at selected positions identified via model entropy, and is evaluated by reward-model scoring against best-of-$N$ and chunk-search baselines. It reports that steering 20–80% of tokens achieves superior alignment–efficiency trade-offs over dense intervention. SIA differs from our method in intervention space, training requirements, inference procedure, and evaluation protocol, so we treat it as related work rather than a baseline. SIA finds position-based gating competitive for safety alignment at low intervention ratios but degrading on helpfulness tasks, a task-dependence that mirrors our finding that position-anchored SBS underperforms position-agnostic STS at matched nominal ratios; whether this persists once realized budgets are equalized is task-dependent (Appendix D.2). Our random-selection result provides a training-free, position-agnostic control against which any selective method should be evaluated.

## Appendix B Prompts and Qualitative Examples

### B.1 Prompts

The prompting baselines were developed through iterative prompt engineering. For toxicity reduction, we selected the variant achieving the lowest mean toxicity on a held-out set while producing coherent continuations; for emotion steering, we retained the prompt that maximized target-emotion probability on a held-out set.

### B.2 Qualitative Examples

We now present several qualitative examples.

## Appendix C Implementation Details

For LLaMA 3.1 8B, we use eleutherai/sae-llama-3.1-8b-32x; for Gemma-2 2B, google/gemma-scope-2b with a dictionary width of 65K features. Both SAEs are applied at the feed-forward sublayer of a single layer only. We select the top-10 SAE features ranked by mean activation differential $\Delta_{i}$ (Eq. 2). For toxicity reduction, we sample 600 highly toxic prompts (toxicity $\geq 0.6$) and 300 low-toxicity prompts (toxicity $<0.2$) from RealToxicityPrompts; 300 toxic prompts paired with the 300 low-toxicity prompts form the discovery set, and the remaining toxic prompts are the evaluation set. For emotion steering, we sample 1k neutral prompts plus 300 fear and 300 sadness samples from GoEmotions; per target emotion, the 300 emotion samples and 300 neutral samples form the discovery set, and the remaining neutral samples are used for evaluation.

During generation, temperature is 0.1, up to 128 tokens are generated, and a repetition penalty of 1.1 is applied. Steered feature activations are clamped to $[0,40]$. SBS uses $W\in\{32,64\}$. Experiments ran on eight NVIDIA L4 and T4 GPUs.

For emotion steering, we truncate each neutral prompt to 50% of its original token length so the steered output reflects the intervention rather than residual emotional content. For toxicity reduction, we retain the full (already highly toxic) input prompt, since truncation risks removing the toxic tokens that make the task challenging.

Toxicity is measured by s-nlp/roberta_toxicity_classifier; emotion classification uses j-hartmann/emotion-english-distilroberta, fine-tuned on Ekman's six emotion classes plus neutral.

Figure 4: Effect of fluency-preserving regularization. With preservation enabled (Preserve ON), perplexity drops sharply (top) and repetition at the 3- and 4-gram level is markedly reduced (bottom) relative to Preserve OFF. Error bars denote standard deviation.

Figure 5: Steering magnitude ($\alpha$) sweep for Gemma-2 2B under full intervention ($p{=}1$). Columns: toxicity reduction, fear, sadness. Effectiveness saturates while perplexity and repetition degrade sharply beyond the selected operating point.

Figure 6: Steering magnitude ($\alpha$) sweep for LLaMA 3.1-8B under full intervention ($p{=}1$). Columns: toxicity reduction, fear, sadness.

### C.1 Fluency-Preserving Regularization Details

#### Norm Preservation.

We rescale the steered activation to restore the original norm:

$$\tilde{\mathbf{h}}_{t}^{(\ell)}\leftarrow\tilde{\mathbf{h}}_{t}^{(\ell)}\cdot\frac{\bigl\|\mathbf{h}_{t}^{(\ell)}\bigr\|}{\bigl\|\tilde{\mathbf{h}}_{t}^{(\ell)}\bigr\|} \tag{5}$$

#### Activation Clamping.

We clamp the scalar projection of the steered activation onto $\mathbf{v}$ to $[\gamma_{\min},\,\gamma_{\max}]$ and apply a residual correction:

$$s_{t}=\tilde{\mathbf{h}}_{t}^{(\ell)}\cdot\mathbf{v},\quad\hat{s}_{t}=\mathrm{clamp}(s_{t},\;\gamma_{\min},\;\gamma_{\max}) \tag{6}$$

$$\tilde{\mathbf{h}}_{t}^{(\ell)}\leftarrow\tilde{\mathbf{h}}_{t}^{(\ell)}+(\hat{s}_{t}-s_{t})\,\mathbf{v} \tag{7}$$

The upper bound prevents over-amplification; the lower bound $\gamma_{\min}\geq 0$ suppresses residual feature activation already present in the unsteered representation.

#### Repetition Penalty.

We apply a standard repetition penalty at decoding, discounting previously generated tokens without modifying hidden representations.

#### Effect of Fluency Preservation.

Toggling preservation while holding all else fixed (Gemma-2-2B toxicity reduction): enabling it reduces perplexity from roughly $82$ to $30$ and cuts repetition by roughly a factor of three across Rep-3 and Rep-4, with non-overlapping error bars. The steering effect is applied identically in both conditions. We enable fluency preservation by default in all reported experiments.

## Appendix D Additional Results

### D.1 Steering Magnitude Calibration

We calibrate $\alpha$ per model and task under full steering ($p{=}1$). Steering effectiveness increases with $\alpha$ up to saturation; perplexity is stable at low magnitudes but shows a sharp inflection as $\alpha$ grows, accompanied by Rep-3 and Rep-4 spikes. We select the operating point maximizing effectiveness while keeping perplexity and repetition just below this inflection: $\alpha=2,2.5,2$ for LLaMA 3.1-8B on toxicity reduction, fear, and sadness respectively, and $\alpha=35,30,35$ for Gemma-2 2B. Because $\alpha$ is calibrated once at $p{=}1$ and held fixed, sparsity results do not benefit from per-$p$ re-tuning.

### D.2 Comparison at Matched Intervention Budget

We report the realized intervention ratio (the empirical fraction of generated tokens actually steered) alongside $p$ and the achieved behavioral effect (Table 1, Gemma-2 2B). At the same nominal $p$, SBS realizes a lower budget than STS because it steers a bounded leading window. Once budgets are matched, no single ordering holds: STS retains an advantage on toxicity reduction over much of the range, while on fear elicitation SBS is competitive at comparable realized budgets.

| Method | $p$ | Realized Ratio | Toxicity Reduction | Fear Prob. Gain |
|---|---|---|---|---|
| STS | 0.1 | 0.108 | $0.122\pm 0.021$ | $0.012\pm 0.010$ |
| | 0.3 | 0.287 | $0.193\pm 0.028$ | $0.055\pm 0.019$ |
| | 0.5 | 0.492 | $0.237\pm 0.030$ | $0.108\pm 0.026$ |
| | 0.7 | 0.706 | $0.252\pm 0.024$ | $0.165\pm 0.024$ |
| | 0.9 | 0.869 | $0.263\pm 0.019$ | $0.251\pm 0.021$ |
| SBS ($W{=}64$) | 0.1 | 0.048 | $0.023\pm 0.013$ | $0.010\pm 0.009$ |
| | 0.3 | 0.149 | $0.068\pm 0.024$ | $0.052\pm 0.020$ |
| | 0.5 | 0.261 | $0.108\pm 0.031$ | $0.092\pm 0.026$ |
| | 0.7 | 0.379 | $0.158\pm 0.036$ | $0.113\pm 0.029$ |
| | 0.9 | 0.531 | $0.232\pm 0.033$ | $0.135\pm 0.028$ |
| SBS ($W{=}32$) | 0.1 | 0.023 | $0.020\pm 0.011$ | $0.008\pm 0.008$ |
| | 0.3 | 0.086 | $0.065\pm 0.021$ | $0.035\pm 0.016$ |
| | 0.5 | 0.134 | $0.105\pm 0.029$ | $0.050\pm 0.021$ |
| | 0.7 | 0.216 | $0.152\pm 0.035$ | $0.060\pm 0.024$ |
| | 0.9 | 0.274 | $0.230\pm 0.033$ | $0.070\pm 0.023$ |

Table 1: Behavioral shift at matched intervention budget on the toxicity and fear tasks (Gemma-2 2B). Realized ratios are shared across both tasks; toxicity is reported as toxicity reduction and fear as mean probability gain.

Figure 7: Normalized residual-stream displacement at the steered layer as a function of steering strength, for three intervention ratios. The shaded band represents the standard deviation.

### D.3 Residual Stream Shift

We measure the shift intervention induces in the residual stream at the steered layer, quantified as the distance between steered and unsteered hidden states (Figure 7). Both increasing $p$ and increasing $\alpha$ enlarge this displacement, but differently: $p$ adds more displaced positions, whereas $\alpha$ deepens the displacement at each intervened token. The two act as substitutable routes to the same behavioral shift, so at a fixed $p$ a modest increase in $\alpha$ is sufficient to recover the effect of denser intervention.