# A Coin Flip Per Token: Bernoulli Sparse Steering of Large Language Models

###### Abstract

STS and SBS are training-free Bernoulli gates for SAE steering; neither requires a reward model or a learned gating policy.

## 1 Introduction

We ask how much of the behavioral effect of dense steering remains when only a fraction of token positions are steered. We introduce Stochastic Token Steering (STS) and Stochastic Block Steering (SBS). The method is training-free, uses a single interpretable sparsity parameter $p$, and adds only an activation-vector addition during inference.

## 2 Related Work

SIA is treated as related work rather than an experimental baseline.

## 3 Method

Let $\mathcal{M}$ be an $L$-layer decoder-only transformer. For a tokenized input $\mathbf{x}=(x_{1},\ldots,x_{T})$, let $\mathbf{h}_{t}^{(\ell)}\in\mathbb{R}^{d}$ denote the residual-stream activation at layer $\ell$ and position $t$. We inject a steering signal into the residual stream at a single intermediate layer $\ell$:

\[
\tilde{\mathbf{h}}_{t}^{(\ell)}=\mathbf{h}_{t}^{(\ell)}+\mathbf{s}_{t},
\tag{1}
\]

where the steering signal $\mathbf{s}_{t}\in\mathbb{R}^{d}$ is defined in Sec. 3.2. We restrict the intervention to a single intermediate layer. Editing the residual stream there allows the signal to propagate through the remaining layers, so the model can integrate it before generating the next token.

### 3.1 Steering Direction Discovery via SAE

We use a Sparse Autoencoder (SAE) trained on residual-stream activations at layer $\ell$ to identify directions associated with a target behavior. The SAE encoder maps an activation $\mathbf{h}$ to a sparse feature vector $f(\mathbf{h})\in\mathbb{R}^{d_{\mathrm{sae}}}$, whose $i$-th coordinate $f_{i}(\mathbf{h})$ is the activation of learned feature $i$. The SAE decoder $\mathbf{W}_{\mathrm{dec}}\in\mathbb{R}^{d_{\mathrm{sae}}\times d}$ reconstructs $\mathbf{h}$ from these features. To isolate behavior-discriminative features, we encode a positive set $\mathcal{P}$ exhibiting the target behavior and a neutral set $\mathcal{N}$ of matched controls, and compute the mean activation differential of each feature:

\[
\Delta_{i}=\frac{1}{|\mathcal{P}|}\sum_{x\in\mathcal{P}}f_{i}(x)-\frac{1}{|\mathcal{N}|}\sum_{x\in\mathcal{N}}f_{i}(x).
\tag{2}
\]

We select the top-$K$ features by $\Delta_{i}$ and form feature directions $\mathbf{v}_{k}=\mathbf{W}_{\mathrm{dec},i_{k}}$ for $k=1,\ldots,K$, where $i_{k}$ indexes the $k$-th selected feature. We aggregate these into a single unit-norm steering direction

\[
\mathbf{v}=\sum_{k=1}^{K}\mathbf{v}_{k}/\bigl\lVert\sum_{k=1}^{K}\mathbf{v}_{k}\bigr\rVert,
\]

so that features are combined in proportion to how strongly they distinguish the target behavior.

### 3.2 Stochastic Activation Steering

The steering signal at position $t$ takes the form

\[
\mathbf{s}_{t}=m_{t}\cdot\alpha\,\mathbf{v},
\]

where $\alpha\in\mathbb{R}$ is the intervention magnitude, $\mathbf{v}$ is the unit-norm steering direction, and $m_{t}\in\{0,1\}$ is a binary gate that decides whether position $t$ is steered. The standard approach, Full Steering (FS), sets $m_{t}=1$ at every position. We instead treat the gate as a random variable, controlling the fraction of steered positions and thereby trading behavioral strength against fluency.

#### Stochastic Token Steering (STS).

At each position $t$, we sample an independent gate $m_{t}\sim\mathrm{Bernoulli}(p)$:

\[
\tilde{\mathbf{h}}_{t}^{(\ell)}=\mathbf{h}_{t}^{(\ell)}+m_{t}\cdot\alpha\,\mathbf{v},~m_{t}\stackrel{{\scriptstyle\text{i.i.d.}}}{{\sim}}\mathrm{Bernoulli}(p).
\tag{3}
\]

In expectation, a fraction $p$ of positions receive the signal while the rest are left unperturbed. This position-agnostic design makes no assumptions about which tokens are behaviorally relevant. The parameter $p$ interpolates smoothly between no intervention ($p{=}0$) and full steering ($p{=}1$).

#### Stochastic Block Steering (SBS).

We treat a leading prompt window of size $W\leq T$ as an atomic steering unit. A single gate $m\sim\mathrm{Bernoulli}(p)$ is sampled once per sequence:

\[
\tilde{\mathbf{h}}_{t}^{(\ell)}=
\begin{cases}
\mathbf{h}_{t}^{(\ell)}+m\cdot\alpha\,\mathbf{v}&t\leq W,\\[2.0pt]
\mathbf{h}_{t}^{(\ell)}&t>W,
\end{cases}
\tag{4}
\]

With probability $p$ the entire window is steered coherently; with probability $1{-}p$ it is left intact.

## 4 Experiments

#### Setup.

We evaluate two tasks spanning opposite intervention goals: toxicity reduction, a suppression task on RealToxicityPrompts, and emotion steering, an elicitation task on GoEmotions. Emotion steering maps GoEmotions annotations onto basic emotions and uses fear and sadness as targets. We use two open-weight models from distinct families and scales, LLaMA-3.1-8B and Gemma-2 2B. For each model, we steer with an SAE at an intermediate transformer layer. We evaluate SBS at window sizes $W\in\{32,64\}$. Additive steering can displace activations off the native activation manifold and induce incoherent or repetitive text, so we pair every steering method with norm preservation, activation clamping, and a repetition penalty.

#### Baselines and Metrics.

We compare STS and SBS against three baselines: No Steering (unmodified generation), Full Steering (intervention at every token, $p{=}1$), and Prompting (a carefully curated task-specific safety or emotion instruction prepended to the input; Appendix B.1). For steering effectiveness we report mean toxicity reduction, the decrease in classifier-predicted toxicity relative to no steering, and mean probability gain, the increase in target-emotion probability relative to unsteered generation. Toxicity is scored by a RoBERTa toxicity classifier and emotion by a DistilRoBERTa emotion classifier (Appendix C). For generation quality we report GPT-2 perplexity and repetition scores, where Rep-$n$ is the fraction of $n$-grams in a generation that recur.

### 4.1 Sparse Steering Effectiveness

Figure 1 reports behavioral gain over no steering for STS and SBS across intervention ratio $p$ on toxicity reduction, fear steering, and sadness steering, with dashed baselines for prompting, full steering, and no steering, and shaded standard-deviation bands. To ensure a fair comparison, we calibrate the steering magnitude $\alpha$ once under full steering ($p{=}1$) and hold it fixed across all methods and ratios (Appendix D.1). Most of the behavioral shift is recoverable at substantially reduced cost. For LLaMA-3.1-8B, STS at $p{=}0.5$ recovers over 95% of full-steering toxicity reduction and recovers over 75% of full-steering gains on both emotion tasks by $p{=}0.7$. This indicates that the majority of per-token interventions under dense steering are redundant. STS overtakes the prompting baseline in five of the six model–task settings. On toxicity reduction for both models and on fear steering for LLaMA-3.1-8B, it does so by $p\!\approx\!0.5$ or below, i.e. while leaving at least half of the tokens unperturbed. On sadness steering for LLaMA-3.1-8B, the crossover occurs only at higher ratios, $p\!\approx\!0.7$. The exception is sadness steering on Gemma-2 2B, where prompting remains ahead of STS across the ratios we test. STS yields a larger behavioral shift than SBS at matched nominal ratios $p$, most clearly on emotion steering: for fear on LLaMA-3.1-8B, STS at $p{=}0.5$ roughly doubles the probability gain of SBS ($W{=}32$). This ordering holds for Gemma-2 2B as well.

### 4.2 Fluency Under Sparse and Dense Steering

Figure 2 reports perplexity and repetition scores (Rep-3, Rep-4) against intervention ratio for Gemma-2 2B on toxicity reduction, with full steering as reference and shaded standard-deviation bands. Degradation grows monotonically with steering density: both perplexity and repetition rise as more positions are steered, and full steering sits at the high end of both metrics. This monotonicity makes sparsity useful: operating points below the full-steering ratio generally have lower fluency cost. The required ratio is task dependent, smaller for toxicity reduction than for emotion steering. Sparse operating points generally sit below the full-steering cost, and the trend holds across SBS window sizes, so it is not a method-specific artifact.

### 4.3 Steering Magnitude and Intervention Ratio

Figure 3 plots best-performing $\alpha^{*}$ for GoEmotions-fear against intervention ratio $p$ for both models; at each $p$, $\alpha^{*}$ maximizes target-emotion probability gain subject to fluency preservation. The best-performing steering magnitude tends to rise as the ratio falls. Empirically, at a fixed magnitude the behavioral effect increases with the steering intervention ratio, and conversely the ratio required to reach a target effect decreases as the per-token strength grows. Appendix D.3 shows that the normalized residual-stream displacement grows with $\alpha$ and $p$.

## 5 Conclusion

Effective inference-time control need not act at every token.

## Limitations

Our evaluation covers two behavioral tasks (toxicity reduction, emotion steering) on two model families. These span suppression and elicitation, but generalization to other steering targets, such as hallucination reduction, persona enforcement, and multi-attribute control, remains unvalidated. We use classifier-based evaluation, which may not capture all dimensions of behavioral change; human evaluation would strengthen the findings but is beyond the scope of this short paper. Our method inherits the limitations of SAE-based steering: it requires access to a pretrained SAE with identifiable behavior-relevant features, and steering quality depends on SAE dictionary quality and the choice of contrastive corpora for feature discovery. Our experiments use a single decoding configuration; how stochastic steering interacts with other decoding settings, such as higher temperatures, nucleus sampling, and beam search, remains unexplored, and the interaction between gate-level randomness and sampling-level randomness is a natural direction for future work.

#### Intended use and dual-use considerations.

This work develops a training-free method for inference-time behavioral control of language models, with the primary motivation of reducing harmful generation, such as toxicity, and shifting model behavior toward beneficial targets, such as emotional regulation. The same mechanism could be inverted to amplify harmful behaviors: SAE features identified for toxicity reduction could be steered in the opposite direction to elicit toxicity. This dual-use concern is shared by activation-steering methods. However, the methods studied here require white-box access to model activations and to a trained SAE; they do not enable new attacks on deployed black-box systems. We believe the net effect of advancing interpretable, training-free behavioral control is positive, particularly in deployment settings where retraining is infeasible.

#### Use of sensitive data.

Our toxicity reduction experiments use prompts and continuations from RealToxicityPrompts, a dataset constructed specifically for research on toxic language and released under terms that permit such use. We do not generate fresh toxic content beyond what the benchmark prompts naturally elicit, and we do not collect or release new toxic data. Toxicity classifications use s-nlp/roberta_toxicity_classifier. Automated toxicity classifiers carry known biases against text containing identity-related terms, which may inflate or deflate reported scores for affected subsets. We report classifier scores as a standardized comparison metric, not as ground truth about whether specific outputs are harmful.

#### Qualitative examples.

Appendix B.2 reproduces verbatim model outputs that include toxic, offensive, and emotionally distressing content, to provide an honest qualitative characterization of the method, including its failure modes. These examples are prefaced by content warnings in both the appendix and at the front of the paper. We have excluded examples containing identity-targeted slurs, even where the benchmark prompts would have elicited them, because the marginal scientific value of including them does not outweigh the harm of reproducing such content in print. A content warning advises reader discretion.

## Appendix A Related Work

#### Prompt-Based Behavioral Control.

Prompt-based steering is fragile: output distributions shift under minor lexical perturbation, adversarial suffixes override safety instructions, and steerability through prompting is asymmetric and bounded.

#### Dense Activation Steering.

Dense activation steering methods apply a steering direction at every generated token; our method applies the signal to a subset of token positions.

#### Sparse Autoencoders for Interpretability and Steering.

SAEs decompose dense, polysemantic activations into sparser, more monosemantic features, and selected decoder directions can be amplified to steer behavior.

#### Detoxification.

Our detoxification task uses RealToxicityPrompts and applies an SAE-derived direction to a random subset of tokens.

#### Selective and Sparse Steering.

SIA sparsifies logit-space intervention by identifying critical junctions via model entropy and steering only at those positions. It is a reward-guided decoding method: it trains a token-level value model $V^{\ast}$ by distilling trajectory-level rewards, reweights the token distribution by $\exp(\beta V^{\ast})$ at selected positions, and is evaluated by reward-model scoring against best-of-$N$ and chunk-search baselines. It reports that steering 20–80% of tokens achieves superior alignment–efficiency trade-offs over dense intervention. SIA trains a separate value model from trajectory rewards, queries learned guidance during decoding, and reports reward-model evaluation with best-of-$N$ selection. These components require auxiliary-model storage and repeated candidate generation. Our method instead gates a fixed SAE direction in an intermediate residual stream, requires no learned selector, and produces a single completion. Because the methods differ in intervention space, training, inference, and evaluation, their reported results are not directly comparable. SIA finds position-based gating competitive for safety alignment at low intervention ratios but degrading on helpfulness tasks, a task-dependence that mirrors SBS underperforming STS at matched nominal $p$; whether this persists once realized budgets are equalized is task-dependent (Appendix D.2). STS is a training-free, position-agnostic control for evaluating selective methods.

#### Fluency–Alignment Trade-offs.

Sparse application can improve the fluency–alignment tradeoff by keeping the residual stream closer to its native distribution.

## Appendix B Prompts and Qualitative Examples

### B.1 Prompts

For toxicity reduction, we tested multiple prompting formulations ranging from simple instructions to structured rule-based prompts, selecting the variant with the lowest mean toxicity across a held-out set of prompts while still producing coherent continuations. For emotion steering, we iterated over phrasing, tone, and level of directiveness, retaining the prompt that maximized target-emotion probability on a held-out set. The final prompts are strong best-effort prompt-only baselines.

### B.2 Qualitative Examples

Several qualitative examples are presented.

## Appendix C Implementation Details

For LLaMA-3.1-8B, we use eleutherai/sae-llama-3.1-8b-32x; for Gemma-2 2B, we use google/gemma-scope-2b with a dictionary width of 65K features. Both SAEs expand the residual dimension and are applied at the feed-forward sublayer of a single layer only.

We identify steering features via a contrastive procedure, selecting the top-10 SAE features ranked by mean activation differential $\Delta_{i}$ (Eq. 2). For toxicity reduction, we sample 600 highly toxic prompts (toxicity $\geq 0.6$) and 300 low-toxicity prompts (toxicity $<0.2$) from RealToxicityPrompts. From the 600 toxic prompts, we reserve 300 as the set for direction discovery, paired with the 300 low-toxicity prompts; the remaining 300 toxic prompts serve as the evaluation set, where the model is steered to detoxify its continuations. For emotion steering, we sample 1k neutral prompts, 300 samples expressing fear, and 300 samples expressing sadness from GoEmotions. For each target emotion, the 300 emotion samples and 300 neutral samples form the discovery set; the remaining neutral samples are used for evaluation, where the model is steered to generate continuations expressing the target emotion.

During generation, we set temperature to 0.1, generate up to 128 tokens, and apply a repetition penalty of 1.1. Steered feature activations are clamped to $[0,40]$. For SBS, we evaluate window sizes $W\in\{32,64\}$. The computational workload was supported by eight NVIDIA L4 and T4 GPUs.

For emotion steering, we truncate each neutral prompt to 50% of its original token length and ask the model to generate a continuation. For toxicity reduction, we retain the full input prompt for a more challenging detoxification setting.

Toxicity probability is measured by s-nlp/roberta_toxicity_classifier, and emotion classification uses j-hartmann/emotion-english-distilroberta-base, fine-tuned on six emotion classes plus neutral.

### C.1 Fluency-Preserving Regularization Details

The additive intervention can drift in norm or direction away from natural activations and degrade fluency, inflate perplexity, and cause repetition. We therefore use lightweight guards that keep the steered activation close to the model’s natural operating regime while leaving the intended behavioral effect intact.

#### Norm Preservation.

We rescale the steered activation to restore the norm of the original:

\[
\tilde{\mathbf{h}}_{t}^{(\ell)}\leftarrow\tilde{\mathbf{h}}_{t}^{(\ell)}\cdot\frac{\bigl\|\mathbf{h}_{t}^{(\ell)}\bigr\|}{\bigl\|\tilde{\mathbf{h}}_{t}^{(\ell)}\bigr\|}
\tag{5}
\]

This constrains the intervention to rotate the representation toward $\mathbf{v}$ without inflating its magnitude, preserving the scale invariants assumed by downstream computations.

#### Activation Clamping.

We clamp the scalar projection of the steered activation onto $\mathbf{v}$ to a range $[\gamma_{\min},\,\gamma_{\max}]$ and apply a residual correction:

\[
s_{t}=\tilde{\mathbf{h}}_{t}^{(\ell)}\cdot\mathbf{v},\quad\hat{s}_{t}=\mathrm{clamp}(s_{t},\;\gamma_{\min},\;\gamma_{\max})
\tag{6}
\]

\[
\tilde{\mathbf{h}}_{t}^{(\ell)}\leftarrow\tilde{\mathbf{h}}_{t}^{(\ell)}+(\hat{s}_{t}-s_{t})\,\mathbf{v}
\tag{7}
\]

The upper bound prevents over-amplification beyond the feature’s natural range, while the lower bound $\gamma_{\min}\geq 0$ actively suppresses any residual feature activation already present in the unsteered representation.

#### Repetition Penalty.

Steering can suppress high-probability continuations and cause repetitive sequences. We apply a standard repetition penalty at the decoding stage, discounting previously generated tokens without further modifying the hidden representations.

#### Effect of Fluency Preservation.

We ablate the guards on a subset of the Gemma-2-2B toxicity-reduction setting, comparing Preserve ON with Preserve OFF while holding all other settings fixed. Figure 4 reports the results, with error bars denoting standard deviation across generations. Enabling preservation reduces perplexity from roughly $82$ to $30$ and cuts Rep-3 and Rep-4 repetition by roughly a factor of three, with non-overlapping standard-deviation error bars in all three comparisons. The behavioral steering effect is applied identically in both conditions, so the improvements reflect the guards keeping generation on the natural manifold rather than weakening the steering signal. We enable fluency preservation by default in all reported experiments.

## Appendix D Additional Results

We first describe how the steering magnitude $\alpha$ is calibrated per model and task (Appendix D.1). We next compare STS and SBS at a matched intervention budget, controlling for the differing effective budgets realized at the same nominal $p$ (Appendix D.2), and probe their differing fluency by measuring induced residual-stream shift (Appendix D.3).

### D.1 Steering Magnitude Calibration

Before evaluating stochastic interventions, we calibrate the steering magnitude $\alpha$ for each model under full steering ($p{=}1$). Figure 5 is Gemma-2 2B and Figure 6 is LLaMA-3.1-8B; columns correspond to toxicity reduction, fear steering, and sadness steering. The figures show steering effectiveness, perplexity, and repetition scores as a function of $\alpha$ under full intervention. For both models, steering effectiveness increases with $\alpha$ up to a saturation region, beyond which generation quality degrades sharply: perplexity remains stable at low magnitudes but exhibits a sharp inflection as $\alpha$ grows, accompanied by spikes in Rep-3 and Rep-4, indicating loss of on-manifold generation. We select, per task, the operating point just below this inflection that maximizes steering effectiveness while keeping perplexity and repetition within the stable region: $\alpha=2,2.5,2$ for LLaMA-3.1-8B on toxicity reduction, fear, and sadness respectively, and $\alpha=35,30,35$ for Gemma-2 2B on the same tasks. Because $\alpha$ is calibrated once at $p{=}1$ and then held fixed across all stochastic settings, the sparsity results reported elsewhere do not benefit from any per-$p$ re-tuning of the magnitude.

### D.2 Comparison at Matched Intervention Budget

To compare methods on equal footing, we report the realized intervention ratio, the empirical fraction of generated tokens that are actually steered, alongside $p$ and the achieved behavioral effect. Table 1 reports this comparison on the toxicity and fear tasks for Gemma-2 2B.

| Method | $p$ | Realized Ratio | Toxicity Reduction | Fear Prob. Gain |
|---|---:|---:|---:|---:|
| STS | 0.1 | 0.108 | $0.122\pm 0.021$ | $0.012\pm 0.010$ |
|  | 0.3 | 0.287 | $0.193\pm 0.028$ | $0.055\pm 0.019$ |
|  | 0.5 | 0.492 | $0.237\pm 0.030$ | $0.108\pm 0.026$ |
|  | 0.7 | 0.706 | $0.252\pm 0.024$ | $0.165\pm 0.024$ |
|  | 0.9 | 0.869 | $0.263\pm 0.019$ | $0.251\pm 0.021$ |
| SBS ($W{=}64$) | 0.1 | 0.048 | $0.023\pm 0.013$ | $0.010\pm 0.009$ |
|  | 0.3 | 0.149 | $0.068\pm 0.024$ | $0.052\pm 0.020$ |
|  | 0.5 | 0.261 | $0.108\pm 0.031$ | $0.092\pm 0.026$ |
|  | 0.7 | 0.379 | $0.158\pm 0.036$ | $0.113\pm 0.029$ |
|  | 0.9 | 0.531 | $0.232\pm 0.033$ | $0.135\pm 0.028$ |
| SBS ($W{=}32$) | 0.1 | 0.023 | $0.020\pm 0.011$ | $0.008\pm 0.008$ |
|  | 0.3 | 0.086 | $0.065\pm 0.021$ | $0.035\pm 0.016$ |
|  | 0.5 | 0.134 | $0.105\pm 0.029$ | $0.050\pm 0.021$ |
|  | 0.7 | 0.216 | $0.152\pm 0.035$ | $0.060\pm 0.024$ |
|  | 0.9 | 0.274 | $0.230\pm 0.033$ | $0.070\pm 0.023$ |

Table 1: Behavioral shift at matched intervention budget on the toxicity and fear tasks (Gemma-2 2B). Realized ratios are shared across both tasks; toxicity is reported as toxicity reduction and fear as mean probability gain.

The two methods realize markedly different budgets at the same nominal $p$: because SBS steers a bounded leading window, its realized ratio grows slower in $p$ than that of STS. Comparisons at matched nominal $p$ therefore allocate less intervention to SBS. Once budgets are matched, no single ordering holds across tasks and operating points: STS retains an advantage on toxicity reduction over much of the range, while on fear elicitation the block-level variant is competitive at comparable realized budgets.

### D.3 Residual Stream Shift

We measure the shift that intervention induces in the residual stream at the steered layer, quantified as the distance between the steered and unsteered hidden states. Figure 7 plots normalized residual-stream displacement at the steered layer as a function of steering strength, for three intervention ratios, with the shaded band representing the standard deviation. Both increasing $p$ and increasing $\alpha$ enlarge this displacement, but they do so differently: $p$ adds more displaced positions, whereas $\alpha$ deepens the displacement at each intervened token. Since the injected direction is aligned with the target feature, the two act as substitutable routes to the same behavioral shift, so at a fixed $p$ a modest increase in $\alpha$ is sufficient to recover the effect of denser intervention.