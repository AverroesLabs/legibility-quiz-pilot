#### Setup.

We evaluate on two tasks with opposite intervention goals: toxicity reduction on RealToxicityPrompts (suppression) and emotion steering on GoEmotions (elicitation), with fear and sadness as targets following Ekman's basic-emotion mapping. Models are LLaMA-3.1-8B and Gemma-2 2B. Each model is steered with an SAE at an intermediate transformer layer, evaluating SBS at window sizes $W\in\{32,64\}$. Every steering method is paired with three corrections: norm preservation, activation clamping, and a decoding-time repetition penalty. Implementation details are in Appendix C.

#### Baselines and Metrics.

Baselines: (1) No Steering, (2) Full Steering (intervention at every token, $p{=}1$), (3) Prompting (task-specific safety or emotion instruction prepended; Appendix B). Steering effectiveness: mean toxicity reduction (decrease in classifier-predicted toxicity vs. no steering, scored by a RoBERTa toxicity classifier) and mean probability gain (increase in target-emotion probability vs. unsteered generation, scored by a DistilRoBERTa emotion classifier; Appendix C). Generation quality: GPT-2 perplexity and Rep-$n$ (fraction of $n$-grams recurring in a generation).

### 4.1 Sparse Steering Effectiveness

Figure 1 reports effectiveness across intervention ratios on toxicity reduction and emotion steering toward fear and sadness; $\alpha$ is calibrated once under full steering ($p{=}1$) and held fixed across all methods and ratios (Appendix D.1). For LLaMA 3.1-8B, STS at $p{=}0.5$ recovers over 95% of full-steering toxicity reduction and over 75% of full-steering gains on both emotion tasks by $p{=}0.7$. STS overtakes prompting in five of six model–task settings: by $p\approx0.5$ or below on toxicity reduction (both models) and fear steering (LLaMA 3.1-8B), and at $p\approx0.7$ for sadness on LLaMA 3.1-8B; prompting remains ahead on sadness steering for Gemma-2 2B. STS yields a larger behavioral shift than SBS at matched $p$: for fear on LLaMA 3.1-8B, STS at $p{=}0.5$ roughly doubles the probability gain of SBS ($W{=}32$); the ordering also holds for Gemma-2 2B.

### 4.2 Fluency Under Sparse and Dense Steering

Figure 2 reports perplexity and repetition (Rep-3, Rep-4) against intervention ratio for Gemma 2–2B on toxicity reduction, with full steering as reference. Both metrics rise monotonically with steering density, with full steering at the high end, so operating points below the full-steering ratio incur lower fluency cost. The required ratio is task dependent (smaller for toxicity reduction than emotion steering, Sec. 4.1), but the trend holds across SBS window sizes.

Figure 2: Generation quality versus intervention ratio: perplexity and repetition scores (Rep-3, Rep-4), shaded bands show standard deviation. Both metrics rise with steering density.

### 4.3 Steering Magnitude and Intervention Ratio

Figure 3 plots the best-performing $\alpha^{*}$ (maximizing behavioral shift subject to fluency preservation) against intervention ratio for both models. $\alpha^{*}$ tends to rise as the ratio falls: at fixed magnitude, behavioral effect increases with ratio, and the ratio required for a target effect decreases as per-token strength grows. Appendix D.3 corroborates that normalized residual-stream displacement grows with $\alpha$ and $p$.

Figure 3: Best-performing steering magnitude $\alpha^{*}$ versus intervention ratio (GoEmotions-fear). At each $p$, $\alpha^{*}$ maximizes target-emotion probability gain subject to fluency preservation.