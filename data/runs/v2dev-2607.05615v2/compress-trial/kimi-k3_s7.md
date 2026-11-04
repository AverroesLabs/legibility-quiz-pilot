#### Prompt-Based Behavioral Control.

Prompt engineering prepends natural-language instructions to elicit properties such as reduced toxicity or increased helpfulness. It is simple and architecture-agnostic but fragile: output distributions shift under minor lexical perturbations, and adversarial suffixes can override safety instructions.

#### Dense Activation Steering.

These methods intervene in the residual stream at every generated token. Activation Addition (ActAdd) computes a steering vector as the difference between mean activations on contrastive prompt pairs and adds it at every forward pass. Inference-Time Intervention (ITI) identifies attention heads linearly predictive of truthfulness and shifts activations along these directions. Contrastive Activation Addition (CAA) averages over large prompt sets with layer-specific vectors. Representation Engineering learns linear directions corresponding to concepts such as honesty or harmlessness. DiffMean uses the difference of class-conditional means, reducing side effects versus naive contrastive approaches. Style Vectors control stylistic attributes at every decoding step. In-Context Vectors extract latent directions from in-context demonstrations. All apply the steering direction at every generated position; our method departs from this.

#### Sparse Autoencoders for Interpretability and Steering.

Sparse autoencoders (SAEs) decompose dense, polysemantic activations into a dictionary of more monosemantic features aligning with human-understandable concepts. GemmaScope provides open SAE checkpoints for Gemma across all layers at multiple dictionary widths. Amplifying SAE features for specific concepts steers LLaMA outputs toward them. AxBench benchmarks activation steering methods, including SAE-based ones, on effectiveness and side effects. We build on SAE steering but do not intervene at every token.

#### Detoxification.

All existing detoxification methods correct at every decoding step. DExperts re-weights token probabilities using expert (non-toxic) and anti-expert (toxic) models. PPLM backpropagates through a toxicity classifier to update hidden states. Self-Debiasing suppresses probabilities of tokens generated under a toxic prompt. GeDi uses class-conditional language models as generative discriminators. The RealToxicityPrompts benchmark shows all such methods incur fluency cost. We instead use SAE-derived directions applied at only a random subset of tokens.

#### Selective and Sparse Steering.

The concurrent Sparse Inference-time Alignment (SIA) identifies "critical junctions" via model entropy and steers only there. It trains a token-level value model $V^{\ast}$ by distilling trajectory-level rewards, reweights the token distribution by $\exp(\beta V^{\ast})$ at selected positions, and is evaluated by reward-model scoring against best-of-$N$ and chunk-search baselines. Steering 20–80% of tokens gives superior alignment–efficiency trade-offs over dense intervention.

SIA differs from our method in intervention space, training requirements, and inference procedure: it trains a value model, requires auxiliary model storage and repeated candidate generation, and uses reward-model evaluation with best-of-$N$ selection. We gate a fixed SAE direction in an intermediate residual stream and produce a single completion without an auxiliary guidance model. The differing evaluation protocols make results not directly commensurable, so we treat SIA as related work, not a baseline.

Other work argues safety alignment concentrates in early tokens, and weak-to-strong decoding steers initial tokens. SIA finds position-based gating competitive for safety at low intervention ratios but degrading on helpfulness tasks—mirroring our finding that position-anchored SBS underperforms position-agnostic STS at matched nominal ratios. Whether this persists under equalized realized budgets is task-dependent (Appendix D.2). Our random-selection result is a training-free, position-agnostic control for evaluating selective methods.

#### Fluency–Alignment Trade-offs.

Inference-time steering degrades generation quality: large steering magnitudes produce disfluent text, with a documented trade-off between behavioral shift and coherence loss. Sparse application improves this trade-off: intervening at fewer positions keeps the residual stream closer to its native distribution, yielding better fluency at matched behavioral shift.