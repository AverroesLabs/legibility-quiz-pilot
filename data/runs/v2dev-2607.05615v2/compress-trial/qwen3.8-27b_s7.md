Dense activation steering methods intervene at every generated token; our method departs from this.

#### Prompt-Based Behavioral Control.

Prompt engineering prepends natural-language instructions to elicit desired properties. It is fragile: output distributions shift under minor lexical perturbations, and adversarial suffixes can override safety instructions.

#### Dense Activation Steering.

ActAdd computes a steering vector as the difference between mean activations on contrastive prompt pairs and adds it to the residual stream at every forward pass. ITI identifies attention heads linearly predictive of truthfulness and shifts activations along those directions. CAA averages over large prompt sets and applies layer-specific vectors. Representation Engineering learns linear directions in activation space corresponding to high-level concepts. DiffMean uses the difference of class-conditional means and shows reduced side effects versus naive contrastive approaches. Style Vectors extract and apply style-specific directions at every decoding step. In-Context Vectors extract latent directions from in-context demonstrations. All apply the steering direction at every generated position.

#### Sparse Autoencoders for Interpretability and Steering.

SAEs decompose dense, polysemantic activations into a dictionary of monosemantic features. SAEs trained on residual stream activations recover interpretable features. GemmaScope provides open SAE checkpoints for the Gemma model family across all layers at multiple dictionary widths. Amplifying SAE features associated with specific concepts can steer LLaMA outputs toward those concepts. AxBench provides standardized comparisons of steering effectiveness and side effects. Our work builds on SAE steering but departs from applying interventions at every token position.

#### Detoxification.

DExperts combines an expert and anti-expert language model to re-weight token probabilities at each position. PPLM backpropagates through a toxicity classifier to update hidden states during generation. Self-Debiasing prompts the model to generate toxic text and suppresses the resulting token probabilities. GeDi uses class-conditional language models as generative discriminators. RealToxicityPrompts is the standard benchmark; all such methods incur some fluency cost. Our approach uses SAE-derived directions and shows that applying the correction at only a random subset of tokens suffices.

#### Selective and Sparse Steering.

SIA identifies critical junctions via model entropy and steers only at those positions. It trains a token-level value model $V^{\ast}$ by distilling trajectory-level rewards, reweights the token distribution by $\exp(\beta V^{\ast})$ at selected positions, and is evaluated by reward-model scoring against best-of-$N$ and chunk-search baselines. SIA reports that steering 20–80% of tokens achieves superior alignment–efficiency trade-offs over dense intervention. SIA trains a separate value model, queries learned guidance during decoding, and uses best-of-$N$ selection, requiring auxiliary model storage and repeated candidate generation. Our method gates a fixed SAE direction in an intermediate residual stream and produces a single completion without an auxiliary guidance model. Their evaluation protocols differ, so results are not directly commensurable. Qi et al. argue safety alignment is concentrated in early token positions; Song et al. focus steering on initial tokens for weak-to-strong decoding. SIA finds position-based gating is competitive for safety at low intervention ratios but degrades on helpfulness, mirroring our finding that position-anchored SBS underperforms position-agnostic STS at matched nominal ratios. Whether this persists once realized budgets are equalized is task-dependent (Appendix D.2). Our random-selection result provides a training-free, position-agnostic control.

#### Fluency–Alignment Trade-offs.

Large steering magnitudes produce disfluent text. Sparse application can improve the trade-off: intervening at fewer positions keeps the residual stream closer to its native distribution, yielding better fluency at matched behavioral shift.