#### Prompt-Based Behavioral Control  
Prompt engineering inserts natural-language instructions before input. It is architecture-agnostic but fragile: small lexical changes shift output distributions, and adversarial suffixes can override safety instructions. Thus deeper representational interventions are needed.

#### Dense Activation Steering  
All methods in this category apply corrections to the residual stream of every generated token.  
- **ActAdd** computes a steering vector as the difference between mean activations on contrastive prompt pairs and adds it to the residual stream at each forward pass.  
- **ITI** identifies attention heads that linearly predict truthfulness and shifts activations along these directions.  
- **CAA** refines contrastive steering by averaging over large prompt sets and applying layer‑specific vectors.  
- **Representation Engineering** learns linear directions associated with high‑level concepts such as honesty or harmlessness.  
- **DiffMean** improves post‑hoc steering by using the difference of class‑conditional means and shows reduced side effects compared to naive contrastive approaches.  
- **Style Vectors** extract steering directions for stylistic attributes and apply them at every decoding step.  
- **In‑Context Vectors** derive latent directions from in‑context demonstrations and apply them during generation.  
All assume a fixed steering direction applied at every token.

#### Sparse Autoencoders for Interpretability and Steering  
SAEs decompose dense activations into monosemantic features.  
- SAE training on residual stream activations recovers interpretable features aligned with human concepts.  
- **GemmaScope** provides SAE checkpoints for Gemma models at all layers and widths.  
- **Louapre (2024)** shows that amplifying an SAE feature linked to a concept (e.g., Eiffel Tower) steers LLaMA outputs toward that concept.  
- **AxBench** benchmarks activation steering, including SAE‑based methods, measuring effectiveness and side effects.  
Our work builds on SAE steering but does not intervene at every token.

#### Detoxification  
All detoxification methods correct the model at every decoding step.  
- **DExperts** re‑weights token probabilities using an expert/anti‑expert pair.  
- **PPLM** back‑propagates through a toxicity classifier to update hidden states.  
- **Self‑Debiasing** prompts toxic generation and suppresses resulting tokens.  
- **GeDi** employs class‑conditional models as generative discriminators.  
Evaluated by RealToxicityPrompts, all maintain some fluency cost. My method uses SAE‑derived directions and demonstrates that correcting only a random subset of tokens is sufficient.

#### Selective and Sparse Steering  
**SIA (Hu et al. 2026)** identifies “critical junctions” via model entropy and steers only then.  
- Trains a token‑level value model \(V^{\ast}\) by distilling trajectory‑level rewards.  
- Re‑weights token distribution by \(\exp(\beta V^{\ast})\) at selected positions.  
- Evaluated by reward‑model scoring against best‑of‑\(N\) and chunk‑search baselines.  
SIA obtains superior alignment‑efficiency trade‑offs when steering 20–80% of tokens.  
SIA’s method differs: it trains a separate value model, queries learned guidance each decoding step, and uses best‑of‑\(N\) for evaluation.  
Our method gates a fixed SAE direction at random tokens, produces a single completion without auxiliary guidance.  
Other studies (Qi et al. 2025a; Song et al. 2025) focus early tokens for safety alignment; SIA finds position‑based gating competitive for safety at low intervention ratios but degrades on helpfulness, matching our task‑dependence observation that position‑anchored SBS underperforms position‑agnostic STS at matched nominal ratios.  
Our random‑selection strategy offers a training‑free, position‑agnostic baseline.

#### Fluency‑Alignment Trade‑offs  
Inference‑time steering typically reduces fluency.  
- ActAdd large steering magnitudes produce disfluent text.  
- CAA systematically studies the trade‑off between behavioral shift and coherence loss.  
- Wu et al. 2025 provide standardized measurements across methods.  
We observe that sparse application of steering improves this trade‑off: fewer interventions keep the residual stream close to its native distribution, yielding better fluency for a given behavioral shift.