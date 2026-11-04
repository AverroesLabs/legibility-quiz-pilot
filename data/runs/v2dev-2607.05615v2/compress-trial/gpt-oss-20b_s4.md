#### Setup  
Evaluation tasks: toxicity reduction (RealToxicityPrompts) and emotion steering (GoEmotions). Emotion targets: fear and sadness. Models: LLaMA‑3.1‑8B and Gemma‑2‑2B. Steering is applied with a SAE at an intermediate transformer layer; SBS uses window sizes W∈{32,64}. Corrections: norm preservation, activation clamping, repetition penalty.  

#### Baselines and Metrics  
Comparisons: (1) No Steering, (2) Full Steering (p=1), (3) Prompting. Metrics: mean toxicity reduction, mean probability gain. Toxicity classifier: RoBERTa; emotion classifier: DistilRoBERTa. Generation quality: GPT‑2 perplexity, repetition scores Rep‑n.  

#### 4.1 Sparse Steering Effectiveness  
α calibrated under full steering (p=1) and fixed for all methods. LLaMA‑3.1‑8B: STS p=0.5 recovers >95 % of full‑steering toxicity reduction; STS p=0.7 recovers >75 % of full‑steering gains on fear and sadness. STS beats prompting in 5/6 model‑task settings. On LLaMA‑3.1‑8B: STS overtakes prompting on toxicity reduction and fear steering at p≈0.5 or lower; for sadness steering at p≈0.7. Exception: sadness steering on Gemma‑2‑2B, prompting remains ahead. STS produces larger behavioral shift than SBS at matched p. Example: fear on LLaMA‑3.1‑8B, STS p=0.5 roughly doubles SBS (W=32) probability gain; same trend for Gemma‑2‑2B.  

#### 4.2 Fluency Under Sparse and Dense Steering  
Perplexity and Rep‑3/Rep‑4 increase monotonically with steering density. Full steering has highest perplexity and repetition. Sparse operating points below full‑steering ratio achieve lower fluency cost. Required ratio is task dependent – smaller for toxicity reduction than for emotion steering. Trend holds for SBS window sizes.  

#### 4.3 Steering Magnitude and Intervention Ratio  
Best‑performing α* rises as intervention ratio p decreases. For a fixed α, behavioral effect grows with p; conversely, the p needed to reach a target effect falls as α increases. Normalized residual‑stream displacement grows with α and p.