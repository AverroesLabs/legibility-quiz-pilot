# A Coin Flip Per Token: Bernoulli Sparse Steering of Large Language Models

**Nima Eshraghi** – The Vanguard Group, Inc. – leizerteran@vanguard.com  
**Lovedeep Gondara** – The Vanguard Group, Inc. – jithinpradeep@vanguard.com  
**Yuqing Huang** – The Vanguard Group, Inc. – tonexu@cs.toronto.edu  
**Sagarika Suresh** – The Vanguard Group, Inc. – fanny@cs.toronto.edu  
**Leizer Teran** – The Vanguard Group, Inc.  
**Jithin Pradeep** – The Vanguard Group, Inc.  
**Xiaotong (Tone) Xu** – University of Toronto  
**Fanny Chevalier** – University of Toronto  

---  

## Abstract  

Activation steering through sparse autoencoders (SAEs) can control large language models (LLMs) without task‑specific fine‑tuning, but dense per‑token intervention introduces constant perturbation that can harm fluency. We introduce Stochastic Token Steering (STS) and Stochastic Block Steering (SBS), where each token or a leading window is independently steered with probability p. Across two model families and two behavioral tasks, steering only a fraction of tokens recovers most of the dense steering behavioral effect while reducing fluency degradation. Across all experimental settings, the optimal steering magnitude increases as the intervention ratio decreases.

---  

## 1 Introduction  

Large language models require behavioral control for safety, persona alignment, and refusal adjustment. Prompt‑based steering is fragile; output distributions shift under minor lexical perturbations and adversarial suffixes. Activation steering injects a steering direction into an intermediate residual stream, but standard practice applies it at every generated token. We ask whether persistent intervention is necessary. We replace deterministic every‑token steering with stochastic gating and examine the trade‑off between behavioral strength and fluency.

We define Stochastic Token Steering (STS) as independent Bernoulli (p) gating of each token, and Stochastic Block Steering (SBS) as a single Bernoulli (p) decision applied to a fixed leading window. Both methods require no reward model or learned gating policy, acting only by adding an activation vector during inference.

---  

## 2 Related Work  

(Only retained brief reference to activation‑steering and sparse‑autoencoder based methods; all background and citations omitted.)

---  

## 3 Method  

Let \(\mathcal{M}\) be an \(L\)‑layer single‑head decoder‑only transformer.  
For a tokenized input \(\mathbf{x} = (x_1,\dots,x_T)\), let \(\mathbf{h}_t^{(\ell)} \in \mathbb{R}^d\) denote the residual‑stream activation at layer \(\ell\) and position \(t\).  
We inject a steering signal at a single intermediate layer \(\ell\):

\[
\tilde{\mathbf{h}}_t^{(\ell)} = \mathbf{h}_t^{(\ell)} + \mathbf{s}_t \quad\text{(1)}
\]

The steering signal \(\mathbf{s}_t \in \mathbb{R}^d\) is defined in §3.2.

### 3.1 Steering Direction Discovery via SAE  

A Sparse Autoencoder (SAE) is trained on residual‑stream activations at layer \(\ell\).  
Encoder \(f(\mathbf{h}) \in \mathbb{R}^{d_{\text{sae}}}\) maps \(\mathbf{h}\) to sparse features; decoder \(\mathbf{W}_{\text{dec}}\in \mathbb{R}^{d_{\text{sae}}\!\times d}\) reconstructs \(\mathbf{h}\).  

Using a positive set \(\mathcal{P}\) exhibiting target behavior and a neutral set \(\mathcal{N}\), we compute the mean differential of feature \(i\):

\[
\Delta_i = \frac{1}{|\mathcal{P}|} \sum_{x\in\mathcal{P}} f_i(x) - \frac{1}{|\mathcal{N}|} \sum_{x\in\mathcal{N}} f_i(x) \quad\text{(2)}
\]

Select the top‑\(K\) features by \(\Delta_i\), form feature directions \(\mathbf{v}_k = \mathbf{W}_{\text{dec},i_k}\) for \(k=1,\dots,K\), and aggregate to a unit‑norm steering direction

\[
\mathbf{v} = \frac{\sum_{k=1}^{K}\mathbf{v}_k}{\Bigl\lVert\sum_{k=1}^{K}\mathbf{v}_k\Bigr\rVert} .
\]

### 3.2 Stochastic Activation Steering  

The signal at position \(t\) is

\[
\mathbf{s}_t = m_t \cdot \alpha \, \mathbf{v},
\]

where \(\alpha \in \mathbb{R}\) is the intervention magnitude, \(\mathbf{v}\) the unit‑norm direction, and \(m_t \in \{0,1\}\) a binary gate.

*Full Steering (FS)* sets \(m_t = 1\) for all positions.  

#### Stochastic Token Steering (STS)  

Sample independent gates \(m_t \sim \text{Bernoulli}(p)\) for each position:

\[
\tilde{\mathbf{h}}_t^{(\ell)} = \mathbf{h}_t^{(\ell)} + m_t \, \alpha \mathbf{v}, \qquad
m_t \stackrel{\text{i.i.d.}}{\sim} \text{Bernoulli}(p) \quad\text{(3)}
\]

### Stochastic Block Steering (SBS)  

Sample a single gate \(m \sim \text{Bernoulli}(p)\) per sequence and apply it to a leading window of size \(W \le T\):

\[
\tilde{\mathbf{h}}_t^{(\ell)} =
\begin{cases}
\mathbf{h}_t^{(\ell)} + m \, \alpha \mathbf{v}, & t \le W,\\
\mathbf{h}_t^{(\ell)}, & t > W,
\end{cases} \quad\text{(4)
}
\]

where \(m\) is constant across the window.

---  

## 4 Experiments  

### Setup  

We evaluate on two tasks: toxicity reduction (RealToxicityPrompts) and emotion steering (GoEmotions). Targets: fear and sadness, mapped from fine‑grained Ekman labels. We use LLaMA‑3.1‑8B (EleutherAI/sae‑llama‑3.1‑8b‑32x11) and Gemma‑2‑2B (google/gemma‑scope‑2b, dictionary width 65 K).  

For each model we intervene at a single intermediate layer. SBS window sizes \(W = 32, 64\).  

Steering magnitudes \(\alpha\) are calibrated once under full steering (\(p = 1\)) and then held fixed across all stochastic settings (Figure 5, 6).  

Decoding uses temperature 0.1, repetition penalty 1.1, maximum length 128 tokens. Activation clamping to \([0,40]\) is applied. Norm preservation and repetition penalty constitute fluency‑preserving regularization (Appendix C.1).  

Each experiment uses 300 prompts for emotion discovery, 600 toxic prompts for toxicity discovery, and 300 neutral prompts for emotion evaluation.  

### Baselines and Metrics  

*No Steering* – unmodified generation.  
*Full Steering* – intervention at every token, \(p=1\).  
*Prompting* – task‑specific safety or emotion instruction prepended (Appendix B).  

Behavioral effectiveness: mean toxicity reduction (difference in RoBERTa toxicity classifier scores) and mean target‑emotion probability gain (DistilRoBERTa emotion classifier).  
Fluency: GPT‑2 perplexity and repetition scores (Rep‑3, Rep‑4).

### 4.1 Sparse Steering Effectiveness  

Figure 1 shows behavioral gain over no steering versus intervention ratio \(p\).  

Table 1 (below) reports the behavioral shift for LLaMA‑3.1‑8B and Gemma‑2‑2B under matched intervention ratios. For LLaMA‑3.1‑8B, STS with \(p=0.5\) recovers over 95 % of full‑steering toxicity reduction; \(p=0.7\) recovers over 75 % of full‑steering gains on both emotion tasks. STS outperforms prompting in five of six model–task settings; for fear on LLaMA‑3.1‑8B, STS overtakes prompting already by \(p \approx 0.5\). SBS produces smaller behavioral shifts at equivalent \(p\).  

### 4.2 Fluency Under Sparse and Dense Steering  

Figure 2 reports perplexity and Rep‑3, Rep‑4 versus intervention ratio for Gemma‑2‑2B (toxicity). Both metrics increase monotonically as more tokens are steered; dense steering (\(p=1\)) lies at the high end.  

Sparse intervention ratios generally incur lower fluency cost than dense steering while achieving comparable behavioral shift. SBS yields similar fluency to STS but with reduced behavioral effect at matched \(p\).

### 4.3 Steering Magnitude and Intervention Ratio  

Figure 3 plots the best‑performing steering magnitude \(\alpha^{*}\) (maximizing behavioral shift with fluency preservation) against the intervention ratio. The optimal \(\alpha^{*}\) rises as \(p\) falls across both models and tasks. At a fixed \(\alpha\), behavioral effect increases with \(p\); conversely, a larger \(\alpha\) can compensate for sparser intervention.

### Numbers Appearing in Results  

Across intervention ratios, reported gains ranged from 0.009 to 0.036, covering values 0.009, 0.011, 0.013, 0.016, 0.019, 0.021, 0.024, 0.026, 0.028, 0.029, 0.030, 0.031, 0.033, 0.036.  

Token counts of 6, 7, 10, 20, 30, 35, 40, 50, 65, 80, 82 appear in the dataset and window size descriptions. Temperature values 0.2 and 0.6 were also used in ablations, as were 2.0 and 2.5 for steering magnitudes.  

---  

## 5 Conclusion  

STS and SBS provide training‑free, inference‑time interventions that replace dense per‑token activation steering with sparse random application controlled by a single parameter \(p\). Across two model families and both suppression and elicitation tasks, steering a fraction of tokens recovers most of the dense steering behavioral shift while improving the fluency‑control trade‑off. The best‑performing steering magnitude grows as intervention ratio falls.  

---  

## Limitations  

Evaluation is limited to toxicity reduction and emotion steering on LLaMA‑3.1‑8B and Gemma‑2‑2B. Generalization to other behaviors (hallucination, persona enforcement, multi‑attribute control) remains untested. Classifier‑based behavior assessment may not capture all dimensions; human evaluation would strengthen findings. SAE‑based steering relies on pretrained dictionaries and contrastive corpora; steering quality depends on dictionary quality. Experiments use a single decoding configuration; interactions with higher temperatures, nucleus sampling, or beam search are unexplored.

---  

## Intended use and dual‑use considerations  

The method offers training‑free inference‑time behavioral control aimed at reducing harmful generation and encouraging beneficial behaviors. Inverting the steering direction could amplify harmful behaviors; however, this requires white‑box access to model activations and a pretrained SAE, and cannot be applied to deployed black‑box systems.  

---  

## Use of sensitive data  

Toxicity reduction experiments use RealToxicityPrompts, a dataset released under a license permitting research use. No new toxic content is generated or released. Toxicity assessments use the s‑nlp/roberta_toxicity_classifier, acknowledging known biases against identity‑related text.  

---  

## Appendix A Related Work  

Key lines of activation‑steering (ActAdd, ITI, CAA, Representation Engineering) and sparse‑autoencoder steering are acknowledged; detailed comparison omitted for brevity.  

---  

## Appendix B Prompts and Qualitative Examples  

### B.1 Prompts  

Prompt engineering followed an iterative process; the final instruction exemplars are listed in the supplementary material (not reproduced here).  

### B.2 Qualitative Examples  

Examples of toxic, offensive, and emotionally distressing outputs are provided under a content warning in the full paper.  

---  

## Appendix C Implementation Details  

Salient implementation choices:  
- SAE for LLaMA‑3.1‑8B: eleutherai/sae‑llama‑3.1‑8b‑32x11; Gemma‑2‑2B: google/gemma‑scope‑2b, dictionary width 65 K.  
- Encoder produces sparse feature vector; top‑10 features selected via Eq. (2).  
- For toxicity: 600 highly toxic prompts (toxicity ≥ 0.6) and 300 low‑toxicity prompts (toxicity < 0.2). 300 toxic prompts reserved for discovery, 300 for evaluation.  
- For emotion: 1 k neutral prompts, 300 fear, 300 sadness; 300 neutral prompts reserved for discovery, 300 neutral for evaluation.  
- Generation: temperature 0.1, up to 128 tokens, repetition penalty 1.1.  
- Activation clamping: \([\;0,\,40\;]\).  
- SBS window sizes: 32, 64.  
- Eight NVIDIA L4 or T4 GPUs used for computation.  

### C.1 Fluency‑Preserving Regularization  

- Norm preservation: scale altered activation to original norm (Eq. (5)).  
- Activation clamping: project onto steering direction and clamp scalar projection to \([\gamma_{\min},\,\gamma_{\max}]\) (Eqs. (6)–(7)).  
- Repetition penalty: standard discount of previously generated tokens.  

Figure 4 (not reproduced) confirms that enabling preservation reduces perplexity from ~82 to ~30 and cuts replication by ~3×.  

---  

## Appendix D Additional Results  

### D.1 Steering Magnitude Calibration  

Steering magnitude \(\alpha\) calibrated once under full steering (\(p=1\)). Optimal \(\alpha\) per task:  
- LLaMA‑3.1‑8B: toxicity 2.0, fear 2.5, sadness 2.0.  
- Gemma‑2‑2B: toxicity 35, fear 30, sadness 35.  

These values were held fixed across all stochastic settings.  

### D.2 Comparison at Matched Intervention Budget  

Realized intervention ratios differ between STS and SBS. Table 1 (below) lists behavioral shift for Gemma‑2‑2B at matched budgets.  

| Method | \(p\) | Realized Ratio | Toxicity Reduction | Fear Probability Gain |
|--------|-------|----------------|--------------------|-----------------------|
| STS | 0.1 | 0.108 | 0.122 ± 0.021 | 0.012 ± 0.010 |
| STS | 0.3 | 0.287 | 0.193 ± 0.028 | 0.055 ± 0.019 |
| STS | 0.5 | 0.492 | 0.237 ± 0.030 | 0.108 ± 0.026 |
| STS | 0.7 | 0.706 | 0.252 ± 0.024 | 0.165 ± 0.024 |
| STS | 0.9 | 0.869 | 0.263 ± 0.019 | 0.251 ± 0.021 |
| SBS ( W = 64 ) | 0.1 | 0.048 | 0.023 ± 0.013 | 0.010 ± 0.009 |
| SBS ( W = 64 ) | 0.3 | 0.149 | 0.068 ± 0.024 | 0.052 ± 0.020 |
| SBS ( W = 64 ) | 0.5 | 0.261 | 0.108 ± 0.031 | 0.092 ± 0.026 |
| SBS ( W = 64 ) | 0.7 | 0.379 | 0.158 ± 0.036 | 0.113 ± 0.029 |
| SBS ( W = 64 ) | 0.9 | 0.531 | 0.232 ± 0.033 | 0.135 ± 0.028 |
| SBS ( W = 32 ) | 0.1 | 0.023 | 0.020 ± 0.011 | 0.008 ± 0.008 |
| SBS ( W = 32 ) | 0.3 | 0.086 | 0.065 ± 0.021 | 0.035 ± 0.016 |
| SBS ( W = 32 ) | 0.5 | 0.134 | 0.105 ± 0.029 | 0.050 ± 0.021 |
| SBS ( W = 32 ) | 0.7 | 0.216 | 0.152 ± 0.035 | 0.060 ± 0.024 |
| SBS ( W = 32 ) | 0.9 | 0.274 | 0.230 ± 0.033 | 0.070 ± 0.023 |

### D.3 Residual Stream Shift  

Figure 7 (not reproduced) shows that both increasing \(p\) and increasing \(\alpha\) enlarge the displacement of the steered residual activation. The two factors act as substitutes: a modest increase in \(\alpha\) at a fixed \(p\) can recover the behavioral effect of denser intervention.

---  

## Table 1 – Behavioral Shift at Matched Intervention Budget (Gemma‑2‑2B)  

| Method | \(p\) | Realized Ratio | Toxicity Reduction | Fear Probability Gain |
|--------|-------|----------------|--------------------|-----------------------|
| STS | 0.1 | 0.108 | 0.122 ± 0.021 | 0.012 ± 0.010 |
| STS | 0.3 | 0.287 | 0.193 ± 0.028 | 0.055 ± 0.019 |
| STS | 0.5 | 0.492 | 0.237 ± 0.030 | 0.108 ± 0.026 |
| STS | 0.7 | 0.706 | 0.252 ± 0.024 | 0.165 ± 0.024 |
| STS | 0.9 | 0.869 | 0.263 ± 0.019 | 0.251 ± 0.021 |
| SBS ( W = 64 ) | 0.1 | 0.048 | 0.023 ± 0.013 | 0.010 ± 0.009 |
| SBS ( W = 64 ) | 0.3 | 0.149 | 0.068 ± 0.024 | 0.052 ± 0.020 |
| SBS ( W = 64 ) | 0.5 | 0.261 | 0.108 ± 0.031 | 0.092 ± 0.026 |
| SBS ( W = 64 ) | 0.7 | 0.379 | 0.158 ± 0.036 | 0.113 ± 0.029 |
| SBS ( W = 64 ) | 0.9 | 0.531 | 0.232 ± 0.033 | 0.135 ± 0.028 |
| SBS ( W = 32 ) | 0.1 | 0.023 | 0.020 ± 0.011 | 0.008 ± 0.008 |
| SBS ( W = 32 ) | 0.3 | 0.086 | 0.065 ± 0.021 | 0.035 ± 0.016 |
| SBS ( W = 32 ) | 0.5 | 0.134 | 0.105 ± 0.029 | 0.050 ± 0.021 |
| SBS ( W = 32 ) | 0.7 | 0.216 | 0.152 ± 0.035 | 0.060 ± 0.024 |
| SBS ( W = 32 ) | 0.9 | 0.274 | 0.230 ± 0.033 | 0.070 ± 0.023 |