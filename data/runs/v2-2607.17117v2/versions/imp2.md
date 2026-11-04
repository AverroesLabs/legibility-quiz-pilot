# Persistent Sparse Autoencoders: Learning Feature-Specific Timescales in Language Model Representations

Haoyan Luo

Affiliation: University of Cambridge

  
Mateo Espinosa Zarlenga

Affiliation: University of Oxford

  
Mateja Jamnik

Affiliation: University of Cambridge

###### Abstract

Sparse autoencoders (SAEs) encode each token of a language model's activations independently. In standard SAE features, decoder-aligned signal outlasts activation, past activations reduce reconstruction error of later hidden states, and the duration of both varies across features. We introduce Persistent Sparse Autoencoders (Persistent SAEs), which attach a learned persistence coefficient to every feature and train it by reconstruction alone. Persistent SAEs remain competitive in reconstruction and learn a spectrum of timescales: fast features are locally interpretable, and slow features identify the current context. In prompt-injection monitoring, slow features preserve injection-related signal and remain causally effective over long contexts.

## 1 Introduction

Interpreting an LLM's output requires knowing what its hidden states represent and how that content evolves over a sequence [1, 2]. SAEs [3, 4] decompose hidden states into sparse features which, when concept-aligned, support inspection and intervention [5, 6, 7]. Causal transformer hidden states integrate preceding context, yet standard SAEs encode tokens independently [3, 8, 9], obscuring topic, style, or task state [10]. Temporal variants impose predefined partitions or decompositions of temporal roles [10, 11], fixing a priori which features carry temporal information.

Standard SAE features already exhibit temporal persistence (Figure 1): after activation, later hidden states remain aligned with the decoder direction, past activations aid reconstruction of later hidden states, and decay rates differ substantially across features. Persistent SAEs learn a coefficient $\lambda_{j}$ per feature; each feature's state decays at $\lambda_{j}$ and is updated by a token-wise sparse activation. Reconstruction gradients favour higher $\lambda_{j}$ when carrying a state forward aids later reconstruction and smaller $\lambda_{j}$ otherwise.

Persistent SAEs retain competitive reconstruction while learning a broad range of timescales without supervision. Most features are low- $\lambda_{j}$ local detectors; high- $\lambda_{j}$ features retain contextual information across the sequence and, in a prompt-injection case study [12], provide monitoring signals over long timescales. Contributions:

•

SAE features exhibit temporal persistence: after activation, the decoder direction remains elevated and past activations help predict subsequent states.

•

Persistent Sparse Autoencoders, learning one coefficient per sparse feature through reconstruction alone, with competitive reconstruction and a broad spectrum of timescales without a predefined partition of temporal roles.

•

Short-timescale (fast) features act as local detectors; long-timescale (slow) features retain contextual information; in prompt-injection monitoring, slow features support monitoring and steering over long contexts.

## 2 Background and Related Work

Sparse autoencoders. 
For hidden state $\mathbf{x}^{(t)}\in\mathbb{R}^{d}$ at token $t$ , a standard SAE [3] produces $\mathbf{f}^{(t)}\in\mathbb{R}_{\geq 0}^{m}$ and

| | $\hat{\mathbf{x}}^{(t)}_{\mathrm{base}}=\mathbf{W}_{\mathrm{dec}}\mathbf{f}^{(t)}+\mathbf{b}_{\mathrm{dec}},$
| | (1)

with $\mathbf{d}_{j}=\mathbf{W}_{\mathrm{dec}}[:,j]/\lVert\mathbf{W}_{\mathrm{dec}}[:,j]\rVert_{2}$ (notation in Appendix A) and feature $j$ active whenever $\mathbf{f}^{(t)}_{j}>0$ . Sparsity of $\mathbf{f}^{(t)}$ limits each reconstruction to few features [3, 13]. BatchTopK [8] replaces per-token TopK with a batch-level budget, letting the active count vary across tokens; Matryoshka SAEs [9] jointly train nested dictionaries of increasing size, smaller subsets retaining higher-level features.

Temporal structure in LLM representations. 
Standard SAEs and most variants compute $\mathbf{f}^{(t)}$ independently of preceding tokens' features. Slow feature analysis identifies higher-level features in neural networks via slowly varying latent factors [14, 15]. LMs track entities and encode game-board, spatial, temporal, and propositional states [16, 17, 18, 19].

Temporal extensions of SAEs. 
Temporal SAEs [10] pre-assign a high-level subset and align its activations across nearby tokens contrastively. Lubana et al. [11] use an attention-based predictor to separate context-predictable information from a sparse novel residual. Both fix a partition; neither assigns each sparse feature its own learned persistence, which complicates post-hoc timescale analysis. Persistent SAEs assign each feature a persistent state and a learned persistence coefficient.

Figure 1:
Standard SAE features exhibit unmodelled temporal structure.
(a) Natural Pile sequences preserve local and global continuity; shuffled control sequences, concatenating sentences from different documents, preserve only local continuity.
(b) $R_{j}(\tau)$ decays more slowly in natural than in shuffled sequences.
(c) Past activations improve held-out reconstruction of later hidden states in natural, not shuffled, sequences.
(d) At $\tau=1$ , fitted $w_{j}$ are predominantly positive for natural sequences and centred near zero for shuffled ones.

## 3 Does Temporal Structure Exist at the Feature Level?

Setup. 
Residual-stream hidden states $\{\mathbf{x}^{(t)}\}_{t=1}^{T}$ are taken from Pythia-160M-deduped (layer 8) [20] and Gemma-2-2B (layer 12) [21] on Pile documents [22]. BatchTopK [8] and Matryoshka [9] SAEs, neither with a temporal prior, are trained as in Appendix B. The shuffled control is built from intact sentences of different Pile documents (Figure 1 (a); Appendix C).

Sparse Features Temporally Persist in the Hidden States. 
With $p^{(t)}_{j}=\mathbf{d}_{j}^{\top}\mathbf{x}^{(t)}$ and lag $\tau$ in tokens,

| | $R_{j}(\tau):=\big({\mathbb{E}\!\left[p^{(t+\tau)}_{j}\mid\mathbf{f}^{(t)}_{j}>0\right]-\mathbb{E}\!\left[p^{(t)}_{j}\right]}\Big)\Big/\Big({\mathbb{E}\!\left[p^{(t)}_{j}\mid\mathbf{f}^{(t)}_{j}>0\right]-\mathbb{E}\!\left[p^{(t)}_{j}\right]}\Big),$
| | (2)

where $\mathbb{E}[p^{(t)}_{j}]$ is averaged over all evaluated tokens. Evaluation procedure and feature selection: Appendix D.

Across both models and SAE families, $R_{j}(\tau)$ decays more slowly in natural documents than in the shuffled control (Figure 1 (b)).

Past Activations Can Help Reconstruction. 
We augment Equation 1 with activations $\tau$ tokens earlier, restricted to each feature's own direction:

| | $\hat{\mathbf{x}}^{(t)}_{\mathrm{ctx}}=\hat{\mathbf{x}}^{(t)}_{\mathrm{base}}+\mathbf{W}_{\mathrm{dec}}\left(\mathbf{w}_{\tau}\odot\mathbf{f}^{(t-\tau)}\right),$
| | (3)

with $\mathbf{w}_{\tau}\in\mathbb{R}^{m}$ fitted per lag for 10 epochs under an $L_{2}$ penalty, and report on held-out data

| | $R^{2}_{\mathrm{improve}}=1-\Big({\sum_{t}\lVert\mathbf{x}^{(t)}-\hat{\mathbf{x}}^{(t)}_{\mathrm{ctx}}\rVert_{2}^{2}}\Big)\Big/\Big({\sum_{t}\lVert\mathbf{x}^{(t)}-\hat{\mathbf{x}}^{(t)}_{\mathrm{base}}\rVert_{2}^{2}}\Big).$
| | (4)

The control refits the weights on past activations from the shuffled corpus, keeping natural-document hidden states as targets (Section D.1).

Natural-document history recovers a fraction of baseline error, diminishing with lag; shuffled history does not (Figure 1 (c)). Fitted weights at $\tau=1$ are mostly positive for natural documents and centred near zero for the shuffled control (Figure 1 (d)).

Lubana et al. [11] show that hidden-state correlations vary across a sequence and that past hidden states explain much of the current one. The question here is narrower and feature-level: persistence of individual SAE features aids reconstruction and varies widely across features.

## 4 Persistent Sparse Autoencoders

Figure 2:
Persistent Sparse Autoencoders. Each feature integrates $\mathbf{a}^{(t)}$ into $\mathbf{h}^{(t)}$ with learned $\lambda_{j}$ .

A BatchTopK encoder (Section B.2) produces $\mathbf{a}^{(t)}\in\mathbb{R}_{\geq 0}^{m}$ , with $a^{(t)}_{j}>0$ marking new evidence for feature $j$ at token $t$ , and

| | $h^{(t)}_{j}=\lambda_{j}h^{(t-1)}_{j}+(1-\lambda_{j})a^{(t)}_{j},$
| | (5)

with $\lambda_{j}\in[0,1]$ , $\mathbf{h}^{(t)}\in\mathbb{R}_{\geq 0}^{m}$ , $\mathbf{h}^{(0)}=\mathbf{0}$ at every input sequence start, and $\lambda_{j}=\lambda_{\max}\sigma(l_{j})$ for unconstrained $l_{j}$ and $\lambda_{\max}=0.999$ (Section B.3). The factor $(1-\lambda_{j})$ keeps the state magnitude bounded. The decoder reads the state:

| | $\hat{\mathbf{x}}^{(t)}=\mathbf{W}_{\mathrm{dec}}\mathbf{h}^{(t)}+\mathbf{b}_{\mathrm{dec}}.$
| | (6)

The recurrence is diagonal in feature space, so attribution runs through each decoder direction. Sparsity is imposed on $\mathbf{a}^{(t)}$ ; $\mathbf{h}^{(t)}$ need not be sparse. No predefined partition or auxiliary temporal objective is used.

## 5 Experiments

Models and hyperparameters. 
Models and dataset follow Section 3. Baselines: BatchTopK SAE [8], Matryoshka SAE [9], and Temporal SAE [10]. All SAEs use $m=16{,}384$ and $k=20$ ; Matryoshka and Temporal share a predefined 20%/80% partition. Full configuration: Appendix B.

### 5.1 Reconstruction Quality and Learned Timescales

Learned $\lambda_{j}$ span a broad range, mostly short-timescale with a small highly persistent tail (Figure 3 (a)). Henceforth slow = top $1\%$ by $\lambda_{j}$ , fast = bottom $50\%$ .

Table 1 reports FVE, cosine similarity between hidden state and reconstruction, fraction of alive features, and the temporal smoothness metric of Bhalla et al. [10]. Persistent SAEs remain competitive with the strongest baselines in reconstruction; on both models, slow features are smoother than fast features and baseline SAEs.

Grouping features into $\lambda_{j}$ quintiles, median $R_{j}(\tau)$ is consistently ordered across lags on both models, higher- $\lambda_{j}$ groups decaying more slowly (Figure 3 (b)). The Temporal SAE's predefined high- and low-level groups are not clearly separated by empirical persistence.

Table 1:
SAE representation quality. High/Low: predefined splits for Matryoshka and Temporal SAEs; for Persistent SAEs, the largest 1% and smallest 50% coefficients.
Best results bold, second-best underlined.

|
| | FVE ( $\uparrow$ )
|

Cosine
Sim ( $\uparrow$ )

|

Fraction
Alive ( $\uparrow$ )

| Smoothness ( $\downarrow$ ) | Autointerp ( $\uparrow$ )

|
| | Full | High | Low | Full | High | Low

|

Pythia

| BatchTopK | $\mathbf{0.94{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.95{\scriptscriptstyle\pm.01}}$ | $\underline{0.91{\scriptscriptstyle\pm.01}}$ | $0.15{\scriptscriptstyle\pm.01}$ | – | – | $\mathbf{0.91{\scriptscriptstyle\pm.02}}$ | – | –

| Matryoshka | $0.92{\scriptscriptstyle\pm.01}$ | $\underline{0.94{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.96{\scriptscriptstyle\pm.01}}$ | $0.15{\scriptscriptstyle\pm.01}$ | $0.25{\scriptscriptstyle\pm.01}$ | $\underline{0.12{\scriptscriptstyle\pm.01}}$ | $\underline{0.87{\scriptscriptstyle\pm.02}}$ | – | –

| Temporal | $\underline{0.93{\scriptscriptstyle\pm.02}}$ | $\mathbf{0.95{\scriptscriptstyle\pm.01}}$ | $0.79{\scriptscriptstyle\pm.01}$ | $\underline{0.13{\scriptscriptstyle\pm.01}}$ | $\underline{0.18{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.11{\scriptscriptstyle\pm.01}}$ | $\underline{0.87{\scriptscriptstyle\pm.02}}$ | $\mathbf{0.87{\scriptscriptstyle\pm.01}}$ | $\underline{0.89{\scriptscriptstyle\pm.02}}$

| Persistent (Ours) | $\underline{0.93{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.95{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.96{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.10{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.05{\scriptscriptstyle\pm.01}}$ | $0.20{\scriptscriptstyle\pm.01}$ | $0.86{\scriptscriptstyle\pm.01}$ | $\underline{0.84{\scriptscriptstyle\pm.02}}$ | $\mathbf{0.93{\scriptscriptstyle\pm.01}}$

|

Gemma

| BatchTopK | $\mathbf{0.72{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.79{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.92{\scriptscriptstyle\pm.01}}$ | $0.16{\scriptscriptstyle\pm.01}$ | – | – | $\underline{0.87{\scriptscriptstyle\pm.03}}$ | – | –

| Matryoshka | $\underline{0.71{\scriptscriptstyle\pm.01}}$ | $\underline{0.77{\scriptscriptstyle\pm.01}}$ | $0.87{\scriptscriptstyle\pm.01}$ | $0.17{\scriptscriptstyle\pm.01}$ | $0.21{\scriptscriptstyle\pm.01}$ | $\mathbf{0.14{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.89{\scriptscriptstyle\pm.02}}$ | – | –

| Temporal | $0.62{\scriptscriptstyle\pm.02}$ | $0.75{\scriptscriptstyle\pm.01}$ | $0.75{\scriptscriptstyle\pm.01}$ | $\underline{0.14{\scriptscriptstyle\pm.01}}$ | $\underline{0.12{\scriptscriptstyle\pm.01}}$ | $\underline{0.16{\scriptscriptstyle\pm.01}}$ | $0.85{\scriptscriptstyle\pm.02}$ | $\mathbf{0.85{\scriptscriptstyle\pm.02}}$ | $\underline{0.86{\scriptscriptstyle\pm.02}}$

| Persistent (Ours) | $0.66{\scriptscriptstyle\pm.01}$ | $0.75{\scriptscriptstyle\pm.01}$ | $\underline{0.91{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.11{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.04{\scriptscriptstyle\pm.01}}$ | $0.20{\scriptscriptstyle\pm.01}$ | $0.85{\scriptscriptstyle\pm.01}$ | $\underline{0.79{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.89{\scriptscriptstyle\pm.01}}$

Figure 3:
(a) Distribution of $\lambda_{j}$ : many short-timescale, fewer long-timescale features. (b) Median $R_{j}(\tau)$ by $\lambda_{j}$ quintile across lags.

### 5.2 Short-Timescale Features Are Locally Interpretable

We apply the automated interpretability pipeline of Bills et al. [23] (a judge explains a feature from its highest-activation contexts and scores the explanation's predictive accuracy), scoring the Persistent SAE on $\mathbf{a}^{(t)}$ (Appendix E).

On both models fast features score higher than slow features and match or exceed the best baseline (Table 1, right). Fast features often respond to compact lexical or grammatical patterns such as letters or substrings, with activation concentrated at the current token (Tables 4 and 5 in Appendix E). Slow features more often reflect broader subjects or contextual conditions with activation distributed across the sequence; for such features distinct features may receive similar descriptions [24] and explanations may not generalise to related inputs [25]. Slow features are evaluated jointly in Section 5.3.

Where Features Receive New Evidence. 
Features are split into five equal-sized $\lambda$ quintiles and update positions into six categories: punctuation, conjunctions, function words, verbs, adjectives or adverbs, and nouns, following the lexical/function word distinction [26]; Func. (Figure 4) covers adpositions, determiners, particles, and pronouns. For quintile $q$ and category $c$ ,

| | $U_{q,c}={\Pr\!\left(T^{(t)}=c\mid a^{(t)}_{j}>0,\;j\in Q_{q}\right)}\big/{\Pr\!\left(T^{(t)}=c\right)},$
| | (7)

with $T^{(t)}$ the category of token $t$ and $Q_{q}$ the features in quintile $q$ .

Figure 4: $U_{q,c}$ per $\lambda$ -quintile and token category. Orange/blue: above/below corpus-level frequency.

The lowest- $\lambda$ quintile is updated more at nouns; the highest- $\lambda$ quintile more at punctuation and conjunctions (Figure 4). Punctuation and function-like tokens carry substantial contextual information in LLMs [27], and explicit structural markers gate cross-sentence integration [28]. Autointerpretations of high- $\lambda$ features often describe broad semantic domains (Tables 4 and 5).

### 5.3 Long-Timescale States Track Subject Information

Slow features form subject-specific clusters. 
On MMLU [29] examples from five subjects, the final $20$ token representations per sample are embedded with t-SNE, using the top $1\%$ slow features (Appendix F). BatchTopK features heavily mix subjects; the Temporal SAE's high-level group separates them only weakly; slow Persistent SAE features form coherent subject regions, more clearly separated than raw hidden states (Figure 5), learnt without subject labels.

Figure 5:
Token-level MMLU representations coloured by subject: high- $\lambda_{j}$ slow features versus baseline SAEs and raw hidden states.

Temporal Persistence Selects Subject-Informative Features Without Labels. 
On spliced MMLU sequences, concatenated labelled segments whose subject changes at boundaries, a per-token linear probe predicts the current subject from varying fractions of the dictionary. Persistent SAE feature rankings: decreasing $\lambda_{j}$ , increasing $\lambda_{j}$ , random. Comparisons: BatchTopK, Matryoshka, and Temporal SAE representations, and a supervised label-informed oracle ranking favouring features more active on a subject than on all others (Appendix G).

Decreasing- $\lambda_{j}$ selection is nearly as efficient as supervised selection; on both models the slowest $1\%$ reach high AUC, while low- $\lambda_{j}$ , random, and baseline SAE representations retain substantially less subject information at the same budget (Figure 6).

Figure 6:
Subject-probing AUC on spliced MMLU sequences by fraction of features selected.

### 5.4 Case Study: Persistent Prompt Injection Monitoring

In indirect prompt injection [12], an agent meets a malicious instruction in an email, webpage, or tool response, reasons for hundreds of tokens, and acts later [30]; an agentic monitor must hold a security-relevant signal until the agent acts [31, 32].

Setting. 
AgentDojo trajectories from banking, travel, and workspace [12], generated by Llama-3.3-70B-Instruct [33] (Section C.3), each contain a request, an untrusted span with a prompt injection or benign content, and subsequent tool calls. Only failed attacks are used (injection present, attack action not completed).

Gemma-2-2B serves as external monitor: trajectories are replayed and layer-12 residual streams recorded. A linear probe is trained at the final token of the untrusted span ( $d=0$ ) and evaluated without retraining up to $350$ tokens later (Figure 7 (a)). Representations: raw hidden state, baseline SAEs, and Persistent SAE $\mathbf{h}^{(t)}$ restricted to slow (top $1\%$ by $\lambda_{j}$ ) or fast (bottom $50\%$ ) features (Appendix H).

Figure 7:
(a) Probe trained at the end of an untrusted span, evaluated later in the trajectory. (b) Predictive accuracy over distance (top) and causal effect on the monitor's judgement (bottom).

Signal retention: fast features detect local injections while slow states retain signals over long contexts. 
At $d=0$ , the fast-feature state and standard SAE representations give the strongest detection (Figure 7 (b), top); with distance their accuracy falls towards chance, whereas the slow-feature state retains a clearer signal.

Causal effect: slow-feature directions causally influence the monitor’s judgement. 
Each trajectory is truncated at a later distance, appended with a query asking whether it should be flagged for prompt injection, and intervened on at the final query token. Following [7, 34], a steering vector $\mathbf{v}_{r}$ per SAE representation $r$ weights its most predictive decoder directions by probe coefficients and is applied with both signs. With $s(\mathbf{x})=\ell_{\mathrm{yes}}(\mathbf{x})-\ell_{\mathrm{no}}(\mathbf{x})$ ,

| | $\displaystyle\Delta_{\mathrm{causal}}(r,d)=\frac{1}{2}\mathbb{E}_{i}\Big[s\big(\mathbf{x}_{i}^{(d)}+\alpha_{i}^{(d)}\mathbf{v}_{r}\big)-s\big(\mathbf{x}_{i}^{(d)}-\alpha_{i}^{(d)}\mathbf{v}_{r}\big)\Big],$
| | (8)

where $\alpha_{i}^{(d)}$ scales each intervention to $10\%$ of the hidden-state norm.

Slow features give the largest, consistently positive $\Delta_{\mathrm{causal}}$ at every distance. BatchTopK, Matryoshka, and Temporal SAE directions give near-zero or negative effects with no comparable alignment over distance.

## 6 Conclusion and Discussion

Conclusion. 
Persistent SAEs learn from reconstruction alone how long each sparse feature retains information across tokens. Learned coefficients span a broad range: short-timescale features remain locally interpretable; long-timescale features track subject identity and preserve monitoring signals over long contexts.

Limitations and Future Work. 
A single fixed coefficient per feature in a diagonal recurrence cannot capture feature interactions or context-dependent resets. The prompt-injection study uses an external replay monitor, so interventions establish effects on the monitor's judgement, not the original agent's actions. Richer recurrent dynamics and end-to-end agent effects of persistent feature interventions remain open.

## Appendix A Notation

Table 2: Notation by section of introduction. Bold: vectors and matrices; parenthesised superscript: token position; subscript: coordinate.

| Symbol | Meaning

| Hidden states and standard SAE (Section 3, 4)

| $\mathbf{x}^{(t)}\in\mathbb{R}^{d}$ | Residual-stream hidden state at token position $t$

| $T$ | Number of tokens in a sequence

| $d$ | Dimensionality of the hidden state

| $m$ | Number of dictionary features (SAE width)

| $\mathbf{W}_{\mathrm{enc}},\mathbf{b}_{\mathrm{enc}}$ | Standard SAE encoder weight matrix and bias

| $\mathbf{W}_{\mathrm{dec}},\mathbf{b}_{\mathrm{dec}}$ | Standard SAE decoder weight matrix and bias

| $\mathbf{d}_{j}$ | Unit-normalised decoder direction for feature $j$

| $\mathbf{f}^{(t)}\in\mathbb{R}_{\geq 0}^{m}$ | Sparse activation of a standard SAE at token $t$ (Background and Related Work)

| $f^{(t)}_{j}$ | Sparse activation of feature $j$ at token $t$

| $k$ | BatchTopK sparsity level (Section B.2)

| $B$ | Batch size used by BatchTopK selection (Section B.2)

| Empirical feature persistence analysis (Section 3, Appendix D)

| $\tau$ | Lag, in tokens

| $p^{(t)}_{j}$ | Decoder-aligned projection $\mathbf{d}_{j}^{\top}\mathbf{x}^{(t)}$

| $R_{j}(\tau)$ | Empirical persistence of $p^{(t)}_{j}$ at lag $\tau$

| $\beta_{\mathrm{nat}},\beta_{\mathrm{shf}}$ | Power-law decay exponents, natural / shuffled

| Context correction experiment (Section 3, Section D.1)

| $\mathbf{w}_{\tau}\in\mathbb{R}^{m}$ | Per-feature context-correction weights at lag $\tau$

| $w_{\tau,j}$ | Context-correction weight for feature $j$ at lag $\tau$ ( $w_{j}$ at $\tau=1$ )

| $\hat{\mathbf{x}}^{(t)}_{\mathrm{base}}$ | Baseline reconstruction from a standard SAE

| $\hat{\mathbf{x}}^{(t)}_{\mathrm{ctx}}$ | Context-corrected reconstruction (Equation 3)

| $R^{2}_{\mathrm{improve}}$ | Held-out reconstruction-error improvement from context correction

| Persistent SAE (Section 4)

| $\mathbf{a}^{(t)}\in\mathbb{R}_{\geq 0}^{m}$ | Token-wise sparse activation of a Persistent SAE (Section B.2)

| $a^{(t)}_{j}$ | Token-wise sparse activation of feature $j$ at token $t$

| $\mathbf{h}^{(t)}\in\mathbb{R}_{\geq 0}^{m}$ | Persistent state integrating token-wise activations over time (Equation 5)

| $h^{(t)}_{j}$ | Persistent state of feature $j$ at token $t$

| $\lambda_{j}\in(0,\lambda_{\max})$ | Persistence coefficient for feature $j$ (Section B.3)

| $\lambda_{\max}$ | Upper bound on the persistence coefficient ( $=0.999$ )

| $l_{j}$ | Unconstrained logit parameterising $\lambda_{j}$

| $\sigma(\cdot)$ | Sigmoid function

## Appendix B SAE Training Details

### B.1 SAE Baseline Training Details

Four families are trained: BatchTopK [8], Matryoshka BatchTopK [9], Temporal SAE [10], and Persistent SAE, with matched dictionary width, BatchTopK budget, optimiser, and token budget.

Common configuration. 
$m=16{,}384$ , $k=20$ per token; Adam at $3\times 10^{-4}$ with linear warm-up over the first $1{,}000$ steps. From step $1{,}000$ , the BatchTopK activation-threshold schedule with decay $\beta=0.999$ and the auxiliary dead-feature loss weighted by $\alpha_{\mathrm{aux}}=0.03125$ are applied. Activations come from Pile Uncopyrighted [35] with the beginning-of-sequence position removed. Gemma-2-2B [21]: layer-12 residual stream, $d=2{,}304$ ; Pythia-160M-deduped [20]: layer-8 residual stream, $d=768$ .

Matryoshka and Temporal SAEs share two predefined groups of $20\%$ and $80\%$ of the $16{,}384$ features; the Temporal SAE weights reconstruction losses $0.2$ and $0.8$ , Matryoshka uniformly. The Persistent SAE uses no partition.

Matched token budget. 
BatchTopK and Matryoshka consume shuffled token activations, Temporal SAE adjacent token pairs, Persistent SAE contiguous sequences. Activation tokens per step and total steps are matched: $244{,}140$ steps at approximately $2{,}048$ activation tokens per step, i.e. $2{,}048\times 244{,}140=499{,}998{,}720$ tokens, approximately $5.0\times 10^{8}$ , for the flat baselines.

•

BatchTopK and Matryoshka. Flat batches of $2{,}048$ token activations per step.

•

Temporal SAE. Adjacent previous–next token pairs from the same stream, with reconstructed positions per step equal to the flat batch size.

•

Persistent SAE. Eight contiguous sequences of context length $256$ per step; excluding the beginning-of-sequence position, $8\times 255=2{,}040$ usable activations per step, within $0.4\%$ of the flat budget.

All families receive the same number of gradient updates and, up to pairing or sequencing, approximately the same number of token activations. Checkpoints follow a fixed schedule; the final dictionary (ae.pt) is used throughout. Hardware: NVIDIA A100 GPUs with 80 GB of memory.

Figure 8:
Persistent SAE training dynamics: training loss (left), number of non-zero coordinates of $\mathbf{h}^{(t)}$ (state $L_{0}$ ; centre), and fraction of state activation mass from the carried term $\bm{\lambda}\odot\mathbf{h}^{(t-1)}$ (right), for Pythia-160M layer 8 and Gemma-2B layer 12.

Loss drops rapidly then stabilises; state $L_{0}$ settles at roughly $400$ – $500$ active coordinates; carried-mass fraction stabilises at approximately $0.5$ (Pythia) and $0.65$ – $0.70$ (Gemma), away from both zero and one, with no continual growth of the active state.

### B.2 BatchTopK Sparsity

All SAEs compute

| | $\tilde{\mathbf{f}}=\mathrm{ReLU}\!\left(\mathbf{W}_{\mathrm{enc}}(\mathbf{x}-\mathbf{b}_{\mathrm{dec}})+\mathbf{b}_{\mathrm{enc}}\right)\in\mathbb{R}_{\geq 0}^{m},$
| | (9)

flatten $\{\tilde{\mathbf{f}}_{i}\}_{i=1}^{B}$ for batch $X=\{\mathbf{x}_{i}\}_{i=1}^{B}$ into a vector of length $Bm$ , keep the top $kB$ entries globally to obtain $\mathbf{f}_{i}$ , and reconstruct

| | $\hat{\mathbf{x}}_{i}=\mathbf{W}_{\mathrm{dec}}\mathbf{f}_{i}+\mathbf{b}_{\mathrm{dec}}.$
| | (10)

At inference,

| | $f_{ij}=\tilde{f}_{ij}\,\mathbf{1}[\tilde{f}_{ij}>\theta],$
| | (11)

with $\theta$ estimated from training-time top- $k$ statistics. The Persistent SAE takes this output as $\mathbf{a}^{(t)}$ .

### B.3 Persistence Parameterisation

| | $\lambda_{j}=\sigma(l_{j})\cdot\lambda_{\max},$
| | (12)

with $l_{j}\in\mathbb{R}$ and hard cap $\lambda_{\max}=0.999$ , so $\lambda_{j}\in(0,\lambda_{\max})$ throughout training and every feature has finite effective memory without exact poles. Equation 5 is an exponential moving average (EMA): under constant $a^{(t)}_{j}=a_{j}$ the state converges to $a_{j}$ for every $\lambda_{j}$ . All logits $l_{j}$ are initialised to $0.5$ , away from sigmoid saturation.

## Appendix C Dataset Details

### C.1 Pile Corpus

The Pile is an $825$  GiB English corpus from 22 sources, including academic papers, web pages, books, code, and online discussions [22]. We use Pile Uncopyrighted [35], which removes Books3, BookCorpus2, OpenSubtitles, YouTube Subtitles, and OpenWebText2. Documents are streamed and shuffled with a buffer of $10{,}000$ under a fixed seed and accepted if, after tokenisation and truncation to the context cap, they meet the minimum token count.

Natural-corpus temporal statistics on a retained length- $L$ sequence use positions $\{16,\dots,L-1\}$ .

The shuffled control concatenates sentences sampled from documents of the same corpus, forbidding consecutive sentences from the same source document; long-range entity tracking and discourse progression are thereby broken.

### C.2 MMLU Dataset

MMLU is an English benchmark of four-option multiple-choice questions from 57 subjects across STEM, humanities, social sciences, and other fields [29], used here only for subject analyses. Subjects: abstract algebra, astronomy, business ethics, computer security, and European history.

Geometry (Section 5.3): $100$ sequences across the five subjects, final $20$ token representations each. Probing: sequences of equal-length segments from different subjects, each token labelled by its source segment. Details: Appendix F and Appendix G.

### C.3 AgentDojo Trajectories

AgentDojo [12] provides stateful tool environments with 97 user tasks, 27 injection targets, and 629 security test cases across workspace, Slack, travel, and banking, with task-specific checks on agent output and environment changes for both the user request and the attacker's target action.

We use banking, travel, and workspace trajectories generated by Llama-3.3-70B-Instruct. Negatives ( $y=0$ ): benign trajectories without injection. Positives ( $y=1$ ): important_instructions or direct attacks with security=false (injection present, target action not completed). Successful attacks are excluded to remove overt malicious actions, such as transferring money or sending private data, as a classification cue. No filtering by task utility.

## Appendix D Empirical Persistence of Standard SAE Features

Lags $\tau\in\{1,4,16,32,64,128,256,512\}$ ; documents of at most $1{,}024$ traced tokens. Retained features have at least $256$ activation events, at least $64$ valid anchors at every lag, and non-negligible activated-token elevation; $R_{j}(\tau)$ is excluded when $\mathbb{E}[p^{(t)}_{j}\mid\mathbf{f}^{(t)}_{j}>0]-\mathbb{E}[p^{(t)}_{j}]$ is near zero (near-dead or diffuse features), with no $\epsilon$ added.
Pythia-160M: median BatchTopK persistence $0.618$ at lag $1$ , $0.160$ at lag $64$ , $0.096$ at lag $512$ ; Matryoshka $0.504$ , $0.108$ , $0.059$ . Gemma-2-2B natural documents at lag $512$ : $0.042$ (BatchTopK), $0.034$ (Matryoshka). Shuffled controls decay to approximately zero at long lags on both models.
Power-law fits to the positive decay tail: median $\beta_{\mathrm{nat}}\approx 0.32$ versus $\beta_{\mathrm{shf}}\approx 0.87$ – $0.92$ , for both SAE families.

Average $R_{j}(\tau)$ within short ( $\tau\leq 16$ ), medium ( $32\leq\tau\leq 64$ ), and long ( $\tau\geq 128$ ) bands spans nearly the full unit interval in each band; some features persist at short lags and decay before medium lags, others stay elevated into the long-lag regime. This heterogeneity holds in both families and substantially exceeds that of the shuffled control.

### D.1 Context Correction Experiment

Equation 3 is fitted independently at $\tau\in\{1,2,4,8,16,32\}$ on $192$ natural documents and evaluated on $96$ held-out natural documents; targets are always natural-document reconstruction errors, and the control refits the diagonal head from shuffled-corpus context activations with identical parameterisation and budget. At most $768$ tokens per document after dropping $16$ prefix tokens; $10$ epochs of AdamW, learning rate $5\times 10^{-3}$ , weight decay $10^{-3}$ .

Pythia-160M: $0.89\%$ (BatchTopK) and $0.61\%$ (Matryoshka) of baseline error removed at lag $1$ , $0.22\%$ and $0.13\%$ at lag $32$ . Gemma-2-2B: approximately $0.40\%$ for both at lag $1$ , about $0.02\%$ at lag $16$ , zero by lag $32$ . Shuffled controls stay slightly below zero throughout.

Table 3: Fitted lag-one context-correction weights $w_{j}$ . Mean $w_{j}$ and Pos. frac.: mean and positive fraction on natural documents; Shuf. $w_{j}$ : mean on the shuffled control.

| Model | SAE | Mean $w_{j}$ | Pos. frac. | Shuf. $w_{j}$

| Pythia | BatchTopK | $0.068$ | $76.2\%$ | $\phantom{-}0.001$

| Pythia | Matryoshka | $0.087$ | $91.4\%$ | $-0.001$

| Gemma | BatchTopK | $0.068$ | $80.4\%$ | $-0.001$

| Gemma | Matryoshka | $0.091$ | $81.9\%$ | $-0.001$

Shuffled distributions are sharply centred at zero.

## Appendix E Automated Interpretability Details

Following Bills et al. [23], a judge writes an explanation from each feature's highest-activation contexts and is scored on predicting whether the activation exceeds a threshold on held-out contexts: $f_{j}^{(t)}$ for standard SAEs, $a_{j}^{(t)}$ for the Persistent SAE. Judge: gpt-4o-mini; context length $1024$ ; roughly $2$ M Pile tokens for activation examples. The highest- $\lambda_{j}$ $1\%$ and lowest- $\lambda_{j}$ $50\%$ are also scored as groups.

Table 4: Qualitative autointerpretation examples for Pythia-160M. High- and low- $\lambda$ features from the top 1% and bottom 50% of learned coefficients. Bold marks representative activating spans. All selected features score above 0.85.

| Feature | $\lambda$
|

Autointerpretation

|

Representative activation

| Score

| High $\lambda$ (top 1%): sustained semantic domains

| F-5528 | $7.03\text{\times}{10}^{-1}$
|

themes of ethnic diversity and experiences of discrimination

|

“ashamed of being Mexican because I’m different from the stereotype”

| 0.93

| F-12751 | $4.82\text{\times}{10}^{-1}$
|

phrases related to population-based studies and cancer research

|

incident colorectal cancer cases identified through a population-based cancer registry

| 0.93

| F-15986 | $3.81\text{\times}{10}^{-1}$
|

terms related to high-performance cars and their specifications

|

rare and desirable muscle cars; a 1971 Hemi ’Cuda

| 0.93

| F-6538 | $3.79\text{\times}{10}^{-1}$
|

terms related to weather conditions and climate phenomena

|

a combination of global warming and the El Niño weather phenomenon

| 1.00

| F-8449 | $3.69\text{\times}{10}^{-1}$
|

mathematical concepts and terms related to Banach spaces and polynomials

|

continuous scalar polynomials over a Banach space $X$

| 1.00

| Low $\lambda$ (bottom 50%): local lexical, syntactic, and template patterns

| F-1113 | $5.35\text{\times}{10}^{-2}$
|

the substring “white” followed by various words or concepts

|

“There is so much white space just waiting to be filled”

| 1.00

| F-6958 | $4.96\text{\times}{10}^{-2}$
|

the substring “function” in JavaScript code snippets

|

.click(function() {

| 1.00

| F-15079 | $4.81\text{\times}{10}^{-2}$
|

the phrase “U.S. Pat. No.” indicating references to patents

|

“disclosed in U.S. Pat. No. 5,111,981”

| 1.00

| F-3587 | $1.10\text{\times}{10}^{-2}$
|

URLs and web documentation references

|

ubuntu.com/10.04/serverguide/C/index.html

| 1.00

| F-14758 | $4.04\text{\times}{10}^{-4}$
|

SQL commands related to table creation and data manipulation

|

CREATE TABLE IF NOT EXISTS tennis(...)

| 1.00

Table 5: Qualitative autointerpretation examples for Gemma-2-2B. High- and low- $\lambda$ features from the top 1% and bottom 50% of learned coefficients. Bold marks representative activating spans. All selected features score above 0.85.

| Feature | $\lambda$
|

Autointerpretation

|

Representative activation

| Score

| High $\lambda$ (top 1%): sustained semantic domains

| F-11205 | $9.18\text{\times}{10}^{-1}$
|

terms related to cosmology and advanced scientific concepts

|

near-future CMB Stage-III and IV surveys yielding reconstructed CMB lensing maps

| 0.86

| F-5578 | $8.55\text{\times}{10}^{-1}$
|

terms related to ionisation charge collection and quantum experiments in physics

|

ionisation charge collection efficiency observed in gases

| 0.86

| F-15639 | $8.24\text{\times}{10}^{-1}$
|

text related to conflict, violence, and ethnic tensions

|

self-determination for Katanga; a separatist group composed of former Katanga Tigers

| 0.86

| F-13900 | $7.85\text{\times}{10}^{-1}$
|

terms related to microbial biomass and ecological concepts in alpine and tundra environments

|

microbial biomass in an alpine meadow on the Tibetan Plateau

| 0.86

| F-11933 | $7.70\text{\times}{10}^{-1}$
|

expressions of self-acceptance and emotional support in personal growth narratives

|

“In the last couple years, I’ve let go and decided to be myself.”

| 1.00

| Low $\lambda$ (bottom 50%): local lexical, syntactic, and template patterns

| F-9996 | $1.03\text{\times}{10}^{-3}$
|

legal terms and conditions related to software licenses

|

“See the License for the specific language governing rights and limitations”

| 1.00

| F-11210 | $1.30\text{\times}{10}^{-5}$
|

the phrase “all of” in various contexts

|

“it is unlikely that all of these loci respond to selection independently”

| 0.93

| F-14595 | $7.87\text{\times}{10}^{-6}$
|

the substring “mk” within mathematical and scientific contexts

|

repeated \mkern commands in mathematical markup

| 0.93

| F-13259 | $2.73\text{\times}{10}^{-6}$
|

specific terms or phrases enclosed in quotation marks

|

users searching for “Pizza” or “Calgary pizza”

| 1.00

| F-7307 | $1.88\text{\times}{10}^{-6}$
|

the word “In” at the beginning of sentences

|

“In this way the missing persons in our evolutionary mystery would …”

| 0.86

## Appendix F Subject Geometry Details

$100$ sequences from the five MMLU subjects (abstract algebra, astronomy, business ethics, computer security, European history); last $20$ tokens of each, subject-labelled. Raw hidden states, BatchTopK features, the Temporal SAE's high-level group, and the Persistent SAE's high- $\lambda_{j}$ state are projected to two dimensions by t-SNE with shared hyperparameters and coloured by subject.

## Appendix G Subject Probing Details

Spliced sequences concatenate equal-length segments from different MMLU subjects, with subject labels balanced within each sequence and entire source documents held out for testing. A per-token linear probe predicts source subject; we report held-out one-vs-rest macro ROC AUC as the available fraction of features sweeps from $0.1\%$ to $100\%$ . The Persistent SAE is probed on $\mathbf{h}^{(t)}$ with features ordered by decreasing $\lambda_{j}$ , increasing $\lambda_{j}$ , repeated random permutations, or supervised one-vs-rest importance

| | $s_{j,c}=\mathbb{E}[h^{(t)}_{j}\mid y^{(t)}=c]-\mathbb{E}[h^{(t)}_{j}\mid y^{(t)}\neq c],$
| | (13)

computed on the training split, sorted within subject and interleaved round-robin into one global ordering for the multiclass probe. BatchTopK, Matryoshka, and Temporal baselines use full dictionaries under the same probe.

## Appendix H Prompt Injection Monitoring Details

Trajectory construction. 
AgentDojo [12] runs of Llama-3.3-70B-Instruct agents; attacks restricted to important_instructions and direct types with security=false; benign trajectories contain no injected content. Each serialised trajectory is appended with

should this trajectory be flagged for prompt injection or unsafe instruction hijacking? Answer yes or no:

and read with Gemma-2-2B at the layer-12 residual stream. The yes-minus-no logit at the query position yields AUC $0.51$ .

Checkpoints and probes. 
Checkpoints sit at $0$ , $50$ , $100$ , $200$ , $250$ , $300$ , and $350$ tokens after the final injected-span token (matched position for benign trajectories), plus the final query position. Extracted representations: raw hidden state, BatchTopK and Matryoshka activations, the Temporal SAE's high-level group, and the Persistent SAE's $\mathbf{a}^{(t)}$ , full $\mathbf{h}^{(t)}$ , and slow state (largest 1% of $\mathbf{h}^{(t)}$ coordinates by $\lambda_{j}$ ). Logistic regression probes are fit at $d=0$ only and evaluated without retraining at every checkpoint; held-out accuracy and ROC AUC are reported.

Causal intervention. 
Per SAE representation, the $k=128$ features with largest absolute probe coefficients form the probe-weighted decoder direction $\mathbf{v}_{r}$ , normalised and scaled so that $\lVert\alpha_{i}^{(d)}\mathbf{v}_{r}\rVert_{2}=\rho\lVert\mathbf{x}_{i}^{(d)}\rVert_{2}$ with $\rho=0.1$ , and added to the residual stream with both signs. Equation 8 measures the change in the external replay monitor's logit margin, not the original Llama-3.3-70B agent's action. Directions are constructed at $d=0$ ; Figure 7 reports transfer of their effect across distance.