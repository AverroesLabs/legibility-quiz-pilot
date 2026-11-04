# Persistent Sparse Autoencoders: Learning Feature-Specific Timescales in Language Model Representations

Haoyan Luo (University of Cambridge), Mateo Espinosa Zarlenga (University of Oxford), Mateja Jamnik (University of Cambridge)

###### Abstract

Standard SAEs encode each token independently. We show temporal persistence emerges in standard SAE features, with widely varying duration. Persistent SAEs learn a per-feature persistence coefficient from reconstruction alone. They retain competitive reconstruction and learn a spectrum of timescales: fast features stay locally interpretable; slow features accumulate context-identifying information and, in prompt-injection monitoring, preserve injection signals that stay causally effective over long contexts.

## 1 Introduction

Causal transformer hidden states integrate preceding context, yet standard SAEs encode each token independently, favouring token-specific features and obscuring carried context (topic, style, task state). Prior temporal SAEs require an a-priori partition of which features carry temporal information, introducing experimenter bias. Can each feature's timescale be learned from data?

Observation (Figure 1): after a feature activates, later hidden states stay aligned with its decoder direction, and past activations help reconstruct later states; decay rates differ widely across features. Persistent SAEs learn a coefficient $\lambda_j$ per feature: each feature keeps a state decaying at $\lambda_j$, updated by a token-wise sparse activation. Reconstruction favours higher $\lambda_j$ when carrying state forward helps, smaller otherwise.

Contributions:
- SAE features exhibit temporal persistence: decoder direction stays elevated after activation; past activations help predict later states.
- Persistent SAEs learn one coefficient per feature through reconstruction alone, retain competitive reconstruction, and discover a broad timescale spectrum without predefined temporal roles.
- Fast features are locally interpretable detectors; slow features retain contextual information; in a prompt-injection case study slow features help monitor and steer over long contexts.

## 2 Background and Related Work

**Sparse autoencoders.** For hidden state $\mathbf{x}^{(t)}\in\mathbb{R}^d$, an SAE gives sparse $\mathbf{f}^{(t)}\in\mathbb{R}^m_{\ge0}$ and

$$\hat{\mathbf{x}}^{(t)}_{\mathrm{base}}=\mathbf{W}_{\mathrm{dec}}\mathbf{f}^{(t)}+\mathbf{b}_{\mathrm{dec}}.\quad(1)$$

Feature $j$: activation $f^{(t)}_j$, direction $\mathbf{d}_j=\mathbf{W}_{\mathrm{dec}}[:,j]/\lVert\mathbf{W}_{\mathrm{dec}}[:,j]\rVert_2$; active when $f^{(t)}_j>0$. BatchTopK uses a batch-level sparsity budget; Matryoshka SAEs jointly train nested dictionaries.

**Temporal structure.** Standard SAEs compute $\mathbf{f}^{(t)}$ independently of preceding tokens. Slow feature analysis shows slowly varying latents reveal higher-level features [14, 15]; LMs track entities and encode board, spatial, temporal and propositional states [16, 17, 18, 19].

**Temporal SAEs.** Temporal SAEs pre-assign a high-level subset and align its activations across nearby tokens contrastively; Lubana et al. use an attention-based predictor to separate context-predictable information from a sparse novel residual. Both fix a partition; neither lets each feature learn its own persistence. Persistent SAEs give each feature a state and an analysable learned coefficient.

Figure 1: (a) Natural Pile sequences preserve local and global continuity; shuffled controls (sentences from different documents) preserve only local. (b) $R_j(\tau)$ decays more slowly in natural sequences. (c) Past activations improve held-out reconstruction in natural, not shuffled, sequences. (d) At $\tau=1$, $w_j$ mostly positive for natural, centred near zero for shuffled.

## 3 Does Temporal Structure Exist at the Feature Level?

**Setup.** Residual-stream states from Pythia-160M-deduped (layer 8) and Gemma-2-2B (layer 12) [21] on Pile documents; BatchTopK and Matryoshka SAEs (no temporal prior; Appendix B). The shuffled control (Appendix C) isolates global continuity.

**Persistence in hidden states.** With $p^{(t)}_j=\mathbf{d}_j^\top\mathbf{x}^{(t)}$:

$$R_j(\tau):=\frac{\mathbb{E}[p^{(t+\tau)}_j\mid f^{(t)}_j>0]-\mathbb{E}[p^{(t)}_j]}{\mathbb{E}[p^{(t)}_j\mid f^{(t)}_j>0]-\mathbb{E}[p^{(t)}_j]},\quad(2)$$

$\mathbb{E}[p^{(t)}_j]$ the average over all tokens; $R_j=1$: elevation fully retained, $0$: decayed to baseline (Appendix D). Across both models and SAE families, persistence decays more slowly in natural documents (Fig. 1b): features inherit document-level persistence without encoding it.

**Past activations help reconstruction.** Restricted context correction:

$$\hat{\mathbf{x}}^{(t)}_{\mathrm{ctx}}=\hat{\mathbf{x}}^{(t)}_{\mathrm{base}}+\mathbf{W}_{\mathrm{dec}}(\mathbf{w}_\tau\odot\mathbf{f}^{(t-\tau)}),\quad(3)$$

$\mathbf{w}_\tau\in\mathbb{R}^m$ acting only along each feature's own direction; fit 10 epochs with $L_2$ penalty; on held-out data

$$R^2_{\mathrm{improve}}=1-\frac{\sum_t\lVert\mathbf{x}^{(t)}-\hat{\mathbf{x}}^{(t)}_{\mathrm{ctx}}\rVert_2^2}{\sum_t\lVert\mathbf{x}^{(t)}-\hat{\mathbf{x}}^{(t)}_{\mathrm{base}}\rVert_2^2}.\quad(4)$$

Control: refit using shuffled-corpus past activations with the same natural targets (Section D.1). Natural past activations recover part of baseline error, decreasing with lag; shuffled do not (Fig. 1c); $\tau=1$ weights mostly positive vs. near zero (Fig. 1d). Standard SAEs leave reconstruction-relevant temporal structure unused. Unlike Lubana et al.'s hidden-state-level findings, this is feature-level: persistence helps reconstruction and varies widely across features.

## 4 Persistent Sparse Autoencoders

A BatchTopK encoder (Section B.2) gives token-wise $\mathbf{a}^{(t)}\in\mathbb{R}^m_{\ge0}$ ($a^{(t)}_j>0$: new evidence), integrated into state $\mathbf{h}^{(t)}$ (Figure 2):

$$h^{(t)}_j=\lambda_jh^{(t-1)}_j+(1-\lambda_j)a^{(t)}_j,\quad(5)$$
$$\hat{\mathbf{x}}^{(t)}=\mathbf{W}_{\mathrm{dec}}\mathbf{h}^{(t)}+\mathbf{b}_{\mathrm{dec}}.\quad(6)$$

Small $\lambda_j$: fast; large: slow. $(1-\lambda_j)$ keeps state bounded. $\mathbf{h}^{(0)}=\mathbf{0}$ per sequence; $\lambda_j=\lambda_{\max}\sigma(l_j)$, $\lambda_{\max}=0.999$ (Section B.3). The diagonal recurrence preserves feature-level attribution. Sparsity is on $\mathbf{a}^{(t)}$; $\mathbf{h}^{(t)}$ need not be sparse. No predefined partitions or auxiliary temporal objectives.

## 5 Experiments

Same models/data as Section 3. Baselines: BatchTopK, Matryoshka, Temporal SAE. All: $m=16{,}384$, $k=20$; Matryoshka and Temporal share a predefined 20%/80% partition (Appendix B).

### 5.1 Reconstruction Quality and Learned Timescales

$\lambda_j$ spans a broad range: mostly short-timescale, small long-timescale tail (Fig. 3a). Slow = top 1% by $\lambda_j$; fast = bottom 50%. Metrics: FVE, cosine similarity, fraction alive, temporal smoothness (Bhalla et al.). Persistent SAEs stay competitive; slow features are smoother than fast features and baselines. Grouping into $\lambda_j$ quintiles, higher quintiles have greater median $R_j(\tau)$ across lags (Fig. 3b); the Temporal SAE's predefined groups are not clearly separated by empirical persistence.

Table 1: High/Low = predefined splits (Matryoshka, Temporal) or top 1%/bottom 50% $\lambda_j$ (Persistent). **Best**, _second_.

| Model | SAE | FVE↑ | Cos↑ | Alive↑ | Smooth↓ Full | High | Low | Autointerp↑ Full | High | Low |
|---|---|---|---|---|---|---|---|---|---|---|
| Pythia | BatchTopK | **0.94±.01** | **0.95±.01** | _0.91±.01_ | 0.15±.01 | – | – | **0.91±.02** | – | – |
| | Matryoshka | 0.92±.01 | _0.94±.01_ | **0.96±.01** | 0.15±.01 | 0.25±.01 | _0.12±.01_ | _0.87±.02_ | – | – |
| | Temporal | _0.93±.02_ | **0.95±.01** | 0.79±.01 | _0.13±.01_ | _0.18±.01_ | **0.11±.01** | _0.87±.02_ | **0.87±.01** | _0.89±.02_ |
| | Persistent | _0.93±.01_ | **0.95±.01** | **0.96±.01** | **0.10±.01** | **0.05±.01** | 0.20±.01 | 0.86±.01 | _0.84±.02_ | **0.93±.01** |
| Gemma | BatchTopK | **0.72±.01** | **0.79±.01** | **0.92±.01** | 0.16±.01 | – | – | _0.87±.03_ | – | – |
| | Matryoshka | _0.71±.01_ | _0.77±.01_ | 0.87±.01 | 0.17±.01 | 0.21±.01 | **0.14±.01** | **0.89±.02** | – | – |
| | Temporal | 0.62±.02 | 0.75±.01 | 0.75±.01 | _0.14±.01_ | _0.12±.01_ | _0.16±.01_ | 0.85±.02 | **0.85±.02** | _0.86±.02_ |
| | Persistent | 0.66±.01 | 0.75±.01 | _0.91±.01_ | **0.11±.01** | **0.04±.01** | 0.20±.01 | 0.85±.01 | _0.79±.01_ | **0.89±.01** |

### 5.2 Short-Timescale Features Are Locally Interpretable

Autointerp pipeline of Bills et al. [23] on token-wise $\mathbf{a}^{(t)}$ (Appendix E). Fast features are easier to explain than slow ones on both models, and match or exceed the best baseline. Fast features respond to compact lexical/grammatical patterns concentrated at the current token; slow features reflect broader subjects distributed across the sequence, a mismatch with token-specific evaluation: distinct features may get similar descriptions [24], explanations may not generalise [25]. Hence slow features are studied jointly (5.3).

**Where Features Receive New Evidence.** Five equal $\lambda$ quintiles; six token categories (punctuation, conjunctions, function words, verbs, adjectives/adverbs, nouns), following the lexical/function distinction [26]; Func. = adpositions, determiners, particles, pronouns.

$$U_{q,c}=\Pr(T^{(t)}=c\mid a^{(t)}_j>0,j\in Q_q)/\Pr(T^{(t)}=c),\quad(7)$$

$>1$: more updates than expected. Lowest-$\lambda$ quintile: more updates at nouns (clear lexical evidence). Highest: more at punctuation and conjunctions, positions associated with organising/integrating context [27, 28]. Update locations don't reveal state content.

### 5.3 Long-Timescale States Track Subject Information

**Slow features form subject-specific clusters.** t-SNE of final 20 tokens of MMLU [29] examples from five subjects (Appendix F), top 1% slow features: BatchTopK mixes subjects, Temporal's high-level group separates little; slow Persistent features form coherent subject regions, clearer than raw hidden states, without labels (Fig. 5).

**Persistence selects subject-informative features without labels.** Spliced MMLU sequences (subject changes at boundaries); per-token linear probe for current subject using varying feature fractions. Rankings: decreasing $\lambda_j$, increasing, random; baselines BatchTopK, Matryoshka, Temporal; supervised label-informed oracle (Appendix G). Decreasing persistence is nearly as efficient as supervised; slowest 1% reach high AUC on both models, while low-$\lambda_j$, random and baselines retain substantially less at the same budget (Fig. 6).

### 5.4 Case Study: Persistent Prompt Injection Monitoring

Agents may read an injection, reason hundreds of tokens, then act [30]; monitors must preserve the signal until action [31].

**Setting.** AgentDojo trajectories (banking, travel, workspace) generated by Llama-3.3-70B-Instruct [33] (Section C.3): request, untrusted span (injection or benign), subsequent tool calls. Only failed attacks used. Gemma-2-2B as external monitor, layer-12 residual. Linear probe trained at final untrusted-span token ($d=0$), evaluated up to 350 tokens later without retraining. Compare raw state, baseline SAEs, Persistent $\mathbf{h}^{(t)}$ restricted to slow (top 1%) or fast (bottom 50%) (Appendix H).

**Signal retention.** At $d=0$, fast-feature state and standard SAEs detect best; with distance they fall toward chance; slow state retains a clearer signal (Fig. 7b top).

**Causal effect.** Truncate at a later distance, append a flagging query, intervene at the final query token. Steering vector $\mathbf{v}_r$ weights most predictive decoder directions by probe coefficients [34]. With $s(\mathbf{x})=\ell_{\mathrm{yes}}-\ell_{\mathrm{no}}$:

$$\Delta_{\mathrm{causal}}(r,d)=\tfrac12\mathbb{E}_i\big[s(\mathbf{x}_i^{(d)}+\alpha_i^{(d)}\mathbf{v}_r)-s(\mathbf{x}_i^{(d)}-\alpha_i^{(d)}\mathbf{v}_r)\big],\quad(8)$$

$\alpha_i^{(d)}$ scaled to 10% of hidden-state norm; positive = shifts toward yes. Slow features give the largest, consistently positive effect at every distance; BatchTopK, Matryoshka, Temporal give near-zero or negative effects with no alignment over distance. One representation supports immediate detection (fast) and long-context retention and causal influence (slow).

## 6 Conclusion and Discussion

**Limitations and Future Work.** One fixed coefficient per feature via a diagonal recurrence: no feature interactions or context-dependent resets. The injection study uses an external replay monitor, so interventions affect the monitor's judgement, not the original agent's actions. Future: richer recurrent dynamics; testing effects on end-to-end agent behaviour.

## Appendix A Notation

Bold: vectors/matrices; superscript $(t)$: token; subscript: coordinate.

| Symbol | Meaning |
|---|---|
| $\mathbf{x}^{(t)}\in\mathbb{R}^d$ | Residual hidden state at $t$ |
| $T$, $d$, $m$ | Sequence length, hidden dim, SAE width |
| $\mathbf{W}_{\mathrm{enc}},\mathbf{b}_{\mathrm{enc}}$; $\mathbf{W}_{\mathrm{dec}},\mathbf{b}_{\mathrm{dec}}$ | Encoder / decoder weights, bias |
| $\mathbf{d}_j$ | Unit decoder direction |
| $\mathbf{f}^{(t)}$, $f^{(t)}_j$ | Standard SAE sparse activation |
| $k$, $B$ | BatchTopK level, batch size |
| $\tau$ | Lag (tokens) |
| $p^{(t)}_j$ | $\mathbf{d}_j^\top\mathbf{x}^{(t)}$ |
| $R_j(\tau)$ | Empirical persistence |
| $\beta_{\mathrm{nat}},\beta_{\mathrm{shf}}$ | Power-law decay exponents |
| $\mathbf{w}_\tau$, $w_{\tau,j}$ | Context-correction weights ($w_j$ at $\tau=1$) |
| $\hat{\mathbf{x}}_{\mathrm{base}}$, $\hat{\mathbf{x}}_{\mathrm{ctx}}$ | Baseline / context-corrected reconstruction |
| $R^2_{\mathrm{improve}}$ | Held-out error improvement |
| $\mathbf{a}^{(t)}$, $a^{(t)}_j$ | Persistent SAE token-wise activation |
| $\mathbf{h}^{(t)}$, $h^{(t)}_j$ | Persistent state |
| $\lambda_j\in(0,\lambda_{\max})$; $\lambda_{\max}=0.999$ | Persistence coefficient; cap |
| $l_j$, $\sigma$ | Logit; sigmoid |

## Appendix B SAE Training Details

### B.1 SAE Baseline Training Details

All four families share width, budget, optimiser, token budget. $m=16{,}384$, $k=20$; Adam, lr $3\times10^{-4}$, linear warm-up over first 1{,}000 steps; from step 1{,}000, BatchTopK threshold schedule with decay $\beta=0.999$ and auxiliary dead-feature loss $\alpha_{\mathrm{aux}}=0.03125$. Data: Pile Uncopyrighted [35], BOS removed. Gemma-2-2B layer 12, $d=2{,}304$; Pythia-160M-deduped layer 8, $d=768$. Matryoshka and Temporal use 20%/80% groups; Temporal weights losses 0.2/0.8, Matryoshka uniform. Persistent uses no partition.

Matched token budget: 244{,}140 steps, ~2{,}048 activation tokens/step; flat baselines $2{,}048\times244{,}140=499{,}998{,}720\approx5.0\times10^8$ tokens.
- BatchTopK, Matryoshka: flat batch of 2{,}048 shuffled tokens.
- Temporal: adjacent previous–next pairs; reconstructed positions/step equal flat batch size.
- Persistent: 8 contiguous sequences of length 256; $8\times255=2{,}040$ tokens/step, within 0.4%.

Same gradient updates; final dictionary (ae.pt) used everywhere; NVIDIA A100 80 GB GPUs.

Figure 8 (Pythia layer 8, Gemma layer 12): loss drops rapidly then stabilises; state $L_0$ settles at ~400–500 active coordinates; fraction of state mass from carried term $\bm\lambda\odot\mathbf{h}^{(t-1)}$ stabilises at ~0.5 (Pythia) and 0.65–0.70 (Gemma). No continual growth of the active state; neither model relies exclusively on new evidence or carried state.

### B.2 BatchTopK Sparsity

$$\tilde{\mathbf{f}}=\mathrm{ReLU}(\mathbf{W}_{\mathrm{enc}}(\mathbf{x}-\mathbf{b}_{\mathrm{dec}})+\mathbf{b}_{\mathrm{enc}})\in\mathbb{R}^m_{\ge0}.\quad(9)$$

For batch $\{\mathbf{x}_i\}_{i=1}^B$, flatten to length $Bm$, keep top $kB$ entries globally;

$$\hat{\mathbf{x}}_i=\mathbf{W}_{\mathrm{dec}}\mathbf{f}_i+\mathbf{b}_{\mathrm{dec}}.\quad(10)$$

Inference: $f_{ij}=\tilde f_{ij}\mathbf{1}[\tilde f_{ij}>\theta]$ (11), $\theta$ estimated from training top-$k$ statistics. The Persistent SAE uses this as $\mathbf{a}^{(t)}$.

### B.3 Persistence Parameterisation

$\lambda_j=\sigma(l_j)\cdot\lambda_{\max}$ (12), $l_j\in\mathbb{R}$, $\lambda_{\max}=0.999$ prevents exact poles, giving finite memory. Eq. 5 is an EMA: under constant $a_j$, state converges to $a_j$ for any $\lambda_j$, so persistence doesn't inflate magnitude. All $l_j$ initialised to 0.5 (moderate persistence, away from saturation).

## Appendix C Dataset Details

### C.1 Pile Corpus

Pile: 825 GiB, 22 sources. Pile Uncopyrighted removes Books3, BookCorpus2, OpenSubtitles, YouTube Subtitles, OpenWebText2. Streaming, shuffle buffer 10{,}000, fixed seed; documents accepted if ≥ minimum tokens after tokenisation and truncation to the cap. For retained length $L$, temporal statistics use positions $\{16,\dots,L-1\}$. Shuffled control: sentences sampled from documents of the same corpus, concatenated with no two consecutive sentences from the same source document, breaking long-range entity tracking and discourse progression.

### C.2 MMLU Dataset

Four-option questions, 57 subjects; used only for subject analyses. Subjects: abstract algebra, astronomy, business ethics, computer security, European history. Geometry: 100 sequences, final 20 tokens each. Probing: equal-length segments from different subjects, tokens labelled by source segment.

### C.3 AgentDojo Trajectories

AgentDojo: 97 user tasks, 27 injection targets, 629 security test cases across workspace, Slack, travel, banking; task-specific checks inspect agent output and environment changes. We use banking, travel, workspace from Llama-3.3-70B-Instruct. $y=0$: benign, no injection. $y=1$: important_instructions or direct attacks with security=false (injection present, target action not completed), avoiding trivial cues from overt malicious actions. No utility filtering.

## Appendix D Empirical Persistence of Standard SAE Features

Lags $\tau\in\{1,4,16,32,64,128,256,512\}$; documents ≤1{,}024 traced tokens. Features kept if ≥256 activation events, ≥64 valid anchors per lag, and non-negligible elevation; $R_j$ undefined/excluded when the denominator is near zero (no $\epsilon$ added). Pythia: BatchTopK median 0.618 (lag 1) → 0.160 (lag 64) → 0.096 (lag 512); Matryoshka 0.504 → 0.108 → 0.059. Gemma natural at lag 512: 0.042 (BatchTopK), 0.034 (Matryoshka). Shuffled ≈0 at long lags on both. Power-law fits to positive tail: $\beta_{\mathrm{nat}}\approx0.32$ vs $\beta_{\mathrm{shf}}\approx0.87$–0.92, both families. Ranking by average $R_j$ in short ($\tau\le16$), medium ($32\le\tau\le64$), long ($\tau\ge128$) bands, distributions span nearly the full unit interval; some features persist only short, others into long lags; heterogeneity in both families, much larger than shuffled.

### D.1 Context Correction Experiment

Lags $\{1,2,4,8,16,32\}$, fit independently; 192 natural documents to fit, 96 held out; targets always natural. Control: same head/budget on shuffled context activations. ≤768 tokens/document after dropping 16 prefix tokens; 10 epochs AdamW, lr $5\times10^{-3}$, weight decay $10^{-3}$. Pythia: 0.89% (BatchTopK), 0.61% (Matryoshka) error removed at lag 1; positive at lag 32 (0.22%, 0.13%). Gemma: ~0.40% at lag 1, ~0.02% at lag 16, vanishing by 32. Shuffled slightly below zero throughout.

Table 3: Lag-one $w_j$ (natural mean, positive fraction; shuffled mean).

| Model | SAE | Mean $w_j$ | Pos. frac. | Shuf. $w_j$ |
|---|---|---|---|---|
| Pythia | BatchTopK | 0.068 | 76.2% | 0.001 |
| Pythia | Matryoshka | 0.087 | 91.4% | −0.001 |
| Gemma | BatchTopK | 0.068 | 80.4% | −0.001 |
| Gemma | Matryoshka | 0.091 | 81.9% | −0.001 |

## Appendix E Automated Interpretability Details

Judge gpt-4o-mini explains from top-activation contexts, scored on predicting above-threshold activation on held-out contexts; $f^{(t)}_j$ for baselines, $a^{(t)}_j$ for Persistent; context 1024, ~2M Pile tokens. Also scored top-1%/bottom-50% $\lambda_j$ groups. Tables 4–5: all scores >0.85; high-λ = sustained semantic domains, low-λ = local lexical/syntactic/template patterns.

Table 4 (Pythia-160M):

| Feat. | λ | Explanation | Activation | Score |
|---|---|---|---|---|
| F-5528 | 7.03e-1 | ethnic diversity, discrimination | "ashamed of being Mexican" | 0.93 |
| F-12751 | 4.82e-1 | population studies, cancer | population-based cancer registry | 0.93 |
| F-15986 | 3.81e-1 | high-performance cars | 1971 Hemi 'Cuda | 0.93 |
| F-6538 | 3.79e-1 | weather, climate | El Niño phenomenon | 1.00 |
| F-8449 | 3.69e-1 | Banach spaces, polynomials | polynomials over a Banach space | 1.00 |
| F-1113 | 5.35e-2 | substring "white" | "white space" | 1.00 |
| F-6958 | 4.96e-2 | "function" in JavaScript | .click(function() { | 1.00 |
| F-15079 | 4.81e-2 | "U.S. Pat. No." | U.S. Pat. No. 5,111,981 | 1.00 |
| F-3587 | 1.10e-2 | URLs, web docs | ubuntu.com/10.04/serverguide | 1.00 |
| F-14758 | 4.04e-4 | SQL table commands | CREATE TABLE IF NOT EXISTS | 1.00 |

Table 5 (Gemma-2-2B):

| Feat. | λ | Explanation | Activation | Score |
|---|---|---|---|---|
| F-11205 | 9.18e-1 | cosmology | CMB lensing maps | 0.86 |
| F-5578 | 8.55e-1 | ionisation charge collection, quantum physics | charge collection in gases | 0.86 |
| F-15639 | 8.24e-1 | conflict, ethnic tensions | Katanga separatists | 0.86 |
| F-13900 | 7.85e-1 | microbial biomass, alpine/tundra | Tibetan Plateau meadow | 0.86 |
| F-11933 | 7.70e-1 | self-acceptance narratives | "decided to be myself" | 1.00 |
| F-9996 | 1.03e-3 | software license terms | "See the License…" | 1.00 |
| F-11210 | 1.30e-5 | phrase "all of" | "all of these loci" | 0.93 |
| F-14595 | 7.87e-6 | substring "mk" in math | \mkern | 0.93 |
| F-13259 | 2.73e-6 | quoted terms | "Calgary pizza" | 1.00 |
| F-7307 | 1.88e-6 | sentence-initial "In" | "In this way…" | 0.86 |

## Appendix F Subject Geometry Details

100 sequences, five subjects, last 20 tokens. Representations: raw hidden states, BatchTopK, Temporal high-level group, Persistent high-$\lambda_j$ state; shared t-SNE hyperparameters.

## Appendix G Subject Probing Details

Subject-balanced spliced sequences; whole source documents held out for test. Held-out one-vs-rest macro ROC AUC vs feature fraction 0.1%–100%. Persistent evaluated on $\mathbf{h}^{(t)}$; orderings: decreasing/increasing $\lambda_j$, repeated random permutations, supervised:
$$s_{j,c}=\mathbb{E}[h^{(t)}_j\mid y^{(t)}=c]-\mathbb{E}[h^{(t)}_j\mid y^{(t)}\ne c]\quad(13)$$
on training split, per-subject rankings interleaved round-robin into one ordering. Baselines use full dictionaries.

## Appendix H Prompt Injection Monitoring Details

**Trajectory construction.** Query appended: "should this trajectory be flagged for prompt injection or unsafe instruction hijacking? Answer yes or no:". The yes-minus-no logit alone gives AUC 0.51, hence internal probes.

**Checkpoints and probes.** Offsets 0, 50, 100, 200, 250, 300, 350 after the injected span's final token (matched position for benign), plus final query. Representations: raw, BatchTopK, Matryoshka, Temporal high-level, Persistent $\mathbf{a}^{(t)}$, $\mathbf{h}^{(t)}$, slow state. Logistic regression fit at $d=0$ only; held-out accuracy and ROC AUC.

**Causal intervention.** Top $k=128$ features by |probe coefficient|; weighted decoder columns form $\mathbf{v}_r$, normalised so $\lVert\alpha\mathbf{v}_r\rVert_2=\rho\lVert\mathbf{x}\rVert_2$, $\rho=0.1$, added with both signs. Direction built at $d=0$ and transferred across distances.