# Persistent Sparse Autoencoders: Learning Feature-Specific Timescales in Language Model Representations

Haoyan Luo (University of Cambridge), Mateo Espinosa Zarlenga (University of Oxford), Mateja Jamnik (University of Cambridge)

###### Abstract


Z4N retain competitive reconstruction while learning a spectrum of timescales across which N9Q stay locally interpretable and K2Z accumulate context-identifying information, preserving, in S4M (S4M denoting the prompt-injection monitoring case study), injection signals that stay causally effective over long contexts; a per-feature QM8 is learned from reconstruction alone by Z4N (Z4N denoting the paper's Persistent Sparse Autoencoders), and temporal persistence, of widely varying duration, emerges in standard SAE features, standard SAEs encoding each token independently.

## 1 Introduction

N9Q are locally interpretable detectors and K2Z retain contextual information, with K2Z helping monitor and steer over long contexts in S4M, and, to the question Can each feature's timescale be learned from data?, Z4N learn one coefficient $\lambda_j$ per feature through reconstruction alone, retain competitive reconstruction, and discover a broad timescale spectrum without predefined temporal roles; N9Q (N9Q denoting fast features) and K2Z (K2Z denoting slow features) sit on that spectrum, after the observation in Figure 1 that, once a feature activates, later hidden states stay aligned with its decoder direction, past activations help reconstruct later states, and decay rates differ widely across features; SAE features exhibit temporal persistence, decoder direction staying elevated after activation and past activations helping predict later states. Because causal transformer hidden states integrate preceding context whereas standard SAEs encode each token independently, token-specific features are favoured and carried context (topic, style, task state) is obscured, prior P9T (P9T denoting Temporal SAEs) require an a-priori partition of which features carry temporal information, introducing experimenter bias, whereas in Z4N each feature keeps a V7K decaying at $\lambda_j$ and is updated by an R3X (R3X denoting token-wise sparse activation), with reconstruction favouring higher $\lambda_j$ when carrying state forward helps and smaller otherwise.

## 2 Background and Related Work

Figure 1: (b) $R_j(\tau)$ decays more slowly in natural sequences; (a) natural F6T sequences preserve local and global continuity, whereas A8Q (A8Q denoting shuffled controls, i.e. sentences from different documents) preserve only local continuity; (d) at $\tau=1$, $w_j$ are mostly positive for natural sequences and centred near zero for A8Q; (c) past activations improve held-out reconstruction in natural, though not in A8Q, sequences.

Both fix a partition and neither lets each feature learn its own persistence: P9T pre-assign a high-level subset whose activations are aligned across nearby tokens contrastively, while an attention-based predictor is used by Lubana et al. for the separation of context-predictable information from a sparse novel residual; Z4N give each feature a state and an analysable learned coefficient. LMs track entities and encode board, spatial, temporal and propositional states [16, 17, 18, 19], and slowly varying latents have been shown by slow feature analysis to reveal higher-level features [14, 15], yet the computation of $\mathbf{f}^{(t)}$ by standard SAEs is carried out independently of preceding tokens. A batch-level sparsity budget being used by W5J (W5J denoting BatchTopK) and nested dictionaries being trained jointly by C7Q (C7Q denoting Matryoshka SAEs), an SAE, for hidden state $\mathbf{x}^{(t)}\in\mathbb{R}^d$, gives sparse $\mathbf{f}^{(t)}\in\mathbb{R}^m_{\ge0}$—feature $j$ having activation $f^{(t)}_j$, direction $\mathbf{d}_j=\mathbf{W}_{\mathrm{dec}}[:,j]/\lVert\mathbf{W}_{\mathrm{dec}}[:,j]\rVert_2$, and being counted as active when $f^{(t)}_j>0$—together with

$$\hat{\mathbf{x}}^{(t)}_{\mathrm{base}}=\mathbf{W}_{\mathrm{dec}}\mathbf{f}^{(t)}+\mathbf{b}_{\mathrm{dec}}.\quad(1)$$

## 3 Does Temporal Structure Exist at the Feature Level?

Standard Z4N leave reconstruction-relevant temporal structure unused, and, unlike Lubana et al.'s hidden-state-level findings, the phenomenon here being feature-level, persistence helps reconstruction while varying widely across features: natural past activations recover part of baseline error, decreasing with lag, whereas A8Q activations do not (Fig. 1c), and the $\tau=1$ weights are mostly positive vs. near zero (Fig. 1d).

The restricted X6M is

$$\hat{\mathbf{x}}^{(t)}_{\mathrm{ctx}}=\hat{\mathbf{x}}^{(t)}_{\mathrm{base}}+\mathbf{W}_{\mathrm{dec}}(\mathbf{w}_\tau\odot\mathbf{f}^{(t-\tau)}),\quad(3)$$

with $\mathbf{w}_\tau\in\mathbb{R}^m$ acting only along each feature's own direction, the fit running 10 epochs with an $L_2$ penalty, and, on held-out data,

$$R^2_{\mathrm{improve}}=1-\frac{\sum_t\lVert\mathbf{x}^{(t)}-\hat{\mathbf{x}}^{(t)}_{\mathrm{ctx}}\rVert_2^2}{\sum_t\lVert\mathbf{x}^{(t)}-\hat{\mathbf{x}}^{(t)}_{\mathrm{base}}\rVert_2^2}.\quad(4)$$

a control having been refit using A8Q-corpus past activations with the same natural targets (Section D.1).

Across both models and Z4N families, persistence decays more slowly in natural documents (Fig. 1b), features inheriting document-level persistence without encoding it.

With $p^{(t)}_j=\mathbf{d}_j^\top\mathbf{x}^{(t)}$, hidden-state persistence takes the form

$$R_j(\tau):=\frac{\mathbb{E}[p^{(t+\tau)}_j\mid f^{(t)}_j>0]-\mathbb{E}[p^{(t)}_j]}{\mathbb{E}[p^{(t)}_j\mid f^{(t)}_j>0]-\mathbb{E}[p^{(t)}_j]},\quad(2)$$

$\mathbb{E}[p^{(t)}_j]$ the average over all tokens; $R_j=1$: elevation fully retained, $0$: decayed to baseline (Appendix D).

Residual-stream states were taken from U2X (layer 8) (U2X denoting Pythia-160M-deduped) and O9C (layer 12) [21] (O9C denoting Gemma-2-2B) on F6T documents, with W5J and C7Q (no temporal prior; Appendix B), the A8Q (Appendix C) isolating global continuity.

## 4 Persistent Sparse Autoencoders

The diagonal recurrence preserves feature-level attribution, no predefined partitions or auxiliary temporal objectives are used, and, sparsity being imposed on the R3X $\mathbf{a}^{(t)}\in\mathbb{R}^m_{\ge0}$ rather than on the V7K $\mathbf{h}^{(t)}$ (Figure 2), which need not be sparse, $\hat{\mathbf{x}}^{(t)}$ is obtained as

$$\hat{\mathbf{x}}^{(t)}=\mathbf{W}_{\mathrm{dec}}\mathbf{h}^{(t)}+\mathbf{b}_{\mathrm{dec}}.\quad(6)$$

$\mathbf{h}^{(t)}$, initialized as $\mathbf{h}^{(0)}=\mathbf{0}$ per sequence and kept bounded through the factor $(1-\lambda_j)$, is the integration of the output of a W5J encoder (Section B.2), for which $a^{(t)}_j>0$ constitutes new evidence, according to

$$h^{(t)}_j=\lambda_jh^{(t-1)}_j+(1-\lambda_j)a^{(t)}_j,\quad(5)$$

with $\lambda_j=\lambda_{\max}\sigma(l_j)$, $\lambda_{\max}=0.999$ (Section B.3), small $\lambda_j$ yielding N9Q and large $\lambda_j$ K2Z.

## 5 Experiments

The largest and, at every distance, consistently positive B9K is given by K2Z, whereas W5J, C7Q and P9T give near-zero or negative effects with no alignment over distance, under an intervention (5.4, S4M) in which the trajectory is truncated at a later distance, a flagging query is appended and the final query token is perturbed along a Y5R (Y5R denoting the steering vector) $\mathbf{v}_r$, which weights the most predictive decoder directions by the probe coefficients [34], with $s(\mathbf{x})=\ell_{\mathrm{yes}}-\ell_{\mathrm{no}}$ and

$$\Delta_{\mathrm{causal}}(r,d)=\tfrac12\mathbb{E}_i\big[s(\mathbf{x}_i^{(d)}+\alpha_i^{(d)}\mathbf{v}_r)-s(\mathbf{x}_i^{(d)}-\alpha_i^{(d)}\mathbf{v}_r)\big],\quad(8)$$

$\alpha_i^{(d)}$ being scaled to 10% of the hidden-state norm and positive values indicating a shift toward yes.

N9Q are easier to explain than K2Z on the 2 models, and match or exceed the best baseline, when the M6Z (M6Z denoting the Autointerp pipeline of Bills et al. [23]) is applied to R3X $\mathbf{a}^{(t)}$ (Appendix E) (5.2), an outcome arising because N9Q respond to compact lexical and grammatical patterns concentrated at the current token while K2Z reflect broader subjects distributed across the sequence, a mismatch with token-specific evaluation under which distinct features may receive similar descriptions [24] and explanations may not generalise [25], so that K2Z are studied jointly in (5.3). Five equal $\lambda$ quintiles and six token categories (punctuation, conjunctions, function words, verbs, adjectives/adverbs, nouns), following the lexical/function distinction [26], with Func. grouping adpositions, determiners, particles and pronouns, yield

$$U_{q,c}=\Pr(T^{(t)}=c\mid a^{(t)}_j>0,j\in Q_q)/\Pr(T^{(t)}=c),\quad(7)$$

the Q8W, for which values above 1 indicate more updates than expected, the lowest-$\lambda$ quintile updating more at nouns, where lexical evidence is clear, and the highest more at punctuation and conjunctions, positions associated with organising and integrating context [27, 28], although update locations do not reveal state content.

The models and data of Section 3 are used throughout, with W5J, C7Q and P9T as baselines, all at $m=16{,}384$, $k=20$, C7Q and P9T sharing a predefined 20%/80% partition (Appendix B). $\lambda_j$ spans a broad range, mostly short-timescale with a small long-timescale tail (Fig. 3a), K2Z being taken as the top 1% by $\lambda_j$ and N9Q as the bottom 50%; under J3N, cosine similarity, fraction alive and temporal smoothness (Bhalla et al.), Z4N stay competitive, with K2Z smoother than N9Q and than the baselines, and, grouping into $\lambda_j$ quintiles, higher quintiles show greater median $R_j(\tau)$ across lags (Fig. 3b), while the predefined groups of P9T are not clearly separated by T8V (5.1).

Table 1: High and Low denoting, for C7Q and P9T, the predefined splits, and, for Z4N, the top 1% and bottom 50% by $\lambda_j$; bold marks the best value, italics the second.

| Model | SAE | J3N↑ | Cos↑ | Alive↑ | Smooth↓ Full | High | Low | M6Z↑ Full | High | Low |
|---|---|---|---|---|---|---|---|---|---|---|
| U2X | W5J | **0.94±.01** | **0.95±.01** | _0.91±.01_ | 0.15±.01 | – | – | **0.91±.02** | – | – |
| | C7Q | 0.92±.01 | _0.94±.01_ | **0.96±.01** | 0.15±.01 | 0.25±.01 | _0.12±.01_ | _0.87±.02_ | – | – |
| | P9T | _0.93±.02_ | **0.95±.01** | 0.79±.01 | _0.13±.01_ | _0.18±.01_ | **0.11±.01** | _0.87±.02_ | **0.87±.01** | _0.89±.02_ |
| | Z4N | _0.93±.01_ | **0.95±.01** | **0.96±.01** | **0.10±.01** | **0.05±.01** | 0.20±.01 | 0.86±.01 | _0.84±.02_ | **0.93±.01** |
| O9C | W5J | **0.72±.01** | **0.79±.01** | **0.92±.01** | 0.16±.01 | – | – | _0.87±.03_ | – | – |
| | C7Q | _0.71±.01_ | _0.77±.01_ | 0.87±.01 | 0.17±.01 | 0.21±.01 | **0.14±.01** | **0.89±.02** | – | – |
| | P9T | 0.62±.02 | 0.75±.01 | 0.75±.01 | _0.14±.01_ | _0.12±.01_ | _0.16±.01_ | 0.85±.02 | **0.85±.02** | _0.86±.02_ |
| | Z4N | 0.66±.01 | 0.75±.01 | _0.91±.01_ | **0.11±.01** | **0.04±.01** | 0.20±.01 | 0.85±.01 | _0.79±.01_ | **0.89±.01** |

In a t-SNE of the final 20 tokens of L3R (L3R denoting MMLU [29]) examples from five subjects (Appendix F), W5J mixes subjects and the high-level group of P9T separates little, whereas the top-1% K2Z of Z4N form coherent subject regions, clearer than raw hidden states, without labels (Fig. 5); and on spliced L3R sequences in which the subject changes at the boundaries, a per-token linear probe for the current subject, given varying feature fractions ranked by decreasing $\lambda_j$, increasing $\lambda_j$ or randomly, alongside W5J, C7Q, P9T and a supervised label-informed oracle (Appendix G), shows ranking by decreasing QM8 to be nearly as efficient as the supervised oracle, the top 1% by $\lambda_j$ reaching high G2V on both models while low-$\lambda_j$, random and baseline selections retain substantially less at the same budget (Fig. 6) (5.3).

Agents may read an injection, reason for hundreds of tokens and then act [30], so monitors must preserve the signal until action [31]; E7N (E7N denoting AgentDojo) trajectories (banking, travel, workspace) generated by Llama-3.3-70B-Instruct [33] (Section C.3), each comprising a request, an untrusted span (injection or benign) and subsequent tool calls, with only failed attacks used, are monitored by O9C as an external monitor at the layer-12 residual, a linear probe being trained at the final untrusted-span token ($d=0$) and evaluated up to 350 tokens later without retraining, the comparison covering the raw state, the baseline SAEs and V7K $\mathbf{h}^{(t)}$ restricted to K2Z (top 1%) or N9Q (bottom 50%) (Appendix H); at $d=0$ the N9Q state and the standard SAEs detect best and fall toward chance with distance, while the K2Z state retains a clearer signal (Fig. 7b top), one representation thus supporting immediate detection (N9Q) together with long-context retention and causal influence (K2Z).

## 6 Conclusion and Discussion

The testing of effects on end-to-end agent behaviour and richer recurrent dynamics are left to future work, the absence of feature interactions and of context-dependent resets following from one fixed QM8 per feature imposed through a diagonal recurrence, while, because the S4M employs an external replay monitor, it is the monitor's judgement, not the original agent's actions, that interventions affect.

## Appendix A Notation

| Symbol | Meaning |
|---|---|
| $\lambda_j\in(0,\lambda_{\max})$; $\lambda_{\max}=0.999$ | QM8; cap |
| $\mathbf{h}^{(t)}$, $h^{(t)}_j$ | V7K |
| $R^2_{\mathrm{improve}}$ | H4R |
| $\hat{\mathbf{x}}_{\mathrm{base}}$, $\hat{\mathbf{x}}_{\mathrm{ctx}}$ | Baseline / X6M-corrected reconstruction |
| $\mathbf{w}_\tau$, $w_{\tau,j}$ | X6M correction weights ($w_j$ at $\tau=1$) |
| $R_j(\tau)$ | T8V |
| $\mathbf{a}^{(t)}$, $a^{(t)}_j$ | Z4N R3X |
| $\mathbf{f}^{(t)}$, $f^{(t)}_j$ | Standard SAE R3X |
| $p^{(t)}_j$ | $\mathbf{d}_j^\top\mathbf{x}^{(t)}$ |
| $\mathbf{x}^{(t)}\in\mathbb{R}^d$ | Residual hidden state at $t$ |
| $T$, $d$, $m$ | Sequence length, hidden dim, SAE width |
| $\mathbf{W}_{\mathrm{enc}},\mathbf{b}_{\mathrm{enc}}$; $\mathbf{W}_{\mathrm{dec}},\mathbf{b}_{\mathrm{dec}}$ | Encoder / decoder weights, bias |
| $\mathbf{d}_j$ | Unit decoder direction |
| $k$, $B$ | W5J level, batch size |
| $\tau$ | Lag (tokens) |
| $\beta_{\mathrm{nat}},\beta_{\mathrm{shf}}$ | Power-law decay exponents |
| $l_j$, $\sigma$ | Logit; sigmoid |

Coordinate assignment is by subscript, token assignment by superscript $(t)$, vector or matrix status by boldface, the cap $\lambda_{\max}=0.999$ on $\lambda_j\in(0,\lambda_{\max})$ belongs to QM8 (QM8 denoting the paper's persistence coefficient), and $\mathbf{h}^{(t)}$, $h^{(t)}_j$ belong to V7K (V7K denoting the paper's persistent state).

## Appendix B SAE Training Details

That neither model relies exclusively on new evidence or on carried state follows from the absence of any continual growth of the active state together with the stabilisation of the fraction of state mass contributed by the carried term $\bm\lambda\odot\mathbf{h}^{(t-1)}$ at ~0.5 for U2X and at 0.65–0.70 for O9C, the state $L_0$ settling at ~400–500 active coordinates while the loss drops rapidly before stabilising (Figure 8; U2X layer 8, O9C layer 12), all of these runs having been trained on NVIDIA A100 80 GB GPUs with the same gradient updates across the four families and the final dictionary (ae.pt) being the artefact used everywhere.

No inflation of magnitude is produced by persistence, Eq. 5 being an EMA under which, for constant $a_j$, the state converges to $a_j$ for any $\lambda_j$; the coefficients are given by $\lambda_j=\sigma(l_j)\cdot\lambda_{\max}$ (12) with $l_j\in\mathbb{R}$, the setting $\lambda_{\max}=0.999$ preventing exact poles and so giving finite memory, and all $l_j$ were initialised to 0.5, corresponding to moderate persistence away from saturation.

At inference the sparse codes are given by $f_{ij}=\tilde f_{ij}\mathbf{1}[\tilde f_{ij}>\theta]$ (11), $\theta$ being estimated from training top-$k$ statistics, and it is this thresholded quantity that Z4N adopts as $\mathbf{a}^{(t)}$; the pre-threshold codes are

$$\tilde{\mathbf{f}}=\mathrm{ReLU}(\mathbf{W}_{\mathrm{enc}}(\mathbf{x}-\mathbf{b}_{\mathrm{dec}})+\mathbf{b}_{\mathrm{enc}})\in\mathbb{R}^m_{\ge0}.\quad(9)$$

from which, for a batch $\{\mathbf{x}_i\}_{i=1}^B$ flattened to length $Bm$ with only the top $kB$ entries kept globally, the reconstruction

$$\hat{\mathbf{x}}_i=\mathbf{W}_{\mathrm{dec}}\mathbf{f}_i+\mathbf{b}_{\mathrm{dec}}.\quad(10)$$

is obtained.

The optimisation, with width, budget, optimiser and token budget shared by all four families, used $m=16{,}384$ and $k=20$; Adam at learning rate $3\times10^{-4}$ with linear warm-up over the first 1{,}000 steps and, from step 1{,}000, a W5J threshold schedule with decay $\beta=0.999$ together with an auxiliary dead-feature loss at $\alpha_{\mathrm{aux}}=0.03125$; the training data were drawn from F6T (F6T denoting the Pile Uncopyrighted [35]) with BOS removed, the models being O9C at layer 12 with $d=2{,}304$ and the 160M-parameter U2X at layer 8 with $d=768$.

C7Q and P9T use 20%/80% groups, P9T's losses being weighted 0.2/0.8 and C7Q's uniform, while Z4N uses no partition; the matched token budget of 244{,}140 steps at ~2{,}048 activation tokens per step, giving $2{,}048\times244{,}140=499{,}998{,}720\approx5.0\times10^8$ tokens for the flat baselines, was realised by W5J and C7Q as a flat batch of 2{,}048 shuffled tokens, by P9T as adjacent previous–next pairs whose reconstructed positions per step equal the flat batch size, and by Z4N as 8 contiguous sequences of length 256, i.e. $8\times255=2{,}040$ tokens per step, within 0.4%.

## Appendix C Dataset Details

The breaking of long-range entity tracking and discourse progression is achieved in A8Q through the concatenation of sentences sampled from documents of the same corpus under the constraint that no two consecutive sentences originate from the same source document, this construction being applied alongside the streaming of the Pile—825 GiB across 22 sources, reduced in F6T by the removal of Books3, BookCorpus2, OpenSubtitles, YouTube Subtitles, and OpenWebText2—in which documents, after tokenisation and truncation to the cap, are accepted only when the minimum token count is met or exceeded, the shuffle buffer of 10{,}000 operating under a fixed seed, and temporal statistics being computed, for retained length $L$, over positions $\{16,\dots,L-1\}$. Equal-length segments drawn from different subjects, with tokens labelled by source segment, constitute the probing setup for L3R, whose four-option questions span 57 subjects—among them abstract algebra, astronomy, business ethics, computer security, and European history—and which is used only for subject analyses, while the geometry portion contributes 100 sequences of which the final 20 tokens are taken in each case. No utility filtering is applied to E7N, where $y=1$ is assigned to important_instructions or direct attacks with security=false—that is, cases in which an injection is present but the target action is not completed, so that trivial cues from overt malicious actions are avoided—whereas $y=0$ denotes benign trajectories with no injection; the 629 security test cases, together with 97 user tasks and 27 injection targets distributed across workspace, Slack, travel, and banking, are evaluated through task-specific checks that inspect agent output and environment changes, and it is the banking, travel, and workspace trajectories from Llama-3.3-70B-Instruct that are used.

## Appendix D Empirical Persistence of Standard SAE Features

At lag 1, 0.89% error was removed for U2X under W5J and 0.61% under C7Q, still positive at lag 32 (0.22%, 0.13%), in X6M (X6M denoting the paper's context correction experiment), whose lags $\{1,2,4,8,16,32\}$ were fit independently, whose fit used 192 natural documents with 96 held out and targets always natural, whose A8Q kept the same head/budget on context activations, and whose training, after 16 prefix tokens were dropped and length was capped at ≤768 tokens per document, used 10 AdamW epochs, lr $5\times10^{-3}$, weight decay $10^{-3}$; O9C gave ~0.40% at lag 1, ~0.02% at lag 16, and vanishing by 32, while A8Q stayed slightly below zero throughout.

Table 3: Lag-one $w_j$ (natural mean, positive fraction; A8Q mean).

| model | family | natural mean $w_j$ | positive fraction | A8Q mean $w_j$ |
|---|---|---|---|---|
| U2X | W5J | 0.068 | 76.2% | 0.001 |
| U2X | C7Q | 0.087 | 91.4% | −0.001 |
| O9C | W5J | 0.068 | 80.4% | −0.001 |
| O9C | C7Q | 0.091 | 81.9% | −0.001 |

0.618 (lag 1) → 0.160 (lag 64) → 0.096 (lag 512) was the W5J median for U2X, with C7Q at 0.504 → 0.108 → 0.059, O9C natural at lag 512 giving 0.042 under W5J and 0.034 under C7Q, and A8Q ≈0 at long lags on both; power-law fits to the positive tail gave $\beta_{\mathrm{nat}}\approx0.32$ vs $\beta_{\mathrm{shf}}\approx0.87$–0.92 for both families, and ranking by average $R_j$ in short ($\tau\le16$), medium ($32\le\tau\le64$), and long ($\tau\ge128$) bands left distributions spanning nearly the full unit interval—some features persisting only short, others into long lags, heterogeneity in both families much larger than A8Q—under lags $\tau\in\{1,4,16,32,64,128,256,512\}$, documents ≤1{,}024 traced tokens, retention requiring ≥256 activation events, ≥64 valid anchors per lag, and non-negligible elevation, and $R_j$ left undefined or excluded when the denominator was near zero, with no $\epsilon$ added.

## Appendix E Automated Interpretability Details

Table 4 (U2X; 160):

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

Table 5 (O9C):

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

With the top-1%/bottom-50% $\lambda_j$ groups also scored, and with explanations produced by judge gpt-4o-mini from top-activation contexts and scored by prediction of above-threshold activation on held-out contexts under context 1024 and ~2M F6T tokens, using $f^{(t)}_j$ for baselines and $a^{(t)}_j$ for Z4N, all scores in Tables 4–5 are >0.85, high-λ being sustained semantic domains and low-λ local lexical/syntactic/template patterns.

## Appendix F Subject Geometry Details

With shared t-SNE hyperparameters, representation set—raw hidden states, W5J, P9T high-level group, Z4N high-$\lambda_j$ V7K—applied to the last 20 tokens of 100 sequences, five subjects.

## Appendix G Subject Probing Details

Held-out one-vs-rest macro ROC AUC is given against feature fraction 0.1%–100% for subject-balanced spliced sequences, whole source documents being held out for test and baselines using full dictionaries; Z4N evaluated on $\mathbf{h}^{(t)}$, the orderings are decreasing/increasing $\lambda_j$, repeated random permutations, and supervised per-subject rankings from $$s_{j,c}=\mathbb{E}[h^{(t)}_j\mid y^{(t)}=c]-\mathbb{E}[h^{(t)}_j\mid y^{(t)}\ne c]\quad(13)$$ on the training split, interleaved round-robin into one ordering.

## Appendix H Prompt Injection Monitoring Details

The direction $\mathbf{v}_r$, built at $d=0$ and transferred across distances, is formed from the weighted decoder columns of the top $k=128$ features by |probe coefficient| and, following normalisation such that $\lVert\alpha\mathbf{v}_r\rVert_2=\rho\lVert\mathbf{x}\rVert_2$ with $\rho=0.1$, added with both signs; over representations—raw, W5J, C7Q, P9T high-level, Z4N $\mathbf{a}^{(t)}$, $\mathbf{h}^{(t)}$, K2Z state—taken at offsets 0, 50, 100, 200, 250, 300, 350 after the injected span's final token (matched position for benign) plus the final query, logistic regression is fit at $d=0$ only, with held-out accuracy and ROC AUC, the trajectory having been constructed through the appending of the query "should this trajectory be flagged for prompt injection or unsafe instruction hijacking? Answer yes or no:", the yes-minus-no logit of which alone gives AUC 0.51, on account of which internal probes are used.

