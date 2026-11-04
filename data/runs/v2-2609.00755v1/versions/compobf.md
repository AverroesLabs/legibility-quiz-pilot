# S³martCirc: Self-supervised Smart Circuit Discovery

Wendy Zheng, Yinhan He (University of Virginia), Liang Wu (Nokia), Jundong Li (University of Virginia)

Code — https://anonymous.4open.science/r/s3martcirc

###### Abstract


Existing circuit discovery methods are outperformed by Q7Z, whose abstraction of node behavior into two general computational roles and definition of a quantitative metric (Z4P denoting the paper's functional interpretation metric) for assigning those roles leave importance and role discovered jointly, while unifying them faces two challenges, (2) role identification subjective rather than quantified and (1) roles tied to specific nodes, limiting generalization, given that a component's importance and functional role are codependent; Mechanistic interpretability (MI) of LLMs usually runs in two stages: functional interpretation after circuit discovery (V3K, e.g., attention heads or feedforward neurons).

## Introduction

Better task-relevant circuits and recovery of known, human-validated circuits across multiple LLM architectures constitute the empirical validation associated with the MI framework (Q7Z denoting S³martCirc) in which discovery and interpretation are coupled through bidirectional X8U (X8U denoting alternating optimization), with the two stages reversible—first identification of the functional operations a task requires, then finding of the nodes implementing them—since importance and functionality are interdependent: a node matters because of its function, and its function is defined by its task contribution, so discovery and interpretation should be one jointly optimized problem, while the obstruction is (1) that interpretations are node-specific, prior roles (e.g., name mover, S-inhibition heads) being bound to a node and a task and giving no shareable abstraction for automated pipelines, and (2) that functional roles are subjective, based on human judgment and hard to formalize; given that LLMs' core computation, matrix multiplication, mainly enables two operations, computational transformation and information propagation, fine-grained, task-specific roles are abstracted, as general functional roles, into a coarse dichotomy abstracting task-specific interpretations into two general, quantifiable categories describing how a node processes information, not what, with semantic specificity traded for generality and quantifiability so it can enter an optimization objective.

## Preliminaries and Definitions

That systematic principles underlie the decomposition of tasks by LLMs is suggested by the role mapping (Fig. 1), in which, for E5M ("When John and Mary went to the store, John gave a drink to [Mary]"), S-inhibition heads, through which subject info is routed, together with previous token heads, are assigned the T9Q role, whereas name mover and negative name mover heads, effecting amplification of specific names in logits, are assigned the M2X role, and, for U8J ("The Three Letter Acronym" → "TLA"), previous token and bridge heads, through which letter info is propagated, are T9Q while letter mover heads are M2X, these constituting two roles (M2X denoting functional nodes, i.e., computational transformation, outputs differing substantially from inputs in representation/semantics; T9Q denoting passthrough nodes, i.e., information propagation, the relocation of input features downstream with minimal modification). The sharing of interpretations (e.g., previous token, mover heads) by circuits for the distinct tasks E5M and U8J is consistent with the reuse of components across tasks with consistent roles. Modeled as a computational graph is an LLM with output distribution $P$ and vocabulary $V$: nodes being components (attention heads or feedforward neurons), edges being information flow via the residual stream; a node effects a mapping of $X\in\mathbb{R}^{T\times d_{m}}$ to $Y\in\mathbb{R}^{T\times d_{n}}$ ($T$ tokens, $d_m$ hidden dim, $d_n$ node output dim; architecture-determined for heads, $d_n=1$ for neurons), and the subgraph responsible for task performance being a circuit; the node set and graph are identical across all compared methods.

## Self-supervised Smart Circuit Discovery

In Q7Z, high $\mathrm{Z4P}$ pushes $c_{n,f}$ up and low $\mathrm{Z4P}$ pushes $c_{n,p}$ up through stage 2, R8D (R8D denoting Important Node Classification), which assigns each candidate a role via a task-dependent metric because nodes doing nontrivial computation should alter relational structure between tokens while propagating nodes should preserve it; with $N_i$ the important nodes, $$L_{\text{class.}}=\sum_{n_{i}\in N_{i}}\Big[-c_{n_{i},f}\,\mathrm{Z4P}(X_{n_{i}},Y_{n_{i}})-\frac{c_{n_{i},p}}{\mathrm{Z4P}(X_{n_{i}},Y_{n_{i}})}\Big]\quad(4)$$ where $$\mathrm{Z4P}(X,Y)=\frac{\|\text{Sim}(X)-\text{Sim}(Y)\|_{F}}{\|\text{Sim}(X)\|_{F}}\quad(3)$$ and, for $A\in\mathbb{R}^{T\times d_{n}}$, $\text{Sim}(A)=\hat{A}\hat{A}^{T}\odot(\mathbf{1}\mathbf{1}^{T}-I)$, with $\hat{A}$ row-wise $L_2$-normalized and the mask excluding self-similarity. The candidates consumed by that assignment are produced by stage 1, V3K (V3K denoting Important Node Discovery), which separates $c_{n,p}+c_{n,f}$ from $c_{n,u}$ and takes nodes with $c_{n,p}+c_{n,f}>c_{n,u}$ as candidates, since removing truly important nodes should degrade performance; with $P_{\text{orig}}$ the full model and $P_{\text{mask}}$ the model with unimportant nodes ($c_{n,u}>c_{n,p}+c_{n,f}$) masked, $$L_{\text{disc.}}=\alpha\sum_{v\in V}P_{\text{orig}}(v)\log\frac{P_{\text{orig}}(v)}{P_{\text{mask}}(v)}-(1-\alpha)\log P_{\text{mask}}(y)\quad(2)$$ $y$ is the ground-truth token and $\alpha\in[0,1]$ trades faithfulness (KL) against task performance (CE). The coefficients just used belong to each node $n$ as learnable masking coefficients $c_{n}=\{c_{n,u},c_{n,p},c_{n,f}\}$, probabilities of being unimportant, T9Q, M2X, with $c_{n,u}+c_{n,p}+c_{n,f}=1$; training uses soft masking and evaluation uses hard assignments $$\hat{c}_{i}=\begin{cases}1,&i=\arg\max_{j}c_{j}\\0,&\text{otherwise}\end{cases}\quad(1)$$ Regularization on both stages (confidence → binary values, validity → sum to one, sparsity → minimal circuits), with $C$ all masking coefficients, is $$L_{\text{sparsity}}=\frac{1}{|C|}\sum_{c_{n}\in C}\Big(\sum_{v\in c_{n}}\tfrac{1}{2}|v(1-v)|+\tfrac{1}{2}|1-(c_{n,u}+c_{n,p}+c_{n,f})|+(c_{n,p}+c_{n,f})\Big)\quad(5)$$ and X8U of the stages is used for bidirectional influence, while role assignment from H7B (H7B denoting Random) initialization is unreliable, so a Q3H optimizes only $L_{\text{disc.}}$ first.

## Experiments

On every task and model Q7Z is found to beat all baselines on both X1R and N7V (effectiveness, RQ1), a superiority attributable to the fact that patching methods, evaluating nodes independently, may miss interactions, while Y4F and S3V, although they likewise use continuous masks, do much worse, whence the guidance contributed by Z4P exceeds what masking alone provides; nor are the circuits Q7Z yields always the smallest, yet the degradation induced at comparable size is far larger, as on E5M with F6S, where -69.00% is obtained with 35 nodes against the -5.67% of G8C with 47.67 nodes.

Table 1: Means±std over three runs are reported; – marks cases in which no circuit was found among the selected nodes, and * marks one run only (48-hour limit).

| Model | Method | U8J K6N | U8J X1R | U8J N7V | E5M K6N | E5M X1R | E5M N7V |
|---|---|---|---|---|---|---|---|
| C4T | D2G | 35.00±4.32 | -54.67±38.33 | -3.54±38.33 | 39.33±10.40 | -70.71±25.89 | -9.11±25.89 |
| | P5L | 37.33±2.05 | -0.17±0.24 | 0.46±0.24 | 48.00±2.16 | -12.73±8.01 | -5.14±8.01 |
| | G8C | 49.67±11.44 | -2.00±1.08 | -0.59±1.08 | 38.00±5.10 | -0.17±0.24 | -0.29±0.24 |
| | S3V | – | – | – | – | – | – |
| | Y4F | 32.33±37.97 | 0.00±0.00 | 0.07±0.00 | 38.33±33.81 | 0.00±0.00 | -0.02±0.00 |
| | H7B | 60.33±7.76 | 0.00±0.00 | 0.04±0.00 | 77.33±3.40 | 0.00±0.00 | -0.11±0.00 |
| | Q7Z | 60.33±7.93 | -97.50±1.08 | -5.00±1.08 | 76.00±5.72 | -95.99±2.85 | -10.42±2.85 |
| F6S | D2G | – | – | – | 13.00±2.00 | -59.17±41.90 | -2.37±41.90 |
| | P5L | – | – | – | 71.67±2.05 | -1.50±0.41 | -0.35±0.41 |
| | G8C | 54.67±2.62 | -2.17±0.24 | -0.34±0.24 | 47.67±3.30 | -5.67±3.70 | -0.35±3.70 |
| | S3V | 29.67±12.66 | 0.00±0.00 | -0.08±0.00 | 7.33±0.94 | 0.00±0.00 | 0.00±0.00 |
| | Y4F | 70.67±3.09 | 0.00±0.00 | 0.01±0.00 | 70.67±3.09 | 0.00±0.00 | 0.01±0.00 |
| | H7B | 60.00±9.93 | 0.00±0.00 | -0.00±0.00 | 62.33±6.18 | 0.00±0.00 | -0.01±0.00 |
| | Q7Z | 47.33±2.05 | -57.33±3.40 | -3.17±3.40 | 35.00±3.74 | -69.00±42.43 | -2.96±42.43 |
| J9W | D2G | 32.00±0.00* | -100.00±0.00* | -1.00±0.00* | 12.00±0.00* | -33.00±0.00* | -0.33±0.00* |
| | P5L | 64.33±3.30 | -0.17±0.24 | -0.48±0.24 | 29.00±3.56 | -0.50±0.41 | -0.91±0.41 |
| | G8C | 32.33±8.34 | -0.17±0.24 | -0.12±0.24 | 35.00±5.35 | -0.33±0.47 | -0.19±0.47 |
| | S3V | 14.67±0.94 | 0.00±0.00 | 0.29±0.00 | – | – | – |
| | Y4F | 46.00±4.24 | 0.00±0.00 | -0.01±0.00 | 46.00±4.24 | 0.00±0.00 | -0.01±0.00 |
| | H7B | 52.67±9.74 | 0.00±0.00 | -0.00±0.00 | 48.00±6.98 | 0.00±0.00 | 0.01±0.00 |
| | Q7Z | 12.00±1.63 | -100.00±0.00 | -10.97±0.00 | 16.33±4.50 | -100.00±0.00 | -12.01±0.00 |

The worst of the variants is B9K, whose U8J X1R drop of only -4.33% against -97.5% arises because V3K carried out on all nodes first yields an overly broad, noisy candidate set, so that the resulting circuits include non-critical nodes and omit key ones; T5R, at ≈88% on E5M and 42% on U8J much better than B9K yet still below Q7Z, indicates that M2X priors help but need X8U; and L6M drops consistently (67% on E5M, 66% on U8J), random coefficients giving unreliable early gradients, which causes unstable optimization and suboptimal circuits (ablations, RQ4).

Performance improves with $k$, stabilizing at $k=6$, the X1R drop moving from -71% ($k=2$) to ≈-96% ($k=6$) while recovery increases monotonically, and $\alpha=0.75$ is best for both metrics, for although KL should dominate for E5M, removing CE ($\alpha=1.0$) degrades sharply, which explains the weaker baselines relying on output-distribution differences, since some critical components drive correct behavior without large logit changes (hyperparameter sensitivity, RQ3); the analysis underlying these numbers, run on E5M with C4T (Fig. 5), counts each iteration as one V3K step plus $k$ R8D steps, with $k\in\{2,4,6,8,10\}$ and $\alpha\in\{0.0,0.25,0.5,0.75,1.0\}$.

Of the manually verified E5M circuit, the percent recovered within end-to-end circuits thresholded at 50/100/150/200 nodes (Fig. 3), 100% being unattainable given the node count, is higher for Q7Z than for most baselines and comparable only to D2G, the optimization-based methods recovering much less even at 200 nodes, so that alignment with human-verified interpretations obtains and credibility is added where ground truth is unavailable (recovery of known circuits, RQ2); and on U8J (Fig. 4), nodes being written as layer.head, e.g., 8.11 = layer 8, head 11, 4 of 8 nodes (50%) are recovered by Q7Z, the classifications matching Fig. 1 — Previous Token 1.0 T9Q, Letter Mover 11.4 M2X — while 8.11 is labeled T9Q and 10.10 M2X, though prior work shows both can act as Bridge or Letter Mover depending on intervention, and because human analysis uses isolated interventions to find what a head can do whereas Q7Z measures the dominant contribution across samples, this reflects multi-functionality, not misclassification.

The variants (Fig. 6) are L6M, in which Q3H is omitted, B9K, in which V3K is run in full and R8D only afterwards, and T5R, in which R8D precedes V3K; each stage refines the other, V3K using evolving roles and R8D working on increasingly focused candidates.

Nodes not connected to input or output having been removed, the end-to-end circuit is obtained, and its size, K6N (K6N denoting Circuit Size), is what is reported; evaluation masks only circuit nodes ($P_{\text{circ}}$), X1R being the percent accuracy drop original→$P_{\text{circ}}$ and N7V the difference in logit of the original model's predicted answer between original and $P_{\text{circ}}$.

Among the baselines, the patching-based ones are D2G, ablating each node with a corrupted activation and measuring the logit-difference change, P5L, a linear approximation in which the clean–corrupted activation difference is multiplied by the gradient of the logit-based metric, and G8C, with integrated gradients along the corrupted→clean path, while the optimization-based ones are S3V (S3V denoting IBCircuit, the information-bottleneck method) and Y4F, gradient-based edge pruning adapted to nodes, and H7B, taking $k$ random nodes, serves as a lower bound.

The models are C4T, F6S and J9W (C4T denoting GPT-2, the well-studied MI benchmark, F6S denoting Llama 3.2 1B, and J9W denoting Qwen 3 1.7B, the latter two for architecture generalization), and the tasks, both with circuits previously uncovered in C4T, are E5M and U8J.

## Related Works

(3) the spanning of attention heads and MLP neurons, (2) the joint optimization of importance and role by X8U, and (1) the sourcing of roles from a task-agnostic quantitative activation metric rather than hand-specified desiderata or ablation signs mark Q7Z as different, while continuous masks under sparsity (e.g., S3V, behavior-specific mask learning) capture interactions better than patching yet model only which components matter; because interventions are costly and isolate components and interpretation is subjective, most work is where-then-what—localize via patching or gradient approximations, then interpret manually—and, with components reused with consistent roles, roles localizing to individual neurons, and structurally distinct circuits functionally equivalent, an abstract, quantitative role is motivated.

## Conclusion

That a node may be labeled differently across tasks follows from roles being defined relative to a task's activations, not intrinsically, and the 2 roles, intentionally coarse, fail to recover fine-grained manual roles—limitations; on C4T (2), F6S (3.2), and J9W (3), baselines are outperformed by Q7Z, by which the discovery of circuits and the interpretation of node roles are jointly carried out.

## Appendix A Implementation Details

P5L and G8C on J9W were run on an RTX A6000 48GB, every other run having been executed on NVIDIA RTX A4000 16GB and A16 16GB GPUs, and the runtimes measured under this allocation are reported, in seconds, in Table 2.

Table 2: Runtime (s)

| | | D2G | P5L | G8C | S3V | Y4F | H7B | Q7Z |
|---|---|---|---|---|---|---|---|---|
| C4T | U8J | 6239.72 | 299.54 | 3452.48 | 426.70 | 3391.87 | 3.98 | 197.14 |
| | E5M | 7943.71 | 267.94 | 2801.64 | 467.19 | 3732.88 | 0.13 | 264.44 |
| F6S | U8J | 61510.09 | 1114.46 | 12770.96 | 1266.76 | 11470.94 | 2.22 | 507.67 |
| | E5M | 64639.54 | 1128.62 | 12344.18 | 1617.78 | 11645.51 | 3.37 | 901.02 |
| J9W | U8J | 168689.44 | 829.43 | 11151.09 | 3845.65 | 19072.51 | 3.54 | 982.16 |
| | E5M | 187663.88 | 783.17 | 11148.12 | 5152.79 | 21000.88 | 2.36 | 1047.59 |

The configurations were 0.001, 0.005, 50, 10, 0.9, 0.75 for J9W E5M; 0.01, 0.005, 60, 60, 0.5, 0.25 for C4T U8J; 0.001, 0.005, 75, 5, 1.0, 0.25 for F6S E5M; 0.001, 0.005, 50, 5, 0.7, 0.75 for J9W U8J; 0.005, 0.0005, 60, 5, 0.3, 0.75 for C4T E5M; and 0.001, 0.005, 50, 5, 1.0, 0.75 for F6S U8J, each sextuple being read as (lr, weight decay, epochs, warmup, sparsity coef., α), with the seed fixed at 42 throughout.

For U8J, the clean and corrupted prompts share the template "The [A_RAND_WORD] [B_RAND_WORD] [C_RAND_WORD] (AB", which was filled, for each pair, with matching random words, the resulting pool being shuffled and its first 1000 items taken; the words had been drawn from the wonderwords RandomWord pool at 500 per letter and were retained only when " Word" tokenizes to two tokens, the first of them the space-prefixed capital, letters left without valid words being dropped, and an abbreviation was counted as valid when three letters tokenize to three tokens. For E5M, corruption consists of the swapping of the second name in "When [NAME_1] and [NAME_2] went to the [PLACE], [NAME_1] handed the [OBJECT] to", a template filled with every valid combination of the names and place/object pairs of Wang et al. that survived the requirement of being single-token, the balanced set of 1000 samples being assembled by drawing $1000/p$ samples per pair, with $p$ the number of valid pairs. Generation was carried out per tokenizer, so that token limits were respected.

## Appendix B Additional Experiment Results

Q7Z's keeping of the highest percent recovered holds when, on U8J (Fig. 7), Section 4.3's first experiment is repeated.

