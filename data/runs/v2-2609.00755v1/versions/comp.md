# S³martCirc: Self-supervised Smart Circuit Discovery

Wendy Zheng, Yinhan He (University of Virginia), Liang Wu (Nokia), Jundong Li (University of Virginia)

Code — https://anonymous.4open.science/r/s3martcirc

###### Abstract

Mechanistic interpretability (MI) of LLMs usually runs in two stages: circuit discovery (find important nodes, e.g., attention heads or feedforward neurons), then functional interpretation. Yet a component's importance and functional role are codependent. Unifying them faces two challenges: (1) roles are tied to specific nodes, limiting generalization; (2) role identification is subjective rather than quantified. S³martCirc abstracts node behavior into two general computational roles and defines a quantitative metric to assign them, so importance and role are discovered jointly. It outperforms existing circuit discovery methods.

## Introduction

The two stages can be reversed: first identify the functional operations a task requires, then find the nodes implementing them. Importance and functionality are interdependent: a node matters because of its function, and its function is defined by its task contribution. Hence discovery and interpretation should be one jointly optimized problem. Obstacles: (1) **node-specific interpretations**: prior roles (e.g., name mover, S-inhibition heads) are bound to a node and task, giving no shareable abstraction for automated pipelines; (2) **subjective functional roles** based on human judgment, hard to formalize.

Key observation: LLMs' core computation, matrix multiplication, mainly enables two operations, computational transformation and information propagation. We abstract fine-grained, task-specific roles into a coarse dichotomy describing *how* a node processes information, not *what*, trading semantic specificity for generality and quantifiability so it can enter an optimization objective.

Contributions:
- **General functional roles**: a dichotomy abstracting task-specific interpretations into two general, quantifiable categories.
- **S³martCirc**: an MI framework coupling discovery and interpretation via bidirectional, alternating optimization.
- **Empirical validation** across multiple LLM architectures: better task-relevant circuits and recovery of known, human-validated circuits.

## Preliminaries and Definitions

LLM with output distribution $P$ and vocabulary $V$ is modeled as a computational graph: nodes are components (attention heads or feedforward neurons), edges are information flow via the residual stream. A circuit is the subgraph responsible for task performance. A node maps $X\in\mathbb{R}^{T\times d_{m}}$ to $Y\in\mathbb{R}^{T\times d_{n}}$ ($T$ tokens, $d_m$ hidden dim, $d_n$ node output dim; architecture-determined for heads, $d_n=1$ for neurons). All compared methods use the same node set and graph.

Circuits for distinct tasks, IOI ("When John and Mary went to the store, John gave a drink to [Mary]") and Acronym Prediction ("The Three Letter Acronym" → "TLA"), share interpretations (e.g., previous token, mover heads), consistent with components being reused across tasks with consistent roles.

Two roles:
- **Functional nodes** (computational transformation): outputs differ substantially from inputs in representation/semantics.
- **Passthrough nodes** (information propagation): relocate input features downstream with minimal modification.

Mapping (Fig. 1): IOI: S-inhibition heads (route subject info) and previous token heads are passthrough; name mover and negative name mover heads (amplify specific names in logits) are functional. Acronym: previous token and bridge heads (propagate letter info) are passthrough; letter mover heads are functional. This suggests systematic principles by which LLMs decompose tasks.

## Self-supervised Smart Circuit Discovery

### S³martCirc Overview

Each node $n$ has learnable masking coefficients $c_{n}=\{c_{n,u},c_{n,p},c_{n,f}\}$: probabilities of being unimportant, passthrough, functional, with $c_{n,u}+c_{n,p}+c_{n,f}=1$. Stage 1, **Important Node Discovery**, separates $c_{n,p}+c_{n,f}$ from $c_{n,u}$; nodes with $c_{n,p}+c_{n,f}>c_{n,u}$ are candidates. Stage 2, **Important Node Classification**, assigns each candidate a role via a task-dependent metric. Training uses soft masking; evaluation uses hard assignments:

$$\hat{c}_{i}=\begin{cases}1,&i=\arg\max_{j}c_{j}\\0,&\text{otherwise}\end{cases}\quad(1)$$

### Important Node Discovery

Removing truly important nodes should degrade performance. With $P_{\text{orig}}$ the full model and $P_{\text{mask}}$ the model with unimportant nodes ($c_{n,u}>c_{n,p}+c_{n,f}$) masked:

$$L_{\text{disc.}}=\alpha\sum_{v\in V}P_{\text{orig}}(v)\log\frac{P_{\text{orig}}(v)}{P_{\text{mask}}(v)}-(1-\alpha)\log P_{\text{mask}}(y)\quad(2)$$

$y$ is the ground-truth token; $\alpha\in[0,1]$ trades faithfulness (KL) against task performance (CE).

### Important Node Classification

Nodes doing nontrivial computation should alter relational structure between tokens; propagating nodes should preserve it. For $A\in\mathbb{R}^{T\times d_{n}}$, $\text{Sim}(A)=\hat{A}\hat{A}^{T}\odot(\mathbf{1}\mathbf{1}^{T}-I)$, with $\hat{A}$ row-wise $L_2$-normalized and the mask excluding self-similarity. Metric:

$$\text{FI}(X,Y)=\frac{\|\text{Sim}(X)-\text{Sim}(Y)\|_{F}}{\|\text{Sim}(X)\|_{F}}\quad(3)$$

With $N_i$ the important nodes:

$$L_{\text{class.}}=\sum_{n_{i}\in N_{i}}\Big[-c_{n_{i},f}\,\text{FI}(X_{n_{i}},Y_{n_{i}})-\frac{c_{n_{i},p}}{\text{FI}(X_{n_{i}},Y_{n_{i}})}\Big]\quad(4)$$

High FI pushes $c_{n,f}$ up; low FI pushes $c_{n,p}$ up.

### S³martCirc Training Process

Regularization on both stages (confidence → binary values, validity → sum to one, sparsity → minimal circuits), with $C$ all masking coefficients:

$$L_{\text{sparsity}}=\frac{1}{|C|}\sum_{c_{n}\in C}\Big(\sum_{v\in c_{n}}\tfrac{1}{2}|v(1-v)|+\tfrac{1}{2}|1-(c_{n,u}+c_{n,p}+c_{n,f})|+(c_{n,p}+c_{n,f})\Big)\quad(5)$$

Training alternates the stages for bidirectional influence. Since role assignment from random initialization is unreliable, a warmup optimizes only $L_{\text{disc.}}$ first.

## Experiments

RQ1 effectiveness; RQ2 recovery of known circuits; RQ3 hyperparameter sensitivity; RQ4 ablations.

### Experiment Setup

**LLMs**: GPT-2 (well-studied MI benchmark), Llama 3.2 1B, Qwen 3 1.7B (architecture generalization). **Tasks**: IOI and Acronym, both with circuits previously uncovered in GPT-2. **Metrics**: nodes not connected to input/output are removed, giving the end-to-end circuit; *Circuit Size* is its size. Evaluation masks only circuit nodes ($P_{\text{circ}}$). *Acc.*: percent accuracy drop original→$P_{\text{circ}}$. *Logit.*: difference in logit of the original model's predicted answer between original and $P_{\text{circ}}$.

**Baselines**: patching-based: Act. (ablate each node with corrupted activation, measure logit-difference change), Attr. (linear approximation: clean–corrupted activation difference × gradient of logit-based metric), Attr. IG (integrated gradients along corrupted→clean path); optimization-based: IBCircuit (information bottleneck), Prune. (gradient-based edge pruning, adapted to nodes); Random ($k$ random nodes, lower bound).

Table 1: Means±std over three runs. "–": no circuit found among selected nodes. *: one run only (48-hour limit).

| Model | Method | Acr. Size | Acr. Acc. | Acr. Logit. | IOI Size | IOI Acc. | IOI Logit. |
|---|---|---|---|---|---|---|---|
| GPT-2 | Act. | 35.00±4.32 | -54.67±38.33 | -3.54±38.33 | 39.33±10.40 | -70.71±25.89 | -9.11±25.89 |
| | Attr. | 37.33±2.05 | -0.17±0.24 | 0.46±0.24 | 48.00±2.16 | -12.73±8.01 | -5.14±8.01 |
| | Attr. IG | 49.67±11.44 | -2.00±1.08 | -0.59±1.08 | 38.00±5.10 | -0.17±0.24 | -0.29±0.24 |
| | IBCircuit | – | – | – | – | – | – |
| | Prune. | 32.33±37.97 | 0.00±0.00 | 0.07±0.00 | 38.33±33.81 | 0.00±0.00 | -0.02±0.00 |
| | Random | 60.33±7.76 | 0.00±0.00 | 0.04±0.00 | 77.33±3.40 | 0.00±0.00 | -0.11±0.00 |
| | S³martCirc | 60.33±7.93 | -97.50±1.08 | -5.00±1.08 | 76.00±5.72 | -95.99±2.85 | -10.42±2.85 |
| Llama | Act. | – | – | – | 13.00±2.00 | -59.17±41.90 | -2.37±41.90 |
| | Attr. | – | – | – | 71.67±2.05 | -1.50±0.41 | -0.35±0.41 |
| | Attr. IG | 54.67±2.62 | -2.17±0.24 | -0.34±0.24 | 47.67±3.30 | -5.67±3.70 | -0.35±3.70 |
| | IBCircuit | 29.67±12.66 | 0.00±0.00 | -0.08±0.00 | 7.33±0.94 | 0.00±0.00 | 0.00±0.00 |
| | Prune. | 70.67±3.09 | 0.00±0.00 | 0.01±0.00 | 70.67±3.09 | 0.00±0.00 | 0.01±0.00 |
| | Random | 60.00±9.93 | 0.00±0.00 | -0.00±0.00 | 62.33±6.18 | 0.00±0.00 | -0.01±0.00 |
| | S³martCirc | 47.33±2.05 | -57.33±3.40 | -3.17±3.40 | 35.00±3.74 | -69.00±42.43 | -2.96±42.43 |
| Qwen | Act. | 32.00±0.00* | -100.00±0.00* | -1.00±0.00* | 12.00±0.00* | -33.00±0.00* | -0.33±0.00* |
| | Attr. | 64.33±3.30 | -0.17±0.24 | -0.48±0.24 | 29.00±3.56 | -0.50±0.41 | -0.91±0.41 |
| | Attr. IG | 32.33±8.34 | -0.17±0.24 | -0.12±0.24 | 35.00±5.35 | -0.33±0.47 | -0.19±0.47 |
| | IBCircuit | 14.67±0.94 | 0.00±0.00 | 0.29±0.00 | – | – | – |
| | Prune. | 46.00±4.24 | 0.00±0.00 | -0.01±0.00 | 46.00±4.24 | 0.00±0.00 | -0.01±0.00 |
| | Random | 52.67±9.74 | 0.00±0.00 | -0.00±0.00 | 48.00±6.98 | 0.00±0.00 | 0.01±0.00 |
| | S³martCirc | 12.00±1.63 | -100.00±0.00 | -10.97±0.00 | 16.33±4.50 | -100.00±0.00 | -12.01±0.00 |

### RQ1: Performance of S³martCirc

S³martCirc beats all baselines on Acc. and Logit. for every task/model. Patching methods evaluate nodes independently and may miss interactions; Prune. and IBCircuit also use continuous masks yet do much worse, so functional interpretation adds guidance beyond masking. S³martCirc is not always smallest but yields far larger degradation at comparable size, e.g., on IOI with Llama, -69.00% with 35 nodes vs. Attr. IG's -5.67% with 47.67 nodes.

### RQ2: Recovering Prior Circuits

Fig. 3: percent of the manually verified IOI circuit recovered within end-to-end circuits thresholded at 50/100/150/200 nodes (100% unattainable given the node count). S³martCirc recovers more than most baselines, comparable only to Act.; optimization-based methods recover much less even at 200 nodes. Thus it aligns with human-verified interpretations, adding credibility where ground truth is unavailable.

Acronym (Fig. 4; nodes as layer.head, e.g., 8.11 = layer 8, head 11): S³martCirc recovers 4 of 8 nodes (50%); classifications match Fig. 1: Previous Token 1.0 passthrough, Letter Mover 11.4 functional. It labels 8.11 passthrough and 10.10 functional, though prior work shows both can act as Bridge or Letter Mover depending on intervention. Human analysis uses isolated interventions to find what a head *can* do; S³martCirc measures the dominant contribution across samples, so this reflects multi-functionality, not misclassification.

### RQ3: Parameter Analysis

IOI, GPT-2 (Fig. 5); each iteration = one discovery step + $k$ classification steps; $k\in\{2,4,6,8,10\}$, $\alpha\in\{0.0,0.25,0.5,0.75,1.0\}$. Performance improves with $k$, stabilizing at $k=6$: Acc. drop from -71% ($k=2$) to ≈-96% ($k=6$); recovery increases monotonically. $\alpha=0.75$ is best for both metrics: KL should dominate for IOI, but removing CE ($\alpha=1.0$) degrades sharply, explaining weaker baselines relying on output-distribution differences; some critical components drive correct behavior without large logit changes.

### RQ4: Ablation Study

Variants (Fig. 6): -NW (no warmup), -DF (full discovery, then classification), -IF (classification, then discovery).

**Importance of Alternating Training.** -DF is worst, Acronym Acc. drop -4.33% vs -97.5%: discovering all nodes first yields an overly broad, noisy candidate set, so circuits include non-critical and omit key nodes. -IF is much better than -DF (≈88% IOI, 42% Acronym) but below S³martCirc: functional priors help but need alternation. Each stage refines the other: discovery uses evolving roles; classification works on increasingly focused candidates.

**Role of Warmup Period.** -NW drops consistently (67% IOI, 66% Acronym): random coefficients give unreliable early gradients, causing unstable optimization and suboptimal circuits.

## Related Works

**Mechanistic Interpretability and Circuit Discovery.** Most work is where-then-what: localize via patching or gradient approximations, then interpret manually; interventions are costly and isolate components, interpretation is subjective. Components are reused with consistent roles, roles localize to individual neurons, and structurally distinct circuits can be functionally equivalent, motivating an abstract, quantitative role.

**Optimization-Based Circuit Discovery.** Continuous masks under sparsity (e.g., IBCircuit, behavior-specific mask learning) capture interactions better than patching but model only which components matter. S³martCirc differs: (1) roles from a task-agnostic quantitative activation metric, not hand-specified desiderata or ablation signs; (2) importance and role optimized jointly by alternation; (3) spans attention heads and MLP neurons.

## Conclusion

S³martCirc jointly discovers circuits and interprets node roles, outperforming baselines on GPT-2, Llama 3.2, and Qwen 3. Limitations: the two roles are intentionally coarse and do not recover fine-grained manual roles; roles are defined relative to a task's activations, not intrinsically, so a node may be labeled differently across tasks.

## Appendix A Implementation Details

### Computation Resources

NVIDIA RTX A4000 16GB and A16 16GB GPUs; Attr. and Attr. IG on Qwen used RTX A6000 48GB.

### Data Generation

Data generated per tokenizer to respect token limits.

**IOI.** Names and place/object pairs from Wang et al., kept if single-token; all valid combinations fill "When [NAME_1] and [NAME_2] went to the [PLACE], [NAME_1] handed the [OBJECT] to"; corrupted = second name swapped. Balanced 1000 samples: $1000/p$ per pair ($p$ valid pairs).

**Acronym.** wonderwords RandomWord: 500 words per letter, kept if " Word" tokenizes to two tokens, the first the space-prefixed capital; letters without valid words dropped. Valid abbreviations: three letters tokenizing to three tokens. For each pair, matching random words fill "The [A_RAND_WORD] [B_RAND_WORD] [C_RAND_WORD] (AB" (clean/corrupted); shuffle, take first 1000.

### Hyperparameters

Seed 42. (lr, weight decay, epochs, warmup, sparsity coef., α):
- GPT-2 Acronym: 0.01, 0.005, 60, 60, 0.5, 0.25
- GPT-2 IOI: 0.005, 0.0005, 60, 5, 0.3, 0.75
- Llama Acronym: 0.001, 0.005, 50, 5, 1.0, 0.75
- Llama IOI: 0.001, 0.005, 75, 5, 1.0, 0.25
- Qwen Acronym: 0.001, 0.005, 50, 5, 0.7, 0.75
- Qwen IOI: 0.001, 0.005, 50, 10, 0.9, 0.75

### Run time

Table 2: Runtime (s)

| | | Act. | Attr. | Attr. IG | IBC. | Prune. | Rand. | Ours |
|---|---|---|---|---|---|---|---|---|
| GPT-2 | Acr. | 6239.72 | 299.54 | 3452.48 | 426.70 | 3391.87 | 3.98 | 197.14 |
| | IOI | 7943.71 | 267.94 | 2801.64 | 467.19 | 3732.88 | 0.13 | 264.44 |
| Llama | Acr. | 61510.09 | 1114.46 | 12770.96 | 1266.76 | 11470.94 | 2.22 | 507.67 |
| | IOI | 64639.54 | 1128.62 | 12344.18 | 1617.78 | 11645.51 | 3.37 | 901.02 |
| Qwen | Acr. | 168689.44 | 829.43 | 11151.09 | 3845.65 | 19072.51 | 3.54 | 982.16 |
| | IOI | 187663.88 | 783.17 | 11148.12 | 5152.79 | 21000.88 | 2.36 | 1047.59 |

## Appendix B Additional Experiment Results

Repeating Section 4.3's first experiment on Acronym (Fig. 7): S³martCirc keeps the highest percent recovered.