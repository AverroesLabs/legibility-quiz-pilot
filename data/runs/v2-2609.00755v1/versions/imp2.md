# S³martCirc: Self-supervised Smart Circuit Discovery

Wendy Zheng

  
Yinhan He

  
Liang Wu
Jundong Li\corresponding

###### Abstract

Large Language Models (LLMs) perform strongly on tasks from text summarization to question answering, yet their internal decision-making remains opaque. Mechanistic interpretability (MI) for LLMs proceeds in two stages: circuit discovery over individual nodes (attention heads, feedforward neurons), followed by functional interpretation of each node's role in a task. Two obstacles stand in the way of merging these stages: (1) functional roles are tied to specific nodes or components, which limits generalization, and (2) roles are assigned by subjective interpretation rather than quantifiable metrics. We propose S³martCirc (Self-supervised Smart Circuit Discovery), a unified framework that discovers circuits and interprets functionality simultaneously. S³martCirc abstracts node behavior into two computational roles shared across tasks, assigns them with a quantitative metric, and learns importance and role jointly rather than in sequence. S³martCirc outperforms existing circuit discovery methods.

1University of Virginia

2Nokia

ncd9cf@virginia.edu,nee7ne@virginia.edu, liang.wu@nokia.com, jundong@virginia.edu

Code — https://anonymous.4open.science/r/s3martcirc

## Introduction

LLMs perform strongly on summarization (Basyal and Sanghvi 2023; Van Veen et al. 2023), translation (Feng et al. 2024; Kleidermacher and Zou 2025) and question answering (Li et al. 2024; Tan et al. 2023; Yang et al. 2023a). Their opacity raises safety concerns (Singh et al. 2024; Wagner et al. 2024) and limits adoption in healthcare and finance (Tatsat and Shater 2025; Yang et al. 2023b). Among interpretation techniques (Chen et al. 2021; He et al. 2025; Hoover et al. 2020; Singh et al. 2024), MI (Bereska and Gavves 2024; Sharkey et al. 2025) reverse-engineers networks into human-understandable algorithms. For LLMs it proceeds in two stages: circuit discovery over attention heads or feedforward neurons, then functional interpretation of the selected nodes (Liu et al. 2025; Nanda et al. 2023; Wang et al. 2022). The two stages can also be run in the opposite order, with the required operations identified first and the implementing nodes located afterwards (Méloux et al. 2025b).

Two obstacles prevent the stages from being merged. (1) Node-specific interpretations: prior roles (e.g., name mover, S-inhibition heads) are fine-grained, bound to a particular node and task, and not shareable across tasks or pipelines. (2) Subjective functional roles: assignment relies on human judgment and resists formalization.

S³martCirc performs circuit discovery and functional interpretation jointly. The core computation of an LLM, matrix multiplication, supports two operations: computational transformation and information propagation. We map task-specific interpretations onto a coarse dichotomy over these two operations. The dichotomy gives up semantic specificity in exchange for generality and quantifiability, and enters the optimization objective directly. Contributions:

•

General functional roles: a dichotomy of computational roles that abstracts prior task-specific interpretations into two general, quantifiable categories.

•

S³martCirc Framework: an MI framework coupling discovery and interpretation through bidirectional, alternating optimization.

•

Empirical validation: experiments across multiple LLM architectures show superior identification of task-relevant circuits and recovery of known, human-validated circuits.

## Preliminaries and Definitions

Let $P$ be the output distribution of an LLM with vocabulary $V$. The LLM is treated as a computational graph whose nodes are attention heads or feedforward neurons and whose edges carry residual-stream flow between them (Figure 2); the circuit for a task is the subgraph responsible for task performance. A node maps $X\in\mathbb{R}^{T\times d_{m}}$ to $Y\in\mathbb{R}^{T\times d_{n}}$, with $T$ tokens, hidden dimension $d_{m}$, and $d_{n}$ fixed by the architecture for heads and $d_{n}=1$ for feedforward neurons. All compared discovery methods share this node set and graph abstraction.

Circuits for different tasks share roles, consistent with component reuse across tasks (Merullo et al. 2024). Figure 1 compares Indirect Object Identification (IOI) (Wang et al. 2022), e.g., "When John and Mary went to the store, John gave a drink to [Mary]", with Acronym Prediction (García-Carrasco et al. 2024a), in which the model predicts the last letter of a three-letter acronym, e.g., "The Three Letter Acronym" $\rightarrow$ "TLA". Both circuits contain previous token and mover heads.

Figure 1: Mapping between node-specific functional roles in discovered circuits and the defined general functional roles.

The two categories are computational transformation (functional nodes, whose outputs differ substantially from their inputs in representation or semantics) and information propagation (passthrough nodes, which carry input features downstream with minimal modification). In IOI, the S-inhibition heads, which route subject information downstream, and the previous token heads, which write previous-token information into the residual stream, are passthrough. The name mover and negative name mover heads, which selectively amplify specific names in the output logits, are functional. In Acronym, the previous token and bridge heads, which propagate letter information, are passthrough, and the letter mover heads, which emphasize the correct output, are functional.

## Self-supervised Smart Circuit Discovery

Figure 2: An overview of the proposed S³martCirc framework.

### S³martCirc Overview

Every node $n$ carries learnable masking coefficients $c_{n}=\{c_{n,u},c_{n,p},c_{n,f}\}$, the probabilities of being unimportant, passthrough or functional, with $c_{n,u}+c_{n,p}+c_{n,f}=1$. The coefficients are optimized in two stages (Figure 2). Important Node Discovery contrasts $c_{n,p}+c_{n,f}$ against $c_{n,u}$; nodes with $c_{n,p}+c_{n,f}>c_{n,u}$ become candidates. Important Node Classification then assigns each candidate a role using a task-dependent metric. Masking is soft in $c_{n}$ during training and hard at evaluation:

| | $\hat{c}_{i}=\begin{cases}1,&i=\arg\max_{j}c_{j}\\
0,&\text{otherwise}\end{cases}$
| | (1)

### Important Node Discovery

Importance is optimized through two objectives. The first is KL divergence from the full model, for faithfulness. The second is cross-entropy on the ground-truth answer, for correctness. Here $P_{\text{orig}}$ is the full-model distribution and $P_{\text{mask}}$ is the distribution with nodes satisfying $c_{n,u}>c_{n,p}+c_{n,f}$ masked:

| | $L_{\text{disc.}}=\alpha\cdot\sum_{v\in V}P_{\text{orig}}(v)\log\frac{P_{\text{orig}}(v)}{P_{\text{mask}}(v)}-(1-\alpha)\cdot(\log P_{\text{mask}}(y)).$
| | (2)

Here $v\in V$ ranges over tokens, $y\in V$ is the ground truth, and $\alpha\in[0,1]$ sets the faithfulness/task-performance trade-off.

### Important Node Classification

Prior work assigns roles qualitatively or heuristically (Wang et al. 2022; Chandna et al. 2025). S³martCirc assigns them by the change in pairwise token similarity across a node. For $A\in\mathbb{R}^{T\times d_{n}}$ (the node's input or output), let $\text{Sim}(A)=\hat{A}\hat{A}^{T}\odot(\mathbf{1}\mathbf{1}^{T}-I),$ where $\hat{A}$ is the row-wise $L_{2}$-normalized $A$, $\mathbf{1}$ is the all-ones column vector and $I$ is the identity. Then

| | $\text{FI}(X,Y)=\frac{\|\text{Sim}(X)-\text{Sim}(Y)\|_{F}}{\|\text{Sim}(X)\|_{F}},$
| | (3)

with $X\in\mathbb{R}^{T\times d_{m}}$ and $Y\in\mathbb{R}^{T\times d_{n}}$ the node's input and output. With $N_{i}$ the candidates ($c_{n,p}+c_{n,f}>c_{n,u}$):

| | $\displaystyle L_{\text{class.}}=\sum_{n_{i}\in N_{i}}\Big[-c_{n_{i},f}\cdot\text{FI}(X_{n_{i}},Y_{n_{i}})-\frac{c_{n_{i},p}}{\text{FI}(X_{n_{i}},Y_{n_{i}})}\Big].$
| | (4)

### S³martCirc Training Process

Both stages add a regularizer with confidence, validity and sparsity terms:

| | $\displaystyle L_{\text{sparsity}}$ | $\displaystyle=\frac{1}{|C|}\sum_{c_{n}\in C}\Big(\underbrace{\sum_{v\in c_{n}}\Big[\frac{1}{2}|v(1-v)|\Big]}_{\text{confidence}}$
| | (5)

|
| | $\displaystyle+\underbrace{\frac{1}{2}|1-(c_{n,u}+c_{n,p}+c_{n,f})|}_{\text{validity}}$
|

|
| | $\displaystyle+\underbrace{(c_{n,p}+c_{n,f})}_{\text{sparsity}}\Big),$
|

where $C$ is the set of all masking coefficients. A warmup phase optimizes $L_{\text{disc.}}$ alone, because role assignment from randomly initialized coefficients is unreliable. After warmup, training alternates between the two stages (Figure 2).

## Experiments

Table 1: Performance of S³martCirc compared to baselines over three runs. Best results are shown in bold, and runner-up results are italicized. Empty cells indicate that no circuit could be found within the set of nodes selected by the method. An asterisk (*) indicates that only one experimental run was completed due to the time limit (48 hours).

|
| | Acronym | IOI

|
| | Circuit Size | Acc. | Logit. | Circuit Size | Acc. | Logit.

| GPT-2 | Act. | $\mathit{35.00_{\pm 4.32}}$ | - $\mathit{54.67_{\pm 38.33}}$ | - $\mathit{3.54_{\pm 38.33}}$ | $39.33_{\pm 10.40}$ | - $\mathit{70.71_{\pm 25.89}}$ | - $\mathit{9.11_{\pm 25.89}}$

| Attr. | $37.33_{\pm 2.05}$ | - $0.17_{\pm 0.24}$ | $0.46_{\pm 0.24}$ | $48.00_{\pm 2.16}$ | - $12.73_{\pm 8.01}$ | - $5.14_{\pm 8.01}$

| Attr. IG | $49.67_{\pm 11.44}$ | - $2.00_{\pm 1.08}$ | - $0.59_{\pm 1.08}$ | $\mathbf{38.00_{\pm 5.10}}$ | - $0.17_{\pm 0.24}$ | - $0.29_{\pm 0.24}$

| IBCircuit | – | – | – | – | – | –

| Prune. | $\mathbf{32.33_{\pm 37.97}}$ | $0.00_{\pm 0.00}$ | $0.07_{\pm 0.00}$ | $\mathit{38.33_{\pm 33.81}}$ | $0.00_{\pm 0.00}$ | - $0.02_{\pm 0.00}$

| Random | $60.33_{\pm 7.76}$ | $0.00_{\pm 0.00}$ | $0.04_{\pm 0.00}$ | $77.33_{\pm 3.40}$ | $0.00_{\pm 0.00}$ | - $0.11_{\pm 0.00}$

| S³martCirc | $60.33_{\pm 7.93}$ | $\mathbf{-97.50_{\pm 1.08}}$ | $\mathbf{-5.00_{\pm 1.08}}$ | $76.00_{\pm 5.72}$ | $\mathbf{-95.99_{\pm 2.85}}$ | $\mathbf{-10.42_{\pm 2.85}}$

| Llama | Act. | – | – | – | $\mathit{13.00_{\pm 2.00}}$ | - $\mathit{59.17_{\pm 41.90}}$ | - $\mathit{2.37_{\pm 41.90}}$

| Attr. | – | – | – | $71.67_{\pm 2.05}$ | - $1.50_{\pm 0.41}$ | - $0.35_{\pm 0.41}$

| Attr. IG | $54.67_{\pm 2.62}$ | - $\mathit{2.17_{\pm 0.24}}$ | - $\mathit{0.34_{\pm 0.24}}$ | $47.67_{\pm 3.30}$ | - $5.67_{\pm 3.70}$ | - $0.35_{\pm 3.70}$

| IBCircuit | $\mathbf{29.67_{\pm 12.66}}$ | $0.00_{\pm 0.00}$ | - $0.08_{\pm 0.00}$ | $\mathbf{7.33_{\pm 0.94}}$ | $0.00_{\pm 0.00}$ | $0.00_{\pm 0.00}$

| Prune. | $70.67_{\pm 3.09}$ | $0.00_{\pm 0.00}$ | $0.01_{\pm 0.00}$ | $70.67_{\pm 3.09}$ | $0.00_{\pm 0.00}$ | $0.01_{\pm 0.00}$

| Random | $60.00_{\pm 9.93}$ | $0.00_{\pm 0.00}$ | - $0.00_{\pm 0.00}$ | $62.33_{\pm 6.18}$ | $0.00_{\pm 0.00}$ | - $0.01_{\pm 0.00}$

| S³martCirc | $\mathit{47.33_{\pm 2.05}}$ | $\mathbf{-57.33_{\pm 3.40}}$ | $\mathbf{-3.17_{\pm 3.40}}$ | $35.00_{\pm 3.74}$ | $\mathbf{-69.00_{\pm 42.43}}$ | $\mathbf{-2.96_{\pm 42.43}}$

| Qwen | Act. | $32.00_{\pm 0.00}$ * | $\mathbf{-100.00_{\pm 0.00}}$ * | - $\mathit{1.00_{\pm 0.00}}$ * | $\mathbf{12.00_{\pm 0.00}}$ * | - $\mathit{33.00_{\pm 0.00}}$ * | - $0.33_{\pm 0.00}$ *

| Attr. | $64.33_{\pm 3.30}$ | - $\mathit{0.17_{\pm 0.24}}$ | - $0.48_{\pm 0.24}$ | $29.00_{\pm 3.56}$ | - $0.50_{\pm 0.41}$ | - $\mathit{0.91_{\pm 0.41}}$

| Attr. IG | $32.33_{\pm 8.34}$ | - $\mathit{0.17_{\pm 0.24}}$ | - $0.12_{\pm 0.24}$ | $35.00_{\pm 5.35}$ | - $0.33_{\pm 0.47}$ | - $0.19_{\pm 0.47}$

| IBCircuit | $\mathit{14.67_{\pm 0.94}}$ | $0.00_{\pm 0.00}$ | $0.29_{\pm 0.00}$ | – | – | –

| Prune. | $46.00_{\pm 4.24}$ | $0.00_{\pm 0.00}$ | - $0.01_{\pm 0.00}$ | $46.00_{\pm 4.24}$ | $0.00_{\pm 0.00}$ | - $0.01_{\pm 0.00}$

| Random | $52.67_{\pm 9.74}$ | $0.00_{\pm 0.00}$ | - $0.00_{\pm 0.00}$ | $48.00_{\pm 6.98}$ | $0.00_{\pm 0.00}$ | $0.01_{\pm 0.00}$

| S³martCirc | $\mathbf{12.00_{\pm 1.63}}$ | $\mathbf{-100.00_{\pm 0.00}}$ | $\mathbf{-10.97_{\pm 0.00}}$ | $\mathit{16.33_{\pm 4.50}}$ | $\mathbf{-100.00_{\pm 0.00}}$ | $\mathbf{-12.01_{\pm 0.00}}$

Research questions: RQ1, effectiveness relative to existing methods; RQ2, recovery of known circuits; RQ3, hyperparameter sensitivity; RQ4, component contributions.

### Experiment Setup

LLMs. GPT-2 (Radford et al. 2019; Maintainers 2022), a standard MI benchmark; Llama 3.2 1B (Grattafiori et al. 2024; Meta 2024); and Qwen 3 1.7B (Team 2025a; Team 2025b), to test generalization across architectures.

Evaluation Tasks. IOI (Wang et al. 2022) and Acronym Prediction (Acronym) (García-Carrasco et al. 2024a). Both have circuits previously established in GPT-2.

Evaluation Metrics. Each method's selected nodes are reduced to an end-to-end circuit by removing nodes disconnected from the input or output; the size of this circuit is the Circuit Size. Evaluation masks only the circuit nodes ($P_{\text{circ}}$). Acc. is the percent accuracy drop from the original model to $P_{\text{circ}}$, and Logit. is the corresponding change in the logit of the original model's predicted answer.

Baselines. Patching-based: Activation Patching (Act.) (Heimersheim and Nanda 2024), which replaces each node's activation with a corrupted one and scores the change in logit difference between the original and an alternative answer; Attribution Patching (Attr.) (Syed et al. 2023), which scores each node by the product of its clean-minus-corrupted activation with the gradient of a logit-difference metric; and Attribution Patching with Integrated Gradients (Attr. IG) (Hanna et al. 2024), which integrates those gradients along the corrupted-to-clean path. Optimization-based: IBCircuit (Bian et al. 2026), which uses an information bottleneck objective, and Pruning (Prune.) (Bhaskar et al. 2024), which applies gradient-based edge pruning, here adapted to the node level. Random selects $k$ nodes.

### RQ1: Performance of S³martCirc

Effectiveness. S³martCirc gives the largest accuracy drop and logit difference on every model/task (Table 1). On GPT-2 Acronym it reaches -97.50%, against -54.67% for the strongest baseline, Activation Patching. Patching-based methods score nodes independently. Prune. and IBCircuit learn continuous masks but have no role term, and both fall well below S³martCirc.

Circuit Size. S³martCirc does not always produce the smallest circuit. At comparable sizes, however, its degradation is far larger. On Llama IOI it reaches -69.00% with 35 nodes, while Attr. IG reaches -5.67% with 47.67 nodes on average.

### RQ2: Recovering Prior Circuits

Figure 3 reports the percentage of the manually verified IOI circuit (Wang et al. 2022) that appears in each method's end-to-end circuit at thresholds of 50, 100, 150, and 200 nodes. Given the size of the node space, 100% is not attainable. S³martCirc recovers more than most baselines and is matched only by Activation Patching. The optimization-based baselines recover substantially less, even at 200 nodes.

For the Acronym circuit (García-Carrasco et al. 2024a), Figure 4 labels nodes as layer.head (e.g., 8.11 is layer 8, attention head 11). S³martCirc recovers four of the eight nodes (50%), and its roles agree with Figure 1. Previous Token head 1.0 is passthrough and Letter Mover head 11.4 is functional. Head 8.11 is classified as passthrough and 10.10 as functional. In prior work, both heads act as Bridge or Letter Mover heads depending on the intervention; there, behavior is assessed under isolated interventions, whereas S³martCirc's labels reflect each head's dominant contribution across samples. These two heads are therefore multi-functional rather than misclassified.

Figure 3: Percent of the IOI circuit recovered by different circuit discovery methods

### RQ3: Parameter Analysis

Figure 5 shows results on GPT-2 IOI for $k\in\{2,4,6,8,10\}$, the number of classification steps per discovery step, and $\alpha\in\{0.0,0.25,0.5,0.75,1.0\}$. Both Acc. and IOI recovery are reported. Accuracy drop improves from -71% at $k=2$ to approximately -96% at $k=6$ and stabilizes there (Figure 5a). Circuit recovery increases monotonically in $k$. $\alpha=0.75$ is best on both measures (Figure 5b), and $\alpha=1.0$ (no cross-entropy term) degrades sharply. Several IOI-critical components contribute to correct behavior without large logit changes.

Figure 4: Comparison between the Acronym circuit and the functional interpretations determined by S³martCirc.

### RQ4: Ablation Study

Figure 6 compares three variants against the full method: S³martCirc-NW, which has no warmup; S³martCirc-DF, which runs Important Node Discovery to completion before Important Node Classification; and S³martCirc-IF, which runs Classification before Discovery.

Importance of Alternating Training. S³martCirc-DF is the weakest variant: -4.33% on Acronym, against -97.5% for S³martCirc. Its candidate set is overly broad, and the resulting circuits include non-critical nodes and omit key ones. S³martCirc-IF improves substantially on S³martCirc-DF (approximately 88% on IOI and 42% on Acronym) but remains below the full method.

Figure 5: Effect of $\alpha$ and $k$ on S³martCirc Performance.

Under alternation, discovery receives updated role information and classification operates on progressively narrower candidate sets.

Role of Warmup Period. Removing warmup degrades performance on both tasks (67% on IOI and 66% on Acronym). Without warmup, early gradients are uninformative, optimization is less stable, and the method converges to suboptimal circuits.

Figure 6: Ablation study to evaluate the effectiveness of different components in S³martCirc and its training process.

## Related Works

Mechanistic Interpretability and Circuit Discovery.
MI identifies the components behind model behavior (Bereska and Gavves 2024; Sharkey et al. 2025; Rai et al. 2024) and is organized around circuits. Most work follows a where-then-what paradigm (Méloux et al. 2025a; García-Carrasco et al. 2024b; Chandna et al. 2025). Components are localized by patching (Wang et al. 2022; Meng et al. 2022; García-Carrasco et al. 2024a; Heimersheim and Nanda 2024) or by gradient approximations (Syed et al. 2023; Hanna et al. 2024), and then interpreted manually. These interventions are costly and treat components in isolation, and the manual interpretation is subjective and hard to scale. Components are reused across tasks with consistent roles (Merullo et al. 2024), roles localize to individual neurons (Nikankin et al. 2026), and structurally distinct circuits can be functionally equivalent (Haklay et al. 2026).

Optimization-Based Circuit Discovery.
Continuous masks under sparsity constraints (Frankle and Carbin 2019; Bhaskar et al. 2024) include IBCircuit's information bottleneck objective balanced against behavioral fidelity (Bian et al. 2026) and behavior-specific mask learning (Li and Janson 2024). These methods capture component interactions better than patching, but most model only which components matter, not how they contribute. S³martCirc differs in three respects: (1) roles come from a task-agnostic, quantitative metric on activations rather than from hand-specified desiderata or ablation signs; (2) importance and role are optimized by alternating between stages rather than read off a single mask; (3) discovery covers both attention heads and MLP neurons.

## Conclusion

S³martCirc jointly discovers circuits and interprets node functionality in LLMs. It uses a task-general dichotomy of abstract roles and a quantitative metric that place interpretation inside an end-to-end optimizable discovery procedure. On GPT-2, Llama 3.2, and Qwen 3, it outperforms existing methods in identifying compact, task-relevant circuits and in recovering human-verified circuits. It has two limitations. The two roles are coarse and do not recover the fine-grained roles of manual analysis. Roles are also defined relative to a task's activations rather than intrinsically, so the same node may receive different labels on different tasks.

## Appendix A Implementation Details

### Computation Resources

Experiments ran on NVIDIA RTX A4000 16GB and NVIDIA A16 16GB GPU clusters. The exceptions were Attribution Patching and Attribution Patching IG on Qwen, which ran on NVIDIA RTX A6000 48GB GPU clusters because of model size.

### Data Generation

Data are generated per tokenizer to respect token limits.

IOI. Names and place/object pairs come from (Wang et al. 2022). Only single-token values are kept. All valid combinations are inserted into “When [NAME_1] and [NAME_2] went to the [PLACE], [NAME_1] handed the [OBJECT] to”. Corrupted counterparts swap the second name. For a balanced set of 1000 samples, $1000/p$ samples are drawn per place/object pair, where $p$ is the number of valid pairs.

Acronym. Words come from the RandomWord class of the wonderwords package, with 500 sampled per letter. A word is kept only if its space-prefixed, capitalized form tokenizes into exactly two tokens, the first being the space-prefixed capital letter. Letters with no valid words are dropped. Valid abbreviations are three-letter sequences from the remaining letters whose concatenation tokenizes into three tokens. For each pair of valid abbreviations, matching random words fill “The [A_RAND_WORD] [B_RAND_WORD] [C_RAND_WORD] (AB” to form clean and corrupted samples. The data are shuffled and the first 1000 samples are kept.

### Hyperparameters

Hyperparameters for Table 1 are listed below. All runs used random seed 42.

•

Acronym with GPT-2 - Learning rate is 0.01, weight decay is 0.005, number of epochs is 60, number of warmup epochs is 60, sparsity coefficient is 0.5, and the logit coefficient (alpha) is 0.25.

•

IOI with GPT-2 - Learning rate is 0.005, weight decay is 0.0005, number of epochs is 60, number of warmup epochs is 5, sparsity coefficient is 0.3, and the logit coefficient is 0.75.

•

Acronym with Llama - Learning rate is 0.001, weight decay is 0.005, number of epochs is 50, number of warmup epochs is 5, sparsity coefficient is 1.0, and the logit coefficient is 0.75.

•

IOI with Llama - Learning rate is 0.001, weight decay is 0.005, number of epochs is 75, number of warmup epochs is 5, sparsity coefficient is 1.0, and the logit coefficient is 0.25.

•

Acronym with Qwen - Learning rate is 0.001, weight decay is 0.005, number of epochs is 50, number of warmup epochs is 5, sparsity coefficient is 0.7, and the logit coefficient is 0.75.

•

IOI with Qwen - Learning rate is 0.001, weight decay is 0.005, number of epochs is 50, number of warmup epochs is 10, sparsity coefficient is 0.9, and the logit coefficient is 0.75.

### Run time

S³martCirc runs efficiently relative to the baselines (Table 2).

Table 2: Runtime comparison between baselines and S³martCirc (in seconds)

|
| | Act. | Attr. | Attr. IG | IBCircuit | Prune. | Random | S³martCirc

| GPT-2 | Acronym | 6239.72 | 299.54 | 3452.48 | 426.70 | 3391.87 | 3.98 | 197.14

| IOI | 7943.71 | 267.94 | 2801.64 | 467.19 | 3732.88 | 0.13 | 264.44

| Llama | Acronym | 61510.09 | 1114.46 | 12770.96 | 1266.76 | 11470.94 | 2.22 | 507.67

| IOI | 64639.54 | 1128.62 | 12344.18 | 1617.78 | 11645.51 | 3.37 | 901.02

| Qwen | Acronym | 168689.44 | 829.43 | 11151.09 | 3845.65 | 19072.51 | 3.54 | 982.16

| IOI | 187663.88 | 783.17 | 11148.12 | 5152.79 | 21000.88 | 2.36 | 1047.59

## Appendix B Additional Experiment Results

The first experiment of Section 4.3 was repeated on Acronym (Figure 7). S³martCirc again recovers the largest percentage of the circuit.

Figure 7: Percentage of the Acronym circuit recovered by different circuit discovery methods