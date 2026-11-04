# S³martCirc: Self-supervised Smart Circuit Discovery

Wendy Zheng

  
Yinhan He

  
Liang Wu
Jundong Li\corresponding

###### Abstract

Large Language Models (LLMs) have demonstrated remarkable performance across diverse tasks, from text summarization to question answering. Despite these capabilities, their black-box nature obscures internal decision-making processes. Mechanistic interpretability (MI) aims to address this by reverse-engineering neural networks into human-understandable algorithms. Current MI approaches for LLMs typically follow a two-stage paradigm: first identifying important components (circuit discovery), where components are typically individual nodes such as an attention head or feedforward neuron, and second determining the role they play in a certain task (functional interpretation). The first stage is commonly carried out with patching-based interventions that score each node in isolation, which is computationally expensive and can miss interactions between components, while the second stage relies heavily on manual analysis. In this work, we focus on making the discovery stage more efficient and better grounded in the model’s actual computation. We propose S³martCirc (Self-supervised Smart Circuit Discovery), a framework that learns soft masks over attention heads and feedforward neurons using self-supervised signals derived from the model itself. S³martCirc combines a faithfulness objective, a correctness objective, and a coefficient regularizer that promotes confident, valid, and sparse circuits, and it evaluates discovered circuits end to end on a common graph abstraction. We describe the framework in detail, establish a unified evaluation protocol spanning three LLM families and two established tasks, and characterize the behavior and cost of existing discovery methods under this protocol.

1University of Virginia

2Nokia

ncd9cf@virginia.edu,nee7ne@virginia.edu, liang.wu@nokia.com, jundong@virginia.edu

Code — https://anonymous.4open.science/r/s3martcirc

## Introduction

Recently, Large Language Models (LLMs) have exhibited extraordinary performance across a wide range of diverse tasks, from text summarization (Basyal and Sanghvi 2023; Van Veen et al. 2023) and translation (Feng et al. 2024; Kleidermacher and Zou 2025) to question answering (Li et al. 2024; Tan et al. 2023; Yang et al. 2023a). However, these models are primarily black boxes whose internal reasoning mechanisms cannot be directly accessed or interpreted by humans. This opacity obscures their decision-making processes and raises safety concerns about potential undesired behaviors (Singh et al. 2024; Wagner et al. 2024). Consequently, it hinders the adoption of LLMs in critical fields such as healthcare and finance, where reliability and trustworthiness are paramount (Tatsat and Shater 2025; Yang et al. 2023b).

To better understand how LLMs operate, various interpretation techniques have been proposed (Chen et al. 2021; He et al. 2025; Hoover et al. 2020; Singh et al. 2024), among which mechanistic interpretability (MI) has emerged as a particularly promising direction (Bereska and Gavves 2024; Sharkey et al. 2025). MI aims to reverse engineer neural networks into human-understandable algorithms by uncovering how components collaborate to implement a specific task. Current approaches for LLMs generally follow a two-stage paradigm: (1) identify important nodes (i.e., attention heads or feedforward neurons) through circuit discovery, and (2) determine the functional role each selected node performs through functional interpretation (Liu et al. 2025; Nanda et al. 2023; Wang et al. 2022).

Both stages face practical obstacles, and the discovery stage is the foundation on which everything else rests: an interpretation can only be as reliable as the set of nodes it is built upon. Discovery is most often performed with patching-based interventions, which replace a node’s activation with a corrupted counterpart and measure the effect on a logit-based metric (Heimersheim and Nanda 2024). While conceptually clear, this procedure scales with the number of nodes and must be repeated for every candidate, which becomes prohibitive for modern models with thousands of heads and many thousands of neurons. Gradient-based approximations (Syed et al. 2023; Hanna et al. 2024) reduce this cost but inherit the limitation of scoring each node independently, so that interactions between components, such as redundancy or compensation, are not reflected in the scores. Optimization-based approaches that learn continuous masks (Bhaskar et al. 2024; Bian et al. 2026) offer a way to account for such interactions, but they differ considerably in their objectives, their graph abstractions, and the granularity at which they operate, which makes them difficult to compare directly.

A second difficulty concerns evaluation. Circuits are reported at different granularities (edges, heads, neurons), with different notions of what it means for a circuit to be complete, and with different metrics of faithfulness. Consequently, it is often unclear whether differences between methods reflect genuine differences in the quality of the selected nodes or merely differences in the evaluation protocol. A shared node set, a shared definition of an end-to-end connected circuit, and shared metrics are therefore prerequisites for drawing meaningful conclusions about discovery methods.

To address these issues, we propose Self-supervised Smart Circuit Discovery (S³martCirc), a framework for learning circuits through soft masks over all attention heads and feedforward neurons. S³martCirc is self-supervised in the sense that its training signal is derived from the model itself: a faithfulness term encourages the masked model to reproduce the full model’s output distribution, and a correctness term encourages it to retain the ability to produce the right answer. These are combined with a coefficient regularizer that pushes masks toward confident, valid, and sparse assignments, and with a warmup schedule that stabilizes early optimization. Our contributions are summarized as follows:

•

S³martCirc Framework: We propose a mask-based MI framework that learns node-level circuits over both attention heads and feedforward neurons using self-supervised faithfulness and correctness objectives together with a structured coefficient regularizer.

•

Unified evaluation protocol: We define a common computational graph abstraction, an end-to-end circuit extraction step, and a consistent set of metrics, allowing patching-based and optimization-based methods to be compared on equal footing.

•

Empirical characterization: We evaluate a broad set of existing discovery methods across three LLM architectures and two established tasks, report their performance and computational cost, and discuss the factors that govern their behavior.

## Preliminaries and Definitions

In this section, we first introduce the notation used throughout the paper. We then examine circuits from two distinct tasks that have been studied extensively in prior work and that serve as our evaluation settings.

Given an LLM, let $P$ be its output probability distribution and $V$ be its vocabulary. We model the LLM as a computational graph in which each node is a model component (i.e. attention head or feedforward layer neuron) and each edge represents information flowing between components through the residual stream, as illustrated in Figure 2. A circuit is the subgraph within the computational graph responsible for the model’s performance on a given task. Each node takes the previous layer’s representation $X\in\mathbb{R}^{T\times d_{m}}$ as input and produces an output $Y\in\mathbb{R}^{T\times d_{n}}$ , where $T$ is the number of input tokens, $d_{m}$ is the model’s hidden dimension, and $d_{n}$ is the node’s output dimension. For attention heads, $d_{n}$ is determined by the architecture. For feedforward neurons, $d_{n}=1$ . All discovery methods we compare operate over the same node set and graph abstraction to ensure a fair comparison.

Current MI research typically identifies and interprets nodes in the context of a single task. When comparing circuits across different tasks, one observes that some components are reused across tasks (Merullo et al. 2024), which underscores the value of discovery methods that are applied consistently across settings. Consider the two tasks shown in Figure 1: the Indirect Object Identification (IOI) task (Wang et al. 2022), where the model identifies the indirect object in sentences (e.g., "When John and Mary went to the store, John gave a drink to [Mary]"), and the Acronym Prediction task (García-Carrasco et al. 2024a), where the model predicts the last letter of a three-letter acronym from a word sequence (e.g., "The Three Letter Acronym" $\rightarrow$ "TLA"). Both tasks have been analyzed in depth in GPT-2, and their circuits have been described in terms of named components such as previous token heads and mover heads.

Figure 1: Circuits reported in prior work for the IOI and Acronym Prediction tasks.

In the IOI circuit, prior analysis identified previous token heads, induction and duplicate token heads, S-inhibition heads, and name mover and negative name mover heads, which together detect the repeated subject, suppress it, and promote the remaining name in the output logit distribution (Wang et al. 2022). In the acronym circuit, previous token heads, bridge heads, and letter mover heads were identified as jointly responsible for retrieving the capital letter of the relevant word and writing it to the output position (García-Carrasco et al. 2024a). These descriptions were obtained through careful, task-specific interventions and provide a valuable reference: a discovery method that is sensitive to the model’s true computation should recover a substantial portion of these components when applied to the same model.

These two tasks are complementary as evaluation settings. IOI requires the model to track entities across a sentence and to suppress a distractor, whereas Acronym requires positional retrieval of sub-word information. Their circuits span early, middle, and late layers, and they involve both attention-based routing and output-facing components. Together, they offer a demanding yet well-understood testbed for discovery methods, and they allow us to examine how methods behave when moved from GPT-2, where the reference circuits were established, to more recent architectures for which no reference exists.

## Self-supervised Smart Circuit Discovery

Figure 2: An overview of the proposed S³martCirc framework.

In this section, we first present an overview of the proposed framework, S³martCirc. We then describe the implementation of each stage in detail and outline the training process.

### S³martCirc Overview

S³martCirc learns, for every node, how strongly it should be retained in the circuit. Specifically, for every node $n$ in the LLM, we introduce a set of learnable masking coefficients $c_{n}=\{c_{n,u},c_{n,i}\}$ , where $c_{n,u}$ denotes the probability that node $n$ is unimportant and $c_{n,i}$ denotes the probability that it is important for the task. These coefficients are constrained such that $c_{n,u}+c_{n,i}=1$ . We optimize these coefficients through the procedure shown in Figure 2. In the first stage, Important Node Discovery, we learn coefficients that preserve task behavior when unimportant nodes are suppressed. Nodes satisfying $c_{n,i}>c_{n,u}$ are selected as candidate circuit nodes. In the second stage, Important Node Classification, the soft coefficients are converted into a discrete circuit and pruned to an end-to-end connected subgraph. During training, node masking is applied softly using the coefficients $c_{n}$ , while hard assignments are used only during evaluation through the following indicator function:

| | $\hat{c}_{i}=\begin{cases}1,&i=\arg\max_{j}c_{j}\\
0,&\text{otherwise}\end{cases}$
| | (1)

Soft masking during training allows gradients to flow to every coefficient, so that the contribution of each node is evaluated in the context of all other nodes rather than in isolation. This is the principal advantage of mask-based discovery over patching: a node whose effect is only visible when another node is absent, or whose contribution is redundant with another node, can be handled naturally by the joint optimization.

### Important Node Discovery

A circuit contains the nodes responsible for performing the task at hand. Consequently, removing any truly important node from the circuit should substantially degrade task performance. We identify important nodes by optimizing two complementary objectives: (1) minimizing the KL divergence between the circuit’s output distribution and that of the full model to ensure that the circuit faithfully reproduces the model’s original computation; and (2) minimizing cross-entropy loss with respect to the ground-truth answer to ensure the circuit produces correct predictions.

Formally, let $P_{\text{orig}}$ denote the output probability distribution of the full LLM, and $P_{\text{mask}}$ denote the distribution obtained when unimportant nodes are masked (i.e., nodes where $c_{n,u}>c_{n,i}$ ). The combined objective becomes

| | $L_{\text{disc.}}=\alpha\cdot\sum_{v\in V}P_{\text{orig}}(v)\log\frac{P_{\text{orig}}(v)}{P_{\text{mask}}(v)}-(1-\alpha)\cdot(\log P_{\text{mask}}(y)).$
| | (2)

Here, $v\in V$ represents a token in the LLM’s vocabulary, and $P_{\text{orig}}(v)$ and $P_{\text{mask}}(v)$ represent the probability assigned to token $v$ by the full model and the model when unimportant nodes are masked, respectively. $y\in V$ is the ground truth token, and $\alpha\in[0,1]$ controls the trade-off between faithfulness to the original model behavior and task performance.

The two terms capture different aspects of what it means for a circuit to be responsible for a behavior. The KL term considers the entire output distribution and therefore penalizes changes in the probability assigned to alternative tokens, not only to the correct one. The cross-entropy term focuses on the ground-truth answer and thus rewards retaining the components that make the correct prediction possible, even when their removal would not dramatically reshape the rest of the distribution. Methods based purely on logit differences or distributional divergence can underweight such components, and the mixing coefficient $\alpha$ allows the balance between the two views to be adjusted per task.

### Important Node Classification

Once the masking coefficients have been learned, they must be turned into a discrete circuit that can be inspected and evaluated. This classification step separates important from unimportant nodes and ensures that the resulting subgraph is meaningful as a circuit rather than an unstructured collection of components.

The first step applies Equation 1 to each node’s coefficients, assigning the node to the circuit if $c_{n,i}$ is the larger of its two coefficients. Because the regularizer described below pushes coefficients toward binary values, most nodes are assigned with high confidence by the end of training, and the hard assignment closely matches the soft mask used during optimization. This reduces the gap between the training-time and evaluation-time models, which is a well-known source of discrepancy in mask-based pruning (Frankle and Carbin 2019).

The second step enforces connectivity. Let $N_{i}$ denote the set of nodes identified as important (i.e., those satisfying $c_{n,i}>c_{n,u}$ ). A node in $N_{i}$ that cannot receive information from the input embeddings through other circuit nodes or the residual stream, or whose output cannot reach the unembedding, cannot participate in an end-to-end computation from the prompt to the prediction. We therefore remove such nodes, yielding the final circuit. The same extraction step is applied to every method we compare, so that differences in reported circuits reflect differences in node selection rather than in post-processing.

This two-step procedure keeps classification deliberately simple and method-agnostic. Its purpose is not to add information beyond the learned coefficients, but to provide a uniform and reproducible bridge between a continuous relaxation and the discrete objects that MI practitioners analyze.

### S³martCirc Training Process

Due to the vast size of LLMs, we introduce coefficient regularization to encourage the identification of minimal circuits while maximizing task performance. The objective consists of three terms that promote (1) confident assignments by enforcing binary values, (2) valid probability distributions over node assignments (i.e., coefficients sum to one), and (3) sparse circuit selection:

| | $\displaystyle L_{\text{sparsity}}$ | $\displaystyle=\frac{1}{|C|}\sum_{c_{n}\in C}\Big(\underbrace{\sum_{v\in c_{n}}\Big[\frac{1}{2}|v(1-v)|\Big]}_{\text{confidence}}$
| | (5)

|
| | $\displaystyle+\underbrace{\frac{1}{2}|1-(c_{n,u}+c_{n,i})|}_{\text{validity}}$
|

|
| | $\displaystyle+\underbrace{c_{n,i}}_{\text{sparsity}}\Big),$
|

where $C$ is the set of all masking coefficients in the model. The first term pushes each coefficient toward binary values (0 or 1), the second term enforces that the two probabilities sum to one, and the third term directly penalizes nodes being included in the circuit, encouraging sparsity.

The overall training objective is the sum of the discovery objective and the regularizer, weighted by a sparsity coefficient that controls how aggressively nodes are removed. Optimizing randomly initialized coefficients with a strong sparsity penalty from the very first step can drive many coefficients toward zero before the faithfulness and correctness signals have had a chance to shape them. We therefore introduce a warmup phase during which the discovery objective ( $L_{\text{disc.}}$ ) dominates, allowing the coefficients to settle around a reasonable set of task-relevant nodes before the full regularized objective is applied. After warmup, coefficients are trained with the complete objective until convergence, and the final circuit is extracted with the classification procedure described above.

## Experiments

Table 1: Performance of existing circuit discovery baselines under our unified evaluation protocol over three runs. Empty cells indicate that no circuit could be found within the set of nodes selected by the method. An asterisk (*) indicates that only one experimental run was completed due to the time limit (48 hours).

|
| | Acronym | IOI

|
| | Circuit Size | Acc. | Logit. | Circuit Size | Acc. | Logit.

| GPT-2 | Act. | $35.00_{\pm 4.32}$ | - $54.67_{\pm 38.33}$ | - $3.54_{\pm 38.33}$ | $39.33_{\pm 10.40}$ | - $70.71_{\pm 25.89}$ | - $9.11_{\pm 25.89}$

| Attr. | $37.33_{\pm 2.05}$ | - $0.17_{\pm 0.24}$ | $0.46_{\pm 0.24}$ | $48.00_{\pm 2.16}$ | - $12.73_{\pm 8.01}$ | - $5.14_{\pm 8.01}$

| Attr. IG | $49.67_{\pm 11.44}$ | - $2.00_{\pm 1.08}$ | - $0.59_{\pm 1.08}$ | $38.00_{\pm 5.10}$ | - $0.17_{\pm 0.24}$ | - $0.29_{\pm 0.24}$

| IBCircuit | – | – | – | – | – | –

| Prune. | $32.33_{\pm 37.97}$ | $0.00_{\pm 0.00}$ | $0.07_{\pm 0.00}$ | $38.33_{\pm 33.81}$ | $0.00_{\pm 0.00}$ | - $0.02_{\pm 0.00}$

| Random | $60.33_{\pm 7.76}$ | $0.00_{\pm 0.00}$ | $0.04_{\pm 0.00}$ | $77.33_{\pm 3.40}$ | $0.00_{\pm 0.00}$ | - $0.11_{\pm 0.00}$

| Llama | Act. | – | – | – | $13.00_{\pm 2.00}$ | - $59.17_{\pm 41.90}$ | - $2.37_{\pm 41.90}$

| Attr. | – | – | – | $71.67_{\pm 2.05}$ | - $1.50_{\pm 0.41}$ | - $0.35_{\pm 0.41}$

| Attr. IG | $54.67_{\pm 2.62}$ | - $2.17_{\pm 0.24}$ | - $0.34_{\pm 0.24}$ | $47.67_{\pm 3.30}$ | - $5.67_{\pm 3.70}$ | - $0.35_{\pm 3.70}$

| IBCircuit | $29.67_{\pm 12.66}$ | $0.00_{\pm 0.00}$ | - $0.08_{\pm 0.00}$ | $7.33_{\pm 0.94}$ | $0.00_{\pm 0.00}$ | $0.00_{\pm 0.00}$

| Prune. | $70.67_{\pm 3.09}$ | $0.00_{\pm 0.00}$ | $0.01_{\pm 0.00}$ | $70.67_{\pm 3.09}$ | $0.00_{\pm 0.00}$ | $0.01_{\pm 0.00}$

| Random | $60.00_{\pm 9.93}$ | $0.00_{\pm 0.00}$ | - $0.00_{\pm 0.00}$ | $62.33_{\pm 6.18}$ | $0.00_{\pm 0.00}$ | - $0.01_{\pm 0.00}$

| Qwen | Act. | $32.00_{\pm 0.00}$ * | $-100.00_{\pm 0.00}$ * | - $1.00_{\pm 0.00}$ * | $12.00_{\pm 0.00}$ * | - $33.00_{\pm 0.00}$ * | - $0.33_{\pm 0.00}$ *

| Attr. | $64.33_{\pm 3.30}$ | - $0.17_{\pm 0.24}$ | - $0.48_{\pm 0.24}$ | $29.00_{\pm 3.56}$ | - $0.50_{\pm 0.41}$ | - $0.91_{\pm 0.41}$

| Attr. IG | $32.33_{\pm 8.34}$ | - $0.17_{\pm 0.24}$ | - $0.12_{\pm 0.24}$ | $35.00_{\pm 5.35}$ | - $0.33_{\pm 0.47}$ | - $0.19_{\pm 0.47}$

| IBCircuit | $14.67_{\pm 0.94}$ | $0.00_{\pm 0.00}$ | $0.29_{\pm 0.00}$ | – | – | –

| Prune. | $46.00_{\pm 4.24}$ | $0.00_{\pm 0.00}$ | - $0.01_{\pm 0.00}$ | $46.00_{\pm 4.24}$ | $0.00_{\pm 0.00}$ | - $0.01_{\pm 0.00}$

| Random | $52.67_{\pm 9.74}$ | $0.00_{\pm 0.00}$ | - $0.00_{\pm 0.00}$ | $48.00_{\pm 6.98}$ | $0.00_{\pm 0.00}$ | $0.01_{\pm 0.00}$

In this section, we first introduce the experiment setup. Then, we address the following research questions: RQ1. How do existing circuit discovery methods behave under a unified evaluation protocol? RQ2. To what extent can known circuits from prior work serve as a reference for automated discovery? RQ3. Which hyperparameters govern the behavior of mask-based discovery? RQ4. What role do the individual components of the S³martCirc training pipeline play?

### Experiment Setup

LLMs. We select three LLMs of varying sizes for our experiments: GPT-2 (Radford et al. 2019; Maintainers 2022), Llama 3.2 1B (Grattafiori et al. 2024; Meta 2024), and Qwen 3 1.7B (Team 2025a; Team 2025b). GPT-2 has been extensively examined in prior MI work, making it a well-understood and widely used benchmark for circuit analysis. Llama 3.2 1B and Qwen 3 1.7B are included to evaluate how discovery methods generalize across model architectures.

Evaluation Tasks. We collect two tasks to perform circuit discovery on: Indirect Object Identification (IOI) (Wang et al. 2022) and Acronym Prediction (Acronym) (García-Carrasco et al. 2024a). Both tasks are drawn from previous MI work where extensive experiments have been performed to uncover the task circuit in GPT-2.

Evaluation Metrics. We use three metrics: Circuit Size, Accuracy Drop (Acc.), and Logit Difference (Logit.). Each method selects a set of important nodes. We then remove nodes that are not connected to the input or output, yielding the final end-to-end circuit; Circuit Size is defined as the size of this circuit. Unlike during training, we evaluate circuit performance by masking only the circuit nodes in the model, denoted as $P_{\text{circ}}$ . Acc. measures the percent drop in accuracy between the original model and $P_{\text{circ}}$ . Lastly, Logit. denotes the difference in the logit assigned to the original model’s predicted answer between the original model and $P_{\text{circ}}$ .

Baselines.
We adopt six circuit discovery methods as baselines. We first consider patching-based approaches: Activation Patching (Act.) (Heimersheim and Nanda 2024) measures each node’s contribution by ablating it (i.e., replacing its activation with a corrupted counterpart) and evaluating the resulting change in logit difference between the model’s original answer and an alternative response. Attribution Patching (Attr.) (Syed et al. 2023) provides a linear approximation of each node’s contribution by multiplying the difference between its clean and corrupted activations with the gradient of a logit-based metric (i.e., logit difference between the original and corrupted outputs). Attribution Patching with Integrated Gradients (Attr. IG) (Hanna et al. 2024) extends attribution patching by integrating gradients along a path between the corrupted and clean activations, yielding a more accurate attribution score. We next consider optimization-based approaches: IBCircuit (Bian et al. 2026) performs circuit discovery by optimizing an information bottleneck objective to identify compact, task-relevant subgraphs. Pruning (Prune.) (Bhaskar et al. 2024) applies gradient-based pruning to remove less important edges; we adapt this method to operate at the node level for fair comparison. Random serves as a lower bound by randomly selecting $k$ nodes as the circuit.

### RQ1: Performance of S³martCirc

Table 1 reports the performance of the baselines under the unified protocol, providing the reference landscape against which mask-based discovery in the S³martCirc framework is situated. Several observations follow.

Behavior of patching-based methods. Activation Patching is the only baseline that produces substantial accuracy drops across most settings, for example -54.67% on Acronym and -70.71% on IOI with GPT-2. However, its variance across runs is large, indicating that the selected nodes are sensitive to the particular corrupted samples used. Its approximations, Attribution Patching and Attr. IG, are far cheaper but yield circuits whose removal barely affects accuracy in most settings. This is consistent with the observation that linear and path-integrated approximations can diverge from the true effect of an intervention, especially for components whose contribution is nonlinear or depends on other components.

Behavior of optimization-based methods. Pruning and IBCircuit, in their node-level adaptations, frequently select circuits whose removal leaves accuracy unchanged, and IBCircuit fails to produce a connected circuit in several settings. These outcomes illustrate how sensitive mask-based methods are to the choice of objective, granularity, and connectivity post-processing: an objective that is well suited to edge-level discovery or to preserving behavior when the circuit is kept may not translate into selecting the nodes whose removal disrupts behavior. This sensitivity is a central motivation for the self-supervised faithfulness and correctness objectives and the structured regularizer in S³martCirc.

Circuit Size. Circuit sizes vary widely across methods and models, and smaller circuits do not systematically correspond to larger performance effects. For example, on IOI with Llama, Attr. IG produces circuits averaging 47.67 nodes with a -5.67% accuracy drop, while IBCircuit produces circuits of only 7.33 nodes with no measurable drop. Circuit size should therefore always be interpreted jointly with the accuracy and logit metrics; a compact circuit is informative only if its components are in fact critical to the behavior.

Cross-model variation. The same method can behave very differently across architectures. Activation Patching, for instance, does not produce a connected circuit on Acronym with Llama, yet reaches a complete accuracy drop on Acronym with Qwen in the single run completed within the time limit. Such variation underlines why evaluation across multiple model families is necessary before drawing conclusions about a discovery method.

### RQ2: Recovering Prior Circuits

Previous work (Wang et al. 2022) manually identified and verified a ground-truth circuit for the IOI task through extensive analysis and validation. This circuit serves as a reliable reference for evaluating whether automated methods can recover the same critical components. Accordingly, a desirable property of circuit discovery methods is the ability to recover nodes from this known IOI circuit. Figure 3 reports the percentage of nodes from the original IOI circuit that are recovered by each method within the final end-to-end discovered circuits. We evaluate this across varying circuit sizes obtained by thresholding at 50, 100, 150, and 200 nodes. Note that perfect recovery (100%) is not achievable due to the vast number of possible nodes to choose from.

Recovery of a reference circuit complements the masking-based metrics of RQ1. Accuracy drop measures whether the selected nodes are necessary for the behavior, whereas recovery measures whether they coincide with components that independent, intervention-heavy analysis has established as part of the mechanism. A method may score well on one of these criteria and poorly on the other: it could select nodes whose removal is damaging for reasons unrelated to the described mechanism, or it could recover many reference nodes while also including large numbers of irrelevant ones. Reporting recovery at several thresholds helps disentangle these possibilities, since recovery at small thresholds rewards precise selection while recovery at large thresholds rewards coverage.

The same reasoning applies to the Acronym Prediction task (García-Carrasco et al. 2024a), whose reference circuit is shown in Figure 4. In the figure, each node is labeled as layer.head (e.g., 8.11 represents layer 8, attention head 11). The acronym circuit is considerably smaller than the IOI circuit, which makes recovery more demanding: each missed node represents a large fraction of the reference. Moreover, prior work reports that some heads in this circuit, such as 8.11 and 10.10, can act as either bridge heads or letter mover heads depending on the specific intervention applied. This multi-functionality illustrates an important caveat for any comparison with human-verified circuits: the reference describes what a head can do under carefully isolated interventions, whereas automated methods estimate a head’s contribution averaged over many samples. Agreement with the reference should therefore be interpreted at the level of node membership rather than as confirmation of a specific mechanistic narrative.

Reference circuits also have limitations of their own. They were established in GPT-2 only, so for Llama and Qwen no comparable reference exists, and they may omit components with small but consistent contributions that fell below the thresholds used in the original analyses. We therefore view recovery as one useful lens among several, most informative when considered alongside the masking-based metrics in Table 1.

Figure 3: Percent of the IOI circuit recovered by different circuit discovery methods

### RQ3: Parameter Analysis

To answer RQ3, we consider the hyperparameters that govern mask-based discovery in S³martCirc, with Figure 5 summarizing the analysis on the IOI task using GPT-2. The most important of these is $\alpha$ , which balances the two objectives in the Important Node Discovery stage, evaluated across $\alpha\in\{0.0,0.25,0.5,0.75,1.0\}$ . The two endpoints correspond to qualitatively different notions of a circuit. At $\alpha=1.0$ , only distributional faithfulness matters, and the masked model is asked to reproduce the full model’s output distribution regardless of whether it predicts the correct answer. At $\alpha=0.0$ , only correctness matters, and the masked model may alter the remaining distribution freely as long as it assigns high probability to the ground-truth token. Intermediate values interpolate between these views, and the appropriate balance can be expected to depend on how concentrated the model’s output distribution is for the task.

Beyond $\alpha$ , the sparsity coefficient determines how strongly circuit membership is penalized and hence directly controls the trade-off between circuit size and the strength of the retained behavior. Larger values produce smaller circuits but risk discarding components that contribute modestly yet consistently. The length of the warmup phase determines how long the discovery objective shapes the coefficients before regularization takes full effect; very short warmups expose the coefficients to the sparsity penalty before a reasonable candidate set has formed, while very long warmups delay convergence. The values used for each model and task are listed in Appendix A, and they vary noticeably across settings, which reflects the differing sizes of the models and the differing structure of the two tasks.

These considerations suggest that, as with other mask-based methods, careful per-setting tuning is needed for reliable behavior. They also connect to the behavior of the baselines in RQ1: methods that rely primarily on output distribution differences emphasize the faithfulness view, and components that support the correct prediction without inducing large changes in logits may be underweighted by such criteria.

Figure 4: The reference Acronym circuit reported in prior work.

### RQ4: Ablation Study

The S³martCirc training pipeline consists of several components whose roles can be examined individually, as summarized in Figure 6: the faithfulness term, the correctness term, the three parts of the coefficient regularizer, the warmup phase, and the connectivity-based extraction step. Each component addresses a distinct failure mode of naive mask learning.

Discovery objectives. The faithfulness and correctness terms in Equation 2 provide the training signal for the masks. Without the faithfulness term, the optimization has no incentive to preserve the model’s broader output behavior and may retain only the narrow set of components needed to boost one token. Without the correctness term, it may preserve the overall shape of the distribution while losing the components responsible for the actual answer. Combining the two makes the learned circuit sensitive to both aspects of the behavior.

Coefficient regularizer. The confidence term reduces the mismatch between soft training-time masks and hard evaluation-time masks; without it, many coefficients may hover near intermediate values, and the hard assignment in Equation 1 becomes a poor approximation of the trained model. The validity term keeps coefficients interpretable as probabilities over the two assignments. The sparsity term is what makes the circuit compact rather than a near-copy of the full model.

Figure 5: Effect of hyperparameters on S³martCirc.

Warmup and extraction. The warmup phase protects early optimization from a sparsity penalty that would otherwise act on uninformed coefficients, allowing the discovery objective to establish a reasonable candidate set first. In particular, without such an initial phase, the optimization may struggle to generate informative gradients for distinguishing important from unimportant components. The connectivity-based extraction step, finally, ensures that every reported circuit corresponds to an end-to-end computation, and because it is shared across all methods, it removes one source of inconsistency in comparisons.

Taken together, these components reflect a design philosophy in which each element of the pipeline is simple and has a clear purpose, so that the behavior of the overall framework can be understood in terms of its parts.

Figure 6: Overview of the components of S³martCirc and its training process.

## Related Works

Mechanistic Interpretability and Circuit Discovery.
Mechanistic interpretability (MI) explains model behavior by identifying the components responsible for it (Bereska and Gavves 2024; Sharkey et al. 2025; Rai et al. 2024), with the circuit as its central abstraction. Most work follows a where-then-what paradigm (Méloux et al. 2025a; García-Carrasco et al. 2024b; Chandna et al. 2025), localizing components via patching-based interventions (Wang et al. 2022; Meng et al. 2022; García-Carrasco et al. 2024a; Heimersheim and Nanda 2024) or gradient-based approximations (Syed et al. 2023; Hanna et al. 2024) and then interpreting their roles manually. Such interventions are costly and evaluate components in isolation, while the interpretation step is subjective and hard to scale. Some work considers reversing the order of these stages, first hypothesizing the operations a task requires and then searching for components that implement them (Méloux et al. 2025b). Recent studies further show that components are reused across tasks (Merullo et al. 2024), that behavior can be localized to individual neurons (Nikankin et al. 2026), and that structurally distinct circuits can be functionally equivalent (Haklay et al. 2026). These findings emphasize the importance of discovery methods that operate at fine granularity, account for component interactions, and are evaluated consistently across tasks and models.

Optimization-Based Circuit Discovery.
A complementary line optimizes continuous masks over components under sparsity constraints (Frankle and Carbin 2019; Bhaskar et al. 2024), e.g., IBCircuit’s information bottleneck objective (Bian et al. 2026) and behavior-specific mask learning (Li and Janson 2024). These better capture component interactions than patching. For example, IBCircuit formulates circuit extraction as an optimization problem balancing behavioral fidelity with an information bottleneck objective (Bian et al. 2026), while other approaches optimize masks to isolate behavior-specific components (Li and Janson 2024). S³martCirc belongs to this family and differs in that (1) its training signal combines distributional faithfulness with correctness on the ground-truth answer; (2) its regularizer explicitly targets confidence, validity, and sparsity of the mask coefficients, together with a warmup schedule; and (3) discovery spans both attention heads and MLP neurons, with a shared end-to-end extraction step used for evaluation.

## Conclusion

We introduce S³martCirc, a self-supervised, mask-based framework for circuit discovery in LLMs. S³martCirc learns soft masks over attention heads and feedforward neurons using faithfulness and correctness objectives derived from the model itself, regularizes the masks toward confident, valid, and sparse assignments, and converts them into end-to-end connected circuits through a simple, method-agnostic extraction procedure. We further establish a unified evaluation protocol across GPT-2, Llama 3.2, and Qwen 3 on two well-studied tasks, and use it to characterize the behavior and computational cost of existing patching-based and optimization-based discovery methods. Our analysis highlights the high variance of exact patching, the weak agreement between its approximations and true intervention effects, and the sensitivity of mask-based methods to objectives and post-processing. Our framework also has limitations that point to future work: it addresses circuit discovery and leaves functional interpretation of the discovered nodes to subsequent analysis, and reference circuits for validating discovery are currently available only for GPT-2.

## Appendix A Implementation Details

### Computation Resources

All experiments were performed on clusters of NVIDIA RTX A4000 16GB and NVIDIA A16 16GB GPUs except for Attribution Patching and Attribution Patching IG on Qwen. Due to the large model size, these experiments were done on clusters of NVIDIA RTX A6000 48GB GPUs.

### Data Generation

We generate the data samples for each experiment due to the need to maintain token limits based on each model’s tokenizer. Below describes the data generation process for each task, given a tokenizer.

IOI. We adopt the names and place/object pairs from (Wang et al. 2022). For each model, we keep only the names and place/object pairs that tokenize as a single token, yielding a valid set of values. We then enumerate all valid combinations and insert each into the template “When [NAME_1] and [NAME_2] went to the [PLACE], [NAME_1] handed the [OBJECT] to”. Corrupted counterparts are formed by swapping the second name, producing clean/corrupted pairs. Finally, to obtain a balanced set of 1000 samples, we draw $1000/p$ samples from each place/object pair, where $p$ is the number of valid pairs.

Acronym. We generate samples using the RandomWord class from the wonderwords package. For each letter of the alphabet, we sample 500 words and keep a word only if its formatted version (i.e., prepending a space and capitalizing the first letter) tokenizes into exactly two tokens, the first of which is the space-prefixed capital letter. Letters with no valid words are discarded, leaving a set of valid letters. Within this set, we identify valid abbreviations, i.e., three-letter sequences whose concatenation tokenizes into three tokens. For each pair of valid abbreviations, we sample random words matching their letters and insert them into the template “The [A_RAND_WORD] [B_RAND_WORD] [C_RAND_WORD] (AB” to form clean and corrupted samples. Finally, we shuffle the generated data and take the first 1000 samples.

### Hyperparameters

Below, we list the hyperparameters used for each model dataset combination. All experiments used a random seed of 42.

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

In Table 2, we compare the runtime of S³martCirc to the baselines. From the table, we see that S³martCirc is relatively efficient, requiring substantially less time than exact patching and the more expensive approximation and pruning methods across all settings.

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

We also repeated the recovery analysis of Section 4.3 for the Acronym task and show the results in Figure 7. As with IOI, recovery is reported at multiple circuit-size thresholds, and the smaller size of the acronym reference circuit makes this setting a more stringent test of precise node selection.

Figure 7: Percentage of the Acronym circuit recovered by different circuit discovery methods