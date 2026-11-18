# IA2: Alignment with ICL Activations

Improves Supervised Fine-Tuning

Aayush Mishra

  
Daniel Khashabi & Anqi LiuDepartment of Computer Science,
Johns Hopkins University

###### Abstract

Supervised Fine-Tuning (SFT) specializes a model by training its weights, or numerical parameters, to produce intended responses to queries. In-Context Learning (ICL) instead adapts a model during inference, when it produces responses. ICL supplies instructions or demonstrations in the prompt without updating weights.

When labeled data are scarce, ICL can generalize better than SFT and produce better-calibrated responses. Calibration measures how closely a model’s confidence agrees with how often it is correct. ICL requires more inference computation, however.

We ask whether ICL’s internal computations can improve SFT. We first show that ICL and SFT produce distinct activation patterns. Activations are internal values computed by the model. These differences indicate that the two methods adapt through different functional mechanisms (§3; Figure 2).

We introduce ICL Activation Alignment (IA2). It is a self-distillation method: the model learns from its own behavior under ICL. IA2 trains an SFT model to reproduce ICL activation patterns and encourages ICL-like internal reasoning. Using IA2 as a priming step before SFT significantly improves output accuracy and calibration. Our experiments cover 12 popular benchmarks and two model families (§5; Figure 1). This result is practically useful and offers a conceptual window into the internal mechanics of model adaptation.

|

00footnotetext: ♡Equal advising. Correspondence to: {amishr24,danielk,aliu.cs}@jhu.edu

## 1 Introduction

Large language models (LLMs) are general-purpose models, but they are often used in specialized applications. For example, a news aggregator may need to classify articles into predefined categories.

SFT is a popular way to adapt LLMs. It uses labeled examples to train models for narrow downstream tasks. Parameter-efficient fine-tuning (PEFT) methods adapt a model through a small set of parameters (Hu et al., 2021; Liu et al., 2022). With these methods, an SFT model often consists of a small parameter set that can be efficiently loaded into or removed from graphics processing unit (GPU) memory. This makes such models extremely useful. However, SFT typically needs many labeled samples to generalize well to new tasks (Le Scao and Rush, 2021). Collecting these samples can be expensive.

ICL instead adapts and steers an LLM during inference (Brown et al., 2020). The model receives a query preceded by demonstrations or instructions. These help it “learn” the demonstrated task and answer accordingly. Prior work finds that ICL generalizes well with few examples and typically produces well-calibrated responses (Duan et al., 2024). Our experiments agree (§5; Figure 1). However, demonstrations and instructions occupy context space. They increase the cost of each query and leave less space for query and response tokens, the units of text processed by the model.

These differences motivate us to compare how ICL and SFT function internally. Their generated text may look similar, but their internal activations differ (§3; Figure 2). Activations record the model’s processing before it produces the next token. Divergent patterns therefore highlight different adaptation mechanisms. Prior work supports this idea (Shen et al., 2024).

We hypothesize that ICL activations contain rich information about how to extract generalizable patterns from context. This information may be absent from SFT models, especially with few training examples. In that setting, SFT models are prone to shortcut learning and overfitting: they can fit target responses without learning patterns that generalize. We therefore ask:

**Can the information-rich ICL activations be used to improve the quality of SFT?**

Recent work has tried to distill context into LLM weights (Snell et al., 2022; Chen et al., 2024b). These methods use training signals only from response text. They may therefore suffer from the same SFT issues. Training a model to reproduce an ICL-conditioned model’s outputs does not ensure that it functions like that model.

We propose IA2 to align a model with its own functional behavior under ICL (§4). Figure 1 summarizes the pipeline:

1. Collect information-rich ICL activations.
2. Train the model to match those activations without the demonstrations.
3. Perform SFT on this primed model.

We use **IA2 → SFT** to denote this sequential pipeline. Priming with IA2 before SFT drastically improves adaptation on text classification and generation tasks, while using the same data (§5; Tables 1–2). IA2 also supplies an important training signal that SFT-only training does not provide (§5.2; Figures 3–4). We trained over 13,000 models across 12 benchmarks to validate these findings. The results show practical benefits and provide a conceptual window into model adaptation.

Our contributions are:

- We show that ICL and SFT, using the same data, do not align in activation space. This highlights a gap in their functional behavior (§3; Figure 2).
- We propose IA2 to enforce alignment with ICL’s functional behavior. This priming step drastically improves SFT performance (Figure 1; §4).
- We show that the IA2 training signal is absent from SFT-only training. This highlights its importance for adaptation quality (§5.2; Figures 3–4).

**Figure 1:** An overview of our improved SFT pipeline. Standard SFT enforces only output-space alignment: agreement between the model’s response and a target response. It uses only step 3 above. In low-data settings, this produces subpar performance and miscalibration. IA2 instead enforces functional alignment with ICL by matching the rich activation patterns produced under ICL. IA2 priming before SFT boosts adaptation quality. The performance charts aggregate results across models and datasets. See §5 for details.

## 2 Background and Notation

**Transformer Language Models:** A standard decoder-only transformer language model (LM), written \(M_{\Theta}\), has parameters \(\Theta\). It contains a stack of self-attention (SA) and linear layers. Self-attention lets the representation at a token position depend on other tokens in the sequence. A linear layer applies a learned linear transformation.

Let \(T=[t_i]_{i=1}^{R}\) be a sequence of \(R\) token vectors. Here, \(t_i\) is the vector at position \(i\), and \(i\) indexes token positions. Each vector has width \(d\). At each layer, the model applies self-attention and linear projections. At the final layer, the last token’s representation passes through the LM head, which maps it to predictions for the next token.

A standard SA operation uses four parameter matrices: \(W_Q\), \(W_K\), \(W_V\), and \(W_O\), all contained in \(\Theta\). They form queries, keys, values, and an output projection, respectively. We abbreviate this set as \(W_{QKVO}\). Its output is:

\[
\operatorname{SA}(T;W_{QKVO})
=
Z
=
\left[
z_i
=
\sigma\left(\frac{q_iK^T}{\sqrt d}\right)V\cdot W_O
\right]_{i=1}^{R}.
\tag{1}
\]

Here:

- \(\sigma(\cdot)\) is Softmax, which converts scores into normalized nonnegative weights.
- \(q_i=t_i\cdot W_Q\) is the query at position \(i\).
- \(K=t_{:i}\cdot W_K\) is the matrix of keys for the first \(i\) tokens.
- \(V=t_{:i}\cdot W_V\) is the corresponding matrix of values.
- \(t_{:i}\) denotes the first \(i\) token vectors.
- A superscript \(T\), as in \(K^T\), denotes matrix transpose. It is distinct from the sequence named \(T\).
- The dot denotes matrix multiplication, and \(\sqrt d\) is the square root of the vector width.
- \(z_i\) is the SA output at position \(i\). The bracketed expression collects all \(R\) outputs into \(Z\).

The output \(Z\) has the same shape as the input \(T\): \(R\times d\), or \(R\) vectors of width \(d\). We study how model behavior changes when either the context \(T\) or the weights \(W_{QKVO}\) change.

**Supervised Fine-Tuning:** Let

\[
\mathcal D_{\mathcal T}=\{(X_i,Y_i)\}_{i=1}^{N}
\]

be a dataset for task \(\mathcal T\). It contains \(N\) input–output pairs. \(X_i\) is the input and \(Y_i\) is its ground-truth output.

SFT updates the model parameters from \(\Theta\) to \(\Theta'\). For a new input \(X_t\), the updated model produces a first response token:

\[
M_{\Theta'}(X_t)=\bar y_t.
\]

The subscript \(t\) identifies the new test sample. Repeatedly applying the model to the growing sequence extends \(\bar y_t\) into the full response \(\bar Y_t\). A bar marks an SFT model’s generated response.

Suppose the ground-truth response \(Y_i\) contains \(G\) tokens. We generate \(G\) tokens to obtain \(\bar Y_i\), then minimize cross-entropy over all generated tokens. Cross-entropy measures disagreement between the model’s token prediction and the target token. The loss is:

\[
\mathcal L_{\mathrm{SFT}}
=
\sum_{i=1}^{N}
\sum_{j=1}^{G}
\operatorname{cross\text{-}entropy}(\bar Y_{ij},Y_{ij}).
\tag{2}
\]

Here, \(\mathcal L_{\mathrm{SFT}}\) is the SFT training loss, \(j\) indexes response tokens, and \(\bar Y_{ij}\) and \(Y_{ij}\) refer to the generated and ground-truth tokens at response position \(j\) for sample \(i\).

**In-Context Learning:** The same dataset can supply demonstrations without weight updates. Let

\[
I=[X_1\circ Y_1\circ\ldots\circ X_n\circ Y_n]
\]

be the demonstration sequence, where \(n\) is the number of demonstrations and \(\circ\) means concatenation. The model receives \(I\) followed by a new query \(X_t\):

\[
M_\Theta(I\circ X_t)=\hat y_t.
\]

Here, \(\hat y_t\) is the first generated token under ICL. A hat marks an ICL-generated response. Repeated generation produces the full response \(\hat Y_t\).

The model processes the demonstrations to understand the task and answer accordingly. Remarkably, \(\hat Y_t\) is often similar to the expected response \(Y_t\). ICL is therefore a useful inference-time adaptation method. It was first illustrated in the GPT-3 language model (Brown et al., 2020) and is widely used to adapt generic LMs during inference.

Today’s frontier models can follow instructions directly. A generic instruction could therefore be considered a form of ICL demonstration. We focus instead on classic ICL, where demonstrations are input–output pairs.

**Activations:** SA uses every token available in the context, so different token sequences produce different outputs. We call the hidden SA outputs **activations**. They record internal processing and can reveal behavioral changes caused by changing context.

If \(M_\Theta\) has \(L\) layers and processes \(R\) tokens, its activation tensor \(A\) has size \(L\times R\times d\). A tensor here is an array indexed by layer, token position, and vector coordinate.

We study:

- \(T=[I\circ X]\) for ICL, where \(I\) contains demonstration tokens and \(X\) contains query tokens.
- \(T=X\) without ICL, under different model weights.

We define activation similarity, \(\operatorname{asim}(A_1,A_2)\), as token-wise cosine similarity between two activation tensors \(A_1\) and \(A_2\). Cosine similarity compares the directions of two vectors. It produces one similarity value for each layer and token position. Thus, \(\operatorname{asim}\) has size \(L\times R\).

## 3 Are ICL and SFT practically the same?

**Figure 2: Layerwise Similarity with ICL Activations.** SFT models have small activation similarity, or asim, with ICL. This indicates different functional behavior during adaptation. Our recommended pipeline, IA2 → SFT, aligns much more closely with ICL than SFT alone and performs much better as a result (Figure 1).

Recent studies claim that ICL in transformers works through an internal gradient-descent mechanism (Von Oswald et al., 2023; Akyürek et al., 2022). Gradient descent normally updates parameters to reduce a training loss. These studies show that self-attention can produce outputs as though the model parameters had been updated using the demonstrations in context. This implies functional equivalence between ICL and parameter updates.

Under this equivalence, we should expect similar activations at output-token positions in two cases: the base model receives ICL demonstrations, and the SFT model receives no demonstrations. We tested whether LLMs exhibit this behavior.

**ICL and SFT produce different activations:** We collected ICL activations, \(A_{\mathrm{ICL}}\), using \(T=[I\circ X]\), for 100 test samples from multiple datasets. These included SST2, a sentiment-classification dataset, and AGNews, a news-categorization dataset. We used models from the Qwen3 (Yang et al., 2025) and Llama-3.2 (Grattafiori et al., 2024) families. The model sizes and layer counts were 1B ↔ 16 layers, 3B ↔ 28 layers, and 4B ↔ 36 layers. Here, B denotes billion parameters, and each arrow links a model size to its layer count.

We then trained SFT models to convergence using the same demonstrations. Convergence means that training reached its stopping point under the training procedure. We collected their activations, \(A_{\mathrm{SFT}}\), using \(T=X\). See §B for experimental details.

We calculated \(\operatorname{asim}(A_{\mathrm{ICL}},A_{\mathrm{SFT}})\) at output-token positions. We averaged across tokens, samples, and datasets, then plotted the results in Figure 2, first row, “SFT only.” Across models, ICL and SFT activations do not align.

The activations align better near the initial and final layers. We expect those layers to process tokens individually. They are more misaligned in the middle layers, where we expect more abstract processing of the whole demonstration set.

**Why should we care about alignment with ICL?** The functional difference also appears in expected calibration error (ECE), a measure of mismatch between confidence and accuracy (Guo et al., 2017). Figure 1 shows that ICL is much better calibrated than SFT at similar, slightly better accuracy.

We suspect that this difference comes from SFT’s output-oriented training signal. That signal can permit shortcuts that fail on new data. In contrast, ICL relies on complex circuits—internal computational pathways—that must work across tasks (Elhage et al., 2021). It extracts generalizable patterns from demonstrations and performs in a well-calibrated manner.

This experiment demonstrates that ICL and SFT adapt through different means. SFT should therefore not be expected to serve as a drop-in replacement for ICL. These differences motivate our question:

**Can the information-rich ICL activations be used to improve the quality of SFT?**

Next, we show that they can and describe our proposed SFT procedure.

## 4 Inducing ICL-like behavior in SFT Models

Our pipeline has two training steps. First, IA2 primes the model to encourage ICL-like functional behavior. Then, further SFT on the aligned weights produces much better adaptation.

**IA2—ICL Activation Alignment:** Section 3 found little activation similarity between SFT and ICL. To increase this similarity, and therefore functional similarity with ICL, we set the following goal at each layer of \(M_\Theta\).

Find modified attention weights

\[
\tilde W_{QKVO}
=
(\tilde W_Q,\tilde W_K,\tilde W_V,\tilde W_O)
\]

such that, at every newly generated token, indexed by \(-1\), meaning the last position:

\[
\operatorname{SA}([I\circ X];W_{QKVO})
\approx
\operatorname{SA}(X;\tilde W_{QKVO})
\quad \forall X\in\mathcal T.
\tag{3}
\]

A tilde marks modified weights. The symbol \(\approx\) means approximate agreement. The condition \(\forall X\in\mathcal T\) means that the goal applies to every query \(X\) from task \(\mathcal T\).

Prior work gives a closed-form solution for linearized attention models (Chen et al., 2024a). Those models replace Softmax with a kernel approximation. We seek a practical, general solution for real, nonlinear transformers.

The goal is to place ICL’s qualities directly into the SFT model’s weights. These weights are not trained to produce similar output text. They are trained to process queries as ICL does at each layer of the base model.

For each training input \(X_i\), we first generate an ICL response \(\hat Y_i\) using

\[
T_i=[I\circ X_i].
\]

The demonstrations \(I\) come from the remaining training samples. We collect activations at every output position. Assuming \(G\) response tokens, this gives

\[
A^i_{\mathrm{ICL}}\in\mathbb R^{L\times G\times d}.
\]

Here, \(\mathbb R^{L\times G\times d}\) denotes real-valued arrays with \(L\) layers, \(G\) response positions, and width \(d\). The superscript \(i\) identifies the training sample.

Next, we attach that same generated response to the query without demonstrations:

\[
T_i=[X_i\circ\hat Y_i].
\]

This treats the ICL response as though the model had generated it from the query alone. We collect the resulting, unaligned output-position activations \(A^i\). We then train the model parameters to minimize the mean squared error between the two activation tensors. The loss is written as:

\[
\mathcal L_{\mathbf{IA2}}
=
\sum_{i=1}^{N}
\left\|A^i-A^i_{\mathrm{ICL}}\right\|.
\tag{4}
\]

Here, \(\mathcal L_{\mathbf{IA2}}\) is the IA2 loss, and the norm \(\|\cdot\|\) measures the activation difference.

**IA2 → SFT:** After IA2, we continue training with the standard SFT cross-entropy loss in Equation (2), using ground-truth target tokens.

Suppose IA2 updates the parameters from \(\Theta\) to \(\Theta'\). We use \(M_{\Theta'}\) to collect responses \(\bar Y_i\) for all training inputs \(X_i\). We minimize the SFT loss between these responses and the ground-truth responses \(Y_i\). This further aligns the outputs with the intended targets.

The complete procedure is:

1. Collect target ICL activations \(A_{\mathrm{ICL}}\). Train with \(\mathcal L_{\mathbf{IA2}}\) until convergence to align functional behavior with ICL.
2. Switch to \(\mathcal L_{\mathrm{SFT}}\), using ground-truth responses \(Y\). Train until convergence to align output behavior with the targets.

## 5 Experimental Results

We study two settings: single-token outputs and multi-token outputs. We describe their differences where relevant.

**Adaptation Tasks:** In the single-token setting, each output \(Y_i\) is one token. It is typically a class label, True/False, or a multiple-choice question (MCQ) option. In the multi-token setting, \(Y_i\) can contain multiple tokens and vary in length. This is the general setting for open-ended generation.

We compare adaptation methods across tasks and models. In many cases, we also evaluate on out-of-distribution (OOD) datasets, whose distribution differs from the training dataset. This tests behavior under distribution shift. An in-distribution (ID) evaluation uses the training task’s dataset distribution.

**Single-token tasks:**

- **Sentiment Classification:** We use SST2 (Socher et al., 2013), Financial Phrasebank, abbreviated FinS (Malo et al., 2014), and Poem Sentiment, abbreviated PoemS (Sheng and Uthus, 2020). We train models on SST2 or FinS and evaluate each on all three datasets. Each training source therefore has one ID and 2 OOD evaluation datasets.
- **True/False:** We use StrategyQA, abbreviated STF (Geva et al., 2021). It requires multi-hop reasoning, or multiple linked reasoning steps, before a True/False conclusion.
- **News Categorization:** We use AGNews, abbreviated AGN (Zhang et al., 2015), and BBCNews, abbreviated BBCN (Greene and Cunningham, 2006). We train on AGN and evaluate on both.
- **MCQ:** We use SciQ (Johannes Welbl, 2017) and QASC (Khot et al., 2019). To add complexity, we remap the choice labels \([A,B,C,\ldots]\) to an unrelated token set. We call the remapped datasets SciQr and QASCr. We train on SciQr and evaluate on both.

**Multi-token tasks:**

- **Grade-school Math:** We use GSM8K (Cobbe et al., 2021) and GSM Symbolic, abbreviated GSM8Ks (Mirzadeh et al., 2024). We train on GSM8K and evaluate on both.
- **Advanced Math:** We use MATH Algebra, abbreviated HMathA (Hendrycks et al., 2021). It contains higher-grade algebra problems.
- **Scientific QA:** Here, QA means question answering. We use SciQ again, but generate answer tokens rather than choose an option.

**Data Setup:** We focus on the data-scarce, few-shot setting in which ICL is typically used. A shot is a training or demonstration example. For each task, we create training datasets

\[
\mathcal D_{\mathcal T}=\{(X_i,Y_i)\}_{i=1}^{N}
\]

with \(N\) varying in powers of \(2\): \([2,4,8,16,\ldots]\). At each \(N\), we create 5 different sets to average out outlier effects.

For every training input \(X_i\), we collect its IA2 target \(A^i_{\mathrm{ICL}}\) using the other \(N-1\) samples as demonstrations, in random order. Thus, IA2 reuses the training samples rather than adding data. All adaptation methods use exactly the same data.

For each task, we sample a separate evaluation set of 500 samples. Section A gives further dataset and processing details.

**Multi-token details:** The model outputs reasoning tokens before the answer, following the format shown in the demonstrations. These are ordinary text, not text inside `<think>` tags. Both the demonstrations and expected outputs use chain-of-thought style, which presents intermediate reasoning before the answer (Wei et al., 2022).

We extract the answer from the generated response before evaluating it. Parsing differs by dataset:

- **GSM8K and GSM8Ks:** Ground-truth responses follow the pattern “`<reasoning steps> #### <numerical answer>`.” We do not evaluate the correctness or conciseness of the reasoning. We extract the numerical answer and require an exact match with the ground truth. To allow slight formatting differences, we also use common alternatives such as matching “`The answer is <value>`.” The exact parsing function is in our code.
- **HMathA:** Answers are expected inside a `\boxed{}` element and are parsed accordingly.
- **SciQ:** We construct responses in a format similar to GSM8K: “`<support text> #### <text answer>`.” We parse this format and count only an exact string match as correct.

**Training Setup:** We use two model families. Qwen3-4B-Base is used on every single-token and multi-token task.

For single-token tasks, we also use Llama-3.2-1B, except on SciQr, where we use Llama-3.2-3B. The 1B Llama model could not improve above the random baseline for any \(N\) up to 128. For multi-token tasks, the second model is Llama-3.2-1B-Instruct. This instruction-following model tests the method on a post-trained model, meaning one that has undergone further training after its initial language-model training.

We train with Low-Rank Adaptation (LoRA), which represents weight changes in a low-rank form. We use rank 8, the size of the low-rank update, and modify the \(W_Q\), \(W_K\), and \(W_O\) matrices in self-attention layers. We chose the models and weight combinations reasonably given available resources. In total, we trained more than 13,000 models.

Multi-token training is much more resource-intensive because answers are long. We therefore use \(N\) only up to 16 for Llama and up to 8 for Qwen in that setting. For efficiency, we cap generated training responses at \(G=200\) tokens. IA2 uses these responses to collect

\[
\hat A_i\in\mathbb R^{L\times200\times d}
\]

for every training input \(x_i\). Here, \(\hat A_i\) is the collected ICL activation tensor, and \(x_i\) denotes the training input also written \(X_i\). For SFT, \(G\) instead equals the number of tokens in the ground-truth answer.

Each method–dataset combination is trained until convergence, for up to 50 epochs. An epoch is one pass through the training data. We try 3 learning rates, which control update size: slow, 1e-4; medium, 3e-4; and fast, 1e-3. We choose the best-performing rate to neutralize the effect of hyperparameter selection on method performance. Hyperparameters are training settings selected outside the learned model weights. Section B gives more details.

**Evaluation:** For single-token tasks, we measure accuracy, the proportion of correct answers, and ECE. Both use the probability distribution of the first generated token.

We first evaluate the base model with ICL demonstrations to establish the ICL benchmark. For each task, we evaluate the validation set 5 times using different demonstrations and report average metrics. We then evaluate each training method without demonstrations and report its best-performing average metrics.

For multi-token tasks, we limit generated evaluation tokens to use computation judiciously. GSM8K and GSM8Ks have a limit of 200, based on their 95th-percentile answer lengths of 184 and 154, respectively. HMathA has a limit of 400 because its 95th-percentile answer length is 307.

For multi-token performance, we measure only accuracy: the number of questions answered correctly in the required format. See §A for answer parsing and other details. Confidence is difficult to measure for open-ended generation, so we do not measure calibration.

### 5.1 IA2 priming boosts SFT performance

**Single-Token results:** Table 1 reports adapted Qwen results for the practical single-token setting \(N=4\). Using the same data, IA2 → SFT outperforms SFT-only training in almost all cases on both accuracy and calibration (Table 1, paired SFT-only and IA2 → SFT columns).

In most cases, IA2 → SFT also exceeds ICL accuracy, with slightly worse calibration (Table 1, ICL and IA2 → SFT columns). IA2 alone reaches high accuracy in many cases while remaining reasonably calibrated like ICL. It does so without training on target response tokens. For example, the SciQr-source/SciQr-evaluation row reports IA2-only accuracy of 90.4 (01.5), compared with ICL accuracy of 87.7 (01.1). This highlights the richness of the IA2 signal.

Table 1 shows one setup. Figure 1 gives aggregate trends across models and datasets. Section C provides tables for other model/\(N\) combinations and a statistical significance test of our improvement.

In the tables, “acc” means accuracy and “ece” means expected calibration error. The arrow \(\uparrow\) means higher is better; \(\downarrow\) means lower is better. “Source” identifies the training dataset, and “Eval” identifies the evaluation dataset. An asterisk, ∗, marks OOD evaluation. “Only” means that no subsequent training step is used.

| Source | Eval | ICL acc ↑ | ICL ece ↓ | SFT only acc ↑ | SFT only ece ↓ | IA2 only acc ↑ | IA2 only ece ↓ | IA2 → SFT acc ↑ | IA2 → SFT ece ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AGN | AGN | 30.0 (02.3) | 0.13 (0.03) | 27.9 (04.3) | 0.52 (0.12) | 24.0 (00.3) | 0.11 (0.01) | 31.8 (03.8) | 0.35 (0.24) |
| AGN | BBCN∗ | 28.8 (00.6) | 0.12 (0.01) | 31.8 (07.1) | 0.56 (0.14) | 24.3 (01.3) | 0.10 (0.02) | 27.9 (04.3) | 0.40 (0.21) |
| FinS | FinS | 63.6 (01.5) | 0.12 (0.01) | 67.4 (14.1) | 0.31 (0.14) | 63.1 (15.0) | 0.24 (0.06) | 78.7 (13.7) | 0.16 (0.11) |
| FinS | PoemS∗ | 56.9 (02.2) | 0.12 (0.02) | 49.9 (04.6) | 0.48 (0.04) | 52.8 (05.6) | 0.15 (0.04) | 60.6 (15.7) | 0.36 (0.15) |
| FinS | SST2∗ | 70.1 (01.3) | 0.17 (0.01) | 59.4 (06.5) | 0.20 (0.10) | 69.8 (13.7) | 0.30 (0.10) | 71.4 (18.9) | 0.20 (0.15) |
| SciQr | QASCr∗ | 56.5 (01.4) | 0.08 (0.01) | 76.5 (03.8) | 0.09 (0.05) | 71.2 (03.0) | 0.12 (0.01) | 79.4 (01.8) | 0.09 (0.01) |
| SciQr | SciQr | 87.7 (01.1) | 0.07 (0.01) | 88.3 (04.6) | 0.07 (0.05) | 90.4 (01.5) | 0.09 (0.01) | 91.7 (01.7) | 0.05 (0.01) |
| SST2 | FinS∗ | 41.9 (00.3) | 0.19 (0.01) | 68.4 (02.3) | 0.30 (0.04) | 71.3 (02.1) | 0.21 (0.03) | 82.4 (12.2) | 0.11 (0.08) |
| SST2 | PoemS∗ | 65.1 (01.2) | 0.11 (0.02) | 56.5 (13.2) | 0.33 (0.14) | 62.4 (09.2) | 0.19 (0.06) | 68.4 (08.4) | 0.30 (0.08) |
| SST2 | SST2 | 85.4 (00.4) | 0.13 (0.00) | 65.2 (10.7) | 0.22 (0.10) | 82.7 (06.4) | 0.28 (0.03) | 90.4 (02.3) | 0.06 (0.01) |
| STF | STF | 66.0 (02.1) | 0.07 (0.01) | 52.4 (04.4) | 0.29 (0.10) | 62.2 (04.1) | 0.16 (0.04) | 62.4 (04.3) | 0.29 (0.08) |

**Table 1:** Performance for \(N=4\) on Qwen3-4B-Base, showing accuracy (acc) and expected calibration error (ece). Parentheses contain standard deviations across 5 runs for the best-performing learning rate. Standard deviation measures variation across runs. The best training method is shown in bold. An asterisk (∗) marks OOD evaluations. Our proposed IA2 → SFT method outperforms standard SFT across the board.

In Table 2 and later tables, “w/o ICL” means the model is evaluated without ICL demonstrations.

| Source | Eval | w/o ICL acc ↑ | ICL acc ↑ | SFT only acc ↑ | IA2 only acc ↑ | IA2 → SFT acc ↑ |
|---|---|---:|---:|---:|---:|---:|
| GSM8K | GSM8K | 56.4 (00.0) | 76.4 (01.1) | 70.9 (02.8) | 77.4 (02.8) | 73.6 (03.7) |
| GSM8K | GSM8Ks∗ | 45.4 (00.0) | 68.4 (01.1) | 64.5 (04.0) | 66.2 (03.7) | 68.8 (05.1) |
| HMathA | HMathA | 21.0 (00.0) | 60.4 (02.1) | 50.4 (01.8) | 47.8 (06.0) | 55.3 (02.1) |
| SciQ | SciQ | 15.4 (00.0) | 37.5 (01.0) | 35.0 (07.1) | 06.9 (13.8) | 40.8 (07.6) |

**Table 2:** Performance for \(N=4\) on Qwen3-4B-Base models and multi-token datasets. The best training method is shown in bold. IA2 → SFT outperforms standard SFT on all datasets.

**Multi-Token results:** Table 2 reports adapted Qwen accuracy for multi-token tasks. IA2 → SFT exceeds SFT-only accuracy in every row, as in the single-token setting.

Compared with the single-token setting, IA2 → SFT sometimes performs worse than ICL. Table 2 shows this on GSM8K and HMathA. We suspect imperfect compression of longer contexts into the same small LoRA weight space. Larger ranks should improve this. Section C contains additional results for other model/\(N\) combinations.

### 5.2 Why is IA2 priming important?

**Figure 3:** We plot \(\operatorname{asim}(A,A_{\mathrm{ICL}})\) against accuracy and ECE for all training methods. Each point represents one training experiment. ECE decreases smoothly as activation similarity increases. However, extreme alignment with ICL activations may leave accuracy gains that can be obtained through SFT on ground-truth responses.

We investigate three training methods: 1) SFT only, 2) IA2 only, and 3) IA2 → SFT.

**Activation Similarity vs Performance:** Figure 2 shows all three methods’ activation similarity with ICL. As expected, IA2 drastically increases similarity. IA2 → SFT retains higher similarity with ICL even after its SFT stage.

Figure 3 relates similarity to performance. IA2 → SFT lies between IA2 only and SFT only in activation similarity, while achieving better accuracy and calibration. SFT-only training and the SFT stage of IA2 → SFT differ only in their starting state. This implies that IA2 provides a rich training signal unavailable through SFT-only training.

Greater activation similarity clearly implies better calibration in Figure 3. However, extreme similarity may not maximize accuracy because the ICL signal is not always correct. Section C shows a case where IA2 models follow ICL performance even when that performance worsens through overfitting.

It is therefore best to combine the signals: IA2 aligns internal functional behavior with ICL, and SFT on ground-truth responses aligns outputs with expected behavior.

**Subspace Overlap:** Activations record functional behavior, but they do not isolate the changes caused by training. We use LoRA adapters, the trained parameter additions to the base model. Unchanged base-model weights still strongly influence the activations. To isolate the changes, we analyze the LoRA weights directly.

**Figure 4:** IA2 → SFT models have significantly high subspace overlap with IA2-only models. SFT-only models align with neither. A subspace is the set of weight directions spanned by the learned updates. This result indicates that IA2 provides a training signal absent from SFT-only training and important for performance.

Figure 4 shows distributions of weight-subspace overlap between pairs of models trained by different methods. We include models from all single-token and multi-token experiments.

First, we use singular value decomposition (SVD), a matrix decomposition, to find basis vectors for the weight matrices. Basis vectors describe the directions spanned by a matrix. We then calculate:

\[
\frac{1}{r}*
\left\|U_1^T\cdot U_2\right\|_F^2.
\]

Here, \(U_1\) and \(U_2\) contain basis vectors for the two methods, \(r\) is the LoRA weight rank, and \(*\) denotes multiplication. The squared Frobenius norm, \(\|\cdot\|_F^2\), is the sum of the squared matrix entries.

Subspace overlap measures how much weight space two methods share. It therefore isolates and compares changes in functional behavior.

SFT-only updates are almost completely orthogonal to those of the other methods: their weight directions have little overlap (Figure 4). The most performant IA2 → SFT models share around \(39\%\) of their spanned weight space, on average, with IA2-only models (Figure 4). This implies that much of the IA2 → SFT performance gain comes from IA2. The subspaces found by IA2 are practically unreachable through SFT-only training.

Overlap between IA2-only and IA2 → SFT models has a wide spread. In many cases, IA2 alone already aligned the outputs. The subsequent SFT stage then changed little before convergence, producing very high overlap.

## 6 Discussion

**IA2 + SFT:** A natural alternative to sequential IA2 → SFT is simultaneous **IA2 + SFT**. This uses both \(\mathcal L_{\mathrm{IA2}}\) and \(\mathcal L_{\mathrm{SFT}}\) at the same time.

This creates a practical difficulty. When \(G>1\), the IA2 target activation tensor \(\hat A\) is collected using the model’s own generated ICL response. That response can differ from the ground truth, even in length. The two objectives are therefore incompatible for joint training on their respective responses.

Instead, we can self-distill using ICL responses, as in Snell et al. (2022), while adding the IA2 signal. We train with the unified loss

\[
\mathcal L_{\mathrm{IA2}}+\beta\cdot\mathcal L_{\mathrm{SFT}},
\]

where \(\beta\) controls the weight of the SFT loss. We try 4 values of \(\beta\) that smoothly span the range between IA2-only and SFT-only training. See §B for details.

Table 3 reports multi-token results after selecting the best learning-rate/\(\beta\) combination across 5 random seeds. A random seed controls the repeatable random choices in a run.

SFT-only and IA2 → SFT performance drops when trained on ICL responses rather than ground truth; compare their columns in Tables 3 and 2. However, IA2 + SFT extracts significantly more performance from ICL responses than SFT alone (Table 3, SFT-only and IA2 + SFT columns). This highlights the richness of the IA2 signal. Section C gives additional IA2 + SFT results.

| Source | Eval | w/o ICL acc ↑ | ICL acc ↑ | SFT only acc ↑ | IA2 only acc ↑ | IA2 → SFT acc ↑ | IA2 + SFT acc ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| GSM8K | GSM8K | 56.4 (00.0) | 76.4 (01.1) | 66.4 (18.1) | 77.4 (02.8) | 67.2 (22.1) | 77.0 (07.8) |
| GSM8K | GSM8Ks∗ | 45.4 (00.0) | 68.4 (01.1) | 59.8 (20.2) | 66.2 (03.7) | 61.6 (24.0) | 69.0 (09.6) |
| HMathA | HMathA | 21.0 (00.0) | 60.4 (02.1) | 49.0 (03.0) | 47.8 (06.0) | 54.0 (01.7) | 53.6 (02.7) |
| SciQ | SciQ | 15.4 (00.0) | 37.5 (01.0) | 27.2 (12.0) | 06.9 (13.8) | 32.8 (10.0) | 34.5 (10.6) |

**Table 3:** Performance for \(N=4\) on Qwen3-4B-Base models trained with ICL responses rather than ground-truth responses. IA2 + SFT performs significantly better than SFT only, highlighting the significance of IA2.

**Knowledge Distillation baseline:** Knowledge distillation (KD) transfers behavior from a teacher model to a student model. A popular approach matches soft labels: the teacher’s probability distribution over tokens rather than only a selected token (Hinton et al., 2015). These probabilities provide a denser training signal.

Here, the teacher is the same model with added ICL context. To compare IA2 with soft-label matching, we train and evaluate Llama models on GSM8K using 1B-Instruct and on SST2 using 1B. Soft-label matching uses ICL responses.

In the multi-token case, soft-label matching significantly improves on standard SFT and nearly reaches IA2 + SFT: the GSM8K row of Table 4 reports 36.1 (03.5), compared with 28.2 (04.5) for SFT only and 37.0 (01.9) for IA2 + SFT. It falls short in the single-token case (Table 4, SST2 row). This highlights the consistency of IA2-based training in reaching high performance. A dash indicates that calibration is not measured.

| Dataset | \(N\) | ICL acc ↑ | ICL ece ↓ | SFT only acc ↑ | SFT only ece ↓ | SFT (soft labels) acc ↑ | SFT (soft labels) ece ↓ | IA2 only acc ↑ | IA2 only ece ↓ | IA2 → SFT acc ↑ | IA2 → SFT ece ↓ | IA2 + SFT acc ↑ | IA2 + SFT ece ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 4 | 34.6 (00.9) | – | 28.2 (04.5) | – | 36.1 (03.5) | – | 29.3 (05.0) | – | 30.3 (03.2) | – | 37.0 (01.9) | – |
| SST2 | 8 | 77.0 (02.2) | 0.10 (0.02) | 52.3 (04.3) | 0.38 (0.14) | 55.9 (08.8) | 0.16 (0.06) | 64.9 (15.3) | 0.30 (0.05) | 64.7 (19.3) | 0.33 (0.19) | 66.9 (18.4) | 0.28 (0.21) |

**Table 4:** Comparison with the knowledge distillation (KD) baseline, SFT on soft labels. Although KD performs better than standard SFT in the multi-token case, IA2-based training consistently provides high performance in single-token and multi-token cases.

**Why IA2 based training does not consistently beat ICL?** In some multi-token cases, ICL exceeds every training method, including IA2-based training. Examples include Qwen on the GSM8K and HMathA math datasets (Table 5).

We suspect this comes from Qwen’s mid-training on science, technology, engineering, and mathematics (STEM) data. This may make ICL extremely sample-efficient. Surprisingly, GSM8K ICL performance at \(N=2\) exceeds performance at \(N=4,8\) (Table 5, GSM8K rows).

Table 5 shows performance as the shot count increases. The gap between ICL and IA2 → SFT closes quickly with more shots. Our key takeaway remains consistent: IA2 → SFT exceeds SFT-only training, as shown in the last column. “wrt” means “with respect to.” The gap columns compare IA2 → SFT with the named method.

| Dataset | \(N\) | ICL | SFT only | IA2 only | IA2 → SFT | Gap wrt ICL | Gap wrt SFT only |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 2 | 81.2 (00.4) | 65.2 (01.0) | 70.7 (01.8) | 74.6 (02.4) | -06.6 | +09.4 |
| GSM8K | 4 | 76.4 (01.1) | 70.9 (02.8) | 77.4 (02.8) | 73.6 (03.7) | -02.8 | +02.7 |
| GSM8K | 8 | 77.2 (00.9) | 73.5 (01.6) | 78.8 (02.5) | 76.0 (02.0) | -01.2 | +02.5 |
| HMathA | 2 | 58.6 (02.0) | 47.8 (04.0) | 33.9 (06.4) | 48.7 (07.0) | -09.9 | +00.9 |
| HMathA | 4 | 60.4 (02.1) | 50.4 (01.8) | 47.8 (06.0) | 55.3 (02.1) | -05.1 | +04.9 |
| HMathA | 8 | 62.1 (01.0) | 52.0 (00.8) | 57.3 (03.7) | 59.4 (00.8) | -02.7 | +07.4 |

**Table 5:** ICL is more sample-efficient than training methods for Qwen models on math datasets. IA2 → SFT quickly closes the performance gap as shots increase and consistently exceeds SFT-only training.

**Other fine-tuning methods:** LoRA is not the only fine-tuning method. Alternatives include full fine-tuning, with full-rank weight updates, and other PEFT methods. These include prompt/prefix tuning, which trains prompt or prefix representations (Lester et al., 2021; Li and Liang, 2021), and activation scaling, which learns factors that scale internal activations, as in (IA)3 (Liu et al., 2022).

Our pipeline does not compete with these methods. It improves them by providing better training signals. To establish this, we repeat experiments on a few datasets with the popular PEFT method (IA)3. We use exactly the §5 methodology, replacing LoRA with (IA)3. Section B gives setup details.

Table 6 reports one setting and shows trends similar to LoRA. Section C gives other results. We leave exploration of IA2 improvements for other SFT methods to future work. As a side note, IA2 also marks a major improvement in method naming compared with (IA)3.

| Source | Eval | ICL acc ↑ | ICL ece ↓ | SFT only acc ↑ | SFT only ece ↓ | IA2 only acc ↑ | IA2 only ece ↓ | IA2 → SFT acc ↑ | IA2 → SFT ece ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SciQr | QASCr∗ | 35.6 (01.4) | 0.06 (0.01) | 32.8 (03.4) | 0.20 (0.08) | 34.2 (02.7) | 0.08 (0.01) | 48.6 (05.5) | 0.25 (0.11) |
| SciQr | SciQr | 60.0 (01.8) | 0.07 (0.01) | 64.5 (05.0) | 0.12 (0.05) | 63.0 (04.6) | 0.10 (0.04) | 70.6 (04.6) | 0.08 (0.02) |
| SST2 | FinS∗ | 56.6 (01.6) | 0.17 (0.01) | 58.2 (10.7) | 0.25 (0.03) | 61.9 (09.5) | 0.18 (0.06) | 59.8 (07.8) | 0.18 (0.03) |
| SST2 | PoemS∗ | 52.3 (03.0) | 0.19 (0.02) | 48.1 (03.2) | 0.44 (0.11) | 48.1 (03.4) | 0.11 (0.04) | 59.8 (13.4) | 0.33 (0.13) |
| SST2 | SST2 | 60.3 (02.8) | 0.19 (0.02) | 52.4 (02.8) | 0.35 (0.05) | 54.2 (04.2) | 0.15 (0.06) | 62.6 (09.1) | 0.28 (0.11) |

**Table 6:** Performance for \(N=4\) on Llama-3.2 models trained with (IA)3. IA2 → SFT achieves significant gains over SFT-only training and exceeds even ICL accuracy in every case.

**Computational Overhead:** IA2 → SFT requires activation collection before training. We run ICL inference on the model with the given dataset to collect these targets. This is a one-time cost. The collected activations greatly benefit inference capabilities, while the resulting model’s inference cost is the same as that of an SFT-only model.

**Potential improvements:** Resource constraints fixed several design choices that could be explored and improved. The most important are LoRA ablations and selective IA2. Ablations would vary settings such as rank and target modules, the weight matrices being updated.

We showed that extreme alignment can hurt performance (Figure 3; §C). Prior work also shows that some layers are more sensitive to ICL than others (Todd et al., 2023). Selectively aligning layers could therefore be more effective.

Prompt optimization before target-activation collection is another possible improvement. ICL performance is sensitive to demonstration order (Lu et al., 2022; Zhao et al., 2021).

Finally, our IA2 → SFT pipeline uses fixed learning rates. The two signals could benefit from finer-grained control. We leave these explorations to future work.

## 7 Related Work

**ICL vs SFT:** Prior work compares ICL and SFT performance, but the verdict is unclear (Mosbach et al., 2023; Duan et al., 2024). We found SFT to perform worse than ICL (Figure 1; Tables 1–2). Other work studies their functional similarity (Dai et al., 2022) and differences (Wang et al., 2023).

Some studies examine miscalibration after ICL or SFT and methods to improve it (Zhang et al., 2024; Li et al., 2025; Li et al., 2024; Xiao et al., 2025). Sia et al. (2024) identify “where” translation happens during ICL by masking parts of the context. Like us, they use activations as a signal of functional behavior.

Recent work shows that SFT is prone to memorization, overfitting, and catastrophic forgetting—the loss of previously learned behavior during further training (Chu et al., 2025; Shenfeld et al., 2025). Our improved pipeline aims to address these problems.

**Context Distillation:** Many methods, like ours, improve adaptation by compressing explicit context signals. These include:

- Using intermediate contemplation tokens for compression (Cheng and Van Durme, 2024; Jiang et al., 2025).
- Internalizing context by generating adapters on the fly (Chen et al., 2025; Charakorn et al., 2025).
- Using related text to improve distillation (Zhu et al., 2025; Choi et al., 2025).
- Compressing context directly into weights (Shen et al., 2025; Deng et al., 2024; Yu et al., 2024; Snell et al., 2022; Chen et al., 2024b; Shin et al., 2024).

These methods distill a context’s effect on outputs through token-response signals. Our method is complementary. It improves distillation through the model’s own ICL processing signals. Prior work supports this idea (Aguilar et al., 2020; Jin et al., 2024; Yang et al., 2024).

**Activation Steering:** A growing line of work extracts “steering” vectors or weights and applies them at targeted internal locations to produce an intended output effect (Postmus and Abreu, 2024; Stolfo et al., 2025; Caccia et al., 2025; Fleshman and Van Durme, 2024; Arditi et al., 2024).

Like our method, these approaches intervene in activation space to affect model function. They use targeted interventions in small application domains. In contrast, our method aims to induce a large behavioral change that avoids potential brittleness and reliability issues (Queiroz Da Silva et al., 2025).

## 8 Conclusion and Future Work

We used activations to study LLMs’ functional behavior. We found a key distinction: ICL and SFT adapt through different mechanisms (§3; Figure 2). ICL often encodes richer, more generalizable patterns.

This motivated IA2, a priming technique that enables ICL-like internal behavior before SFT. It shifts the model into a more adaptable weight subspace that SFT alone cannot access (§5.2; Figure 4). Extensive experiments show that IA2 consistently improves SFT accuracy and calibration (§5; Figure 1).

Future work will test effects on output diversity and catastrophic forgetting. This will help measure how IA2 can support LLM post-training. We also aim to study IA2 on post-trained models tuned through reinforcement learning (RL), which trains behavior using a reward signal.

## Reproducibility Statement

We provide code for data download and processing, training, evaluation, analysis, plotting, and related steps on GitHub. The details in §5 and the appendix, together with a detailed README guide in the code repository, should be sufficient to reproduce the results fully. We used repeatable random seeds.

## Acknowledgments

This work is supported by the Office of Naval Research (ONR; N00014-24-1-2089), an Amazon Research Award, and a grant from Open Philanthropy. We thank Andrea Wynn for their constructive feedback on earlier versions of this document.
