# IA2: Alignment with ICL Activations

Improves Supervised Fine-Tuning

Aayush Mishra

  
Daniel Khashabi & Anqi LiuDepartment of Computer Science,
Johns Hopkins University

###### Abstract

Supervised Fine-Tuning (SFT) specializes model behavior by training model weights to produce intended responses to queries. In contrast, In-Context Learning (ICL) adapts a model during inference by placing instructions or demonstrations in its prompt. In data-scarce settings, ICL can generalize better and produce better-calibrated responses than SFT. Calibration describes how well a model’s confidence matches its correctness. These benefits come at the cost of more inference computation.

We ask whether ICL’s internal computations can improve SFT. We first show that ICL and SFT produce distinct activation patterns (§3, Figure 2). Activations are the hidden outputs of self-attention operations, which combine information from tokens in the context. Their differences indicate that the two methods adapt through different functional mechanisms.

We introduce ICL Activation Alignment (IA2), a self-distillation technique: the model learns from its own behavior under ICL. IA2 trains a model to reproduce ICL activation patterns without the demonstrations. It therefore encourages ICL-like internal reasoning. Applying IA2 as a priming step before SFT significantly improves output accuracy and calibration. Our empirical study covers 12 popular benchmarks and two model families (§5; Tables 1–2). This finding is practically useful and offers a conceptual window into the internal mechanisms of model adaptation.

|

00footnotetext: ♡Equal advising. Correspondence to: {amishr24,danielk,aliu.cs}@jhu.edu

## 1 Introduction

Large language models (LLMs) are general-purpose models, but they are often used in specialized applications. For example, a news aggregator may need to classify articles into predefined categories.

A popular adaptation method is Supervised Fine-Tuning (SFT). It trains an LLM on labeled samples for a narrow downstream task. Parameter-efficient fine-tuning (PEFT) methods adapt a model using a small set of trainable parameters (Hu et al., 2021; Liu et al., 2022). These parameter sets can be loaded into and removed from graphics processing unit (GPU) memory efficiently. This makes SFT models extremely useful. However, SFT typically needs many labeled samples to generalize well to new tasks (Le Scao and Rush, 2021). Collecting those samples can be expensive.

In-Context Learning (ICL) instead adapts and steers an LLM during inference (Brown et al., 2020). The model receives a query preceded by demonstrations or instructions. These help it “learn” the demonstrated task and answer accordingly, without updating its weights. Prior work shows that ICL generalizes well with few examples and typically produces well-calibrated responses (Duan et al., 2024). Our experiments concur (§5). But ICL has a cost. Demonstrations and instructions occupy context space and increase the cost of each query. That space could otherwise hold more query and response tokens, the units of text processed by the model.

These differences motivate us to compare the functional behavior of ICL and SFT. Their output-token behavior may look similar, but their internal activations differ (§3, Figure 2). Activations record internal processing before the model produces the next token. Divergent patterns therefore highlight different mechanisms of adaptation, an interpretation supported by Shen et al. (2024).

We hypothesize that ICL activations contain rich information about how to extract generalizable patterns from context. This information may be absent from SFT models, especially with few training examples. In that setting, SFT models are prone to shortcut learning or overfitting: learning responses that do not generalize to new data. We therefore ask:

> Can the information rich ICL activations be used to improve the quality of SFT?

Recent work has tried to distill context into LLM weights (Snell et al., 2022; Chen et al., 2024b). These methods use response text as their training signal. They may therefore suffer from the same SFT problems. Training a model to reproduce an ICL-conditioned model’s outputs does not ensure that it functions like that model.

We propose ICL Activation Alignment (IA2), a self-distillation method that aligns a model with its own functional behavior under ICL (§4). Figure 1 outlines the pipeline:

1. Collect information-rich ICL activations.
2. Train the model to match them, enforcing functional alignment with ICL.
3. Perform SFT on this primed model.

Using the same training data, IA2 priming drastically improves adaptation on text-classification and generation tasks (§5; Tables 1–2). We also show that IA2 supplies an important training signal unavailable through SFT-only training (§5.2, Figures 3–4). We trained over 13,000 models spanning 12 benchmarks to validate these findings. The results show practical benefits and offer a conceptual window into model adaptation.

In summary:

- $\star$ ICL and SFT do not align in the model’s activation space, even when they use the same data. This highlights a gap in their functional behavior (§3, Figure 2).
- $\star$ IA2—ICL Activation Alignment—enforces alignment with ICL’s functional behavior. Using it before SFT drastically improves SFT performance (Figure 1, §4).
- $\star$ The IA2 training signal is not present in SFT-only training. This signal is important for adaptation quality (§5, especially §5.2).

**Figure 1:** An overview of our improved SFT pipeline. Standard SFT enforces only output-space alignment: agreement between the model’s response and a target response, corresponding to step 3 above. In low-data settings, this results in subpar performance and miscalibration. IA2 instead enforces functional alignment with ICL by matching the rich activation patterns produced during ICL. IA2 priming before SFT improves adaptation quality. The performance-comparison charts aggregate results across models and datasets. See section 5 for details.

## 2 Background and Notation

**Transformer Language Models:** A language model (LM) predicts the next token in a sequence. We write a standard decoder-only transformer LM as $M_{\Theta}$, where $\Theta$ is its set of model parameters, or weights. It consists of stacked self-attention (SA) and linear layers. Self-attention combines information from tokens in the sequence. Linear layers apply learned projections to token vectors.

Let $T=[t_i]_{i=1}^{R}$ be an ordered sequence of $R$ token vectors. Here, $i$ indexes token positions, $t_i$ is the vector at position $i$, and $d$ is the vector dimension. The model applies self-attention and linear projections to each token at each layer. At the final layer, it passes the last token projection through the LM head, the output component that predicts the most likely next token.

Self-attention is affected by other tokens in the sequence. A standard SA operation uses 4 weight matrices: $W_Q$, $W_K$, $W_V$, and $W_O$, all belonging to $\Theta$. These form query, key, value, and output projections. We abbreviate this set as $W_{QKVO}$.

At token position $i$, define the query vector $q_i=t_i\cdot W_Q$, key matrix $K=t_{:i}\cdot W_K$, and value matrix $V=t_{:i}\cdot W_V$. The notation $t_{:i}$ means the first $i$ tokens, and $\cdot$ denotes multiplication. A superscript $T$ on a matrix denotes its transpose, not the token sequence $T$. The function $\sigma$ is Softmax, which converts scores into probability weights. The factor $\sqrt d$ is the square root of the vector dimension. With $z_i$ denoting the output at position $i$, the full output sequence $Z$ is:

\[
\operatorname{SA}(T;W_{QKVO})
=Z=
\left[
z_i=\sigma\left(\frac{q_iK^{T}}{\sqrt d}\right)V\cdot W_O
\right]_{i=1}^{R}.
\tag{1}
\]

$Z$ has the same shape as $T$: $R\times d$, or $R$ vectors of dimension $d$. We study how model behavior changes when we change either the context $T$ or the weights $W_{QKVO}$.

**Supervised Fine-Tuning:** Let $\mathcal T$ denote a task. Its training dataset is
$\mathcal D_{\mathcal T}=\{(X_i,Y_i)\}_{i=1}^{N}$, a set of $N$ input–output pairs. $X_i$ is the input and $Y_i$ is the ground-truth output for sample $i$.

SFT updates the weights from $\Theta$ to $\Theta'$ to produce relevant responses to new inputs. For a test input $X_t$, we write
$M_{\Theta'}(X_t)=\bar y_t$.
The subscript $t$ identifies the test sample. Here, $\bar y_t$ is the first token of the full generated response $\bar Y_t$. Repeatedly applying $M_{\Theta'}$ to the growing sequence extends that response. A bar marks responses generated by the fine-tuned model.

If $Y_i$ contains $G$ tokens, we generate $G$ tokens to obtain $\bar Y_i$. We then minimize cross-entropy, a loss that penalizes low predicted probability for the ground-truth token. In the following equation, $j$ indexes response-token positions; $\bar Y_{ij}$ and $Y_{ij}$ are the model prediction and ground-truth token at position $j$. The summation signs add the loss over all samples and output tokens:

\[
\mathcal L_{\text{SFT}}
=
\sum_{i=1}^{N}\sum_{j=1}^{G}
\operatorname{cross\text{-}entropy}(\bar Y_{ij},Y_{ij}).
\tag{2}
\]

**In-Context Learning:** The same dataset can adapt $M_\Theta$ without weight updates. Let $n$ be the number of demonstrations placed in the prompt. Define their concatenated token sequence as
$I=[X_1\circ Y_1\circ\ldots\circ X_n\circ Y_n]$,
where $\circ$ means concatenation, or joining sequences in order.

Given a new test sample $X_t$, the model processes the demonstrations and produces
$M_\Theta(I\circ X_t)=\hat y_t$.
A hat marks an ICL-generated response: $\hat y_t$ is the response token, and $\hat Y_t$ denotes the full response. Remarkably, $\hat Y_t$ is often similar to the expected response $Y_t$. ICL therefore provides useful inference-time adaptation.

ICL was first illustrated in GPT-3 (Brown et al., 2020) and is widely used to adapt generic LMs during inference. Today’s frontier models can follow instructions directly, so a generic instruction can be considered a form of ICL demonstration. We focus on classic ICL with input–output demonstration pairs.

**Activations:** Because SA uses the tokens in its context, different token sequences produce different hidden SA outputs. We call these outputs activations. They provide a footprint of internal processing and can help us study how behavior changes with context.

If $M_\Theta$ has $L$ layers and processes $R$ tokens, its activation tensor $A$ has size $L\times R\times d$: one $d$-dimensional vector per layer and token position. We study $T=[I\circ X]$ for ICL and $T=X$ without ICL, under different model weights. Here, $I$ contains demonstration tokens and $X$ contains query tokens.

We define activation similarity, $\operatorname{asim}(A_1,A_2)$, as token-wise cosine similarity between two activation tensors $A_1$ and $A_2$. Cosine similarity compares the directions of their activation vectors. It produces one value per layer and token position, so $\operatorname{asim}$ has size $L\times R$.

## 3 Are ICL and SFT practically the same?

**Figure 2: Layerwise Similarity with ICL Activations.** SFT models have small activation similarity, or $\operatorname{asim}$, with ICL. This indicates different functional behavior during adaptation. Our recommended pipeline, IA2 → SFT, means IA2 followed by SFT. It aligns much more closely with ICL than SFT alone and performs much better as a result (Figure 1).

Recent studies claim that ICL in transformers operates through an internal gradient-descent mechanism (Von Oswald et al., 2023; Akyürek et al., 2022). Gradient descent updates parameters to reduce a training loss. These studies show that self-attention can produce outputs as if the parameters had been updated using the demonstrations in context. This implies a functional equivalence between ICL and parameter updates.

Under this equivalence, we should expect similar activations at output-token positions in two cases: the base model with ICL demonstrations, and an SFT model without demonstrations. We investigated whether LLMs actually exhibit this pattern.

**ICL and SFT produce different activations:** We collected ICL activations, $A_{\text{ICL}}$, using $T=[I\circ X]$ for 100 test samples from multiple datasets. These included SST2, a sentiment-classification dataset, and AGNews, a news-categorization dataset. We used the Qwen3 (Yang et al., 2025) and Llama-3.2 (Grattafiori et al., 2024) model families. The model sizes and layer counts were 1B ↔ 16 layers, 3B ↔ 28 layers, and 4B ↔ 36 layers. Here, B means billion parameters, and ↔ pairs a size with its layer count.

We trained SFT models on the same demonstrations until convergence, meaning training continued until it settled. We then collected their activations, $A_{\text{SFT}}$, using $T=X$. See §B for experiment details.

We computed $\operatorname{asim}(A_{\text{ICL}},A_{\text{SFT}})$ at output-token positions. Figure 2, first row, “SFT only,” plots the averages across tokens, samples, and datasets. ICL and SFT activations are not aligned across the models. They align better near the initial and final layers, where we expect tokens to be processed individually. They are less aligned in the middle layers, where we expect the whole demonstration set to be processed abstractly.

**Why should we care about alignment with ICL?** The functional difference also appears in expected calibration error (ECE; Guo et al., 2017), a measure of mismatch between confidence and correctness. Lower ECE means better calibration. Figure 1 shows that ICL is much better calibrated than SFT at similar, slightly better accuracy levels. For example, Table 1’s AGN → AGN row reports ICL accuracy of 30.0 (02.3) and ECE of 0.13 (0.03), compared with SFT-only accuracy of 27.9 (04.3) and ECE of 0.52 (0.12). AGN is the table code for AGNews.

We suspect that SFT’s output-oriented training signal permits shortcuts that can fail on new data. In contrast, ICL relies on complex circuits—internal computational mechanisms—that must work across many tasks (Elhage et al., 2021). It extracts generalizable patterns from demonstrations and performs in a well-calibrated manner.

This experiment demonstrates that ICL and SFT achieve adaptation through different means. SFT should therefore not be expected to replace ICL directly. These differences motivate our research question:

> Can the information rich ICL activations be used to improve the quality of SFT?

Next, we show that they can and describe our proposed SFT procedure.

## 4 Inducing ICL-like behavior in SFT Models

Our SFT pipeline has two stages. First, IA2 primes the model by encouraging ICL-like functional behavior. Further SFT on these ICL-aligned weights then produces much better adaptation.

**IA2—ICL Activation Alignment:** Section 3 found little activation similarity between SFT and ICL. To increase that similarity, and thus functional similarity with ICL, we set the following goal for every layer of $M_\Theta$.

Find modified attention weights
$\tilde W_{QKVO}=(\tilde W_Q,\tilde W_K,\tilde W_V,\tilde W_O)$.
The tilde marks modified weights. For every newly generated token, at index $-1$—the last position—we want:

\[
\operatorname{SA}([I\circ X];W_{QKVO})
\approx
\operatorname{SA}(X;\tilde W_{QKVO})
\quad \forall X\in\mathcal T.
\tag{3}
\]

Here, $\approx$ means approximately equal, and $\forall X\in\mathcal T$ means for every query $X$ in task $\mathcal T$. The left side uses the original weights with demonstrations. The right side uses modified weights without demonstrations.

Chen et al. (2024a) give a closed-form solution for linearized attention models, using a kernel approximation to Softmax. We seek a practical, general solution for real nonlinear transformers. Our goal is to put ICL’s qualities directly into the SFT model’s weights. These modified weights are not encouraged to produce similar outputs. Instead, they are encouraged to process queries as ICL does at each layer of the base model.

For each training input $X_i$, we first generate an ICL response $\hat Y_i$ using $T_i=[I\circ X_i]$. We construct $I$ from the remaining samples in the dataset. At every output position, we collect target activations
$A^i_{\text{ICL}}\in\mathbb R^{L\times G\times d}$,
assuming $G$ response tokens. The notation $\mathbb R^{L\times G\times d}$ denotes a real-valued tensor with those dimensions.

Next, we attach the same generated response to the query without demonstrations:
$T_i=[X_i\circ\hat Y_i]$.
This treats the ICL response as though the model had produced it with only the query in context. It gives an unaligned activation tensor $A^i$ at the corresponding output positions.

We train the model parameters to minimize the mean squared error between the two activation tensors:

\[
\mathcal L_{\textbf{IA2}}
=
\sum_{i=1}^{N}
\|A^i-A^i_{\text{ICL}}\|.
\tag{4}
\]

The double bars denote a norm, or magnitude, of the activation difference.

**IA2 → SFT:** After IA2, we continue training with the standard SFT cross-entropy loss in Equation 2, using target tokens. Suppose IA2 changes the parameters from $\Theta$ to $\Theta'$. We collect responses $\bar Y_i$ for all inputs $X_i$ using $M_{\Theta'}$. We then minimize SFT loss between these responses and the ground-truth responses $Y_i$. This further aligns the outputs with the intended targets.

The full pipeline is:

1. Collect target ICL activations $A_{\text{ICL}}$. Train with $\mathcal L_{\textbf{IA2}}$ until convergence to align functional behavior with ICL.
2. Switch to $\mathcal L_{\text{SFT}}$ using ground-truth responses $Y$. Train until convergence to align output behavior with the targets.

## 5 Experimental Results

We study single-token and multi-token adaptation. These settings have different characteristics, which we note when presenting the results.

**Adaptation Tasks:** In the single-token setting, each output $Y_i$ is one token. It is typically a class label, True/False value, or multiple-choice question (MCQ) choice. In the multi-token setting, $Y_i$ can contain several tokens and vary in length. This is the general setting for open-ended generation.

We compare adaptation methods across tasks and models. In many cases, we also evaluate on out-of-distribution (OOD) datasets: datasets that differ from the training source. In-distribution (ID) evaluation uses the training-source distribution.

**Single-token tasks:**

- **Sentiment Classification:** We use SST2 (Socher et al., 2013), Financial Phrasebank [FinS] (Malo et al., 2014), and Poem Sentiment [PoemS] (Sheng and Uthus, 2020). We train on SST2 and FinS separately. For each training source, we evaluate on all three datasets, giving one ID and 2 OOD evaluation datasets.
- **True/False:** We use StrategyQA [STF] (Geva et al., 2021). It requires multi-hop reasoning—several connected reasoning steps—before reaching a True/False conclusion.
- **News Categorization:** We use AGNews [AGN] (Zhang et al., 2015) and BBCNews [BBCN] (Greene and Cunningham, 2006). We train on AGN and evaluate on both.
- **MCQ:** We use SciQ (Johannes Welbl, 2017) and QASC (Khot et al., 2019). To add complexity, we replace the choice labels [A, B, C …] with an unrelated set of tokens. We call these remapped datasets SciQr and QASCr. We train on SciQr and evaluate on both.

**Multi-token tasks:**

- **Grade-school Math:** We use GSM8K (Cobbe et al., 2021) and GSM Symbolic [GSM8Ks] (Mirzadeh et al., 2024). We train on GSM8K and evaluate on both.
- **Advanced Math:** We use MATH Algebra [HMathA] (Hendrycks et al., 2021), which contains higher-grade algebra problems.
- **Scientific QA:** For scientific question answering (QA), we use SciQ again. Here, the model generates answer tokens rather than choosing an option.

**Data Setup:** We focus on the data-scarce, few-shot setting where ICL is typically used. For each task, we create training datasets
$\mathcal D_{\mathcal T}=\{(X_i,Y_i)\}_{i=1}^{N}$
with sizes increasing by powers of $2$: $[2,4,8,16,\ldots]$. We create 5 different training sets at each $N$ to average out outlier effects.

For every training query $X_i$, we collect its IA2 target tensor $A^i_{\text{ICL}}$ using the remaining $N-1$ training samples as demonstrations, in random order. Thus, all adaptation methods use exactly the same data. IA2 reuses the training samples rather than adding examples, keeping the comparison fair.

For evaluation, we sample a separate set of 500 examples for each task. Section A provides additional dataset and processing details.

**Multi-token details:** The model produces reasoning steps as ordinary text before giving an answer in the format shown by the demonstrations. These reasoning tokens are not enclosed in `<think>` tags, which would mark a separate thinking segment. Both demonstrations and expected outputs use chain-of-thought style: written reasoning steps followed by an answer (Wei et al., 2022).

Evaluation extracts the answer from the generated text. The extraction method depends on the dataset. In the patterns below, angle-bracketed phrases are placeholders, and `####` separates reasoning or support text from the answer.

- **GSM8K and GSM8Ks:** Ground-truth responses follow `<reasoning steps> #### <numerical answer>`. We do not evaluate the correctness or conciseness of the reasoning. We extract only the numerical answer and match it exactly against the ground truth to calculate accuracy. We also use common parsing patterns such as `The answer is <value>` to allow slight format variation. The exact parsing function is in our code.
- **HMathA:** Answers are expected inside a `\boxed{}` element, which encloses the final answer, and are parsed accordingly.
- **SciQ:** We create ground-truth responses in a format similar to GSM8K: `<support text> #### <text answer>`. We parse this format and count only exact string matches as correct.

**Training Setup:** We use two model families. Qwen3-4B-Base is used on every single-token and multi-token task. For single-token tasks, we also use Llama-3.2-1B, except on SciQr, where we use Llama-3.2-3B. The 1B Llama model could not improve above the random baseline for any $N$ up to 128.

For multi-token tasks, the secondary model is Llama-3.2-1B-Instruct. “Instruct” identifies the instruction-following, post-trained model. This tests our method on models that have already undergone post-training.

We train low-rank adaptation modules, or LoRAs, with rank 8. These modules modify a low-dimensional weight-update space rather than all model weights. We target the $W_Q$, $W_K$, and $W_O$ self-attention matrices. We chose the models and weight combinations according to available resources, as the experiments required training more than 13,000 models.

Multi-token training is much more resource-intensive because answers are long. We therefore use $N$ up to 16 for Llama and $N$ up to 8 for Qwen in this setting. For efficiency, we cap generated responses at $G=200$ tokens during training. IA2 uses this limit when collecting the activation tensor
$\hat A_i\in\mathbb R^{L\times200\times d}\ \forall x_i$,
where $\hat A_i$ denotes the collected target tensor and $x_i$ denotes a training input. For SFT, $G$ is the number of tokens in the ground-truth answer.

We train each method–dataset combination until convergence, for up to 50 epochs. An epoch is a pass through the training data. We try 3 learning rates, which control update size: slow, 1e-4; medium, 3e-4; and fast, 1e-3. We choose the best-performing rate to neutralize the effect of hyperparameter selection on method performance. Section B provides additional details.

**Evaluation:** For single-token tasks, we measure accuracy and ECE using the first generated token’s probability distribution. Accuracy measures correct answers; ECE measures calibration.

We first evaluate the base model with ICL demonstrations. For each task, we evaluate its validation set 5 times with different demonstration sets and report average metrics. We then evaluate trained models without demonstrations and report their best-performing average metrics.

For multi-token tasks, we limit generation during evaluation to conserve compute. GSM8K and GSM8Ks have 95th-percentile answer lengths of 184 and 154 tokens, respectively, so we set their generation limit to 200. HMathA has a 95th-percentile answer length of 307, so we relax its limit to 400.

For multi-token performance, we measure only accuracy: the number of questions answered correctly in the required format. See §A for answer parsing and other details. We do not measure calibration because confidence is hard to measure for open-ended answers.

### 5.1 IA2 priming boosts SFT performance

**Single-Token results:** Table 1 reports Qwen3-4B-Base results for the practical setting of $N=4$. “SFT only” uses only the SFT objective, and “IA2 only” uses only the activation-alignment objective. In the tables, “Source” is the training dataset and “Eval” is the evaluation dataset. The labels `acc` and `ece` mean accuracy and expected calibration error. An upward arrow (↑) means higher is better; a downward arrow (↓) means lower is better. An asterisk (∗) marks OOD evaluation.

IA2 → SFT outperforms SFT-only training in almost all cases for both accuracy and calibration, using the same amount of data. For example, Table 1’s SST2 → SST2 row reports accuracy of 90.4 (02.3) and ECE of 0.06 (0.01) for IA2 → SFT, compared with 65.2 (10.7) and 0.22 (0.10) for SFT only.

In most cases, IA2 → SFT also exceeds ICL accuracy, with slightly less calibration. The FinS → FinS row, for example, reports accuracy of 78.7 (13.7) versus ICL’s 63.6 (01.5), with ECE of 0.16 (0.11) versus 0.12 (0.01).

IA2-only training often achieves high accuracy and reasonable, ICL-like calibration without training on target-response tokens. For example, the SciQr → SciQr row reports 90.4 (01.5) accuracy and 0.09 (0.01) ECE for IA2 only. This highlights the richness of the IA2 signal.

Table 1 presents one setup. Figure 1 shows aggregate trends across models and datasets. Section C provides detailed tables for other model/$N$ combinations and a statistical-significance test of the improvement.

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

**Table 1:** Performance for $N=4$ on Qwen3-4B-Base, showing accuracy (acc) and Expected Calibration Error (ece). Numbers in parentheses are standard deviations across 5 runs for the best-performing learning rate. Best training method shown in bold. (∗) marks OOD evaluations. Our proposed IA2 → SFT method outperforms standard SFT across the board.

The `w/o ICL` column in Table 2 means the base model evaluated without ICL demonstrations.

| Source | Eval | w/o ICL acc ↑ | ICL acc ↑ | SFT only acc ↑ | IA2 only acc ↑ | IA2 → SFT acc ↑ |
|---|---|---:|---:|---:|---:|---:|
| GSM8K | GSM8K | 56.4 (00.0) | 76.4 (01.1) | 70.9 (02.8) | 77.4 (02.8) | 73.6 (03.7) |
| GSM8K | GSM8Ks∗ | 45.4 (00.0) | 68.4 (01.1) | 64.5 (04.0) | 66.2 (03.7) | 68.8 (05.1) |
| HMathA | HMathA | 21.0 (00.0) | 60.4 (02.1) | 50.4 (01.8) | 47.8 (06.0) | 55.3 (02.1) |
| SciQ | SciQ | 15.4 (00.0) | 37.5 (01.0) | 35.0 (07.1) | 06.9 (13.8) | 40.8 (07.6) |

**Table 2:** Performance for $N=4$ on Qwen3-4B-Base models on multi-token datasets. Best training method shown in bold. IA2 → SFT outperforms standard SFT on all datasets.

**Multi-Token results:** Table 2 reports adapted Qwen accuracy in the multi-token setting. IA2 → SFT exceeds SFT-only accuracy in every row: GSM8K, GSM8Ks, HMathA, and SciQ.

Compared with the single-token setting, IA2 → SFT sometimes performs worse than ICL. Table 2 reports 73.6 (03.7) versus 76.4 (01.1) on GSM8K, and 55.3 (02.1) versus 60.4 (02.1) on HMathA. We suspect that longer contexts are imperfectly compressed into the same small LoRA weight space. Larger ranks should improve this. Section C contains more tables for other model/$N$ combinations.

### 5.2 Why is IA2 priming important?

**Figure 3:** Scatter plots of $\operatorname{asim}(A,A_{\text{ICL}})$ against accuracy and ECE for all training methods show how activation similarity affects performance. Here, $A$ is the trained model’s activation tensor and $A_{\text{ICL}}$ is the ICL target. Each point represents one training experiment. ECE decreases smoothly as similarity increases. But extreme ICL activation alignment may leave accuracy gains that can be obtained through SFT on ground-truth responses.

We examine three methods: 1) SFT only, 2) IA2 only, and 3) IA2 → SFT.

**Activation Similarity vs Performance:** Figure 2 shows each method’s activation similarity with ICL. As expected, IA2 drastically increases similarity. Importantly, IA2 → SFT retains higher ICL similarity after the SFT stage.

Figure 3 relates similarity to accuracy and calibration. IA2 → SFT lies between IA2 only and SFT only in activation similarity, while achieving better accuracy and calibration. SFT-only training and the SFT stage of IA2 → SFT differ only in their starting state. Their different outcomes therefore imply that IA2 supplies a rich training signal unavailable through SFT-only training.

Greater activation similarity implies better calibration in Figure 3. However, extreme similarity may not produce the best accuracy because ICL is not always correct. Section C shows a case where IA2 models follow ICL performance even when it worsens through overfitting. We therefore recommend combining the signals: IA2 aligns internal functional behavior with ICL, and SFT on ground-truth responses aligns outputs with the expected behavior.

**Subspace Overlap:** Activations reflect functional behavior, but they do not isolate changes caused by each training method. We use LoRA adapters, so unchanged base-model weights strongly influence the activations. To isolate training-induced changes, we analyze the LoRA weights directly.

A weight subspace is the set of directions spanned by the weight updates. Subspace overlap measures how much of this space two methods share.

**Figure 4:** IA2 → SFT models have significantly high subspace overlap with IA2-only models. SFT-only models do not align with either. This indicates that IA2 provides a training signal absent from SFT-only training and important for performance.

Figure 4 shows overlap distributions for pairs of methods, using models from all single-token and multi-token experiments. We first apply singular value decomposition (SVD), a matrix decomposition used here to find basis vectors that span the weight space. We then calculate:

\[
\frac{1}{r}*\|U_1^T\cdot U_2\|_F^2.
\]

Here, $U_1$ and $U_2$ contain basis vectors for the two methods being compared, and $r$ is the LoRA weight rank. The asterisk denotes multiplication. The squared Frobenius norm, $\|\cdot\|_F^2$, sums the squared entries of a matrix. The formula measures shared weight space, thereby isolating and comparing differences in functional behavior.

SFT-only updates are almost completely orthogonal to those of the other two methods: their weight-update directions are nearly perpendicular. In contrast, the most performant IA2 → SFT models share around $39\%$ of their spanned weight space, on average, with IA2-only models (Figure 4). This implies that much of the sequential models’ performance gain comes from IA2. The subspaces found by IA2 are practically unreachable through SFT-only training.

Overlap between IA2 only and IA2 → SFT has a wide spread. In many cases, the IA2 model is already output-aligned. The subsequent SFT stage then makes little change before convergence, producing very high overlap.

## 6 Discussion

**IA2 + SFT:** A natural alternative to sequential IA2 → SFT is IA2 + SFT: optimize both $\mathcal L_{\text{IA2}}$ and $\mathcal L_{\text{SFT}}$ simultaneously. This poses a practical challenge. For any response longer than one token, $G>1$, the target activation tensor $\hat A$ follows the model’s own generated ICL response. This response may differ from the dataset’s ground-truth response, even in length. The two objectives are therefore incompatible for joint training on those different responses.

We can instead self-distill using ICL responses, as in Snell et al. (2022), while adding the IA2 signal. We test the unified loss
$\mathcal L_{\text{IA2}}+\beta\cdot\mathcal L_{\text{SFT}}$,
where $\beta$ controls the contribution of SFT loss. We vary $\beta$ over 4 values that smoothly span IA2-only to SFT-only training. Details are in §B.

Table 3 reports multi-token results after selecting the best learning-rate/$\beta$ combination across 5 random seeds. A random seed fixes the randomness used in a run. Relative to training on ground-truth responses in Table 2, SFT-only and IA2 → SFT accuracy drops in every Table 3 row when trained on ICL responses. However, IA2 + SFT extracts significantly more performance from ICL responses than SFT alone. For example, the GSM8K → GSM8K row reports 77.0 (07.8) for IA2 + SFT versus 66.4 (18.1) for SFT only. This highlights the richness of IA2’s signal. Section C provides additional joint-training results.

| Source | Eval | w/o ICL acc ↑ | ICL acc ↑ | SFT only acc ↑ | IA2 only acc ↑ | IA2 → SFT acc ↑ | IA2 + SFT acc ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| GSM8K | GSM8K | 56.4 (00.0) | 76.4 (01.1) | 66.4 (18.1) | 77.4 (02.8) | 67.2 (22.1) | 77.0 (07.8) |
| GSM8K | GSM8Ks∗ | 45.4 (00.0) | 68.4 (01.1) | 59.8 (20.2) | 66.2 (03.7) | 61.6 (24.0) | 69.0 (09.6) |
| HMathA | HMathA | 21.0 (00.0) | 60.4 (02.1) | 49.0 (03.0) | 47.8 (06.0) | 54.0 (01.7) | 53.6 (02.7) |
| SciQ | SciQ | 15.4 (00.0) | 37.5 (01.0) | 27.2 (12.0) | 06.9 (13.8) | 32.8 (10.0) | 34.5 (10.6) |

**Table 3:** Performance for $N=4$ on Qwen3-4B-Base models trained with ICL responses instead of ground-truth responses. IA2 + SFT performs significantly better than SFT only, highlighting the significance of IA2.

**Knowledge Distillation baseline:** Knowledge distillation (KD) transfers a teacher model’s behavior to a student. A popular method is soft-label matching (Hinton et al., 2015). Rather than supplying only a target token, the teacher supplies probability mass for each token, giving a denser training signal.

Our teacher is the same model enhanced with ICL context. To compare IA2-based training with soft-label matching, we train and evaluate Llama-3.2-1B-Instruct on GSM8K and Llama-3.2-1B on SST2, using soft labels on ICL responses.

Table 4 shows that soft-label matching performs significantly better than standard SFT in the multi-token case and nearly reaches IA2 + SFT. On GSM8K, the accuracies are 36.1 (03.5), 28.2 (04.5), and 37.0 (01.9), respectively. But soft-label matching lacks in the single-token case. On SST2, its accuracy is 55.9 (08.8), compared with 64.9 (15.3) for IA2 only, 64.7 (19.3) for IA2 → SFT, and 66.9 (18.4) for IA2 + SFT. These results highlight the consistency of IA2-based training in achieving high performance. Dashes in Table 4 mark calibration values that are not reported.

| Dataset | $N$ | ICL acc ↑ | ICL ece ↓ | SFT only acc ↑ | SFT only ece ↓ | SFT (soft labels) acc ↑ | SFT (soft labels) ece ↓ | IA2 only acc ↑ | IA2 only ece ↓ | IA2 → SFT acc ↑ | IA2 → SFT ece ↓ | IA2 + SFT acc ↑ | IA2 + SFT ece ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 4 | 34.6 (00.9) | – | 28.2 (04.5) | – | 36.1 (03.5) | – | 29.3 (05.0) | – | 30.3 (03.2) | – | 37.0 (01.9) | – |
| SST2 | 8 | 77.0 (02.2) | 0.10 (0.02) | 52.3 (04.3) | 0.38 (0.14) | 55.9 (08.8) | 0.16 (0.06) | 64.9 (15.3) | 0.30 (0.05) | 64.7 (19.3) | 0.33 (0.19) | 66.9 (18.4) | 0.28 (0.21) |

**Table 4:** Comparison with the Knowledge Distillation (KD) baseline, SFT on soft labels. Although KD performs better than standard SFT in the multi-token case, IA2-based training provides consistently high performance in single-token and multi-token cases.

**Why IA2 based training does not consistently beat ICL?** In some multi-token cases, such as Qwen on the math datasets GSM8K and HMathA, ICL exceeds all training methods, including IA2-based training. We suspect that Qwen’s mid-training on science, technology, engineering, and mathematics (STEM) data makes its ICL extremely sample-efficient. Surprisingly, GSM8K ICL accuracy at $N=2$ is higher than at $N=4,8$: Table 5 reports 81.2 (00.4), 76.4 (01.1), and 77.2 (00.9), respectively.

Table 5 shows how performance evolves with the number of shots, or training examples. It also shows the IA2 → SFT gap relative to ICL. This gap closes quickly with more shots. The last column supports our main takeaway: IA2 → SFT performs better than SFT only. “wrt” means “with respect to”; the gap columns compare IA2 → SFT with the named method.

| Dataset | $N$ | ICL | SFT only | IA2 only | IA2 → SFT | Gap wrt ICL | Gap wrt SFT only |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 2 | 81.2 (00.4) | 65.2 (01.0) | 70.7 (01.8) | 74.6 (02.4) | -06.6 | +09.4 |
| GSM8K | 4 | 76.4 (01.1) | 70.9 (02.8) | 77.4 (02.8) | 73.6 (03.7) | -02.8 | +02.7 |
| GSM8K | 8 | 77.2 (00.9) | 73.5 (01.6) | 78.8 (02.5) | 76.0 (02.0) | -01.2 | +02.5 |
| HMathA | 2 | 58.6 (02.0) | 47.8 (04.0) | 33.9 (06.4) | 48.7 (07.0) | -09.9 | +00.9 |
| HMathA | 4 | 60.4 (02.1) | 50.4 (01.8) | 47.8 (06.0) | 55.3 (02.1) | -05.1 | +04.9 |
| HMathA | 8 | 62.1 (01.0) | 52.0 (00.8) | 57.3 (03.7) | 59.4 (00.8) | -02.7 | +07.4 |

**Table 5:** ICL is more sample-efficient than training methods for Qwen models on math datasets. IA2 → SFT quickly closes the performance gap as the number of shots increases and consistently beats SFT-only training.

**Other fine-tuning methods:** LoRA is not the only way to fine-tune a model. Alternatives include full fine-tuning, which uses full-rank updates; prompt/prefix tuning, which learns prompts or prefixes (Lester et al., 2021; Li and Liang, 2021); and activation scaling, which learns adjustments to activation scale, called (IA)3 (Liu et al., 2022).

Our pipeline does not compete with these methods. It improves them by supplying better training signals. To establish this, we repeat experiments on a few datasets using the popular PEFT method (IA)3 instead of LoRA. We otherwise follow the same methodology as §5. Section B gives setup details.

Table 6 reports one setting and shows trends similar to LoRA. IA2 → SFT exceeds SFT-only and ICL accuracy in every row. For example, on SciQr → QASCr, its accuracy is 48.6 (05.5), compared with 32.8 (03.4) for SFT only and 35.6 (01.4) for ICL. Section C contains other results.

We leave exploration of IA2 improvements to other SFT methods for future work. As a side note, IA2 also marks a major improvement in method naming compared with (IA)3.

| Source | Eval | ICL acc ↑ | ICL ece ↓ | SFT only acc ↑ | SFT only ece ↓ | IA2 only acc ↑ | IA2 only ece ↓ | IA2 → SFT acc ↑ | IA2 → SFT ece ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SciQr | QASCr∗ | 35.6 (01.4) | 0.06 (0.01) | 32.8 (03.4) | 0.20 (0.08) | 34.2 (02.7) | 0.08 (0.01) | 48.6 (05.5) | 0.25 (0.11) |
| SciQr | SciQr | 60.0 (01.8) | 0.07 (0.01) | 64.5 (05.0) | 0.12 (0.05) | 63.0 (04.6) | 0.10 (0.04) | 70.6 (04.6) | 0.08 (0.02) |
| SST2 | FinS∗ | 56.6 (01.6) | 0.17 (0.01) | 58.2 (10.7) | 0.25 (0.03) | 61.9 (09.5) | 0.18 (0.06) | 59.8 (07.8) | 0.18 (0.03) |
| SST2 | PoemS∗ | 52.3 (03.0) | 0.19 (0.02) | 48.1 (03.2) | 0.44 (0.11) | 48.1 (03.4) | 0.11 (0.04) | 59.8 (13.4) | 0.33 (0.13) |
| SST2 | SST2 | 60.3 (02.8) | 0.19 (0.02) | 52.4 (02.8) | 0.35 (0.05) | 54.2 (04.2) | 0.15 (0.06) | 62.6 (09.1) | 0.28 (0.11) |

**Table 6:** Performance for $N=4$ on Llama-3.2 models trained using (IA)3. IA2 → SFT achieves significant gains over SFT only and exceeds ICL accuracy in all cases.

**Computational Overhead:** IA2 → SFT requires an activation-collection step before training. We run ICL inference on the model with the given dataset to collect its rich activations. This is a one-time cost. The resulting capability improvements require no extra inference cost relative to SFT-only models.

**Potential improvements:** Resource constraints fixed several design choices that merit further exploration. The most important include controlled comparisons of LoRA settings—such as rank and target modules—and selective IA2.

Extreme alignment can hurt performance (§5.2, Figure 3). Prior work also shows that some layers are more sensitive to ICL than others (Todd et al., 2023). Selectively aligning layers could therefore be more effective.

Another minor improvement could be prompt optimization before collecting target activations. ICL performance is sensitive to demonstration order (Lu et al., 2022; Zhao et al., 2021).

Finally, our IA2 → SFT pipeline uses fixed learning rates. The two training signals may benefit from finer-grained control. We leave these explorations for future work.

## 7 Related Work

**ICL vs SFT:** Prior studies compare ICL and SFT performance, but the verdict remains unclear (Mosbach et al., 2023; Duan et al., 2024). In our work, SFT performed worse than ICL (§3, Figure 1). Other work studies their functional similarity (Dai et al., 2022) and differences (Wang et al., 2023).

Several studies examine miscalibration under ICL or SFT and ways to improve it (Zhang et al., 2024; Li et al., 2025; Li et al., 2024; Xiao et al., 2025). Sia et al. (2024) use context masking—hiding parts of the context—to identify “where” translation happens during ICL. Like us, they use activations as a signal of functional behavior.

Recent work finds SFT prone to memorization, overfitting, and catastrophic forgetting: loss of previously learned behavior during further training (Chu et al., 2025; Shenfeld et al., 2025). Our improved SFT pipeline aims to address these problems.

**Context Distillation:** Like our work, a large body of research improves LLM adaptation by compressing explicit context signals. Approaches include:

- Intermediate contemplation tokens for compression (Cheng and Van Durme, 2024; Jiang et al., 2025).
- Internalizing context by generating adapters on the fly (Chen et al., 2025; Charakorn et al., 2025).
- Using related text to enhance distillation (Zhu et al., 2025; Choi et al., 2025).
- Compressing context directly into weights (Shen et al., 2025; Deng et al., 2024; Yu et al., 2024; Snell et al., 2022; Chen et al., 2024b; Shin et al., 2024).

These methods use token-response signals to distill the effect of context on model outputs. Our work is complementary. It improves distillation through the model’s own ICL-processing signals, an idea supported by Aguilar et al. (2020), Jin et al. (2024), and Yang et al. (2024).

**Activation Steering:** A growing line of work extracts “steering” vectors or weights. These are applied at targeted internal model locations to produce an intended output effect (Postmus and Abreu, 2024; Stolfo et al., 2025; Caccia et al., 2025; Fleshman and Van Durme, 2024; Arditi et al., 2024).

Like our method, these approaches intervene in activation space to influence model functionality. Unlike their targeted approach and small application domain, our method aims to induce a large behavioral change. This avoids potential brittleness and reliability issues (Queiroz Da Silva et al., 2025).

## 8 Conclusion and Future Work

We used model activations to probe LLM functional behavior and found a key distinction: ICL and SFT adapt through different mechanisms (§3, Figure 2). ICL often encodes richer, more generalizable patterns.

This finding motivated IA2, an SFT priming technique that enables ICL-like internal behavior. IA2 shifts models into a more adaptable weight subspace inaccessible to SFT alone. The subspace analysis in §5.2 and Figure 4 supports this finding: the most performant IA2 → SFT models share around $39\%$ of their spanned weight space with IA2-only models, while SFT-only updates are almost completely orthogonal to both.

Extensive experiments show that IA2 consistently improves SFT accuracy and calibration (§5; Tables 1–2). Future work will examine effects on output diversity and catastrophic forgetting to measure IA2’s usefulness in LLM post-training. We also aim to study IA2 on post-trained models tuned through reinforcement learning (RL).

## Reproducibility Statement

We provide code on GitHub for data downloading and processing, training, evaluation, analysis, plotting, and other steps. The details in §5 and the appendix, together with a detailed README instruction file, should be sufficient to reproduce our results fully. We used repeatable random seeds.

## Acknowledgments

This work is supported by the Office of Naval Research (ONR; N00014-24-1-2089), an Amazon Research Award, and a grant from Open Philanthropy. We thank Andrea Wynn for their constructive feedback on earlier versions of this document.
