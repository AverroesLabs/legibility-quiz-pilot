# IA2: Alignment with ICL Activations

Improves Supervised Fine-Tuning

Aayush Mishra

  
Daniel Khashabi & Anqi LiuDepartment of Computer Science,
Johns Hopkins University

###### Abstract

Supervised Fine-Tuning (SFT) specializes a model’s behavior by training its weights to produce intended responses to queries. In contrast, In-Context Learning (ICL) adapts a model during inference—the process of producing a response—through instructions or demonstrations in its prompt. When data are scarce, ICL can generalize better to new examples and produce better-calibrated responses than SFT. Better calibration means that the model’s confidence more closely matches how often it is correct. However, ICL requires more computation during inference.

We ask: Can ICL’s internal computations improve SFT? We first show that ICL and SFT produce distinct activation patterns. Activations are internal outputs that reflect the model’s processing. These distinct patterns indicate that the two methods adapt models through different functional mechanisms. To use ICL’s rich functionality, we introduce ICL Activation Alignment (IA2). This self-distillation technique trains a model using signals from the same model. It aims to reproduce ICL activation patterns in SFT models and encourages ICL-like internal reasoning. Using IA2 as a priming step before SFT significantly improves output accuracy and calibration. Extensive experiments on 12 popular benchmarks and two model families show these gains. This finding offers both practical benefits and a conceptual view of how model adaptation works internally.

00footnotetext: ♡Equal advising. Correspondence to: {amishr24,danielk,aliu.cs}@jhu.edu

## 1 Introduction

Large language models (LLMs) serve general purposes, but people often use them in specialized applications. For example, a news aggregator app may need to classify articles into predefined categories. Supervised Fine-Tuning (SFT) is a popular way to adapt LLMs. It uses labeled examples to train models for specific downstream tasks.

Parameter efficient fine tuning (PEFT) methods train only a small set of parameters (Hu et al., 2021; Liu et al., 2022). With these methods, SFT models often consist of small parameter sets that users can efficiently load into or remove from graphics processing unit (GPU) memory. This makes them extremely useful. However, SFT usually needs many labeled examples to generalize well to new tasks (Le Scao and Rush, 2021). Collecting these examples can be expensive.

In contrast, In-Context Learning (ICL) (Brown et al., 2020) adapts and guides LLM behavior during inference. The model receives a query after demonstrations or instructions in the prompt. These help it “learn” the demonstrated task and respond accordingly. Prior work (Duan et al., 2024) shows that ICL generalizes well in few-shot settings, where only a few examples are available. It also typically produces well-calibrated responses. Our experiments agree with these findings (§5).

However, ICL has a cost. Its demonstrations and instructions occupy valuable context space and increase the cost of each query. The model could otherwise use that space to process more query and response tokens, the units of text it handles.

These subtle but important differences motivate us to study how ICL and SFT function. Their behavior in token space—the text they produce—may look similar. Yet they produce different internal activations (§3). Activations record the model’s internal processing before it produces the next token. Different activation patterns therefore highlight different adaptation mechanisms. Prior work supports this idea (Shen et al., 2024).

We hypothesize that ICL activations contain rich information about how to extract generalizable patterns from context. SFT models may lack this information, especially in few-shot settings. In these settings, they can learn shortcuts or overfit target responses, fitting training examples without generalizing well. We therefore ask:

**Can the information-rich ICL activations improve the quality of SFT?**

Recent work has tried to distill context into LLM weights (Snell et al., 2022; Chen et al., 2024b). However, these methods use training signals only from response texts. They may therefore face the same SFT problems described above. Training a model to reproduce an ICL-conditioned model’s outputs does not ensure that the two models function alike.

To address this problem, we propose ICL Activation Alignment (IA2). This self-distillation method enforces alignment with the model’s own functional behavior during ICL (§4). Figure 1 outlines the method: (1) collect information-rich ICL activations, (2) enforce functional alignment with ICL, and then (3) perform SFT on the primed model.

Priming models with IA2 before SFT, using the same data, drastically improves adapted models on text classification and generation tasks (§5). We also show that IA2 supplies an important training signal that SFT-only training does not provide. We trained over 13,000 models across 12 benchmarks to validate these findings. The results show IA2’s practical benefits and offer a conceptual view of model adaptation’s internal mechanisms.

In summary:

- We show that ICL and SFT do not align in the model’s activation space, even with the same data. This highlights a gap in their functional behavior (§3).
- We propose IA2—ICL Activation Alignment—to enforce alignment with ICL’s functional behavior. This priming step drastically improves SFT models’ performance (Figure 1, §4).
- We show that SFT-only training does not provide the IA2 training signal. This highlights its importance for improving adaptation (§5).

**Figure 1:** Overview of our improved SFT pipeline. Standard SFT enforces alignment only in output space: it aligns the model’s response with a target response through step 3 above. This leads to subpar performance and miscalibration when data are limited. In contrast, IA2 enforces functional alignment with ICL by matching the rich activation patterns that ICL produces. IA2 priming before SFT improves adaptation. Performance comparison charts, aggregated across models and datasets, show this improvement. See section 5 for details.

## 2 Background and Notation

**Transformer Language Models:** A standard decoder-only transformer language model (LM), \(M_{\Theta}\), has parameters \(\Theta\). It consists of stacked self-attention (SA) and linear layers. Self-attention combines information from tokens in the sequence. Linear layers transform token vectors through linear projections.

The model processes a sequence of token vectors, \(T=[t_i]_{i=1}^{R}\), by applying self-attention and linear projections to each token at each layer. Here, \(R\) is the number of tokens. It passes the last token’s projection at the final layer through the LM head, the component that predicts the most likely next token.

Self-attention differs from the other operation because other tokens in the sequence affect it. A standard self-attention operation uses 4 weight matrices: \(W_Q\), \(W_K\), \(W_V\), and \(W_O\), all in \(\Theta\). We call this set \(W_{QKVO}\). Its output at each token position is:

\[
\text{SA}(T;W_{QKVO})=Z=
\left[
z_i=\sigma\left(\frac{q_iK^T}{\sqrt{d}}\right)V\cdot W_O
\right]_{i=1}^{R}.
\tag{1}
\]

Here, \(\sigma(.)\) denotes Softmax, which converts scores into probabilities. The query vector is \(q_i=t_i\cdot W_Q\), the key matrix is \(K=t_{:i}\cdot W_K\), and the value matrix is \(V=t_{:i}\cdot W_V\). The notation \(t_{:i}\) means the first \(i\) tokens. The superscript \(T\) denotes matrix transpose. The output \(Z\) has the same shape, \(R\times d\), as the input \(T\), where \(d\) is the vector dimension.

We study how changes to context \(T\) and weights \(W_{QKVO}\) interact to change model behavior.

**Supervised Fine-Tuning:** Let

\[
\mathcal{D}_{\mathcal{T}}=\{(X_i,Y_i)\}_{i=1}^{N}
\]

be a set of \(N\) input–output pairs that illustrate task \(\mathcal{T}\). Here, \(X\) denotes an input and \(Y\) its output. We can fine-tune the model weights from \(\Theta\) to \(\Theta'\) so the model produces relevant responses to new inputs. For a new input \(X_t\),

\[
M_{\Theta'}(X_t)=\bar{y}_t.
\]

The token \(\bar{y}_t\) begins the full response \(\bar{Y}_t\). Repeatedly applying \(M_{\Theta'}\) to the growing sequence extends that response.

If the ground truth response \(Y_i\) contains \(G\) tokens, we generate \(G\) new tokens to obtain \(\bar{Y}_i\). We then minimize cross-entropy loss, which penalizes differences from target tokens, over all generated tokens. The SFT loss is:

\[
\mathcal{L}_{\text{SFT}}
=
\sum_{i=1}^{N}\sum_{j=1}^{G}
\text{cross-entropy}(\bar{Y}_{ij},Y_{ij}).
\tag{2}
\]

**In-Context Learning:** We can also use \(\mathcal{D}_{\mathcal{T}}\) without updating weights. We prompt \(M_{\Theta}\) with a sequence of demonstration tokens,

\[
I=[X_1\circ Y_1\circ\ldots\circ X_n\circ Y_n],
\]

followed by a new test sample \(X_t\). The symbol \(\circ\) means concatenation: joining sequences in order. The model processes the prompt’s examples to understand the task and produces a task-appropriate response:

\[
M_{\Theta}(I\circ X_t)=\hat{y}_t.
\]

Remarkably, the full response \(\hat{Y}_t\) often resembles the expected response \(Y_t\). ICL therefore offers a useful way to adapt models during inference.

Brown et al. (2020) first illustrated ICL in GPT-3. People now widely use it to adapt general-purpose language models during inference. Today’s frontier models can follow instructions directly, so a general instruction could count as a form of ICL demonstration. Here, we focus on classic ICL, using demonstrations made of input–output pairs.

**Activations:** Self-attention uses every token in the context, so different token sequences produce different outputs. We call the hidden self-attention outputs *activations*. These outputs reflect the model’s internal processing. We can use them to study how model behavior changes with context.

If \(M_{\Theta}\) has \(L\) layers and processes \(R\) tokens, it produces an activation tensor \(A\) of size \(L\times R\times d\). A tensor is a multidimensional array. We study token sequences \(T=[I\circ X]\) for ICL and \(T=X\) without ICL under different model weights. Here, \(I\) contains the ICL demonstration tokens, and \(X\) contains the query tokens.

We define activation similarity, \(\operatorname{asim}(A_1,A_2)\), as the token-wise cosine similarity between two activation tensors. Cosine similarity compares vector directions. The resulting asim array has size \(L\times R\).

## 3 Are ICL and SFT practically the same?

**Figure 2: Layerwise Similarity with ICL Activations.** SFT models have low activation similarity, or asim, with ICL. This indicates that they adapt through different functional behavior. Our recommended pipeline, IA2 → SFT, aligns much more closely with ICL than SFT alone and performs much better as a result (Figure 1). The arrow means that IA2 training comes before SFT.

Recent studies claim that ICL in transformers works through an internal gradient descent mechanism, a process that updates parameters to reduce loss (Von Oswald et al., 2023; Akyürek et al., 2022). They show that self-attention can produce outputs as though the model had updated its parameters using the ICL demonstrations in context. This implies functional equivalence between the two methods.

Under this equivalence, we should expect similar activations at output token positions from two settings: the base model with ICL demonstrations and the SFT model without them. We investigated whether LLMs actually show this pattern.

**ICL and SFT produce different activations:** We collected ICL activations \(A_{\text{ICL}}\), using \(T=[I\circ X]\), for 100 test samples from several datasets, including SST2 and AGNews. We used models from the Qwen3 (Yang et al., 2025) and Llama-3.2 (Grattafiori et al., 2024) families. Their sizes and layer counts were 1 billion parameters with 16 layers, 3 billion with 28 layers, and 4 billion with 36 layers.

We then trained SFT models on the same ICL demonstrations until convergence, when training stopped improving, and collected activations \(A_{\text{SFT}}\) using \(T=X\). Section B gives experiment details. We calculated \(\operatorname{asim}(A_{\text{ICL}},A_{\text{SFT}})\) at output token positions. We then plotted averages across tokens, samples, and datasets. Figure 2 shows these results as “SFT only” in the first row.

ICL and SFT activations do not align across models. They align better near the initial and final layers, where we expect the model to process tokens individually. They misalign in the middle layers, where we expect the model to process the whole demonstration set at an abstract level.

**Why should we care about alignment with ICL?** The functional difference between ICL and SFT appears not only in activation patterns but also in expected calibration error (ECE) (Guo et al., 2017), as Figure 1 shows. ECE measures the gap between confidence and accuracy. At similar—and slightly better—accuracy, ICL responses are much better calibrated than SFT responses.

We suspect that SFT’s output-focused training signal allows it to learn shortcuts that fail on new data. In contrast, ICL relies on complex circuits, or interacting internal components, that must work across many tasks (Elhage et al., 2021). It therefore extracts generalizable patterns from demonstrations and performs in a well-calibrated way.

This experiment shows that ICL and SFT adapt models through different means. We should not expect SFT to serve as a drop-in replacement for ICL. These differences motivate our question:

**Can the information-rich ICL activations improve the quality of SFT?**

Next, we show that they can and describe our proposed SFT procedure.

## 4 Inducing ICL-like behavior in SFT Models

Our SFT pipeline has two steps. First, IA2 primes the model by encouraging ICL-like functional behavior. Further SFT on these ICL-aligned weights then produces much better adaptation.

**IA2—ICL Activation Alignment:** In §3, SFT showed little activation similarity with ICL. To increase this similarity, and thus functional similarity with ICL, we set the following IA2 goal for each layer of \(M_{\Theta}\):

Find

\[
\tilde{W}_{QKVO}
=
(\tilde{W}_Q,\tilde{W}_K,\tilde{W}_V,\tilde{W}_O)
\]

such that, for every newly generated token at index \(-1\),

\[
\text{SA}([I\circ X];W_{QKVO})
\approx
\text{SA}(X;\tilde{W}_{QKVO})
\quad \forall X\in\mathcal{T}.
\tag{3}
\]

Here, \(\forall\) means “for every,” and \(\approx\) means “approximately equal.”

Prior work gives a closed-form solution—a direct formula—for this goal in linearized attention models (Chen et al., 2024a). Those models use a kernel approximation for Softmax to make attention linear. We seek a practical, general solution for real, nonlinear transformers.

Our goal puts ICL’s qualities directly into the SFT model’s weights. It does not encourage the modified weights to produce similar outputs. Instead, it encourages them to process all queries as ICL does at each layer of the base model.

To perform IA2, we first generate responses with ICL, using \(T_i=[I\circ X_i]\). This gives a response \(\hat{Y}_i\) for each training sample \(X_i\). For each sample, we construct \(I\) from the dataset’s remaining samples. We collect activations at each output position. Assuming \(G\) response tokens, this gives

\[
A^i_{\text{ICL}}\in\mathbb{R}^{L\times G\times d},
\]

a real-valued activation tensor.

Next, we attach the response only to the query:

\[
T_i=[X_i\circ\hat{Y}_i].
\]

This treats the model as though it had produced the ICL response with only the query in context. It gives an unaligned activation tensor \(A^i\) at the output token positions. We then train the model parameters to minimize the mean squared error between the two activations:

\[
\mathcal{L}_{\textbf{IA2}}
=
\sum_{i=1}^{N}
\left\|A^i-A^i_{\text{ICL}}\right\|.
\tag{4}
\]

The double bars denote a norm, which measures the size of the difference.

**IA2 → SFT:** After IA2, we continue training with the standard SFT cross-entropy loss on target tokens (Equation 2). Suppose IA2 changes the model parameters from \(\Theta\) to \(\Theta'\). We collect responses \(\bar{Y}_i\) for all inputs \(X_i\) using \(M_{\Theta'}\). We then minimize the SFT loss between \(\bar{Y}_i\) and ground truth \(Y_i\) for every training sample. This further aligns the outputs with the intended targets.

The full pipeline has two steps:

1. Collect target ICL activations \(A_{\text{ICL}}\), then perform IA2 using \(\mathcal{L}_{\textbf{IA2}}\) until convergence. This aligns the model’s functional behavior with ICL.
2. Switch to \(\mathcal{L}_{\text{SFT}}\), using ground truth target responses \(Y\), and train until convergence. This aligns the model with target output behavior.

## 5 Experimental Results

We run experiments in two settings with different features: single-token and multi-token outputs. We note differences between them when presenting results.

**Adaptation Tasks:** In the single-token setting, each output \(Y_i\) contains one token. This token typically represents a category or class label, True/False, or a multiple-choice question (MCQ) choice. In the multi-token setting, \(Y_i\) can contain several tokens and vary in length. This is the general setting for open-ended generation.

We test and compare our proposed SFT pipeline across tasks and models in both settings. In many cases, we also evaluate on out-of-distribution (OOD) datasets. These differ in distribution from the training data and let us study behavior under distribution shift. In-distribution (ID) datasets match the training distribution.

**Single-token tasks:**

- **Sentiment Classification:** We use SST2 (Socher et al., 2013), Financial Phrasebank [FinS] (Malo et al., 2014), and Poem Sentiment [PoemS] (Sheng and Uthus, 2020). We train models on SST2 and FinS. For each training set, we evaluate on all three datasets. This gives one ID and 2 OOD evaluation datasets.
- **True/False:** We use StrategyQA [STF] (Geva et al., 2021). It requires multi-hop reasoning—several linked reasoning steps—before a True/False conclusion.
- **News Categorization:** We use AGNews [AGN] (Zhang et al., 2015) and BBCNews [BBCN] (Greene and Cunningham, 2006). We train on AGN and evaluate on both.
- **MCQ:** We use SciQ (Johannes Welbl, 2017) and QASC (Khot et al., 2019). To add complexity, we replace the MCQ choices [A, B, C …] with an unrelated token set. We call the resulting datasets SciQr and QASCr. We train on SciQr and evaluate on both.

**Multi-token tasks:**

- **Grade-school Math:** We use GSM8K (Cobbe et al., 2021) and GSM Symbolic [GSM8Ks] (Mirzadeh et al., 2024). We train on GSM8K and evaluate on both.
- **Advanced Math:** We use MATH Algebra [HMathA] (Hendrycks et al., 2021), which contains higher-grade algebra problems.
- **Scientific QA:** For scientific question answering (QA), we again use SciQ. This time, the model generates the answer’s tokens rather than choosing an option.

**Data Setup:** We focus on the data-scarce, few-shot setting where people typically use ICL. For each task, we create several training datasets,

\[
\mathcal{D}_{\mathcal{T}}=\{(X_i,Y_i)\}_{i=1}^{N},
\]

by varying \(N\) in powers of \(2:[2,4,8,16,\ldots]\). We create 5 different sets at each \(N\) to average out outlier effects in the final quantitative analysis.

For every training sample \(X_i\), we collect the ICL activation tensor \(A^i_{\text{ICL}}\) for IA2 using the remaining \(N-1\) samples as demonstrations in random order. This reuses training samples. It ensures that all adaptation methods use exactly the same data, keeping the comparison fair. For evaluation, we sample a different set of 500 examples for each task. Section A gives more details about datasets and processing.

**Multi-token details:** Here, the model produces reasoning tokens before the answer, following the format shown in the demonstrations. These tokens appear as standard text, not inside `<think>` tags. Both demonstrations and expected outputs use a chain-of-thought style, which presents reasoning steps before an answer (Wei et al., 2022).

We must extract the answer from the generated text to evaluate performance. Each dataset uses a different method:

- For GSM8K and GSM8Ks, ground truth answers follow the pattern “`<reasoning steps> #### <numerical answer>`”. We do not evaluate the reasoning steps’ correctness or conciseness. We extract only the numerical answer and calculate accuracy through exact matches with the ground truth. We also use other common parsing methods, such as matching “`The answer is <value>`”, to allow slight format changes. Our code contains the exact parsing function.
- For HMathA, we expect answers inside a `\boxed{}` element and parse them accordingly.
- For SciQ, we create ground truth responses in a format similar to GSM8K: “`<support text> #### <text answer>`”. We parse responses using this format and count only exact string matches as correct.

**Training Setup:** We test each training method across all tasks using two model families. We use Qwen3-4B-Base for every single-token and multi-token task. We also use Llama-3.2-1B for every single-token task except SciQr, where we use Llama-3.2-3B. We make this exception because the 1B Llama model could not improve above the random baseline for any \(N\) up to 128.

For multi-token tasks, our second model is Llama-3.2-1B-Instruct. Using an Instruct model tests our method on post-trained models, which have received additional training after base-model training.

We train with low-rank adaptation (LoRA), which updates weights through low-rank matrices. We use rank 8 to modify the \(W_Q\), \(W_K\), and \(W_O\) matrices in self-attention layers. We chose these models and weight combinations based reasonably on available resources. In total, we trained more than 13,000 models.

Multi-token training uses many more resources because each sample has a long answer. We therefore use only \(N\) up to 16 for Llama and up to 8 for Qwen in this setting. For efficiency, we also cap generated tokens during training at \(G=200\). IA2 uses this cap when collecting the activation tensor

\[
\hat{A}_i\in\mathbb{R}^{L\times 200\times d}
\quad\text{for every }x_i.
\]

For SFT, \(G\) equals the ground truth answer’s token count.

We train each method–dataset combination until convergence, for up to 50 epochs, or passes through the training data. We use 3 learning rates: slow, 1e-4; medium, 3e-4; and fast, 1e-3. We select the best-performing rate to neutralize hyperparameter selection’s effect on method performance. Hyperparameters are training settings chosen outside the learned model weights. Section B gives additional details.

**Evaluation:** For single-token tasks, we measure accuracy and expected calibration error (ECE). Because each answer contains one token, we calculate both metrics from the first generated token’s probability distribution.

We first evaluate the base model with ICL demonstrations to establish the ICL benchmark. We evaluate each task’s validation set 5 times with different ICL demonstrations and report average metrics. We then evaluate each training method without ICL demonstrations and report its best-performing average metrics.

For multi-token tasks, we limit generated tokens during evaluation to use computation carefully. We set the limit to 200 for GSM8K and GSM8Ks, based on their 95th-percentile answer lengths of 184 and 154, respectively. For HMathA, we raise the limit to 400 because its 95th-percentile answer length is 307.

We measure only accuracy: how many questions the model answers correctly in the required format. Section A gives answer-parsing and other details. We do not measure calibration because confidence is hard to measure for open-ended generation.

### 5.1 IA2 priming boosts SFT performance

**Single-Token results:** Table 1 reports adapted Qwen models’ performance on single-token tasks in the practical setting of \(N=4\). Our IA2 → SFT method outperforms SFT-only training in accuracy and calibration in almost all cases, using the same amount of data. In most cases, it also outperforms ICL in accuracy, with slightly worse calibration.

IA2-only training reaches high accuracy in many cases and remains reasonably calibrated, like ICL, without training on target response tokens. This highlights the richness of the IA2 training signal. Table 1 shows one setup; Figure 1 captures aggregate trends across models and datasets. Section C gives detailed tables for other model/\(N\) combinations and a statistical significance test of our improvement.

In the tables, “accuracy ↑” means higher accuracy is better, and “ECE ↓” means lower expected calibration error is better.

| Dataset: Source | Dataset: Eval | ICL accuracy ↑ | ICL ECE ↓ | SFT only accuracy ↑ | SFT only ECE ↓ | IA2 only accuracy ↑ | IA2 only ECE ↓ | IA2 → SFT accuracy ↑ | IA2 → SFT ECE ↓ |
|---|---|---|---|---|---|---|---|---|---|
| AGN | AGN | 30.0 (02.3) | 0.13 (0.03) | 27.9 (04.3) | 0.52 (0.12) | 24.0 (00.3) | 0.11 (0.01) | 31.8 (03.8) | 0.35 (0.24) |
| | BBCN∗ | 28.8 (00.6) | 0.12 (0.01) | 31.8 (07.1) | 0.56 (0.14) | 24.3 (01.3) | 0.10 (0.02) | 27.9 (04.3) | 0.40 (0.21) |
| FinS | FinS | 63.6 (01.5) | 0.12 (0.01) | 67.4 (14.1) | 0.31 (0.14) | 63.1 (15.0) | 0.24 (0.06) | 78.7 (13.7) | 0.16 (0.11) |
| | PoemS∗ | 56.9 (02.2) | 0.12 (0.02) | 49.9 (04.6) | 0.48 (0.04) | 52.8 (05.6) | 0.15 (0.04) | 60.6 (15.7) | 0.36 (0.15) |
| | SST2∗ | 70.1 (01.3) | 0.17 (0.01) | 59.4 (06.5) | 0.20 (0.10) | 69.8 (13.7) | 0.30 (0.10) | 71.4 (18.9) | 0.20 (0.15) |
| SciQr | QASCr∗ | 56.5 (01.4) | 0.08 (0.01) | 76.5 (03.8) | 0.09 (0.05) | 71.2 (03.0) | 0.12 (0.01) | 79.4 (01.8) | 0.09 (0.01) |
| | SciQr | 87.7 (01.1) | 0.07 (0.01) | 88.3 (04.6) | 0.07 (0.05) | 90.4 (01.5) | 0.09 (0.01) | 91.7 (01.7) | 0.05 (0.01) |
| SST2 | FinS∗ | 41.9 (00.3) | 0.19 (0.01) | 68.4 (02.3) | 0.30 (0.04) | 71.3 (02.1) | 0.21 (0.03) | 82.4 (12.2) | 0.11 (0.08) |
| | PoemS∗ | 65.1 (01.2) | 0.11 (0.02) | 56.5 (13.2) | 0.33 (0.14) | 62.4 (09.2) | 0.19 (0.06) | 68.4 (08.4) | 0.30 (0.08) |
| | SST2 | 85.4 (00.4) | 0.13 (0.00) | 65.2 (10.7) | 0.22 (0.10) | 82.7 (06.4) | 0.28 (0.03) | 90.4 (02.3) | 0.06 (0.01) |
| STF | STF | 66.0 (02.1) | 0.07 (0.01) | 52.4 (04.4) | 0.29 (0.10) | 62.2 (04.1) | 0.16 (0.04) | 62.4 (04.3) | 0.29 (0.08) |

**Table 1:** Performance for \(N=4\) on Qwen3-4B-Base, showing accuracy and expected calibration error. Parentheses show standard deviations, which measure variation, across 5 runs for the best-performing learning rate. Bold indicates the best training method. The symbol ∗ marks OOD evaluations. Our proposed IA2 → SFT method outperforms standard SFT across the board.

| Dataset: Source | Dataset: Eval | Without ICL accuracy ↑ | ICL accuracy ↑ | SFT only accuracy ↑ | IA2 only accuracy ↑ | IA2 → SFT accuracy ↑ |
|---|---|---|---|---|---|---|
| GSM8K | GSM8K | 56.4 (00.0) | 76.4 (01.1) | 70.9 (02.8) | 77.4 (02.8) | 73.6 (03.7) |
| | GSM8Ks∗ | 45.4 (00.0) | 68.4 (01.1) | 64.5 (04.0) | 66.2 (03.7) | 68.8 (05.1) |
| HMathA | HMathA | 21.0 (00.0) | 60.4 (02.1) | 50.4 (01.8) | 47.8 (06.0) | 55.3 (02.1) |
| SciQ | SciQ | 15.4 (00.0) | 37.5 (01.0) | 35.0 (07.1) | 06.9 (13.8) | 40.8 (07.6) |

**Table 2:** Performance for \(N=4\) on Qwen3-4B-Base models on multi-token datasets. Bold indicates the best training method. IA2 → SFT outperforms standard SFT on all datasets.

**Multi-Token results:** Table 2 reports adapted Qwen models’ accuracy in the multi-token setting. As in the single-token setting, IA2 → SFT outperforms SFT-only training in every case. Compared with the single-token setting, it is noteworthy that IA2 → SFT sometimes performs worse than ICL.

We suspect that the same small LoRA weight space cannot perfectly compress longer contexts. Larger ranks should improve this. Section C provides more tables for other model/\(N\) combinations.

### 5.2 Why is IA2 priming important?

**Figure 3:** Scatter plots of \(\operatorname{asim}(A,A_{\text{ICL}})\) against accuracy/ECE for all training methods show how activation similarity affects performance metrics. Each point represents one training experiment. ECE falls smoothly as similarity increases. However, extreme alignment with ICL activations may miss some accuracy gains that SFT on ground truth responses can provide.

We investigate properties of three training methods: 1) SFT only, 2) IA2 only, and 3) IA2 → SFT. We make the following observations.

**Activation Similarity vs Performance:** Figure 2 shows all 3 methods’ activation similarity with ICL. As expected, IA2 drastically increases this similarity. Importantly, IA2 → SFT retains higher similarity with ICL activations even after SFT training.

Figure 3 shows the relationship between ICL activation similarity and performance. IA2 → SFT lies between IA2 and SFT in activation similarity while achieving better accuracy and calibration. SFT-only and IA2 → SFT training differ only in their starting state. This implies that IA2 supplies a rich training signal that SFT-only training cannot provide.

Greater activation similarity clearly implies better calibration. However, extreme similarity may not yield the best accuracy because the ICL signal is not always correct. In §C, we show one case where IA2 models follow ICL performance even when it worsens because of overfitting.

It is therefore best to combine both training signals. IA2 improves internal functional alignment with ICL. SFT on ground truth responses aligns the model with expected output behavior.

**Subspace Overlap:** Activation patterns reflect the model’s functional behavior, but they do not isolate changes caused by different training methods. We use LoRA adapters, and unchanged base-model weights strongly influence activation patterns. To isolate behavioral changes, we analyze LoRA weights directly.

**Figure 4:** IA2 → SFT models have significantly high subspace overlap with IA2-only models, while SFT-only models align with neither. Subspace overlap measures how much of the weight space two models share. This indicates that IA2 provides a training signal absent from SFT-only training and important for performance.

Figure 4 shows the distribution of weight subspace overlap between pairs of models trained with different methods. We use models from all single-token and multi-token experiments.

We first apply singular value decomposition (SVD) to the weight matrices to find basis vectors, which describe the directions spanning each subspace. We then calculate overlap using:

\[
\frac{1}{r}*\left\|U_1^T\cdot U_2\right\|_F^2.
\]

Here, \(U_1\) and \(U_2\) contain the basis vectors for the two methods, and \(r\) is the LoRA weights’ rank. The subscript \(F\) denotes the Frobenius norm, a measure of matrix size. Subspace overlap measures how much weight space the methods share. It thus isolates and compares differences in their functional behavior.

SFT-only training produces weight updates almost completely orthogonal to those of the other two methods. Orthogonal updates point in perpendicular directions. In contrast, the best-performing IA2 → SFT models share around \(39\%\) of their spanned weight space, on average, with IA2-only models.

This implies that IA2 accounts for much of IA2 → SFT models’ performance gains. It also implies that SFT-only training cannot practically reach the subspaces IA2 identifies.

Overlap between IA2-only and IA2 → SFT models varies widely. In many cases, the IA2 model already had aligned outputs. Further SFT therefore changed little before convergence, producing very high overlap.

## 6 Discussion

**IA2 + SFT:** A natural alternative to sequential IA2 → SFT training is IA2 + SFT: using \(\mathcal{L}_{\text{IA2}}\) and \(\mathcal{L}_{\text{SFT}}\) at the same time. This creates a practical challenge. For any \(G>1\), the target activation tensor \(\hat{A}\) uses the model’s own generated response tokens. We call this the ICL response. It can differ greatly from the dataset’s ground truth response, even in length. This makes the two objectives incompatible for joint training.

However, we can perform self-distillation using ICL responses, as in Snell et al. (2022), while adding an IA2 signal. We train with the combined loss

\[
\mathcal{L}_{\text{IA2}}+\beta\cdot\mathcal{L}_{\text{SFT}},
\]

where \(\beta\) controls the SFT loss’s contribution. We vary \(\beta\) across 4 values that smoothly span the range between IA2-only and SFT-only training. Section B gives details.

We select the best-performing learning-rate/\(\beta\) combination across 5 random seeds, which control random choices in the experiments. Table 3 reports multi-token performance. SFT-only and IA2 → SFT performance falls compared with training on ground truth responses in Table 2. However, IA2 + SFT extracts significantly more performance from ICL responses than SFT-only training. This highlights the richness of the IA2 signal. Section C gives additional IA2 + SFT results.

| Dataset: Source | Dataset: Eval | Without ICL accuracy ↑ | ICL accuracy ↑ | SFT only accuracy ↑ | IA2 only accuracy ↑ | IA2 → SFT accuracy ↑ | IA2 + SFT accuracy ↑ |
|---|---|---|---|---|---|---|---|
| GSM8K | GSM8K | 56.4 (00.0) | 76.4 (01.1) | 66.4 (18.1) | 77.4 (02.8) | 67.2 (22.1) | 77.0 (07.8) |
| | GSM8Ks∗ | 45.4 (00.0) | 68.4 (01.1) | 59.8 (20.2) | 66.2 (03.7) | 61.6 (24.0) | 69.0 (09.6) |
| HMathA | HMathA | 21.0 (00.0) | 60.4 (02.1) | 49.0 (03.0) | 47.8 (06.0) | 54.0 (01.7) | 53.6 (02.7) |
| SciQ | SciQ | 15.4 (00.0) | 37.5 (01.0) | 27.2 (12.0) | 06.9 (13.8) | 32.8 (10.0) | 34.5 (10.6) |

**Table 3:** Performance for \(N=4\) on Qwen3-4B-Base models trained with ICL responses rather than ground truth responses. IA2 + SFT performs significantly better than SFT only, highlighting IA2’s importance.

**Knowledge Distillation baseline:** Knowledge distillation trains a student model to reproduce a stronger teacher model’s behavior. A popular method uses soft label matching (Hinton et al., 2015). Instead of supplying only a target token, the teacher supplies probabilities for each token. This provides a denser training signal.

Here, we use the same model, enhanced with ICL context, as the teacher. To compare IA2-based training with soft label matching, we train and evaluate Llama models on GSM8K using 1B-Instruct and on SST2 using 1B. We use soft label matching on ICL responses.

Table 4 shows that soft label matching performs significantly better than standard SFT in the multi-token case, nearly reaching IA2 + SFT performance. However, it falls short in the single-token case. This highlights IA2-based training’s consistency in achieving high performance.

| Dataset | \(N\) | ICL accuracy ↑ | ICL ECE ↓ | SFT only accuracy ↑ | SFT only ECE ↓ | SFT (soft labels) accuracy ↑ | SFT (soft labels) ECE ↓ | IA2 only accuracy ↑ | IA2 only ECE ↓ | IA2 → SFT accuracy ↑ | IA2 → SFT ECE ↓ | IA2 + SFT accuracy ↑ | IA2 + SFT ECE ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GSM8K | 4 | 34.6 (00.9) | – | 28.2 (04.5) | – | 36.1 (03.5) | – | 29.3 (05.0) | – | 30.3 (03.2) | – | 37.0 (01.9) | – |
| SST2 | 8 | 77.0 (02.2) | 0.10 (0.02) | 52.3 (04.3) | 0.38 (0.14) | 55.9 (08.8) | 0.16 (0.06) | 64.9 (15.3) | 0.30 (0.05) | 64.7 (19.3) | 0.33 (0.19) | 66.9 (18.4) | 0.28 (0.21) |

**Table 4:** Comparison with the Knowledge Distillation (KD) baseline, SFT on soft labels. Although KD outperforms standard SFT in the multi-token case, IA2-based training consistently provides high performance in both single-token and multi-token cases.

**Why IA2 based training does not consistently beat ICL?** In some multi-token cases, ICL outperforms every training method, including IA2-based methods. Examples include Qwen on the math datasets GSM8K and HMathA.

We suspect that Qwen’s mid-training on science, technology, engineering, and mathematics (STEM) data makes ICL extremely sample-efficient. Surprisingly, GSM8K ICL performance at \(N=2\) exceeds performance at \(N=4,8\).

Table 5 shows how performance changes with shot count, the number of examples. It also shows the gap between ICL and IA2 → SFT, which closes quickly with more shots. The final column supports our consistent key takeaway: IA2 → SFT performs better than SFT only.

| Dataset | \(N\) | ICL | SFT only | IA2 only | IA2 → SFT | Gap with respect to ICL | Gap with respect to SFT only |
|---|---|---|---|---|---|---|---|
| GSM8K | 2 | 81.2 (00.4) | 65.2 (01.0) | 70.7 (01.8) | 74.6 (02.4) | -06.6 | +09.4 |
| | 4 | 76.4 (01.1) | 70.9 (02.8) | 77.4 (02.8) | 73.6 (03.7) | -02.8 | +02.7 |
| | 8 | 77.2 (00.9) | 73.5 (01.6) | 78.8 (02.5) | 76.0 (02.0) | -01.2 | +02.5 |
| HMathA | 2 | 58.6 (02.0) | 47.8 (04.0) | 33.9 (06.4) | 48.7 (07.0) | -09.9 | +00.9 |
| | 4 | 60.4 (02.1) | 50.4 (01.8) | 47.8 (06.0) | 55.3 (02.1) | -05.1 | +04.9 |
| | 8 | 62.1 (01.0) | 52.0 (00.8) | 57.3 (03.7) | 59.4 (00.8) | -02.7 | +07.4 |

**Table 5:** For Qwen models on math datasets, ICL is more sample-efficient than training methods. However, IA2 → SFT quickly closes the performance gap as the number of shots increases and consistently beats SFT-only training.

**Other fine-tuning methods:** LoRA is not the only way to fine-tune models. Alternatives include full fine-tuning at full rank and other parameter-efficient methods. Prompt and prefix tuning adapt models through trainable prompt or prefix representations (Lester et al., 2021; Li and Liang, 2021). Activation scaling, used by (IA)3, adapts models by scaling activations (Liu et al., 2022).

Our pipeline does not compete with these methods. It improves them by providing better training signals. To establish this, we repeat experiments on several datasets using the popular PEFT method (IA)3. We follow exactly the methodology in §5, replacing LoRA training with (IA)3. Section B gives setup details.

Table 6 reports one setting and shows trends similar to LoRA. Section C provides other results. We leave tests of IA2’s improvements to other SFT methods for future work. As a side note, IA2 also marks a major improvement in method naming compared with (IA)3.

| Dataset: Source | Dataset: Eval | ICL accuracy ↑ | ICL ECE ↓ | SFT only accuracy ↑ | SFT only ECE ↓ | IA2 only accuracy ↑ | IA2 only ECE ↓ | IA2 → SFT accuracy ↑ | IA2 → SFT ECE ↓ |
|---|---|---|---|---|---|---|---|---|---|
| SciQr | QASCr∗ | 35.6 (01.4) | 0.06 (0.01) | 32.8 (03.4) | 0.20 (0.08) | 34.2 (02.7) | 0.08 (0.01) | 48.6 (05.5) | 0.25 (0.11) |
| | SciQr | 60.0 (01.8) | 0.07 (0.01) | 64.5 (05.0) | 0.12 (0.05) | 63.0 (04.6) | 0.10 (0.04) | 70.6 (04.6) | 0.08 (0.02) |
| SST2 | FinS∗ | 56.6 (01.6) | 0.17 (0.01) | 58.2 (10.7) | 0.25 (0.03) | 61.9 (09.5) | 0.18 (0.06) | 59.8 (07.8) | 0.18 (0.03) |
| | PoemS∗ | 52.3 (03.0) | 0.19 (0.02) | 48.1 (03.2) | 0.44 (0.11) | 48.1 (03.4) | 0.11 (0.04) | 59.8 (13.4) | 0.33 (0.13) |
| | SST2 | 60.3 (02.8) | 0.19 (0.02) | 52.4 (02.8) | 0.35 (0.05) | 54.2 (04.2) | 0.15 (0.06) | 62.6 (09.1) | 0.28 (0.11) |

**Table 6:** Performance for \(N=4\) on Llama-3.2 models trained with (IA)3. IA2 → SFT significantly improves on SFT-only training and exceeds even ICL accuracy in every case.

**Computational Overhead:** Before training, IA2 → SFT requires an activation-collection step. We run ICL inference on the model using the given dataset. This one-time cost collects rich ICL activations that greatly improve the model’s capabilities. During inference, the trained model runs at the same cost as an SFT-only model.

**Potential improvements:** Resource limits forced us to fix several design choices that could support further study and improvement. The most important include ablations—tests that vary individual choices—of LoRA rank, target modules, and other parameters, along with selective IA2.

We showed that extreme alignment can hurt performance. Prior work also shows that some layers are more sensitive to ICL than others (Todd et al., 2023). Selectively aligning layers may therefore work better.

A smaller improvement could come from optimizing prompts before collecting target activations. ICL performance depends on demonstration order (Lu et al., 2022; Zhao et al., 2021).

Finally, we use fixed learning rates for IA2 → SFT. Finer control could benefit the two training signals. We leave these studies for future work.

## 7 Related Work

**ICL vs SFT:** Prior work has studied performance differences between ICL and SFT, but the verdict remains unclear (Mosbach et al., 2023; Duan et al., 2024). In our work, SFT performed worse than ICL. Other studies examine functional similarities (Dai et al., 2022) and differences (Wang et al., 2023) between them.

Some studies show how ICL or SFT adaptation can become miscalibrated and how to improve calibration (Zhang et al., 2024; Li et al., 2025; Li et al., 2024; Xiao et al., 2025). Sia et al. (2024) use context masking, which hides parts of the context, to identify “where” translation occurs in LLMs during ICL. Like us, they use activations as a signal of functional behavior.

Importantly, recent work shows that SFT is prone to memorization, overfitting, and catastrophic forgetting—the loss of previously learned behavior (Chu et al., 2025; Shenfeld et al., 2025). Our improved SFT pipeline aims to address these problems.

**Context Distillation:** Like our work, many studies aim to improve LLM adaptation by compressing explicit context signals. Some use intermediate contemplation tokens for compression (Cheng and Van Durme, 2024; Jiang et al., 2025). Others internalize context by generating adapters as needed (Chen et al., 2025; Charakorn et al., 2025), use related text to improve distillation (Zhu et al., 2025; Choi et al., 2025), or compress context directly into weights (Shen et al., 2025; Deng et al., 2024; Yu et al., 2024; Snell et al., 2022; Chen et al., 2024b; Shin et al., 2024).

These studies use response-token signals to distill contexts’ effects on model outputs. Our work complements them. It improves distillation through the model’s own ICL processing signals. Prior work supports this idea (Aguilar et al., 2020; Jin et al., 2024; Yang et al., 2024).

**Activation Steering:** A growing line of work extracts “steering” vectors or weights and applies them at targeted locations inside a model to create an intended output effect (Postmus and Abreu, 2024; Stolfo et al., 2025; Caccia et al., 2025; Fleshman and Van Durme, 2024; Arditi et al., 2024).

Like our method, these approaches intervene in activation space to influence how the model functions. However, they use targeted interventions within a small application domain. Our method instead aims to induce a broad behavioral change that avoids potential brittleness and reliability problems (Queiroz Da Silva et al., 2025).

## 8 Conclusion and Future Work

We used model activations to study LLMs’ functional behavior and found a key distinction. ICL and SFT adapt models through different mechanisms, and ICL often encodes richer, more generalizable patterns.

This finding motivated IA2, an SFT priming method that enables ICL-like internal behavior. This simple step moves models into a more adaptable weight subspace that SFT alone cannot reach. Extensive experiments show that IA2 consistently improves SFT accuracy and calibration.

In future work, we aim to test other effects of IA2-based SFT, including effects on diversity and catastrophic forgetting. These tests will help measure how IA2 can support LLM post-training. We also aim to study IA2’s effects on post-trained models tuned through reinforcement learning (RL).

## Reproducibility Statement

We provide code on GitHub for data download and processing, training, evaluation, analysis, plotting, and other steps. The details in this paper (§5 and the appendix), together with a detailed README file, should suffice to reproduce our results fully. We used repeatable random seeds.

## Acknowledgments

This work is supported by ONR (N00014-24-1-2089), an Amazon Research Award, and a grant from Open Philanthropy. We thank Andrea Wynn for their constructive feedback on earlier versions of this document.
