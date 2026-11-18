# IA2: Alignment with ICL Activations

Improves Supervised Fine-Tuning

Aayush Mishra

  
Daniel Khashabi & Anqi LiuDepartment of Computer Science,
Johns Hopkins University

###### Abstract

Supervised Fine-Tuning (SFT) specializes model behavior by training model weights to produce intended target responses for queries. In contrast, In-Context Learning (ICL) adapts models during inference by providing instructions or demonstrations in the prompt, without updating model weights. In data-scarce settings, ICL can offer better generalizability and more calibrated responses than SFT, but ICL requires more inference compute. These contrasting properties motivate our question: Can ICL’s internal computations be used to improve the qualities of SFT?

We first show that ICL and SFT produce distinct activation patterns. Because activations reflect a model’s internal processing, these distinct patterns indicate that ICL and SFT achieve adaptation through different functional mechanisms. To use ICL’s rich functionality, we introduce ICL Activation Alignment (IA2), a self-distillation technique that aims to replicate ICL’s activation patterns in SFT models and thereby incentivize ICL-like internal reasoning. Our empirical results on 12 popular benchmarks and two model families show that performing IA2 as a priming step before SFT significantly improves the accuracy and calibration of model outputs. Thus, IA2 is practically useful and also offers a conceptual window into the inner mechanics of model adaptation.

00footnotetext: ♡Equal advising. Correspondence to: {amishr24,danielk,aliu.cs}@jhu.edu

## 1 Introduction

Large language models (LLMs) are general-purpose models, but LLMs are often used in specialized applications. For example, a news aggregator app may need to classify articles into predefined categories. A popular approach for adapting LLMs to such applications is Supervised Fine-Tuning (SFT), which uses a dataset of labeled samples to train and adapt model weights on narrow downstream tasks. With parameter-efficient fine-tuning (PEFT) techniques (Hu et al., 2021; Liu et al., 2022), SFT models are often just a small set of parameters that can be efficiently loaded into or removed from graphics processing unit (GPU) memory. This small parameter footprint makes SFT models extremely useful. However, SFT typically requires a large set of labeled samples to generalize well on new tasks (Le Scao and Rush, 2021), and collecting those labeled samples can be expensive.

In contrast, In-Context Learning (ICL) (Brown et al., 2020) adapts and steers LLM behavior during inference. The model receives a query preceded by in-context demonstrations or instructions, which help the model “learn” the demonstrated task and answer accordingly. Prior work (Duan et al., 2024) shows—and our experiments in §5 concur—that ICL generalizes well in a few-shot setting and typically produces well-calibrated responses. However, ICL demonstrations and instructions occupy valuable context space and increase the cost of running each query. Without those demonstrations or instructions, the same context space could instead be used for more query and response tokens.

These differences motivate us to investigate whether ICL and SFT also differ in their functional behavior. We find that, although the methods’ token-space behavior may look similar on the surface, models produce disparate internal activations under ICL and SFT (§3). Activations are a footprint of the model’s internal processing before the model produces the next token. Therefore, divergent activation patterns highlight a difference in how ICL and SFT achieve adaptation, an idea also supported by prior work (Shen et al., 2024).

We hypothesize that, when a model performs ICL, the model’s activations contain rich information about how to extract generalizable patterns from the context. This information may be absent in SFT models, especially in a few-shot setting, where SFT models are prone to shortcut learning or overfitting on target responses. Hence, we ask:

**Can the information-rich ICL activations be used to improve the quality of SFT?**

Recent works have tried to distill context into LLM weights (Snell et al., 2022; Chen et al., 2024b). However, those methods use training signals only from response texts and may therefore suffer from the same SFT issues described above. In particular, training a model to reproduce the outputs of an ICL-conditioned model does not ensure that the trained model functions like an ICL-conditioned model.

To address this distinction between output behavior and functional behavior, we propose ICL Activation Alignment (IA2), a self-distillation method that enforces alignment with a model’s own functional behavior when performing ICL (§4). Figure 1 outlines the proposed pipeline: (1) collect information-rich ICL activations, (2) enforce functional alignment with ICL, and then (3) perform SFT on the primed model. Using the same data, priming models with IA2 before SFT drastically improves the adapted models’ performance on a variety of text classification and generation tasks (§5). We also show that IA2 provides an important training signal unavailable through SFT-only training. We trained over 13,000 models spanning 12 benchmarks to validate these findings. The findings demonstrate practical benefits and offer a conceptual window into the inner mechanics of model adaptation.

In summary:

- $\star$ We show that ICL and SFT, even with the same data, do not align in the model’s activation space. This lack of alignment highlights a gap in their functional behavior (§3).
- $\star$ We propose IA2—ICL Activation Alignment—to enforce alignment with ICL’s functional behavior. This priming step drastically improves SFT model performance (Figure 1, §4).
- $\star$ We show that the IA2 training signal is not present in SFT-only training, highlighting the importance of this signal for improving adaptation quality (§5).

**Figure 1:** An overview of our improved SFT pipeline. Standard SFT enforces only output-space alignment: alignment between a model’s response and a target response, corresponding to only step 3 above. In low-data settings, this output-only alignment results in subpar performance and mis-calibration. In contrast, IA2 enforces functional alignment with ICL by matching the rich activation patterns produced during ICL. Performing IA2 priming before SFT boosts adaptation quality. The performance comparison charts show this improvement aggregated across models and datasets. See §5 for details.

## 2 Background and Notation

**Transformer Language Models:** A standard decoder-only transformer-based language model (LM), denoted $M_{\Theta}$, has parameters $\Theta$ and consists of a stack of self-attention (SA) and linear layers. Let $T=[t_i]_{i=1}^{R}$ be a sequence of $R$ token vectors, where $t_i$ is the vector at token position $i$ and $d$ is the token-vector dimension. The model applies self-attention and linear projections to each token at each layer. The last token’s projection at the final layer then passes through the LM head to predict the most likely next token.

Self-attention is special because the output for a token is affected by other tokens in the sequence. A standard self-attention operation uses four weight matrices, $W_Q$, $W_K$, $W_V$, and $W_O$, all belonging to $\Theta$. We denote this set of matrices by $W_{QKVO}$. The output at each token position is:

$$
\text{SA}(T;W_{QKVO})=Z=
\left[
z_i=\sigma\left(\frac{q_iK^T}{\sqrt{d}}\right)V\cdot W_O
\right]_{i=1}^{R}.
\tag{1}
$$

Here, $\sigma(.)$ denotes Softmax; $q_i=t_i\cdot W_Q$ is the query vector; $K=t_{:i}\cdot W_K$ and $V=t_{:i}\cdot W_V$ are the key and value representations for the first $i$ tokens; and $t_{:i}$ denotes those first $i$ tokens. The superscript $T$ denotes a transpose. The output $Z$ has the same shape, $R\times d$, as the input $T$. In this work, we study how model behavior changes when either the context $T$ or the self-attention weights $W_{QKVO}$ change.

**Supervised Fine-Tuning:** Let $\mathcal{T}$ denote a task, and let $\mathcal{D}_{\mathcal{T}}=\{(X_i,Y_i)\}_{i=1}^{N}$ be a dataset of $N$ input–output pairs illustrating that task. Here, $X_i$ is an input and $Y_i$ is its ground-truth output. SFT updates the model weights from $\Theta$ to $\Theta^{\prime}$ so that the model produces relevant responses for new inputs. For a new input $X_t$,

$$
M_{\Theta^{\prime}}(X_t)=\bar y_t,
$$

where $\bar y_t$ is the first token of the complete generated response $\bar Y_t$. Repeated application of $M_{\Theta^{\prime}}$ to the growing sequence extends this response.

For training, suppose the ground-truth response $Y_i$ contains $G$ tokens. We generate $G$ new tokens to obtain $\bar Y_i$ and minimize cross-entropy over all generated tokens. With $j$ indexing response-token positions, the SFT loss is:

$$
\mathcal{L}_{\text{SFT}}
=
\sum_{i=1}^{N}\sum_{j=1}^{G}
\text{cross-entropy}(\bar Y_{ij},Y_{ij}).
\tag{2}
$$

**In-Context Learning:** The same dataset $\mathcal{D}_{\mathcal{T}}$ can also be used without weight updates. Let $\circ$ denote sequence concatenation, and let

$$
I=[X_1\circ Y_1\circ\ldots\circ X_n\circ Y_n]
$$

be the concatenated tokens of $n$ demonstrations. The model receives these demonstration tokens followed by a new test input $X_t$. By processing the examples in its prompt, the model understands the task and produces a task-appropriate response:

$$
M_{\Theta}(I\circ X_t)=\hat y_t.
$$

Here, $\hat y_t$ is the first token of the ICL-generated response $\hat Y_t$. Remarkably, the complete response $\hat Y_t$ is often similar to the expected response $Y_t$. Therefore, ICL is a useful inference-time adaptation method. ICL was first illustrated in GPT-3 (Brown et al., 2020) and is widely used to adapt generic language models at inference time. Today’s frontier models can follow instructions directly, so a generic instruction can also be considered a form of ICL demonstration. In this work, however, we focus on classic ICL with a demonstration set of input–output pairs.

**Activations:** Because self-attention uses every token in the context, different token sequences produce different self-attention outputs. We call these hidden self-attention outputs *activations*. Activations are a footprint of the model’s internal processing and can therefore be used to study how model behavior changes with context.

If $M_{\Theta}$ has $L$ layers and processes $R$ tokens, the activation tensor $A$ has size $L\times R\times d$. We study token sequences $T=[I\circ X]$ for ICL and $T=X$ without ICL under different model weights. In these expressions, $I$ contains demonstration tokens and $X$ contains query tokens.

We define *activation similarity*, written $\operatorname{asim}(A_1,A_2)$, as token-wise cosine similarity between two activation tensors $A_1$ and $A_2$. The resulting activation-similarity tensor has size $L\times R$.

## 3 Are ICL and SFT practically the same?

**Figure 2: Layerwise Similarity with ICL Activations.** SFT models have small activation similarity, $\operatorname{asim}$, with ICL, indicating different functional behavior for adaptation. Our recommended pipeline, IA2 → SFT, aligns much more closely with ICL than SFT-only training and performs much better as a result (see Figure 1).

Recent studies (Von Oswald et al., 2023; Akyürek et al., 2022) claim that ICL in transformer models works through an internal gradient-descent mechanism. These studies show that self-attention can produce outputs as if model parameters had been updated using the ICL demonstrations in context. This finding implies a functional equivalence between ICL and parameter updates. If that equivalence holds, a base model with ICL demonstrations should produce activations at output-token positions similar to those of an SFT model without ICL demonstrations. We investigated whether LLMs actually exhibit this predicted similarity.

**ICL and SFT produce different activations:** We collected ICL activation tensors, denoted $A_{\text{ICL}}$, using $T=[I\circ X]$ for 100 test samples from multiple datasets, including the SST2 sentiment-classification dataset and the AGNews news-categorization dataset. We used models from the Qwen3 (Yang et al., 2025) and Llama-3.2 (Grattafiori et al., 2024) families, with the following model-size and layer-count correspondences: 1B $\leftrightarrow$ 16 layers, 3B $\leftrightarrow$ 28 layers, and 4B $\leftrightarrow$ 36 layers, where B denotes billions of parameters.

We then trained SFT models on the same demonstrations until convergence and collected the SFT activation tensors, denoted $A_{\text{SFT}}$, using $T=X$. Experiment details appear in §B. We calculated $\operatorname{asim}(A_{\text{ICL}},A_{\text{SFT}})$ at output-token positions and averaged the similarity across tokens, samples, and datasets. The “SFT only” plots in the first row of Figure 2 show these averages.

Across the different models, ICL and SFT activations are not aligned. However, activations align better near the initial and final layers, where we expect tokens to be processed at an individual level. In contrast, activations are misaligned in the middle layers, where we expect the whole demonstration set to be processed at an abstract level.

**Why should we care about alignment with ICL?** The functional difference between ICL and SFT is not confined to hidden activation patterns: the difference also appears in expected calibration error (ECE) (Guo et al., 2017), as shown in Figure 1. At similar—and slightly better—accuracy levels, ICL responses are much more calibrated than SFT responses.

We suspect that SFT’s output-oriented training signal allows SFT models to learn shortcuts that can fail on new data. In contrast, ICL relies on complex circuits (Elhage et al., 2021) that must work across a variety of tasks. We therefore suspect that ICL extracts generalizable patterns from demonstrations and performs in a well-calibrated manner.

This experiment demonstrates that ICL and SFT achieve adaptation through different means. Consequently, SFT should not be expected to function as a drop-in replacement for ICL. These differences motivate the research question:

**Can the information-rich ICL activations be used to improve the quality of SFT?**

Next, we show that ICL activations can improve SFT and describe our proposed SFT procedure.

## 4 Inducing ICL-like behavior in SFT Models

Our SFT pipeline has two steps. First, IA2 primes the model by incentivizing ICL-like functional behavior. Then, further SFT on the ICL-aligned weights improves adaptation qualities.

**IA2—ICL Activation Alignment:** In §3, SFT models had little activation similarity with ICL. To increase activation similarity, and hence functional similarity with ICL, we define the following goal for each layer of $M_{\Theta}$. Find modified self-attention weights

$$
\tilde W_{QKVO}=(\tilde W_Q,\tilde W_K,\tilde W_V,\tilde W_O)
$$

such that, for every newly generated token at the last position, indexed by $-1$,

$$
\text{SA}([I\circ X];W_{QKVO})
\approx
\text{SA}(X;\tilde W_{QKVO})
\quad \forall X\in\mathcal T.
\tag{3}
$$

Thus, the goal compares self-attention under the original weights with demonstrations in context against self-attention under modified weights with only the query in context.

Prior work (Chen et al., 2024a) provides a closed-form solution to this goal for linearized attention models using a kernel approximation to Softmax. In contrast, we seek a practical general solution for real, non-linear transformers. Our goal is to achieve ICL’s qualities directly in the SFT model’s weights. The modified weights are not incentivized to produce similar outputs; instead, the modified weights are incentivized to process every input query similarly to ICL at each layer of the base model.

To perform IA2, we first generate responses using ICL. For every training input $X_i$, we use $T_i=[I\circ X_i]$ to obtain an ICL response $\hat Y_i$. For each input, we construct the demonstration sequence $I$ from the remaining samples in the dataset. We collect activations at each output position, yielding a target tensor $A_{\text{ICL}}^i\in\mathbb R^{L\times G\times d}$ when the response has $G$ tokens.

Next, we attach the ICL-generated response to the query without demonstrations, giving $T_i=[X_i\circ\hat Y_i]$. This sequence treats the model as if the model had produced the ICL response with only the query in context. Processing this sequence gives an unaligned activation tensor $A^i$ at the output-token positions. We then train the model parameters to minimize the mean squared error between the unaligned activations and the target ICL activations:

$$
\mathcal{L}_{\textbf{IA2}}
=
\sum_{i=1}^{N}\left\|A^i-A_{\text{ICL}}^i\right\|.
\tag{4}
$$

**IA2 → SFT:** After IA2, we continue training with the standard SFT cross-entropy loss on target tokens (Equation 2). Suppose IA2 updates the parameters from $\Theta$ to $\Theta^{\prime}$. We obtain responses $\bar Y_i$ for every training input $X_i$ using $M_{\Theta^{\prime}}$ and minimize the SFT loss between these responses and the ground-truth responses $Y_i$. This second loss further aligns the model’s outputs with the intended targets.

The complete pipeline is therefore:

1. Collect target ICL activations $A_{\text{ICL}}$ and train with $\mathcal{L}_{\textbf{IA2}}$ until convergence. This step aligns the model’s functional behavior with ICL.
2. Switch to $\mathcal{L}_{\text{SFT}}$ using ground-truth target responses $Y$, and train until convergence. This step aligns the model with the target output behavior.

## 5 Experimental Results

We study two settings with different characteristics: single-token output and multi-token output. We identify differences between the settings when presenting results.

**Adaptation Tasks:** In the single-token setting, each sample’s output $Y_i$ is one token, typically a category or class label, a True/False answer, or a multiple-choice question (MCQ) choice. In the multi-token setting, $Y_i$ may contain multiple tokens and vary in length, as in general open-ended generation.

For both settings, we compare adaptation methods across tasks and models. In many cases, we also evaluate on out-of-distribution (OOD) datasets to study behavior under distribution shift. An in-distribution (ID) evaluation uses the training dataset’s distribution. The tasks are:

**Single-token tasks:**

- **Sentiment Classification:** We use SST2 (Socher et al., 2013), Financial Phrasebank, abbreviated FinS (Malo et al., 2014), and Poem Sentiment, abbreviated PoemS (Sheng and Uthus, 2020). We train on SST2 and FinS and evaluate each trained set on all three datasets. Thus, each training source has one ID and 2 OOD evaluation datasets.
- **True/False:** We use StrategyQA, abbreviated STF (Geva et al., 2021), which requires multi-hop reasoning before reaching a True/False conclusion.
- **News Categorization:** We use AGNews, abbreviated AGN (Zhang et al., 2015), and BBCNews, abbreviated BBCN (Greene and Cunningham, 2006). We train on AGNews and evaluate on both datasets.
- **MCQ:** We use SciQ (Johannes Welbl, 2017) and QASC (Khot et al., 2019). To add complexity, we remap the choices [A, B, C …] to an unrelated token set. We call the remapped datasets SciQr and QASCr. We train on SciQr and evaluate on both remapped datasets.

**Multi-token tasks:**

- **Grade-school Math:** We use GSM8K (Cobbe et al., 2021) and GSM Symbolic, abbreviated GSM8Ks (Mirzadeh et al., 2024). We train on GSM8K and evaluate on both datasets.
- **Advanced Math:** We use MATH Algebra, abbreviated HMathA (Hendrycks et al., 2021), which contains higher-grade Algebra problems.
- **Scientific Question Answering (QA):** We use SciQ again, but the model generates the answer tokens rather than selecting an option.

**Data Setup:** Our focus is the data-scarce, few-shot setting in which ICL is typically used. For each task, we therefore construct multiple training datasets $\mathcal D_{\mathcal T}=\{(X_i,Y_i)\}_{i=1}^{N}$ by varying the number of samples $N$ in orders of $2:[2,4,8,16,\ldots]$. At each value of $N$, we create 5 different sets to average out outlier effects in the final quantitative analysis.

For IA2, we collect a target ICL activation tensor $A_{\text{ICL}}^i$ for every training input $X_i$ by using the remaining $N-1$ samples as demonstrations in random order. This procedure reuses training samples, so every adaptation method receives exactly the same data and the comparison remains fair. For evaluation, we sample a separate set of 500 samples for each task. Further dataset and processing details appear in §A.

**Multi-token details:** In this setting, the model generates a sequence of reasoning tokens followed by an answer in the format shown in the demonstrations. The reasoning tokens are standard text, not text inside `<think>` tags. Both the demonstrations and the expected outputs use a chain-of-thought style (Wei et al., 2022). Because evaluation concerns the answer, the answer must be extracted from the generated text. We use dataset-specific parsing methods:

- **GSM8K and GSM8Ks:** Ground-truth responses follow the pattern “`<reasoning steps> #### <numerical answer>`”. We do not evaluate the correctness or conciseness of the reasoning steps. Instead, we parse the numerical answer and require an exact match with the ground-truth answer to calculate accuracy. To allow slight output-format variations, we also use common parsing techniques such as matching “`The answer is <value>`”. The exact parsing function is available in our code.
- **HMathA:** Answers are expected inside a `\boxed{}` element and are parsed accordingly.
- **SciQ:** We construct ground-truth responses in a format similar to GSM8K: “`<support text> #### <text answer>`”. We parse answers using this format and count only exact string matches as correct.

**Training Setup:** We use two model families to test each training method across tasks. Qwen3-4B-Base is used on every single-token and multi-token task. For single-token tasks, the second model is Llama-3.2-1B, except on SciQr, where we use Llama-3.2-3B. We use the larger Llama model on SciQr because the 1B model could not improve above the random baseline for any $N$ up to 128.

For multi-token tasks, the secondary model is Llama-3.2-1B-Instruct. Using an Instruct model tests IA2’s efficacy on post-trained models.

We train low-rank adaptation modules (LoRAs) with rank 8 to modify the self-attention matrices $W_Q$, $W_K$, and $W_O$. We chose these models and weight combinations reasonably on the basis of available resources, because the experiments involved training more than 13,000 models in total.

Multi-token training requires substantially more resources because each sample has a long answer. Consequently, for multi-token tasks, we use $N$ only up to 16 for Llama models and up to 8 for Qwen models. For efficiency, we also fix the maximum number of generated training tokens to $G=200$. In IA2, this limit is used when collecting the target activation tensor, denoted here by $\hat A_i\in\mathbb R^{L\times 200\times d}$ for every input $x_i$. For SFT, $G$ instead equals the number of tokens in the ground-truth answer.

Each method–dataset combination is trained until convergence, up to 50 epochs, using 3 learning rates: slow, 1e-4; medium, 3e-4; and fast, 1e-3. We select the best-performing learning rate to neutralize the effect of hyperparameter selection on method performance. Additional training details appear in §B.

**Evaluation:** For single-token tasks, we use accuracy and expected calibration error (ECE). Because each answer contains one token, both metrics are calculated from the first generated token’s probability distribution.

We first evaluate the base model with demonstrations to establish the ICL performance benchmark. For this benchmark, we evaluate each task’s validation set 5 times with different demonstrations and report average metrics. We then evaluate each trained method without demonstrations and report the best-performing average metrics.

For multi-token evaluation, we restrict the maximum number of generated tokens to use compute judiciously. For GSM8K and GSM8Ks, the limit is 200 tokens, based on their 95 percentile answer lengths of 184 and 154, respectively. For HMathA, we relax the limit to 400 tokens because the 95 percentile answer length is 307. We evaluate only accuracy: the number of questions answered correctly in the required format. Answer-parsing details appear above and in §A. We do not measure calibration for multi-token tasks because answer confidence is hard to measure for open-ended generation.

### 5.1 IA2 priming boosts SFT performance

**Single-Token results:** Table 1 reports accuracy and ECE for Qwen3-4B-Base models with $N=4$ training samples. Using the same amount of data, IA2 → SFT outperforms SFT-only training in almost all cases on both metrics. For example, in the SST2-source, SST2-evaluation row, accuracy is 90.4 (02.3) with IA2 → SFT versus 65.2 (10.7) with SFT only, while ECE is 0.06 (0.01) versus 0.22 (0.10). In most cases, IA2 → SFT also has higher accuracy than ICL, with slightly less calibration.

IA2-only training reaches high accuracy in many cases while remaining reasonably calibrated like ICL, despite not training on target response tokens. For example, in the SciQr-source, SciQr-evaluation row of Table 1, IA2 only achieves accuracy of 90.4 (01.5) and ECE of 0.09 (0.01). These results highlight the richness of the IA2 training signal.

Table 1 shows one setup; Figure 1 shows aggregate performance trends across models and datasets. Detailed tables for other model/$N$ combinations and a statistical significance test of the improvement appear in §C.

In Table 1, “Source” identifies the training dataset and “Eval” identifies the evaluation dataset. The dataset labels are defined in the task list above. “acc” denotes accuracy, for which higher is better ($\uparrow$), and “ece” denotes expected calibration error, for which lower is better ($\downarrow$). An asterisk ($\ast$) marks an OOD evaluation.

| Source | Eval | ICL acc ↑ | ICL ece ↓ | SFT only acc ↑ | SFT only ece ↓ | IA2 only acc ↑ | IA2 only ece ↓ | IA2 → SFT acc ↑ | IA2 → SFT ece ↓ |
|---|---|---|---|---|---|---|---|---|---|
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

**Table 1:** Performance report for $N=4$ on the Qwen3-4B-Base model, showing accuracy (acc) and Expected Calibration Error (ece). Numbers in parentheses are standard deviations across 5 runs for the best-performing learning rate. Best training method shown in bold. An asterisk (∗) highlights OOD evaluations. Our proposed IA2 → SFT training method outperforms standard SFT across the board.

Table 2 uses the same source/evaluation convention and reports multi-token accuracy. “w/o ICL” means the base model without ICL demonstrations.

| Source | Eval | w/o ICL acc ↑ | ICL acc ↑ | SFT only acc ↑ | IA2 only acc ↑ | IA2 → SFT acc ↑ |
|---|---|---|---|---|---|---|
| GSM8K | GSM8K | 56.4 (00.0) | 76.4 (01.1) | 70.9 (02.8) | 77.4 (02.8) | 73.6 (03.7) |
| GSM8K | GSM8Ks∗ | 45.4 (00.0) | 68.4 (01.1) | 64.5 (04.0) | 66.2 (03.7) | 68.8 (05.1) |
| HMathA | HMathA | 21.0 (00.0) | 60.4 (02.1) | 50.4 (01.8) | 47.8 (06.0) | 55.3 (02.1) |
| SciQ | SciQ | 15.4 (00.0) | 37.5 (01.0) | 35.0 (07.1) | 06.9 (13.8) | 40.8 (07.6) |

**Table 2:** Performance report for $N=4$ on Qwen3-4B-Base models on multi-token datasets. Best training method shown in bold. IA2 → SFT outperforms standard SFT across all datasets.

**Multi-Token results:** Table 2 reports the accuracy of adapted Qwen models in the multi-token setting. As in the single-token setting, IA2 → SFT outperforms SFT-only training; for the multi-token setting, the improvement occurs in all cases. The IA2 → SFT and SFT-only accuracies are, respectively, 73.6 (03.7) versus 70.9 (02.8) on GSM8K, 68.8 (05.1) versus 64.5 (04.0) on GSM8Ks, 55.3 (02.1) versus 50.4 (01.8) on HMathA, and 40.8 (07.6) versus 35.0 (07.1) on SciQ.

However, compared with the single-token setting, IA2 → SFT sometimes performs worse than ICL. In Table 2, for example, GSM8K accuracy is 73.6 (03.7) with IA2 → SFT versus 76.4 (01.1) with ICL, and HMathA accuracy is 55.3 (02.1) versus 60.4 (02.1). We suspect that these shortfalls result from imperfect compression of longer contexts into the same small LoRA weight space, and that larger ranks should improve this compression. Additional tables for other model/$N$ combinations appear in §C.

### 5.2 Why is IA2 priming important?

**Figure 3:** Activation similarity, $\operatorname{asim}(A,A_{\text{ICL}})$, plotted against Accuracy/ECE for all training methods to show the relationship between activation similarity and performance metrics. Here, $A$ is a trained model’s activation tensor and $A_{\text{ICL}}$ is the ICL activation tensor. Each point represents one training experiment. As similarity increases, ECE decreases smoothly. However, extreme ICL activation alignment may leave accuracy gains available that can be obtained by applying SFT to ground-truth responses.

We investigate properties of three training methods: (1) SFT only, (2) IA2 only, and (3) IA2 → SFT.

**Activation Similarity vs Performance:** Figure 2 shows each method’s activation similarity with ICL. As expected, IA2 drastically increases similarity. Importantly, IA2 → SFT retains higher similarity with ICL activations even after the subsequent SFT training.

Figure 3 relates ICL activation similarity to accuracy and ECE. IA2 → SFT lies between IA2 only and SFT only in activation similarity while achieving better accuracy and calibration. SFT-only training and the SFT stage of IA2 → SFT differ only in their starting state. Therefore, the difference in their outcomes implies that IA2 provides a rich training signal unavailable through SFT-only training.

Greater activation similarity clearly implies better calibration in Figure 3. However, extreme activation similarity may not yield the best accuracy because the ICL signal is not always right. Indeed, §C presents a case in which IA2 models follow ICL performance even when ICL performance worsens because of overfitting. Therefore, the best approach is to combine the two signals: IA2 improves internal functional alignment with ICL, while SFT on ground-truth responses aligns outputs with expected behavior.

**Subspace Overlap:** Although activation patterns are a footprint of functional behavior, activation patterns do not isolate behavioral changes induced by different training methods. This limitation arises because we use LoRA adaptors, so the unchanged base-model weights substantially influence the activations. To isolate the behavioral changes, we therefore analyze the LoRA weights directly.

**Figure 4:** IA2 → SFT models have significantly high subspace overlap with IA2-only models, whereas SFT-only models do not align with either. This difference indicates that IA2 supplies a training signal absent from SFT-only training and important for performance.

Figure 4 shows the distribution of weight-subspace overlap between pairs of models trained by different methods, using models from all single-token and multi-token experiments. We first perform singular value decomposition (SVD) on the weight matrices to obtain basis vectors. For two compared methods, let $U_1$ and $U_2$ denote their basis vectors, and let $r$ be the rank of the LoRA weights. We calculate subspace overlap as

$$
\frac{1}{r}*\left\|U_1^T\cdot U_2\right\|_F^2,
$$

where $\|\cdot\|_F$ denotes the Frobenius norm. Subspace overlap measures how much weight space the two methods share, thereby isolating and comparing the difference in their functional behavior.

SFT-only training produces weight updates almost completely orthogonal to those from the other two methods. In contrast, the most performant IA2 → SFT models share around $39\%$ of their spanned weight space, on average, with IA2-only models. This overlap implies that much of the IA2 → SFT performance gain comes from IA2 and that the subspaces identified by IA2 are practically unreachable through SFT-only training.

The overlap between IA2-only and IA2 → SFT models has a high spread. In many cases, the IA2 model was already output-aligned, so the subsequent SFT stage changed the model little before convergence. Those cases therefore produced very high subspace overlap.

## 6 Discussion

**IA2 + SFT:** A natural alternative to sequential IA2 → SFT is simultaneous IA2 + SFT: training with both $\mathcal L_{\text{IA2}}$ and $\mathcal L_{\text{SFT}}$ at the same time. However, simultaneous training poses a practical challenge. For any response length $G>1$, the target IA2 activation tensor $\hat A$ is collected using the model’s own generated response tokens. We call this generated response the *ICL response*. The ICL response may differ substantially from the dataset’s ground-truth response, even in length. Consequently, the two objectives are incompatible for joint training when they use these different responses.

We can, however, perform self-distillation using ICL responses, as in Snell et al. (2022), while adding the IA2 signal. We train with the unified loss

$$
\mathcal L_{\text{IA2}}+\beta\cdot\mathcal L_{\text{SFT}},
$$

where $\beta$ weights the SFT loss. We vary $\beta$ over 4 values to span the spectrum smoothly between IA2-only and SFT-only training; details appear in §B. After selecting the best-performing learning-rate/$\beta$ combination across 5 random seeds, we report multi-token results in Table 3.

Relative to ground-truth training in Table 2, training on ICL responses lowers SFT-only and IA2 → SFT performance. For example, on GSM8K, SFT-only accuracy changes from 70.9 (02.8) in Table 2 to 66.4 (18.1) in Table 3, while IA2 → SFT changes from 73.6 (03.7) to 67.2 (22.1). Nevertheless, IA2 + SFT extracts significantly more performance from ICL responses than SFT only. In Table 3, IA2 + SFT and SFT-only accuracies are, respectively, 77.0 (07.8) versus 66.4 (18.1) on GSM8K, 69.0 (09.6) versus 59.8 (20.2) on GSM8Ks, 53.6 (02.7) versus 49.0 (03.0) on HMathA, and 34.5 (10.6) versus 27.2 (12.0) on SciQ. These results highlight the richness of the IA2 training signal. Further IA2 + SFT results appear in §C.

| Source | Eval | w/o ICL acc ↑ | ICL acc ↑ | SFT only acc ↑ | IA2 only acc ↑ | IA2 → SFT acc ↑ | IA2 + SFT acc ↑ |
|---|---|---|---|---|---|---|---|
| GSM8K | GSM8K | 56.4 (00.0) | 76.4 (01.1) | 66.4 (18.1) | 77.4 (02.8) | 67.2 (22.1) | 77.0 (07.8) |
| GSM8K | GSM8Ks∗ | 45.4 (00.0) | 68.4 (01.1) | 59.8 (20.2) | 66.2 (03.7) | 61.6 (24.0) | 69.0 (09.6) |
| HMathA | HMathA | 21.0 (00.0) | 60.4 (02.1) | 49.0 (03.0) | 47.8 (06.0) | 54.0 (01.7) | 53.6 (02.7) |
| SciQ | SciQ | 15.4 (00.0) | 37.5 (01.0) | 27.2 (12.0) | 06.9 (13.8) | 32.8 (10.0) | 34.5 (10.6) |

**Table 3:** Performance report for $N=4$ on Qwen3-4B-Base models trained with ICL responses instead of ground-truth responses. IA2 + SFT performs significantly better than SFT only, highlighting the significance of IA2.

**Knowledge Distillation baseline:** Soft-label matching is a popular method for distilling a stronger teacher model’s behavior into a student model (Hinton et al., 2015). Instead of supplying only target tokens, the teacher provides a denser training signal through the probability mass associated with each token. In our setup, the teacher is the same model enhanced with ICL context.

To compare IA2-based training with soft-label matching, we train and evaluate Llama models with soft-label matching on ICL responses. We use Llama-3.2-1B-Instruct on GSM8K and Llama-3.2-1B on SST2. Table 4 shows that, in the multi-token GSM8K case, soft-label matching performs significantly better than standard SFT and almost reaches IA2 + SFT: accuracy is 36.1 (03.5) for soft labels, 28.2 (04.5) for standard SFT, and 37.0 (01.9) for IA2 + SFT. However, soft-label matching lacks performance in the single-token case. On SST2, soft-label accuracy is 55.9 (08.8), compared with 66.9 (18.4) for IA2 + SFT. These comparisons highlight the consistency of IA2-based training in achieving high performance.

In Table 4, “SFT (soft labels)” is the Knowledge Distillation (KD) baseline. A dash indicates that ECE is not measured for the multi-token task.

| Dataset | $N$ | ICL acc ↑ | ICL ece ↓ | SFT only acc ↑ | SFT only ece ↓ | SFT (soft labels) acc ↑ | SFT (soft labels) ece ↓ | IA2 only acc ↑ | IA2 only ece ↓ | IA2 → SFT acc ↑ | IA2 → SFT ece ↓ | IA2 + SFT acc ↑ | IA2 + SFT ece ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GSM8K | 4 | 34.6 (00.9) | – | 28.2 (04.5) | – | 36.1 (03.5) | – | 29.3 (05.0) | – | 30.3 (03.2) | – | 37.0 (01.9) | – |
| SST2 | 8 | 77.0 (02.2) | 0.10 (0.02) | 52.3 (04.3) | 0.38 (0.14) | 55.9 (08.8) | 0.16 (0.06) | 64.9 (15.3) | 0.30 (0.05) | 64.7 (19.3) | 0.33 (0.19) | 66.9 (18.4) | 0.28 (0.21) |

**Table 4:** Comparison with the Knowledge Distillation (KD) baseline, SFT on soft labels. Although the KD baseline outperforms standard SFT in the multi-token case, IA2-based training provides consistently high performance in both single-token and multi-token cases.

**Why IA2 based training does not consistently beat ICL?** In some multi-token cases, including Qwen on the math datasets GSM8K and HMathA, ICL performance exceeds all training methods, including IA2-based training. We suspect that Qwen’s mid-training on science, technology, engineering, and mathematics (STEM) data makes ICL extremely sample-efficient. Surprisingly, GSM8K ICL performance at $N=2$ is higher than at $N=4,8$: Table 5 reports 81.2 (00.4), 76.4 (01.1), and 77.2 (00.9), respectively.

Table 5 shows performance as the shot count, or number of training samples $N$, increases. The table also reports the IA2 → SFT gap with respect to ICL and SFT only. The gap between IA2 → SFT and ICL closes quickly with more shots: the GSM8K gap changes from -06.6 at $N=2$ to -01.2 at $N=8$, and the HMathA gap changes from -09.9 to -02.7. Our key takeaway remains consistent: IA2 → SFT outperforms SFT only, as shown by every positive entry in the final column.

| Dataset | $N$ | ICL | SFT only | IA2 only | IA2 → SFT | Gap wrt ICL | Gap wrt SFT only |
|---|---|---|---|---|---|---|---|
| GSM8K | 2 | 81.2 (00.4) | 65.2 (01.0) | 70.7 (01.8) | 74.6 (02.4) | -06.6 | +09.4 |
| GSM8K | 4 | 76.4 (01.1) | 70.9 (02.8) | 77.4 (02.8) | 73.6 (03.7) | -02.8 | +02.7 |
| GSM8K | 8 | 77.2 (00.9) | 73.5 (01.6) | 78.8 (02.5) | 76.0 (02.0) | -01.2 | +02.5 |
| HMathA | 2 | 58.6 (02.0) | 47.8 (04.0) | 33.9 (06.4) | 48.7 (07.0) | -09.9 | +00.9 |
| HMathA | 4 | 60.4 (02.1) | 50.4 (01.8) | 47.8 (06.0) | 55.3 (02.1) | -05.1 | +04.9 |
| HMathA | 8 | 62.1 (01.0) | 52.0 (00.8) | 57.3 (03.7) | 59.4 (00.8) | -02.7 | +07.4 |

**Table 5:** ICL is more sample-efficient than training methods for Qwen models on math datasets. However, IA2 → SFT quickly closes the performance gap as the number of shots increases and consistently outperforms SFT-only training. “wrt” means “with respect to.”

**Other fine-tuning methods:** LoRA is not the only fine-tuning method. Alternatives include full, full-rank fine-tuning and other parameter-efficient methods, such as prompt/prefix tuning (Lester et al., 2021; Li and Liang, 2021) and activation scaling, called (IA)3 (Liu et al., 2022). Our pipeline does not compete with these methods; rather, the pipeline improves these methods by providing better training signals.

To establish this point, we repeat experiments on a few datasets using the popular PEFT method (IA)3. We follow exactly the methodology in §5, replacing only LoRA training with (IA)3; setup details appear in §B. Table 6 reports one setting and shows trends similar to LoRA. For example, with SciQr as both the source and evaluation dataset, IA2 → SFT achieves accuracy of 70.6 (04.6), compared with 64.5 (05.0) for SFT only. Other results appear in §C. We defer exploration of IA2 improvements to other SFT methods to future work. On a side note, IA2 also marks a major improvement in method naming compared to (IA)3.

| Source | Eval | ICL acc ↑ | ICL ece ↓ | SFT only acc ↑ | SFT only ece ↓ | IA2 only acc ↑ | IA2 only ece ↓ | IA2 → SFT acc ↑ | IA2 → SFT ece ↓ |
|---|---|---|---|---|---|---|---|---|---|
| SciQr | QASCr∗ | 35.6 (01.4) | 0.06 (0.01) | 32.8 (03.4) | 0.20 (0.08) | 34.2 (02.7) | 0.08 (0.01) | 48.6 (05.5) | 0.25 (0.11) |
| SciQr | SciQr | 60.0 (01.8) | 0.07 (0.01) | 64.5 (05.0) | 0.12 (0.05) | 63.0 (04.6) | 0.10 (0.04) | 70.6 (04.6) | 0.08 (0.02) |
| SST2 | FinS∗ | 56.6 (01.6) | 0.17 (0.01) | 58.2 (10.7) | 0.25 (0.03) | 61.9 (09.5) | 0.18 (0.06) | 59.8 (07.8) | 0.18 (0.03) |
| SST2 | PoemS∗ | 52.3 (03.0) | 0.19 (0.02) | 48.1 (03.2) | 0.44 (0.11) | 48.1 (03.4) | 0.11 (0.04) | 59.8 (13.4) | 0.33 (0.13) |
| SST2 | SST2 | 60.3 (02.8) | 0.19 (0.02) | 52.4 (02.8) | 0.35 (0.05) | 54.2 (04.2) | 0.15 (0.06) | 62.6 (09.1) | 0.28 (0.11) |

**Table 6:** Performance report for $N=4$ on Llama-3.2 models trained using (IA)3. IA2 → SFT achieves significant gains over SFT-only training, exceeding even ICL accuracy in all cases.

**Computational Overhead:** IA2 → SFT requires an activation-collection step before training. During this step, we perform ICL inference on the model using the given dataset. Collecting the rich ICL activations is a one-time cost. The collected activations greatly benefit the model’s capabilities, while inference with the resulting model costs the same as inference with an SFT-only model.

**Potential improvements:** Resource constraints led us to fix several design choices that could be explored and improved. The most important possibilities include ablations of LoRA parameters—such as rank and target modules—and selective IA2.

Our results show that extreme alignment can hurt performance, and prior work (Todd et al., 2023) shows that some layers are more sensitive to ICL than others. Therefore, selectively aligning layers could be more effective. Another minor improvement could be prompt optimization before collecting target activations, because ICL performance is sensitive to demonstration order (Lu et al., 2022; Zhao et al., 2021). Finally, we use fixed learning rates for IA2 → SFT, but the two training signals could benefit from finer-grained control. We defer these explorations to future work.

## 7 Related Work

**ICL vs SFT:** Prior work has studied performance differences between ICL and SFT (Mosbach et al., 2023; Duan et al., 2024), but the verdict remains unclear. In our work, SFT performed worse than ICL. Other studies examine functional similarities (Dai et al., 2022) and differences (Wang et al., 2023) between ICL and SFT. Some studies show how adaptation through ICL or SFT can be miscalibrated and how calibration can be improved (Zhang et al., 2024; Li et al., 2025; Li et al., 2024; Xiao et al., 2025).

Sia et al. (2024) identify “where” translation happens in LLMs during ICL through context masking. Like our work, that study uses activations as a signal of functional behavior. Importantly, recent work (Chu et al., 2025; Shenfeld et al., 2025) shows that SFT is prone to memorization/overfitting and catastrophic forgetting. Our improved SFT pipeline aims to address these problems.

**Context Distillation:** Like our work, a large body of research seeks to improve LLM adaptation by compressing explicit context signals. Approaches include using intermediate contemplation tokens for compression (Cheng and Van Durme, 2024; Jiang et al., 2025), internalizing context by generating adapters on the fly (Chen et al., 2025; Charakorn et al., 2025), using related text to enhance distillation (Zhu et al., 2025; Choi et al., 2025), and compressing context directly into weights (Shen et al., 2025; Deng et al., 2024; Yu et al., 2024; Snell et al., 2022; Chen et al., 2024b; Shin et al., 2024).

These methods aim to distill a given context’s effect on model outputs using token-response signals. Our work is complementary: IA2 improves distillation through the model’s own ICL processing signals, an idea supported by prior work (Aguilar et al., 2020; Jin et al., 2024; Yang et al., 2024).

**Activation Steering:** A growing line of work extracts “steering” vectors or weights and applies those vectors or weights at targeted locations inside a model to produce an intended output effect (Postmus and Abreu, 2024; Stolfo et al., 2025; Caccia et al., 2025; Fleshman and Van Durme, 2024; Arditi et al., 2024). These methods resemble IA2 because both intervene in activation space to influence model functionality.

However, activation-steering methods use targeted interventions and have a small application domain. In contrast, our method aims to incite a large behavioral change in the model, which avoids potential brittleness and reliability issues (Queiroz Da Silva et al., 2025).

## 8 Conclusion and Future Work

We used model activations to probe LLMs’ functional behavior and found a key distinction: ICL and SFT achieve adaptation through different mechanisms, and ICL often encodes richer, more generalizable patterns. Motivated by this distinction, we introduced IA2, an SFT priming technique that enables ICL-like internal behavior.

IA2 shifts models into a more adaptable weight subspace inaccessible to SFT alone. The activation-similarity and subspace-overlap analyses in §5.2 support this distinction, while the extensive experiments in §5 show that IA2 consistently improves SFT accuracy and calibration.

In future work, we aim to test other effects of IA2-based SFT, including effects on diversity and catastrophic forgetting, to measure how IA2 can help with LLM post-training. We also aim to study IA2’s effect on post-trained models tuned with reinforcement learning (RL).

## Reproducibility Statement

We provide code for data download and processing, training, evaluation, analysis, plotting, and other steps on github. The details in §5 and the appendix, together with the detailed README file, should be sufficient to reproduce our results fully because we used repeatable random seeds.

## Acknowledgments

This work is supported by ONR (N00014-24-1-2089), an Amazon Research Award, and a grant from Open Philanthropy. We thank Andrea Wynn for their constructive feedback on earlier versions of this document.
