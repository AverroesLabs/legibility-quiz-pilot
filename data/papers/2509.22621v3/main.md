# IA2: Alignment with ICL Activations

Improves Supervised Fine-Tuning

Aayush Mishra

  
Daniel Khashabi & Anqi LiuDepartment of Computer Science,
Johns Hopkins University

###### Abstract

Supervised Fine-Tuning (SFT) is used to specialize model behavior by training weights to produce intended target responses for queries. In contrast, In-Context Learning (ICL) adapts models during inference with instructions or demonstrations in the prompt. ICL can offer better generalizability and more calibrated responses compared to SFT in data scarce settings, at the cost of more inference compute. In this work, we ask the question: Can ICL’s internal computations be used to improve the qualities of SFT? We first show that ICL and SFT produce distinct activation patterns, indicating that the two methods achieve adaptation through different functional mechanisms. Motivated by this observation and to use ICL’s rich functionality, we introduce ICL Activation Alignment (IA2), a self-distillation technique which aims to replicate ICL’s activation patterns in SFT models and incentivizes ICL-like internal reasoning. Performing IA2 as a priming step before SFT significantly improves the accuracy and calibration of model outputs, as shown by our extensive empirical results on 12 popular benchmarks and two model families. This finding is not only practically useful, but also offers a conceptual window into the inner mechanics of model adaptation.

|

00footnotetext: ♡Equal advising. Correspondence to: {amishr24,danielk,aliu.cs}@jhu.edu

## 1 Introduction

LLMs are general purpose models but are often used in specialized applications. For example, a news aggregator app may need to classify articles into predefined categories.
A popular approach for adapting LLMs is Supervised Fine-Tuning (SFT) which uses a dataset of labeled samples to train and adapt LLMs on narrow downstream tasks. With the power of parameter efficient fine tuning (PEFT) techniques (Hu et al., 2021; Liu et al., 2022), SFT models are often just a small set of parameters which can be efficiently loaded on/off GPU memory making them extremely useful. However, SFT typically requires a large set of labeled samples to generalize well on new tasks (Le Scao and Rush, 2021), which can be expensive to collect.

In contrast, In-Context Learning (ICL) (Brown et al., 2020) is used to adapt and steer LLM behavior during inference time. The model is given a query preceded by in-context demonstrations or instructions, which help it “learn” the demonstrated task and answer accordingly. Prior work (Duan et al., 2024) shows (and our experiments concur (§5)) that ICL generalizes well in a few-shot setting and typically produces well calibrated responses. However, using ICL comes at a cost. ICL demonstrations/instructions use up valuable context space (increasing the cost of running each query) which could otherwise be used for processing more query and response tokens.

These subtle but important differences between ICL and SFT motivate us to investigate the difference in their functional behavior. We find that while their token space behavior may look similar on the surface, models produce disparate internal activations under ICL and SFT (§3). As activations are a footprint of the model’s internal processing before it produces the next token, divergent activation patterns highlight a difference in how the two methods achieve adaptation, an idea supported by prior work (Shen et al., 2024). We hypothesize that when the model performs ICL, its activations contain rich information about how to extract generalizable patterns from the context. This information may be absent in SFT models especially in a few-shot setting, where they are prone to shortcut learning/overfitting on target responses. Hence, we ask the research question:

Can the information rich ICL activations be used to improve the quality of SFT?

Recent works have tried to distill context in the weights of LLMs (Snell et al., 2022; Chen et al., 2024b), but they use training signals only from response texts, which may suffer from the same SFT issues highlighted above.
Simply training a model to reproduce the outputs of an ICL-conditioned model does not ensure that it functions like an ICL-conditioned model.

To address this, we propose ICL Activation Alignment (IA2), a self-distillation method to enforce alignment with the model’s own functional behavior when performing ICL (§4). See Figure 1 for an overview of our proposed method: (1) collect information rich ICL activations and (2) enforce functional alignment with ICL. Then (3) perform SFT on this primed model. We show that priming models with IA2 before SFT using the same data, drastically improves the performance of the adapted model on a variety of text classification and generation tasks (§5). In addition, we show that IA2 provides an important training signal that is unavailable with SFT only training. We trained over 13,000 models spanning 12 benchmarks, to validate our findings, which not only signify the practical benefits of IA2, but offer a conceptual window into the inner mechanics of model adaptation. In summary:

$\star$

We show that ICL and SFT with the same data, do not align in the model’s activation space, highlighting a gap in their functional behavior (§3).

$\star$

We propose IA2—ICL Activation Alignment, to enforce alignment with ICL’s functional behavior. This priming step drastically improves the performance of SFT models (see Figure 1, §4).

$\star$

We show that the IA2 training signal is not present in SFT only training, highlighting its importance in improving the quality of adaptation (§5).

Figure 1: An overview of our improved SFT pipeline. Standard SFT only enforces output space alignment (between the model’s response and a target response—only step 3 above), resulting in subpar performance and mis-calibration in low-data settings. In contrast, our method IA2 enforces functional alignment with ICL by matching the rich activation patterns produced when performing ICL. IA2 priming before SFT boosts the quality of adaptation. We show this improvement through performance comparison charts aggregated across models and datasets. See section  5 for details.

## 2 Background and Notation

Transformer Language Models: A standard decoder-only transformer based LM $M_{\Theta}$ with parameters $\Theta$ consists of a stack of self-attention (SA) and linear layers. A sequence of token vectors $T=[t_{i}]_{i=1}^{R}$ is processed by applying the SA and linear projections on each token at each layer, until the last token projection at the final layer is passed through the LM head to predict the most likely next token in the sequence. The SA operation is special because it is affected by other tokens in the sequence. For a standard SA operation which uses 4 weight matrices, $W_{Q},W_{K},W_{V}$ and $W_{O}\in\Theta$ (we will call it the set $W_{QKVO}$ for brevity), the output at each token position is given by:

| | $\text{SA}(T;W_{QKVO})=Z=\big[z_{i}=\sigma\big(\frac{q_{i}K^{T}}{\sqrt{d}}\big)V\cdot W_{O}\big]_{i=1}^{R}$
| | (1)

where $\sigma(.)$ denotes Softmax, $q_{i}=t_{i}\cdot W_{Q},K=t_{:i}\cdot W_{K}$ , $V=t_{:i}\cdot W_{V}$ ( $t_{:i}$ means first $i$ tokens). Note that $Z$ has the same shape ( $R\times d$ ) as the input $T$ . In this work, we study the interplay of changes in model behavior induced by changing the context $T$ or the weights $W_{QKVO}$ .

Supervised Fine-Tuning: Given a set $\mathcal{D}_{\mathcal{T}}=\{(X_{i},Y_{i})\}_{i=1}^{N}$ of $N$ input ( $X$ )–output ( $Y$ ) pairs illustrating a task $\mathcal{T}$ , we can fine-tune the model weights $\Theta\rightarrow\Theta^{\prime}$ to produce relevant responses for new inputs (say $X_{t}$ ), i.e., $M_{\Theta^{\prime}}(X_{t})=\bar{y}_{t}$ . Here, $\bar{y}_{t}$ is the first token of the whole response $\bar{Y}_{t}$ which can be extended by repeated application of $M_{\Theta^{\prime}}$ on the growing sequence. This process is performed as follows. If the ground truth response $Y_{i}$ contains $G$ tokens, we generate $G$ new tokens with the model to get $\bar{Y}_{i}$ and minimize the cross-entropy loss over all generated tokens. The SFT loss is defined as:

| | $\mathcal{L}_{\text{SFT}}=\sum_{i=1}^{N}\sum_{j=1}^{G}\text{cross-entropy}(\bar{Y}_{ij},Y_{ij})$
| | (2)

In-Context Learning:
Using $\mathcal{D}_{\mathcal{T}}$ , $M_{\Theta}$ can also be prompted with the concatenated sequence of demonstration tokens $I=[X_{1}\circ Y_{1}\circ\ldots\circ X_{n}\circ Y_{n}]$ along with a new test sample $X_{t}$ to get a task-appropriate response without any weight updates. $M_{\Theta}$ processes the examples in its prompt to understand the task and produces a response $M_{\Theta}(I\circ X_{t})=\hat{y}_{t}$ accordingly. Remarkably, $\hat{Y}_{t}$ is often similar to the expected response $Y_{t}$ . Hence, ICL serves as a useful inference-time adaptation method. ICL was first illustrated in GPT-3 (Brown et al., 2020) and is widely used to adapt generic LMs at the the inference-time. Today’s frontier models have the ability to follow instructions directly, so a generic instruction could be considered a form of ICL demonstrations. In this work, we will focus on the classic ICL setting with a demonstration set of input-output pairs.

Activations: As the SA operation uses every token in the context, it produces different outputs for different sequences of tokens. We denote the hidden SA outputs as activations. Activations are a footprint of the model’s internal processing which can be used to study model behavior changes with changing contexts. If $M_{\Theta}$ consists of $L$ layers, and it processes $R$ tokens, we get an activation tensor $A$ of size $L\times R\times d$ . In this work, we study sequences of tokens $T=[I\circ X]$ for ICL; and $T=X$ without ICL under different model weights, where $I$ denotes tokens of ICL demos and $X$ denotes tokens of the query. Here, we also define activation similarity (asim $(A_{1},A_{2})$ ) as the token-wise cosine similarity between two activation tensors. Note that asim is of size $L\times R$ .

## 3 Are ICL and SFT practically the same?

Figure 2:
Layerwise Similarity with ICL Activations.
SFT models have small asim with ICL. This indicates differing functional behavior for adaptation. Our recommended pipeline (IA2 → SFT) aligns much more with ICL than using only SFT, and performs much better as a result (see  Figure 1).

Recent studies (Von Oswald et al., 2023; Akyürek et al., 2022) claim that ICL in transformer models works by an internal gradient descent mechanism. They show that the self-attention operation can produce outputs as if the model parameters were updated using ICL demos in context. This implies a functional equivalence between the two. Under this equivalence, we should expect the base model with ICL demos to produce similar activations at the output token positions as the SFT model produces without ICL demos. We investigated whether this phenomenon is actually exhibited by LLMs.

ICL and SFT produce different activations: We collected ICL activations $A_{\text{ICL}}$ (using $T=[I\circ X]$ ) for 100 test samples from multiple datasets (SST2, AGNews, etc.) in a variety of models (Qwen3 (Yang et al., 2025) and Llama-3.2 (Grattafiori et al., 2024) family, 1B $\leftrightarrow$ 16 layers, 3B $\leftrightarrow$ 28 layers, and 4B $\leftrightarrow$ 36 layers). We then trained SFT models using the same ICL demos until convergence, and collected activations $A_{\text{SFT}}$ (using $T=X$ ). Experiment details can be found in §B. Then, we calculated asim $(A_{\text{ICL}},A_{\text{SFT}})$ at output token positions, and plotted the average across tokens, samples, and datasets (see Figure 2–SFT only in the first row). We see that ICL and SFT activations are not aligned across different models. It is noteworthy that the activations align better near the initial and final layers where we expect the tokens to be processed at an individual level, but are misaligned in the middle where we expect the whole demonstration set to be processed at an abstract level.

Why should we care about alignment with ICL?
The difference in functional behavior of ICL and SFT is not just hidden in activation patterns. It also surfaces in the form of expected calibration error (Guo et al., 2017) as seen in Figure 1. ICL is much more calibrated in its responses than SFT at similar (slightly better) accuracy levels. We suspect that this is due to the output-oriented nature of the SFT training signal, which can allow it to learn shortcuts that can fail on new data. In contrast, as ICL relies on complex circuits (Elhage et al., 2021) that need to work for a variety of tasks, it extracts generalizable patterns from the demos and performs in a well-calibrated manner. This experiment demonstrates that ICL and SFT achieve adaptation through different means, and SFT should not be expected to work as a drop-in replacement for ICL. Motivated by these differences, we ask the following research question:

Can the information rich ICL activations be used to improve the quality of SFT?

Next, we will show that this is indeed the case and describe our proposed SFT procedure.

## 4 Inducing ICL-like behavior in SFT Models

In this section, we will discuss our two-step SFT pipeline. First, we propose our priming step (IA2) which incentivizes ICL-like functional behavior. Then, we discuss how performing further SFT on these ICL-aligned weights results in much better adaptation qualities.

IA2—ICL Activation Alignment: In §3, we found SFT to have little asim with ICL. To increase asim, and hence the functional similarity with ICL, we design the following goal (IA2) for each layer of $M_{\Theta}$ :

Find $\tilde{W}_{QKVO}=(\tilde{W}_{Q},\tilde{W}_{K},\tilde{W}_{V},\tilde{W}_{O})$ such that for every newly generated token (at index: $-1$ ):

| | $\text{SA}([I\circ X];W_{QKVO})\approx\text{SA}(X;\tilde{W}_{QKVO})\quad\forall X\in\mathcal{T}.$
| | (3)

Prior work (Chen et al., 2024a) has shown a closed-form solution for the above on linearized attention models (using a kernel approximation for softmax). We aim to find a practical general solution in real non-linear transformers. Our goal tries to achieve the qualities of ICL directly in the weights of the SFT model. These modified weights are not incentivized to produce similar outputs, but to process all inputs (queries) similar to how ICL does at each layer of the base model.

To perform IA2, we first generate model responses using ICL ( $T_{i}=[I\circ X_{i}]$ ), giving us $\hat{Y}_{i}$ for all training samples $X_{i}$ . We use the remaining samples in the dataset to construct $I$ for each sample. We collect activations at each output position giving us $A^{i}_{\text{ICL}}\in\mathbb{R}^{L\times G\times d}$ assuming $G$ response tokens. Next, we provide the response attached only with the query $T_{i}=[X_{i}\circ\hat{Y}_{i}]$ , as if the model had produced the ICL response with only the query in context. This gives an unaligned activation tensor $A^{i}$ at the output token positions. Then, we simply train the model to minimize the mean squared error between the two activations w.r.t. model parameters:

| | $\mathcal{L}_{\textbf{{IA2}}}=\sum_{i=1}^{N}||A^{i}-A^{i}_{\text{ICL}}||.$
| | (4)

IA2 → SFT: After IA2, we continue to train the model with the standard cross-entropy loss for SFT (Equation 2) on target tokens. Say IA2 training updates the model parameters from $\Theta$ to $\Theta^{\prime}$ . We collect output responses $\bar{Y}_{i}$ for all $X_{i}$ using $M_{\Theta^{\prime}}$ and minmize the SFT loss between $\bar{Y}_{i}$ and ground truth $Y_{i}$ for all training samples. This loss further aligns the model’s outputs with our intended targets. Overall, our proposed training pipeline has the following two steps:

1.

Collect target ICL activations $A_{\text{ICL}}$ and perform IA2 using $\mathcal{L}_{\textbf{{IA2}}}$ until convergence. This aligns the model’s functional behavior with ICL.

2.

Switch to $\mathcal{L}_{\text{SFT}}$ with ground truth target responses $Y$ , and train until convergence to align with target output behavior.

## 5 Experimental Results

Our study includes experiments in two different settings: single-token and multi-token, with different characteristics. We will highlight any differences as we present our results.

Adaptation Tasks: In the single-token setting, the output $Y_{i}$ for each sample is a single token, typically a category/class label, True/False, or MCQ choice. In the multi-token setting, the output $Y_{i}$ can be multiple tokens long and variable in length. This is the general case for open-ended generation. For both settings, we test and compare our proposed SFT pipeline on various tasks using multiple models. In many cases, we also test the adaptation methods on OOD datasets to study their behavior under distribution shift. We enlist the tasks under each category below:

Single-token tasks:

•

Sentiment Classification: We use SST2 (Socher et al., 2013), Financial Phrasebank [FinS] (Malo et al., 2014) and Poem Sentiment [PoemS] (Sheng and Uthus, 2020) datasets. We use SST2 and FinS for training models and use all three for evaluation on each set. This creates one in-distribution (ID) and 2 out-of-distribution (OOD) evaluation datasets.

•

True/False: We use StrategyQA [STF] (Geva et al., 2021) which involves multi-hop reasoning before coming to a True/False conclusion.

•

News Categorization: We use AGNews [AGN] (Zhang et al., 2015) and BBCNews [BBCN] (Greene and Cunningham, 2006). We train on AGN and evaluate on both.

•

MCQ: We use SciQ (Johannes Welbl, 2017), and QASC (Khot et al., 2019). To create additional complexity, we remap the MCQ choices [A, B, C …] to an unrelated token set. Hence, we call the datasets SciQr and QASCr. We train on SciQr and evaluate on both.

Multi-token tasks:

•

Grade-school Math: We use GSM8K (Cobbe et al., 2021), and GSM Symbolic [GSM8Ks]  (Mirzadeh et al., 2024). We train on GSM8K and evaluated on both.

•

Advanced Math: We use MATH Algebra [HMathA] (Hendrycks et al., 2021) which consists of higher grade Algebra problems.

•

Scientific QA: We used SciQ again, but this time for generating the tokens in the the answer instead of choosing between options.

Data Setup: As our focus is on the data-scarce few-shot setting where ICL is typically used, we create multiple training datasets $\mathcal{D}_{\mathcal{T}}=\{(X_{i},Y_{i})\}_{i=1}^{N}$ for each task by varying $N$ in orders of $2:[2,4,8,16,...]$ . We create 5 different sets at each $N$ value to average out outlier effects on our final quantitative analysis. Importantly, we collect ICL activation tensors $A^{i}_{\text{ICL}}\hskip 2.0pt\forall X_{i}$ for IA2 training by using the remaining $N-1$ samples as ICL demonstrations (in random order), effectively reusing training samples. This makes sure that we use the exact same data for all adaptation methods, keeping the comparison fair. For evaluation, we sampled a different set of 500 samples for each task. Additional details about the datasets and how they are processed can be found in §A.

Multi-token details:
Here, the model outputs a sequence of reasoning tokens (not inside the $<$ think $>$ tags, just standard text) before the answer in a specific format illustrated in the demonstrations. The demonstrations and expected output are both chain-of-thought (Wei et al., 2022) style. The actual answer needs to be extracted from the generated text to evaluate model performance and different datasets have different method for parsing the answer from response tokens.

•

For GSM8K and GSM8Ks, the ground truth answers have the following pattern: “ $<$ reasoning steps $>$ #### $<$ numerical answer $>$ ”. We do not evaluate the correctness or conciseness of the reasoning steps and only parse the numerical answer out from the response and match it exactly with the ground truth answer for calculating accuracy. We also use other common answer parsing techniques like matching with “The answer is $<$ value $>$ ”, to allow for slight variations in output format. Our exact parsing function can be seen in our code.

•

For HMathA, the answers are expected to be inside a \boxed{} element and parsed accordingly.

•

For SciQ, we generate the ground truth format similar to the GSM8K pattern, i.e., “ $<$ support text $>$ #### $<$ text answer $>$ ”. We parsed the answers according to this format and considered an exact string match as the only correct answer.

Training Setup: We use two model families to test each training method across all tasks. We use the Qwen3-4B-Base model on every task (single/multi-token). In addition, we also use the Llama-3.2-1B model on every single-token task except SciQr for which we use Llama-3.2-3B. This is because the 1B Llama model was unable to improve above random baseline for any $N$ upto 128. For the multi-token tasks, we use the Llama-3.2-1B-Instruct model as the secondary model. Using an Instruct model tests our method’s efficacy on post-trained models. For training models, we use LoRAs with rank 8 to modify the $W_{Q},W_{K}$ and $W_{O}$ matrices of self-attention layers. We chose these models and weight combinations reasonably on the basis of available resources, as we trained more than 13,000 models in total for our experiments. Training in multi-token setting is much more resource extensive. As each sample has a long answer, we decided to only use $N$ upto 16 for Llama and $N$ upto 8 for Qwen models in multi-token setting. We also fix the maximum generated tokens ( $G=200$ ) during training for efficiency. This is used in IA2 for collecting the activation tensor $\hat{A}_{i}\in\mathbb{R}^{L\times 200\times d}\hskip 2.0pt\forall x_{i}$ . For SFT, $G$ is equal to the number of tokens in the ground truth answer. Each (method, dataset) combination is trained until convergence (upto 50 epochs) with 3 learning rates (slow:1e-4, medium:3e-4, fast:1e-3), and the best performing one is chosen to neutralize the effect of hyperparameter selection on method performance. Additional details in §B.

Evaluation: In the single-token setting, we use accuracy and expected calibration error (ECE) as performance metrics. Given that the model’s answer is a single token, we calculate the accuracy and ECE on the basis of the first generated token’s probability distribution. First, we evaluate the base model with ICL demos to set the ICL performance benchmark. For this, we evaluate the validation set of each task 5 times with different ICL demos to report their average metrics. Then, we evaluate each training method without ICL demos and report their best-performing average metrics. In the multi-token setting, we restricted the maximum generated tokens during evaluation to judiciously use compute resources. We set the limit to 200 for GSM8K and GSM8Ks datasets. This is on the basis of their 95 percentile answer lengths (184 and 154 respectively). We relax this limit to 400 for HMathA, as the 95 percentile answer lengths are 307. For performance, we consider only accuracy, i.e., number of questions answered correctly in the required format (see answer parsing and other details in §A). As the answer confidence is hard to measure for open-ended generation tasks, we do not measure calibration.

### 5.1 IA2 priming boosts SFT performance

Single-Token results:
In Table 1, we report the performance metrics of adapted Qwen models for the practical setting of $N=4$ for single-token tasks. We find that our proposed IA2 → SFT method outperforms SFT only training in almost all cases in terms of both accuracy and calibration, all while using the same amount of data. In most cases, IA2 → SFT also outperforms ICL in terms of accuracy, with slightly less calibration. Notice that IA2 only training reaches high accuracy in many cases, while being reasonably calibrated like ICL, without being trained on target response tokens. This highlights the richness of IA2 training signal. While this table shows the results for one setup,  Figure 1 captures the aggregate performance trends across models and datasets. Detailed tabulated results for some other model/ $N$ combinations, and a statistical significance test of our improvement can be seen in §C.

| Dataset | Adaptation Method

| Source | Eval | ICL | SFT only | IA2 only | IA2 $\rightarrow$ SFT

| acc $\uparrow$ | ece $\downarrow$ | acc $\uparrow$ | ece $\downarrow$ | acc $\uparrow$ | ece $\downarrow$ | acc $\uparrow$ | ece $\downarrow$

|
AGN | AGN | 30.0 (02.3) | 0.13 (0.03) | 27.9 (04.3) | 0.52 (0.12) | 24.0 (00.3) | 0.11 (0.01) | 31.8 (03.8) | 0.35 (0.24)

| BBCN∗ | 28.8 (00.6) | 0.12 (0.01) | 31.8 (07.1) | 0.56 (0.14) | 24.3 (01.3) | 0.10 (0.02) | 27.9 (04.3) | 0.40 (0.21)

| FinS | FinS | 63.6 (01.5) | 0.12 (0.01) | 67.4 (14.1) | 0.31 (0.14) | 63.1 (15.0) | 0.24 (0.06) | 78.7 (13.7) | 0.16 (0.11)

| PoemS∗ | 56.9 (02.2) | 0.12 (0.02) | 49.9 (04.6) | 0.48 (0.04) | 52.8 (05.6) | 0.15 (0.04) | 60.6 (15.7) | 0.36 (0.15)

| SST2∗ | 70.1 (01.3) | 0.17 (0.01) | 59.4 (06.5) | 0.20 (0.10) | 69.8 (13.7) | 0.30 (0.10) | 71.4 (18.9) | 0.20 (0.15)

| SciQr | QASCr∗ | 56.5 (01.4) | 0.08 (0.01) | 76.5 (03.8) | 0.09 (0.05) | 71.2 (03.0) | 0.12 (0.01) | 79.4 (01.8) | 0.09 (0.01)

| SciQr | 87.7 (01.1) | 0.07 (0.01) | 88.3 (04.6) | 0.07 (0.05) | 90.4 (01.5) | 0.09 (0.01) | 91.7 (01.7) | 0.05 (0.01)

| SST2 | FinS∗ | 41.9 (00.3) | 0.19 (0.01) | 68.4 (02.3) | 0.30 (0.04) | 71.3 (02.1) | 0.21 (0.03) | 82.4 (12.2) | 0.11 (0.08)

| PoemS∗ | 65.1 (01.2) | 0.11 (0.02) | 56.5 (13.2) | 0.33 (0.14) | 62.4 (09.2) | 0.19 (0.06) | 68.4 (08.4) | 0.30 (0.08)

| SST2 | 85.4 (00.4) | 0.13 (0.00) | 65.2 (10.7) | 0.22 (0.10) | 82.7 (06.4) | 0.28 (0.03) | 90.4 (02.3) | 0.06 (0.01)

| STF | STF | 66.0 (02.1) | 0.07 (0.01) | 52.4 (04.4) | 0.29 (0.10) | 62.2 (04.1) | 0.16 (0.04) | 62.4 (04.3) | 0.29 (0.08)

Table 1:
Performance report for $N=4$ on Qwen3-4B-Base model, showing accuracy (acc) and Expected Calibration Error (ece). Numbers in parentheses show standard deviations across 5 runs for the best performing learning rate. Best training method shown in bold. (∗) highlights OOD evaluations. Our proposed IA2 → SFT training method outperforms standard SFT across the board.

| Dataset | Adaptation Method

| Source | Eval | w/o ICL | ICL | SFT only | IA2 only | IA2 $\rightarrow$ SFT

| acc $\uparrow$ | acc $\uparrow$ | acc $\uparrow$ | acc $\uparrow$ | acc $\uparrow$

|
GSM8K | GSM8K | 56.4 (00.0) | 76.4 (01.1) | 70.9 (02.8) | 77.4 (02.8) | 73.6 (03.7)

| GSM8Ks∗ | 45.4 (00.0) | 68.4 (01.1) | 64.5 (04.0) | 66.2 (03.7) | 68.8 (05.1)

| HMathA | HMathA | 21.0 (00.0) | 60.4 (02.1) | 50.4 (01.8) | 47.8 (06.0) | 55.3 (02.1)

| SciQ | SciQ | 15.4 (00.0) | 37.5 (01.0) | 35.0 (07.1) | 06.9 (13.8) | 40.8 (07.6)

Table 2:
Performance report for $N=4$ on Qwen3-4B-Base models, on multi-token datasets. Best training method shown in bold. IA2 → SFT outperforms standard SFT across all datasets.

Multi-Token results:
In Table 2, we report the accuracy of adapted Qwen models in the multi-token setting. Like single-token, our proposed IA2 → SFT method outperforms SFT only training, in all cases. It is noteworthy that compared to the single-token setting, IA2 → SFT performs worse than ICL sometimes. We suspect that this is due to imperfect compression of longer contexts in the same small LoRA weight space, which should improve with larger ranks. More tabulated results for other model/ $N$ combinations are present in §C.

### 5.2 Why is IA2 priming important?

Figure 3: We scatter asim $(A,A_{\text{ICL}})$
vs Accuracy/ECE for all training methods to show the impact of asim on performance metrics. Each point corresponds to one training experiment. With increasing similarity, ECE goes down smoothly. But extreme ICL activation alignment may leave some accuracy gains on the table that can be sourced using SFT on ground truth responses.

We investigate several properties of the three training methods: 1) SFT only, 2) IA2 only and 3) IA2 → SFT models, and make the following observations.

Activation Similarity vs Performance:  Figure 2 shows the activation similarity of all 3 methods with ICL. IA2 drastically increases this similarity as expected. Importantly, IA2 → SFT retains higher similarity with ICL activations even after SFT training. In Figure 3 we show the relationship between ICL activation similarity and performance metrics. IA2 → SFT sits comfortably between IA2 and SFT in terms of activation similarity, while achieving better accuracy and calibration. Importantly, note that SFT only and IA2 → SFT training only differs in the start state. This implies that IA2 offers a rich training signal which is unavailable via SFT only training. It is clear that more activation similarity implies better calibration. However, extreme activation similarity may not be the optimal for achieving the best accuracy, as the ICL signal is not always right. In fact, in §C, we show one case where IA2 models follow ICL performance even if becomes worse due to overfitting. Therefore, it is best to combine the two training signals: IA2 to improve the internal functional alignment with ICL and SFT on ground truth responses to align with expected output behavior.

Subspace Overlap: Although activation patterns are a footprint of the model’s functional behavior, it does not isolate changes in behavior induced by different training methods. This is because we use LoRA adaptors and the activation patterns have significant influence from the unchanged base model weights. To isolate the change in behavior, we analyze LoRA weights directly.

Figure 4: IA2 → SFT models have significantly high subspace overlap with IA2 only models, while SFT only models do not align with either. This indicates a training signal in IA2 that is absent from SFT only training and, which is important for performance.

Figure 4shows the weight subspace overlap distribution between pairs of models trained using different methods. We use models from all single/multi token experiments. We first perform SVD on the weight matrices to find basis vectors. Then we calculate subspace overlap by the formula: $\frac{1}{r}*||U_{1}^{T}\cdot U_{2}||_{F}^{2}$ where $U_{1}$ and $U_{2}$ are the basis vectors corresponding to the two methods being compared, and $r$ is the rank of LoRA weights. Subspace overlap measures how much of the weight space is shared between two methods, in effect isolating and comparing the difference in their functional behavior. We find that SFT only training results in weight updates which are almost completely orthogonal to the other two methods. Meanwhile, the most performant IA2 → SFT models share around $39\%$ of their spanned weight space on average with IA2 only trained models. This implies that a lot of the performance gains of IA2 → SFT models are achieved because of IA2, and the subspaces identified by IA2 are practically unreachable by SFT only training. The high spread of subspace overlap between IA2 only and IA2 → SFT models is because in many cases, the IA2 model itself was already output aligned, so the secondary SFT training did not change the model much before convergence, resulting in very high subspace overlap.

## 6 Discussion

IA2 + SFT: One natural alternative to IA2 → SFT training is IA2 + SFT, i.e., using both $\mathcal{L}_{\text{{{IA2}} }}$ and $\mathcal{L}_{\text{SFT}}$ at the same time rather than sequentially. This approach poses a practical challenge. $\hat{A}$ collected as the target activation tensor for IA2 uses the model’s own generated response tokens for any $G>1$ . We call it the ICL response, which could be very different (even in length) from the ground truth response in the dataset. This makes the two objectives incompatible to train together. However, we could perform self-distillation using ICL responses, similar to Snell et al. (2022) but including an additional IA2 signal. We perform this training with a unified loss $\mathcal{L}_{\text{{{IA2}} }}+\beta\cdot\mathcal{L}_{\text{SFT}}$ and vary $\beta$ with 4 values spanning the spectrum between IA2 only and SFT only training smoothly (details in §B). After choosing the best performing (learning rate, $\beta$ ) combination across 5 random seeds, we report the performance for multi-token experiments in Table 3. Note that the performance of SFT only and IA2 → SFT training drops when compared to training on ground truth responses (from Table 2), but IA2 + SFT extracts significantly more performance out of ICL responses when compared to SFT only training. This highlights the richness of the IA2 training signal. Additional results of IA2 + SFT training can be found in §C.

| Dataset | Adaptation Method

| Source | Eval | w/o ICL | ICL | SFT only | IA2 only | IA2 $\rightarrow$ SFT | IA2 + SFT

| acc $\uparrow$ | acc $\uparrow$ | acc $\uparrow$ | acc $\uparrow$ | acc $\uparrow$ | acc $\uparrow$

|
GSM8K | GSM8K | 56.4 (00.0) | 76.4 (01.1) | 66.4 (18.1) | 77.4 (02.8) | 67.2 (22.1) | 77.0 (07.8)

| GSM8Ks∗ | 45.4 (00.0) | 68.4 (01.1) | 59.8 (20.2) | 66.2 (03.7) | 61.6 (24.0) | 69.0 (09.6)

| HMathA | HMathA | 21.0 (00.0) | 60.4 (02.1) | 49.0 (03.0) | 47.8 (06.0) | 54.0 (01.7) | 53.6 (02.7)

| SciQ | SciQ | 15.4 (00.0) | 37.5 (01.0) | 27.2 (12.0) | 06.9 (13.8) | 32.8 (10.0) | 34.5 (10.6)

Table 3:
Performance report for $N=4$ on Qwen3-4B-Base models when trained with ICL responses instead of ground truth responses. IA2 + SFT performs significantly better than SFT only highlighting the significance of IA2.

Knowledge Distillation baseline: A popular approach to distill the behavior of a stronger teacher into the student model is through soft label matching (Hinton et al., 2015). The teacher model provides a denser training signal through the probability mass associated with each token. In our work, we treat the same model (enhanced with ICL context) as the teacher. To test how IA2 based training performs in comparison to soft label matching, we train and evaluate Llama models on GSM8K (1B-Instruct) and SST2 (1B) with soft label matching on ICL responses. In Table 4, we observe that soft-label matching performs significantly better than standard SFT in the multi-token case almost reaching IA2 + SFT performance, but lacks in the single-token case. This highlights the consistency of IA2 based training in achieving high performance.

| Dataset | $N$ | ICL | SFT only | SFT (soft labels) | IA2 only | IA2 $\rightarrow$ SFT | IA2 + SFT

| acc $\uparrow$ | ece $\downarrow$ | acc $\uparrow$ | ece $\downarrow$ | acc $\uparrow$ | ece $\downarrow$ | acc $\uparrow$ | ece $\downarrow$ | acc $\uparrow$ | ece $\downarrow$ | acc $\uparrow$ | ece $\downarrow$

|
GSM8K | 4 | 34.6 (00.9) | – | 28.2 (04.5) | – | 36.1 (03.5) | – | 29.3 (05.0) | – | 30.3 (03.2) | – | 37.0 (01.9) | –

| SST2 | 8 | 77.0 (02.2) | 0.10 (0.02) | 52.3 (04.3) | 0.38 (0.14) | 55.9 (08.8) | 0.16 (0.06) | 64.9 (15.3) | 0.30 (0.05) | 64.7 (19.3) | 0.33 (0.19) | 66.9 (18.4) | 0.28 (0.21)

Table 4: Comparison with Knowledge Distillation (KD) baseline (SFT on soft labels). Although KD baseline performs better than standard SFT in the multi-token case, IA2 based training provides consistently high performance for single/multi token cases.

Why IA2 based training does not consistently beat ICL? In some multi-token cases, like the Qwen model on math datasets (GSM8K and HMathA), we find that ICL performance exceeds all training methods including IA2 based training. We suspect that this is due to mid-training of Qwen on STEM data, making ICL extremely sample efficient (surprisingly, ICL performance for GSM8K at $N=2$ is larger than that at $N=4,8$ ). In Table 5, we illustrate the evolution of performance with shot count. We also highlight the gap between ICL and IA2 → SFT performance, which closes quickly with more shots. Our key takeaway—IA2 → SFT performs better than SFT only—remains consistent as highlighted by the last column.

| Dataset | $N$ | ICL | SFT only | IA2 only | IA2 $\rightarrow$ SFT | Gap wrt ICL | Gap wrt SFT only

|
GSM8K | 2 | 81.2 (00.4) | 65.2 (01.0) | 70.7 (01.8) | 74.6 (02.4) | -06.6 | +09.4

| 4 | 76.4 (01.1) | 70.9 (02.8) | 77.4 (02.8) | 73.6 (03.7) | -02.8 | +02.7

| 8 | 77.2 (00.9) | 73.5 (01.6) | 78.8 (02.5) | 76.0 (02.0) | -01.2 | +02.5

| HMathA | 2 | 58.6 (02.0) | 47.8 (04.0) | 33.9 (06.4) | 48.7 (07.0) | -09.9 | +00.9

| 4 | 60.4 (02.1) | 50.4 (01.8) | 47.8 (06.0) | 55.3 (02.1) | -05.1 | +04.9

| 8 | 62.1 (01.0) | 52.0 (00.8) | 57.3 (03.7) | 59.4 (00.8) | -02.7 | +07.4

Table 5: ICL is more sample efficient than training methods for Qwen models in math datasets. But IA2 $\rightarrow$ SFT quickly closes the gap in performance with increasing number of shots, and consistently beats SFT only training.

Other fine-tuning methods: LoRA is not the only method to fine-tune models. One could perform full fine tuning (full-rank) or other parameter efficient methods like prompt/prefix tuning (Lester et al., 2021; Li and Liang, 2021) and activation scaling ((IA)3 Liu et al. (2022)). Our proposed pipeline is not in competition, rather an improvement over these methods as it provides better training signals. To establish this, we repeat our experiments on a few datasets with the popular PEFT method (IA)3. We followed the exact same methodology from §5, just replaced LoRA training with (IA)3 (details about setup in §B). We report the metrics for one setting in Table 6 which shows trends similar to LoRA. Other results can be seen in §C. We defer the exploration of improvement in other SFT methods through IA2 for future work. On a side note, IA2 also marks a major improvement in method naming compared to (IA)3.

| Dataset | Adaptation Method

| Source | Eval | ICL | SFT only | IA2 only | IA2 $\rightarrow$ SFT

| acc $\uparrow$ | ece $\downarrow$ | acc $\uparrow$ | ece $\downarrow$ | acc $\uparrow$ | ece $\downarrow$ | acc $\uparrow$ | ece $\downarrow$

|
SciQr | QASCr∗ | 35.6 (01.4) | 0.06 (0.01) | 32.8 (03.4) | 0.20 (0.08) | 34.2 (02.7) | 0.08 (0.01) | 48.6 (05.5) | 0.25 (0.11)

| SciQr | 60.0 (01.8) | 0.07 (0.01) | 64.5 (05.0) | 0.12 (0.05) | 63.0 (04.6) | 0.10 (0.04) | 70.6 (04.6) | 0.08 (0.02)

| SST2 | FinS∗ | 56.6 (01.6) | 0.17 (0.01) | 58.2 (10.7) | 0.25 (0.03) | 61.9 (09.5) | 0.18 (0.06) | 59.8 (07.8) | 0.18 (0.03)

| PoemS∗ | 52.3 (03.0) | 0.19 (0.02) | 48.1 (03.2) | 0.44 (0.11) | 48.1 (03.4) | 0.11 (0.04) | 59.8 (13.4) | 0.33 (0.13)

| SST2 | 60.3 (02.8) | 0.19 (0.02) | 52.4 (02.8) | 0.35 (0.05) | 54.2 (04.2) | 0.15 (0.06) | 62.6 (09.1) | 0.28 (0.11)

Table 6:
Performance report for $N=4$ on Llama-3.2 models trained using (IA)3. IA2 → SFT achieves significant gains over SFT only training, exceeding even ICL accuracy in all cases.

Computational Overhead: IA2 → SFT requires a data (activations) collection step before training, where we perform ICL inference on the model with the given dataset. This is a one-time cost to collect the rich ICL activations, which greatly benefits the model’s capabilities during inference that runs at the same cost as SFT only models.

Potential improvements: We made several design choices, which were fixed due to resource constraints, that have potential for exploration and improvements. The most important ones include ablations around LoRA parameters (rank, target modules, etc.) and selective IA2. We showed that extreme alignment can hurt performance and previous work (Todd et al., 2023) has shown that some layers are more sensitive to ICL than others. Therefore, selectively aligning layers can be more effective. Another minor improvement could include prompt optimization before collecting target activations, as ICL performance is sensitive to the order of demos (Lu et al., 2022; Zhao et al., 2021). Lastly, we perform the IA2 → SFT pipeline with fixed learning rates, but the two signals could benefit from finer grained control. We defer these explorations to future work.

## 7 Related Work

ICL vs SFT: Prior works have studied the performance differences between ICL and SFT (Mosbach et al., 2023; Duan et al., 2024), but the verdict is unclear. In our work, we found SFT to perform worse than ICL. Other works have studied the functional similarity (Dai et al., 2022) and differences (Wang et al., 2023) of ICL with SFT. Some works show how adaptation through ICL/SFT can be miscalibrated and how to improve it (Zhang et al., 2024; Li et al., 2025; Li et al., 2024; Xiao et al., 2025). Sia et al. (2024) identify “where”
translation happens in LLMs during ICL through context masking, using activations as a signal for functional behavior similar to our work. Importantly, recent work (Chu et al., 2025; Shenfeld et al., 2025) has shown SFT to be prone to memorization/overfitting and catastrophic forgetting. Our improved SFT pipeline aims to address these problems.

Context Distillation: A large body of work targets to improve LLM adaptation using some form of compression of explicit context signals like us. These range from using intermediate contemplation tokens for compression (Cheng and Van Durme, 2024; Jiang et al., 2025), internalizing context through generating adapters on the fly (Chen et al., 2025; Charakorn et al., 2025), using related text to enhance distillation (Zhu et al., 2025; Choi et al., 2025), or compressing context directly into weights (Shen et al., 2025; Deng et al., 2024; Yu et al., 2024; Snell et al., 2022; Chen et al., 2024b; Shin et al., 2024). These works aim to distill the effect of given contexts on model outputs using token response signals. Our work is complementary to these works and improves the distillation effects through the model’s own ICL processing signals, an idea with supporting prior work (Aguilar et al., 2020; Jin et al., 2024; Yang et al., 2024).

Activation Steering: Lastly, a growing line of work attempts to extract “steering” vectors or weights to apply on targeted locations inside the model to create an intended effect on the model output (Postmus and Abreu, 2024; Stolfo et al., 2025; Caccia et al., 2025; Fleshman and Van Durme, 2024; Arditi et al., 2024). These methods are similar to our work in that they intervene in the activation space to influence the model’s functionality. In contrast to their targeted approach and small application domain, our method aims to incite a large behavioral change in the model which avoids potential brittleness and reliability issues (Queiroz Da Silva et al., 2025).

## 8 Conclusion and Future Work

In this work, we used model activations to probe the functional behavior of LLMs and found a key distinction: ICL and SFT achieve adaptation through different mechanisms, and ICL often encodes richer, more generalizable patterns. Motivated by this, we introduced IA2, an SFT priming technique that enables ICL-like internal behavior in the model. This simple step shifts models into a more adaptable weight subspace, which is inaccessible to SFT alone. Extensive experiments show that IA2 consistently improves the accuracy and calibration of SFT. In the future, we aim to test other effects like impact on diversity and catastrophic forgetting of IA2 based SFT to measure how IA2 can help in post-training LLMs. We also aim to study the effect of IA2 on post-trained (RL-tuned) models.

## Reproducibility Statement

We provide codes used for data download and processing, training, evaluation, analysis, plotting, etc. on github. All details in this paper (§5 and the appendix) along with a detailed README file should be sufficient to fully reproduce our results as we used repeatable random seeds.

## Acknowledgments

This work is supported by ONR (N00014-24-1-2089), an Amazon Research Award, and a grant from Open Philanthropy. We thank Andrea Wynn for their constructive feedback on earlier versions of this document.
