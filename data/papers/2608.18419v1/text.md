\correspondingauthor

# Mechanistic Interpretability of Structure-Aware Numerical Reasoning in LLaMA 3.1–8B

Conference: The fourth International Workshop on Rich Media with Generative AI ; November 10–14, 2026; Rio de Janeiro, BrazilThe fourth International Workshop on Rich Media with Generative AI (RichMediaGAI ’26), November 10–14, 2026, Rio de Janeiro, BrazilDOI: 10.1145/3841458.3841545ISBN: 979-8-4007-2947-8/2026/11CCS: Computing methodologies Natural language processingCCS: Time SeriesCCS: Mechanistic Interpretability

Rahul Chowdhury

Affiliation: Northeastern University, Boston, MA, USA

email: chowdhury.rah@northeastern.edu

,
Timothy A Rupprecht

Affiliation: EmbodyX Inc., San Mateo, CA, USA

email: tarupprecht@gmail.com

,
Senhao Cao

Affiliation: Northeastern University, Boston, MA, USA

email: senhao.cao@gmail.com

,
Jiahao Liu

Affiliation: Mobi.ai, Boston, MA, USA

email: jiahao@takemobi.com

,
Octavia Camps

Affiliation: Northeastern University, Boston, MA, USA

email: O.Camps@northeastern.edu

,
David Bau

Affiliation: Northeastern University, Boston, MA, USA

email: davidbau@northeastern.edu

,
Pu Zhao

Affiliation: Northeastern University, Boston, MA, USA

email: p.zhao@northeastern.edu

and
Yanzhi Wang

Affiliation: Northeastern University, Boston, MA, USA

email: yanzhiwang@northeastern.edu

© cc

###### Abstract.

Recent work has shown that large language models (LLMs) exhibit strong numerical sequence modeling capabilities and show promise in time-series prediction. While LLMs display in-context learning capabilities, the mechanisms with which they accomplish time-series prediction remain unclear. Specifically, whether they truly understand the underlying structure, which at a minimum requires
reasoning over first differences in the sequence of numbers. To study this, we investigate Llama 3.1-8B from a mechanistic interpretability point of view. Mechanistic interpretability is an emerging field concerned with the reverse engineering of the algorithms learned by neural networks such as LLMs. To assess Llamas’ numerical sequence modeling capabilities and to facilitate our mechanistic interpretability analysis, we create a sequence modeling task that cannot be solved without picking up structural cues. Specifically, we sample $n$ random numbers and repeat them with an offset. We find that Llama displays strong performance on our tasks suggesting that it can pick up on the underlying structure. To understand the mechanisms that allow it to do so, we perform probing experiments and activation patching based counterfactual analysis. Probing reveals that the model computes and stores first differences in its internal representations without explicit supervision, indicating that it tracks structural information about the sequence. Activation patching reveals that Llama retrieves the relevant first-difference with a mechanism similar to an induction circuit and subsequently adds it to the current value. Notably, our work represents one of the first studies to identify this form of concept induction in LLMs.

††cc-license: by

## 1. Introduction

Figure 1.
Illustrates how the model identifies a repeated first-difference (delta) pattern after differencing, then extrapolates by locating and copying the delta that immediately follows the final observed one, and adding it to last number.

Large Language Models (LLMs) can zero-shot extrapolate sequences, and generate novel sequences and functions in response to complex queries (Mirchandani et al., 2023; Shen et al., 2025d; Shen et al., 2024; Zhao et al., 2024b; Zhan et al., 2024c; Bubeck et al., 2023). How do they do this? Did they encounter these trends during training and simply extrapolate using a rule learned specifically for those patterns? Or did they learn to infer the underlying structure of sequence data more generally?

If LLMs can understand the ordinal structure of numerical data — identifying a hidden pattern, reasoning about it, and applying a consistent extrapolation algorithm — the implications extend well beyond memorization. It would suggest these models are learning abstract structural reasoning, a capability with relevance far beyond language: reasoning over time-series data, understanding geometric relationships between image pixels, or maintaining structural coherence across connected points in 3D data (Shen et al., 2025a; Shen et al., 2026; Shen et al., 2025e; Shen et al., 2025b; Shen et al., 2025f; Shen et al., 2025c; Shen et al., 2025d; Zhan et al., 2024a; Zhan et al., 2024b).

In this work, we study the LLaMA 3.1–8B model through the lens of mechanistic interpretability. To probe its structural reasoning capabilities, we introduce a non-trivial sequence modeling task, shown in Figure 2, that cannot be solved via token-level copying through induction heads (Elhage et al., 2021). Our interpretability analysis combines linear probing with activation patching. Probing shows that first-difference representations are locally stored and linearly decodable across layers. Patching experiments further reveal that the model identifies structurally critical tokens, retrieves the correct offset through induction over these latent differences, and performs the final arithmetic step by adding that offset to the last number in the sequence. These computations are precise, localized, and structurally grounded.

To our knowledge, this is the first mechanistic interpretability study to uncover an internal mechanism by which an LLM performs induction over latent structure in numerical sequences—where extrapolation arises from arithmetic operations over internal representations retrieved through, and we illustrate it in Figure 1.

Figure 2.
(a) Shows the input signal, which consists of all unique values which means no number appears twice in the sequence. (b) Shows the deltas of the input signal, revealing that a cycle or pattern emerges only after applying the differencing operation.

## 2. Related Work

Mirchandani et al. (2023) showed that LLMs can perform sequence completion tasks, attributing this ability to in-context learning. However, they did not examine whether such performance stems from memorization, nor did they investigate the underlying mechanisms or the model’s understanding of temporal structure. (Gruver et al., 2024) proposed LLM-Time that explored ways to make a pre-trained LLMs fit for time-series forecasting, demonstrating that pretrained LLMs can perform well on time-series forecasting tasks without fine-tuning.
Time-GPT (Garza and Mergenthaler-Canseco, 2023) is a foundation model trained exclusively on time-series data, and Lag-Llama (Rasul et al., 2023) is another time-series-specific transformer. While these models highlight effectiveness of transformers for sequence modeling (Zhao et al., 2024a; Zhan et al., 2021; Wu et al., 2022; Yang et al., 2023; Li et al., 2022; Li et al., 2023; Lin et al., 2025; Zhao et al., 2025b; Liu et al., 2025; Guan et al., 2021; Wang et al., 2018), they fall outside our study scope, as we focus on general-purpose LLM.

Akyürek et al. (2022) and Kantamneni et al. (2024) examined in-context learning using toy models but did not explore whether LLMs can infer sequential structure in tasks they were not explicitly trained for.
(Lan et al., 2023) applied mechanistic interpretability to identify shared circuits in simple sequence continuation tasks, such as extending short increasing number sequences. While (Lan et al., 2023) demonstrated LLMs’ ability to handle familiar patterns, it did not address whether these models can reason over more abstract or irregular numerical structures. In contrast, our work investigates whether a LLM can recognize and extrapolate more complex sequential structures—such as sequences defined by arbitrary first differences. This goes beyond surface-level extension and explores the model’s ability to infer and extend abstract structure in number sequences.

## 3. Problem Setting

### 3.1. Dataset

We design a dataset to have strings containing numbers followed by a comma, and each instance finishes with a comma so that the predicted next token is a number. The numbers in the whole dataset were between 0 and 999. We construct each sequence with first differences from integers in the set containing all integers between $-9$ and $9$ inclusive, excluding $0$ . Each sequence consists of unique numbers whose deltas are also unique up to the 17th position, after which the delta repeats.

We design the dataset to specifically test whether the LLM can uncover latent numerical structure and specifically track delta within a sequence.
As we illustrate in Figure 2 and in an abbreviated form in Table 1, each sequence comprises two segments: the first is a random walk with unique deltas and no apparent pattern; the second segment reuses the same set of deltas, in the same order, as the first segment. This structure ensures that the model must recognize the pattern in delta and identify the unique location from which the model can copy the delta from.

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 23 | 24 | 25 | 26 | 27 | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 55 | 56 | 57 | 58 | 59

| Token | S | 403 | , | 405 | , | 400 | 411 | , | 417 | , | 409 | 416 | , | 436 | , | 438 | , | 433 | 444 | , | 450 | , | 442

| Delta
|
|
| | 2
| | -5 | 7
| | 6
| | -8 | -2
| | 20
| | 2
| | -5 | 7
| | 6
| | -8

Table 1.
Table shows selected index positions, tokens, and their corresponding first differences from the tokenized input. A recurrence in the first difference after token 33 marks the onset of a repeating cycle.

Algorithm 1 Predict Next Number via First-Difference Pattern Detection

Input: Token sequence $S=[\text{BOS},s_{1},s_{2},\ldots,s_{n}\texttt{,}]$

Output: Predicted next number $s_{n+1}$

1:

 Compute first differences:

2:

  

$D=[s_{2}-s_{1},\ s_{3}-s_{2},\ \ldots,\ s_{n}-s_{n-1}]$

3:

 Identify repeating pattern $P$ in $D$

4:

 Locate the index $k$ where $P$ first appears in $D$

5:

 Determine the phase position $p$ of $s_{n}$ within pattern $P$

6:

 Let $d_{p}=D[p]$ be the corresponding first difference

7:

 Compute prediction: $s_{n+1}=s_{n}+d_{p}$

8:

 return $s_{n+1}$

### 3.2. Model

We conduct all experiments using LLaMA 3.1–8B (Grattafiori et al., 2024), a transformer (Vaswani, 2017) based model with 32 layers and 32 attention heads per layer. One key reason for choosing this model is that its tokenizer represents each integer from 0 to 999 as a single token. This property makes the analysis more tractable by reducing the number of tokens per sequence, thereby simplifying both intervention and observation during interpretability experiments. As a result, LLaMA 3.1–8B was used consistently across all experiments.

### 3.3. Software

We extensively use NNsight and NDIF framework (Fiotto-Kaufman et al., 2024) for tracing activations, applying interventions, and analyzing internal representations of the LLaMA 3.1–8B model.

### 3.4. Performance Evaluation

We evaluate LLaMA 3.1-8B on the sequence prediction task as shown in Figure 1 and Figure 2, and we formally describe it in Algorithm 1. We prompt the model to predict the 30th number given the first 29 elements of each input sequence. The evaluation dataset consists of 10,000 sequences, each formatted as a comma-separated string of integers. Every instance ends with a comma, indicating that the next token to be predicted should be a number.

Due to this formatting, we tokenize each number and each comma separately, as we illustrate in abbreviated form in Table 2. Including the initial special token, the model processes 59 tokens before generating the 60th token as output. This prediction corresponds to the 30th number in the sequence.

#### 3.4.1. Metrics

We employ two complementary evaluation metrics:

Mean Absolute Error (MAE) quantifies average prediction error:

| (1)
| | $\text{MAE}=\frac{1}{n}\sum_{i=1}^{n}\left|y_{i}-\hat{y}_{i}\right|,$
|

where $y_{i}$ and $\hat{y}_{i}$ denote the ground truth and predicted values, respectively, and $n$ is the number of instances.

Coefficient of Determination ( $R^{2}$ ) measures explained variance:

| (2)
| | $R^{2}=1-\frac{\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^{2}}{\sum_{i=1}^{n}(y_{i}-\bar{y})^{2}},$
|

where $\bar{y}$ is the mean of ground truth. Values close to 1 indicate strong predictive performance, while values near 0 or negative suggest performance close to or worse than a naive mean predictor.

#### 3.4.2. Results

LLaMA 3.1-8B achieves an MAE of 4.2748 and an $R^{2}$ of 0.9958 on our evaluation dataset. The high $R^{2}$ value indicates that the model explains approximately 99.58% of the variance in the target sequences, demonstrating strong pattern recognition capabilities for numerical sequence prediction.

## 4. Is Numerical Information Represented Locally in the Model?

Figure 3. Layerwise linear decodability of first differences and final numbers. Mean absolute error (MAE; top) and coefficient of determination ( $R^{2}$ ; bottom) for linear probes trained to decode delta (left) at 27th position and final number (right) at 57th position from hidden states across layers. Solid green lines denote performance on true targets; dashed red lines denote control probes trained on shuffled labels. Delta show consistently strong linear decodability, while final number representations gradually lose decodability.

To predict the next number, the model must add the final observed value to a previously seen first difference. This raises a core interpretability question: does the model explicitly encode these numerical components in structured, localized representations, or does it rely solely on implicit pattern recognition?

We use probing experiments to assess the representational content of the model’s hidden states. Unlike patching—which tests whether an activation is functionally necessary by measuring how altering it affects the model’s output—probing asks whether specific information is present in the representation, regardless of whether the model ultimately uses it. Here, we probe for the presence and location of numerically meaningful quantities in the hidden layers by training linear regressors to decode two key values: (1) the first difference from the 27th token position (where it is expected to be stored), and (2) the final number from the 57th token position. If successful, this would indicate that the model explicitly encodes these arithmetic components in localized, interpretable forms, even without task-specific supervision, and uses these interpretable representations for prediction.

### 4.1. Probing Setup

We collect hidden states from two key token positions: the 27th token position, which is expected to encode the first difference that must be added to the final number to predict the next number, and the 57th token position, which contains the final number in the input sequence. At each layer, we train linear regression models using ordinary least squares on 4,500 training samples and 500 held-out samples for testing:

| (3)
| | $\small\hat{\boldsymbol{\beta}}=\operatorname*{arg\,min}_{\boldsymbol{\beta}}\left\|\mathbf{Y}-\mathbf{X}\boldsymbol{\beta}\right\|^{2},$
|

where $\mathbf{X}\in\mathrm{R}^{n\times d}$ is the matrix of hidden states (with $d=4096$ dimensional representations across $n=4500$ training examples), $\mathbf{Y}\in\mathrm{R}^{n}$ is the target vector (either the first difference or the final number), and $\boldsymbol{\beta}\in\mathrm{R}^{d}$ is the learned weight vector.

To establish baselines, we implement control conditions that test whether observed decodability reflects genuine localization or spurious correlations. For first difference decoding, we train control probes using activations from the 57th token to predict the first difference of 27th position, testing whether first difference information is accessible from positions nearer to the last token position. For final number decoding, we train control probes using activations from the 27th token to predict the number at the 57th position. These ensure that successful decoding reflects position-specific encoding rather than global numerical information distribution.

### 4.2. Results

Figure 3 presents the results of probing experiments. The top row reports the MAE of the linear regression model on held-out test samples, while the bottom row shows the $R^{2}$ . We compute these metrics at each layer to evaluate the linearity of the target quantities throughout the model’s depth.

The results confirm that both the first difference (at the 27th token position) and the final number (at the 57th token position) are linearly decodable from the hidden states, as evidenced by low MAE values and high $R^{2}$ scores approaching 1. Notably, the first difference becomes linearly accessible starting after the 7th layer, where the MAE drops sharply and approaches its minimum, while the $R^{2}$ score increases and stabilizes above 0.8. Although performance slightly decays with depth, it remains relatively stable across the first 16 layers.
In contrast, the linear decodability of the final number at the 57th position shows a gradual decline across deeper layers, with MAE increasing and $R^{2}$ decreasing after layer 8. This suggests that the final number representation becomes increasingly entangled with other contextual information as it propagates through model.

Crucially, the control conditions demonstrate substantially worse performance across all layers for both targets. For first differences, control probes using 57th token activations achieve negative $R^{2}$ values and high MAE, indicating no meaningful linear relationship. Similarly, control probes attempting to decode final numbers from 27th token activations show consistently poor performance. This confirms that the observed decodability is position-specific and not due to general numerical information in the representations.

Together, these results indicate that the numerical components necessary for the model’s arithmetic composition-the first difference and the final number-are linearly encoded in specific locations.

## 5. Is the Next Number Computed by Adding a Retrieved First Difference?

Figure 4.
Illustrates the query-hidden state patching technique that we use to test whether the model performs arithmetic composition by retrieving a first difference from an earlier position. The clean sequence (bottom) contains a structured delta pattern, while the corrupt sequence (top) is flat-valued. We transplant activations from the clean run into the corrupt run from a given layer $l$ onward. We patch hidden states (shown with blue arrows) at layers $l$ to $L$ at one token position at a time except the beginning-of-sequence and final two tokens. We patch the query vector of the final token (shown with orange arrows) from the clean run at layers $l+1$ to $L$ , ensuring the model attends to the correct location encoding the target first difference. This setup isolates and tests the causal contribution of induction and addition behaviors in the model’s prediction pipeline.

Having established that the model encodes first differences and final numbers in localized representations, we now ask whether it functionally uses those representations to perform arithmetic composition. We test whether the model retrieves a previously seen first difference from a particular token position and perform arithmetic operation on it.

To answer this question, we use activation patching
(Wang et al., 2023; Meng et al., 2022; Vig et al., 2020; Zhao et al., 2025a; Zhao et al., 2026a; Zhao et al., 2026b; Goldowsky-Dill et al., 2023; Finlayson et al., 2021), following the patching methodology of (Dumas et al., 2024). We transplant activations from a clean sequence into a corrupt one to determine if the model retrieves the delta from the hypothesized position (delta that follows the latest one) and adds that to a new number of an unseen sequence.

### Clean vs. Corrupt Sequence

We define two types of input sequences:

•

Clean sequence: $x^{\text{clean}}=(x_{1},\dots,x_{29})$ , where the deltas $\Delta_{i}$ begin to repeat after $i=17$ , and the model predicts: $x_{30}=x_{29}+\Delta^{*}$ where $\Delta^{*}$ is the difference retrieved from an earlier position. In tokenized form, $x_{29}$ appears at token 57, and the output $x_{30}$ is generated at token 58.

•

Corrupt sequence: $x^{\text{corrupt}}=(100,100,\dots,100)$ , so that $\Delta_{i}=0$ for all $i$ .

What makes the corrupt sequence particularly well-suited for evaluating the extraction and addition of the first difference is its deliberately flattened structure, which eliminates any intrinsic gradient or variation in the input. The corrupt sequence serves as an ideal control, as it lacks the algorithm present in the clean sequence, allowing us to test whether the model can still retrieve and apply a meaningful first difference from an alternate source.

### 5.1. Patching Setup

Let $H_{t}^{(l)}$ and $Q_{t}^{(l)}$ denote the hidden state and query vector at token $t$ , layer $l$ . We patch hidden states to all tokens except the beginning token, the final number token (57), and the final token (58):

| | $H_{t}^{\text{corrupt},(l:L)}\leftarrow H_{t}^{\text{clean},(l:L)}$
|

The query vector of final token (58) is patched from one layer above:

| | $Q_{58}^{\text{corrupt},(l+1:L)}\leftarrow Q_{58}^{\text{clean},(l+1:L)}$
|

In particular, following (Dumas et al., 2024), we patch layer $l$ and all subsequent layers, as we illustrate in Figure 4. Patching downstream layers ensures that transient causal signals are preserved and not suppressed as noise by later layers, thereby, improving sensitivity to the influence of critical nodes on the final prediction.

For the final token, we specifically patch the query vector across layers $l+1$ through $L$ . This is necessary because the corrupt sequence does not learn in-context to retrieve target first differences from earlier tokens. By patching the query vector of the final token from the clean forward pass, we ensure that the model attends to the position encoding the appropriate first difference, thereby, facilitating the intended retrieval behavior.

### 5.2. Causal Metric

To quantify causal influence, we use difference in predicted probabilities of counterfactual label $y_{\text{cf}}=100+\Delta^{*}$ as evaluation metric:

| (4)
| | $\Delta P=\mathrm{E}_{x\sim\mathcal{D}}\left[P_{\text{patched}}(y_{\text{cf}}\mid x)-P_{\text{corrupt}}(y_{\text{cf}}\mid x)\right]$
|

where $P_{\text{patched}}(y_{\text{cf}}\mid x)$ is the probability assigned to the counterfactual label $y_{\text{cf}}$ under the patched forward pass, and $P_{\text{corrupt}}(y_{\text{cf}}\mid x)$ is the corresponding probability under the corrupt forward pass. A higher $\Delta P$ indicates stronger causal contribution of the patched components to the model’s ability to predict the correct label.

### 5.3. Results

Figure 5 presents the results of the patching experiment conducted on 100 data instances with zero MAE to ensure a strong response signal. We observe a significant increase in the probability difference score ( $\Delta P$ ), highlighted in red in Figure 5, at the 27th token position, beginning from layer 14 onward. This indicates that, from layer 15 onward, the query vector of the final token (position 58) causally interacts with the hidden state at position 27. The arithmetic operation is highly localized: only at these layers does the model cleanly separate the corrupt final number and compose it with the correct first difference. Earlier interventions disrupt this alignment due to representational entanglement between the corrupt and patched sequences, which impairs computation.

This finding supports the hypothesis that the model performs induction over first differences by locating and copying a locally stored delta representation—analogous to the [A][B]…[A][?] pattern discussed in (Elhage et al., 2021)—that follows the final observed one, though this behavior operates at a structural rather than token level. The arithmetic operation that follows induction—adding the copied delta to the final number at position 57 to generate the next token—is generalizable, as the model performs it even in the absence of a guiding algorithm in the corrupt sequence. This confirms that the model engages in both induction and arithmetic composition in numerical sequence extrapolation.

Figure 5.
Heatmap of probability difference scores ( $\Delta P$ ) across layers and token positions, following patching of the final query vector from layers $l{+}1$ to $L$ and hidden states from layers $l$ to $L$ . A sharp causal effect emerges at token 27, highlighted in red, indicating that the final query attends to position 27 to retrieve the locally stored first difference that follows the last observed one.

## 6. Can the Model Identify Functionally Critical Tokens for Delta Retrieval and Extrapolation?

Although previous experiments demonstrate that the model stores first differences in localized representations and retrieves the appropriate ones for arithmetic composition, they do not tell if the model uncovers the underlying algorithm. In particular, uncovering requires identifying the onset of a repeating pattern, aligning it with the final observed difference, and selecting the correct delta to apply.
To investigate this, we use patching to isolate the causal contribution of each attention head and assess which heads drive successful extrapolation by routing value from appropriate positions.

### Patching Setup

We use the same clean and corrupt sequences described in the previous section, where the clean input contains a repeating first-difference pattern and the corrupt input is a flat sequence of 100s. The target counterfactual label remains: $y_{\text{cf}}=100+\Delta^{*}$ .

Let $Q_{s}^{(l,h)}$ , $K_{t}^{(l,h)}$ , $V_{t}^{(l,h)}$ , and $W_{O}^{(l,h)}$ denote the query, key, value vectors, and output projection matrix for head $(l,h)$ in layer $l$ at token $s$ attending to token positions $t$ . The attention weights at head $(l,h)$ are computed as:

| | $\alpha_{t,s}^{(l,h)}=\mathrm{softmax}\left(\frac{(Q_{s}^{(l,h)})^{\top}K_{t}^{(l,h)}}{\sqrt{d_{h}}}\right)$
|

The corresponding output of attention head $(l,h)$ at token $t$ is:

| | $z_{s}^{(l,h)}=W_{O}^{(l,h)}\left(\sum_{t=1}^{T}\alpha_{t,s}^{(l,h)}\cdot V_{t}^{(l,h)}\right)$
|

To evaluate the causal effect of head $(l,h)$ at the final token position $s=58$ , we patch just its output, as we illustrate in Figure 9 in the appendix , from the clean run into the corrupt run:

| | $z_{58}^{\mathrm{corrupt},(l,h)}\leftarrow z_{58}^{\mathrm{clean},(l,h)}$
|

This setup isolates the contribution of a single attention head by substituting its output at the final token while preserving the rest of the corrupt context.

### 6.1. Causal Metric

To determine which tokens most significantly contribute to the model’s prediction, we construct a profile of attention heads that integrates two key signals: (1) the causal effect of each head, and (2) the value-weighted attention mass it allocates across token positions. We compute the causal effect as the change in the model’s predicted probability for the counterfactual label when we patch head’s output at position $s=58$ from the clean to the corrupt run, as we describe in the previous section. The second component—the value-weighted attention— identifies how much of its value is added to the residual stream through this head.

We define the token importance score $\mathrm{TI}_{t}$ as:

| (5)
| | $\small\mathrm{TI}_{t}=\frac{1}{N}\sum_{n=1}^{N}\lambda_{n}^{(l,h)}\cdot w_{n,t}^{(l,h)},$
|

where $N$ is the number of evaluation samples, $\lambda_{n}^{(l,h)}=P_{\mathrm{patched}}(y_{\mathrm{cf}}\mid x_{n})-P_{\mathrm{corrupt}}(y_{\mathrm{cf}}\mid x_{n})$ is the causal effect of patching head $(l,h)$ in sample $n$ , and $w_{n,t}^{(l,h)}$ is the value-weighted attention score at the attended position $t$ .
The value-weighted attention score $w_{n,t}^{(l,h)}$ is defined using the attention weights and value norms:

| (6)
| | $w_{n,t}^{(l,h)}=\alpha_{t,s}^{(l,h)}\cdot\left\|V_{t}^{(l,h)}\right\|_{2},$
|

This metric $\mathrm{TI}_{t}$ weights causal influence with value-aware attention, providing a principled measure of token-level functional relevance to the model’s output.

### 6.2. Results

Figure 6 presents the results of the causal attention-based analysis over 100 sequences with zero MAE, and the top five most relevant tokens are identified as positions 57, 33, 27, 55, and 25. As we show in Table 1, these correspond to the critical points that the model needs for identification and extrapolation of the pattern: the final number (57), the onset of the repeating delta pattern (33), the delta that gets added to the final value (27), the number that the model needs to compute the last delta (55), and a token (25) that both matches the final delta and enables retrieval of the next one.

These results also suggest the model emphasizes tokens that support first-difference alignment, with token 27 playing a key role in extrapolation. The pattern of attention supports the hypothesis that the model recognizes the pattern and must have first used the final first difference as a retrieval anchor to locate the appropriate delta, and, upon recognizing the repeating structure, may further leverage phase alignment and positional encoding to guide delta selection.

Figure 6.
Token importance scores ( $\mathrm{TI}_{t}$ ) across sequence positions. Peaks highlight the model’s use of structural alignment for pattern detection and delta retrieval.

## 7. Does the Model Prioritize Delta Induction Over Positional Cues?

Prior experiments demonstrate that the model stores and composes first differences to extrapolate future values. However, they do not reveal whether retrieval is governed by semantic identity of delta or by its positional and phase-based alignment in input.

To investigate this, we perform a key swapping intervention between the 27th and 25th tokens—two positions with high and comparable importance scores as shown in Figure 6.

### 7.1. Rotary Position Embedding (RoPE)

Let $x\in\mathbf{R}^{d}$ denote an input vector, where $d$ is even. Following the formulation introduced by (Su et al. 2024)(Su et al., 2024), we partition $x$ into $d/2$ adjacent 2D pairs:

| | $x=\left[\,(x_{1}^{(1)},x_{1}^{(2)}),\,(x_{2}^{(1)},x_{2}^{(2)}),\,\dots,\,(x_{d/2}^{(1)},x_{d/2}^{(2)})\,\right]$
|

Let $W_{q},W_{k}\in\mathbf{R}^{d\times d}$ be the query and key projection matrices. For position $m$ , the RoPE-transformed query is:

| | $Q_{m}=\text{RoPE}(W_{q}x,m)=\bigoplus_{i=1}^{d/2}R(m\theta_{i})\,(W_{q}x)_{i}$
|

and similarly, for position $n$ , the RoPE-transformed key is:

| | $K_{n}=\text{RoPE}(W_{k}x,n)=\bigoplus_{i=1}^{d/2}R(n\theta_{i})\,(W_{k}x)_{i}$
|

Here, $R(m\theta_{i})$ is a 2D rotation matrix applied to the $i$ -th 2D component:

| | $R(m\theta_{i})=\begin{bmatrix}\cos(m\theta_{i})&-\sin(m\theta_{i})\\
\sin(m\theta_{i})&\cos(m\theta_{i})\end{bmatrix}$
|

Here, $R(n\theta_{i})$ is a 2D rotation matrix applied to the $i$ -th 2D component:

| | $R(n\theta_{i})=\begin{bmatrix}\cos(n\theta_{i})&-\sin(n\theta_{i})\\
\sin(n\theta_{i})&\cos(n\theta_{i})\end{bmatrix}$
|

The operator $\bigoplus$ denotes vector concatenation across the $d/2$ rotated 2D components:

| | $\bigoplus_{i=1}^{d/2}v_{i}=[v_{1};\,v_{2};\,\dots;\,v_{d/2}]$
|

where each $v_{i}\in\mathbf{R}^{2}$ , yielding a final vector in $\mathbf{R}^{d}$ .

The query and key vectors are independently rotated using their respective absolute positions $m$ and $n$ , and the relative positional information gets encoded through their dot product $Q_{m}^{\top}K_{n}$ .

### 7.2. Patching Setup

LLaMA 3.1-8B uses rotary position embedding (RoPE) (Su et al., 2024) where positional information is local to key and query.
We swap Keys $K_{27}^{\text{clean},(l:L)}\leftrightarrow K_{25}^{\text{clean},(l:L)}$ in layer $l$ and all subsequent layers to redirect positional and phase-based cues while preserving $V_{27}^{\text{clean},(l:L)}$ and $V_{25}^{\text{clean},(l:L)}$ by patching the values from the original run to maintain their first difference content across layers $l+1$ through $L$ , as we illustrate in Figure 10 in the appendix. This setup attempts to disentangle two types of cues bound to the delta representation.

### 7.3. Causal Metric

To quantify the causal impact of an intervention, we measure the change in the model’s confidence for both the counterfactual label $y_{\text{cf}}=x_{29}+\Delta_{25}$ and the ground truth label $y_{\text{gt}}=x_{29}+\Delta_{27}$ . The causal effect on the counterfactual label is defined as:

| (7)
| | $\Delta P_{\text{cf}}=\mathrm{E}_{x\sim\mathcal{D}}\left[P_{\text{patched}}(y_{\text{cf}}\mid x)-P_{\text{corrupt}}(y_{\text{cf}}\mid x)\right],$
|

and for the ground truth label:

| (8)
| | $\Delta P_{\text{gt}}=\mathrm{E}_{x\sim\mathcal{D}}\left[P_{\text{patched}}(y_{\text{gt}}\mid x)-P_{\text{corrupt}}(y_{\text{gt}}\mid x)\right].$
|

We also report the absolute probabilities assigned after the intervention: $P_{\text{patched}}(y_{\text{cf}}\mid x)$ and $P_{\text{patched}}(y_{\text{gt}}\mid x)$ .
These metrics together capture both the direction and magnitude of the model’s response to the intervention.

### 7.4. Results

Figure 7 presents the probability difference following the key-swapping intervention $K_{27}^{\text{clean},(l:L)}\leftrightarrow K_{25}^{\text{clean},(l:L)}$ that we conduct over 100 instances with zero MAE. The intervention leads to a significant increase in $\Delta P_{\text{cf}}$ and a corresponding drop in $\Delta P_{\text{gt}}$ , suggesting that phase or position-based cues contribute meaningfully for delta retrieval and composition. However, as shown in Figure 8, which reports the absolute probabilities after intervention, the drop in $P_{\text{patched}}(y_{\text{gt}})$ was not sufficient for $P_{\text{patched}}(y_{\text{cf}})$ to overtake it. This indicates that although key-based redirection influences the model’s behavior, it is ultimately not strong enough to override the original delta-based composition since the value vector must contain information that this first difference comes after the last observed one. The model’s affinity for delta-based composition supports the hypothesis that it first performed induction over first differences to learn the underlying structure: internally simulated the composition process, and identified the correct algorithm, potentially refining it through phase-based alignment.

Figure 7. Shows key-swap intervention shifts confidence from ground truth to counterfactual, revealing the role of position-aware attention alignment in delta retrieval.

Figure 8. Shows ground truth remains more probable post key-swap, indicating the model prioritizes delta-based composition over redirected attention.

## 8. Conclusion

This work demonstrates the LLM, despite not being explicitly trained for high-precision numerical tasks, exhibit an emergent ability to perform structural extrapolation in numerical sequences without any explicit supervision. We show that the LLM is capable of recognizing trends, inferring position-dependent rules encoded as distinct first differences, and systematically composing these rules to generate accurate predictions over evolving input sequences. Our analyses reveal that the LLM exhibit reasoning behaviors, operating over latent structure. Patching provided strong evidence that the model identifies numerical patterns, computes first differences even before recurring structures emerge, stores these differences locally, and later retrieves them to generate accurate predictions. Probing confirmed that the observed effects are not incidental but reflect functional computations over structured internal representations.

We believe these findings would invite future work on extending these insights to other reasoning tasks where latent structure plays a critical role.

## Appendix

## Appendix A Illustrating Patching Experiments

Figure 9.
Illustrates the head-level patching intervention that we use to isolate the functional role of individual attention heads in pattern recognition and delta retrieval. The clean sequence (bottom) contains structured first-difference patterns, while the corrupt sequence (top) is flat-valued. At each layer $l$ , we patch only the output of a single attention head (at the final token position) from the clean forward pass into the corrupt one (shown with orange arrow), while all other heads and components remain unchanged. This targeted intervention tests whether the selected head causally contribute to the recognition of repeating structure, retrieval of a critical first difference, and contributes to accurate extrapolation in the final prediction.

Figure 10.
Illustrates the key-swapping intervention that we use to disentangle the model’s reliance on phase (positional) cues versus delta-based representations during extrapolation. We swap keys from token positions 25 and 27—both identified as causally important—are swapped from layer $l$ onward (indicated by orange and gray arrows), while preserving the respective value vectors from layer $l$ onward. This setup tests whether the model selects the correct first difference for addition based on the original delta identity stored in the value vector or shifts its prediction in response to the redirected positional cue, thereby revealing its bias toward phase.
