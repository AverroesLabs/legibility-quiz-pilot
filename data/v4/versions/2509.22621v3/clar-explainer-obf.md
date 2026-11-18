The paper asks whether activations—context-dependent internal processing outputs—from in-context learning, which adapts a model through examples without changing its weights, can improve supervised fine-tuning, which adapts weights using labeled examples. Its ICL Activation Alignment (IA2) method first trains a model to reproduce these activations, then trains it on target responses. Across 12 benchmarks and two model families, the paper reports improved accuracy and calibration over supervised fine-tuning alone. Activation comparisons and weight-space analysis indicate different adaptation mechanisms and a training signal unavailable through ordinary supervised fine-tuning. The method does not consistently exceed in-context learning, especially on some multi-token mathematics tasks. The paper also examines simultaneous training, soft-label distillation, and an alternative parameter-efficient tuning method.

## Glossary

*This glossary defines the paper’s terms, codes, and notation.*

- **Activation:** A hidden self-attention output; a context-dependent footprint of processing before next-token production.
- **Activation steering:** Interventions at internal locations using steering vectors or weights to affect outputs.
- **Functional alignment:** Alignment of layerwise processing rather than only output similarity.
- **Output alignment:** Training to match intended target-response tokens.
- **Self-distillation:** Transferring the same model’s context-conditioned signals into its trained weights.
- **Context distillation:** Compressing contextual effects into weights, commonly through token-response signals.
- **Soft labels:** A teacher’s probability mass for each token.
- **LLM; LM:** The generic language models adapted and evaluated in the paper.
- **Decoder-only transformer:** The studied LM architecture, which stacks self-attention and linear layers.
- **SA:** Self-attention, the token-processing operation in Equation (1).
- **LM head:** The component receiving the final-layer last-token projection to predict the next token.
- **Softmax; \(\sigma\):** The function used in the attention expression in Equation (1).
- **PEFT:** Parameter-efficient fine-tuning; adaptation using small parameter sets.
- **GPU memory:** The memory onto which PEFT parameter sets can be loaded and from which they can be removed.
- **LoRA:** The tuning method used with rank 8 and selected self-attention matrices.
- **Rank; \(r\):** The LoRA rank; \(r\) also normalizes Subspace Overlap.
- **(IA)3:** The alternative PEFT method that uses activation scaling.
- **Full-rank tuning; prompt/prefix tuning:** Other fine-tuning methods discussed as alternatives to LoRA.
- **SVD:** The operation used to obtain weight-matrix basis vectors for subspace comparison.
- **Weight subspace; spanned weight space:** The space compared using weight-matrix basis vectors.
- **Orthogonal weight updates:** The relationship reported between H9C updates and those of the other methods.
- **Generalization:** Performance on new data rather than only the training examples.
- **Shortcuts; target-response overfitting:** Risks attributed to output-oriented, few-shot weight adaptation.
- **Catastrophic forgetting:** A fine-tuning problem cited in related work and proposed for future investigation.
- **Chain-of-thought style:** Ordinary-text reasoning followed by an answer in the demonstrated format.
- **MCQ:** The choice-based setting whose choice labels are remapped to unrelated tokens.
- **QA:** The scientific-answer setting that generates answer tokens rather than selecting options.
- **ID; OOD:** In-distribution and out-of-distribution evaluation.
- **Single-token output:** One token, typically a category, True/False value, or MCQ choice.
- **Multi-token output:** A variable-length, potentially multiple-token response.
- **Calibration; miscalibration:** The response property evaluated using expected calibration error in single-token tasks.
- **Cross-entropy:** The target-token training objective in Equation (2).
- **Mean squared error:** The activation-alignment objective described before Equation (4).
- **Epoch; convergence:** The training unit capped at 50, and the stopping condition used for training.
- **Standard deviation:** The parenthesized variability over five runs at the best-performing learning rate.
- **Random seeds:** Repeatable random choices used across training and reproduction.
- **Q7M; SFT:** Supervised Fine-Tuning, labeled weight adaptation.
- **V2K; ICL:** In-Context Learning, adaptation from contextual demonstrations without weight updates.
- **R8D; IA2:** ICL Activation Alignment, the activation-alignment priming operation.
- **H9C:** SFT only.
- **J3N:** IA2 only, without target-response-token training.
- **P4Z:** IA2 → SFT, sequential activation priming and supervised fine-tuning.
- **T6B:** IA2 + SFT, simultaneous activation and target-token training.
- **M8G; KD:** Knowledge Distillation, transferring teacher behavior through soft-label matching.
- **F5W:** SFT using soft labels.
- **U6R; acc:** Accuracy.
- **D8P; ECE:** Expected calibration error.
- **K4X; asim:** Activation similarity, computed as token-wise cosine similarity.
- **A5Y:** Subspace Overlap, the shared-weight-space measure defined in §5.2.
- **B7V; SST2:** The sentiment dataset used for training and evaluation.
- **Z2H; Financial Phrasebank; FinS:** The financial sentiment dataset.
- **C9L; Poem Sentiment; PoemS:** The poem sentiment evaluation dataset.
- **N5Q; StrategyQA; STF:** The True/False dataset requiring multi-hop reasoning.
- **W3F; AGNews; AGN:** The news-categorization training and evaluation dataset.
- **Y8J; BBCNews; BBCN:** The additional news-categorization evaluation dataset.
- **E2T; SciQr:** The remapped-choice dataset derived from SciQ.
- **G6A; QASCr:** The remapped-choice evaluation dataset derived from QASC.
- **QASC:** The source dataset for G6A.
- **L9S; GSM8K:** The grade-school mathematics training and evaluation dataset.
- **X4C; GSM Symbolic; GSM8Ks:** The additional grade-school mathematics evaluation dataset.
- **V7B; MATH Algebra; HMathA:** The higher-grade algebra dataset.
- **O3D; SciQ:** The scientific QA dataset and source dataset for E2T.
- **Qwen3; Llama-3.2:** The two model families studied.
- **Qwen3-4B-Base:** The Qwen model used on every single- and multi-token task.
- **Llama-3.2-1B; Llama-3.2-3B:** The Llama models used for single-token tasks.
- **Llama-3.2-1B-Instruct:** The secondary multi-token model, used to test post-trained models.
- **GPT-3:** The model in which ICL was first illustrated, as cited in the paper.
- **1B, 3B, 4B:** Model-size labels corresponding here to 16, 28, and 36 layers.
- **STEM-data mid-training:** The proposed explanation for Qwen’s sample-efficient contextual adaptation.
- **RL-tuned models:** Post-trained models proposed for future testing.
- **\(\mathcal T\):** A task.
- **\(\mathcal D_{\mathcal T}\):** The task’s dataset of \(N\) input–output pairs.
- **\(N\); \(n\):** Dataset size and the number of concatenated demonstrations, respectively.
- **\(X_i,X_t,X\); \(x_i\):** An input or query, indexed where applicable.
- **\(Y_i,Y_t\):** A ground-truth or expected response.
- **\(i,j,t\):** Sample or token indices; \(t\) marks the queried example in the response notation.
- **\(\Theta,\Theta'\); \(M_\Theta,M_{\Theta'}\):** Original and adapted parameters, and the models using them.
- **\(\bar Y_i,\bar Y_t\); \(\bar y_t\):** Generated responses under weight adaptation, and the first token of \(\bar Y_t\).
- **\(\hat Y_i,\hat Y_t\); \(\hat y_t\):** Context-conditioned responses, and the predicted contextual response token.
- **\(G\):** The response-token count used for generation or target-token training.
- **\(T=[t_i]_{i=1}^{R}\):** A sequence of \(R\) token vectors.
- **\(R,d,L\):** Token count, vector dimension, and model-layer count.
- **\(t_{:i}\):** The first \(i\) tokens.
- **\(W_Q,W_K,W_V,W_O\); \(W_{QKVO}\):** The self-attention weight matrices, collectively denoted \(W_{QKVO}\).
- **\(q_i,K,V\):** \(t_iW_Q\), \(t_{:i}W_K\), and \(t_{:i}W_V\).
- **\(Z=[z_i]\):** The self-attention output, with shape \(R\times d\).
- **\(I\); \(\circ\):** Concatenated demonstrations; sequence concatenation.
- **\(A\); \(A^i\):** An activation tensor; an unaligned output-position tensor for sample \(i\).
- **\(A_{\mathrm{V2K}},A^i_{\mathrm{V2K}}\):** Context-conditioned reference activations.
- **\(A_{\mathrm{Q7M}}\):** Activations collected after supervised fine-tuning without demonstrations.
- **\(\hat A_i\); \(\hat A\):** The capped training activation tensor; the target tensor discussed for joint training.
- **\(\tilde W_{QKVO}\):** The adapted attention matrices sought by activation alignment.
- **\(-1\):** The index of each newly generated token in the alignment condition.
- **\(\mathcal L_{\mathrm{Q7M}}\); \(\mathcal L_{\mathrm{R8D}}\):** The target-token loss and activation-alignment loss.
- **\(\beta\):** The weight on target-token loss in simultaneous training.
- **\(U_1,U_2\):** Basis vectors from the two methods’ compared weight matrices.
- **\(\|\cdot\|;\ \|\cdot\|_F^2\):** The norm notation in Equation (4); the squared norm in Subspace Overlap.
- **\({}^T\):** Transposition in the attention and overlap formulas.
- **\(\mathbb R\); \(\in\); \(\forall\):** Real-valued tensor notation; membership; “for every.”
- **\(\sum\); \(\times\); \(\cdot\); \(*\):** Summation; tensor dimensions or multiplication; multiplication.
- **\(\approx\); \(\rightarrow\); \(+\):** Approximate alignment; adaptation or sequential training; combined losses or simultaneous training.
- **\(\sqrt d\):** The attention scaling denominator.
- **\([A,B,C,\ldots]\):** The original MCQ choice labels before remapping.
- **Slow, medium, fast:** Learning rates 1e-4, 3e-4, and 1e-3.
- **↑; ↓:** Higher-is-better and lower-is-better table indicators.
- **∗:** OOD evaluation.
- **Gap wrt V2K; Gap wrt H9C:** The reported P4Z gaps relative to V2K and H9C.
- **`<think>`:** Tags not used around the ordinary-text reasoning in these experiments.
- **`####`:** The separator before numerical or text answers in specified ground-truth formats.
- **`\boxed{}`:** The required and parsed V7B answer enclosure.
- **`<reasoning steps>`, `<numerical answer>`, `<value>`, `<support text>`, `<text answer>`:** Placeholders in the response-format descriptions.
- **GitHub code; README:** The reproduction code and its detailed accompanying instructions.
- **ONR; N00014-24-1-2089:** A supporting organization and its listed award.
- **♡:** Equal advising.

# IA2: Alignment with ICL Activations

Improves Supervised Fine-Tuning

Aayush Mishra

Daniel Khashabi & Anqi Liu  
Department of Computer Science, Johns Hopkins University

♡ Equal advising. Correspondence: {amishr24,danielk,aliu.cs}@jhu.edu

###### Abstract

*Activation-alignment priming improves supervised fine-tuning and exposes differences between adaptation mechanisms.*

Across 12 popular benchmarks and two model families, P4Z—IA2 followed by supervised fine-tuning—significantly improves U6R, or accuracy, and calibration, the response property measured here by expected calibration error. This provides practically useful adaptation and a conceptual window into its internal mechanics.

R8D, or ICL Activation Alignment (IA2), is self-distillation: it trains a model to replicate its own context-conditioned activation patterns. Those patterns come from V2K, or In-Context Learning (ICL), which adapts through demonstrations without weight updates. R8D incentivizes V2K-like internal reasoning in weight-trained models. Its priming signal is unavailable through H9C, or supervised fine-tuning alone.

**Claims and evidence**
- P4Z improves accuracy and calibration across the reported 12 benchmarks and two model families. Examples: Table 1, B7V→B7V, P4Z/H9C U6R 90.4/65.2 and D8P 0.06/0.22; Table 6, B7V→B7V, 62.6/52.4 and 0.28/0.35.
- R8D provides a distinct priming signal: Figure 4 and the §5.2 weight-space table, “Most performant P4Z and J3N,” report around \(39\%\) shared spanned weight space.

## 1 Introduction

*The paper asks whether contextual activation signals can improve labeled weight adaptation.*

Generic LLMs, the language models adapted here, serve specialized applications. One example is classifying news-aggregator articles into predefined categories. Parameter-efficient fine-tuning (PEFT) techniques (Hu et al., 2021; Liu et al., 2022) often represent adapted models with small parameter sets. These sets can be efficiently loaded onto and removed from GPU memory.

Generalization through Q7M—Supervised Fine-Tuning (SFT), or labeled weight adaptation—typically requires many labeled samples (Le Scao and Rush, 2021). Collecting them can be expensive. V2K demonstrations and instructions instead consume context capacity and increase per-query inference compute. They displace query and response tokens.

The research question is whether information-rich V2K activations can improve Q7M quality. Activations are hidden processing outputs before next-token production. The hypothesis is that they contain information about extracting generalizable contextual patterns. That information may be absent from few-shot weight adaptation, which is prone to shortcuts or target-response overfitting. Similar token-space behavior need not imply similar internal processing. Divergent activations indicate distinct adaptation mechanisms, consistent with Shen et al. (2024).

Context-to-weight distillation using response-text signals (Snell et al., 2022; Chen et al., 2024b) can inherit the same output-training problems. Reproducing a context-conditioned model’s outputs does not ensure reproducing its functioning.

**Figure 1:** Standard Q7M enforces output-space agreement between response and target—the third step of the proposed pipeline—and produces subpar performance and miscalibration under low data. The pipeline collects rich V2K activations, enforces functional alignment, then trains against target responses; its comparison charts aggregate performance across models and datasets; details: §5.

**Claims and evidence**
- Standard Q7M has low-data performance and calibration problems: Figure 1; Table 1, B7V→B7V, H9C/P4Z U6R 65.2/90.4 and D8P 0.22/0.06.

## 2 Background and Notation

*This section defines weight adaptation, contextual adaptation, self-attention, and activation comparison.*

For task \(\mathcal T\), let \(\mathcal D_{\mathcal T}=\{(X_i,Y_i)\}_{i=1}^{N}\) contain \(N\) input–output pairs. Here, \(X_i\) is an input and \(Y_i\) its ground-truth response. Q7M changes model parameters \(\Theta\rightarrow\Theta'\). The adapted model produces
\(M_{\Theta'}(X_t)=\bar y_t\), where \(\bar y_t\) is the first token of response \(\bar Y_t\). Repeated application to the growing sequence extends the response.

For ground truth \(Y_i\) containing \(G\) tokens, generate \(G\) tokens \(\bar Y_i\). Minimize cross-entropy, the target-token loss, across all generated tokens. The indices \(i\) and \(j\) identify samples and response tokens:

\[
\mathcal L_{\mathrm{Q7M}}
=\sum_{i=1}^{N}\sum_{j=1}^{G}
\operatorname{cross\text{-}entropy}(\bar Y_{ij},Y_{ij}).
\tag{2}
\]

A decoder-only transformer LM, or language model, \(M_\Theta\) stacks self-attention (SA) and linear layers. It processes \(R\) token vectors \(T=[t_i]_{i=1}^{R}\) using SA and linear projections at every token and layer. The final-layer last-token projection passes through the LM head, which predicts the most likely next token. SA depends on other sequence tokens.

Let \(W_Q,W_K,W_V,W_O\in\Theta\) be the attention weight matrices, collectively \(W_{QKVO}\). Then

\[
\operatorname{SA}(T;W_{QKVO})
=Z=
\left[
z_i=\sigma\left(\frac{q_iK^T}{\sqrt d}\right)V\cdot W_O
\right]_{i=1}^{R},
\tag{1}
\]

where \(\sigma\) is Softmax, \(q_i=t_iW_Q\), \(K=t_{:i}W_K\), and \(V=t_{:i}W_V\). The notation \(t_{:i}\) denotes the first \(i\) tokens; superscript \(T\) denotes transposition. Both the output \(Z=[z_i]\) and input \(T\) have shape \(R\times d\), where \(d\) is the vector dimension. The studied behavioral changes arise from varying context \(T\) or weights \(W_{QKVO}\).

Let
\(I=[X_1\circ Y_1\circ\ldots\circ X_n\circ Y_n]\)
contain \(n\) concatenated demonstrations; \(\circ\) denotes concatenation. V2K processes these examples to understand the task and produces
\(M_\Theta(I\circ X_t)=\hat y_t\)
without weight updates. The predicted token \(\hat y_t\) begins a contextual response \(\hat Y_t\), often similar to expected \(Y_t\).

First illustrated in GPT-3 (Brown et al., 2020), this widely used adaptation of generic LMs also encompasses generic instructions as demonstrations for instruction-following frontier models. The present setting restricts demonstrations to classic input–output pairs.

Hidden SA outputs are activations: context-dependent footprints of processing before next-token production. They are useful for examining behavioral changes. A model with \(L\) layers processing \(R\) tokens supplies a real-valued activation tensor \(A\in\mathbb R^{L\times R\times d}\).

K4X, or activation similarity (asim), is token-wise cosine similarity between two activation tensors. It has shape \(L\times R\). The studied sequences are \(T=[I\circ X]\), with demonstrations, and \(T=X\), without demonstrations under differing weights.

## 3 Are V2K and Q7M practically the same?

*Behavioral and activation comparisons indicate that V2K and Q7M adapt differently.*

V2K responses are much better calibrated than Q7M responses at similar, slightly better U6R. This agrees with the few-shot generalization and calibration findings of Duan et al. (2024). The proposed explanation is that output-oriented training learns shortcuts that fail on new data. Complex V2K circuits (Elhage et al., 2021), required across varied tasks, instead extract generalizable demonstration patterns. The methods achieve adaptation differently. Q7M is not a drop-in V2K replacement.

Recent internal-gradient-descent accounts (Von Oswald et al., 2023; Akyürek et al., 2022) show SA producing outputs as though contextual demonstrations had updated parameters. Such functional equivalence predicts similar output-position activations for a base model with demonstrations and a trained model without them.

The observed activations do not align across models. Initial and final layers align better than middle layers. Individual-token processing is expected in the initial and final layers; abstract processing of the entire demonstration set is expected in the middle layers.

**Figure 2: Layerwise similarity with V2K activations.** H9C has small K4X, indicating differing adaptation functionality; P4Z aligns much more closely and performs much better. The first-row H9C curves average output-token K4X across tokens, samples, and datasets.

For this investigation, collect contextual activations \(A_{\mathrm{V2K}}\) using \(T=[I\circ X]\) on 100 test samples from multiple datasets. These include B7V, or SST2, and W3F, or AGNews. Then train using those same demonstrations until convergence, the training stopping condition, and collect trained-model activations \(A_{\mathrm{Q7M}}\) using \(T=X\).

Models span Qwen3 (Yang et al., 2025) and Llama-3.2 (Grattafiori et al., 2024). Their size–depth correspondences are \(1\mathrm B\leftrightarrow16\) layers, \(3\mathrm B\leftrightarrow28\) layers, and \(4\mathrm B\leftrightarrow36\) layers. The output-position comparison is
\(\mathrm{K4X}(A_{\mathrm{V2K}},A_{\mathrm{Q7M}})\).
Experiment details: §B.

**Claims and evidence**
- V2K has better calibration at similar, slightly better accuracy: Table 1, W3F→W3F, V2K/H9C U6R 30.0/27.9 and D8P 0.13/0.52; Z2H→C9L∗, 56.9/49.9 and 0.12/0.48.
- The activation patterns differ, with better initial/final than middle-layer alignment: Figure 2, H9C curves, averaged across output tokens, samples, and datasets.
- The comparison uses 100 test samples and the stated 1B/16-layer, 3B/28-layer, and 4B/36-layer correspondences: this section’s experimental setup; further details in §B.

## 4 Inducing V2K-like behavior in Q7M Models

*The method first aligns contextual activations and then trains on ground-truth responses.*

After priming changes \(\Theta\) to \(\Theta'\), collect \(\bar Y_i\) for every \(X_i\) using \(M_{\Theta'}\). Train against ground truth \(Y_i\) with \(\mathcal L_{\mathrm{Q7M}}\) until convergence. This aligns intended target-token behavior.

The preceding R8D priming operation instead aligns layerwise processing, not output similarity. It seeks adapted attention matrices
\(\tilde W_{QKVO}=(\tilde W_Q,\tilde W_K,\tilde W_V,\tilde W_O)\)
satisfying the following condition for every newly generated token at index \(-1\):

\[
\operatorname{SA}([I\circ X];W_{QKVO})
\approx
\operatorname{SA}(X;\tilde W_{QKVO})
\qquad \forall X\in\mathcal T.
\tag{3}
\]

Here, \(\approx\) expresses the sought approximate alignment, and \(\forall\) means “for every.”

Chen et al. (2024a) give a closed-form solution for linearized attention using a Softmax kernel approximation. The target here is a practical general solution for real nonlinear transformers. It places V2K qualities directly into adapted weights, processing every query as the base model does at each layer.

For each training input \(X_i\), construct \(I\) from the remaining dataset samples. Generate \(\hat Y_i\) from \(T_i=[I\circ X_i]\). Collect output-position activations \(A^i_{\mathrm{V2K}}\in\mathbb R^{L\times G\times d}\) for \(G\) response tokens.

Next supply \(T_i=[X_i\circ\hat Y_i]\). Treat the response as generated with only the query in context. This produces unaligned output-position activations \(A^i\). Minimize the mean squared error with respect to model parameters until convergence:

\[
\mathcal L_{\mathrm{R8D}}
=\sum_{i=1}^{N}\left\|A^i-A^i_{\mathrm{V2K}}\right\|.
\tag{4}
\]

The norm notation \(\|\cdot\|\) expresses the activation difference in this objective.

## 5 Experimental Results

*The experiments compare adaptation methods across single- and multi-token tasks, including distribution shifts.*

Each \((\text{method},\text{dataset})\) combination trains to convergence, up to 50 epochs, the training units used here. Three learning rates are tested: slow:1e-4, medium:3e-4, and fast:1e-3. The best performer is selected to neutralize hyperparameter-selection effects. Over 13,000 models were trained.

Resource-based choices fix rank-8 LoRAs, the tuning parameter sets used here, targeting SA matrices \(W_Q,W_K,W_O\). Multi-token training is substantially more resource-extensive with long answers. It restricts \(N\) to 16 for Llama and 8 for Qwen.

For activation collection, training caps generation at \(G=200\), yielding training activation tensors
\(\hat A_i\in\mathbb R^{L\times200\times d}\ \forall x_i\).
Target-token training uses the ground-truth response length for \(G\). Further training details: §B.

Single-token evaluation uses the first generated token’s probability distribution for both U6R, or accuracy (acc), and D8P, or expected calibration error (ECE; Guo et al., 2017). Base-model demonstrations are evaluated five times per task validation set with different demonstrations, and average metrics are reported. Trained methods are evaluated without demonstrations, and best-performing averages are reported.

Multi-token evaluation counts only correctly answered questions in the required format; parsing and other details are in §A. Open-ended answer confidence is difficult to measure, so calibration is not measured. Generation limits judiciously conserve compute:
- 200 for L9S, or GSM8K, and X4C, or GSM Symbolic [GSM8Ks]. Their 95-percentile answer lengths are 184 and 154.
- 400 for V7B, or MATH Algebra [HMathA]. Its corresponding length is 307.

Qwen3-4B-Base is used on every single- and multi-token task. Llama-3.2-1B additionally covers every single-token task except E2T, or SciQr, the remapped-choice dataset derived from SciQ. E2T uses Llama-3.2-3B: the 1B model could not improve above random baseline for any \(N\) up to 128. Multi-token experiments instead use Llama-3.2-1B-Instruct as the secondary model, testing efficacy on post-trained models.

Single-token outputs are one token, typically a category/class, True/False, or MCQ choice. The MCQ setting uses choice labels remapped to unrelated tokens. Multi-token outputs have variable, potentially multiple-token length: the general open-generation case. Both settings compare adaptation across multiple models and tasks, frequently including distribution-shift evaluation.

- **Sentiment:** B7V = SST2 (Socher et al., 2013), alongside Z2H = Financial Phrasebank [FinS] (Malo et al., 2014) and C9L = Poem Sentiment [PoemS] (Sheng and Uthus, 2020). Training is separate on B7V and Z2H. Each trained model is evaluated on all three datasets, producing one in-distribution (ID) and two out-of-distribution (OOD) datasets.
- **True/False:** N5Q = StrategyQA [STF] (Geva et al., 2021), requiring multi-hop reasoning before its binary conclusion.
- **News categorization:** Training uses W3F = AGNews [AGN] (Zhang et al., 2015). Evaluation uses W3F and Y8J = BBCNews [BBCN] (Greene and Cunningham, 2006).
- **MCQ:** Choices \([A,B,C,\ldots]\) are remapped into unrelated tokens for added complexity. E2T = SciQr, derived from O3D, is used for training and evaluation. G6A = QASCr, derived from QASC (Khot et al., 2019), is used for evaluation.
- **Grade-school mathematics:** Training uses L9S = GSM8K (Cobbe et al., 2021). Evaluation uses L9S and X4C = GSM Symbolic [GSM8Ks] (Mirzadeh et al., 2024).
- **Advanced mathematics:** V7B = MATH Algebra [HMathA] (Hendrycks et al., 2021), containing higher-grade algebra problems.
- **Scientific QA:** O3D = SciQ (Johannes Welbl, 2017), generating answer tokens rather than selecting options. This is also the source dataset for E2T.

Few-shot training datasets
\(\mathcal D_{\mathcal T}=\{(X_i,Y_i)\}_{i=1}^{N}\)
vary \(N\) by powers of two, \([2,4,8,16,\ldots]\). Five distinct sets per \(N\) average outliers. Each target tensor \(A^i_{\mathrm{V2K}}\ \forall X_i\) uses the other \(N-1\) samples as randomly ordered demonstrations. Exactly the same training data are reused across adaptation methods for fair comparison. Evaluation samples a separate set of 500 examples per task. Dataset and processing details: §A.

Multi-token demonstrations and expected outputs are chain-of-thought style (Wei et al., 2022): ordinary-text reasoning precedes an answer in the demonstrated format. The reasoning is not inside `<think>` tags.

Parsing is dataset-specific:
- L9S and X4C ground truths follow “\(<\)reasoning steps\(>\) #### \(<\)numerical answer\(>\)”. Reasoning correctness and conciseness are unevaluated. The parsed number must exactly match ground truth. Common alternatives, including “The answer is \(<\)value\(>\)”, permit slight formatting variation. The exact parser is in the code.
- V7B answers are expected and parsed inside `\boxed{}`.
- O3D ground truths are constructed as “\(<\)support text\(>\) #### \(<\)text answer\(>\)”. Parsing uses that format, and only exact string matches count as correct.

Here, `####` separates reasoning or support text from the answer; the bracketed phrases are format placeholders.

**Claims and evidence**
- Llama-3.2-1B did not improve above random baseline on E2T for any \(N\) up to 128: the model-selection result stated in this section; E2T therefore uses Llama-3.2-3B.

### 5.1 R8D priming boosts Q7M performance

*Priming generally improves single-token accuracy and calibration and improves multi-token accuracy over H9C.*

P4Z outperforms H9C in almost all single-token cases for U6R and calibration with identical data. In most cases, it also exceeds V2K U6R, with slightly worse calibration.

J3N, or IA2 only without target-response-token training, often reaches high U6R and reasonable, V2K-like calibration. This evidences a rich training signal. Other model/\(N\) reports and a statistical-significance test of the improvement are provided in §C.

| Source | Eval | V2K U6R ↑ | V2K D8P ↓ | H9C U6R ↑ | H9C D8P ↓ | J3N U6R ↑ | J3N D8P ↓ | P4Z U6R ↑ | P4Z D8P ↓ |
|---|---|---|---|---|---|---|---|---|---|
| W3F | W3F | 30.0 (02.3) | 0.13 (0.03) | 27.9 (04.3) | 0.52 (0.12) | 24.0 (00.3) | 0.11 (0.01) | 31.8 (03.8) | 0.35 (0.24) |
| | Y8J∗ | 28.8 (00.6) | 0.12 (0.01) | 31.8 (07.1) | 0.56 (0.14) | 24.3 (01.3) | 0.10 (0.02) | 27.9 (04.3) | 0.40 (0.21) |
| Z2H | Z2H | 63.6 (01.5) | 0.12 (0.01) | 67.4 (14.1) | 0.31 (0.14) | 63.1 (15.0) | 0.24 (0.06) | 78.7 (13.7) | 0.16 (0.11) |
| | C9L∗ | 56.9 (02.2) | 0.12 (0.02) | 49.9 (04.6) | 0.48 (0.04) | 52.8 (05.6) | 0.15 (0.04) | 60.6 (15.7) | 0.36 (0.15) |
| | B7V∗ | 70.1 (01.3) | 0.17 (0.01) | 59.4 (06.5) | 0.20 (0.10) | 69.8 (13.7) | 0.30 (0.10) | 71.4 (18.9) | 0.20 (0.15) |
| E2T | G6A∗ | 56.5 (01.4) | 0.08 (0.01) | 76.5 (03.8) | 0.09 (0.05) | 71.2 (03.0) | 0.12 (0.01) | 79.4 (01.8) | 0.09 (0.01) |
| | E2T | 87.7 (01.1) | 0.07 (0.01) | 88.3 (04.6) | 0.07 (0.05) | 90.4 (01.5) | 0.09 (0.01) | 91.7 (01.7) | 0.05 (0.01) |
| B7V | Z2H∗ | 41.9 (00.3) | 0.19 (0.01) | 68.4 (02.3) | 0.30 (0.04) | 71.3 (02.1) | 0.21 (0.03) | 82.4 (12.2) | 0.11 (0.08) |
| | C9L∗ | 65.1 (01.2) | 0.11 (0.02) | 56.5 (13.2) | 0.33 (0.14) | 62.4 (09.2) | 0.19 (0.06) | 68.4 (08.4) | 0.30 (0.08) |
| | B7V | 85.4 (00.4) | 0.13 (0.00) | 65.2 (10.7) | 0.22 (0.10) | 82.7 (06.4) | 0.28 (0.03) | 90.4 (02.3) | 0.06 (0.01) |
| N5Q | N5Q | 66.0 (02.1) | 0.07 (0.01) | 52.4 (04.4) | 0.29 (0.10) | 62.2 (04.1) | 0.16 (0.04) | 62.4 (04.3) | 0.29 (0.08) |

**Table 1:** Qwen3-4B-Base, \(N=4\), single-token dataset/adaptation-method performance. Parentheses give standard deviations over five runs at the best-performing learning rate; bold denotes best training method, ∗ denotes OOD evaluation. P4Z outperforms standard Q7M across the board.

In the tables, ↑ means higher is better and ↓ means lower is better.

Multi-token P4Z exceeds H9C in every case but sometimes underperforms V2K, unlike the single-token results. The proposed explanation is imperfect compression of longer contexts into the same small LoRA weight space. This is expected to improve with larger ranks. Additional model/\(N\) reports are provided in §C.

| Source | Eval | Without V2K U6R ↑ | V2K U6R ↑ | H9C U6R ↑ | J3N U6R ↑ | P4Z U6R ↑ |
|---|---|---|---|---|---|---|
| L9S | L9S | 56.4 (00.0) | 76.4 (01.1) | 70.9 (02.8) | 77.4 (02.8) | 73.6 (03.7) |
| | X4C∗ | 45.4 (00.0) | 68.4 (01.1) | 64.5 (04.0) | 66.2 (03.7) | 68.8 (05.1) |
| V7B | V7B | 21.0 (00.0) | 60.4 (02.1) | 50.4 (01.8) | 47.8 (06.0) | 55.3 (02.1) |
| O3D | O3D | 15.4 (00.0) | 37.5 (01.0) | 35.0 (07.1) | 06.9 (13.8) | 40.8 (07.6) |

**Table 2:** Qwen3-4B-Base, \(N=4\), multi-token dataset/adaptation-method performance; bold denotes best training method.

**Claims and evidence**
- P4Z generally improves single-token U6R and calibration over H9C. Table 1, P4Z/H9C U6R and D8P by Source→Eval row: W3F→W3F, 31.8/27.9 and 0.35/0.52; W3F→Y8J∗, 27.9/31.8 and 0.40/0.56; Z2H→Z2H, 78.7/67.4 and 0.16/0.31; Z2H→C9L∗, 60.6/49.9 and 0.36/0.48; Z2H→B7V∗, 71.4/59.4 and 0.20/0.20; E2T→G6A∗, 79.4/76.5 and 0.09/0.09; E2T→E2T, 91.7/88.3 and 0.05/0.07; B7V→Z2H∗, 82.4/68.4 and 0.11/0.30; B7V→C9L∗, 68.4/56.5 and 0.30/0.33; B7V→B7V, 90.4/65.2 and 0.06/0.22; N5Q→N5Q, 62.4/52.4 and 0.29/0.29.
- P4Z often exceeds V2K U6R with slightly worse calibration: Table 1, Z2H→Z2H, P4Z/V2K 78.7/63.6 and D8P 0.16/0.12; B7V→C9L∗, 68.4/65.1 and 0.30/0.11.
- J3N can perform well without target-token training: Table 1, E2T→E2T, J3N U6R 90.4 and D8P 0.09; V2K 87.7 and 0.07.
- P4Z exceeds H9C in every reported multi-token case: Table 2, L9S→L9S 73.6/70.9; L9S→X4C∗ 68.8/64.5; V7B→V7B 55.3/50.4; O3D→O3D 40.8/35.0.
- P4Z sometimes underperforms V2K: Table 2, L9S→L9S 73.6/76.4 and V7B→V7B 55.3/60.4. Longer-context compression is the proposed explanation.

### 5.2 Why is R8D priming important?

*Activation and weight-space analyses identify a distinct signal from priming that subsequent target training retains.*

H9C weight updates are almost completely orthogonal to those of the other methods. The most performant P4Z models substantially share J3N’s spanned weight space. Much of their gain is attributable to R8D, whose identified subspaces are practically unreachable through H9C.

The wide J3N–P4Z distribution arises when priming already produces output alignment. Subsequent target training then changes little before convergence, leaving very high overlap.

**Figure 4:** P4Z has significantly high A5Y, or Subspace Overlap, with J3N, while H9C aligns with neither, indicating an R8D signal absent from H9C and important for performance.

| Model pair | Mean shared spanned weight space |
|---|---|
| Most performant P4Z and J3N | around \(39\%\) |

The three training conditions are baseline H9C = SFT only, priming without target training J3N = IA2 only, and sequential adaptation P4Z = IA2 → SFT. All appear in Figure 2’s K4X comparison with V2K. Priming drastically increases K4X, and sequential training retains elevated K4X. H9C and P4Z differ only in their starting state.

P4Z has intermediate K4X between the other conditions while achieving better U6R and calibration. Increasing similarity improves calibration, but extreme similarity need not optimize U6R. The V2K signal is not always correct. One case in §C has J3N following V2K even as overfitting worsens performance. Functional alignment and ground-truth output alignment are best combined.

**Figure 3:** Scatterplots of \(\mathrm{K4X}(A,A_{\mathrm{V2K}})\) against U6R/D8P place one training experiment at each point. D8P decreases smoothly with increasing similarity; extreme alignment can leave U6R gains obtainable from ground-truth training.

Under LoRA, unchanged base weights significantly influence activations. Activation patterns alone therefore cannot isolate training-induced behavioral changes.

Direct weight analysis instead uses all single- and multi-token experiments. It takes SVD basis vectors—the weight-matrix basis vectors obtained for comparison. A5Y measures shared weight space, isolating and comparing functional-behavior differences through

\[
\frac{1}{r}*\left\|U_1^T\cdot U_2\right\|_F^2,
\]

where \(U_1,U_2\) are the compared methods’ basis vectors and \(r\) is LoRA rank. The notation \(\|\cdot\|_F^2\) is the squared norm used in this overlap measure.

**Claims and evidence**
- H9C updates are nearly orthogonal to the other methods, while performant P4Z models share J3N’s space: Figure 4; the unnumbered table, “Most performant P4Z and J3N,” reports around \(39\%\).
- Priming raises activation similarity, and sequential target training retains elevated similarity: Figure 2, H9C, J3N, and P4Z comparisons.
- Calibration improves with similarity, but extreme similarity need not maximize accuracy: Figure 3, K4X–D8P and K4X–U6R scatterplots; the overfitting case is referred to §C.
- P4Z combines the signals with better performance: Table 1, B7V→B7V, P4Z/J3N/H9C U6R 90.4/82.7/65.2 and D8P 0.06/0.28/0.22.

## 6 Discussion

*The discussion tests simultaneous training and soft-label distillation, examines limits, and considers other tuning methods and improvements.*

T6B, or IA2 + SFT, applies activation and target-token training simultaneously. It extracts significantly more performance from V2K responses than H9C. H9C and P4Z trained on those responses perform worse than when trained on ground truths.

| Source | Eval | Without V2K U6R ↑ | V2K U6R ↑ | H9C U6R ↑ | J3N U6R ↑ | P4Z U6R ↑ | T6B U6R ↑ |
|---|---|---|---|---|---|---|---|---|
| L9S | L9S | 56.4 (00.0) | 76.4 (01.1) | 66.4 (18.1) | 77.4 (02.8) | 67.2 (22.1) | 77.0 (07.8) |
| | X4C∗ | 45.4 (00.0) | 68.4 (01.1) | 59.8 (20.2) | 66.2 (03.7) | 61.6 (24.0) | 69.0 (09.6) |
| V7B | V7B | 21.0 (00.0) | 60.4 (02.1) | 49.0 (03.0) | 47.8 (06.0) | 54.0 (01.7) | 53.6 (02.7) |
| O3D | O3D | 15.4 (00.0) | 37.5 (01.0) | 27.2 (12.0) | 06.9 (13.8) | 32.8 (10.0) | 34.5 (10.6) |

**Table 3:** Qwen3-4B-Base, \(N=4\), multi-token dataset/adaptation-method performance when training on V2K responses rather than ground truths; T6B’s significant improvement over H9C highlights R8D’s significance.

T6B applies \(\mathcal L_{\mathrm{R8D}}\) and \(\mathcal L_{\mathrm{Q7M}}\) together rather than sequentially. However, target tensor \(\hat A\) uses the model’s own generated response tokens whenever \(G>1\). That V2K response can differ from ground truth even in length. The ground-truth objectives are incompatible for joint training.

Self-distillation on V2K responses, analogous to Snell et al. (2022) with an additional activation signal, uses

\[
\mathcal L_{\mathrm{R8D}}+\beta\cdot\mathcal L_{\mathrm{Q7M}},
\]

where \(\beta\) weights the target-token loss. It varies over four values smoothly spanning J3N to H9C; details are in §B. Multi-token reports select the best \((\text{learning rate},\beta)\) combination across five random seeds. Further T6B results are provided in §C.

**Computational overhead:** P4Z adds one pretraining activation-collection pass: V2K inference on the given dataset. This one-time cost greatly benefits subsequent capabilities. Inference itself costs the same as H9C.

M8G, or Knowledge Distillation (KD), transfers a stronger teacher’s behavior into a student through soft-label matching. Soft labels provide each token’s probability mass (Hinton et al., 2015). Here, the teacher is the same model enhanced with contextual demonstrations. F5W is the comparator that uses SFT with these soft labels.

| Dataset | \(N\) | V2K U6R ↑ | V2K D8P ↓ | H9C U6R ↑ | H9C D8P ↓ | F5W U6R ↑ | F5W D8P ↓ | J3N U6R ↑ | J3N D8P ↓ | P4Z U6R ↑ | P4Z D8P ↓ | T6B U6R ↑ | T6B D8P ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L9S | 4 | 34.6 (00.9) | – | 28.2 (04.5) | – | 36.1 (03.5) | – | 29.3 (05.0) | – | 30.3 (03.2) | – | 37.0 (01.9) | – |
| B7V | 8 | 77.0 (02.2) | 0.10 (0.02) | 52.3 (04.3) | 0.38 (0.14) | 55.9 (08.8) | 0.16 (0.06) | 64.9 (15.3) | 0.30 (0.05) | 64.7 (19.3) | 0.33 (0.19) | 66.9 (18.4) | 0.28 (0.21) |

**Table 4:** Comparison with M8G, using target-token training on soft labels; the baseline improves on standard Q7M in the multi-token case, while R8D-based training provides consistently high single-/multi-token performance.

F5W trains and evaluates Llama-3.2-1B-Instruct on L9S and Llama-3.2-1B on B7V using V2K responses. It significantly improves over standard Q7M in the multi-token case and nearly reaches T6B, but is lacking in the single-token case.

**Failure to consistently exceed V2K:** On some multi-token Qwen mathematics cases, including L9S and V7B, V2K exceeds every training method. The proposed explanation is Qwen’s STEM-data mid-training, making contextual adaptation extremely sample efficient. Surprisingly, L9S V2K at \(N=2\) exceeds its \(N=4,8\) performance. P4Z’s gap to V2K closes quickly with more shots, while superiority to H9C remains consistent.

| Dataset | \(N\) | V2K | H9C | J3N | P4Z | Gap wrt V2K | Gap wrt H9C |
|---|---|---|---|---|---|---|---|
| L9S | 2 | 81.2 (00.4) | 65.2 (01.0) | 70.7 (01.8) | 74.6 (02.4) | -06.6 | +09.4 |
| | 4 | 76.4 (01.1) | 70.9 (02.8) | 77.4 (02.8) | 73.6 (03.7) | -02.8 | +02.7 |
| | 8 | 77.2 (00.9) | 73.5 (01.6) | 78.8 (02.5) | 76.0 (02.0) | -01.2 | +02.5 |
| V7B | 2 | 58.6 (02.0) | 47.8 (04.0) | 33.9 (06.4) | 48.7 (07.0) | -09.9 | +00.9 |
| | 4 | 60.4 (02.1) | 50.4 (01.8) | 47.8 (06.0) | 55.3 (02.1) | -05.1 | +04.9 |
| | 8 | 62.1 (01.0) | 52.0 (00.8) | 57.3 (03.7) | 59.4 (00.8) | -02.7 | +07.4 |

**Table 5:** Shot-count evolution for Qwen mathematics datasets, where V2K is more sample efficient than training methods.

“Gap wrt V2K” and “Gap wrt H9C” report P4Z’s gaps relative to those methods.

**Other fine-tuning methods:** Full-rank tuning, prompt/prefix tuning (Lester et al., 2021; Li and Liang, 2021), and activation scaling via (IA)3 (Liu et al., 2022) are alternatives to LoRA. The pipeline improves training signals rather than competing with these methods.

Experiments on a few datasets replace LoRA with the popular PEFT method (IA)3. They retain the exact §5 methodology; setup is in §B. Trends resemble LoRA, and additional reports are provided in §C. Exploration with other Q7M methods is deferred. R8D also marks a major method-naming improvement over (IA)3.

| Source | Eval | V2K U6R ↑ | V2K D8P ↓ | H9C U6R ↑ | H9C D8P ↓ | J3N U6R ↑ | J3N D8P ↓ | P4Z U6R ↑ | P4Z D8P ↓ |
|---|---|---|---|---|---|---|---|---|---|
| E2T | G6A∗ | 35.6 (01.4) | 0.06 (0.01) | 32.8 (03.4) | 0.20 (0.08) | 34.2 (02.7) | 0.08 (0.01) | 48.6 (05.5) | 0.25 (0.11) |
| | E2T | 60.0 (01.8) | 0.07 (0.01) | 64.5 (05.0) | 0.12 (0.05) | 63.0 (04.6) | 0.10 (0.04) | 70.6 (04.6) | 0.08 (0.02) |
| B7V | Z2H∗ | 56.6 (01.6) | 0.17 (0.01) | 58.2 (10.7) | 0.25 (0.03) | 61.9 (09.5) | 0.18 (0.06) | 59.8 (07.8) | 0.18 (0.03) |
| | C9L∗ | 52.3 (03.0) | 0.19 (0.02) | 48.1 (03.2) | 0.44 (0.11) | 48.1 (03.4) | 0.11 (0.04) | 59.8 (13.4) | 0.33 (0.13) |
| | B7V | 60.3 (02.8) | 0.19 (0.02) | 52.4 (02.8) | 0.35 (0.05) | 54.2 (04.2) | 0.15 (0.06) | 62.6 (09.1) | 0.28 (0.11) |

**Table 6:** Llama-3.2 models, \(N=4\), trained with (IA)3. P4Z significantly exceeds H9C, exceeding even V2K U6R in all cases.

**Potential improvements:** Resource constraints fixed design choices. Deferred explorations include LoRA rank and target-module ablations, selective R8D, prompt optimization, and finer-grained control of the two training signals rather than fixed learning rates.

Extreme alignment can hurt performance. Layer sensitivity to contextual demonstrations also varies (Todd et al., 2023), so selective layer alignment may be more effective. Demonstration order affects V2K performance (Lu et al., 2022; Zhao et al., 2021), permitting prompt optimization before target-activation collection.

**Claims and evidence**
- T6B extracts more performance from V2K responses than H9C: Table 3, T6B/H9C U6R, L9S→L9S 77.0/66.4; L9S→X4C∗ 69.0/59.8; V7B→V7B 53.6/49.0; O3D→O3D 34.5/27.2.
- H9C and P4Z do worse on V2K responses than ground truths: Tables 3/2, respectively, H9C and P4Z U6R—L9S→L9S 66.4/70.9 and 67.2/73.6; L9S→X4C∗ 59.8/64.5 and 61.6/68.8; V7B→V7B 49.0/50.4 and 54.0/55.3; O3D→O3D 27.2/35.0 and 32.8/40.8.
- P4Z adds one activation-collection pass and has H9C’s inference cost: the “Computational overhead” account in this section.
- Soft-label training improves multi-token performance and nearly reaches T6B: Table 4, L9S, \(N=4\), F5W/H9C/T6B U6R 36.1/28.2/37.0. Its single-token performance is lacking relative to R8D methods: B7V, \(N=8\), F5W/J3N/P4Z/T6B 55.9/64.9/64.7/66.9.
- V2K can exceed every training method: Table 5, V7B, \(N=2\), V2K/H9C/J3N/P4Z 58.6/47.8/33.9/48.7; \(N=4\), 60.4/50.4/47.8/55.3; \(N=8\), 62.1/52.0/57.3/59.4. STEM-data mid-training is the proposed explanation.
- L9S V2K is better at \(N=2\) than \(N=4,8\): Table 5, L9S rows, 81.2, 76.4, and 77.2.
- P4Z’s V2K gap closes with more shots, and its H9C advantage persists: Table 5, L9S gaps at \(N=2,4,8\), -06.6/-02.8/-01.2 and +09.4/+02.7/+02.5; V7B, -09.9/-05.1/-02.7 and +00.9/+04.9/+07.4.
- With (IA)3, P4Z exceeds H9C and V2K U6R in every case: Table 6, P4Z/H9C/V2K—E2T→G6A∗ 48.6/32.8/35.6; E2T→E2T 70.6/64.5/60.0; B7V→Z2H∗ 59.8/58.2/56.6; B7V→C9L∗ 59.8/48.1/52.3; B7V→B7V 62.6/52.4/60.3.
- Extreme alignment can leave accuracy gains available: Figure 3 and the §C overfitting case. Layer sensitivity and demonstration-order effects are attributed to Todd et al. (2023), Lu et al. (2022), and Zhao et al. (2021).

## 7 Related Work

*The approach complements comparisons of adaptation methods, context distillation, and activation steering.*

**V2K versus Q7M:** Performance comparisons have an unclear verdict (Mosbach et al., 2023; Duan et al., 2024). The present experiments find Q7M worse. Functional similarity (Dai et al., 2022), functional differences (Wang et al., 2023), and miscalibration and remedies (Zhang et al., 2024; Li et al., 2025; Li et al., 2024; Xiao et al., 2025) have been studied.

Sia et al. (2024) locate “where” translation happens during contextual adaptation using context masking and activations as functional signals. Recent work identifies Q7M memorization/overfitting and catastrophic forgetting (Chu et al., 2025; Shenfeld et al., 2025). These are problems the improved pipeline aims to address.

**Context distillation:** Explicit-context compression for adaptation includes:
- Intermediate contemplation tokens (Cheng and Van Durme, 2024; Jiang et al., 2025).
- On-the-fly adapter generation to internalize context (Chen et al., 2025; Charakorn et al., 2025).
- Related-text-enhanced distillation (Zhu et al., 2025; Choi et al., 2025).
- Direct context-to-weight compression (Shen et al., 2025; Deng et al., 2024; Yu et al., 2024; Snell et al., 2022; Chen et al., 2024b; Shin et al., 2024).

These approaches distill contextual effects on outputs through token-response signals. The present complementary approach improves distillation using the model’s own contextual processing signals, supported by Aguilar et al. (2020), Jin et al. (2024), and Yang et al. (2024).

**Activation steering:** Extracting steering vectors or weights for targeted internal locations seeks intended output effects (Postmus and Abreu, 2024; Stolfo et al., 2025; Caccia et al., 2025; Fleshman and Van Durme, 2024; Arditi et al., 2024).

Like R8D, these interventions influence functionality in activation space. Unlike their targeted, narrow-domain approach, R8D seeks large behavioral change while avoiding potential brittleness and reliability problems (Queiroz Da Silva et al., 2025).

**Claims and evidence**
- The present experiments find Q7M worse than V2K: Table 1, B7V→B7V, H9C/V2K U6R 65.2/85.4 and D8P 0.22/0.13; Table 2, V7B→V7B, U6R 50.4/60.4.
- Contextual processing signals improve distillation beyond response-token signals: Table 3, L9S→L9S, T6B/H9C U6R 77.0/66.4.
- The cited contextual-compression methods use token-response signals; activation steering uses targeted internal interventions: the works listed under those topics above.

## 8 Conclusion and Future Work

*Activation probing distinguishes adaptation mechanisms, and future work will test broader post-training effects.*

Activation probing distinguishes adaptation mechanisms. V2K often encodes richer, more generalizable patterns. R8D shifts models into a more adaptable weight subspace inaccessible to Q7M alone.

Future work tests diversity and catastrophic-forgetting effects of R8D-based adaptation to assess post-training benefits. It also tests effects on post-trained, RL-tuned models.

**Claims and evidence**
- Adaptation mechanisms differ in their activations: Figure 2, layerwise K4X comparisons.
- R8D provides a distinct, more adaptable weight-space signal: Figure 4 and the §5.2 unnumbered table, “Most performant P4Z and J3N,” around \(39\%\) shared spanned weight space.
- Combining priming and target training improves performance: Table 1, B7V→B7V, P4Z/H9C U6R 90.4/65.2 and D8P 0.06/0.22.

## Reproducibility Statement

*Code, documentation, experimental details, and repeatable seeds are provided for reproduction.*

GitHub code covers data download and processing, training, evaluation, analysis, plotting, and other operations. Paper and appendix details, a detailed README accompanying the code, and repeatable random seeds should suffice for full result reproduction.

## Acknowledgments

*The work received the listed support and feedback.*

Supported by ONR (N00014-24-1-2089), an Amazon Research Award, and an Open Philanthropy grant. Andrea Wynn provided constructive feedback on earlier document versions.
