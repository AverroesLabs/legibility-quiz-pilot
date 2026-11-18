This paper asks whether the internal computations of in-context learning (ICL) can improve supervised fine-tuning (SFT). It compares their activation patterns and finds that they differ, even when they use the same data. The proposed method, ICL Activation Alignment (IA2), trains a model without demonstrations to match its own activations with demonstrations. SFT then trains this primed model on ground-truth responses. Experiments cover 12 benchmarks, two model families, and more than 13,000 trained models. The paper reports improvements in SFT accuracy and calibration. It also finds that IA2 and SFT identify different weight subspaces. IA2-based training does not consistently outperform ICL, especially on some multi-token math tasks. The paper also examines simultaneous IA2 and SFT, soft-label distillation, and an alternative parameter-efficient tuning method.

**Glossary**

- **Adaptation:** Specializing or steering a general-purpose model for a task.
- **SFT / Supervised Fine-Tuning:** Training model weights on labeled input–output pairs to produce intended responses.
- **ICL / In-Context Learning:** Adapting model behavior at inference time through demonstrations or instructions in the prompt, without weight updates.
- **IA2 / ICL Activation Alignment:** Self-distillation that trains a model without demonstrations to match its own activations with ICL demonstrations.
- **Priming:** Performing IA2 before SFT.
- **IA2 → SFT:** Sequential IA2 priming followed by SFT.
- **IA2 + SFT:** Training with IA2 and SFT losses simultaneously.
- **SFT only / IA2 only:** Training with only the named objective.
- **Self-distillation:** Using the same model, enhanced with ICL context, as the teacher.
- **Teacher / student:** The ICL-conditioned model supplying training signals and the model trained from those signals.
- **LLM / LM:** A general-purpose language model that can be adapted to specialized applications.
- **Transformer / decoder-only transformer:** The model architecture described as a stack of self-attention and linear layers.
- **SA / self-attention:** An operation whose output at a token position depends on other tokens in the sequence.
- **Linear layer / projection:** A transformation applied to token vectors within the model.
- **LM head:** The component that receives the final-layer token projection to predict the next token.
- **Token:** An element of an input or output sequence processed by the model.
- **Token space / output space:** The space of model responses, contrasted with internal activations.
- **Activation / hidden SA output:** An internal self-attention output used as a footprint of model processing.
- **Activation space:** The model’s internal activation patterns.
- **Functional behavior / functional alignment:** How a model processes inputs internally, and its alignment with ICL processing.
- **Output alignment:** Matching model responses to intended target responses.
- **Activation similarity / asim:** Token-wise cosine similarity between two activation tensors.
- **Cosine similarity:** The similarity measure used to compare activation tensors token by token.
- **Layerwise similarity:** Activation similarity examined separately across model layers.
- **Context / prompt:** The token sequence supplied to the model.
- **Demonstration / demo:** An input–output example placed in the ICL prompt.
- **Query:** The input for which the model must produce a response.
- **Inference:** Producing responses from the model, including ICL adaptation without weight updates.
- **Inference compute / context space:** Resources consumed by processing query, response, and demonstration tokens.
- **Few-shot / data-scarce setting:** Adaptation using small training or demonstration sets.
- **Shot count:** The number of samples, denoted by \(N\).
- **Generalization:** Performance on new inputs or tasks.
- **Calibration:** A response quality evaluated here with expected calibration error.
- **ECE / ece / expected calibration error:** The calibration metric used for single-token responses.
- **Accuracy / acc:** The performance metric based on correct answers; multi-token answers must also follow the required format.
- **\(\uparrow\) / \(\downarrow\):** Table indicators that higher accuracy and lower ECE are preferred.
- **Standard deviation:** The quantity in table parentheses describing variation across runs.
- **Convergence:** The stopping point used for IA2 and SFT training, subject to the stated epoch limit.
- **Epoch:** A training unit, with training limited to up to 50 epochs.
- **Learning rate:** A training hyperparameter tested at slow, medium, and fast settings.
- **Random seed / random run:** A repeatable source of variation used across training and evaluation runs.
- **Hyperparameter selection:** Choosing the best-performing learning rate or learning-rate–\(\beta\) combination.
- **Shortcut learning / overfitting / memorization:** Behaviors discussed as possible weaknesses of output-oriented SFT.
- **Catastrophic forgetting:** A problem associated with SFT in prior work and proposed for future IA2 evaluation.
- **Circuit:** An internal mechanism that ICL relies on across tasks.
- **Internal gradient descent:** A proposed account in which attention behaves as if parameters were updated from demonstrations.
- **Gradient descent:** The parameter-update mechanism invoked by that account of ICL.
- **Cross-entropy:** The loss used to align generated tokens with target tokens in SFT.
- **Mean squared error:** The activation-matching objective described for IA2.
- **Softmax / \(\sigma(\cdot)\):** The function used in the self-attention equation.
- **Linearized attention / kernel approximation:** The approximation under which prior work gives a closed-form alignment solution.
- **Non-linear transformer:** The practical model setting targeted by IA2 rather than the linearized approximation.
- **PEFT / parameter efficient fine tuning:** Adaptation through a small set of parameters.
- **LoRA / LoRAs / LoRA adaptor:** The weight-adaptation method used with rank 8 in the main experiments.
- **Rank / \(r\):** The LoRA weight rank; \(r\) also normalizes the subspace-overlap formula.
- **Target modules:** The weight components selected for adaptation.
- **Full fine tuning / full-rank:** Updating model weights rather than using only parameter-efficient components.
- **Prompt/prefix tuning:** Alternative parameter-efficient methods named in the discussion.
- **(IA)3 / activation scaling:** The alternative PEFT method tested in place of LoRA.
- **GPU memory:** The memory on which small SFT parameter sets can be loaded and unloaded.
- **Weight space / subspace:** The space spanned by adapted weights.
- **Subspace overlap:** The shared weight space measured by \(\frac{1}{r}\|U_1^T U_2\|_F^2\).
- **SVD:** The procedure used to obtain basis vectors from weight matrices.
- **Basis vectors / \(U_1,U_2\):** Vectors obtained from the weight matrices for the two methods being compared.
- **Orthogonal weight updates:** Updates described as having almost no shared subspace with the other methods.
- **Single-token setting:** Tasks whose answer is one token, such as a category, True/False value, or MCQ choice.
- **Multi-token setting:** Tasks whose response may contain multiple tokens and vary in length.
- **Open-ended generation:** The general multi-token response setting.
- **Chain-of-thought:** Demonstration and response formats containing reasoning before the answer.
- **Answer parsing:** Extracting the final answer from generated response text.
- **Exact match:** Requiring the parsed answer to match the ground-truth answer exactly.
- **Ground truth / target response:** The dataset response used for supervised training or evaluation.
- **ID / in-distribution:** Evaluation on the training dataset’s distribution.
- **OOD / out-of-distribution:** Evaluation on another dataset to study distribution shift.
- **Distribution shift:** A change between training and evaluation distributions.
- **MCQ:** The task format of choosing among answer options.
- **True/False:** The binary answer format used for StrategyQA.
- **Multi-hop reasoning:** Reasoning required before the StrategyQA True/False conclusion.
- **QA:** The question-answering task label used for scientific questions.
- **Sentiment classification:** Assigning sentiment categories using SST2, Financial Phrasebank, or Poem Sentiment.
- **News categorization:** Classifying news using AGNews or BBCNews.
- **SST2:** A sentiment-classification dataset used for training and evaluation.
- **Financial Phrasebank / FinS:** A sentiment dataset used for training and evaluation.
- **Poem Sentiment / PoemS:** A sentiment dataset used for evaluation.
- **StrategyQA / STF:** A dataset requiring multi-hop reasoning before a True/False answer.
- **AGNews / AGN:** A news-categorization dataset used for training and evaluation.
- **BBCNews / BBCN:** A news-categorization dataset used for evaluation.
- **SciQ:** A scientific dataset used both for MCQ selection and answer generation.
- **QASC:** An MCQ dataset used for evaluation after choice remapping.
- **SciQr / QASCr:** SciQ and QASC with MCQ choices remapped to unrelated tokens.
- **GSM8K:** The grade-school math dataset used for training and evaluation.
- **GSM Symbolic / GSM8Ks:** The additional grade-school math evaluation dataset.
- **MATH Algebra / HMathA:** A dataset of higher-grade algebra problems.
- **Qwen3 / Qwen3-4B-Base:** A model family and the Qwen base model used across all tasks.
- **Llama-3.2:** The second model family used in the experiments.
- **Llama-3.2-1B / Llama-3.2-3B:** Models used for single-token tasks, with the 3B model used for SciQr.
- **Llama-3.2-1B-Instruct:** The secondary multi-token model used to test a post-trained model.
- **1B / 3B / 4B:** Model-size labels associated in §3 with 16, 28, and 36 layers.
- **Base model:** The model before the adaptation procedure being evaluated.
- **Instruct / post-trained model:** The model category used to test IA2 after prior training beyond the base model.
- **RL-tuned:** A post-trained model category proposed for future study.
- **GPT-3:** The model in which the paper says ICL was first illustrated.
- **STEM:** The data category cited as a suspected reason for Qwen’s sample-efficient math ICL.
- **Random baseline:** The performance level that Llama-3.2-1B did not exceed on SciQr up to \(N=128\).
- **Knowledge distillation / KD:** Transferring teacher behavior through soft-label matching.
- **Soft labels / soft-label matching:** Training signals using the teacher’s token probability mass.
- **ICL response:** A response generated by the model with ICL context.
- **Context distillation / compression:** Internalizing the effect of explicit context in model weights or other compressed representations.
- **Contemplation tokens:** Intermediate tokens used for compression in related work.
- **Adapter generation:** Producing adaptation components on the fly to internalize context.
- **Activation steering / steering vectors or weights:** Interventions at selected model locations to influence outputs.
- **Context masking:** The method cited for locating translation behavior during ICL.
- **Selective IA2:** Aligning selected layers rather than all layers.
- **Ablation:** A proposed study of choices such as LoRA rank and target modules.
- **Prompt optimization:** Improving the prompt before collecting target activations.
- **Computational overhead:** The one-time ICL activation-collection cost before IA2 training.
- **\(\mathcal{T}\):** The adaptation task.
- **\(\mathcal{D}_{\mathcal{T}}\):** The task dataset \(\{(X_i,Y_i)\}_{i=1}^{N}\).
- **\(N\):** The number of training samples or shots.
- **\(N-1\):** The remaining samples used as demonstrations for each IA2 training sample.
- **\(X,X_i,X_t,x_i\):** Query inputs; \(X_t\) denotes a new test input.
- **\(Y,Y_i,Y_t\):** Ground-truth responses.
- **\(Y_{ij}\):** The \(j\)-th ground-truth token for sample \(i\).
- **\(\bar y_t,\bar Y_t,\bar Y_i,\bar Y_{ij}\):** A fine-tuned model’s first response token, full response, sample response, and response token.
- **\(\hat y_t,\hat Y_t,\hat Y_i\):** The first token or full response generated with ICL.
- **\(M_\Theta\):** The language model with parameters \(\Theta\).
- **\(\Theta,\Theta'\):** Model parameters before and after an update.
- **\(M_{\Theta'}\):** The model with updated parameters.
- **\(T,T_i\):** A token sequence supplied to the model.
- **\(t_i\):** The token vector at position \(i\).
- **\(t_{:i}\):** The first \(i\) tokens.
- **\(R\):** The number of processed tokens.
- **\(d\):** The token-vector dimension in the stated tensor shapes.
- **\(L\):** The number of model layers.
- **\(G\):** The number of response tokens; capped at 200 for IA2 training.
- **\(i,j,n\):** Indices for samples or token positions; \(n\) indexes the demonstrations in \(I\).
- **\(I\):** The concatenated tokens of ICL demonstrations.
- **\(\circ\):** Token-sequence concatenation.
- **\(W_Q,W_K,W_V,W_O\):** The four weight matrices used by the stated SA operation.
- **\(W_{QKVO}\):** The set of those four attention matrices.
- **\(\tilde W_{QKVO}\):** Modified attention weights sought by IA2.
- **\(q_i,K,V\):** The projections \(t_iW_Q\), \(t_{:i}W_K\), and \(t_{:i}W_V\).
- **\(Z,z_i\):** The self-attention output sequence and its output at token position \(i\).
- **\(A,A_1,A_2,A^i\):** Activation tensors; \(A^i\) is the unaligned output-position tensor for sample \(i\).
- **\(A_{\mathrm{ICL}},A^i_{\mathrm{ICL}}\):** Activations collected with ICL, including the sample-specific target tensor.
- **\(A_{\mathrm{SFT}}\):** Activations collected from an SFT model without ICL demonstrations.
- **\(\hat A_i,\hat A\):** The activation tensor notation used for IA2 collection and its target in the discussion.
- **\(\mathbb{R}^{L\times G\times d}\):** The stated shape of a sample’s ICL response-activation tensor.
- **\(\mathcal L_{\mathrm{SFT}}\):** The summed cross-entropy loss on target response tokens.
- **\(\mathcal L_{\mathrm{IA2}}\):** The summed activation-matching loss in Equation 4.
- **\(\beta\):** The weight on SFT loss in simultaneous IA2 + SFT training.
- **\(\sum\):** Summation over the samples or tokens specified in an equation.
- **\(\|\cdot\|\):** The activation-difference norm written in Equation 4.
- **\(\|\cdot\|_F^2\):** The squared matrix norm used in the subspace-overlap formula.
- **Superscript \(T\):** Matrix transpose in the attention and overlap equations.
- **\(\cdot\):** Multiplication in the stated equations.
- **\(\sqrt d\):** The scaling denominator in the attention equation.
- **\(\approx\):** The approximate equality sought between ICL and modified-weight attention outputs.
- **\(\forall,\in\):** “For every” and set membership in the stated objectives.
- **\(-1\):** The newly generated token index in Equation 3.
- **w/o ICL:** Evaluation without ICL demonstrations.
- **w.r.t.:** With respect to model parameters in the IA2 training description.
- **wrt:** With respect to the comparison method in Table 5’s gap columns.
- **\(\ast\):** The table marker for OOD evaluation.
- **–:** An unreported ECE entry in Table 4.
- **`<think>`:** Tags explicitly not used for the multi-token reasoning text.
- **`####`:** The separator before the answer in GSM8K, GSM8Ks, and constructed SciQ responses.
- **`The answer is <value>`:** An additional math-answer pattern accepted by parsing.
- **`\boxed{}`:** The element from which HMathA answers are parsed.
- **`[A, B, C …]`:** The original MCQ choice labels remapped for SciQr and QASCr.
- **`1e-4`, `3e-4`, `1e-3`:** The slow, medium, and fast learning rates tested.
- **§A / §B / §C:** Appendix sections cited for datasets, experimental setup, and additional results.
- **README / github:** The documentation file and code-hosting location named in the reproducibility statement.
- **ONR / `N00014-24-1-2089`:** A named supporter and its grant identifier.
- **Amazon Research Award / Open Philanthropy:** The other named funding sources.

# IA2: Alignment with ICL Activations

Improves Supervised Fine-Tuning

Aayush Mishra

  
Daniel Khashabi & Anqi LiuDepartment of Computer Science,
Johns Hopkins University

###### Abstract

*The paper proposes matching ICL activations before SFT and reports improved adaptation.*

Supervised Fine-Tuning (SFT) specializes model behavior by training weights to produce intended responses to queries. In-Context Learning (ICL) instead adapts a model during inference—response generation—through instructions or demonstrations in its prompt. In data-scarce settings, ICL can generalize better and produce more calibrated responses than SFT, but requires more inference compute. Calibration is evaluated here through expected calibration error.

We ask whether ICL’s internal computations can improve SFT. We first show that ICL and SFT produce distinct activations—the hidden outputs of self-attention, an operation affected by other context tokens. These patterns indicate different functional mechanisms for adaptation.

We introduce ICL Activation Alignment (IA2), a self-distillation technique that uses the model’s own ICL processing as a teacher. IA2 aims to reproduce ICL activation patterns in an SFT model and incentivize ICL-like internal reasoning. Using IA2 as a priming step before SFT significantly improves accuracy and calibration across 12 popular benchmarks and two model families. The finding offers practical benefits and a conceptual window into model adaptation.

**Claims and evidence**
- ICL and SFT produce distinct activation patterns: Figure 2, the “SFT only” layerwise comparisons described in §3.
- IA2 priming improves SFT: Table 1, FinS → FinS, accuracy 67.4 versus 78.7 and ece 0.31 versus 0.16; Table 2, HMathA → HMathA, accuracy 50.4 versus 55.3.
- The study covers 12 benchmarks and two model families: §5 identifies the tasks and Qwen3 and Llama-3.2 models.

00footnotetext: ♡Equal advising. Correspondence to: {amishr24,danielk,aliu.cs}@jhu.edu

## 1 Introduction

*ICL and SFT differ in cost, calibration, and internal processing; IA2 uses this difference to improve SFT.*

LLMs are general-purpose language models often used in specialized applications. A news aggregator, for example, may classify articles into predefined categories. SFT adapts LLMs to narrow downstream tasks using labeled samples. Parameter efficient fine tuning (PEFT) methods adapt only a small parameter set (Hu et al., 2021; Liu et al., 2022). These SFT parameters can be efficiently loaded onto or removed from GPU memory. However, SFT typically needs many labeled samples to generalize to new tasks (Le Scao and Rush, 2021), and collecting them can be expensive.

ICL (Brown et al., 2020) instead adapts and steers behavior at inference time. A query is preceded by demonstrations—input–output examples—or instructions. These help the model “learn” the task and answer accordingly. Prior work (Duan et al., 2024), consistent with our experiments (§5), finds that ICL generalizes well in few-shot settings, where only a small number of examples are available, and typically produces well-calibrated responses. But demonstrations and instructions consume context space and increase each query’s cost. That space could otherwise hold more query and response tokens.

These differences motivate us to compare ICL’s and SFT’s functional behavior: how they process inputs internally. Their token-space behavior—their responses—may look similar, but their activations differ (§3). Activations record internal processing before the next token is produced. Their divergence therefore highlights different adaptation mechanisms, consistent with Shen et al. (2024).

We hypothesize that ICL activations contain rich information about extracting generalizable patterns from context. SFT models may lack this information, especially with few samples, when they can learn shortcuts or overfit target responses. We ask:

> Can the information rich ICL activations be used to improve the quality of SFT?

Recent work distills context into LLM weights (Snell et al., 2022; Chen et al., 2024b), but uses only response-text training signals. These signals may retain the SFT problems above. Reproducing an ICL-conditioned model’s outputs does not ensure ICL-like internal functioning.

IA2 addresses this by aligning a model with its own functional behavior under ICL (§4). Figure 1 outlines the pipeline: (1) collect information-rich ICL activations; (2) enforce functional alignment with ICL; and (3) perform SFT on the primed model.

Priming with IA2 before SFT, using the same data, drastically improves performance on text classification and generation tasks (§5). IA2 also supplies an important training signal unavailable from SFT alone. We trained over 13,000 models across 12 benchmarks. The findings demonstrate practical benefits and illuminate adaptation mechanisms.

In summary:
- ICL and SFT using the same data do not align in activation space, revealing a functional gap (§3).
- IA2 enforces alignment with ICL’s functional behavior. This priming step drastically improves SFT performance (Figure 1, §4).
- The IA2 training signal is absent from SFT-only training, highlighting its importance for adaptation (§5).

Figure 1: An overview of our improved SFT pipeline. Standard SFT only enforces output space alignment (between the model’s response and a target response—only step 3 above), resulting in subpar performance and mis-calibration in low-data settings. In contrast, our method IA2 enforces functional alignment with ICL by matching the rich activation patterns produced when performing ICL. IA2 priming before SFT boosts the quality of adaptation. We show this improvement through performance comparison charts aggregated across models and datasets. See section 5 for details.

**Claims and evidence**
- ICL and SFT do not align in activation space: Figure 2, “SFT only,” using the same demonstrations for ICL and SFT.
- IA2 → SFT improves adaptation with the same data: Table 1, SST2 → SST2, accuracy 65.2 versus 90.4 and ece 0.22 versus 0.06; Table 2, SciQ → SciQ, accuracy 35.0 versus 40.8.
- IA2 supplies a different training signal: Figure 4; §5.2 reports around \(39\%\) average weight-subspace overlap between IA2 only and IA2 → SFT, while SFT-only updates are almost completely orthogonal.

## 2 Background and Notation

*This section defines the model, SFT and ICL procedures, and activation similarity.*

**Transformer Language Models:** A standard decoder-only transformer LM, \(M_\Theta\), has parameters \(\Theta\) and a stack of self-attention (SA) and linear layers. It processes token vectors \(T=[t_i]_{i=1}^{R}\), where \(R\) is the sequence length. At each layer, SA and linear projections act on each token. The final layer’s last-token projection passes through the LM head, which predicts the most likely next token.

SA is affected by other tokens in the sequence. For an SA operation with 4 weight matrices, \(W_Q,W_K,W_V,W_O\in\Theta\), collectively \(W_{QKVO}\), the output is:

\[
\operatorname{SA}(T;W_{QKVO})=Z
=\left[z_i=\sigma\left(\frac{q_iK^T}{\sqrt d}\right)V\cdot W_O\right]_{i=1}^{R}.
\tag{1}
\]

Here, \(\sigma(\cdot)\) denotes Softmax; \(q_i=t_i\cdot W_Q\), \(K=t_{:i}\cdot W_K\), and \(V=t_{:i}\cdot W_V\). The notation \(t_{:i}\) means the first \(i\) tokens, and superscript \(T\) denotes transpose. Both output \(Z\) and input \(T\) have shape \(R\times d\), where \(d\) is the token-vector dimension. We study changes in model behavior induced by changing either context \(T\) or weights \(W_{QKVO}\).

**Supervised Fine-Tuning:** Let
\[
\mathcal D_{\mathcal T}=\{(X_i,Y_i)\}_{i=1}^{N}
\]
be \(N\) input–output pairs illustrating task \(\mathcal T\). SFT updates weights \(\Theta\rightarrow\Theta'\) to produce relevant responses to new inputs \(X_t\):
\[
M_{\Theta'}(X_t)=\bar y_t.
\]
Here, \(\bar y_t\) is the first token of response \(\bar Y_t\). Repeatedly applying \(M_{\Theta'}\) to the growing sequence extends that response.

If ground-truth response \(Y_i\) has \(G\) tokens, we generate \(G\) tokens to obtain \(\bar Y_i\) and minimize cross-entropy, the token-target matching loss, over all generated tokens:

\[
\mathcal L_{\text{SFT}}
=\sum_{i=1}^{N}\sum_{j=1}^{G}
\operatorname{cross\text{-}entropy}(\bar Y_{ij},Y_{ij}).
\tag{2}
\]

The index \(j\) selects a response token.

**In-Context Learning:** The same dataset can supply demonstration tokens
\[
I=[X_1\circ Y_1\circ\ldots\circ X_n\circ Y_n],
\]
where \(\circ\) denotes concatenation and \(n\) indexes demonstrations. The model receives \(I\) followed by a new test sample \(X_t\), without weight updates:
\[
M_\Theta(I\circ X_t)=\hat y_t.
\]
It processes the prompt examples to understand the task and respond. The full ICL response \(\hat Y_t\) is often similar to expected response \(Y_t\).

ICL was first illustrated in GPT-3 (Brown et al., 2020) and is widely used for inference-time adaptation of generic LMs. Today’s frontier models can directly follow instructions, so a generic instruction can be considered a form of ICL demonstration. Here, we focus on classic ICL with input–output demonstration pairs.

**Activations:** SA uses every context token, so different sequences produce different hidden SA outputs, which we call activations. They are a footprint of internal processing and can reveal behavioral changes across contexts. A model with \(L\) layers processing \(R\) tokens produces an activation tensor \(A\) of size \(L\times R\times d\).

We study \(T=[I\circ X]\) for ICL and \(T=X\) without ICL under different model weights. Here, \(I\) contains demonstration tokens and \(X\) contains query tokens. Activation similarity,
\(\operatorname{asim}(A_1,A_2)\), is the token-wise cosine similarity between two activation tensors. Its size is \(L\times R\).

## 3 Are ICL and SFT practically the same?

*ICL and SFT produce different internal activations and differ in calibration.*

Figure 2:
Layerwise Similarity with ICL Activations.
SFT models have small asim with ICL. This indicates differing functional behavior for adaptation. Our recommended pipeline (IA2 → SFT) aligns much more with ICL than using only SFT, and performs much better as a result (see Figure 1).

Recent studies (Von Oswald et al., 2023; Akyürek et al., 2022) claim that transformer ICL operates through internal gradient descent: attention produces outputs as if model parameters had been updated using context demonstrations. This implies functional equivalence between ICL and parameter updates. Under this equivalence, a base model with demonstrations should produce activations at output-token positions similar to those of an SFT model without demonstrations. We test whether LLMs actually exhibit this behavior.

**ICL and SFT produce different activations:** We collected \(A_{\text{ICL}}\), the activations with \(T=[I\circ X]\), for 100 test samples from multiple datasets, including SST2, a sentiment dataset, and AGNews, a news-categorization dataset. We used Qwen3 (Yang et al., 2025) and Llama-3.2 (Grattafiori et al., 2024) models: 1B \(\leftrightarrow\) 16 layers, 3B \(\leftrightarrow\) 28 layers, and 4B \(\leftrightarrow\) 36 layers.

We trained SFT models on the same demonstrations until convergence, then collected \(A_{\text{SFT}}\) with \(T=X\). Experimental details are in §B, the cited setup appendix. We calculated \(\operatorname{asim}(A_{\text{ICL}},A_{\text{SFT}})\) at output-token positions and averaged across tokens, samples, and datasets. Figure 2’s first row, “SFT only,” shows the result.

ICL and SFT activations are not aligned across models. Alignment is better near initial and final layers, where we expect individual-token processing. It is worse in the middle layers, where we expect abstract processing of the full demonstration set.

**Why should we care about alignment with ICL?** The functional difference also appears in expected calibration error (Guo et al., 2017), as Figure 1 shows. ICL is much more calibrated than SFT at similar, slightly better accuracy.

We suspect SFT’s output-oriented signal permits shortcuts that fail on new data. ICL instead relies on complex circuits—internal mechanisms that must work across tasks (Elhage et al., 2021). It extracts generalizable demonstration patterns and performs in a well-calibrated manner.

This experiment demonstrates different means of adaptation. SFT should not be expected to replace ICL directly. We therefore ask:

> Can the information rich ICL activations be used to improve the quality of SFT?

The next section describes our procedure and shows that this is possible.

**Claims and evidence**
- ICL and SFT have different activations across models: Figure 2, “SFT only,” from the comparison using 100 test samples and models with 16, 28, and 36 layers.
- Alignment is better near initial and final layers and worse in the middle: Figure 2, the layerwise “SFT only” comparisons.
- ICL is better calibrated at similar, slightly better accuracy: Figure 1, the aggregated accuracy and expected-calibration-error comparisons.

## 4 Inducing ICL-like behavior in SFT Models

*IA2 matches ICL activations first; subsequent SFT aligns responses with ground-truth targets.*

Our two-step pipeline first uses IA2 to incentivize ICL-like functional behavior. It then applies SFT to the ICL-aligned weights, producing much better adaptation qualities.

**IA2—ICL Activation Alignment:** Section 3 found little activation similarity between SFT and ICL. To increase asim, and thus functional similarity with ICL, we seek modified attention weights at each layer:
\[
\tilde W_{QKVO}=(\tilde W_Q,\tilde W_K,\tilde W_V,\tilde W_O).
\]
For each newly generated token, at index \(-1\), the goal is:

\[
\operatorname{SA}([I\circ X];W_{QKVO})
\approx
\operatorname{SA}(X;\tilde W_{QKVO})
\quad\forall X\in\mathcal T.
\tag{3}
\]

Here, \(\approx\) expresses the desired approximate match, and \(\forall X\in\mathcal T\) states that it should hold for every task input.

Chen et al. (2024a) give a closed-form solution for linearized attention, using a kernel approximation to Softmax. We seek a practical, general solution for real non-linear transformers. The modified weights are not encouraged to produce similar outputs, but to process queries at each layer as the base model does under ICL.

First, for each training sample \(X_i\), we generate an ICL response \(\hat Y_i\) using \(T_i=[I\circ X_i]\). The remaining dataset samples construct \(I\). We collect output-position activations:
\[
A^i_{\text{ICL}}\in\mathbb R^{L\times G\times d},
\]
assuming \(G\) response tokens.

Next, we provide only the query and that response,
\[
T_i=[X_i\circ\hat Y_i],
\]
as if the model had produced the ICL response without demonstrations. This produces an unaligned output-position activation tensor \(A^i\). We train model parameters to minimize the mean squared error between the two activations:

\[
\mathcal L_{\textbf{IA2}}
=\sum_{i=1}^{N}\|A^i-A^i_{\text{ICL}}\|.
\tag{4}
\]

The norm notation is the activation-difference expression used in this loss.

**IA2 → SFT:** After IA2, we continue training with the standard SFT cross-entropy loss on target tokens (Equation 2). If IA2 changes \(\Theta\) to \(\Theta'\), we collect responses \(\bar Y_i\) from \(M_{\Theta'}\) for every \(X_i\), then minimize SFT loss against ground truth \(Y_i\). This further aligns outputs with intended targets.

The pipeline is:

1. Collect target activations \(A_{\text{ICL}}\). Train with \(\mathcal L_{\textbf{IA2}}\) until convergence to align functional behavior with ICL.
2. Switch to \(\mathcal L_{\text{SFT}}\) with ground-truth responses \(Y\). Train until convergence to align output behavior.

**Claims and evidence**
- SFT has little asim with ICL: Figure 2, “SFT only,” as described in §3.
- IA2 followed by SFT produces better adaptation: Table 1, FinS → FinS, accuracy 67.4 versus 78.7 and ece 0.31 versus 0.16; Table 2, GSM8Ks evaluation after GSM8K training, accuracy 64.5 versus 68.8.

## 5 Experimental Results

*The experiments compare adaptation methods on single-token and multi-token tasks using the same training data.*

We study two settings with different characteristics: single-token and multi-token responses. We note differences as we present results.

**Adaptation Tasks:** In the single-token setting, \(Y_i\) is one token: typically a class label, True/False answer, or MCQ answer choice. MCQ tasks require selection among options. In the multi-token setting, \(Y_i\) may contain multiple tokens and vary in length, as in open-ended generation.

We compare the pipeline across tasks and models. We also often use out-of-distribution (OOD) evaluation—testing on another dataset—to examine distribution shift. In-distribution (ID) evaluation uses the training dataset’s distribution.

**Single-token tasks:**

- **Sentiment Classification:** SST2 (Socher et al., 2013), Financial Phrasebank [FinS] (Malo et al., 2014), and Poem Sentiment [PoemS] (Sheng and Uthus, 2020). We train on SST2 or FinS and evaluate on all three. Each training set therefore has one ID and 2 OOD evaluations.
- **True/False:** StrategyQA [STF] (Geva et al., 2021), which requires multi-hop reasoning before a True/False conclusion.
- **News Categorization:** AGNews [AGN] (Zhang et al., 2015) and BBCNews [BBCN] (Greene and Cunningham, 2006). We train on AGN and evaluate on both.
- **MCQ:** SciQ (Johannes Welbl, 2017) and QASC (Khot et al., 2019). To add complexity, we remap choices `[A, B, C …]` to unrelated tokens. We call the remapped datasets SciQr and QASCr. We train on SciQr and evaluate on both.

**Multi-token tasks:**

- **Grade-school Math:** GSM8K (Cobbe et al., 2021) and GSM Symbolic [GSM8Ks] (Mirzadeh et al., 2024). We train on GSM8K and evaluate on both.
- **Advanced Math:** MATH Algebra [HMathA] (Hendrycks et al., 2021), containing higher-grade algebra problems.
- **Scientific QA:** SciQ again, but generating answer tokens rather than selecting an option.

**Data Setup:** We focus on the data-scarce, few-shot setting in which ICL is typically used. For each task, we construct training datasets
\[
\mathcal D_{\mathcal T}=\{(X_i,Y_i)\}_{i=1}^{N}
\]
with \(N\) varying in orders of \(2:[2,4,8,16,\ldots]\). We create 5 sets per \(N\) to average out outlier effects.

For IA2, each \(A^i_{\text{ICL}}\) is collected using the remaining \(N-1\) samples as demonstrations in random order. This reuses training samples and gives every adaptation method exactly the same data. Evaluation uses a separate sample of 500 examples per task. Dataset and processing details are in §A.

**Multi-token details:** The model generates reasoning before the answer in a demonstrated format. These are ordinary text tokens, not text inside `<think>` tags. Demonstrations and expected outputs use chain-of-thought style—reasoning followed by an answer (Wei et al., 2022). Evaluation extracts the answer from generated text using dataset-specific parsing:

- **GSM8K and GSM8Ks:** Ground truth follows `"<reasoning steps> #### <numerical answer>"`. We do not evaluate reasoning correctness or conciseness. We extract the numerical answer and require an exact match. We also accept common patterns such as `"The answer is <value>"` to allow slight format variation. The exact parsing function is in our code.
- **HMathA:** Answers are expected inside a `\boxed{}` element and parsed accordingly.
- **SciQ:** We construct `"<support text> #### <text answer>"`, parse that format, and count only exact string matches as correct.

**Training Setup:** We use two model families. Qwen3-4B-Base is used for every single- and multi-token task. Llama-3.2-1B is also used for every single-token task except SciQr, where we use Llama-3.2-3B. The 1B Llama model could not improve above the random baseline for any \(N\) up to 128. For multi-token tasks, the secondary model is Llama-3.2-1B-Instruct. This tests IA2 on a post-trained model.

We train LoRAs—weight-adaptation components—with rank 8, modifying \(W_Q,W_K,W_O\) in SA layers. Model and weight choices reflect available resources: we trained more than 13,000 models in total.

Multi-token training is much more resource extensive because answers are long. We therefore use \(N\) only up to 16 for Llama and 8 for Qwen. During training, we cap generated tokens at \(G=200\) for efficiency. IA2 uses this cap to collect
\[
\hat A_i\in\mathbb R^{L\times200\times d}\quad\forall x_i.
\]
For SFT, \(G\) is the ground-truth answer’s token count.

Each method–dataset combination trains to convergence, up to 50 epochs, with 3 learning rates: slow, `1e-4`; medium, `3e-4`; and fast, `1e-3`. We select the best-performing rate to neutralize hyperparameter-selection effects on method performance. Further details are in §B.

**Evaluation:** Single-token evaluation uses accuracy (acc) and expected calibration error (ECE, also written ece), both based on the first generated token’s probability distribution.

We first benchmark the base model with ICL. Each task’s validation set is evaluated 5 times with different demonstrations, and we report average metrics. Each training method is then evaluated without demonstrations, reporting its best-performing average metrics.

For multi-token evaluation, we limit generated tokens to conserve compute. GSM8K and GSM8Ks use a limit of 200, based on their 95 percentile answer lengths of 184 and 154. HMathA uses 400 because its 95 percentile answer length is 307. We measure only accuracy: questions answered correctly in the required format (§A). We do not measure calibration because confidence is difficult to measure for open-ended generation.

**Claims and evidence**
- Llama-3.2-1B did not improve above the random baseline on SciQr: the training-setup report states this for every \(N\) up to 128; it provides no table row.
- The experiments reuse the same adaptation data: the data setup uses \(N-1\) remaining samples as demonstrations for each IA2 sample and 5 training sets per \(N\).
- The study trained more than 13,000 models: the training-setup report states this total.

### 5.1 IA2 priming boosts SFT performance

*IA2 → SFT generally improves single-token accuracy and calibration and improves multi-token accuracy over SFT only.*

**Single-Token results:** Table 1 reports adapted Qwen models at the practical setting \(N=4\). IA2 → SFT outperforms SFT-only training in almost all cases for both accuracy and calibration, using the same amount of data. In most cases, its accuracy also exceeds ICL’s, with slightly less calibration.

IA2-only training often reaches high accuracy while remaining reasonably calibrated like ICL, despite receiving no target-response-token training. This highlights the richness of the IA2 signal. Figure 1 summarizes aggregate trends across models and datasets. Additional model/\(N\) tables and a statistical significance test are in §C, the cited additional-results appendix.

| Source | Eval | ICL acc ↑ | ICL ece ↓ | SFT only acc ↑ | SFT only ece ↓ | IA2 only acc ↑ | IA2 only ece ↓ | IA2 → SFT acc ↑ | IA2 → SFT ece ↓ |
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

Table 1:
Performance report for \(N=4\) on Qwen3-4B-Base model, showing accuracy (acc) and Expected Calibration Error (ece). Numbers in parentheses show standard deviations across 5 runs for the best performing learning rate. Best training method shown in bold. (∗) highlights OOD evaluations. Our proposed IA2 → SFT training method outperforms standard SFT across the board.

Here and in later tables, ↑ marks higher preferred accuracy, ↓ marks lower preferred ECE, and ∗ identifies an OOD evaluation. Parentheses give the reported standard deviations.

| Source | Eval | w/o ICL acc ↑ | ICL acc ↑ | SFT only acc ↑ | IA2 only acc ↑ | IA2 → SFT acc ↑ |
|---|---|---|---|---|---|---|
| GSM8K | GSM8K | 56.4 (00.0) | 76.4 (01.1) | 70.9 (02.8) | 77.4 (02.8) | 73.6 (03.7) |
| | GSM8Ks∗ | 45.4 (00.0) | 68.4 (01.1) | 64.5 (04.0) | 66.2 (03.7) | 68.8 (05.1) |
| HMathA | HMathA | 21.0 (00.0) | 60.4 (02.1) | 50.4 (01.8) | 47.8 (06.0) | 55.3 (02.1) |
| SciQ | SciQ | 15.4 (00.0) | 37.5 (01.0) | 35.0 (07.1) | 06.9 (13.8) | 40.8 (07.6) |

Table 2:
Performance report for \(N=4\) on Qwen3-4B-Base models, on multi-token datasets. Best training method shown in bold. IA2 → SFT outperforms standard SFT across all datasets.

The “w/o ICL” column evaluates the model without demonstrations.

**Multi-Token results:** Table 2 reports adapted Qwen accuracy for multi-token tasks. IA2 → SFT outperforms SFT-only training in every case. Unlike the single-token setting, it sometimes performs worse than ICL. We suspect that longer contexts are imperfectly compressed into the same small LoRA weight space; larger ranks should improve this. Further model/\(N\) tables are in §C.

**Claims and evidence**
- IA2 → SFT improves single-token accuracy and calibration over SFT only in almost all cases: Table 1, AGN → AGN, accuracy 27.9 versus 31.8 and ece 0.52 versus 0.35; FinS → FinS, 67.4 versus 78.7 and 0.31 versus 0.16; SST2 → SST2, 65.2 versus 90.4 and 0.22 versus 0.06.
- IA2 → SFT usually exceeds ICL accuracy with slightly less calibration: Table 1, FinS → FinS, accuracy 63.6 versus 78.7 and ece 0.12 versus 0.16; FinS → PoemS∗, 56.9 versus 60.6 and 0.12 versus 0.36.
- IA2 only can achieve high accuracy and reasonable calibration without target-token training: Table 1, SciQr → SciQr, accuracy 90.4 and ece 0.09; SST2 → FinS∗, accuracy 71.3 and ece 0.21.
- IA2 → SFT exceeds SFT-only multi-token accuracy in all reported datasets: Table 2, GSM8K, 70.9 versus 73.6; GSM8Ks∗, 64.5 versus 68.8; HMathA, 50.4 versus 55.3; SciQ, 35.0 versus 40.8.
- IA2 → SFT sometimes remains below ICL on multi-token tasks: Table 2, GSM8K, 73.6 versus 76.4; HMathA, 55.3 versus 60.4.

### 5.2 Why is IA2 priming important?

*Activation similarity and weight-subspace overlap indicate that IA2 supplies a signal unavailable from SFT alone.*

Figure 3: We scatter asim \((A,A_{\text{ICL}})\)
vs Accuracy/ECE for all training methods to show the impact of asim on performance metrics. Each point corresponds to one training experiment. With increasing similarity, ECE goes down smoothly. But extreme ICL activation alignment may leave some accuracy gains on the table that can be sourced using SFT on ground truth responses.

We examine three methods: 1) SFT only, 2) IA2 only, and 3) IA2 → SFT.

**Activation Similarity vs Performance:** Figure 2 compares all 3 methods with ICL. IA2 drastically raises activation similarity, as expected. IA2 → SFT retains greater similarity even after SFT.

Figure 3 relates similarity to performance. IA2 → SFT lies between IA2 only and SFT only in activation similarity while achieving better accuracy and calibration. SFT only and IA2 → SFT differ only in their starting state. This implies that IA2 supplies a rich training signal unavailable through SFT alone.

More activation similarity clearly implies better calibration. But extreme similarity may not maximize accuracy because the ICL signal is not always correct. Section C includes a case where IA2 follows ICL performance even when that performance worsens through overfitting. Thus, combining the signals is best: IA2 aligns internal functioning with ICL, and SFT on ground truth aligns outputs with expected behavior.

**Subspace Overlap:** Activation patterns reflect functional behavior but do not isolate training-induced changes. Our LoRA adaptors leave the base weights unchanged, and those weights strongly influence activations. We therefore analyze LoRA weights directly.

Figure 4: IA2 → SFT models have significantly high subspace overlap with IA2 only models, while SFT only models do not align with either. This indicates a training signal in IA2 that is absent from SFT only training and, which is important for performance.

Figure 4 shows distributions of weight-subspace overlap—the weight space shared by two methods—across all single- and multi-token experiments. We first use SVD to obtain basis vectors from weight matrices. We then calculate:
\[
\frac{1}{r}*\|U_1^T\cdot U_2\|_F^2,
\]
where \(U_1,U_2\) are the two methods’ basis vectors, \(r\) is LoRA rank, and \(\|\cdot\|_F^2\) is the squared matrix norm used in the overlap calculation.

This isolates and compares differences in functional behavior. SFT-only updates are almost completely orthogonal to both other methods. The most performant IA2 → SFT models share around \(39\%\) of their spanned weight space, on average, with IA2-only models.

This implies that IA2 produces much of the IA2 → SFT performance gain and identifies subspaces practically unreachable through SFT alone. Overlap between IA2 only and IA2 → SFT varies widely because many IA2 models are already output-aligned. Subsequent SFT then converges with little model change, producing very high overlap.

**Claims and evidence**
- IA2 raises similarity to ICL, and subsequent SFT retains more similarity than SFT only: Figure 2, the three methods’ layerwise comparisons.
- Greater similarity accompanies lower ECE, while extreme similarity can leave accuracy gains unrealized: Figure 3, asim versus accuracy/ECE; §C supplies the cited overfitting case.
- SFT-only weight updates are almost completely orthogonal to both IA2 methods: Figure 4, the SFT-only pairwise overlap distributions.
- IA2 → SFT retains a substantial IA2 weight subspace: Figure 4 and this section’s reported average of around \(39\%\) overlap with IA2 only.
- Already output-aligned IA2 models can change little during subsequent SFT: Figure 4’s broad IA2-only versus IA2 → SFT overlap distribution, explained in the text.

## 6 Discussion

*The discussion tests simultaneous losses, soft-label distillation, shot counts, and (IA)3, then identifies costs and future improvements.*

**IA2 + SFT:** A natural alternative to sequential IA2 → SFT is simultaneous IA2 + SFT: using \(\mathcal L_{\text{IA2}}\) and \(\mathcal L_{\text{SFT}}\) together. This creates a practical problem. For any \(G>1\), target activation tensor \(\hat A\) uses the model’s own generated ICL-response tokens. This response can differ from ground truth, even in length, making the two objectives incompatible for joint training.

Instead, we can self-distill ICL responses, as in Snell et al. (2022), while adding IA2. We train with:
\[
\mathcal L_{\text{IA2}}+\beta\cdot\mathcal L_{\text{SFT}},
\]
where \(\beta\) weights SFT loss. We test 4 \(\beta\) values spanning IA2-only to SFT-only training (§B). We select the best learning-rate–\(\beta\) combination across 5 random seeds.

Table 3 reports multi-token results. Compared with ground-truth training in Table 2, SFT-only and IA2 → SFT performance drops. But IA2 + SFT extracts significantly more performance from ICL responses than SFT only does. This highlights the richness of IA2’s signal. Further IA2 + SFT results are in §C.

| Source | Eval | w/o ICL acc ↑ | ICL acc ↑ | SFT only acc ↑ | IA2 only acc ↑ | IA2 → SFT acc ↑ | IA2 + SFT acc ↑ |
|---|---|---|---|---|---|---|---|
| GSM8K | GSM8K | 56.4 (00.0) | 76.4 (01.1) | 66.4 (18.1) | 77.4 (02.8) | 67.2 (22.1) | 77.0 (07.8) |
| | GSM8Ks∗ | 45.4 (00.0) | 68.4 (01.1) | 59.8 (20.2) | 66.2 (03.7) | 61.6 (24.0) | 69.0 (09.6) |
| HMathA | HMathA | 21.0 (00.0) | 60.4 (02.1) | 49.0 (03.0) | 47.8 (06.0) | 54.0 (01.7) | 53.6 (02.7) |
| SciQ | SciQ | 15.4 (00.0) | 37.5 (01.0) | 27.2 (12.0) | 06.9 (13.8) | 32.8 (10.0) | 34.5 (10.6) |

Table 3:
Performance report for \(N=4\) on Qwen3-4B-Base models when trained with ICL responses instead of ground truth responses. IA2 + SFT performs significantly better than SFT only highlighting the significance of IA2.

**Knowledge Distillation baseline:** Knowledge distillation (KD) commonly transfers a stronger teacher’s behavior through soft-label matching (Hinton et al., 2015). Soft labels supply a denser signal through the probability mass assigned to each token. Here, the teacher is the same model enhanced with ICL context.

We compare soft-label matching with IA2-based training on Llama models: GSM8K with 1B-Instruct and SST2 with 1B. Both use ICL responses. Table 4 shows that soft-label matching significantly improves on standard SFT for multi-token responses, nearly reaching IA2 + SFT, but falls short for single-token responses. This highlights the consistency of IA2-based high performance.

| Dataset | \(N\) | ICL acc ↑ | ICL ece ↓ | SFT only acc ↑ | SFT only ece ↓ | SFT (soft labels) acc ↑ | SFT (soft labels) ece ↓ | IA2 only acc ↑ | IA2 only ece ↓ | IA2 → SFT acc ↑ | IA2 → SFT ece ↓ | IA2 + SFT acc ↑ | IA2 + SFT ece ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GSM8K | 4 | 34.6 (00.9) | – | 28.2 (04.5) | – | 36.1 (03.5) | – | 29.3 (05.0) | – | 30.3 (03.2) | – | 37.0 (01.9) | – |
| SST2 | 8 | 77.0 (02.2) | 0.10 (0.02) | 52.3 (04.3) | 0.38 (0.14) | 55.9 (08.8) | 0.16 (0.06) | 64.9 (15.3) | 0.30 (0.05) | 64.7 (19.3) | 0.33 (0.19) | 66.9 (18.4) | 0.28 (0.21) |

Table 4: Comparison with Knowledge Distillation (KD) baseline (SFT on soft labels). Although KD baseline performs better than standard SFT in the multi-token case, IA2 based training provides consistently high performance for single/multi token cases.

The dash marks unreported ECE entries.

**Why IA2 based training does not consistently beat ICL?** In some multi-token settings, such as Qwen on GSM8K and HMathA, ICL exceeds every training method, including IA2-based training. We suspect Qwen’s mid-training on STEM data makes ICL extremely sample-efficient. Surprisingly, GSM8K ICL at \(N=2\) exceeds its performance at \(N=4,8\).

Table 5 shows performance as shot count increases. The gap between ICL and IA2 → SFT closes quickly. The key finding remains consistent: IA2 → SFT exceeds SFT only, as the final column shows.

| Dataset | \(N\) | ICL | SFT only | IA2 only | IA2 → SFT | Gap wrt ICL | Gap wrt SFT only |
|---|---|---|---|---|---|---|---|
| GSM8K | 2 | 81.2 (00.4) | 65.2 (01.0) | 70.7 (01.8) | 74.6 (02.4) | -06.6 | +09.4 |
| | 4 | 76.4 (01.1) | 70.9 (02.8) | 77.4 (02.8) | 73.6 (03.7) | -02.8 | +02.7 |
| | 8 | 77.2 (00.9) | 73.5 (01.6) | 78.8 (02.5) | 76.0 (02.0) | -01.2 | +02.5 |
| HMathA | 2 | 58.6 (02.0) | 47.8 (04.0) | 33.9 (06.4) | 48.7 (07.0) | -09.9 | +00.9 |
| | 4 | 60.4 (02.1) | 50.4 (01.8) | 47.8 (06.0) | 55.3 (02.1) | -05.1 | +04.9 |
| | 8 | 62.1 (01.0) | 52.0 (00.8) | 57.3 (03.7) | 59.4 (00.8) | -02.7 | +07.4 |

Table 5: ICL is more sample efficient than training methods for Qwen models in math datasets. But IA2 \(\rightarrow\) SFT quickly closes the gap in performance with increasing number of shots, and consistently beats SFT only training.

Here, “wrt” means with respect to the named comparison method.

**Other fine-tuning methods:** LoRA is not the only option. Alternatives include full fine tuning, which is full-rank; prompt/prefix tuning (Lester et al., 2021; Li and Liang, 2021); and activation scaling, (IA)3 (Liu et al., 2022).

Our pipeline improves rather than competes with these methods because it supplies better training signals. To demonstrate this, we repeat experiments on several datasets with the popular PEFT method (IA)3. The methodology is unchanged from §5 except that (IA)3 replaces LoRA (§B).

Table 6 gives one setting and shows trends similar to LoRA; further results are in §C. We leave exploration of other SFT methods to future work. As a side note, IA2 also marks a major improvement in method naming compared to (IA)3.

| Source | Eval | ICL acc ↑ | ICL ece ↓ | SFT only acc ↑ | SFT only ece ↓ | IA2 only acc ↑ | IA2 only ece ↓ | IA2 → SFT acc ↑ | IA2 → SFT ece ↓ |
|---|---|---|---|---|---|---|---|---|---|
| SciQr | QASCr∗ | 35.6 (01.4) | 0.06 (0.01) | 32.8 (03.4) | 0.20 (0.08) | 34.2 (02.7) | 0.08 (0.01) | 48.6 (05.5) | 0.25 (0.11) |
| | SciQr | 60.0 (01.8) | 0.07 (0.01) | 64.5 (05.0) | 0.12 (0.05) | 63.0 (04.6) | 0.10 (0.04) | 70.6 (04.6) | 0.08 (0.02) |
| SST2 | FinS∗ | 56.6 (01.6) | 0.17 (0.01) | 58.2 (10.7) | 0.25 (0.03) | 61.9 (09.5) | 0.18 (0.06) | 59.8 (07.8) | 0.18 (0.03) |
| | PoemS∗ | 52.3 (03.0) | 0.19 (0.02) | 48.1 (03.2) | 0.44 (0.11) | 48.1 (03.4) | 0.11 (0.04) | 59.8 (13.4) | 0.33 (0.13) |
| | SST2 | 60.3 (02.8) | 0.19 (0.02) | 52.4 (02.8) | 0.35 (0.05) | 54.2 (04.2) | 0.15 (0.06) | 62.6 (09.1) | 0.28 (0.11) |

Table 6:
Performance report for \(N=4\) on Llama-3.2 models trained using (IA)3. IA2 → SFT achieves significant gains over SFT only training, exceeding even ICL accuracy in all cases.

**Computational Overhead:** Before training, IA2 → SFT requires collecting activations through ICL inference on the dataset. This is a one-time cost. It greatly benefits model capabilities, while subsequent inference costs the same as for SFT-only models.

**Potential improvements:** Resource constraints fixed several design choices that could be explored. The most important are ablations—tests of alternative choices—for LoRA parameters such as rank and target modules, and selective IA2, which aligns only selected layers.

Extreme alignment can hurt performance, and Todd et al. (2023) found some layers more sensitive to ICL than others. Selective layer alignment may therefore be more effective. Prompt optimization before activation collection may also help because demonstration order affects ICL performance (Lu et al., 2022; Zhao et al., 2021). Finally, our fixed learning rates could be replaced by finer-grained control of the two signals. We leave these explorations to future work.

**Claims and evidence**
- Training on ICL responses reduces SFT-only and IA2 → SFT performance compared with ground-truth training: Tables 2 and 3, respectively:
  - GSM8K: SFT only 70.9 versus 66.4; IA2 → SFT 73.6 versus 67.2.
  - GSM8Ks∗: 64.5 versus 59.8; 68.8 versus 61.6.
  - HMathA: 50.4 versus 49.0; 55.3 versus 54.0.
  - SciQ: 35.0 versus 27.2; 40.8 versus 32.8.
- IA2 + SFT extracts more performance from ICL responses than SFT only: Table 3, GSM8K, 77.0 versus 66.4; GSM8Ks∗, 69.0 versus 59.8; HMathA, 53.6 versus 49.0; SciQ, 34.5 versus 27.2.
- Soft-label SFT nearly reaches IA2 + SFT on the multi-token task but falls short on the single-token task: Table 4, GSM8K at \(N=4\), accuracy 36.1 versus 37.0, with standard SFT at 28.2; SST2 at \(N=8\), accuracy 55.9 versus 66.9, with standard SFT at 52.3.
- GSM8K ICL is stronger at \(N=2\) than at \(N=4,8\): Table 5, ICL accuracy 81.2, 76.4, and 77.2.
- IA2 → SFT closes the ICL gap as shots increase: Table 5, GSM8K gaps -06.6, -02.8, -01.2; HMathA gaps -09.9, -05.1, -02.7 at \(N=2,4,8\).
- IA2 → SFT consistently exceeds SFT only in these math comparisons: Table 5, final-column gaps +09.4, +02.7, +02.5 for GSM8K and +00.9, +04.9, +07.4 for HMathA.
- The gains also occur with (IA)3: Table 6, SFT-only versus IA2 → SFT accuracy is 32.8 versus 48.6 for SciQr → QASCr∗; 64.5 versus 70.6 for SciQr → SciQr; 58.2 versus 59.8 for SST2 → FinS∗; 48.1 versus 59.8 for SST2 → PoemS∗; and 52.4 versus 62.6 for SST2 → SST2.
- IA2 → SFT exceeds ICL accuracy in every Table 6 row: ICL versus IA2 → SFT accuracy is 35.6 versus 48.6, 60.0 versus 70.6, 56.6 versus 59.8, 52.3 versus 59.8, and 60.3 versus 62.6, respectively.
- Activation collection adds a one-time cost, while inference costs the same as SFT only: the “Computational Overhead” account; no numerical cost comparison is reported.

## 7 Related Work

*The paper relates IA2 to comparisons of ICL and SFT, context distillation, and activation steering.*

**ICL vs SFT:** Prior studies compare ICL and SFT performance (Mosbach et al., 2023; Duan et al., 2024), but the verdict remains unclear. We find SFT worse than ICL. Other work examines their functional similarities (Dai et al., 2022) and differences (Wang et al., 2023), or studies miscalibration and its remedies (Zhang et al., 2024; Li et al., 2025; Li et al., 2024; Xiao et al., 2025).

Sia et al. (2024) use context masking to locate “where” translation occurs during ICL. Like us, they use activations as a functional-behavior signal. Recent work also shows SFT’s susceptibility to memorization, overfitting, and catastrophic forgetting—problems associated with SFT that our pipeline aims to address (Chu et al., 2025; Shenfeld et al., 2025).

**Context Distillation:** Many methods improve LLM adaptation by compressing explicit context signals. They use:
- Intermediate contemplation tokens for compression (Cheng and Van Durme, 2024; Jiang et al., 2025).
- Adapters generated on the fly to internalize context (Chen et al., 2025; Charakorn et al., 2025).
- Related text to improve distillation (Zhu et al., 2025; Choi et al., 2025).
- Direct compression of context into weights (Shen et al., 2025; Deng et al., 2024; Yu et al., 2024; Snell et al., 2022; Chen et al., 2024b; Shin et al., 2024).

These approaches distill context effects on outputs through response-token signals. Our method complements them and improves distillation through the model’s own ICL processing signals, an idea supported by Aguilar et al. (2020), Jin et al. (2024), and Yang et al. (2024).

**Activation Steering:** Another line of work extracts steering vectors or weights and applies them at selected internal locations to influence outputs (Postmus and Abreu, 2024; Stolfo et al., 2025; Caccia et al., 2025; Fleshman and Van Durme, 2024; Arditi et al., 2024).

Like IA2, these methods intervene in activation space. Unlike their targeted approach and small application domain, IA2 aims for a large behavioral change that avoids potential brittleness and reliability problems (Queiroz Da Silva et al., 2025).

**Claims and evidence**
- This study finds SFT worse than ICL: Figure 1’s aggregated comparisons; Table 2, GSM8K accuracy 70.9 for SFT only versus 76.4 for ICL, and HMathA 50.4 versus 60.4.
- The statements about prior approaches are supported by the cited studies; this section reports no separate numerical experiment for them.

## 8 Conclusion and Future Work

*IA2 uses ICL-like internal processing to improve SFT, while further effects and RL-tuned models remain to be studied.*

We used activations to probe LLM functional behavior. ICL and SFT adapt through different mechanisms, and ICL often encodes richer, more generalizable patterns.

This motivated IA2, an SFT priming method that enables ICL-like internal behavior. The simple step moves models into a more adaptable weight subspace inaccessible to SFT alone. Extensive experiments show consistent improvements in SFT accuracy and calibration.

Future work will examine diversity and catastrophic forgetting to assess IA2’s role in LLM post-training. We also plan to study IA2 on post-trained, RL-tuned models.

**Claims and evidence**
- ICL and SFT use different adaptation mechanisms: Figure 2, the “SFT only” activation comparisons described in §3.
- IA2 identifies a weight subspace unavailable to SFT alone: Figure 4; §5.2 reports around \(39\%\) average overlap between IA2 only and IA2 → SFT, with SFT-only updates almost completely orthogonal.
- IA2 improves SFT accuracy and calibration: Table 1, SST2 → SST2, accuracy 65.2 versus 90.4 and ece 0.22 versus 0.06; Table 2, HMathA → HMathA, accuracy 50.4 versus 55.3; Figures 1 and 3 show the aggregate and activation-similarity comparisons.

## Reproducibility Statement

*The provided code, experimental details, and repeatable seeds are intended to support reproduction.*

We provide code on github for data download and processing, training, evaluation, analysis, plotting, and other steps. Section 5, the appendix, and a detailed README file should provide enough information to reproduce the results fully. We used repeatable random seeds.

## Acknowledgments

*The work received the listed funding and feedback.*

This work is supported by ONR (N00014-24-1-2089), an Amazon Research Award, and a grant from Open Philanthropy. We thank Andrea Wynn for their constructive feedback on earlier versions of this document.
