# Persistent Sparse Autoencoders: Learning Feature-Specific Timescales in Language Model Representations

Haoyan Luo

Affiliation: University of Cambridge

  
Mateo Espinosa Zarlenga

Affiliation: University of Oxford

  
Mateja Jamnik

Affiliation: University of Cambridge

###### Abstract

Sparse autoencoders (SAEs) decompose language model activations into sparse features, yet these models traditionally encode each token independently, failing to expose information that persists across a sequence. We first show that temporal persistence can naturally emerge in standard SAE features: after a feature activates, the hidden state remains aligned with its direction, and past activations help reconstruct later hidden states. How long this lasts varies widely across features.
We therefore introduce Persistent Sparse Autoencoders (Persistent SAEs), an extension of standard SAEs that learns a persistence coefficient for each feature, allowing the model to learn feature-specific timescales from reconstruction alone.
Our experiments show that Persistent SAEs retain competitive reconstruction quality while learning a spectrum of timescales: short-timescale (fast) features stay locally interpretable, whereas long-timescale (slow) features accumulate information that identifies the current context.
Moreover, we show in a prompt-injection monitoring case study that slow features preserve injection-related signals and remain causally effective over long contexts. These results suggest that Persistent SAEs offer new opportunities for interpreting and monitoring language models via persistent sparse features.

## 1 Introduction

Understanding the output of a large language model (LLM) requires identifying what information is represented in its hidden states and how that information evolves throughout the sequence [1, 2]. Sparse autoencoders (SAEs) [3, 4], methods that decompose an LLM’s hidden states into a set of sparse features, have become a popular tool for approaching this problem. Specifically, when features are aligned with human-interpretable concepts, an SAE’s decomposition enables easy inspection of concepts learnt by an LLM, and provides a mechanism for model control through inspection of and intervention on these features [5, 6, 7].

Although causal transformer hidden states integrate the preceding context, standard SAEs encode each token independently [3, 8, 9]. This atemporal formulation favours token-specific features but obscures the contextual information a model carries across a sequence, such as the current topic, style, or task state [10].
To alleviate this constraint, recent methods have incorporated temporal structure into SAEs through predefined partitions and decompositions of temporal roles [10, 11].
However, these approaches require specifying, a priori, which features should carry temporal information, leading to experimenter bias in the features the SAE learns.
It therefore remains unclear whether each feature’s timescale can be learned directly from the data, rather than being predefined by the model architecture.

This paper explores this question starting from the observation that temporal persistence can naturally emerge in standard SAE features. As shown in Figure 1, we observe that, after a feature activates, later hidden states remain more strongly aligned with its decoder direction, and past activations contain information that helps reconstruct subsequent hidden states.
The persistence decays at substantially different rates across features, suggesting that different features operate at different timescales.
Yet, standard SAEs encode each position independently and therefore do not explicitly represent this persistence.
To fill this gap, we introduce Persistent Sparse Autoencoders, which learn a persistence coefficient $\lambda_{j}$ for every feature. Each feature keeps a persistent state that decays at its persistent coefficient and is updated by a token-wise sparse activation.
If carrying a feature’s state forward helps reconstruct later hidden states, there is an incentive to learn a higher $\lambda_{j}$ during optimisation; if it does not, the objective favours a smaller $\lambda_{j}$ and faster decay.

Our experiments show that Persistent SAEs retain competitive reconstruction quality while learning a broad range of feature timescales unsupervisedly.
Most are low- $\lambda_{j}$ fast features that behave as locally interpretable detectors, as in a standard SAE. High- $\lambda_{j}$ slow features accumulate information that retains contextual information across the sequence.
We show how, in a prompt-injection monitoring case study [12], slow features can serve as powerful monitoring signals for malicious prompt injections over longer time scales.
These results show that SAE reconstruction can organise sparse features by timescale, separating token-wise activations from a compact persistent state.
Therefore, our main contributions can be summarised as follows:

•

We show that SAE features exhibit temporal persistence: after a feature is activated, its decoder direction remains elevated, and past activations can help predict subsequent states.

•

We introduce a new SAE variant, termed Persistent Sparse Autoencoders, that explicitly models this temporal persistence by learning one coefficient per sparse feature through reconstruction alone. Persistent SAEs retain competitive reconstruction quality and discover a broad spectrum of timescales without a predefined partition of temporal roles.

•

We show that short-timescale (fast) features act as locally interpretable detectors, while long-timescale (slow) features accumulate information that retains contextual information in a sequence. Through a prompt-injection monitoring case study, we demonstrate the potential of slow features to monitor and steer models over long contexts, opening new opportunities to interpret and control LLMs via explicitly persistent features.

## 2 Background and Related Work

Sparse autoencoders. 
Given an LLM’s hidden state $\mathbf{x}^{(t)}\in\mathbb{R}^{d}$ at token $t$ , a standard SAE [3] encodes $\mathbf{x}^{(t)}$ as non-negative sparse activations $\mathbf{f}^{(t)}\in\mathbb{R}_{\geq 0}^{m}$ from which it can be reconstructed as

| | $\hat{\mathbf{x}}^{(t)}_{\mathrm{base}}=\mathbf{W}_{\mathrm{dec}}\mathbf{f}^{(t)}+\mathbf{b}_{\mathrm{dec}}.$
| | (1)

Under this representation, the $j$ -th feature is represented by the activation $\mathbf{f}^{(t)}_{j}$ and its corresponding normalised decoder direction $\mathbf{d}_{j}=\mathbf{W}_{\mathrm{dec}}[:,j]/\lVert\mathbf{W}_{\mathrm{dec}}[:,j]\rVert_{2}$ (see Appendix A for a notation summary). Intuitively, we think of feature $j$ as active when $\mathbf{f}^{(t)}_{j}>0$ .
Because $\mathbf{f}^{(t)}$ is sparse by construction, each reconstruction depends on only a small set of features, making the contribution of individual features easier to inspect [3, 13].
Subsequent SAEs improve how representational capacity is allocated and organised: BatchTopK [8] replaces the fixed per-token TopK constraint with a batch-level sparsity budget, allowing the number of active features to adapt across tokens, while Matryoshka SAEs [9] jointly train nested dictionaries of increasing size so that smaller subsets retain higher-level features as larger dictionaries add more specific ones.

Temporal structure in LLM representations. 
Although autoregressive LLMs build $\mathbf{x}^{(t)}$ from the preceding context, standard SAEs and most SAE variants compute features $\mathbf{f}^{(t)}$ independently of the features observed in preceding tokens. SAE features, therefore, do not explicitly expose information carried across tokens. Yet, traditional slow feature analysis shows that slowly varying latent factors reveal higher-level features in neural networks [14, 15].
Prior studies likewise show that language models track entities and that their hidden states encode game-board, spatial, temporal, and propositional states [16, 17, 18, 19]. These findings motivate mechanisms that explicitly encode temporal persistence in SAE features.

Temporal extensions of SAEs. 
Recent works, specifically temporal formulations of SAE [10, 11], have made progress toward capturing feature persistence. Temporal SAEs [10] pre-assign a high-level feature subset and use a contrastive objective to align their activations across nearby tokens. Lubana et al. [11] instead use an attention-based predictor to separate context-predictable information from a sparse novel residual representation. Both prescribe temporal structure via a fixed partition. However, neither allows each sparse feature to learn its own temporal persistence, complicating post-hoc analysis of the expected timescale of different features. Persistent SAEs fill this gap by assigning each feature a persistent state and a learned persistence coefficient that can be easily analysed.

Figure 1:
Standard SAE features exhibit a temporal structure that is not explicitly modelled.
(a) Natural Pile sequences preserve local and global continuity, whereas the shuffled control sequences formed by concatenating sentences from different documents preserve local continuity while breaking global continuity.
(b) Empirical feature persistence $R_{j}(\tau)$ decays more slowly in natural sequences than in shuffled sequences.
(c) Past feature activations improve held-out reconstruction of later hidden states in natural sequences, but not in shuffled ones.
(d) At lag $\tau=1$ , fitted context-correction weights $w_{j}$ are predominantly positive for natural sequences but centred near zero for shuffled ones.

## 3 Does Temporal Structure Exist at the Feature Level?

We first test whether standard SAEs inherit temporal structure from the hidden states they reconstruct. Using a shared setup, we ask whether the signal along $\mathbf{d}_{j}$ remains after feature $j$ activates, and whether past activations improve the reconstruction of subsequent tokens’ hidden states.

Setup. 
We analyse residual-stream hidden states $\{\mathbf{x}^{(t)}\}_{t=1}^{T}$ from Pythia-160M-deduped (layer 8) [20] and Gemma-2-2B (layer 12) [21] on Pile documents [22]. We train BatchTopK [8] and Matryoshka [9] SAEs, neither of which has a temporal prior. We describe training configurations in Appendix B.

Natural Pile documents are continuous at two scales: tokens form coherent sentences (local continuity), and the sentences collectively develop a coherent document-level context (global continuity). Our shuffled control isolates the contribution of global continuity by concatenating intact sentences drawn from different Pile documents. Natural documents and the shuffled control therefore both preserve local continuity, but only natural documents preserve global continuity. Comparing them lets us test whether feature persistence reflects connections across the document beyond the local coherence within individual sentences (Figure 1 (a)). Appendix C gives the construction details of the dataset.

Sparse Features Temporally Persist in the Hidden States. 
We first test whether the signal aligned with feature $j$ persists after it activates. To achieve this, for each token position $t$ , we define $p^{(t)}_{j}=\mathbf{d}_{j}^{\top}\mathbf{x}^{(t)}$ as the projection of the hidden state $\mathbf{x}^{(t)}$ for the $t$ -th token onto the direction representing feature $j$ .
Intuitively, a larger $p^{(t)}_{j}$ indicates a stronger alignment between the hidden state and feature direction $\mathbf{d}_{j}$
Conditioning on feature $j$ being active at token $t$ , we then define its empirical persistence at lag $\tau$ , where $\tau$ is the number of subsequent tokens, as

| | $R_{j}(\tau):=\big({\mathbb{E}\!\left[p^{(t+\tau)}_{j}\mid\mathbf{f}^{(t)}_{j}>0\right]-\mathbb{E}\!\left[p^{(t)}_{j}\right]}\Big)\Big/\Big({\mathbb{E}\!\left[p^{(t)}_{j}\mid\mathbf{f}^{(t)}_{j}>0\right]-\mathbb{E}\!\left[p^{(t)}_{j}\right]}\Big),$
| | (2)

where $\mathbb{E}[p^{(t)}_{j}]$ is the feature’s average projection across all evaluated tokens.
Thus here: larger $R_{j}(\tau)$ indicates greater persistence: $R_{j}(\tau)=1$ means that the activated-token elevation is fully retained, while $R_{j}(\tau)=0$ means that it has decayed to baseline activation after $\tau$ tokens. Appendix D provides the full evaluation procedure and feature-selection criteria for this measure.

In Figure 1 (b), we observe that across both models and SAE families, empirical feature persistence decays more slowly in natural documents than in the shuffled control. This suggests that standard SAE features inherit the persistence associated with global continuity, even when the SAE does not explicitly encode it.
These results suggest that feature-aligned signals persist with document-level context, not only at the tokens that activate the feature. Since standard SAE features already exhibit this temporal structure, it may be possible to model it explicitly and test whether past activations help reconstruct the hidden states of subsequent tokens.

Past Activations Can Help Reconstruction. 
We further test whether $\mathbf{f}^{(t-\tau)}$ can reduce an SAE’s reconstruction error for token $t$ .
We test this using a restricted context correction experiment: starting from the reconstruction in Equation 1, we add a contribution from the activations $\tau$ tokens earlier:

| | $\hat{\mathbf{x}}^{(t)}_{\mathrm{ctx}}=\hat{\mathbf{x}}^{(t)}_{\mathrm{base}}+\mathbf{W}_{\mathrm{dec}}\left(\mathbf{w}_{\tau}\odot\mathbf{f}^{(t-\tau)}\right).$
| | (3)

The weights $\mathbf{w}_{\tau}\in\mathbb{R}^{m}$ specify how much each past feature activation affects the reconstruction only along its corresponding feature direction. Therefore, the weight vector $\mathbf{w}_{\tau}$ directly tests whether a feature’s past activation predicts a later contribution along the same direction. In this setting, for a given lag, we fit $\mathbf{w}_{\tau}$ for 10 epochs with an $L_{2}$ penalty and measure the fraction of the standard SAE’s reconstruction error that the context correction recovers on held-out data:

| | $R^{2}_{\mathrm{improve}}=1-\Big({\sum_{t}\lVert\mathbf{x}^{(t)}-\hat{\mathbf{x}}^{(t)}_{\mathrm{ctx}}\rVert_{2}^{2}}\Big)\Big/\Big({\sum_{t}\lVert\mathbf{x}^{(t)}-\hat{\mathbf{x}}^{(t)}_{\mathrm{base}}\rVert_{2}^{2}}\Big).$
| | (4)

For the control, we refit the context correction weights using past activations from the shuffled corpus while keeping the same natural-document hidden states as targets. This tests whether the reconstruction gain requires the actual document history. We describe the training details in Section D.1.

In Figure 1 (c), we observe that past activations in natural documents recover a fraction of the baseline reconstruction error, with gains decreasing as the lag grows. However, this effect is not observed when using the shuffled control activations. Correspondingly, Figure 1 (d) shows that the fitted context correction weights at $\tau=1$ are mostly positive for natural documents, but centred near zero for the shuffled control. Together, these results imply that standard SAEs leave reconstruction-relevant temporal structure unused within their features.

We note that Lubana et al. [11] show that hidden-state correlations vary across a sequence, and that past hidden states explain much of the current hidden state. In contrast, our experiments above explore a narrower, feature-level question: whether individual SAE features capture this temporal structure. We find that their persistence helps reconstruction and varies widely across features, motivating the explicit learning of temporal persistence for each feature in the next section.

## 4 Persistent Sparse Autoencoders

In this section, we introduce Persistent SAEs, which make the temporal structure observed in the previous section explicit by separating a token-wise sparse activation from a persistent state (Figure 2).

Figure 2:
Persistent Sparse Autoencoders. Each feature integrates its token-wise sparse activation $\mathbf{a}^{(t)}$ into a persistent state $\mathbf{h}^{(t)}$ with its learned persistence coefficient $\lambda_{j}$ .

As in standard BatchTopK encoders (see Section B.2 for details), a Persistent SAE encoder produces token-wise sparse activation $\mathbf{a}^{(t)}\in\mathbb{R}_{\geq 0}^{m}$ , where $a^{(t)}_{j}>0$ means feature $j$ finds new evidence at token $t$ . In contrast to standard SAEs, however, each feature integrates this activation into a persistent state $\mathbf{h}^{(t)}\in\mathbb{R}_{\geq 0}^{m}$ using a learned persistence coefficient $\lambda_{j}\in[0,1]$ :

| | $h^{(t)}_{j}=\lambda_{j}h^{(t-1)}_{j}+(1-\lambda_{j})a^{(t)}_{j}.$
| | (5)

Intuitively, a small $\lambda_{j}$ yields a short-timescale, fast feature that stays close to the current token, and a large $\lambda_{j}$ gives a long-timescale, slow feature that integrates over many tokens across the sequence.
The factor $(1-\lambda_{j})$ prevents activations from accumulating excessively, keeping the state magnitude bounded.
During training, we initialise $\mathbf{h}^{(0)}=\mathbf{0}$ at the start of every input sequence. We parameterise $\lambda_{j}=\lambda_{\max}\sigma(l_{j})$ , where $l_{j}$ is an unconstrained learned parameter and $\lambda_{\max}=0.999$ keeps every feature’s timescale finite. Section B.3 provides further details about the persistence parametrisation.

The decoder reconstructs the hidden state from this persistent state:

| | $\hat{\mathbf{x}}^{(t)}=\mathbf{W}_{\mathrm{dec}}\mathbf{h}^{(t)}+\mathbf{b}_{\mathrm{dec}}.$
| | (6)

The recurrence is diagonal in feature space: feature $j$ carries only its own activation forward, preserving feature-level attribution through its corresponding decoder direction. Sparsity is imposed on the token-wise activation $\mathbf{a}^{(t)}$ , so the accumulated state $\mathbf{h}^{(t)}$ integrates sparse updates but is itself not necessarily sparse.
During reconstruction training, the objective favours a larger $\lambda_{j}$ when carrying feature $j$ forward helps reconstruct later hidden states. The model therefore discovers each feature’s timescale without the predefined feature partitions or auxiliary temporal objectives that previous work requires.

## 5 Experiments

In this section, we evaluate the quality of the representations learnt by Persistent SAEs and explore what appears at each learned timescale. Specifically, we first test whether adding a persistent state preserves reconstruction quality and interpretable token-wise activations (Sections 5.1 and 5.2). We then ask what information the accumulated states of slow features capture (Section 5.3). Finally, as a demonstration of the practical uses of our proposed approach, we show how persistent states can improve long-context monitoring in a prompt injection case study (Section 5.4).

Models and hyperparameters. 
Throughout all experiments, we use the same models and dataset as in Section 3. We compare with BatchTopK SAE [8] and Matryoshka SAE [9] as strong standard baselines, and Temporal SAE [10] as the closest temporally-structured baseline.
All SAEs are trained with $m=16{,}384$ features and a BatchTopK budget of $k=20$ . Matryoshka and Temporal SAEs both use the same predefined 20%/80% feature partition. Appendix B gives the full training configuration.

### 5.1 Reconstruction Quality and Learned Timescales

We first examine how the learned persistence coefficients are distributed across features. Figure 3 (a) shows that they span a broad range. Most features are short-timescale, with a small tail of long-timescale, highly persistent features.
Based on this distribution, we define slow features as the highest $1\%$ by $\lambda_{j}$ and fast features as the lowest $50\%$ for subsequent analyses.

We then study the quality of our method’s representations by looking at: (1) the fraction of variance explained (FVE); (2) the cosine similarity between the hidden state and its reconstruction; (3) the fraction of active features; and (4) the degree to which features change smoothly over time, using the temporal smoothness metric from Bhalla et al. [10]. Table 1 shows that Persistent SAEs remain competitive with the strongest baselines in reconstruction quality. Across both models, slow features change more smoothly over time than fast features and baselines SAEs. This suggests that Persistent SAEs preserve fast features for token-specific detail while concentrating temporal continuity in a small slow subset, without preassigning temporal roles.

Moreover, we test whether the learned coefficients correspond to the empirical persistence observed in Section 3. We group features into quintiles by $\lambda_{j}$ and measure their median $R_{j}(\tau)$ across lags. Figure 3 (b) shows a consistent ordering across both models: higher- $\lambda_{j}$ groups decay more slowly and remain more persistent over longer lags. By contrast, the Temporal SAE’s predefined high-level and low-level groups are not clearly separated by empirical persistence. This alignment shows that the coefficients learnt through reconstruction capture meaningful differences in feature timescales.

Table 1:
SAE representation quality. High/Low denote predefined splits for Matryoshka and Temporal SAEs. For Persistent SAEs, the largest 1% and smallest 50% coefficients are used for High/Low partitions.
Best results are bold and second-best results are underlined. Persistent SAEs remain competitive while separating smooth, slow features from locally interpretable fast features.

|
| | FVE ( $\uparrow$ )
|

Cosine
Sim ( $\uparrow$ )

|

Fraction
Alive ( $\uparrow$ )

| Smoothness ( $\downarrow$ ) | Autointerp ( $\uparrow$ )

|
| | Full | High | Low | Full | High | Low

|

Pythia

| BatchTopK | $\mathbf{0.94{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.95{\scriptscriptstyle\pm.01}}$ | $\underline{0.91{\scriptscriptstyle\pm.01}}$ | $0.15{\scriptscriptstyle\pm.01}$ | – | – | $\mathbf{0.91{\scriptscriptstyle\pm.02}}$ | – | –

| Matryoshka | $0.92{\scriptscriptstyle\pm.01}$ | $\underline{0.94{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.96{\scriptscriptstyle\pm.01}}$ | $0.15{\scriptscriptstyle\pm.01}$ | $0.25{\scriptscriptstyle\pm.01}$ | $\underline{0.12{\scriptscriptstyle\pm.01}}$ | $\underline{0.87{\scriptscriptstyle\pm.02}}$ | – | –

| Temporal | $\underline{0.93{\scriptscriptstyle\pm.02}}$ | $\mathbf{0.95{\scriptscriptstyle\pm.01}}$ | $0.79{\scriptscriptstyle\pm.01}$ | $\underline{0.13{\scriptscriptstyle\pm.01}}$ | $\underline{0.18{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.11{\scriptscriptstyle\pm.01}}$ | $\underline{0.87{\scriptscriptstyle\pm.02}}$ | $\mathbf{0.87{\scriptscriptstyle\pm.01}}$ | $\underline{0.89{\scriptscriptstyle\pm.02}}$

| Persistent (Ours) | $\underline{0.93{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.95{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.96{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.10{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.05{\scriptscriptstyle\pm.01}}$ | $0.20{\scriptscriptstyle\pm.01}$ | $0.86{\scriptscriptstyle\pm.01}$ | $\underline{0.84{\scriptscriptstyle\pm.02}}$ | $\mathbf{0.93{\scriptscriptstyle\pm.01}}$

|

Gemma

| BatchTopK | $\mathbf{0.72{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.79{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.92{\scriptscriptstyle\pm.01}}$ | $0.16{\scriptscriptstyle\pm.01}$ | – | – | $\underline{0.87{\scriptscriptstyle\pm.03}}$ | – | –

| Matryoshka | $\underline{0.71{\scriptscriptstyle\pm.01}}$ | $\underline{0.77{\scriptscriptstyle\pm.01}}$ | $0.87{\scriptscriptstyle\pm.01}$ | $0.17{\scriptscriptstyle\pm.01}$ | $0.21{\scriptscriptstyle\pm.01}$ | $\mathbf{0.14{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.89{\scriptscriptstyle\pm.02}}$ | – | –

| Temporal | $0.62{\scriptscriptstyle\pm.02}$ | $0.75{\scriptscriptstyle\pm.01}$ | $0.75{\scriptscriptstyle\pm.01}$ | $\underline{0.14{\scriptscriptstyle\pm.01}}$ | $\underline{0.12{\scriptscriptstyle\pm.01}}$ | $\underline{0.16{\scriptscriptstyle\pm.01}}$ | $0.85{\scriptscriptstyle\pm.02}$ | $\mathbf{0.85{\scriptscriptstyle\pm.02}}$ | $\underline{0.86{\scriptscriptstyle\pm.02}}$

| Persistent (Ours) | $0.66{\scriptscriptstyle\pm.01}$ | $0.75{\scriptscriptstyle\pm.01}$ | $\underline{0.91{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.11{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.04{\scriptscriptstyle\pm.01}}$ | $0.20{\scriptscriptstyle\pm.01}$ | $0.85{\scriptscriptstyle\pm.01}$ | $\underline{0.79{\scriptscriptstyle\pm.01}}$ | $\mathbf{0.89{\scriptscriptstyle\pm.01}}$

Figure 3:
Learned persistence coefficients align with empirical persistence. (a) Persistence coefficients $\lambda_{j}$ span a broad range, with many short-timescale features and fewer long-timescale features. (b) We divide Persistent SAE features into quintiles by their learned $\lambda_{j}$ . Higher- $\lambda_{j}$ quintiles retain greater median empirical persistence $R_{j}(\tau)$ across different lags.

### 5.2 Short-Timescale Features Are Locally Interpretable

We next ask how learned timescales relate to feature interpretability. We use the automated interpretability pipeline of Bills et al. [23], in which a judge model explains a feature from its highest-activation contexts then scores how well that explanation predicts activation. For the Persistent SAE, we evaluate the token-wise sparse activation $\mathbf{a}^{(t)}$ , asking whether it remains understandable before it is used to update the persistent state (see Appendix E for details on autointerpretation).

Our experiments, summarised in Table 1 (right), suggest that, across both models, fast features are easier to explain locally than slow features. Thus, adding recurrence preserves locally interpretable activations for fast features, while the interpretability score changes systematically with the learned timescale.
More importantly, we observe that fast features match or exceed the best baseline score on both models.
Qualitative examples (Tables 4 and 5 in Appendix E) suggest that fast features often respond to compact lexical or grammatical patterns, such as letters or substrings.
Their activation is concentrated at the current token, making these features easy to characterise from short local contexts.
Slow features more often reflect broader subjects or contextual conditions whose activation is distributed across the sequence.
This exposes a mismatch between persistent representations and a token-specific feature evaluation:
distinct features may receive similar descriptions [24], and explanations may not generalise to related inputs [25]. Hence, in Section 5.3 we study slow features jointly, asking whether their state forms a coherent context representation.

Where Features Receive New Evidence. 
The preceding analysis asks how well individual feature activations can be explained at the tokens where they occur. Before examining the accumulated states of slow features, we investigate which kinds of tokens provide new activation to features at different timescales? We divide features into five equal-sized $\lambda$ quintiles and group their update positions into six token categories: punctuation, conjunctions, function words, verbs, adjectives or adverbs, and nouns.
This grouping follows the classical distinction between lexical/content words and grammatical/function words [26].
Function words (Func. in Figure 4) include adpositions, determiners, particles, and pronouns.
For each quintile $q$ and token category $c$ , we compute the relative update frequency

| | $U_{q,c}={\Pr\!\left(T^{(t)}=c\mid a^{(t)}_{j}>0,\;j\in Q_{q}\right)}\big/{\Pr\!\left(T^{(t)}=c\right)},$
| | (7)

where $T^{(t)}$ is the category of token $t$ and $Q_{q}$ is the set of features in quintile $q$ . This ratio compares how often features are updated at category $c$ with how often that category occurs in the corpus. A value above one means more updates than expected, while a value below one means fewer.

Figure 4: Where features receive evidence. Each cell shows how often features in a $\lambda$ -quintile are updated at a token category relative to the category corpus-level frequency. Orange and blue indicate frequencies above and below the corpus-level baseline.

Figure 4 shows the clearest difference between the two ends of the timescale distribution. Features in the lowest- $\lambda$ quintile receive more updates at nouns. This aligns with their stronger local interpretability in Table 1, as nouns provide clear lexical evidence at the current token. Features in the highest- $\lambda$ quintile, in contrast, receive updates more often at punctuation and conjunctions.
Recent analyses of LLMs likewise find that punctuation and other function-like tokens can carry substantial contextual information [27], and that explicit structural markers gate the integration of information across sentences
[28].
Thus, short-timescale features receive relatively more evidence from lexical content, whereas long-timescale features are preferentially updated at positions associated with organising and integrating context.
These update locations, however, do not by themselves reveal what information the resulting persistent states encode. Indeed, qualitative autointerpretations of high- $\lambda$ features often describe broader semantic domains (Tables 4 and 5). We therefore turn from where slow features receive activation to what their accumulated states represent.

### 5.3 Long-Timescale States Track Subject Information

Slow features form subject-specific clusters. 
To test whether the accumulated states of slow features retain higher-level contextual information, we visualise how slow features are represented on MMLU [29] examples from five subjects. For each sample, we collect the representations of its final $20$ tokens and visualise them using t-SNE (see Appendix F for details). We select the highest $1\%$ of slow features for visualisation. Figure 5 shows that BatchTopK features heavily mix subjects, whereas the Temporal SAE’s predefined high-level group affords limited separation. In contrast, slow Persistent SAE features form coherent subject regions with a clearer visual separation than even the raw hidden states. Importantly, Persistent SAEs learn this structure without subject labels.

Figure 5:
Token-level MMLU representations, coloured by subject. The Persistent SAE’s high- $\lambda_{j}$ slow features reveal clearer subject structure than baseline SAEs or the raw hidden states.

Temporal Persistence Selects Subject-Informative Features Without Labels. 
We test this observation quantitatively on spliced MMLU sequences, created by concatenating labelled segments from different subjects so that the subject changes at segment boundaries. At each token, a linear probe predicts the current subject using varying fractions of the feature dictionary for training the probe. We compare three rankings of Persistent SAE features: decreasing $\lambda_{j}$ , increasing $\lambda_{j}$ , and random order. We compare against BatchTopK, Matryoshka, and Temporal SAE representations. We also include a supervised ranking as an oracle baseline. For each subject, the supervised ranking favours features that activate more strongly on that subject than on all other subjects. Because this ranking uses the subject labels directly, it serves as a label-informed oracle baseline (see Appendix G for details).

As seen in Figure 6, selecting features by decreasing persistence is nearly as efficient as supervised selection. Across both models, the slowest $1\%$ of features reach high AUC, while low- $\lambda_{j}$ , random, and baseline SAE representations retain substantially less subject information at the same budget. Thus, persistence provides an unsupervised criterion for selecting a compact state from a much wider sparse dictionary that tracks the current subject of a sequence.

Figure 6:
Subject-probing AUC on spliced MMLU sequences. Selecting a fraction of features in decreasing order of $\lambda_{j}$ recovers most of the subject signal, approaching supervised feature selection.

### 5.4 Case Study: Persistent Prompt Injection Monitoring

The preceding experiments suggest that fast features detect interpretable patterns at the current token, while slow features integrate token-wise activations into a compact persistent state. Indirect prompt injection [12] provides a practical test of this hypothesis. In this setting, an agent may encounter a malicious instruction in an email, webpage, or tool response, continue reasoning for hundreds of tokens, and act later [30]. A useful agentic monitor must therefore do more than detect the injected text when it appears, and should preserve a security-relevant signal until the agent acts [31, 32].

Setting. 
We use AgentDojo trajectories from the banking, travel, and workspace domains [12], generated by Llama-3.3-70B-Instruct [33] (see Section C.3 for details). Each trajectory contains a request, an untrusted span with a prompt injection or benign content, and the agent’s subsequent tool calls. To prevent the monitor from relying on malicious actions to identify an attack, we consider only failed attacks, where an injection is present but the agent does not complete the attack’s action.

We use Gemma-2-2B as an external monitor: we replay each trajectory through the monitor model and record the layer-12 residual stream. We train a linear probe at the final token of the untrusted span ( $d=0$ ), where the injection is directly visible, and evaluate the same probe up to $350$ tokens later without retraining (Figure 7 (a)). We compare the raw hidden state and baseline SAE representations with the Persistent SAE state $\mathbf{h}^{(t)}$ restricted to the slow features (highest $1\%$ by $\lambda_{j}$ ) or fast features (lowest $50\%$ ), following the definitions in Section 5.1. This evaluation measures how long each representation preserves the injection signal (see Appendix H for evaluation details).

Figure 7:
Persistent monitoring of prompt injection. (a) A probe trained at the end of an untrusted span is evaluated later in the trajectory. (b) Slow states retain more predictive information (top) and the strongest causal effect on the monitor’s judgement (bottom).

Signal retention: fast features detect local injections while slow states retain signals over long contexts. 
At $d=0$ , the fast-feature state and standard SAE representations provide the strongest signals for detecting an injection (Figure 7 (b), top). As the evaluation moves further beyond the untrusted span, their accuracy falls towards chance. By contrast, the slow-feature state retains a clearer signal at longer distances. Fast and slow features therefore play complementary roles: fast features detect injected text when it appears, while slow states preserve the signal through the subsequent context.

Causal effect: slow-feature directions causally influence the monitor’s judgement. 
Probe accuracy shows that a representation contains injection information, but not whether the corresponding direction influences the monitor’s own judgement. To test this, we truncate each trajectory at a later distance, append a query asking whether it should be flagged for prompt injection, and intervene in the model at the final query token.
Building on prior work that selects SAE features for steering and uses linear classifiers to identify task-relevant sparse subspaces [7, 34], we construct a steering vector $\mathbf{v}_{r}$ for each SAE representation $r$ by weighting its most predictive decoder directions by their probe coefficients. We then apply interventions in both the positive and negative directions. If $s(\mathbf{x})=\ell_{\mathrm{yes}}(\mathbf{x})-\ell_{\mathrm{no}}(\mathbf{x})$ is the monitor’s logit margin, we measure

| | $\displaystyle\Delta_{\mathrm{causal}}(r,d)=\frac{1}{2}\mathbb{E}_{i}\Big[s\big(\mathbf{x}_{i}^{(d)}+\alpha_{i}^{(d)}\mathbf{v}_{r}\big)-s\big(\mathbf{x}_{i}^{(d)}-\alpha_{i}^{(d)}\mathbf{v}_{r}\big)\Big].$
| | (8)

Here, $\alpha_{i}^{(d)}$ scales each intervention to $10\%$ of the hidden-state norm. A positive $\Delta_{\mathrm{causal}}$ means that moving in the probe direction shifts the monitor towards answering yes.

We show that slow features produce the largest and most consistently positive intervention effect at every distance.
By contrast, directions constructed from BatchTopK, Matryoshka, and Temporal SAE features produce near-zero or negative effects and show no comparable alignment over distance.
Thus, slow-feature directions remain aligned with the monitor’s judgement long after the local injection signal has passed.
Together, the probing and intervention results show complementary roles for fast and slow features. Fast features provide strong local detection, while the slow features retain a compact injection-related signal that remains both decodable and aligned with the monitor’s judgement. The Persistent SAE therefore supports both immediate detection and long-context retention and causal influence in a single representation, potentially enabling more robust monitoring and control of LLMs.

## 6 Conclusion and Discussion

[This section has been omitted.]

## Appendix A Notation

Table 2 lists the symbols used in the main text and appendices, in order of first appearance, as a reference for the terminology introduced in Sections 3–5.4. We write vectors and matrices in bold, index token positions with a parenthesised superscript, and index coordinates within a vector with a subscript.

Table 2: Notation used throughout the paper, grouped by the section in which each symbol is introduced. Bold symbols denote vectors and matrices, a parenthesised superscript denotes a token position, and a subscript denotes a coordinate within a vector.

| Symbol | Meaning

| Hidden states and standard SAE (Section 3, 4)

| $\mathbf{x}^{(t)}\in\mathbb{R}^{d}$ | Residual-stream hidden state at token position $t$

| $T$ | Number of tokens in a sequence

| $d$ | Dimensionality of the hidden state

| $m$ | Number of dictionary features (SAE width)

| $\mathbf{W}_{\mathrm{enc}},\mathbf{b}_{\mathrm{enc}}$ | Standard SAE encoder weight matrix and bias

| $\mathbf{W}_{\mathrm{dec}},\mathbf{b}_{\mathrm{dec}}$ | Standard SAE decoder weight matrix and bias

| $\mathbf{d}_{j}$ | Unit-normalised decoder direction for feature $j$

| $\mathbf{f}^{(t)}\in\mathbb{R}_{\geq 0}^{m}$ | Sparse activation of a standard SAE at token $t$ (Background and Related Work)

| $f^{(t)}_{j}$ | Sparse activation of feature $j$ at token $t$

| $k$ | BatchTopK sparsity level (Section B.2)

| $B$ | Batch size used by BatchTopK selection (Section B.2)

| Empirical feature persistence analysis (Section 3, Appendix D)

| $\tau$ | Lag, in tokens

| $p^{(t)}_{j}$ | Decoder-aligned projection $\mathbf{d}_{j}^{\top}\mathbf{x}^{(t)}$

| $R_{j}(\tau)$ | Empirical persistence of $p^{(t)}_{j}$ at lag $\tau$

| $\beta_{\mathrm{nat}},\beta_{\mathrm{shf}}$ | Power-law decay exponents, natural / shuffled

| Context correction experiment (Section 3, Section D.1)

| $\mathbf{w}_{\tau}\in\mathbb{R}^{m}$ | Per-feature context-correction weights at lag $\tau$

| $w_{\tau,j}$ | Context-correction weight for feature $j$ at lag $\tau$ ( $w_{j}$ at $\tau=1$ )

| $\hat{\mathbf{x}}^{(t)}_{\mathrm{base}}$ | Baseline reconstruction from a standard SAE

| $\hat{\mathbf{x}}^{(t)}_{\mathrm{ctx}}$ | Context-corrected reconstruction (Equation 3)

| $R^{2}_{\mathrm{improve}}$ | Held-out reconstruction-error improvement from context correction

| Persistent SAE (Section 4)

| $\mathbf{a}^{(t)}\in\mathbb{R}_{\geq 0}^{m}$ | Token-wise sparse activation of a Persistent SAE (Section B.2)

| $a^{(t)}_{j}$ | Token-wise sparse activation of feature $j$ at token $t$

| $\mathbf{h}^{(t)}\in\mathbb{R}_{\geq 0}^{m}$ | Persistent state integrating token-wise activations over time (Equation 5)

| $h^{(t)}_{j}$ | Persistent state of feature $j$ at token $t$

| $\lambda_{j}\in(0,\lambda_{\max})$ | Persistence coefficient for feature $j$ (Section B.3)

| $\lambda_{\max}$ | Upper bound on the persistence coefficient ( $=0.999$ )

| $l_{j}$ | Unconstrained logit parameterising $\lambda_{j}$

| $\sigma(\cdot)$ | Sigmoid function

## Appendix B SAE Training Details

### B.1 SAE Baseline Training Details

We train four SAE families for comparison: BatchTopK [8],
Matryoshka BatchTopK [9], Temporal SAE [10],
and our Persistent SAE. Each family uses the same dictionary width, BatchTopK activation budget,
optimiser, and token budget. This matched setup isolates the effect of how each SAE
represents temporal information, rather than differences in the amount of training
data or the number of optimisation steps.

Common configuration. 
Every SAE has a dictionary of $m=16{,}384$ features and a BatchTopK activation
budget of $k=20$ per token. We train with Adam at a learning rate of $3\times 10^{-4}$ and use
a linear warm-up over the first $1{,}000$ steps. From step $1{,}000$ , we apply the
BatchTopK activation-threshold schedule with decay $\beta=0.999$ , together with the
auxiliary dead-feature loss weighted by $\alpha_{\mathrm{aux}}=0.03125$ . Training
activations come from Pile Uncopyrighted [35], with the beginning-of-sequence
position removed. For Gemma-2-2B [21], we use the layer-12 residual stream
with hidden width $d=2{,}304$ ; for Pythia-160M-deduped [20],
we use the layer-8 residual stream with $d=768$ .

The Matryoshka and Temporal SAEs both divide their dictionaries into the same two
predefined feature groups, containing $20\%$ and $80\%$ of the $16{,}384$ features.
The Temporal SAE weights its reconstruction losses by $0.2$ and $0.8$ ,
respectively, whereas Matryoshka uses uniform reconstruction weights. The Persistent SAE
does not use a predefined partition; it instead learns a persistence coefficient for every
feature.

Matched token budget. 
The four families process activations in different units. BatchTopK and Matryoshka
use shuffled token activations, the Temporal SAE uses adjacent token pairs, and
the Persistent SAE uses contiguous sequences so that it can carry state across tokens.
We therefore match the number of activation tokens processed per optimisation step
and the total number of steps, rather than the number of these structural units. Each
model is trained for $244{,}140$ steps with an effective batch of approximately
$2{,}048$ activation tokens per step. For the flat baselines, this gives
$2{,}048\times 244{,}140=499{,}998{,}720$ tokens, or approximately $5.0\times 10^{8}$
tokens in total.

•

BatchTopK and Matryoshka. The activation buffer returns a flat
batch of $2{,}048$ token activations at each step.

•

Temporal SAE. The buffer returns adjacent previous–next token
pairs from the same activation stream. We keep the number of reconstructed token
positions per step equal to the flat-baseline batch size.

•

Persistent SAE. The recurrent state requires contiguous
sequences rather than shuffled tokens. We use eight sequences of context length
$256$ per step. After excluding the beginning-of-sequence position, this gives
$8\times 255=2{,}040$ usable token activations per step, within $0.4\%$ of the
flat-baseline budget.

All families consequently receive the same number of gradient updates and, up to
the pairing or sequencing required by their objectives, approximately the same number
of token activations. We save checkpoints on a fixed schedule and use the final dictionary
(ae.pt) for every downstream evaluation. All experiments were run on NVIDIA A100 GPUs with 80 GB of memory.

Figure 8:
Persistent SAE training dynamics.
Training loss (left), the number of non-zero coordinates in the persistent state $\mathbf{h}^{(t)}$ (state $L_{0}$ ; centre), and the fraction of state activation mass contributed by the carried term $\bm{\lambda}\odot\mathbf{h}^{(t-1)}$ (right) for Pythia-160M at layer 8 and Gemma-2B at layer 12.
After a short initial transient, the loss and active-state size stabilise, while the carried-mass fraction remains well away from both zero and one; the models therefore use both carried state and new token-wise activations without continual growth in the active state.

Figure 8 shows the training dynamics of the Persistent SAEs.
For both models, the loss decreases rapidly and then remains stable, while the state $L_{0}$ settles at roughly $400$ – $500$ active coordinates despite continued training.
The carried-mass fraction stabilises at approximately $0.5$ for Pythia and $0.65$ – $0.70$ for Gemma.
Thus, the recurrence does not cause the active state to grow continually, and neither model relies exclusively on new evidence or carried state.

### B.2 BatchTopK Sparsity

For every SAE in this paper, sparsity is imposed using a BatchTopK mechanism in which a hidden state $\mathbf{x}\in\mathbb{R}^{d}$ is first mapped to non-negative pre-activations

| | $\tilde{\mathbf{f}}=\mathrm{ReLU}\!\left(\mathbf{W}_{\mathrm{enc}}(\mathbf{x}-\mathbf{b}_{\mathrm{dec}})+\mathbf{b}_{\mathrm{enc}}\right)\in\mathbb{R}_{\geq 0}^{m},$
| | (9)

and only the largest pre-activations are retained across the batch.
For a batch $X=\{\mathbf{x}_{i}\}_{i=1}^{B}$ , the pre-activations $\{\tilde{\mathbf{f}}_{i}\}_{i=1}^{B}$ are flattened into a single vector of length $Bm$ , and the sparse activation $\mathbf{f}_{i}$ is obtained by keeping only the top $kB$ entries globally and setting all others to zero.
The reconstruction is then

| | $\hat{\mathbf{x}}_{i}=\mathbf{W}_{\mathrm{dec}}\mathbf{f}_{i}+\mathbf{b}_{\mathrm{dec}}.$
| | (10)

In practice, after training with exact BatchTopK selection, inference uses an equivalent thresholded form

| | $f_{ij}=\tilde{f}_{ij}\,\mathbf{1}[\tilde{f}_{ij}>\theta],$
| | (11)

where the threshold $\theta$ is estimated from training-time top- $k$ statistics.
The Persistent SAE uses the resulting sparse activation as its token-wise sparse activation $\mathbf{a}^{(t)}$ ; only its downstream use differs.

### B.3 Persistence Parameterisation

The persistence coefficients for the Persistent SAE are parameterised as

| | $\lambda_{j}=\sigma(l_{j})\cdot\lambda_{\max},$
| | (12)

where $l_{j}\in\mathbb{R}$ is an unconstrained logit, $\sigma$ is the sigmoid function, and $\lambda_{\max}=0.999$ is a hard cap that prevents exact poles.
This ensures $\lambda_{j}\in(0,\lambda_{\max})$ throughout training.
The factor $(1-\lambda_{j})$ in Equation 5 makes the recurrence an exponential moving average (EMA) instead of an unnormalised sum. Under a constant activation $a^{(t)}_{j}=a_{j}$ , the state converges to $a_{j}$ for every $\lambda_{j}$ , so persistence does not systematically increase the state magnitude. The cap below one prevents exact poles and gives every feature finite effective memory. Before training, we initialise every logit $l_{j}$ to $0.5$ . This gives each feature moderate initial persistence while keeping the sigmoid away from saturation, so training can readily adjust its timescale in either direction.

## Appendix C Dataset Details

We use three datasets for distinct parts of the evaluation. Pile documents provide the hidden states used to train and evaluate the SAEs, MMLU questions provide subject-labelled text for analysing the persistent state, and AgentDojo trajectories support the prompt-injection monitoring case study. We describe each dataset and our use of it below.

### C.1 Pile Corpus

The original Pile is an $825$  GiB English-language corpus assembled from 22 sources, including academic papers, web pages, books, code, and online discussions [22]. We use the Pile Uncopyrighted release [35], a derived version that removes the Books3, BookCorpus2, OpenSubtitles, YouTube Subtitles, and OpenWebText2 components. Documents are accessed in streaming mode and shuffled with a buffer of size $10{,}000$ using a fixed seed. A document is accepted if, after tokenisation with the model tokeniser and truncation to the context cap, it contains at least the required minimum number of tokens.

For the natural corpus, each accepted document is passed through the model, and the chosen residual-stream hidden states are retained.
If a sequence has retained length $L$ , temporal statistics are computed on token positions $\{16,\dots,L-1\}$ after dropping the initial tokens.

The shuffled control is constructed at the sentence level rather than by permuting tokens. Sentences are sampled from documents drawn from the same corpus and concatenated subject to a simple adjacency constraint that prevents consecutive sentences from coming from the same source document. This preserves local continuity while breaking global continuity, including long-range entity tracking and discourse progression.

### C.2 MMLU Dataset

Massive Multitask Language Understanding (MMLU) is an English-language benchmark of four-option multiple-choice questions from 57 subjects spanning STEM, the humanities, the social sciences, and other professional and academic fields [29]. We use MMLU only for the subject analyses; it is not used to train the SAEs. Specifically, we use questions from abstract algebra, astronomy, business ethics, computer security, and European history. These subjects cover distinct areas while providing a subject label for every example.

For the geometry analysis in Section 5.3, we sample $100$ sequences across the five subjects and retain the final $20$ token representations from each sequence. For subject probing, we construct sequences by joining equal-length segments from different subjects and assign each token the subject label of its source segment. Appendix F and Appendix G provide the sampling and evaluation details for the two analyses.

### C.3 AgentDojo Trajectories

Indirect prompt injection occurs when an agent retrieves attacker-controlled text from emails, webpages, documents, or tool outputs and follows instructions inside that text. AgentDojo [12] evaluates this setting with stateful tool environments containing 97 user tasks, 27 injection targets, and 629 security test cases across workspace, Slack, travel, and banking domains. Its task-specific checks evaluate whether the legitimate user request and the attacker’s target action were completed by inspecting the agent output and changes to the environment.

For our monitoring dataset, we use trajectories from the banking, travel, and workspace domains generated by Llama-3.3-70B-Instruct. Benign trajectories without an injection form the negative class ( $y=0$ ). The positive class ( $y=1$ ) contains runs with an important_instructions or direct attack for which the AgentDojo injection checker returns security=false. This value means that an injection was present but the agent did not complete the attacker’s target action. We retain these failed attacks deliberately: successful attacks often contain an overt malicious action, such as transferring money or sending private data, which could provide a trivial downstream classification cue. This restriction asks whether the monitor retains a trace of the injection even when no observable attack success occurs. We do not filter by task utility, so the legitimate user request may either succeed or fail.

## Appendix D Empirical Persistence of Standard SAE Features

We evaluate lags $\tau\in\{1,4,16,32,64,128,256,512\}$ on documents of at most $1{,}024$ traced tokens. A feature is retained only if it has at least $256$ activation events, at least $64$ valid anchors at every lag, and a non-negligible activated-token elevation. In particular, $R_{j}(\tau)$ is undefined and excluded when its denominator $\mathbb{E}[p^{(t)}_{j}\mid\mathbf{f}^{(t)}_{j}>0]-\mathbb{E}[p^{(t)}_{j}]$ is near zero, as occurs for near-dead or diffuse features; we do not add an $\epsilon$ to the numerator or denominator.
On Pythia-160M, median persistence for BatchTopK falls from $0.618$ at lag $1$ to $0.160$ at lag $64$ and is still $0.096$ at lag $512$ ; Matryoshka follows the same pattern, from $0.504$ to $0.108$ and $0.059$ . On Gemma-2-2B, natural-document persistence remains measurable at lag $512$ , reaching $0.042$ (BatchTopK) and $0.034$ (Matryoshka). On both models the shuffled control decays to approximately zero at long lags.
Fitting power-law descriptors to the positive decay tail yields median exponents $\beta_{\mathrm{nat}}\approx 0.32$ versus $\beta_{\mathrm{shf}}\approx 0.87$ – $0.92$ , a gap that holds for both SAE families.

Individual features exhibit substantially different persistence profiles across lags. When features are ranked by their average $R_{j}(\tau)$ within short ( $\tau\leq 16$ ), medium ( $32\leq\tau\leq 64$ ), and long ( $\tau\geq 128$ ) lag bands, the resulting distributions span nearly the full unit interval at every band. Some features are strongly persistent at short lags but decay before medium lags; others retain elevated projections well into the long-lag regime. This heterogeneity is present in both SAE families, and is substantially larger than what is observed in the shuffled control.

### D.1 Context Correction Experiment

We fit the per-feature context correction of Equation 3 independently at each lag $\tau\in\{1,2,4,8,16,32\}$ . For each lag, we use $192$ natural documents to fit $\mathbf{w}_{\tau}$ and $96$ held-out natural documents to evaluate $R^{2}_{\mathrm{improve}}$ . The target is always the natural-document reconstruction error. For the control, we refit the diagonal head to predict these same natural targets from context activations drawn from the shuffled corpus, using the same parameterisation and optimisation budget. Each document contributes at most $768$ tokens after dropping $16$ prefix tokens. We optimise for $10$ epochs with AdamW, learning rate $5\times 10^{-3}$ , and weight decay $10^{-3}$ .

On Pythia-160M the correction removes $0.89\%$ (BatchTopK) and $0.61\%$ (Matryoshka) of the baseline reconstruction error at lag $1$ , decreasing but staying positive at lag $32$ ( $0.22\%$ and $0.13\%$ ). On Gemma-2-2B both SAEs remove approximately $0.40\%$ at lag $1$ , falling to about $0.02\%$ at lag $16$ and vanishing by lag $32$ . The shuffled controls stay slightly below zero throughout, so unrelated context never improves reconstruction; the gain requires feature activations from the actual history of the target document.

Table 3 summarises the fitted lag-one weights. On natural documents the mean weight is positive and a large majority of features take a positive weight, indicating that past activation of many features predicts a positive contribution along the same feature direction at the next token. The shuffled distributions are sharply centred at zero. This gives a feature-level reading of the reconstruction gain: it comes largely from carrying feature-aligned activity forward through a coherent document, which is precisely the behaviour the persistent state of Section 4 makes explicit.

Table 3: Fitted lag-one context-correction weights $w_{j}$ . Mean $w_{j}$ and Pos. frac. are the mean and the positive fraction on natural documents; Shuf. $w_{j}$ is the mean on the shuffled control.

| Model | SAE | Mean $w_{j}$ | Pos. frac. | Shuf. $w_{j}$

| Pythia | BatchTopK | $0.068$ | $76.2\%$ | $\phantom{-}0.001$

| Pythia | Matryoshka | $0.087$ | $91.4\%$ | $-0.001$

| Gemma | BatchTopK | $0.068$ | $80.4\%$ | $-0.001$

| Gemma | Matryoshka | $0.091$ | $81.9\%$ | $-0.001$

## Appendix E Automated Interpretability Details

We follow the automated interpretability pipeline of Bills et al. [23]. For each feature we collect its highest-activation contexts, prompt a judge model to write a short natural-language explanation, and score how well that explanation predicts whether the evaluated feature activation exceeds a threshold on held-out contexts. For standard SAEs, this activation is $f_{j}^{(t)}$ ; for the Persistent SAE, we use the token-wise activation $a_{j}^{(t)}$ . We use gpt-4o-mini as the judge, a context length of $1024$ tokens, and roughly $2$ M tokens from the Pile for the activation examples. This evaluates the Persistent SAE on its token-wise activations, matching the token-specific representations of the baselines. We also score the highest- $\lambda_{j}$ $1\%$ and lowest- $\lambda_{j}$ $50\%$ of Persistent SAE features to evaluate their interpretability as groups.

Table 4: Qualitative autointerpretation examples for Pythia-160M. High- and low- $\lambda$ features are drawn from the top 1% and bottom 50% of the learned persistence coefficients, respectively. Bold text marks representative activating spans. The selected high- $\lambda$ features receive token-wise evidence in coherent semantic contexts, whereas the low- $\lambda$ features respond to local strings, syntax, or recurring templates. All selected features have an autointerpretation score above 0.85.

| Feature | $\lambda$
|

Autointerpretation

|

Representative activation

| Score

| High $\lambda$ (top 1%): sustained semantic domains

| F-5528 | $7.03\text{\times}{10}^{-1}$
|

themes of ethnic diversity and experiences of discrimination

|

“ashamed of being Mexican because I’m different from the stereotype”

| 0.93

| F-12751 | $4.82\text{\times}{10}^{-1}$
|

phrases related to population-based studies and cancer research

|

incident colorectal cancer cases identified through a population-based cancer registry

| 0.93

| F-15986 | $3.81\text{\times}{10}^{-1}$
|

terms related to high-performance cars and their specifications

|

rare and desirable muscle cars; a 1971 Hemi ’Cuda

| 0.93

| F-6538 | $3.79\text{\times}{10}^{-1}$
|

terms related to weather conditions and climate phenomena

|

a combination of global warming and the El Niño weather phenomenon

| 1.00

| F-8449 | $3.69\text{\times}{10}^{-1}$
|

mathematical concepts and terms related to Banach spaces and polynomials

|

continuous scalar polynomials over a Banach space $X$

| 1.00

| Low $\lambda$ (bottom 50%): local lexical, syntactic, and template patterns

| F-1113 | $5.35\text{\times}{10}^{-2}$
|

the substring “white” followed by various words or concepts

|

“There is so much white space just waiting to be filled”

| 1.00

| F-6958 | $4.96\text{\times}{10}^{-2}$
|

the substring “function” in JavaScript code snippets

|

.click(function() {

| 1.00

| F-15079 | $4.81\text{\times}{10}^{-2}$
|

the phrase “U.S. Pat. No.” indicating references to patents

|

“disclosed in U.S. Pat. No. 5,111,981”

| 1.00

| F-3587 | $1.10\text{\times}{10}^{-2}$
|

URLs and web documentation references

|

ubuntu.com/10.04/serverguide/C/index.html

| 1.00

| F-14758 | $4.04\text{\times}{10}^{-4}$
|

SQL commands related to table creation and data manipulation

|

CREATE TABLE IF NOT EXISTS tennis(...)

| 1.00

Table 5: Qualitative autointerpretation examples for Gemma-2-2B. High- and low- $\lambda$ features are drawn from the top 1% and bottom 50% of the learned persistence coefficients, respectively. Bold text marks representative activating spans. The selected high- $\lambda$ features receive token-wise evidence in coherent semantic contexts, whereas the low- $\lambda$ features respond to local strings, syntax, or recurring templates. All selected features have an autointerpretation score above 0.85.

| Feature | $\lambda$
|

Autointerpretation

|

Representative activation

| Score

| High $\lambda$ (top 1%): sustained semantic domains

| F-11205 | $9.18\text{\times}{10}^{-1}$
|

terms related to cosmology and advanced scientific concepts

|

near-future CMB Stage-III and IV surveys yielding reconstructed CMB lensing maps

| 0.86

| F-5578 | $8.55\text{\times}{10}^{-1}$
|

terms related to ionisation charge collection and quantum experiments in physics

|

ionisation charge collection efficiency observed in gases

| 0.86

| F-15639 | $8.24\text{\times}{10}^{-1}$
|

text related to conflict, violence, and ethnic tensions

|

self-determination for Katanga; a separatist group composed of former Katanga Tigers

| 0.86

| F-13900 | $7.85\text{\times}{10}^{-1}$
|

terms related to microbial biomass and ecological concepts in alpine and tundra environments

|

microbial biomass in an alpine meadow on the Tibetan Plateau

| 0.86

| F-11933 | $7.70\text{\times}{10}^{-1}$
|

expressions of self-acceptance and emotional support in personal growth narratives

|

“In the last couple years, I’ve let go and decided to be myself.”

| 1.00

| Low $\lambda$ (bottom 50%): local lexical, syntactic, and template patterns

| F-9996 | $1.03\text{\times}{10}^{-3}$
|

legal terms and conditions related to software licenses

|

“See the License for the specific language governing rights and limitations”

| 1.00

| F-11210 | $1.30\text{\times}{10}^{-5}$
|

the phrase “all of” in various contexts

|

“it is unlikely that all of these loci respond to selection independently”

| 0.93

| F-14595 | $7.87\text{\times}{10}^{-6}$
|

the substring “mk” within mathematical and scientific contexts

|

repeated \mkern commands in mathematical markup

| 0.93

| F-13259 | $2.73\text{\times}{10}^{-6}$
|

specific terms or phrases enclosed in quotation marks

|

users searching for “Pizza” or “Calgary pizza”

| 1.00

| F-7307 | $1.88\text{\times}{10}^{-6}$
|

the word “In” at the beginning of sentences

|

“In this way the missing persons in our evolutionary mystery would …”

| 0.86

## Appendix F Subject Geometry Details

For the t-SNE visualisation we draw $100$ sequences from five MMLU subjects (abstract algebra, astronomy, business ethics, computer security, and European history) and keep the last $20$ tokens of each sequence, so that every token carries a subject label. For each representation—raw hidden states, BatchTopK features, the Temporal SAE’s high-level group, and the Persistent SAE’s high- $\lambda_{j}$ state—we collect the token-level vectors and project them to two dimensions with t-SNE using a common set of hyperparameters, colouring each point by its subject.

## Appendix G Subject Probing Details

We construct spliced sequences by concatenating equal-length segments drawn from different MMLU subjects, balancing subject labels within each sequence and holding out entire source documents for the test split, so that a token’s label is the subject of the segment it falls in. We treat each token as a labelled example and fit a linear probe to predict its source subject from the token-level representation. We report held-out one-vs-rest macro ROC AUC as a function of the fraction of features made available, swept from $0.1\%$ to $100\%$ . We evaluate the Persistent SAE on its persistent state $\mathbf{h}^{(t)}$ and order its features by decreasing $\lambda_{j}$ , increasing $\lambda_{j}$ , repeated random permutations, or supervised one-vs-rest importance. For each subject $c$ , the supervised score for feature $j$ is

| | $s_{j,c}=\mathbb{E}[h^{(t)}_{j}\mid y^{(t)}=c]-\mathbb{E}[h^{(t)}_{j}\mid y^{(t)}\neq c],$
| | (13)

computed on the training split. This score measures how much larger the persistent state of feature $j$ is for subject $c$ than for the other subjects. We sort features by this score within each subject and interleave the subject-specific rankings in round-robin order, ensuring that every subject contributes its most discriminative features. This produces a single global feature ordering used by the multiclass probe. For the BatchTopK, Matryoshka, and Temporal baselines, we use their full dictionaries under the same probe.

## Appendix H Prompt Injection Monitoring Details

Trajectory construction. 
Trajectories are taken from AgentDojo [12] runs of Llama-3.3-70B-Instruct agents. Attack trajectories are restricted to the important_instructions and direct attack types and to runs with security=false. In this dataset, these positives contain an injection whose targeted action the agent did not execute; benign trajectories contain no injected content. Each trajectory is serialised and appended with the monitoring query

should this trajectory be flagged for prompt injection or unsafe instruction hijacking? Answer yes or no:

and read with Gemma-2-2B; all representations are taken from the layer-12 residual stream. The model’s own answer to this query is close to uninformative: the yes-minus-no answer logit at the query position gives an AUC of $0.51$ , which is why we train internal monitors rather than relying on prompting.

Checkpoints and probes. 
For each trajectory, we locate the final token of the injected span (for benign trajectories, the token at the matched position) and define checkpoints at fixed offsets of $0$ , $50$ , $100$ , $200$ , $250$ , $300$ , and $350$ tokens after it, together with the final query position. At each checkpoint we extract, from the same layer, the raw hidden state, the BatchTopK and Matryoshka sparse activations, the Temporal SAE’s high-level feature group, and the Persistent SAE’s token-wise activation $\mathbf{a}^{(t)}$ , full state $\mathbf{h}^{(t)}$ , and slow state (the largest 1% of $\mathbf{h}^{(t)}$ coordinates by $\lambda_{j}$ ). A logistic regression probe is fit on each representation using checkpoint $d=0$ only and then evaluated, without retraining, at every other checkpoint; we report held-out accuracy and ROC AUC.

Causal intervention. 
For each SAE representation, we select the $k=128$ features with the largest absolute probe coefficients and combine their weighted decoder columns into a probe-weighted decoder direction $\mathbf{v}_{r}$ . The direction is normalised and scaled so that $\lVert\alpha_{i}^{(d)}\mathbf{v}_{r}\rVert_{2}=\rho\lVert\mathbf{x}_{i}^{(d)}\rVert_{2}$ , with $\rho=0.1$ , then added to the residual stream with both signs. Equation 8 measures the resulting change in the external replay monitor’s logit margin, not a change in the original Llama-3.3-70B agent’s action. Interventions use the direction constructed at $d=0$ , so Figure 7 reports how its causal effect transfers across distance.
