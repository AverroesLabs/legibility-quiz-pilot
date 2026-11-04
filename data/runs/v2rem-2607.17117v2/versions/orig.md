# Persistent Sparse Autoencoders: Learning Feature-Specific Timescales in Language Model Representations

Haoyan Luo

Affiliation: University of Cambridge

  
Mateo Espinosa Zarlenga

Affiliation: University of Oxford

  
Mateja Jamnik

Affiliation: University of Cambridge

###### Abstract

Sparse autoencoders (SAEs) decompose language model activations into sparse features, but they encode each token independently and so do not expose information that persists across a sequence. Contextual properties such as topic, register, task state, and the presence of untrusted instructions extend over many tokens, yet an atemporal decomposition can only describe them indirectly through whichever token-level features happen to fire. In this paper we argue that persistence should be treated as a first-class property of sparse features rather than imposed through fixed partitions of the dictionary. We introduce Persistent Sparse Autoencoders (Persistent SAEs), an extension of standard SAEs that separates a token-wise sparse activation from a persistent feature state, so that every feature is associated with its own timescale rather than a role assigned in advance. We describe the design principles behind this formulation, including bounded state dynamics, preservation of feature-level attribution through decoder directions, and training budgets matched to standard SAE baselines. We then set out an evaluation protocol covering reconstruction fidelity, temporal smoothness, automated interpretability, subject-level structure, and long-context monitoring of indirect prompt injection in agent trajectories. Together, the formulation and protocol provide a framework for studying how sparse features carry information over time in language models, and for asking whether such features can support interpretation and monitoring over long contexts.

## 1 Introduction

Understanding the output of a large language model (LLM) requires identifying what information is represented in its hidden states and how that information evolves throughout the sequence [1, 2]. Sparse autoencoders (SAEs) [3, 4], methods that decompose an LLM’s hidden states into a set of sparse features, have become a popular tool for approaching this problem. Specifically, when features are aligned with human-interpretable concepts, an SAE’s decomposition enables easy inspection of concepts learnt by an LLM, and provides a mechanism for model control through inspection of and intervention on these features [5, 6, 7].

Although causal transformer hidden states integrate the preceding context, standard SAEs encode each token independently [3, 8, 9]. This atemporal formulation favours token-specific features but obscures the contextual information a model carries across a sequence, such as the current topic, style, or task state [10].
To alleviate this constraint, recent methods have incorporated temporal structure into SAEs through predefined partitions and decompositions of temporal roles [10, 11].
However, these approaches require specifying, a priori, which features should carry temporal information, leading to experimenter bias in the features the SAE learns.
It therefore remains unclear how each feature’s timescale could instead be associated with the feature itself, rather than being predefined by the model architecture.

The distinction matters for interpretability practice. When an analyst inspects an SAE feature, the evidence they examine is typically a set of short, high-activation contexts. Such contexts are well suited to features that respond to a specific token or phrase, but they are poorly suited to properties that are only meaningful at the level of a paragraph or a document. A feature that tracks whether a passage concerns abstract algebra, for example, may be easier to describe from the passage as a whole than from any single token. Conversely, imposing a document-level view on a feature that responds to a substring such as “function” in JavaScript code adds little and may blur its description. A decomposition that treats all features identically in time therefore forces a single granularity onto a representation that is, in all likelihood, heterogeneous.

This paper approaches the problem by making persistence an explicit, per-feature component of the SAE. We introduce Persistent Sparse Autoencoders, which keep the token-wise sparse encoder of a standard SAE but route each feature’s activations through a persistent state before decoding. Each feature is associated with its own timescale, and the decoder reconstructs the hidden state from these states rather than from the instantaneous activations alone. Because the state of each feature is indexed by the same dictionary element as its decoder direction, contributions to the reconstruction remain attributable to individual features, which preserves the basic inspectability that motivates SAEs in the first place.

Our second goal is methodological. Temporal extensions of SAEs are difficult to evaluate, because the usual metrics for SAEs—reconstruction error, sparsity, and token-level interpretability scores—were designed for atemporal decompositions. A persistent feature may be interpretable in a different sense from a token-level feature, and may be useful for different downstream tasks. We therefore set out an evaluation protocol that examines Persistent SAEs along several complementary axes: reconstruction fidelity and temporal smoothness; local interpretability of the token-wise activations; the kinds of tokens at which features receive new evidence; whether persistent states organise sequences by subject; and whether persistent states support long-context monitoring of indirect prompt injection in agentic settings [12].

The prompt-injection setting in particular illustrates why persistence is relevant beyond descriptive interpretability. An agent may read a malicious instruction embedded in a tool response, continue reasoning for hundreds of tokens, and only then act. A monitor that inspects the agent’s internal representations must therefore retain a security-relevant signal over a long span, not merely detect the injected text at the moment it appears. Representations that explicitly carry information forward are natural candidates for this role, and evaluating them in this setting connects the study of temporal SAE features to a practical safety concern.

Our main contributions can be summarised as follows:

•

We discuss why temporal structure at the level of individual SAE features is a distinct question from temporal structure in hidden states, and outline the controls needed to separate document-level context from local coherence.

•

We introduce a new SAE variant, termed Persistent Sparse Autoencoders, that associates each sparse feature with a persistent state and a feature-specific timescale, without a predefined partition of temporal roles, while preserving attribution to decoder directions.

•

We describe a matched training setup against BatchTopK, Matryoshka, and Temporal SAE baselines, and an evaluation protocol spanning reconstruction, interpretability, subject structure, and long-context prompt-injection monitoring and steering.

## 2 Background and Related Work

Sparse autoencoders. 
Given an LLM’s hidden state $\mathbf{x}^{(t)}\in\mathbb{R}^{d}$ at token $t$ , a standard SAE [3] encodes $\mathbf{x}^{(t)}$ as non-negative sparse activations $\mathbf{f}^{(t)}\in\mathbb{R}_{\geq 0}^{m}$ from which it can be reconstructed as

| | $\hat{\mathbf{x}}^{(t)}_{\mathrm{base}}=\mathbf{W}_{\mathrm{dec}}\mathbf{f}^{(t)}+\mathbf{b}_{\mathrm{dec}}.$
| | (1)

Under this representation, the $j$ -th feature is represented by the activation $\mathbf{f}^{(t)}_{j}$ and its corresponding normalised decoder direction $\mathbf{d}_{j}=\mathbf{W}_{\mathrm{dec}}[:,j]/\lVert\mathbf{W}_{\mathrm{dec}}[:,j]\rVert_{2}$ (see Appendix A for a notation summary). Intuitively, we think of feature $j$ as active when $\mathbf{f}^{(t)}_{j}>0$ .
Because $\mathbf{f}^{(t)}$ is sparse by construction, each reconstruction depends on only a small set of features, making the contribution of individual features easier to inspect [3, 13].
Subsequent SAEs improve how representational capacity is allocated and organised: BatchTopK [8] replaces the fixed per-token TopK constraint with a batch-level sparsity budget, allowing the number of active features to adapt across tokens, while Matryoshka SAEs [9] jointly train nested dictionaries of increasing size so that smaller subsets retain higher-level features as larger dictionaries add more specific ones.

The appeal of SAEs rests on two properties that any extension should preserve. The first is sparsity: an analyst can reason about a reconstruction in terms of a handful of active features rather than thousands of dense coordinates. The second is attribution: each feature corresponds to a single decoder direction, so its contribution to the hidden state can be read off, ablated, or amplified. Interventions that steer model behaviour through SAE features [5, 6, 7] depend on both properties, as do interpretability workflows that describe features from their activating contexts. An extension that improves reconstruction at the cost of either property would undermine the reasons SAEs are used.

Temporal structure in LLM representations. 
Although autoregressive LLMs build $\mathbf{x}^{(t)}$ from the preceding context, standard SAEs and most SAE variants compute features $\mathbf{f}^{(t)}$ independently of the features observed in preceding tokens. SAE features, therefore, do not explicitly expose information carried across tokens. Yet, traditional slow feature analysis shows that slowly varying latent factors reveal higher-level features in neural networks [14, 15].
Prior studies likewise show that language models track entities and that their hidden states encode game-board, spatial, temporal, and propositional states [16, 17, 18, 19]. These findings motivate mechanisms that explicitly encode temporal persistence in SAE features.

The slow feature analysis literature is a useful conceptual reference point. Its central premise is that the factors one cares about in a signal often change more slowly than the raw observations, so that seeking slowly varying projections tends to recover meaningful latent structure. In language, a document’s topic or genre changes far more slowly than the identity of individual tokens, and a model that answers questions about a passage must maintain such factors across the passage. At the same time, language contains many properties that are intrinsically fast: syntactic category, the identity of a substring, the boundary of a quotation. A useful decomposition should therefore not simply favour slowness, but accommodate a range of timescales and make the timescale of each feature legible.

Temporal extensions of SAEs. 
Recent works, specifically temporal formulations of SAE [10, 11], have made progress toward capturing feature persistence. Temporal SAEs [10] pre-assign a high-level feature subset and use a contrastive objective to align their activations across nearby tokens. Lubana et al. [11] instead use an attention-based predictor to separate context-predictable information from a sparse novel residual representation. Both prescribe temporal structure via a fixed partition. However, neither allows each sparse feature to have its own temporal persistence, complicating post-hoc analysis of the expected timescale of different features. Persistent SAEs aim to fill this gap by assigning each feature a persistent state with a feature-specific timescale that can be analysed directly.

There are also broader connections to recurrent and state-space sequence models, which maintain hidden states that evolve under learned dynamics. Such models are typically dense and are designed for prediction rather than for interpretability. Persistent SAEs borrow the idea of a carried state but restrict it so that each coordinate of the state remains tied to a single dictionary element. This restriction sacrifices expressiveness relative to general recurrent models, but it is precisely what allows each persistent coordinate to be read as the state of a nameable feature.

Figure 1:
Overview of the atemporal and persistent views of sparse features.
(a) A standard SAE encodes each token’s hidden state independently, so contextual properties appear only through whichever token-level features fire at each position.
(b) A Persistent SAE retains a token-wise sparse activation but carries each feature’s information forward in a persistent state, which the decoder uses for reconstruction.
(c) Because each persistent coordinate is tied to one decoder direction, the contribution of every feature to the reconstruction remains attributable.

## 3 Does Temporal Structure Exist at the Feature Level?

Before introducing an architecture that makes persistence explicit, it is worth being precise about what it would mean for standard SAE features to exhibit temporal structure, and why this question differs from the well-established observation that hidden states themselves are temporally correlated.

Setup. 
Throughout the paper we work with residual-stream hidden states $\{\mathbf{x}^{(t)}\}_{t=1}^{T}$ from Pythia-160M-deduped (layer 8) [20] and Gemma-2-2B (layer 12) [21] on Pile documents [22]. We train BatchTopK [8] and Matryoshka [9] SAEs, neither of which has a temporal prior, as standard references. We describe training configurations in Appendix B.

Hidden-state versus feature-level structure. 
It is well known that consecutive hidden states of a causal transformer are correlated, and Lubana et al. [11] show that past hidden states explain a large part of the current hidden state. This does not by itself imply that individual SAE features carry temporal information. A sparse decomposition may assign the persistent part of the hidden state to a small number of features, to many features weakly, or to none at all, absorbing it instead into a bias term or into features that fire on almost every token. Whether temporal structure is visible at the level of individual features is therefore an empirical question about the decomposition, not about the model.

Natural Pile documents are continuous at two scales: tokens form coherent sentences (local continuity), and the sentences collectively develop a coherent document-level context (global continuity). Any analysis of feature persistence must disentangle these two sources. A feature that fires on the first word of a multi-word expression will tend to be followed by related signal on the next few tokens simply because the expression continues, and this kind of persistence says little about whether the representation tracks document-level context. Conversely, signal that is maintained across sentence boundaries in a coherent document is more plausibly attributable to global continuity. Appendix C describes the corpus we use and how sequences are prepared, and Appendix D discusses methodological considerations for measuring persistence in a way that respects this distinction.

Projections onto decoder directions. 
A natural starting point for feature-level analysis is the projection $p^{(t)}_{j}=\mathbf{d}_{j}^{\top}\mathbf{x}^{(t)}$ of the hidden state onto the direction representing feature $j$ . Unlike the sparse activation $f^{(t)}_{j}$ , which is zero whenever the feature falls below the sparsity threshold, the projection is defined at every token and so can reveal signal that is present in the hidden state but not strong enough to be selected by the encoder. Tracking projections after a feature activates is one way to ask whether the information a feature represents lingers in the residual stream. However, projections are confounded by the geometry of the dictionary: decoder directions are not orthogonal, so the projection onto one feature direction can rise because a correlated feature is active. Any analysis built on projections must therefore be interpreted with care, and comparisons across features with very different activation frequencies require normalisation.

Past activations as a source of reconstruction signal. 
A complementary angle asks whether a feature’s activation at an earlier token is informative about the hidden state at a later token along the same direction. If it were, a standard SAE—which reconstructs each token only from its own activations—would be leaving reconstruction-relevant information unused. Framing the question in terms of the same feature direction, rather than an arbitrary linear map from past activations, is important for interpretability: it keeps any effect attributable to a single feature, rather than to a dense mixture of features. It also distinguishes the question from general sequence prediction, in which past hidden states are trivially informative about future ones.

Why an architectural answer. 
Diagnostics of this kind are informative but indirect. They are computed post hoc on a decomposition that was trained without any notion of time, and they inherit the confounds described above. They also leave open how a practitioner would use any temporal structure that they reveal. An alternative is to build persistence into the decomposition itself and to let the training objective determine how each feature uses it. This turns the question of feature-level temporal structure into a question about what an SAE learns when it is permitted to carry information across tokens, and it produces a representation that can be inspected and used directly. We pursue this alternative in the next section, while noting that the diagnostic perspective remains useful for interpreting what the resulting model learns.

## 4 Persistent Sparse Autoencoders

In this section, we introduce Persistent SAEs, which make persistence explicit by separating a token-wise sparse activation from a persistent state (Figure 2).

Figure 2:
Persistent Sparse Autoencoders. Each feature integrates its token-wise sparse activation $\mathbf{a}^{(t)}$ into a persistent state $\mathbf{h}^{(t)}$ governed by a feature-specific timescale.

As in standard BatchTopK encoders (see Section B.2 for details), a Persistent SAE encoder produces token-wise sparse activation $\mathbf{a}^{(t)}\in\mathbb{R}_{\geq 0}^{m}$ , where $a^{(t)}_{j}>0$ means feature $j$ finds new evidence at token $t$ . In contrast to standard SAEs, however, each feature integrates this activation into a persistent state $\mathbf{h}^{(t)}\in\mathbb{R}_{\geq 0}^{m}$ :

| | $\mathbf{h}^{(t)}=\Phi_{\bm{\theta}}\big(\mathbf{h}^{(t-1)},\mathbf{a}^{(t)}\big),$
| | (2)

where $\Phi_{\bm{\theta}}$ is a state-update map with parameters $\bm{\theta}$ that include a timescale for each feature. We place three requirements on this map. First, it should act feature-wise, so that the $j$ -th coordinate of the state depends only on the history of feature $j$ ; this keeps each persistent coordinate tied to a single dictionary element. Second, it should be stable, so that the magnitude of the state remains bounded over long sequences and the decoder sees inputs on a consistent scale regardless of position. Third, it should admit timescales ranging from effectively memoryless, in which case the state reduces to the token-wise activation and the feature behaves as in a standard SAE, to long-lived, in which case the state summarises evidence accumulated over many tokens.

During training, we initialise $\mathbf{h}^{(0)}=\mathbf{0}$ at the start of every input sequence. Section B.3 discusses the parameterisation considerations for the timescale parameters in more detail.

The decoder reconstructs the hidden state from this persistent state:

| | $\hat{\mathbf{x}}^{(t)}=\mathbf{W}_{\mathrm{dec}}\mathbf{h}^{(t)}+\mathbf{b}_{\mathrm{dec}}.$
| | (3)

Because the state update acts feature-wise, feature $j$ carries only its own activation forward, preserving feature-level attribution through its corresponding decoder direction. Sparsity is imposed on the token-wise activation $\mathbf{a}^{(t)}$ , so the accumulated state $\mathbf{h}^{(t)}$ integrates sparse updates but is itself not necessarily sparse. This separation means that a Persistent SAE offers two complementary views of the same feature: the token-wise activation, which indicates where new evidence arrives, and the persistent state, which indicates what the feature currently represents given the history.

Relation to standard and temporal SAEs. 
Persistent SAEs contain standard SAEs as a special case: if every feature’s timescale is set to be memoryless, the state equals the token-wise activation and Equation 3 reduces to Equation 1. They differ from Temporal SAEs [10] in that no subset of features is designated as high-level in advance and no contrastive objective is imposed across neighbouring tokens. They differ from the predictive decomposition of Lubana et al. [11] in that the carried information is not produced by a dense attention-based predictor but by feature-wise states that remain individually interpretable. In all three cases, the motivating idea is the same—that some information in the hidden state is best described at a timescale longer than one token—but the Persistent SAE locates the timescale in each feature rather than in a partition of the dictionary.

Practical considerations. 
Carrying state across tokens changes how data must be presented during training. Standard SAEs are trained on shuffled token activations, which decorrelates consecutive batches and simplifies optimisation. A Persistent SAE instead requires contiguous sequences, so that the state at each token reflects the actual preceding context. This introduces correlations within each batch, which we address by drawing several sequences per step, and it requires matching token budgets carefully when comparing against flat baselines (Appendix B). Inference is equally straightforward: the state is updated once per token, at a cost linear in the dictionary size, and can be maintained online as a sequence is processed.

## 5 Experiments

In this section, we describe how we evaluate the representations learnt by Persistent SAEs and how we examine what appears at each timescale. Specifically, we first consider whether adding a persistent state preserves reconstruction quality and interpretable token-wise activations (Sections 5.1 and 5.2). We then ask what information the accumulated states of persistent features might capture (Section 5.3). Finally, as a demonstration of the practical uses of our proposed approach, we describe how persistent states can be evaluated for long-context monitoring in a prompt injection case study (Section 5.4).

Models and hyperparameters. 
Throughout all experiments, we use the same models and dataset as in Section 3. We compare with BatchTopK SAE [8] and Matryoshka SAE [9] as strong standard baselines, and Temporal SAE [10] as the closest temporally-structured baseline.
All SAEs are trained with $m=16{,}384$ features and a BatchTopK budget of $k=20$ . Matryoshka and Temporal SAEs both use the same predefined 20%/80% feature partition. Appendix B gives the full training configuration.

The choice of baselines reflects the two questions the evaluation is meant to answer. BatchTopK and Matryoshka SAEs establish the reconstruction and interpretability that an atemporal decomposition achieves under the same width and sparsity; any temporal extension should be judged against them on these standard criteria. The Temporal SAE establishes what a temporally structured but partition-based design provides, and so isolates the effect of associating timescales with individual features rather than with a predefined group.

### 5.1 Reconstruction Quality and Learned Timescales

The first requirement for any SAE extension is that it should not degrade the decomposition it extends. We evaluate the quality of each method’s representations by looking at: (1) the fraction of variance explained (FVE); (2) the cosine similarity between the hidden state and its reconstruction; (3) the fraction of features that remain alive; and (4) the degree to which features change smoothly over time, using the temporal smoothness metric from Bhalla et al. [10]. Table 1 summarises these evaluation dimensions together with the representation evaluated for each SAE family.

The smoothness metric deserves comment, because it is the dimension on which temporal and atemporal decompositions are expected to differ most. Smoothness measures how much a feature’s value changes between consecutive tokens, normalised by its overall scale. A decomposition can be made arbitrarily smooth by sacrificing reconstruction, so smoothness is only informative when read alongside FVE and cosine similarity. For the Matryoshka and Temporal SAEs, smoothness can be reported separately for the predefined high-level and low-level groups. For the Persistent SAE, there is no predefined group, so any grouping must be derived from the model’s own timescale parameters; this makes the comparison asymmetric by design, and we report it with that caveat in mind.

Timescales as an object of analysis. 
Beyond aggregate metrics, a Persistent SAE exposes a timescale for every feature, which invites questions that cannot be asked of standard SAEs. How are timescales distributed across the dictionary? Do features with long timescales differ systematically from features with short timescales in what they represent? Do the timescales correspond to anything observable in the hidden states, such as how long a feature’s direction remains elevated after it activates? The last question is important because a timescale parameter is a property of the model, not of the data, and it need not correspond to any measurable persistence in the underlying representation. Relating the parameters to projection-based measures of persistence (Appendix D) is one way to check whether they are meaningful, and a corresponding check for the Temporal SAE is whether its predefined groups differ in the same measures.

Table 1:
SAE evaluation dimensions. High/Low denote predefined splits for Matryoshka and Temporal SAEs. For Persistent SAEs, groupings are derived from the timescale parameters rather than predefined.

| Dimension | Metric | Direction | Representation evaluated
|---|---|---|---
| Fidelity | FVE | $\uparrow$ | Reconstruction from Equation 1 (baselines) or Equation 3 (Persistent)
| Fidelity | Cosine Sim | $\uparrow$ | Same as above
| Capacity | Fraction Alive | $\uparrow$ | Token-wise sparse activation
| Temporal | Smoothness (Full / High / Low) | $\downarrow$ | Feature values across consecutive tokens
| Interpretability | Autointerp (Full / High / Low) | $\uparrow$ | Token-wise activation $\mathbf{f}^{(t)}$ or $\mathbf{a}^{(t)}$

### 5.2 Short-Timescale Features Are Locally Interpretable

We next ask how timescales relate to feature interpretability. We use the automated interpretability pipeline of Bills et al. [23], in which a judge model explains a feature from its highest-activation contexts then scores how well that explanation predicts activation. For the Persistent SAE, we evaluate the token-wise sparse activation $\mathbf{a}^{(t)}$ , asking whether it remains understandable before it is used to update the persistent state (see Appendix E for details on autointerpretation).

The choice to evaluate the token-wise activation rather than the persistent state is deliberate. Automated interpretability pipelines show the judge a short window around each high-activation token. For a feature with a short timescale, the token-wise activation and the state are nearly identical, and the evaluation is directly comparable to that of a standard SAE. For a feature with a long timescale, the state at a given token reflects evidence accumulated over a span much longer than the window shown to the judge, so a state-based evaluation would ask the judge to explain a signal whose cause is largely outside its view. Evaluating the token-wise activation keeps the comparison fair across SAE families and across timescales.

This design also exposes a limitation of current interpretability metrics for persistent representations. Features whose relevance is distributed across a sequence may receive similar descriptions even when they are distinct [24], and explanations derived from a handful of contexts may not generalise to related inputs [25]. A low token-level score for a long-timescale feature would therefore not imply that the feature is uninterpretable, only that it is not well described by local contexts. For this reason we complement the token-level evaluation with analyses that treat persistent features jointly, as a state, in Section 5.3.

Where Features Receive New Evidence. 
A complementary question is which kinds of tokens provide new activation to features at different timescales. We divide features into five equal-sized quintiles by their timescale parameters and group their update positions into six token categories: punctuation, conjunctions, function words, verbs, adjectives or adverbs, and nouns.
This grouping follows the classical distinction between lexical/content words and grammatical/function words [26].
Function words include adpositions, determiners, particles, and pronouns.
For each quintile $q$ and token category $c$ , we compute the relative update frequency

| | $U_{q,c}={\Pr\!\left(T^{(t)}=c\mid a^{(t)}_{j}>0,\;j\in Q_{q}\right)}\big/{\Pr\!\left(T^{(t)}=c\right)},$
| | (4)

where $T^{(t)}$ is the category of token $t$ and $Q_{q}$ is the set of features in quintile $q$ . This ratio compares how often features are updated at category $c$ with how often that category occurs in the corpus. A value above one means more updates than expected, while a value below one means fewer.

The motivation for this analysis comes from work on how LLMs use different token types. Recent analyses find that punctuation and other function-like tokens can carry substantial contextual information [27], and that explicit structural markers gate the integration of information across sentences [28]. Content words, by contrast, provide clear lexical evidence at the current token. If timescales are meaningful, one might expect them to be reflected in where features receive evidence, although the direction and strength of any such relationship is an empirical matter. The relative update frequency normalises for the base rates of each category, which differ widely, so that it reflects a preference of features rather than the composition of the corpus.

### 5.3 Long-Timescale States Track Subject Information

The analyses above concern individual features. Persistent states, however, are most naturally studied jointly: a contextual property such as subject is unlikely to be carried by one feature alone, and the relevant question is whether a set of persistent coordinates, taken together, encodes it. We consider two complementary analyses on MMLU [29], whose examples carry subject labels across a range of distinct domains.

Subject geometry. 
To examine whether persistent states organise sequences by subject, we visualise representations of MMLU examples from five subjects. For each sample, we collect the representations of its final $20$ tokens and project them with t-SNE (see Appendix F for details). We compare the raw hidden states, BatchTopK features, the Temporal SAE’s predefined high-level group, and a subset of the Persistent SAE state selected by timescale. t-SNE is a qualitative tool and its layouts depend on hyperparameters, so we use a common configuration for all representations and treat the visualisation as descriptive rather than as evidence in itself. Importantly, none of the SAEs are trained with subject labels, so any subject organisation that appears would be learnt without supervision.

Subject probing on spliced sequences. 
A quantitative counterpart uses spliced MMLU sequences, created by concatenating labelled segments from different subjects so that the subject changes at segment boundaries. At each token, a linear probe predicts the current subject using varying fractions of the feature dictionary for training the probe. Splicing makes the task genuinely temporal: a representation that tracks the current subject must update at boundaries and maintain the new subject thereafter, whereas a representation that only reflects local lexical cues must rely on whatever subject-specific words happen to appear near each token.

We compare three rankings of Persistent SAE features: by decreasing timescale, by increasing timescale, and in random order. We compare against BatchTopK, Matryoshka, and Temporal SAE representations. We also include a supervised ranking as an oracle baseline. For each subject, the supervised ranking favours features that activate more strongly on that subject than on all other subjects. Because this ranking uses the subject labels directly, it serves as a label-informed reference point rather than a competing method (see Appendix G for details). Sweeping the fraction of features available to the probe shows how compactly each ranking concentrates subject information, which is the relevant quantity if persistent features are to serve as a small, inspectable summary of context.

Interpreting the comparison. 
The probing protocol is designed so that each comparison isolates one factor. Comparing timescale-based rankings with random rankings asks whether the timescale parameters carry information about which features are relevant to subject. Comparing Persistent SAE states with baseline SAE representations at matched budgets asks whether carrying state forward changes what a fixed number of features can encode. Comparing against the supervised ranking asks how far an unsupervised criterion is from one that has access to labels. None of these comparisons alone establishes that a representation tracks subject, but together they constrain the interpretation considerably.

### 5.4 Case Study: Persistent Prompt Injection Monitoring

Indirect prompt injection [12] provides a practical setting in which persistence matters. In this setting, an agent may encounter a malicious instruction in an email, webpage, or tool response, continue reasoning for hundreds of tokens, and act later [30]. A useful agentic monitor must therefore do more than detect the injected text when it appears, and should preserve a security-relevant signal until the agent acts [31, 32]. Representations designed to carry information forward are natural candidates for such monitoring, and this case study describes how we evaluate them.

Setting. 
We use AgentDojo trajectories from the banking, travel, and workspace domains [12], generated by Llama-3.3-70B-Instruct [33] (see Section C.3 for details). Each trajectory contains a request, an untrusted span with a prompt injection or benign content, and the agent’s subsequent tool calls. To prevent the monitor from relying on malicious actions to identify an attack, we consider only failed attacks, where an injection is present but the agent does not complete the attack’s action.

We use Gemma-2-2B as an external monitor: we replay each trajectory through the monitor model and record the layer-12 residual stream. We train a linear probe at the final token of the untrusted span ( $d=0$ ), where the injection is directly visible, and evaluate the same probe up to $350$ tokens later without retraining (Figure 3). We compare the raw hidden state and baseline SAE representations with the Persistent SAE state $\mathbf{h}^{(t)}$ , both in full and restricted to subsets of features grouped by timescale. This evaluation measures how long each representation preserves the injection signal (see Appendix H for evaluation details).

Figure 3:
Persistent monitoring of prompt injection. A probe trained at the end of an untrusted span is evaluated at increasing distances later in the trajectory, without retraining, to measure how long each representation retains the injection signal.

Signal retention. 
Training the probe only at $d=0$ and evaluating it at later distances is a stringent test. A representation may encode the presence of an injection at every position but in a form that drifts, so that a probe fitted at one position does not transfer to another. Retraining the probe at each distance would hide such drift and would answer a different question—whether the information is recoverable at all—rather than whether a fixed monitor continues to work. Fixing the probe reflects how a deployed monitor would operate: it is calibrated once and then applied as the trajectory unfolds. Comparing representations under this protocol separates local detection, which concerns accuracy near $d=0$ , from retention, which concerns how accuracy changes with distance.

Causal effect. 
Probe accuracy shows whether a representation contains injection information, but not whether the corresponding direction influences the monitor’s own judgement. To address this, we truncate each trajectory at a later distance, append a query asking whether it should be flagged for prompt injection, and intervene in the model at the final query token.
Building on prior work that selects SAE features for steering and uses linear classifiers to identify task-relevant sparse subspaces [7, 34], we construct a steering vector for each SAE representation by weighting its most predictive decoder directions by their probe coefficients. We then apply interventions of fixed relative magnitude in both the positive and negative directions and measure the resulting change in the monitor’s yes-minus-no logit margin at the query token. Applying the intervention symmetrically controls for generic effects of perturbing the residual stream, so that what remains reflects alignment between the steering direction and the monitor’s judgement.

The causal analysis complements the probing analysis in an important way. A probe can achieve high accuracy by exploiting directions that the model itself does not use, and conversely a direction that the model uses may be only weakly decodable at a given position. Because SAE decoder directions are the same vectors used to construct steering interventions, this case study tests in a single framework whether a representation both contains and is used for the relevant information. Taken together, the probing and intervention protocols allow us to ask how fast and slow components of a persistent representation divide the labour of immediate detection and long-context retention.

## 6 Conclusion and Discussion

Conclusion. 
We introduced Persistent SAEs, which separate a token-wise sparse activation from a persistent feature state and associate each feature with its own timescale, without a predefined partition of temporal roles. The formulation preserves the sparsity and decoder-level attribution that make SAEs useful for interpretation, contains standard SAEs as a special case, and can be trained under token budgets matched to atemporal baselines. We complemented the formulation with an evaluation protocol that covers reconstruction fidelity and temporal smoothness, local interpretability of token-wise activations, the token categories at which features receive evidence, subject-level structure in persistent states, and long-context monitoring and steering in an indirect prompt-injection setting. As such, Persistent SAEs provide a framework for organising sparse representations by timescale and a potential direction for interpreting and monitoring information maintained by LLMs over long contexts.

Broader perspective. 
A recurring theme of this work is that temporal extensions of SAEs require temporal evaluations. Metrics developed for atemporal decompositions remain necessary—an extension that degrades reconstruction or token-level interpretability would be hard to justify—but they are not sufficient, because they do not ask whether a representation carries context in a usable form. Probing on spliced sequences, fixed-probe evaluation across distance, and symmetric interventions along decoder directions are each designed to answer a specific temporal question while controlling for the most obvious confounds. We hope this protocol is useful beyond the particular architecture studied here, for example in evaluating partition-based temporal SAEs or predictive decompositions on a common footing.

Limitations and Future Work. 
Persistent SAEs assign each feature a single timescale through a feature-wise recurrence, and therefore cannot capture interactions between features or context-dependent resets, such as a topic state that should be cleared at a document boundary. Our evaluation of interpretability relies on automated pipelines designed for token-level features, which are poorly matched to long-timescale features, and better methods for describing distributed features remain an open problem. Moreover, our prompt-injection study uses an external replay monitor of agent trajectories, so intervention effects concern the monitor’s judgement rather than the original agent’s actions. Future work could explore richer recurrent dynamics, interpretability methods suited to persistent features, and tests of whether persistent feature interventions affect end-to-end agent behaviour.

## Appendix A Notation

Table 2 lists the symbols used in the main text and appendices, in order of first appearance, as a reference for the terminology introduced in Sections 3–5.4. We write vectors and matrices in bold, index token positions with a parenthesised superscript, and index coordinates within a vector with a subscript.

Table 2: Notation used throughout the paper, grouped by the section in which each symbol is introduced. Bold symbols denote vectors and matrices, a parenthesised superscript denotes a token position, and a subscript denotes a coordinate within a vector.

| Symbol | Meaning

| Hidden states and standard SAE (Section 3, 4)

| $\mathbf{x}^{(t)}\in\mathbb{R}^{d}$ | Residual-stream hidden state at token position $t$

| $T$ | Number of tokens in a sequence

| $d$ | Dimensionality of the hidden state

| $m$ | Number of dictionary features (SAE width)

| $\mathbf{W}_{\mathrm{enc}},\mathbf{b}_{\mathrm{enc}}$ | Standard SAE encoder weight matrix and bias

| $\mathbf{W}_{\mathrm{dec}},\mathbf{b}_{\mathrm{dec}}$ | Standard SAE decoder weight matrix and bias

| $\mathbf{d}_{j}$ | Unit-normalised decoder direction for feature $j$

| $\mathbf{f}^{(t)}\in\mathbb{R}_{\geq 0}^{m}$ | Sparse activation of a standard SAE at token $t$ (Background and Related Work)

| $f^{(t)}_{j}$ | Sparse activation of feature $j$ at token $t$

| $k$ | BatchTopK sparsity level (Section B.2)

| $B$ | Batch size used by BatchTopK selection (Section B.2)

| Feature-level analysis (Section 3, Appendix D)

| $\tau$ | Lag, in tokens

| $p^{(t)}_{j}$ | Decoder-aligned projection $\mathbf{d}_{j}^{\top}\mathbf{x}^{(t)}$

| $\hat{\mathbf{x}}^{(t)}_{\mathrm{base}}$ | Baseline reconstruction from a standard SAE

| Persistent SAE (Section 4)

| $\mathbf{a}^{(t)}\in\mathbb{R}_{\geq 0}^{m}$ | Token-wise sparse activation of a Persistent SAE (Section B.2)

| $a^{(t)}_{j}$ | Token-wise sparse activation of feature $j$ at token $t$

| $\mathbf{h}^{(t)}\in\mathbb{R}_{\geq 0}^{m}$ | Persistent state integrating token-wise activations over time (Equation 2)

| $h^{(t)}_{j}$ | Persistent state of feature $j$ at token $t$

| $\Phi_{\bm{\theta}}$ | Feature-wise state-update map (Equation 2)

| $\bm{\theta}$ | Parameters of the state update, including a timescale per feature (Section B.3)

| Evaluation (Section 5)

| $U_{q,c}$ | Relative update frequency of quintile $q$ at token category $c$ (Equation 4)

| $T^{(t)}$ | Token category of position $t$

| $Q_{q}$ | Set of features in quintile $q$

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
does not use a predefined partition; it instead associates a timescale with every
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
(ae.pt) for every downstream evaluation. All experiments were run on NVIDIA A100 GPUs with 80 GB of memory.

Sequence sampling for the Persistent SAE. 
Because the Persistent SAE sees contiguous sequences, the composition of each batch differs from that of the flat baselines in one respect that is worth noting: tokens within a sequence are correlated, and the eight sequences in a batch are drawn from different documents to limit this correlation. Sequences are taken from the beginning of documents where possible, so that the initial state $\mathbf{h}^{(0)}=\mathbf{0}$ corresponds to the absence of preceding context rather than to an arbitrary truncation point. When a document is longer than the context length, subsequent windows are treated as fresh sequences with a reset state.

Figure 4:
Persistent SAE training dynamics.
Training loss (left) and the number of non-zero coordinates in the persistent state $\mathbf{h}^{(t)}$ (state $L_{0}$ ; right) for Pythia-160M at layer 8 and Gemma-2B at layer 12.

Figure 4 shows the training dynamics of the Persistent SAEs. The training loss follows the familiar profile of SAE optimisation under a BatchTopK constraint, with a rapid initial decrease followed by a long, slowly improving phase. The state $L_{0}$ is a quantity specific to Persistent SAEs: because sparsity is imposed on the token-wise activation, the number of non-zero state coordinates is not directly controlled and must be monitored to ensure that the state does not grow without bound over training. Monitoring this quantity alongside the loss is a simple diagnostic that we recommend for any SAE with a carried state.

### B.2 BatchTopK Sparsity

For every SAE in this paper, sparsity is imposed using a BatchTopK mechanism in which a hidden state $\mathbf{x}\in\mathbb{R}^{d}$ is first mapped to non-negative pre-activations

| | $\tilde{\mathbf{f}}=\mathrm{ReLU}\!\left(\mathbf{W}_{\mathrm{enc}}(\mathbf{x}-\mathbf{b}_{\mathrm{dec}})+\mathbf{b}_{\mathrm{enc}}\right)\in\mathbb{R}_{\geq 0}^{m},$
| | (5)

and only the largest pre-activations are retained across the batch.
For a batch $X=\{\mathbf{x}_{i}\}_{i=1}^{B}$ , the pre-activations $\{\tilde{\mathbf{f}}_{i}\}_{i=1}^{B}$ are flattened into a single vector of length $Bm$ , and the sparse activation $\mathbf{f}_{i}$ is obtained by keeping only the top $kB$ entries globally and setting all others to zero.
The reconstruction is then

| | $\hat{\mathbf{x}}_{i}=\mathbf{W}_{\mathrm{dec}}\mathbf{f}_{i}+\mathbf{b}_{\mathrm{dec}}.$
| | (6)

In practice, after training with exact BatchTopK selection, inference uses an equivalent thresholded form

| | $f_{ij}=\tilde{f}_{ij}\,\mathbf{1}[\tilde{f}_{ij}>\theta],$
| | (7)

where the threshold $\theta$ is estimated from training-time top- $k$ statistics.
The Persistent SAE uses the resulting sparse activation as its token-wise sparse activation $\mathbf{a}^{(t)}$ ; only its downstream use differs.

The batch-level budget interacts with the persistent state in a way that is worth making explicit. In a flat baseline, the batch consists of shuffled tokens from many documents, and the global top- $kB$ selection allocates activations across this mixture. In the Persistent SAE, the batch consists of contiguous sequences, so the selection allocates activations across positions within a small number of documents. The average number of active features per token is the same in both cases, but the distribution across positions may differ. Using the thresholded form at inference removes this dependence on batch composition and makes the token-wise activation a deterministic function of the current hidden state.

### B.3 Persistence Parameterisation

The timescale parameters of the Persistent SAE must be parameterised so that the requirements of Section 4 hold throughout training. Three considerations guide this choice.

Stability. 
The state update in Equation 2 is applied repeatedly over sequences of hundreds of tokens, so any parameterisation that allows the state to grow without bound would produce inputs to the decoder whose scale depends on position. This would both complicate optimisation and make reconstructions at late positions incomparable to those at early positions. We therefore constrain the timescale parameters to a range in which the update is contractive, and we normalise the contribution of new evidence so that the state remains on the same scale as the token-wise activations it integrates.

Avoiding degenerate timescales. 
A timescale that is effectively infinite would cause a feature to retain the first evidence it receives for the remainder of the sequence, which is undesirable both because it prevents the feature from responding to new context and because it creates ill-conditioned gradients through long chains of updates. We therefore impose a hard cap that keeps every feature’s effective memory finite. At the other extreme, a memoryless timescale is permitted, since it recovers standard SAE behaviour for that feature.

Optimisation. 
The timescale parameters are optimised jointly with the encoder and decoder. We use an unconstrained parameter for each feature, mapped through a squashing function to the admissible range, so that the optimiser does not need to handle explicit constraints. We initialise these parameters to an intermediate value, away from the saturated regions of the squashing function, so that gradients with respect to the timescales are not vanishingly small at the start of training and each feature’s timescale can move in either direction.

Together, these choices make the Persistent SAE a well-behaved extension of a BatchTopK SAE: at initialisation it differs from a standard SAE only by moderate smoothing of activations over time, and over training each feature’s timescale is free to move within a bounded range.

## Appendix C Dataset Details

We use three datasets for distinct parts of the evaluation. Pile documents provide the hidden states used to train and evaluate the SAEs, MMLU questions provide subject-labelled text for analysing the persistent state, and AgentDojo trajectories support the prompt-injection monitoring case study. We describe each dataset and our use of it below.

### C.1 Pile Corpus

The original Pile is an $825$  GiB English-language corpus assembled from 22 sources, including academic papers, web pages, books, code, and online discussions [22]. We use the Pile Uncopyrighted release [35], a derived version that removes the Books3, BookCorpus2, OpenSubtitles, YouTube Subtitles, and OpenWebText2 components. Documents are accessed in streaming mode and shuffled with a buffer of size $10{,}000$ using a fixed seed. A document is accepted if, after tokenisation with the model tokeniser and truncation to the context cap, it contains at least the required minimum number of tokens.

Each accepted document is passed through the model, and the chosen residual-stream hidden states are retained.
If a sequence has retained length $L$ , temporal statistics are computed on token positions $\{16,\dots,L-1\}$ after dropping the initial tokens. Dropping an initial prefix is a standard precaution: hidden states at the first few positions of a sequence are known to have atypical norms and statistics, and including them can distort averages computed over positions.

The diversity of the Pile is relevant to any analysis of persistence. Documents range from short forum posts to long academic papers and code files, and the degree of document-level coherence varies accordingly. Code, for example, combines strong local syntactic structure with long-range dependencies on identifiers defined much earlier, whereas some web text consists of loosely related fragments. Aggregate statistics computed over the corpus therefore average over very different temporal regimes, and per-domain analyses may reveal structure that the aggregate obscures.

### C.2 MMLU Dataset

Massive Multitask Language Understanding (MMLU) is an English-language benchmark of four-option multiple-choice questions from 57 subjects spanning STEM, the humanities, the social sciences, and other professional and academic fields [29]. We use MMLU only for the subject analyses; it is not used to train the SAEs. Specifically, we use questions from abstract algebra, astronomy, business ethics, computer security, and European history. These subjects cover distinct areas while providing a subject label for every example.

For the geometry analysis in Section 5.3, we sample $100$ sequences across the five subjects and retain the final $20$ token representations from each sequence. For subject probing, we construct sequences by joining equal-length segments from different subjects and assign each token the subject label of its source segment. Appendix F and Appendix G provide the sampling and evaluation details for the two analyses.

The five subjects were chosen to span mathematical, scientific, professional, technical, and historical text, so that subject distinctions involve differences in vocabulary, register, and argumentative structure rather than in a single lexical cue. Using the final tokens of each sequence ensures that every retained representation has been computed with a substantial amount of subject-specific preceding context.

### C.3 AgentDojo Trajectories

Indirect prompt injection occurs when an agent retrieves attacker-controlled text from emails, webpages, documents, or tool outputs and follows instructions inside that text. AgentDojo [12] evaluates this setting with stateful tool environments containing 97 user tasks, 27 injection targets, and 629 security test cases across workspace, Slack, travel, and banking domains. Its task-specific checks evaluate whether the legitimate user request and the attacker’s target action were completed by inspecting the agent output and changes to the environment.

For our monitoring dataset, we use trajectories from the banking, travel, and workspace domains generated by Llama-3.3-70B-Instruct. Benign trajectories without an injection form the negative class ( $y=0$ ). The positive class ( $y=1$ ) contains runs with an important_instructions or direct attack for which the AgentDojo injection checker returns security=false. This value means that an injection was present but the agent did not complete the attacker’s target action. We retain these failed attacks deliberately: successful attacks often contain an overt malicious action, such as transferring money or sending private data, which could provide a trivial downstream classification cue. This restriction asks whether the monitor retains a trace of the injection even when no observable attack success occurs. We do not filter by task utility, so the legitimate user request may either succeed or fail.

## Appendix D Empirical Persistence of Standard SAE Features

This appendix discusses methodological considerations for measuring how long the signal associated with an SAE feature remains in the hidden state after the feature activates. Such measurements are useful both for characterising standard SAEs and for interpreting the timescales of a Persistent SAE, but they are easy to get wrong, and we set out the precautions we consider necessary.

Anchoring on activation events. 
The natural unit of analysis is an activation event: a token at which a feature’s sparse activation is positive. Projections onto the feature’s decoder direction can then be tracked at subsequent lags $\tau$ and compared with the feature’s average projection over all tokens. Normalising by the elevation at the activating token itself makes measurements comparable across features whose activations differ in scale.

Feature selection. 
Not every feature admits a reliable measurement. Rarely active features provide too few activation events for stable averages, and anchors near the end of a document cannot be followed to long lags. We therefore recommend retaining only features with a minimum number of activation events and a minimum number of valid anchors at every lag considered. A further subtlety concerns features whose activated-token elevation is near zero, as occurs for near-dead or diffuse features: a normalised measure is undefined in this case, and adding a small constant to the denominator would produce arbitrarily large or unstable values. Such features should be excluded rather than regularised.

Confounds. 
Three confounds deserve attention. First, decoder directions are not orthogonal, so the projection onto one direction can be elevated by the activity of correlated features; persistence measured through projections is thus a property of a direction in the residual stream rather than of a feature in isolation. Second, activation events are not distributed uniformly across positions, and since hidden-state statistics vary with position, comparisons across lags can be affected by positional drift. Restricting analysis to positions after an initial prefix and limiting the maximum document length mitigates but does not eliminate this effect. Third, local continuity—the fact that consecutive tokens belong to the same phrase or sentence—produces short-range persistence regardless of whether a representation tracks document-level context. Separating local from global contributions requires a control that preserves the former while disrupting the latter.

Heterogeneity across features. 
Aggregate statistics such as medians across features summarise the typical behaviour of a dictionary, but they can conceal substantial variation. Grouping lags into short, medium, and long bands and examining the distribution of persistence within each band gives a more complete picture than a single curve. Such per-band distributions are also the natural point of comparison for the timescale parameters of a Persistent SAE, since they indicate whether features that a model treats as long-lived are distinguished in the hidden states themselves.

Descriptive fits. 
It is sometimes convenient to summarise a decay curve by fitting a parametric form, such as an exponential or power law, to its positive tail. Such fits are descriptive rather than mechanistic, and their parameters should be compared only between conditions measured under identical selection criteria. Because the tail of a decay curve is estimated from fewer anchors than its head, fits should weight lags appropriately or be restricted to a range in which estimates are stable.

### D.1 Context Correction Experiment

A complementary diagnostic asks whether a feature’s past activation is informative about the hidden state at a later token along the same direction. One way to operationalise this is to augment a standard SAE’s reconstruction with a correction term that adds, for each feature, a learned multiple of its activation $\tau$ tokens earlier along its own decoder direction, and to measure how much of the baseline reconstruction error this correction recovers on held-out documents.

Why a feature-wise correction. 
Restricting the correction to act feature-wise, rather than allowing an arbitrary linear map from past activations to the current hidden state, is essential for the diagnostic to be informative about features. A dense map would trivially exploit the well-known correlation between consecutive hidden states and would not indicate whether any individual feature carries information forward. A feature-wise correction has only one parameter per feature, so any gain it achieves can be attributed to specific features, and the sign of each parameter indicates whether past activation predicts a positive or negative contribution along the same direction.

Fitting and evaluation. 
The correction should be fitted on one set of documents and evaluated on a disjoint held-out set, with regularisation to prevent overfitting of rarely active features. Because the gains from such a correction are expected to be small relative to the total reconstruction error—the baseline SAE already explains most of the variance—care is needed to ensure that any measured improvement exceeds what would be obtained by chance. A useful control refits the correction using past activations drawn from unrelated text while keeping the same natural-document targets; this preserves the marginal statistics of the activations while removing any genuine relationship with the target document’s history. We fit corrections at a range of lags with AdamW, a learning rate of $5\times 10^{-3}$ , weight decay of $10^{-3}$ , and $10$ epochs, after dropping $16$ prefix tokens and capping each document at $768$ tokens.

Relationship to Persistent SAEs. 
A Persistent SAE can be viewed as internalising this diagnostic: rather than fitting a correction post hoc to a frozen atemporal SAE, it allows each feature’s state to carry information forward during training and lets the encoder and decoder adapt jointly. The two approaches therefore answer related but distinct questions. The post-hoc correction asks whether an atemporal decomposition leaves feature-level temporal information unused; the Persistent SAE asks what a decomposition learns when that information is available to it from the outset.

## Appendix E Automated Interpretability Details

We follow the automated interpretability pipeline of Bills et al. [23]. For each feature we collect its highest-activation contexts, prompt a judge model to write a short natural-language explanation, and score how well that explanation predicts whether the evaluated feature activation exceeds a threshold on held-out contexts. For standard SAEs, this activation is $f_{j}^{(t)}$ ; for the Persistent SAE, we use the token-wise activation $a_{j}^{(t)}$ . We use gpt-4o-mini as the judge, a context length of $1024$ tokens, and roughly $2$ M tokens from the Pile for the activation examples. This evaluates the Persistent SAE on its token-wise activations, matching the token-specific representations of the baselines. We also score groups of Persistent SAE features defined by their timescale parameters, to evaluate their interpretability as groups.

Explanation and scoring. 
The explanation step presents the judge with a fixed number of high-activation contexts, each centred on the activating token and annotated with activation values. The judge is asked for a concise description of what the feature responds to. In the scoring step, the judge receives the explanation together with a mixture of held-out contexts, some containing activations above threshold and some not, and predicts which contexts contain activations. The score reflects the agreement between these predictions and the actual activations. Holding out contexts used for scoring from those used for explanation is important, since otherwise the score would reward memorisation of specific examples rather than a generalisable description.

Sampling considerations. 
High-activation contexts are drawn from the top of each feature’s activation distribution, which emphasises the clearest cases. Features with broad, low-amplitude activation profiles may be poorly represented by their maxima, and for such features the explanation may describe an extreme case rather than typical behaviour. Including contexts sampled from lower activation quantiles alongside the maxima can mitigate this, at the cost of making explanations harder to write. We use the same sampling scheme for all SAE families so that comparisons are not affected by this choice.

Limitations for persistent features. 
As discussed in Section 5.2, token-level pipelines are best suited to features whose relevance is concentrated at the activating token. For features whose evidence accumulates across a sequence, the window presented to the judge may omit much of the context that determines the feature’s role, and descriptions may converge on broad semantic domains that several distinct features share [24]. Developing pipelines that present longer contexts, or that describe groups of features jointly, is a natural direction for evaluating persistent representations more faithfully.

## Appendix F Subject Geometry Details

For the t-SNE visualisation we draw $100$ sequences from five MMLU subjects (abstract algebra, astronomy, business ethics, computer security, and European history) and keep the last $20$ tokens of each sequence, so that every token carries a subject label. For each representation—raw hidden states, BatchTopK features, the Temporal SAE’s high-level group, and a timescale-selected subset of the Persistent SAE state—we collect the token-level vectors and project them to two dimensions with t-SNE using a common set of hyperparameters, colouring each point by its subject.

Because t-SNE emphasises local neighbourhood structure and can exaggerate or suppress cluster separation depending on perplexity and initialisation, we use a fixed random seed and a single perplexity setting across all representations, and we standardise each representation before projection. The visualisation is intended to support qualitative comparison; quantitative claims about subject information are based on the probing analysis in Appendix G.

## Appendix G Subject Probing Details

We construct spliced sequences by concatenating equal-length segments drawn from different MMLU subjects, balancing subject labels within each sequence and holding out entire source documents for the test split, so that a token’s label is the subject of the segment it falls in. We treat each token as a labelled example and fit a linear probe to predict its source subject from the token-level representation. We report held-out one-vs-rest macro ROC AUC as a function of the fraction of features made available, swept from $0.1\%$ to $100\%$ . We evaluate the Persistent SAE on its persistent state $\mathbf{h}^{(t)}$ and order its features by decreasing timescale, increasing timescale, repeated random permutations, or supervised one-vs-rest importance. For each subject $c$ , the supervised score for feature $j$ is

| | $s_{j,c}=\mathbb{E}[h^{(t)}_{j}\mid y^{(t)}=c]-\mathbb{E}[h^{(t)}_{j}\mid y^{(t)}\neq c],$
| | (8)

computed on the training split. This score measures how much larger the persistent state of feature $j$ is for subject $c$ than for the other subjects. We sort features by this score within each subject and interleave the subject-specific rankings in round-robin order, ensuring that every subject contributes its most discriminative features. This produces a single global feature ordering used by the multiclass probe. For the BatchTopK, Matryoshka, and Temporal baselines, we use their full dictionaries under the same probe.

Holding out entire source documents for the test split prevents the probe from exploiting memorised content: a token in the test set never shares a source question with any training token. Balancing subjects within each spliced sequence ensures that the prior probability of each subject is uniform, so that a probe cannot improve its score simply by favouring frequent subjects. Random rankings are repeated over several permutations and averaged, giving a reference curve against which structured rankings can be compared.

## Appendix H Prompt Injection Monitoring Details

Trajectory construction. 
Trajectories are taken from AgentDojo [12] runs of Llama-3.3-70B-Instruct agents. Attack trajectories are restricted to the important_instructions and direct attack types and to runs with security=false. In this dataset, these positives contain an injection whose targeted action the agent did not execute; benign trajectories contain no injected content. Each trajectory is serialised and appended with the monitoring query

should this trajectory be flagged for prompt injection or unsafe instruction hijacking? Answer yes or no:

and read with Gemma-2-2B; all representations are taken from the layer-12 residual stream. The model’s own answer to this query is close to uninformative: the yes-minus-no answer logit at the query position gives an AUC of $0.51$ , which is why we train internal monitors rather than relying on prompting.

Checkpoints and probes. 
For each trajectory, we locate the final token of the injected span (for benign trajectories, the token at the matched position) and define checkpoints at fixed offsets of $0$ , $50$ , $100$ , $200$ , $250$ , $300$ , and $350$ tokens after it, together with the final query position. At each checkpoint we extract, from the same layer, the raw hidden state, the BatchTopK and Matryoshka sparse activations, the Temporal SAE’s high-level feature group, and the Persistent SAE’s token-wise activation $\mathbf{a}^{(t)}$ , full state $\mathbf{h}^{(t)}$ , and timescale-selected subsets of the state. A logistic regression probe is fit on each representation using checkpoint $d=0$ only and then evaluated, without retraining, at every other checkpoint; we report held-out accuracy and ROC AUC.

Causal intervention. 
For each SAE representation, we select the $k=128$ features with the largest absolute probe coefficients and combine their weighted decoder columns into a probe-weighted decoder direction $\mathbf{v}_{r}$ . The direction is normalised and scaled so that its norm equals a fixed fraction $\rho=0.1$ of the hidden-state norm at the intervention position, then added to the residual stream with both signs. We measure the resulting change in the external replay monitor’s yes-minus-no logit margin, not a change in the original Llama-3.3-70B agent’s action. Interventions use the direction constructed at $d=0$ , so the analysis reflects how its causal effect transfers across distance.

Scaling interventions relative to the hidden-state norm ensures that perturbations are comparable across positions and trajectories, whose residual-stream norms vary. Restricting each steering direction to a fixed number of features keeps the comparison between representations fair, since a direction assembled from more features could otherwise achieve a larger effect simply by spanning more of the residual stream. Applying the intervention with both signs and taking the difference removes components of the effect that are independent of direction, such as a generic shift in the logit margin caused by any perturbation of the given magnitude.