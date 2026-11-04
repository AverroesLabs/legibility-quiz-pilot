# Probe Generalization as Subspace Selection for OOD Deception Detection

Daniel Yoo

  
Adrians Skapars

###### Abstract

Linear probes can be used to detect behaviors and concepts inside language model activations, but may fail to transfer to out-of-distribution examples. We study the generalization of Llama-3.1-8B-Instruct deception probes trained on a single source dataset and evaluated on 3 held-out deception detection datasets, and we propose to analyze this problem in the basis of principal components (PCs) of the source-domain activations rather than in the raw activation space. Within this basis, we lay out a set of complementary subspace-selection procedures: a target-guided greedy search that serves as a reference point, rankings derived from the standardized weights of source-trained and target-trained probes, and an interpretability-driven procedure in which an LLM judge scores natural-language descriptions of each PC's most and least activating examples for their likely relevance to out-of-distribution deception. We describe each procedure, the controls we pair it with, and the practical considerations involved in applying it, including how to standardize target data, how to handle ambiguous labels, and how to reduce noise in judge scores. Broadly, we argue that the out-of-distribution robustness of probes is usefully studied as a question of subspace selection, and that interpretable bases such as source PCs make this question tractable to investigate.

Machine Learning, ICML

## 1 Introduction

Linear probes are standard tooling for reading information from language model activations (Alain and Bengio, 2018; Hewitt and Liang, 2019).
Linear probes are cheap to train and expose some semantically meaningful information in the otherwise uninterpretable representation space.
However, probes that look strong in-distribution are sometimes unreliable out-of-distribution (Kirch et al., 2026). Existing work shows that key behavioral concepts are linearly represented in model activations (Marks and Tegmark, 2024; Park et al., 2024; Zou et al., 2025), so probes should have the capacity to detect these concepts, yet they appear to learn spurious correlations in practice (Orgad et al., 2025; Wang et al., 2025). This is unacceptable for probe-based monitoring of harmful interactions, setting a clear challenge:
how can we build probes that only learn generalizing features?

To tackle this, we work in the basis of source-domain principal components (PCs) rather than raw activations, avoiding the assumption that the model's latent dimensions provide a privileged basis. Principal components offer a simple, data-driven way to study structure in representation spaces, and prior work even suggests that the choice of PCs matters: Mu et al. (2018) shows that removing the top PCs of an embedding space improves downstream performance, and Kantamneni et al. (2025) use top-PC projections as a probe baseline. Thus, the focus of our investigation becomes: which subset of principal components supports generalizable probe training and how can that subset be identified?

The PC basis has several attractive properties for this question. First, it is computed entirely from source data, so any selection procedure defined over it can in principle be applied before a deployment distribution is known. Second, each PC is a single direction with a well-defined positive and negative pole, so its behavior can be inspected by looking at the examples that activate it most strongly in either direction. Third, a subset of PCs defines an orthogonal subspace, which makes it straightforward to train, compare, and ablate probes restricted to different subsets while keeping all other parts of the pipeline fixed.

We use deception probes on Llama-3.1-8B-Instruct (Grattafiori et al., 2024) as our testbed,
training them on a single source dataset (Roleplaying deception scenarios) and evaluating on 3 held-out target datasets (Insider Trading Report, Insider Trading Confirm, and Sandbagging) (Goldowsky-Dill et al., 2025).
Working in the source-domain PCA basis, we explore strategies for selecting a subset of training dimensions that avoid spurious correlations and transfer to target domains.

Deception is a natural testbed for this kind of study. The behavior of interest, a model knowingly asserting something it believes to be false, is conceptually stable across settings, but the surface form in which it appears varies enormously: a roleplayed excuse, a misleading trading report, and a deliberately incorrect answer on a capability evaluation share little vocabulary or structure. A probe that latches onto the vocabulary of one setting will therefore fail on the others, while a probe that captures the underlying contrast should not. This makes the gap between in-distribution and out-of-distribution behavior especially visible.

This work contributes the following:

•

A subspace-selection framing of probe transfer (Section 4.2).
We formulate cross-domain probe transfer in terms of which source PCs a probe is permitted to use, and define a target-guided greedy search over source PCs as a reference procedure for this question.

•

Weight-based selection rules
(Section 4.3). We describe how to rank PCs by the standardized weights of source-trained or target-trained probes, giving two simple hypotheses about which directions might be relevant for transfer.

•

An interpretability-driven selection rule that uses no target data (Section 4.5).
We describe a procedure in which an LLM judge interprets each PC and rates how likely it is to encode a transferable behavioral contrast rather than surface patterns, and we use these ratings to define a candidate probe subspace.

Figure 1: Overview of PC-based subspace selection for OOD deception probing. Activations are projected onto the source-domain PCA basis, candidate PC directions are filtered, and probes are trained on the selected low-dimensional subspace.

## 2 Related Work

#### Linear concept directions in activation space.

A related line of work studies whether high-level behavioral or semantic properties are represented as directions or low-dimensional subspaces in activation space. Marks and Tegmark (2024) report linear structure in true/false representations across datasets, while Park et al. (2024) formalize the linear representation hypothesis and connect it to both probing and steering. Similarly, Zou et al. (2025) model high-level properties such as honesty and emotion as population-level directions that can be read from and intervened on in model activations.

This perspective has also been used to explain and control model behavior. (Arditi et al., 2024) show that a single direction can mediate refusal behavior, suggesting that some high-level behaviors may be localized in simple linear features. Lu et al. (2026) similarly identify an "assistant" axis corresponding to assistant-like behavior. Together, these works suggest that linear structure can support both mechanistic understanding and activation-space interventions, while also raising questions about when such directions capture stable concepts rather than context-dependent correlations.

#### Linear probes and their limitations.

Linear and lightweight probes have a long history in interpretability as tools for analyzing what information is represented in neural networks (Alain and Bengio, 2018; Hewitt and Liang, 2019). Recent work has applied such probes to safety-relevant signals in LLM activations, including latent truthfulness without direct supervision (Burns et al., 2024) and strategic deception (Goldowsky-Dill et al., 2025).

At the same time, methodological skepticism is central to the probing literature. Hewitt and Liang (2019) argue that high probe accuracy alone is insufficient evidence that a model already represents the target property, since the probe may itself learn the task or exploit surface-level correlates. Similarly, Ravichander et al. (2021) question whether probing reliably establishes what information is encoded or merely recoverable. Recent work also finds that probes with strong in-distribution performance can generalize unevenly out of distribution (kramár2026buildingproductionreadyprobesgemini). These concerns motivate careful evaluation of whether safety probes identify robust internal signals rather than dataset-specific artifacts.

A recurring theme across this literature is that the quality of a probe cannot be judged from in-distribution accuracy alone. Control tasks, selectivity measures, and held-out evaluation sets have all been proposed as ways of separating what a representation encodes from what a sufficiently flexible classifier can extract. Our setting adds a further dimension to this concern: even when the probe is a simple linear model, the particular direction it learns may combine signal that is genuinely about the behavior with signal that merely co-occurs with the behavior in the training data.

#### Interpreting features with natural-language explanations.

Another approach aims to interpret internal representations by assigning natural-language descriptions to neurons or learned features. Bills et al. (2023) introduced the use of LLMs to generate explanations of neurons from high-activation examples. This approach has since been extended to the automatic labeling of features in sparse autoencoders ((Paulo et al., 2025), (Bricken et al., 2023)), enabling large-scale interpretation of learned feature dictionaries.

Recent work has further explored whether natural-language feature descriptions can be used not only for interpretation but also for intervention. For example, Casademunt et al. (2025) identify concepts through automated natural-language interpretation and ablate them during fine-tuning to reduce misaligned behavior. Our interpretability-driven procedure follows this line of work, applying automated explanation to PCs rather than to neurons or dictionary features, and using the resulting descriptions to decide which directions a probe should be allowed to use.

## 3 Setup

#### Datasets.

Following Goldowsky-Dill et al. (2025), we use 4 of their deception detection datasets (Roleplaying, Insider Trading Report, Insider Trading Confirm, Sandbagging), with descriptions being provided in Appendix A.

Roleplaying is our source domain and the other 3 are target test sets. Each response is labeled as honest (0), ambiguous (-1), or deceptive (1) by GPT-4o (OpenAI, 2024), with the ambiguous responses being removed for our experiments.

The three targets span a range of distances from the source. Insider Trading Report and Insider Trading Confirm share a scenario with each other but not with Roleplaying, and differ in whether the deceptive act is an initial omission or a later explicit confirmation. Sandbagging differs from the source both in topic and in the form deception takes, since the dishonesty consists of deliberately wrong answers rather than misleading statements about events.

#### Language models.

Following Ying et al. (2026), we use layer 15 residual stream activations of Llama-3.1-8B-Instruct (Grattafiori et al., 2024), the layer with the highest cross-domain AUROC in their analysis. To extract activations, we pass each prompt through the model and average token activations over just the assistant's response.

#### Probe training and evaluation.

We standardize source activations to zero mean and unit variance, then apply the same source statistics to standardize all target activations. We run PCA on the scaled source activations and retain the top 100 PCs. All probes are logistic regression classifiers ( $\lambda=10$ ); for experiments using a subset $S$ of these 100 PCs, activations are projected onto $S$ before training and evaluation, while the baseline probes train directly on the full standardized activations.

Using source statistics for standardizing the targets is deliberate. Re-standardizing each target with its own statistics would leak distributional information about the target into the representation and blur the distinction between source-only and target-informed procedures. Fixing the transformation on the source keeps every selection rule comparable in what it is allowed to see.

We use the top 100 PCs because alignment with SAE decoder features (Bricken et al., 2023) decreases with PC index (Appendix C), meaning lower-index PCs are more amenable to natural-language interpretation and hence comparable with interpretability-driven selection methods. This is not a claim that low-index PCs generalize better; it is a choice made so that every direction considered by the selection procedures below is one that can plausibly be described in natural language.

## 4 Experiments

### 4.1 Setting a baseline

Figure 2: Probe configurations used as reference points: source-trained probes on full representations, source-trained probes on the top-100 PCs, and target-trained probes on the top-100 source PCs (5-fold CV).

Our reference point is a probe trained on the full standardized source representations and evaluated directly on each target. This is the configuration most commonly used in practice, and any subspace-selection rule should be judged by whether it improves on it.

We pair this baseline with two further configurations. The first is a source-trained probe restricted to the top-100 source PCs, which isolates the effect of the dimensionality reduction itself from the effect of later selection steps. The second is a probe trained on target data within the same 100-PC subspace using 5-fold cross-validation. This configuration is not a deployable method, since it requires labeled target examples, but it is useful for interpretation: it indicates how much target-relevant signal the subspace contains when the probe is free to place its weight wherever the target data suggests.

Taken together, these three configurations define the space within which source-only selection rules operate. A selection rule can only reweight or discard directions that already exist in the 100-PC basis, so comparisons between source-trained and target-trained probes within that basis are the natural frame for the procedures that follow.

### 4.2 Showing a transferable subspace exists

Figure 3: Configurations compared in the subspace-existence analysis: the full-representation baseline, top-k PCs by variance (k matched to greedy), and greedy oracle PCs.

The central question of this section is whether a subset of source PCs can be found on which a source-trained probe transfers well, and how such a subset might be characterized.
To study this, we introduce a greedy algorithm for PC selection.
Starting from an empty set, we iteratively add PCs from the source dataset.
At each step, we search over the top 100 PCs and add the one that most improves AUROC on the target.
To evaluate each candidate, we project source activations onto the current subset and train a logistic regression probe on those projections.

Probe weights are always fit on source data, but the choice of PC subsets is guided by performance on labeled examples from the target dataset.
Using 5-fold cross-validation over the target dataset, we perform greedy PC selection on the non-held-out target folds and evaluate the selected source-trained probe on the held-out target fold. We stop when no candidate PC improves AUROC by more than 0.001.

The greedy probe should be read as an upper bound on transfer performance, not an unbiased OOD estimate, because subset selection uses some target examples. Its role is diagnostic: it asks how well a source-trained probe could do if the right directions were chosen, independently of whether those directions could be identified without target data.
To ground this reference point, we compare against simply selecting the top $k$ PCs by variance, where $k$ matches the number of PCs selected by the greedy algorithm. This control tests whether any advantage of a selected subset is attributable merely to its size or to the amount of source variance it captures, rather than to the identity of the chosen directions.

Greedy forward selection has well-known weaknesses. It can miss subsets whose members are individually unhelpful but jointly useful, and its output depends on the stopping threshold. We accept these limitations because the procedure is simple, its behavior is easy to inspect, and the subsets it returns are small enough to be examined direction by direction in later sections.

### 4.3 Testing hypotheses for transferable principal components

Figure 4: PC-selection rules compared in this section: ranking by source-probe weights, by target-probe weights, and greedy oracle selection.

What property might make a PC transferable, and can that property be detected from the source side alone?
To approach this, we consider 2 simple hypotheses about what a transferable subset would contain.
The first posits that the transferable PCs are simply the ones that separate the target classes well by themselves across the test set, so that a target-guided procedure would pick them because it has access to target data.
The second posits that they are the PCs the source probe already leans on most heavily, so that a target-guided procedure would merely be rediscovering the source signal in a more compact form.

We compare 2 rankings of PCs, both based on the standardized coefficient (the absolute value of each probe coefficient multiplied by the standard deviation of the corresponding PC) given to each PC by a probe trained on activations projected on the source-PCA basis.

The 'source-PCA probe' ranks PCs by the weights learned by a probe trained on source data, while the 'target-PCA probe' ranks PCs by the weights learned by a probe trained on target data. In both cases, the activations are projected onto the source-PCA basis.
In both cases, the probes are then trained on source data that has been projected onto the top $k$ PCs, selected according to their respective rankings.
Each hypothesis makes a concrete prediction: if it were correct, the corresponding ranking should behave like the greedy reference.

To set $k$ , we run the greedy method on the full target dataset rather than per fold, giving a single stable PC set whose size determines $k$ for each target. Because this uses the full dataset rather than cross-validation, the resulting subset is not identical to the per-fold subsets described in the previous section.

The two hypotheses are worth distinguishing because they suggest different remedies. If transferable directions were simply the most target-discriminative ones, improving transfer would be a matter of acquiring a small amount of target data. If they were the most source-discriminative ones, the problem would lie in regularization or in the compactness of the representation. Neither remedy addresses the possibility that transfer depends on a property of directions that is not captured by discriminability on either domain, which motivates the interpretability analysis that follows.
Further details on the PC sets involved are given in Appendix D.

### 4.4 Interpreting the transferable subspace

A complementary way to characterize candidate directions is to ask what each one encodes in human-interpretable terms.
To do this, we adopt the interpretability method of Bills et al. (2023) and Casademunt et al. (2025).

For each of the top 100 source PCs, we find the 8 most and 8 least PC-activating examples from the source data, before prompting an LLM to generate a natural-language description of the positive pole, the negative pole, and the PC direction as a whole.
The LLM judge then rates the generalizability of each direction on a 1–10 scale (OOD score), with the full judge prompt and model detail being provided in Appendix B.

The judge prompt is designed to push against two failure modes. The first is overgeneralization, in which a direction with a narrow and specific pattern is described in broad terms such as sentiment or honesty. The second is overrating surface cues, in which a direction that tracks lexical habits or task-specific phrasing is scored as generally useful because it happens to correlate with deception in the source examples. The prompt therefore asks for the narrowest accurate description and reserves high scores for directions that seem tied to a general strategy rather than to local wording.

Because the judge sees only source examples, its scores are available before any target data is collected. This is the property that distinguishes the interpretability procedure from the selection rules of the previous two sections, and it is what makes it a candidate for a deployable selection method rather than a diagnostic one.
Examples of the judge's output format are discussed in Appendix E.

### 4.5 Selecting a subspace by interpreting PCs

Figure 5: Configurations compared for interpretability-based selection: the interp-selected subspace versus three controls, namely the full-representation baseline, the top PCs by variance with matched size, and the remaining PCs after ablating the interp-selected PCs.

Building on the OOD scores, we define a probe subspace consisting of the PCs that received the highest observed score from the judge, and train a source probe on activations projected onto this subspace. No target data is used at any point in this procedure.
Because LLM judge scores are noisy, we run the scoring twice and only keep PCs that received the highest score in both runs. Requiring agreement across runs trades recall for precision: some useful directions may be discarded, but the retained set is less likely to include directions that were scored highly by chance.

We evaluate the interp-selected probe against three controls. The first is the full-representation baseline from Section 4.1. The second is a probe trained on the same number of top source PCs by variance, which tests whether the selected subspace behaves differently from an arbitrary low-dimensional projection of the same size. The third is a probe trained on the complement of the selected PCs within the top 100, which tests whether the directions removed by the selection are themselves the ones that carry transferable signal.

These controls are designed to rule out the most obvious alternative explanations for any difference between the interp-selected probe and the baseline. Dimensionality reduction alone, a preference for high-variance directions, and the mere removal of directions are each addressed by one control. The greedy reference from Section 4.2 provides an additional point of comparison, indicating how much room for improvement exists on each target for any source-trained probe within the 100-PC basis.

We expect the informativeness of these comparisons to vary across targets. Where the baseline already transfers well, there is little headroom for any selection rule to distinguish itself. Where the target is far from the source, as with Sandbagging, the comparisons are more demanding and more revealing of whether a selection rule has captured something general.

## 5 Limitations

Our framework is narrow in scope and there are several limitations that should be considered by the reader.

Single model and source dataset.
All experiments use a single model (Llama-3.1-8B-Instruct, layer 15) and a single source dataset (Roleplaying (Goldowsky-Dill et al., 2025)). Moreover, one target, Insider Trading Confirm, is relatively easy for the baseline, which limits how much it can discriminate between selection rules. Whether the framework behaves similarly for other model families or other datasets is an open question we leave to future work.

Distribution shift varies by target.
Sandbagging is the most out-of-distribution target, and any selection rule restricted to source PCs is limited by how well the source basis covers the directions relevant to that target. Conclusions drawn from a single distant target should be treated with caution, and a broader collection of targets at varying distances from the source would make the framework more informative.

Linear pipeline. Our probe pipeline is fully linear (PCA projection followed by logistic regression), so the subspaces it studies are linear by construction. Whether the same framing extends to nonlinear probes or other readout locations remains an open question.

Judge reliability.
The OOD score depends on a single judge model and a single prompt. Different judges, prompts, or numbers of exemplars could produce different rankings, and our two-run agreement filter mitigates only some of this variability.

No causal validation.
Any structural claims about what a PC "encodes" are based on top-activating examples and an LLM judge's verbalization of the contrast.
These are descriptive claims, not causal ones. We have not ablated PCs during inference to verify that they carry the contrast they appear to encode.

## 6 Conclusion

We have presented a framework for studying out-of-distribution deception probing in terms of subspace selection over source-domain principal components. Within this framework we defined a target-guided greedy reference procedure, two weight-based ranking rules corresponding to simple hypotheses about transferable directions, and an interpretability-driven rule in which an LLM judge scores natural-language descriptions of each PC without access to target data, together with controls that isolate the effects of dimensionality, variance, and ablation.

A key motivation is that robust probing requires identifying not only whether a behavioral signal is linearly accessible, but where in activation space that signal is represented. Linear probes trained over full representations may combine general behavioral features with dataset-specific surface cues. These heuristics can support high in-distribution accuracy while hurting transfer. Sparse subspaces of interpretable directions offer a way to make this distinction explicit. This frames OOD probing as a representation-selection problem: the goal is not only to fit a classifier, but also to choose a subspace that preserves the intended behavioral feature and filters out spurious correlation. We hope that the procedures described here provide a useful starting point for future work along these lines.

## Acknowledgment

This work was conducted during the 11th edition of AI Safety Camp, which provided the compute to run our experiments. Thanks to Prasad Mahadik and Arya Hariharan for useful discussions around this work.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none
which we feel must be specifically highlighted here.

## Appendix A Dataset Details

Roleplaying: the model is placed in a simple scenario with an opportunity to lie.

Insider Trading Report: acting as a trader, the model decides whether to honestly report that it used insider information.

Insider Trading Confirm: after hiding earlier use of insider information, the model is asked to confirm it did not use any.

Sandbagging: the model is told to deliberately under-perform on some tasks while behaving normally on others.

| Dataset | Total | Label Counts

| Examples | (-1 / 0 / 1)

| Roleplaying | 371 | 140 / 59 / 172

| Insider Trading Report | 473 | 218 / 194 / 61

| Insider Trading Confirm | 151 | 29 / 38 / 84

| Sandbagging | 1000 | 379 / 375 / 246

Table 1: Dataset label distributions.

## Appendix B LLM Judge Prompt

We use GPT-5-mini (OpenAI, 2025) to interpret directions.

You are interpreting a principal component direction in a neural representation space.
This direction is bipolar:

- Positive activations mean movement toward one pole.

- Negative activations mean movement toward the opposite pole.

- Larger magnitudes mean a stronger match to that pole.

Each line is formatted as token<TAB>activation, with activations normalized to integers from -10 to +10.

The examples below were selected from a dataset with this setup: The model is instructed to Roleplay in a simple scenario.
The model is asked a question which gives a strategic chance to lie.

High-positive examples:

{positive_examples}

High-negative examples:

{negative_examples}

Infer the dominant contrast captured by the direction using both the positive and negative examples.

Important instructions:

- Do not assume the direction is a broad sentiment direction unless the examples clearly support that.

- Prefer the narrowest accurate description over a broad generic one.

- Do not overgeneralize beyond what is supported by the examples.

- Rate how likely this direction is to be useful for out-of-distribution deception on a scale from 1 to 10, and briefly explain why.

- Give low scores to directions that are mainly surface cues, lexical habits, or task-specific phrasing, even if they correlate with deception in these examples.

- A direction that is noisy, brittle, context-dependent, or likely to fail under different wording or domains should usually be scored around 1 to 4, not 5 or above.

- Reserve scores of 5 or higher for directions that seem plausibly tied to a more general deception-relevant strategy or behavior, rather than local wording.

Write your answer in exactly this format:

Positive pole: ...

Negative pole: ...

Shared direction: ...

OOD deception usefulness (1-10): ...

## Appendix C Lower PCs Show Stronger Decoder Alignment

Figure 6: Lower PCs show stronger decoder alignment, both for the best-matching SAE feature and for the mean of the top 10 features (Spearman $\rho$ = -0.857 and -0.917, respectively). We used andyrdt/saes-llama-3.1-8b-instruct (andyrdt, 2024)

.

## Appendix D PC Set Comparisons

The selection rules in Section 4.3 each produce a ranked list of PCs, and comparing these lists is a natural way to understand how the rules relate to one another. A convenient summary is the Jaccard overlap between the PC sets returned by two methods for the same target, which measures the fraction of selected directions they share. Low overlap indicates that two rules are drawing on largely different parts of the representation space, while high overlap indicates that they agree on which directions matter.

For reference, the PCs most heavily weighted by the source-trained probe, ranked by standardized coefficient, are listed below. These are the directions a probe trained on Roleplaying relies on most when it is free to use the entire 100-PC basis.

| Selection method | Dataset / target
|

Selected PCs

| Source probe weights | Source
|

PC7, PC8, PC11, PC12, PC4, PC5, PC2, PC3, PC6, PC27, PC22, PC21, PC1, PC41, PC18

Table 2: PCs ranked highest by the standardized coefficients of a source-only probe.

Most of these directions lie among the lowest-index PCs, which is unsurprising since low-index PCs capture the largest share of source variance and the source probe is trained on the same distribution. Whether such directions reflect the behavioral contrast itself or features specific to the Roleplaying scenarios is exactly the question the interpretability procedure is designed to address.

## Appendix E PC Interpretability

For each PC, the judge returns a description of the positive pole, a description of the negative pole, a summary of the shared direction, and an OOD usefulness score with a short justification. In practice, the justifications tend to fall into a small number of recurring patterns. Many directions are described as contrasts between denial or deflection on one side and admission or detailed explanation on the other. The judge frequently qualifies such descriptions by noting reliance on explicit lexical markers such as "Yes" and "No", on hedging phrases, or on domain-specific vocabulary, and lowers its score accordingly.

Other directions are described primarily in stylistic terms, such as assertiveness, formality, or the use of enumerations and résumé-like phrasing. These are typically judged to be surface cues that could correlate with deception in a particular dataset without reflecting a general strategy. A smaller number of directions are described in terms of communicative tactics, such as defensive or public-relations-style hedging, which the judge treats as more plausibly general while still flagging the risk of false positives from legitimate uses of the same tactics.

The judge's justifications also make explicit the trade-off between generality and reliability that motivates the prompt design. Directions tied to broad tactics are rated as more likely to transfer but are frequently characterized as noisy, while directions tied to narrow lexical patterns are rated as brittle even when they appear clean within the source data. This pattern is consistent with the instructions in Appendix B, and it illustrates the kind of distinction the interpretability-driven selection rule relies on when assembling a candidate subspace.