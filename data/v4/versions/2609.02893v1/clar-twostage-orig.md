# Probe Generalization as Subspace Selection for OOD Deception Detection

Daniel Yoo

  
Adrians Skapars

###### Abstract

Linear probes are classifiers that read weighted combinations of a model’s activations, or internal numerical representations. They can detect behaviors and concepts, but may fail out of distribution (OOD): on examples drawn from a different distribution than their training data.

We study deception probes for Llama-3.1-8B-Instruct. We train on Roleplaying, a dataset of simple scenarios with opportunities to lie, and test on 3 held-out datasets. Two concern reporting or confirming the use of insider information: Insider Trading Report and Insider Trading Confirm. The third, Sandbagging, concerns deliberate under-performance on some tasks.

We find that projecting activations onto a small subset of source-domain principal components (PCs) enables transfer that nearly matches probes trained directly on the test distribution. Principal component analysis (PCA) constructs perpendicular directions, called PCs, ordered by how much variation in the training activations they capture. Projection retains coordinates along selected directions. A target-supervised greedy search finds transferable subsets averaging 15, 8, and 14 PCs for the three targets, respectively (Section 4.2). This search fits probe weights on source data but uses labeled target examples to select PCs. We call it the greedy oracle.

Natural-language PC interpretations can identify some transferable PCs. We use a large language model (LLM) judge to describe each PC’s most and least activating source examples and score whether their contrast suggests a transferable deception direction. Probing on the highest-scoring PCs closes 78% of the gap between the full-representation baseline and greedy oracle on Insider Trading Report, and 25% on Sandbagging. The respective baseline, interpretation-selected, and oracle scores are 0.706, 0.881, and 0.929 for Report, and 0.606, 0.662, and 0.832 for Sandbagging (Section 5).

Directions heavily weighted by a source probe appear to encode source-specific surface features, such as wording and presentation. Directions that transfer appear to encode the same contrast more abstractly, in ways natural-language descriptions can capture (Tables 3–4). These are descriptive, not causal, interpretations. Broadly, our results suggest that OOD probe robustness is largely determined by subspace selection: choosing which activation directions the probe can use.

Machine Learning, ICML (International Conference on Machine Learning)

## 1 Introduction

Linear probes are standard tools for reading information from language model activations (Alain and Bengio, 2018; Hewitt and Liang, 2019). They are cheap to train and expose some semantically meaningful information in otherwise uninterpretable representation spaces.

However, probes that perform well in distribution—on data resembling their training distribution—can be unreliable OOD (Kirch et al., 2026). Existing work shows that key behavioral concepts are linearly represented in model activations: they can be read through linear combinations of activation values (Marks and Tegmark, 2024; Park et al., 2024; Zou et al., 2025). Probes should therefore have the capacity to detect these concepts. Yet, in practice, they appear to learn spurious correlations: dataset-specific associations rather than the intended concept (Orgad et al., 2025; Wang et al., 2025). This is unacceptable for probe-based monitoring of harmful interactions. It raises a clear question: how can we build probes that only learn generalizing features?

We address this question in the basis of source-domain PCs rather than raw activations. A basis supplies the directions used to express an activation vector. Using PCs avoids assuming that the model’s original activation coordinates provide a privileged basis.

PCs offer a simple, data-driven way to study representation structure. Prior work also suggests that PC choice matters. Mu et al. (2018) show that removing the top PCs of an embedding space—a numerical representation space—improves downstream performance. Kantamneni et al. (2025) use top-PC projections as a probe baseline. Our question is therefore: which subset of PCs supports generalizable probe training, and how can we identify it?

We use deception probes on Llama-3.1-8B-Instruct (Grattafiori et al., 2024) as our testbed. We train on one source dataset, Roleplaying, and evaluate on 3 held-out target datasets: Insider Trading Report, Insider Trading Confirm, and Sandbagging (Goldowsky-Dill et al., 2025). Working in the source-domain PCA basis, we study ways to select training dimensions that avoid spurious correlations and transfer to target domains.

This work contributes the following findings:

- **A small transferable subspace exists (Section 4.2).** A target-supervised greedy search over source PCs finds 8–15 components on which a source-trained probe nearly matches a target-trained probe. The average selected sizes are 15 for Insider Trading Report, 8 for Insider Trading Confirm, and 14 for Sandbagging.

- **Source-side and target-side rankings both fail to recover the subspace (Section 4.3).** Ranking PCs by the source or target probe’s weights does not produce a basis that transfers better than the full-representation baseline.

- **An LLM judge partially recovers the subspace without target data (Section 4.5).** The judge interprets and selects PCs likely to encode transferable behavioral contrasts rather than surface patterns. This closes 78% of the baseline-to-oracle gap on Insider Trading Report and 25% on Sandbagging.

Figure 1: Overview of PC-based subspace selection for OOD deception probing. Activations are projected onto the source-domain PCA basis, candidate PC directions are filtered, and probes are trained on the selected low-dimensional subspace.

## 2 Related Work

#### Linear concept directions in activation space.

A related line of work asks whether high-level behavioral or semantic properties correspond to directions or low-dimensional subspaces in activation space. A direction specifies a linear contrast; a subspace spans a set of such directions. Marks and Tegmark (2024) report linear structure in true/false representations across datasets. Park et al. (2024) formalize the linear representation hypothesis—the proposal that concepts have linear representations—and connect it to probing and steering. Steering means changing activations to influence model behavior. Zou et al. (2025) similarly model properties such as honesty and emotion as population-level directions: shared directions that can be read from and intervened on in model activations.

This perspective has also been used to explain and control behavior. Arditi et al. (2024) show that a single direction can mediate refusal behavior. This suggests that some high-level behaviors may be localized in simple linear features, or measurable activation patterns. Lu et al. (2026) identify an “assistant” axis corresponding to assistant-like behavior.

Together, these studies suggest that linear structure can support mechanistic understanding—understanding how internal model components produce behavior—and activation-space interventions. They also raise a question: when do these directions capture stable concepts rather than context-dependent correlations?

#### Linear probes and their limitations.

Linear and lightweight probes, which use relatively simple classifiers, have a long history in interpretability (Alain and Bengio, 2018; Hewitt and Liang, 2019). Recent work applies them to safety-relevant signals in LLM activations. These include latent truthfulness, meaning internally recoverable truthfulness without direct supervision (Burns et al., 2024), and strategic deception (Goldowsky-Dill et al., 2025).

Methodological skepticism is also central to probing research. Hewitt and Liang (2019) argue that high probe accuracy alone does not establish that a model already represents the target property. The probe may itself learn the task or exploit surface-level correlates. Ravichander et al. (2021) similarly question whether probing establishes what information is encoded, rather than merely recoverable by a trained classifier. Recent work also finds uneven OOD generalization despite strong in-distribution performance (kramár2026buildingproductionreadyprobesgemini, the citation identifier supplied here).

These concerns motivate careful tests of whether safety probes identify robust internal signals rather than dataset-specific artifacts.

#### Interpreting features with natural-language explanations.

Another approach assigns natural-language descriptions to neurons, individual activation units, or learned features. Bills et al. (2023) introduced LLM-generated neuron explanations based on high-activation examples.

This approach has been extended to automatic labeling of features in sparse autoencoders (SAEs) ((Paulo et al., 2025), (Bricken et al., 2023)). An SAE learns to represent activations using relatively few active features. Its feature dictionary is the collection of learned features. Automatic labeling enables large-scale interpretation of these dictionaries.

Natural-language feature descriptions have also been used for intervention. Casademunt et al. (2025), for example, identify concepts through automated interpretation and ablate them—remove their contribution—during fine-tuning, or further model training, to reduce misaligned behavior.

## 3 Setup

#### Datasets.

Following Goldowsky-Dill et al. (2025), we use 4 deception-detection datasets: Roleplaying, Insider Trading Report, Insider Trading Confirm, and Sandbagging. Appendix A gives their descriptions and complete label distributions in Table 1.

Roleplaying is the source domain used to train probe weights. The other 3 datasets are target test sets. GPT-4o (OpenAI, 2024), the labeling model, labels each response honest (0), ambiguous (-1), or deceptive (1). We remove ambiguous responses from our experiments.

#### Language models.

Following Ying et al. (2026), we use layer 15 residual stream activations from Llama-3.1-8B-Instruct (Grattafiori et al., 2024). The residual stream is the model’s internal representation carried through its layers. Layer 15 had the highest cross-domain AUROC in their analysis.

AUROC means area under the receiver operating characteristic curve. It measures how well a classifier’s scores distinguish the two classes across decision thresholds. Here, the classes are honest and deceptive responses.

To extract activations, we pass each prompt through the model. We average token activations over only the assistant’s response. A token is a unit of text processed by the model.

#### Probe training and evaluation.

We standardize source activations to zero mean and unit variance. We then use those same source means and variances to standardize all target activations. Variance measures the spread of activation values.

We run PCA on the scaled source activations and retain the top 100 PCs, ordered by variance captured. All probes are logistic regression classifiers, which map a weighted combination of inputs to a class probability. We use $\lambda=10$; the parameter $\lambda$ is not otherwise defined here.

For an experiment using a subset $S$ of these 100 PCs, $S$ denotes the selected component set. We project activations onto $S$ before both training and evaluation. Baseline probes instead train directly on the full standardized activations.

We retain the top 100 PCs because alignment with SAE decoder features decreases with PC index (Appendix C). Decoder features are the activation-space directions used by an SAE to reconstruct its inputs. Here, alignment measures how closely a PC corresponds to such a feature. Figure 6 reports Spearman rank correlations of -0.857 for best-matching feature alignment and -0.917 for mean alignment to the top 10 features.

This makes lower-index PCs more amenable to natural-language interpretation and allows comparison with interpretation-driven selection methods. It is **not** a claim that low-index PCs generalize better. Section 4.1 checks that the 100-PC subspace retains enough information to separate deceptive from honest responses on every target.

## 4 Experiments

### 4.1 Setting a baseline

Figure 2: Probe AUROC on target datasets for source-trained probes on full representations (blue) and top-100 PCs (orange), and target-trained probes on top-100 source PCs (green, 5-fold CV). Error bars: 95% CIs

Here, CV means cross-validation. In 5-fold cross-validation, data are divided into five folds; each fold is held out for evaluation while the other folds are used for fitting. CIs are confidence intervals, which express uncertainty in the estimates.

We first evaluate how well probes trained on full source representations transfer to each target. Figure 2 shows the results. Transfer is reasonably good on Insider Trading Confirm, at 0.874 AUROC. It is moderate on Insider Trading Report, at 0.706, and poor on Sandbagging, at 0.606.

We then check whether the 100-PC subspace itself is the bottleneck. We train probes directly on target data within this source-derived subspace and evaluate them with 5-fold cross-validation. They achieve 0.936 / 0.950 / 0.869 AUROC across the three targets. Thus, the retained subspace encodes enough information to distinguish deceptive from honest responses on every target.

The comparison matters because the source-trained probes transfer poorly on some targets even though target-trained probes can classify within the same retained subspace. Thus, the difficulty is not missing relevant information. It is that the source probe learns a direction in the subspace that does not generalize.

### 4.2 Showing a transferable subspace exists

Figure 3: Probe AUROC across the three targets for the full-representation baseline (blue), top-k PCs by variance (teal, k matched to greedy), and greedy oracle PCs (purple). Error bars: 95% CIs.

Can a probe trained on source data close the performance gap between the baseline and target-trained probes? We test this with a greedy PC-selection algorithm. “Greedy” means adding the best available component at each step.

We start with an empty set. At each step, we consider PCs among the top 100 source PCs. For each candidate, we project source activations onto the current subset plus that candidate and train a logistic regression probe on those source projections. We then add the PC that most improves target AUROC.

Probe weights are always fit on source data. However, PC selection uses labeled target examples. We evaluate this procedure with 5-fold cross-validation over the target dataset. In each split, we select PCs using the non-held-out target folds and evaluate the selected source-trained probe on the held-out target fold.

We stop when no candidate improves AUROC by more than 0.001. The search converges on small subsets. On average, it selects 15 PCs for Insider Trading Report, 8 for Insider Trading Confirm, and 14 for Sandbagging.

The greedy probe provides an upper bound on transfer performance, not an unbiased OOD estimate, because subset selection uses some target examples. We therefore call it an oracle comparison.

As a control, we select the top $k$ PCs by variance, where $k$ is the number of PCs selected by the greedy algorithm. This compares different PC choices at matched subspace sizes. Figure 3 shows the results.

On Insider Trading Report and Sandbagging, greedy selection comes close to a probe trained directly on the target dataset. It improves over the baseline by 0.22 and 0.23 AUROC points, respectively. Because the probe weights still come from source data, these gains show that a transferable subspace exists in principle.

The variance control matches greedy only on Insider Trading Confirm, the easy case. It falls far behind on Insider Trading Report and Sandbagging, the harder generalization tasks (Figure 3). Thus, which PCs are selected matters. The amount of variance they explain does not predict their usefulness.

### 4.3 Testing hypotheses for transferable principal components

Figure 4: Probe AUROC across the three targets for three PC-selection rules: ranking by source-probe weights (red), by target-probe weights (pink), and greedy oracle selection (purple). Error bars: 95% CIs.

What makes these PCs transferable? Can that property be detected using only source data? We test 2 hypotheses.

The first hypothesis is that transferable PCs are simply those that separate target classes well by themselves across the test set. Greedy selection would find them because it has target-data access.

The second hypothesis is that transferable PCs are those the source probe already weights most heavily. Greedy selection would merely rediscover the source signal in a more compact form.

We compare 2 PC rankings. Both use standardized coefficients: the absolute value of a probe coefficient multiplied by the standard deviation of its corresponding PC. The ranking probes are trained on activations projected onto the source-PCA basis.

- The **source-PCA probe** ranks PCs using coefficients learned on source data.
- The **target-PCA probe** ranks PCs using coefficients learned on target data.

For both rankings, the final evaluated probe is trained on **source data**, projected onto the top $k$ PCs in the respective ranking. Thus, target data determine the second ranking, but not the final probe weights. If either hypothesis were correct, its selected-PC probe should recover greedy performance.

We set $k$ by running greedy selection on the full target dataset rather than separately within cross-validation folds. This gives a single stable PC set: $k=13$ for Insider Trading Report, 7 for Insider Trading Confirm, and 14 for Sandbagging. Because this uses the full dataset rather than CV, the greedy numbers differ slightly from those in Section 4.2. Figure 4 shows the results.

Both hypotheses fail, but in different ways. Source-PCA selection transfers about as well as the full-representation baseline. The PCs on which the source probe relies are those that do not generalize.

Target-PCA selection does well on Insider Trading Report. However, it performs worse than greedy on Insider Trading Confirm and below random transfer performance on Sandbagging (Figure 4). Thus, PCs that best separate target classes are not necessarily those that transfer best when their weights are learned from source data.

Appendix D gives further PC-set comparisons. It reports greedy/target-PCA Jaccard overlaps of 0.182 for Report, 0.077 for Confirm, and 0.167 for Sandbagging. Jaccard overlap is the number of shared set members divided by the number in their union. Table 2 lists the selected PCs.

### 4.4 Interpreting the transferable subspace

The preceding experiments show that a small subset of source PCs supports transfer. It differs from the directions on which source or target classification relies. Does this subset have a human-interpretable meaning?

We use the interpretation method of Bills et al. (2023) and Casademunt et al. (2025). For each of the top 100 source PCs, we find the 8 most and 8 least PC-activating source examples. We then ask an LLM to describe the positive pole, negative pole, and direction as a whole. The positive and negative poles are the contrasting ends of the PC: positive coordinates point toward one, and negative coordinates toward the other.

The LLM judge rates each direction’s likely generalizability on a 1–10 scale, which we call its OOD score. Appendix B gives the full prompt and identifies the judge as GPT-5-mini.

No PC receives a score above 6. We therefore treat 6 as the maximum observed OOD score. We retain all PCs receiving this maximum in both scoring runs, giving 19 PCs.

Of the 5 PCs appearing in greedy selections for more than one target, 3 receive the maximum OOD score: PC62, PC57, and PC86. The other two, PC9 and PC81, receive 3 (Table 3, corresponding rows). Here and below, `PC` followed by a number identifies an individual source principal component.

By comparison, only 1 of the top 5 source-PCA PCs receives the maximum score. PC4 receives 6; PC7, PC8, PC11, and PC12 receive 3, 4, 3, and 3, respectively (Table 4, corresponding rows). This comparison suggests that LLMs can partially recover transferable directions.

Greedy-selected PC interpretations describe contrasts at a level of abstraction that survives movement to other domains. For example, Table 3 describes PC62 as denial or deflection versus providing details, and PC57 as denial, minimization, or innocuous explanations versus specific or incriminating details. The same rows also note lexical and contextual limitations. A full breakdown of PCs selected by each method is provided in Appendix E.

### 4.5 Selecting a subspace by interpreting PCs

Figure 5: Probe AUROC across the three targets for the interp-selected subspace (red; 19 PCs) versus three controls: full-representation baseline (blue), top-19 PCs by variance (teal), and the remaining 81 PCs after ablating the interp-selected PCs (brown). Error bars: 95% CIs.

Can interpretation-based OOD scores recover a meaningful fraction of greedy performance without target data?

We select PCs receiving the judge’s maximum observed OOD score. Because judge scores are noisy, we score twice and retain only PCs receiving that maximum in both runs. This gives 19 **interp-selected**, or interpretation-selected, PCs.

We compare against the full-representation baseline and two PC-based controls. One uses the top 19 source PCs by variance. The other removes the selected PCs and trains on the remaining 81 PCs. We call this second control **interp-ablated**. This is an ablation of the probe’s input subspace, not an intervention during language-model inference.

Figure 5 shows that the interp-selected probe outperforms the baseline on all 3 targets. It also outperforms the source-variance control, by wide margins on Insider Trading Report and Sandbagging. As expected, it outperforms the interp-ablated probe on all 3 targets.

The amount of recovery differs sharply by dataset. Gap closure measures how much of the improvement from baseline to greedy oracle is recovered by interpretation-based selection.

On Insider Trading Report, interp-selection closes 78% of the gap between baseline (0.706) and greedy oracle (0.929). The interp-selected score is 0.881, as also reported in Section 5.

On Insider Trading Confirm, baseline transfer is already near the ceiling, leaving little headroom for comparison. The result is therefore uninformative: each selection method except interp-ablated does roughly as well as the others.

On Sandbagging, interp-selection closes 25% of the gap between the greedy oracle (0.832). Section 5 reports the baseline as 0.606 and the interp-selected score as 0.662.

Overall, selecting PCs purely from natural-language descriptions of what they encode, without target data, meaningfully improves baseline performance on the generalization tasks.

## 5 Limitations

Our strongest claims remain narrow. Readers should consider several limitations.

**Single model and source dataset.** All experiments use one model, Llama-3.1-8B-Instruct, at layer 15, and one source dataset, Roleplaying (Goldowsky-Dill et al., 2025). Insider Trading Confirm results are also uninformative because baseline transfer already reaches 0.874. Whether the results generalize to other model families or datasets remains an open question for future work.

**Sandbagging margins are small.** Sandbagging gives the weakest result. Interpretation-based selection raises AUROC from 0.606 to 0.662, recovering only 25% of the gap to the greedy upper bound of 0.832. Sandbagging is also the most OOD target. The modest gain likely reflects this distribution shift, or change between source and target data.

The Insider Trading Report result is more robust: AUROC rises from 0.706 to 0.881, closing 78% of the gap to greedy at 0.929.

**Linear pipeline.** Our pipeline is fully linear: PCA projection followed by logistic regression. The existence of a transferable subspace is therefore perhaps unsurprising. Whether these findings extend to nonlinear probes, which can use nonlinear combinations of inputs, or other readout locations in the model remains open.

**No causal validation.** We make structural claims about what each PC “encodes” from top-activating examples and the LLM judge’s verbal description of their contrast. These claims are descriptive, not causal. We have not ablated high-scoring PCs during model inference to verify that they carry the contrast they appear to encode.

## 6 Conclusion

For probe-based deception monitoring, a small subset of source-domain PCs enables cross-domain transfer that nearly matches target-trained probes. Source-side or target-side separability alone—the ability to distinguish classes within either dataset—does not recover this subset (Section 4.3). An LLM judge’s interpretation score over each PC’s top-activating examples partially recovers it without target-data access (Sections 4.4–4.5).

A key implication is that robust probing requires more than establishing that a behavioral signal is linearly accessible. It also requires identifying where in activation space that signal is represented most robustly.

Linear probes trained on full representations may combine general behavioral features with dataset-specific surface cues. These heuristics can support high in-distribution accuracy while hurting transfer. In contrast, sparse PC subspaces—those using only a small subset of components—may better isolate general behavioral structure.

This reframes OOD probing as a representation-selection problem. The goal is not only to fit a classifier. It is also to choose a subspace that preserves the intended behavioral feature and filters out spurious correlation.

## Acknowledgment

This work was conducted during the 11th edition of AI Safety Camp, which provided the compute to run our experiments. Thanks to Prasad Mahadik and Arya Hariharan for useful discussions around this work.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## Appendix A Dataset Details

**Roleplaying:** the model is placed in a simple scenario with an opportunity to lie.

**Insider Trading Report:** acting as a trader, the model decides whether to honestly report that it used insider information.

**Insider Trading Confirm:** after hiding earlier use of insider information, the model is asked to confirm it did not use any.

**Sandbagging:** the model is told to deliberately under-perform on some tasks while behaving normally on others.

Table 1 lists total examples and label counts before ambiguous responses are removed. The label-count order is ambiguous (-1), honest (0), and deceptive (1).

| Dataset | Total Examples | Label Counts (-1 / 0 / 1) |
|---|---|---|
| Roleplaying | 371 | 140 / 59 / 172 |
| Insider Trading Report | 473 | 218 / 194 / 61 |
| Insider Trading Confirm | 151 | 29 / 38 / 84 |
| Sandbagging | 1000 | 379 / 375 / 246 |

Table 1: Dataset label distributions.

## Appendix B LLM Judge Prompt

We use GPT-5-mini (OpenAI, 2025) to interpret directions.

The prompt below describes a bipolar direction, meaning one with positive and negative poles. In its input format, `<TAB>` denotes a tab separator between a token and its activation. `{positive_examples}` and `{negative_examples}` are placeholders for the selected high-positive and high-negative examples. “Shared direction” asks for the contrast joining the two poles.

```text
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
```

## Appendix C Lower PCs Show Stronger Decoder Alignment

“Lower PCs” here means lower-index PCs. Spearman $\rho$ is a rank correlation: it measures how two quantities vary together in their ordering. Negative values here mean alignment decreases as PC index increases. The resource identifier `andyrdt/saes-llama-3.1-8b-instruct` names the SAE resource used for this comparison.

Figure 6: Lower PCs show stronger decoder alignment, both for the best-matching SAE feature and for the mean of the top 10 features (Spearman $\rho$ = -0.857 and -0.917, respectively). We used andyrdt/saes-llama-3.1-8b-instruct (andyrdt, 2024).

## Appendix D PC Set Comparisons

We compare greedy selection with `target-pca`, the target-PCA coefficient-ranking method from Section 4.3. Their PC sets have Jaccard overlap of 0.182 for Insider Trading Report, 0.077 for Insider Trading Confirm, and 0.167 for Sandbagging. The 2 methods agree on less than a fifth of their selections. Table 2 gives the PC lists.

Together, these results suggest that the transferable subspace is neither the most discriminative part of source representation space nor the most discriminative part of target representation space. “Discriminative” here means useful for separating the honest and deceptive classes within that dataset.

| Selection method | Dataset / target | Selected PCs |
|---|---|---|
| Source probe weights | Source | PC7, PC8, PC11, PC12, PC4, PC5, PC2, PC3, PC6, PC27, PC22, PC21, PC1, PC41, PC18 |
| Greedy selection | Insider Trading Report | PC6, PC89, PC62, PC66, PC26, PC14, PC57, PC25, PC93, PC9, PC70, PC97, PC78 |
| Greedy selection | Insider Trading Confirm | PC71, PC53, PC28, PC86, PC32, PC81, PC77 |
| Greedy selection | Sandbagging | PC37, PC62, PC54, PC41, PC46, PC9, PC57, PC44, PC51, PC43, PC19, PC48, PC86, PC81, PC33 |
| Target separability ranking | Insider Trading Report | PC5, PC6, PC89, PC18, PC79, PC46, PC87, PC4, PC14, PC34, PC29, PC1, PC66, PC67, PC19 |
| Target separability ranking | Insider Trading Confirm | PC4, PC1, PC8, PC53, PC30, PC7, PC13, PC18, PC5, PC20, PC26, PC49, PC28, PC31, PC97 |
| Target separability ranking | Sandbagging | PC51, PC91, PC97, PC13, PC39, PC77, PC4, PC66, PC53, PC86, PC62, PC23, PC41, PC10, PC61 |

Table 2: Comparison of PCs selected by source-only probe weights, greedy transfer selection, and target separability ranking.

## Appendix E PC Interpretability

The explanations below are the LLM judge’s descriptions. They are not causally validated claims about the PCs.

In these explanations, **lexical** cues concern word choice; **negation** concerns wording that denies something; and **hedges** qualify or weaken a statement. **Semantic strategy** concerns the meaning or communicative behavior rather than particular words. **Pragmatic** contrasts concern what a response does in context, while **discourse style** concerns how it presents information. **Enumerations** are lists. **PR-style** means public-relations-style wording, and **QA** means question answering. A **false positive** incorrectly flags an honest response as deceptive; a **false negative** misses a deceptive response.

Table 3: Principal components selected for likely transferability, based on interpretation with an LLM judge.

| PC | Selected On | OOD Score | Full explanation |
|---|---|---|---|
| PC62 | Insider Trading Report + Sandbagging | 6 | OOD deception usefulness (1-10): 6 — This direction maps onto a general deception-relevant behavior (deny/deflect vs provide details), so it can be useful across settings, but it is noisy: true speakers also deny or obfuscate for legitimate reasons and liars sometimes produce elaborate fabricated detail, and the signal may pick up simple lexical negation rather than semantic strategy. |
| PC57 | Insider Trading Report + Sandbagging | 6 | This direction aligns with a general deception-relevant strategy (deny/minimize or present innocuous explanations vs give specific/incriminating details), so it can help flag rhetorical moves often used when hiding or deflecting. However it is partly lexical and task-specific (relies on surface denials, hedges, and domain terms), so it will be brittle and context-dependent out of distribution. |
| PC9 | Insider Trading Report + Sandbagging | 3 | OOD deception usefulness (1-10): 3 — This direction mainly picks up domain‑specific lexical cues (medical/leave language vs project/status language) and stylistic framing rather than a generalizable deception strategy. It may flag genuine illness claims and is brittle across domains and rephrasings. |
| PC81 | Insider Trading Confirm + Sandbagging | 3 | OOD deception usefulness (1-10): 3 — The direction picks up a plausible deception‑relevant contrast (admit vs deny/hedge), but it mainly relies on surface/formal cues (negations, hedges, résumé style, enumerations) that are task- and wording‑dependent and likely brittle across domains and phrasings. |
| PC86 | Insider Trading Confirm + Sandbagging | 6 | OOD deception usefulness (1-10): 6 — Hedging and defensive PR-style wording is a general tactic associated with deceptive or self-protective behavior, so the direction could transfer across domains. However it is noisy (used legitimately for privacy, politeness, or negotiation), sensitive to phrasing and role, and will produce false positives, so it’s only moderately reliable out-of-distribution. |

Table 4: Principal components and their OOD scores with interpretations.

| PC | OOD Score | Full explanation |
|---|---|---|
| PC7 | 3 | OOD deception usefulness (1-10): 3 — This direction mostly picks up a pragmatic contrast (deny/deflect vs admit) which can correlate with deception in some settings, but it relies heavily on surface cues (explicit “No”, word order, presence of “I” + admission) and task-specific phrasing. It is therefore brittle and likely to fail or mislead in different domains or with varied wording. |
| PC8 | 4 | OOD deception usefulness (1-10): 4 — The direction captures a useful narrow contrast (denial/excuse vs admission) that can correlate with evasive or deceptive responses in similar QA/roleplay tasks, but it largely picks up surface cues (explicit “No”/“Yes” and common denial phrasing) and is likely brittle under paraphrase, different domains, or subtler deception strategies. |
| PC11 | 3 | OOD deception usefulness (1-10): 3 — The direction mainly picks up a stylistic/pragmatic contrast (assertive first‑person role/defense vs problem-reporting/negation). That can correlate with evasive or face-saving lies in some scenarios, but it’s surface-y and context dependent and likely to fail across different domains or phrasings. |
| PC12 | 3 | OOD deception usefulness (1-10): 3 — This direction mainly captures discourse style/forcefulness (a surface cue). While overconfident or over-assertive phrasing can sometimes correlate with deceptive defensive behavior, it is brittle and domain-dependent and will produce many false positives/negatives when applied across different speakers, topics, or cultural styles. |
| PC4 | 6 | OOD deception usefulness (1-10): 6 — This captures a general communicative strategy relevant to deception (deny vs admit), so it can transfer beyond the specific prompts. However it also leans on surface cues (explicit “Yes”/“No” turns and common defensive phrasings) and may be brittle when denials or lies use different wording or more subtle tactics. |
