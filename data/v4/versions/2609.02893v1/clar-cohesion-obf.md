# Probe Generalization as Subspace Selection for OOD Deception Detection

Daniel Yoo  
Adrians Skapars

###### Abstract

Directions heavily weighted by a source probe appear to encode source-specific surface features. In contrast, directions that transfer out of distribution (OOD) appear to encode the same contrast more abstractly, in a form capturable in natural language. Thus, probe OOD robustness appears largely determined by subspace selection.

Machine Learning, ICML (International Conference on Machine Learning)

## 1 Introduction

Linear probes cheaply expose semantically meaningful information in otherwise uninterpretable language-model representations (Alain and Bengio, 2018; Hewitt and Liang, 2019). Although linear probes detect behaviors and concepts, linear probes sometimes fail out of distribution despite strong in-distribution performance (Kirch et al., 2026). Behavioral concepts have linear representations (Marks and Tegmark, 2024; Park et al., 2024; Zou et al., 2025), so probes should have detection capacity. However, probes appear to learn spurious correlations in practice (Orgad et al., 2025; Wang et al., 2025). Such unreliability is unacceptable for harmful-interaction monitoring.

We work in coordinates defined by source principal components (PCs, coded **Q7**), obtained through principal component analysis (PCA, coded **N4D**). Working in source principal-component coordinates avoids privileging the model’s latent dimensions because the coordinates instead provide data-driven representation structure. Prior work shows that removing leading components improves downstream embedding performance (Mu et al., 2018), while leading-component projections serve as a probing baseline in Kantamneni et al. (2025). Here, the selection problem is to identify dimensions that permit generalizing, rather than spurious, source-trained classification.

Figure 1: OOD deception-probing pipeline: projection into the source PCA (N4D) basis, filtering candidate principal-component (Q7) directions, and probe training in the selected low-dimensional subspace.

## 2 Related Work

#### Linear concept directions in activation space.

True/false representations exhibit cross-dataset linear structure (Marks and Tegmark, 2024). Park et al. (2024) formalize the linear representation hypothesis and connect the hypothesis to probing and steering. Honesty and emotion admit population-level activation directions permitting readout and intervention (Zou et al., 2025). A single direction mediates refusal (Arditi et al., 2024), and an “assistant” axis corresponds to assistant-like behavior (Lu et al., 2026). Such simple features support mechanistic understanding and activation interventions. However, the distinction between stable concepts and context-dependent correlations remains unresolved.

#### Linear probes and their limitations.

Linear and lightweight probes have a long interpretability history (Alain and Bengio, 2018; Hewitt and Liang, 2019), including work on latent truthfulness without direct supervision (Burns et al., 2024) and strategic deception (Goldowsky-Dill et al., 2025). However, high accuracy need not establish that a model already represents the property being probed: the probe can learn the task or exploit surface correlates (Hewitt and Liang, 2019). Therefore, encoded information and merely recoverable information remain distinct concerns (Ravichander et al., 2021). Strong in-distribution probes generalize unevenly (kramár2026buildingproductionreadyprobesgemini), so safety-probe evaluation must distinguish robust internal signals from dataset artifacts.

#### Interpreting features with natural-language explanations.

Bills et al. (2023) introduced large language model (LLM)-generated neuron explanations from high-activation examples. Automatic sparse-autoencoder feature labeling extends this approach to large feature dictionaries ((Paulo et al., 2025), (Bricken et al., 2023)). Natural-language descriptions also support intervention: Casademunt et al. (2025) automatically identify concepts and ablate the identified concepts during fine-tuning, reducing misaligned behavior.

## 3 Setup

The model is **Llama-3.1-8B-Instruct**, coded **E6**, from Grattafiori et al. (2024). We use the model’s layer 15 residual-stream activations, following Ying et al. (2026), whose analysis assigns layer 15 the highest cross-domain discrimination. Each prompt is passed through Llama-3.1-8B-Instruct, and activations are averaged over assistant-response tokens only.

The four datasets come from Goldowsky-Dill et al. (2025). **Roleplaying**, coded **V8**, is the single source dataset; no other source dataset is used. The three held-out targets are a reporting task, **Insider Trading Report**, coded **C2R**; a confirmation task, **Insider Trading Confirm**, coded **W6**; and an underperformance task, **Sandbagging**, coded **J9**. The response labeler is **GPT-4o**, coded **K8** (OpenAI, 2024). GPT-4o assigns honest responses label 0, ambiguous responses label -1, and deceptive responses label 1. Ambiguous responses are then removed.

Source activations receive zero-mean, unit-variance standardization. The same source statistics are also used to standardize target activations. PCA (N4D) then retains the leading 100 principal-component directions (Q7). Let \(S\) denote a selected subset of these retained directions; projection onto subset \(S\) precedes probe training and evaluation. All classifiers use logistic regression with parameter \(\lambda=10\). In contrast to the subspace probes, the **full-representation baseline**, coded **B8Q**, trains directly on all standardized activations.

The choice to retain 100 components is determined by interpretability comparability, not by superior low-index generalization. Alignment with sparse-autoencoder (SAE) decoder features, coded **U9** (Bricken et al., 2023), decreases with principal-component index, making lower indices more amenable to natural-language interpretation. Appendix C and the decoder-alignment rows in Appendix A report this relationship: Spearman rank-correlation coefficients are \(\rho=-0.857\) for the best-matching feature and \(\rho=-0.917\) for the mean of the top 10 features.

The discrimination measure used below is the **area under the receiver operating characteristic curve** (AUROC), coded **F3**. Cross-validation is abbreviated **CV**. Error bars denote 95% confidence intervals (CIs).

## 4 Experiments

### 4.1 Setting a baseline

For the source-trained full-representation baseline, transfer is reasonably good on Insider Trading Confirm, moderate on Insider Trading Report, and poor on Sandbagging: the respective target AUROCs are 0.874, 0.706, and 0.606 (Appendix A, “B8Q F3” row).

However, target-trained classification within the retained source subspace separates deceptive from honest responses on every target. The target-trained AUROCs under 5-fold cross-validation are 0.950 on Insider Trading Confirm, 0.936 on Insider Trading Report, and 0.869 on Sandbagging (Appendix A, “Target-trained F3, 5-fold CV” row). Thus, relevant information is present in the retained source subspace, but the source-trained direction fails to generalize.

Figure 2: Target AUROC (F3) for source-trained full representations, blue; source-trained leading 100 principal components (Q7), orange; and target-trained leading 100 source principal components, green, with 5-fold cross-validation.

### 4.2 Showing a transferable subspace exists

We use a target-guided search called the **greedy oracle**, coded **H2**, to test whether a transferable subspace exists. Let \(k\) denote the number of selected directions. The variance-selection comparator, **top-k PCs by variance**, coded **Y4**, selects \(k\) directions ranked by explained variance.

The greedy-oracle search starts with an empty subset and considers each of the leading 100 source principal-component directions. For each candidate addition, the search projects source activations onto the candidate subset and fits logistic-regression weights on the source data. The search then adds the direction that most improves target AUROC. Search stops when no candidate improvement exceeds 0.001.

Target 5-fold cross-validation performs subset selection on the non-held-out target folds and evaluates the selected source-trained probe on the held-out target fold. Thus, target labels guide subset selection but never probe-weight fitting. Because target labels guide selection, this procedure is a transfer upper bound, not an unbiased OOD estimate.

Small source-trained subspaces nearly match target-trained probes, particularly on Insider Trading Report and Sandbagging (Figures 2 and 3). Appendix A reports a small greedy-oracle subset range of 8–15 directions, with mean target-cross-validation subset sizes of 15 for Insider Trading Report, 8 for Insider Trading Confirm, and 14 for Sandbagging. The same appendix reports greedy-oracle improvements over the full-representation baseline of 0.22 AUROC points on Insider Trading Report and 0.23 on Sandbagging.

In contrast, variance selection matches the greedy oracle only on the easy Insider Trading Confirm case and falls far behind on the harder Insider Trading Report and Sandbagging tasks (Figure 3). Thus, explained variance does not predict usefulness.

Figure 3: Target AUROC (F3) for the full-representation baseline (B8Q), blue; top-k PCs by variance (Y4), with \(k\) matched to the greedy oracle, teal; and the greedy oracle (H2), purple.

### 4.3 Testing hypotheses for transferable components

We test two hypotheses: transferable directions might be directions with individually strong target-class separation, accessible to target-guided selection; alternatively, transferable directions might be directions already heavily weighted by the source probe, compactly rediscovered by selection.

Both hypotheses are tested through rankings based on a magnitude measure called the **standardized coefficient**, coded **X4**. For each direction, the standardized coefficient is the absolute probe coefficient multiplied by the corresponding principal-component standard deviation. The ranking probes operate on activations projected through the source PCA.

A source-trained ranking probe, the **source-PCA probe**, coded **T6**, supplies one standardized-coefficient ordering. A target-trained ranking probe, the **target-PCA probe**, coded **M9**, supplies the other ordering. After ranking, both evaluated probes train on source data projected onto their top \(k\) ranked directions. Thus, target training supplies the target-PCA ranking, but the evaluated probe weights are still fitted on source data.

For these comparisons, greedy-oracle selection on the full target supplies a stable set and \(k\), rather than fold-specific selections. Consequently, the full-target greedy-oracle numbers differ slightly from the cross-validation results. Appendix A reports full-target greedy-oracle \(k\) values of 13 for Insider Trading Report, 7 for Insider Trading Confirm, and 14 for Sandbagging.

Neither the source-side nor the target-side ranking recovers the transferable subspace or a basis transferring better than the full-representation baseline (Figure 4). The source-PCA probe transfers about as well as the full-representation baseline: source-reliant directions do not generalize. The target-PCA probe does well on Insider Trading Report, performs worse than the greedy oracle on Insider Trading Confirm, and performs below random on Sandbagging (Figure 4). Therefore, target-class separation and transfer under source-fitted weighting select different directions.

Figure 4: Target AUROC (F3) for the source-PCA probe (T6), red; the target-PCA probe (M9), pink; and the greedy oracle (H2), purple.

### 4.4 Interpreting the transferable subspace

To interpret directions, we use an **LLM judge**, coded **C7**. The judge model is **GPT-5-mini**, coded **P2** (OpenAI, 2025); Appendix B gives the prompt.

For each leading source principal-component direction, the judge receives the eight most-activating and eight least-activating source examples. Following Bills et al. (2023) and Casademunt et al. (2025), the judge generates descriptions of the positive pole, negative pole, and whole direction. The judge then assigns a generalization rating called the **OOD score**, coded **R5**, on a 1–10 scale.

Selection retains all directions that reach the observed maximum score in both scoring runs. The scores are noisy. The observed maximum was 6, no direction exceeded 6, and 19 directions reached that maximum in both runs (Appendix E, selection-observation table).

Transferable subsets differ from directions relied on by source or target classification, as shown by the direction sets and overlaps in Appendix D. Interpretations of greedy-oracle-selected directions express contrasts at an abstraction that survives domain changes, but only partial recovery occurs. For example, among the five directions selected by the greedy oracle for more than one target, three received the maximum rating; among the top five source-PCA selections, one received the maximum rating (Appendix E, selection-observation table and Tables 3–4).

### 4.5 Selecting a subspace by interpreting components

The intersection of directions receiving the maximum score in both interpretation runs defines the **interp-selected PC probe**, coded **L2**. This probe uses the 19 retained directions. The controls are the complementary 81 directions, used by the **interp-ablated PC probe**, coded **A7**, and an equal-dimensional variance-ranked subset, the **top-k PCs by variance** control (Y4). All three methods train as source probes.

Language-only selection, without target data, meaningfully improves generalization over the full-representation baseline. The interp-selected probe exceeds the full-representation baseline on all targets and exceeds the equal-dimensional variance control, with wide gains on Insider Trading Report and Sandbagging (Figure 5). On Insider Trading Report, the interp-selected AUROC is 0.881 versus the baseline’s 0.706; on Sandbagging, the interp-selected AUROC is 0.662 versus the baseline’s 0.606 (Appendix A, “L2 F3” and “B8Q F3” rows). The interp-selected probe also exceeds the interp-ablated probe everywhere (Figure 5).

However, recovery varies sharply across targets. On Insider Trading Report, interpretation-based selection substantially closes the baseline–greedy-oracle gap, recovering 78% of that gap; on Sandbagging, recovery is only partial, at 25%. The greedy-oracle AUROCs used for these recovery comparisons are 0.929 and 0.832, respectively (Appendix A, recovery-comparison rows). Insider Trading Confirm has little headroom and is uninformative: every selection method except the interp-ablated probe is roughly tied (Figure 5).

Figure 5: Target AUROC (F3) for the interp-selected PC probe (L2), red; the full-representation baseline (B8Q), blue; top-k PCs by variance (Y4) at the interp-selected probe’s dimension count, teal; and the interp-ablated PC probe (A7), brown.

## 5 Limitations

Claims remain narrow. The single-model, single-source scope leaves other model families and datasets unresolved. In addition, Insider Trading Confirm’s already-high baseline—0.874 AUROC in Appendix A—makes its results uninformative.

Sandbagging is the weakest result and the most out-of-distribution target. Its modest improvement, from the full-representation baseline’s 0.606 AUROC to the interp-selected probe’s 0.662, likely reflects distribution shift. Insider Trading Report’s improvement, from 0.706 to 0.881, is more robust.

The pipeline is fully linear: PCA projection is followed by logistic regression. This linearity makes transferable-subspace existence perhaps unsurprising. However, extension to nonlinear probes or other readout locations remains open.

Statements about what directions “encode” are descriptive: the statements are based on extreme-activation examples and LLM-judge verbalizations, not causal evidence. High-rated directions have not been ablated during inference to verify the apparent contrast.

## 6 Conclusion

Robust behavioral probing requires identifying where a linearly accessible signal is represented most robustly, not merely establishing the signal’s accessibility. Full-representation probes may mix general behavioral features with dataset-specific surface cues, supporting high in-distribution accuracy while harming transfer. In contrast, sparse directional subspaces may isolate more general behavioral structure. Therefore, OOD probing becomes representation selection as well as classifier fitting: preserve the intended behavioral feature and filter spurious correlation.

## Acknowledgment

Conducted during the 11th AI Safety Camp, which supplied experimental compute. Prasad Mahadik and Arya Hariharan provided useful discussions.

## Impact Statement

The work aims to advance Machine Learning; many societal consequences are possible, none requiring specific highlighting in the authors’ view.

## Appendix A Dataset Details

Roleplaying (V8) places Llama-3.1-8B-Instruct (E6) in a simple scenario offering an opportunity to lie. In Insider Trading Report (C2R), the model acts as a trader deciding whether to honestly report insider-information use. Insider Trading Confirm (W6) asks the model, after concealing earlier insider-information use, to confirm non-use. Sandbagging (J9) instructs deliberate underperformance on some tasks and normal behavior on others.

In Table 1, label -1 means ambiguous, label 0 means honest, and label 1 means deceptive. Ambiguous responses are removed before the probing experiments.

Table 1: Dataset label distributions.

| Dataset | Total examples | Label counts (-1 / 0 / 1) |
|---|---:|---|
| V8 | 371 | 140 / 59 / 172 |
| C2R | 473 | 218 / 194 / 61 |
| W6 | 151 | 29 / 38 / 84 |
| J9 | 1000 | 379 / 375 / 246 |

The following table records AUROC (F3). The full-representation baseline is B8Q; the target-guided greedy oracle is H2; and the interpretation-selected probe is L2. Target-trained entries are obtained inside the leading-100 source principal-component subspace. Cross-validation (CV) uses 5 folds where specified, and \(k\) denotes the selected dimension count. Greedy-oracle gains are retained as reported.

| Observation | C2R | W6 | J9 |
|---|---:|---:|---:|
| B8Q F3 | 0.706 | 0.874 | 0.606 |
| Target-trained F3, 5-fold CV | 0.936 | 0.950 | 0.869 |
| H2 improvement over B8Q, F3 points | 0.22 | — | 0.23 |
| H2 F3 used for recovery comparison | 0.929 | — | 0.832 |
| L2 F3 | 0.881 | — | 0.662 |
| Baseline–H2 gap recovered by L2 | 78% | — | 25% |
| Mean H2 subset size, target CV | 15 | 8 | 14 |
| Full-target H2 \(k\) | 13 | 7 | 14 |

The additional observations below give the small greedy-oracle subset range and decoder-alignment results. Spearman \(\rho\) is the rank-correlation coefficient used to measure the relationship between principal-component index and decoder alignment.

| Additional observation | Value |
|---|---|
| Small H2 subset range | 8–15 |
| Decoder alignment, best-matching feature, Spearman \(\rho\) | -0.857 |
| Decoder alignment, mean of top 10 features, Spearman \(\rho\) | -0.917 |

## Appendix B LLM Judge Prompt

Direction interpretation uses GPT-5-mini (P2), OpenAI (2025), as the LLM judge. In the prompt below, Q7 means a principal-component direction, E6 means Llama-3.1-8B-Instruct, V8 means the Roleplaying source dataset, and R5 means the OOD generalization score.

> Interpret a bipolar Q7 direction in neural representation space: positive activations move toward one pole, negative activations toward its opposite, larger magnitudes indicate stronger pole matches. Lines have format `token<TAB>activation`; activations are normalized integers from -10 to +10.
>
> Source setup: E6 is instructed to enact a simple V8 scenario and asked a question offering a strategic opportunity to lie.
>
> High-positive examples:
>
> {positive_examples}
>
> High-negative examples:
>
> {negative_examples}
>
> Infer the dominant contrast from both sets. Assume broad sentiment only with clear support; prefer the narrowest accurate description and avoid unsupported generalization.
>
> Rate usefulness for OOD deception and briefly explain. Give low ratings to mainly surface cues, lexical habits, or task-specific phrasing even when correlated with deception here. Noisy, brittle, context-dependent directions or directions likely to fail across wording or domains usually warrant 1 to 4, not 5 or above. Reserve 5 or higher for plausible general deception-relevant strategies or behaviors rather than local wording.
>
> Use exactly:
>
> Positive pole: ...
>
> Negative pole: ...
>
> Shared direction: ...
>
> R5 (1-10): ...

## Appendix C Lower Q7 Show Stronger Decoder Alignment

Lower-index principal-component directions (Q7) align more strongly with sparse-autoencoder decoder vectors, called **SAE decoder features** (U9). Alignment is measured using both the best-matching decoder feature and mean top-feature alignment, with `andyrdt/saes-llama-3.1-8b-instruct` (andyrdt, 2024). The corresponding Spearman \(\rho\) values are -0.857 for the best-matching feature and -0.917 for the mean of the top 10 features (Appendix A, additional-observation table).

Figure 6: Lower principal components (Q7) exhibit stronger decoder alignment for the best-matching feature and the mean of the top 10 features.

## Appendix D Q7 Set Comparisons

We compare principal-component (Q7) selections using **Jaccard overlap**, coded **D3**: the size of the intersection of two selected sets divided by the size of their union. The greedy oracle (H2) and the target-PCA ranking probe (M9) agree on less than a fifth of selections: Jaccard overlaps are 0.182 for Insider Trading Report (C2R), 0.077 for Insider Trading Confirm (W6), and 0.167 for Sandbagging (J9), as shown below. This agreement is consistent with transferable space being neither the most source-discriminative nor the most target-discriminative region.

| Target | H2–M9 D3 |
|---|---:|
| C2R | 0.182 |
| W6 | 0.077 |
| J9 | 0.167 |

Table 2 gives direction sets from source-only probe weights, transfer selection, and target separability ranking. In the table, T6 is the source-PCA ranking probe, H2 is the greedy oracle, and M9 is the target-PCA ranking probe. The selected Q7 indices are principal-component indices.

Table 2: Direction sets from source-only probe weights, transfer selection, and target separability ranking.

| Selection method | Dataset / target | Selected Q7 indices |
|---|---|---|
| T6 | Source | 7, 8, 11, 12, 4, 5, 2, 3, 6, 27, 22, 21, 1, 41, 18 |
| H2 | C2R | 6, 89, 62, 66, 26, 14, 57, 25, 93, 9, 70, 97, 78 |
| H2 | W6 | 71, 53, 28, 86, 32, 81, 77 |
| H2 | J9 | 37, 62, 54, 41, 46, 9, 57, 44, 51, 43, 19, 48, 86, 81, 33 |
| M9 | C2R | 5, 6, 89, 18, 79, 46, 87, 4, 14, 34, 29, 1, 66, 67, 19 |
| M9 | W6 | 4, 1, 8, 53, 30, 7, 13, 18, 5, 20, 26, 49, 28, 31, 97 |
| M9 | J9 | 51, 91, 97, 13, 39, 77, 4, 66, 53, 86, 62, 23, 41, 10, 61 |

## Appendix E Q7 Interpretability

The following table summarizes principal-component (Q7) interpretation and selection. The OOD score is R5; the interp-selected probe is L2; the complementary interp-ablated probe is A7; the greedy oracle is H2; and the source-PCA ranking probe is T6. A maximum-rated direction has the maximum observed score of 6.

| Selection observation | Value |
|---|---:|
| Maximum observed R5; no direction exceeded it | 6 |
| Scoring runs | 2 |
| Directions reaching maximum in both runs, retained for L2 | 19 |
| Complement directions used by A7 | 81 |
| Directions selected by H2 for more than one target | 5 |
| Maximum-rated directions among those shared H2 selections | 3 out of 5 |
| Maximum-rated directions among the top T6 selections | 1 of the top 5 |

Table 3 gives LLM-judge (C7) interpretations of directions shared across target selections. In the table, C2R is Insider Trading Report, W6 is Insider Trading Confirm, and J9 is Sandbagging. “PR-style” means public-relations-style wording.

Table 3: Directions selected for likely transferability from C7 interpretation.

| Q7 index | Selected on | R5 | Full explanation |
|---|---|---:|---|
| 62 | C2R + J9 | 6 | General deception-relevant denial/deflection versus detail provision can be useful across settings; noise includes truthful denial or obfuscation for legitimate reasons and liars’ elaborate fabricated details. Lexical negation rather than semantic strategy may drive the signal. |
| 57 | C2R + J9 | 6 | General denial/minimization or innocuous explanations versus specific/incriminating details can flag rhetorical hiding or deflection. Partial dependence on surface denials, hedges, and domain terms makes it lexical, task-specific, brittle, and context-dependent OOD. |
| 9 | C2R + J9 | 3 | Mainly domain-specific medical/leave versus project/status vocabulary and stylistic framing, not generalizable deception strategy; genuine illness claims may be flagged, with brittleness across domains and rephrasings. |
| 81 | W6 + J9 | 3 | Plausibly deception-relevant admission versus denial/hedging, mainly carried by surface/formal negations, hedges, résumé style, and enumerations; task- and wording-dependent, likely brittle across domains and phrasings. |
| 86 | W6 + J9 | 6 | Hedging and defensive PR-style wording are general deceptive or self-protective tactics potentially transferable across domains; legitimate privacy, politeness, and negotiation uses add noise, phrasing and role sensitivity, and false positives, limiting OOD reliability to moderate. |

Table 4 gives directions, OOD-score (R5) ratings, and interpretations. In the explanations, QA means question answering, and V8 means the Roleplaying source dataset.

Table 4: Directions, R5 ratings, and interpretations.

| Q7 index | R5 | Full explanation |
|---|---:|---|
| 7 | 3 | Pragmatic denial/deflection versus admission may correlate with deception in some settings; heavy dependence on explicit “No,” word order, “I” plus admission, and task-specific phrasing makes it brittle, likely to fail or mislead across domains or varied wording. |
| 8 | 4 | Narrow denial/excuse versus admission can correlate with evasion or deception in similar QA/V8 tasks; explicit “No”/“Yes” and common denial phrasing dominate, with likely brittleness under paraphrase, domain change, or subtler deception strategies. |
| 11 | 3 | Stylistic/pragmatic assertive first-person role/defense versus problem-reporting/negation can correlate with evasive or face-saving lies in some scenarios; surface-level, context-dependent, likely to fail across domains or phrasings. |
| 12 | 3 | Discourse style/forcefulness is a surface cue; overconfidence or over-assertion sometimes correlates with deceptive defensiveness, yet brittleness and domain dependence produce many false positives/negatives across speakers, topics, and cultural styles. |
| 4 | 6 | General communicative denial versus admission is deception-relevant and can transfer beyond particular prompts; dependence on explicit “Yes”/“No” turns and common defensive phrasing risks brittleness with differently worded denial or lies and subtler tactics. |
