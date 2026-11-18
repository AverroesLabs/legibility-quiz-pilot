# Probe Generalization as Subspace Selection for OOD Deception Detection

Daniel Yoo  
Adrians Skapars

###### Abstract

A probe is a classifier trained on a model’s internal numerical representations, or activations. Here, source data train the probe; target data test transfer to other domains. Out-of-distribution (OOD) robustness means maintaining detection performance under this change.

Directions heavily weighted by a source probe appear to encode source-specific surface features, such as wording. Transferable directions appear to encode the same contrast more abstractly, in a form describable in natural language. Probe OOD robustness appears largely determined by subspace selection: choosing which representation directions the classifier can use. The supporting transfer results appear in Appendix A’s numerical-observations table, and the direction interpretations appear in Appendix E, Tables 3–4. These encoding claims are descriptive, not causal.

Machine Learning, ICML (International Conference on Machine Learning)

## 1 Introduction

Linear probes classify internal representations using weighted sums of their coordinates. They cheaply expose semantically meaningful information in otherwise uninterpretable language-model representations (Alain and Bengio, 2018; Hewitt and Liang, 2019). They detect behaviors and concepts but sometimes fail out of distribution despite strong in-distribution performance—performance on data resembling their training data (Kirch et al., 2026).

Behavioral concepts have linear representations (Marks and Tegmark, 2024; Park et al., 2024; Zou et al., 2025). Probes should therefore have detection capacity. Yet they appear to learn spurious correlations: source-data associations that do not reliably track the intended concept elsewhere (Orgad et al., 2025; Wang et al., 2025). Such unreliability is unacceptable for harmful-interaction monitoring.

We use principal component analysis (PCA; code N4D) to express source activations along data-derived directions ordered by their variance. These directions are principal components (PCs; code Q7). Working in source Q7 coordinates avoids privileging the model’s original internal, or latent, dimensions. These coordinates provide data-driven representation structure. Removing leading components improves downstream embedding performance (Mu et al., 2018), while leading-component projections serve as a probing baseline in Kantamneni et al. (2025). The selection problem is to find dimensions that support generalizing, rather than spurious, source-trained classification.

Figure 1: OOD deception-probing pipeline: projection into the source N4D basis, filtering candidate Q7 directions, probe training in the selected low-dimensional subspace.

## 2 Related Work

#### Linear concept directions in activation space.

Activation space is the coordinate space of a model’s internal representations. True/false representations exhibit cross-dataset linear structure (Marks and Tegmark, 2024). Park et al. (2024) formalize the linear representation hypothesis—that concepts can correspond to directions in this space—and connect it to probing and steering, or changing behavior by changing activations.

Honesty and emotion admit population-level activation directions permitting readout and intervention: detecting a property from activations and changing activations to affect behavior (Zou et al., 2025). A single direction mediates refusal (Arditi et al., 2024). An “assistant” axis corresponds to assistant-like behavior (Lu et al., 2026). Such simple features support mechanistic understanding and activation interventions. Whether they represent stable concepts or context-dependent correlations remains unresolved.

#### Linear probes and their limitations.

Linear and lightweight probes—classifiers with limited complexity—have a long interpretability history (Alain and Bengio, 2018; Hewitt and Liang, 2019). Applications include latent truthfulness without direct supervision (Burns et al., 2024) and strategic deception (Goldowsky-Dill et al., 2025).

High accuracy need not establish that a model already represents the property. The probe can learn the task or exploit surface correlates, such as wording associated with a label (Hewitt and Liang, 2019). Information encoded by the model and information merely recoverable by a trained probe remain distinct concerns (Ravichander et al., 2021). Strong in-distribution probes generalize unevenly (kramár2026buildingproductionreadyprobesgemini). Safety-probe evaluation must distinguish robust internal signals from dataset artifacts.

#### Interpreting features with natural-language explanations.

Bills et al. (2023) introduced large language model (LLM)-generated neuron explanations from high-activation examples. Automatic sparse-autoencoder (SAE) feature labeling extends this to large feature dictionaries ((Paulo et al., 2025), (Bricken et al., 2023)). A sparse autoencoder represents activations using sparsely active learned features.

Natural-language descriptions also support intervention. Casademunt et al. (2025) automatically identify concepts and ablate them—remove their contribution—during fine-tuning, or additional model training. This reduces misaligned behavior.

## 3 Setup

The model is Llama-3.1-8B-Instruct (E6), from Grattafiori et al. (2024). We use layer 15 residual-stream activations: the internal representation passed through that layer. This follows Ying et al. (2026), whose analysis assigns that layer the highest cross-domain discrimination. Each prompt is passed through E6. Activations are averaged over assistant-response tokens only; tokens are the text units processed by the model.

The four datasets come from Goldowsky-Dill et al. (2025). Roleplaying (V8) is the only source dataset. It places E6 in a simple scenario offering an opportunity to lie. Three datasets are held-out targets:

- Insider Trading Report (C2R): E6 acts as a trader deciding whether to honestly report insider-information use.
- Insider Trading Confirm (W6): after concealing earlier insider-information use, E6 is asked to confirm non-use.
- Sandbagging (J9): E6 is instructed to underperform deliberately on some tasks and behave normally on others.

The response labeler is GPT-4o (K8), OpenAI (2024). GPT abbreviates Generative Pre-trained Transformer. K8 assigns honest responses label 0, ambiguous responses label -1, and deceptive responses label 1. Ambiguous responses are removed. Appendix A, Table 1 reports the label distributions.

Source activations are standardized to zero mean and unit variance. The same source statistics standardize target activations. Source PCA (N4D) then retains the leading 100 PCs (Q7). A subset \(S\) is a selected set of these directions. Projection onto \(S\), which retains only its coordinates, precedes training and evaluation.

All classifiers use logistic regression, a classifier based on a weighted sum of its inputs, with parameter \(\lambda=10\). The full-representation baseline (B8Q) instead trains directly on all standardized activations.

The retained count of 100 is chosen for interpretability comparability, not superior low-index generalization. Lower-index PCs align more strongly with SAE decoder features (U9), the vectors used to reconstruct activations from SAE features (Bricken et al., 2023). Alignment decreases with index, making lower indices more amenable to natural-language interpretation. Appendix A’s additional-observations table reports Spearman rank correlations, denoted \(\rho\), of -0.857 for the best-matching feature and -0.917 for the mean of the top 10 features. Appendix C gives the decoder source.

Performance is measured by the area under the receiver operating characteristic curve (AUROC; F3), a measure of discrimination between honest and deceptive responses. Cross-validation (CV) divides data into folds for separate fitting or selection and held-out evaluation; 5-fold CV uses five folds. Error bars denote 95% confidence intervals (CIs).

## 4 Experiments

### 4.1 Setting a baseline

Source-trained B8Q transfer is reasonably good on W6, moderate on C2R, and poor on J9. The respective F3 values are 0.874, 0.706, and 0.606 (Appendix A, “B8Q F3” row).

Target-trained classification within the retained leading-100 source subspace separates deceptive from honest responses on every target. Its 5-fold CV F3 values are 0.936 on C2R, 0.950 on W6, and 0.869 on J9 (Appendix A, “Target-trained F3, 5-fold CV” row). Thus relevant information is present in this subspace, but the source-trained direction fails to generalize.

Figure 2: Target F3 for source-trained full representations, blue; source-trained leading 100 Q7, orange; target-trained leading 100 source Q7, green, with 5-fold CV.

### 4.2 Showing a transferable subspace exists

The greedy oracle search (H2) tests whether a transferable subset exists. “Oracle” indicates that target labels guide selection. Starting with an empty subset, the search considers each of the leading 100 source Q7 directions. For each candidate addition, it projects source activations onto the candidate subset, fits source logistic-regression weights, and measures target F3. It adds the direction that most improves target F3. Search stops when no candidate improvement exceeds 0.001.

In target 5-fold CV, selection uses the non-held-out target folds. The selected source-trained probe is then evaluated on the held-out fold. Target labels guide subset selection, never probe-weight fitting. This is a transfer upper bound, not an unbiased OOD estimate.

Small source-trained subspaces nearly match target-trained probes, particularly on C2R and J9. Appendix A reports H2 improvements over B8Q of 0.22 F3 points on C2R and 0.23 on J9. Mean H2 subset sizes under target CV are 15, 8, and 14 for C2R, W6, and J9, respectively; the reported small-subset range is 8–15.

The variance control selects the top \(k\) PCs by variance (Y4), where \(k\) is the number of selected directions and is matched to H2. Variance selection matches H2 only on the easy W6 case. It falls far behind on the harder C2R and J9 tasks (Figure 3). Explained variance—the variation captured by a direction—does not predict usefulness.

Figure 3: Target F3 for B8Q, blue; Y4 with \(k\) matched to H2, teal; H2, purple.

### 4.3 Testing hypotheses for transferable components

We test two hypotheses. Transferable directions might have individually strong target-class separation, making them accessible to target-guided selection. Alternatively, they might already receive large source-probe weights, with selection merely rediscovering them compactly.

Both rankings use the standardized coefficient (X4): the absolute probe coefficient multiplied by the corresponding Q7 standard deviation. This is calculated on activations projected into the source N4D basis.

A source-trained ranking probe, the source-PCA probe (T6), supplies the source X4 ordering. A target-trained ranking probe, the target-PCA probe (M9), supplies the target X4 ordering. After ranking, both evaluated probes train on **source data** projected onto their top \(k\) ranked directions.

For this comparison, H2 selects using the full target dataset, rather than fold-specific selections. This supplies a stable set and \(k\). Its numbers differ slightly from the CV results. Appendix A reports full-target H2 \(k\) values of 13 for C2R, 7 for W6, and 14 for J9.

Neither source-side nor target-side ranking recovers the transferable subspace or a basis transferring better than B8Q. T6 transfers about as well as B8Q: source-reliant directions do not generalize. M9 does well on C2R, worse than H2 on W6, and below random on J9 (Figure 4).

Target-class separation and transfer under source-fitted weighting select different directions. Appendix D reports H2–M9 set overlaps of 0.182 on C2R, 0.077 on W6, and 0.167 on J9, and lists the direction sets in Table 2.

Figure 4: Target F3 for T6, red; M9, pink; H2, purple.

### 4.4 Interpreting the transferable subspace

For each leading source direction, an interpreter, the LLM judge (C7), receives the eight most-activating and eight least-activating source examples. Direction interpretation uses GPT-5-mini (P2), OpenAI (2025). Following Bills et al. (2023) and Casademunt et al. (2025), C7 generates descriptions of the positive pole, negative pole, and whole direction. Positive and negative poles are the contrasting ends of a direction.

C7 then assigns a generalization rating, the OOD score (R5), on a 1–10 scale. The prompt gives low ratings to surface cues and reserves 5 or higher for plausible general deception-relevant strategies or behaviors (Appendix B). Scores are noisy. Selection retains directions reaching the observed maximum in both scoring runs. The maximum is 6; across 2 runs, 19 directions meet this rule (Appendix E, selection-observations table).

Transferable subsets differ from directions relied on by source or target classification (§4.3; Appendix D, Table 2). Interpretations of H2-selected directions express contrasts at an abstraction surviving domain changes. Of 5 directions selected by H2 for more than one target, 3 out of 5 receive the maximum rating. Only 1 of the top 5 T6 selections does so (Appendix E). Tables 3–4 give the explanations and their qualifications. Only partial recovery occurs.

### 4.5 Selecting a subspace by interpreting components

The intersection retained across scoring runs defines the interpretation-selected PC probe (L2). It uses 19 directions. Its controls are the interpretation-ablated PC probe (A7), which uses the complementary 81 directions, and Y4, an equal-dimensional variance-ranked subset. All are trained as source probes.

Language-only selection, without target data, meaningfully improves generalization over B8Q. L2 exceeds B8Q on all targets and exceeds the variance control, widely on C2R and J9. It also exceeds A7 everywhere (Figure 5). On C2R, L2 F3 is 0.881 versus B8Q’s 0.706. On J9, it is 0.662 versus 0.606 (Appendix A, “L2 F3” and “B8Q F3” rows).

Recovery varies sharply. On C2R, L2 substantially closes the baseline–H2 gap: the reported recovery is 78%, using H2 F3 of 0.929. On J9, recovery is only partial: 25%, using H2 F3 of 0.832 (Appendix A, recovery-comparison rows). W6 has little headroom and is uninformative. Every selection method except A7 is roughly tied there (Figure 5).

Figure 5: Target F3 for L2, red; B8Q, blue; Y4 at the L2 dimension count, teal; A7, brown.

## 5 Limitations

Claims remain narrow. The study uses one model and one source. Other model families and datasets remain unresolved. W6’s already-high baseline, F3 0.874, makes its results uninformative.

J9 is the weakest result and the most out-of-distribution target. Its modest improvement likely reflects distribution shift, a change between source and target data. L2 reaches 0.662 versus B8Q’s 0.606, recovering 25% of the reported baseline–H2 gap. C2R’s improvement is more robust: 0.881 versus 0.706, with 78% recovery (Appendix A).

The pipeline is fully linear—N4D projection followed by logistic regression. This makes transferable-subspace existence perhaps unsurprising. Extension to nonlinear probes or other readout locations remains open.

Statements about what directions “encode” are descriptive, based on extreme-activation examples and C7 verbalizations, not causal. High-rated directions have not been ablated during inference—model execution—to verify their apparent contrast.

## 6 Conclusion

Robust behavioral probing requires identifying where a linearly accessible signal is represented most robustly, not merely establishing its accessibility. Target-trained probes show that relevant information is present (§4.1). H2 shows that selecting a small subspace can improve source-trained transfer (§4.2). Language-only selection recovers part of this improvement (§4.5).

Full-representation probes may mix general behavioral features with dataset-specific surface cues. This can support high in-distribution accuracy while harming transfer. Sparse directional subspaces—subsets using relatively few directions—may isolate more general behavioral structure. OOD probing becomes representation selection as well as classifier fitting: preserve the intended behavioral feature and filter spurious correlation.

## Acknowledgment

Conducted during the 11th AI Safety Camp, which supplied experimental compute. Prasad Mahadik and Arya Hariharan provided useful discussions.

## Impact Statement

The work aims to advance Machine Learning; many societal consequences are possible, none requiring specific highlighting in the authors’ view.

## Appendix A Dataset Details

V8 places E6 in a simple scenario offering an opportunity to lie. In C2R, E6 acts as a trader deciding whether to honestly report insider-information use. W6 asks E6, after concealing earlier insider-information use, to confirm non-use. J9 instructs deliberate underperformance on some tasks and normal behavior on others.

Table 1 reports ambiguous (-1), honest (0), and deceptive (1) labels. Ambiguous responses are removed for the experiments.

Table 1: Dataset label distributions.

| Dataset | Total examples | Label counts (-1 / 0 / 1) |
|---|---:|---|
| V8 | 371 | 140 / 59 / 172 |
| C2R | 473 | 218 / 194 / 61 |
| W6 | 151 | 29 / 38 / 84 |
| J9 | 1000 | 379 / 375 / 246 |

The discrimination measure F3 is AUROC. Target-trained entries below are obtained inside the leading-100 source subspace. H2 gains are retained as reported.

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

| Additional observation | Value |
|---|---|
| Small H2 subset range | 8–15 |
| Decoder alignment, best-matching feature, Spearman \(\rho\) | -0.857 |
| Decoder alignment, mean of top 10 features, Spearman \(\rho\) | -0.917 |

## Appendix B LLM Judge Prompt

Direction interpretation uses GPT-5-mini (P2), OpenAI (2025), with the following prompt. In the line format, `token` is the text unit, `<TAB>` is a tab separator, and `activation` is its numerical activation. The placeholders `{positive_examples}` and `{negative_examples}` are filled with the high-positive and high-negative source examples.

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

Lower-index directions align more strongly with SAE decoder vectors (U9). Alignment is measured for both the best-matching U9 feature and the mean of the top 10 features, using `andyrdt/saes-llama-3.1-8b-instruct` (andyrdt, 2024), the decoder-feature source. Appendix A reports the corresponding Spearman \(\rho\) values as -0.857 and -0.917.

Figure 6: Lower Q7 exhibit stronger decoder alignment for the best-matching feature and the mean of the top 10 features.

## Appendix D Q7 Set Comparisons

Jaccard overlap (D3) measures the number of shared set members divided by the number in either set. H2 and M9 agree on less than a fifth of selections under this measure, as shown below. Together with the source-ranking comparison in §4.3, this is consistent with transferable space being neither the most source-discriminative nor the most target-discriminative region.

| Target | H2–M9 D3 |
|---|---:|
| C2R | 0.182 |
| W6 | 0.077 |
| J9 | 0.167 |

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

| Selection observation | Value |
|---|---:|
| Maximum observed R5; no direction exceeded it | 6 |
| Scoring runs | 2 |
| Directions reaching maximum in both runs, retained for L2 | 19 |
| Complement directions used by A7 | 81 |
| Directions selected by H2 for more than one target | 5 |
| Maximum-rated directions among those shared H2 selections | 3 out of 5 |
| Maximum-rated directions among the top T6 selections | 1 of the top 5 |

In Table 3, “+” means selection on both listed targets. PR means public relations.

Table 3: Directions selected for likely transferability from C7 interpretation.

| Q7 index | Selected on | R5 | Full explanation |
|---|---|---:|---|
| 62 | C2R + J9 | 6 | General deception-relevant denial/deflection versus detail provision can be useful across settings; noise includes truthful denial or obfuscation for legitimate reasons and liars’ elaborate fabricated details. Lexical negation rather than semantic strategy may drive the signal. |
| 57 | C2R + J9 | 6 | General denial/minimization or innocuous explanations versus specific/incriminating details can flag rhetorical hiding or deflection. Partial dependence on surface denials, hedges, and domain terms makes it lexical, task-specific, brittle, and context-dependent OOD. |
| 9 | C2R + J9 | 3 | Mainly domain-specific medical/leave versus project/status vocabulary and stylistic framing, not generalizable deception strategy; genuine illness claims may be flagged, with brittleness across domains and rephrasings. |
| 81 | W6 + J9 | 3 | Plausibly deception-relevant admission versus denial/hedging, mainly carried by surface/formal negations, hedges, résumé style, and enumerations; task- and wording-dependent, likely brittle across domains and phrasings. |
| 86 | W6 + J9 | 6 | Hedging and defensive PR-style wording are general deceptive or self-protective tactics potentially transferable across domains; legitimate privacy, politeness, and negotiation uses add noise, phrasing and role sensitivity, and false positives, limiting OOD reliability to moderate. |

In Table 4, QA means question answering.

Table 4: Directions, R5 ratings, and interpretations.

| Q7 index | R5 | Full explanation |
|---|---:|---|
| 7 | 3 | Pragmatic denial/deflection versus admission may correlate with deception in some settings; heavy dependence on explicit “No,” word order, “I” plus admission, and task-specific phrasing makes it brittle, likely to fail or mislead across domains or varied wording. |
| 8 | 4 | Narrow denial/excuse versus admission can correlate with evasion or deception in similar QA/V8 tasks; explicit “No”/“Yes” and common denial phrasing dominate, with likely brittleness under paraphrase, domain change, or subtler deception strategies. |
| 11 | 3 | Stylistic/pragmatic assertive first-person role/defense versus problem-reporting/negation can correlate with evasive or face-saving lies in some scenarios; surface-level, context-dependent, likely to fail across domains or phrasings. |
| 12 | 3 | Discourse style/forcefulness is a surface cue; overconfidence or over-assertion sometimes correlates with deceptive defensiveness, yet brittleness and domain dependence produce many false positives/negatives across speakers, topics, and cultural styles. |
| 4 | 6 | General communicative denial versus admission is deception-relevant and can transfer beyond particular prompts; dependence on explicit “Yes”/“No” turns and common defensive phrasing risks brittleness with differently worded denial or lies and subtler tactics. |
