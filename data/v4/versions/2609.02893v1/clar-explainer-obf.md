**Summary.** The paper asks whether selecting subspaces of language-model representations improves out-of-distribution (OOD) deception detection. It compares probes—classifiers trained to detect a property—in full representations and selected directions. One selection method uses target labels without fitting probe weights on them. Another uses natural-language interpretations without target data. Small, target-guided subspaces nearly match target-trained probes. Selecting directions by source weights, target-class separation, or explained variance does not recover the same transferable subspaces. Interpretation-based selection improves over the full-representation baseline on all targets, but recovery varies sharply. The study uses one model and one source dataset. Its interpretations are descriptive, not causal.

**Glossary.**

- **Activation / representation / latent dimensions** — Internal model values in which probes detect information; this study uses residual-stream activations.
- **Activation space / subspace / direction / basis** — The representation space, a selected part of it, an individual direction, and the directions used as coordinates.
- **Linear probe / classifier / readout** — A detector fitted to representations; all evaluated classifiers here use logistic regression.
- **Probe weights / coefficients** — Values fitted by a probe to weight representation directions.
- **Source / source-trained** — The Roleplaying dataset, or training on that dataset.
- **Target / held-out target / target-trained** — A transfer-evaluation dataset, or training on that dataset.
- **In-distribution / OOD / out of distribution / transfer / generalization** — Performance within the training setting, or performance beyond it on held-out targets.
- **Behavioral concept / linear representation** — A behavior or concept described as accessible through a linear direction.
- **Surface feature / surface correlate / spurious correlation / dataset artifact** — Source- or wording-specific information that a probe may use without robust transfer.
- **Intervention / steering / ablation** — Changing behavior through activation directions or removing identified concepts or features.
- **Encoded information / recoverable information** — Information already represented by a model versus information a probe can recover; the paper distinguishes these concerns.
- **LLM / interpreter / judge / C7** — The language-model judge that describes directions and rates their likely OOD usefulness.
- **Natural-language explanation / interpretation** — A verbal description inferred from highly positive and highly negative activation examples.
- **Positive pole / negative pole / bipolar direction** — The two opposing sides of a direction, matched by positive and negative activations.
- **Machine Learning / ICML** — The field and accompanying label printed below the abstract.
- **E6 / Llama-3.1-8B-Instruct** — The model supplying layer 15 residual-stream activations.
- **Residual stream / readout location** — The model location from which activations are taken; this study uses layer 15.
- **Assistant-response tokens** — The response tokens over which activations are averaged.
- **V8 / Roleplaying** — The sole source dataset, with scenarios offering an opportunity to lie.
- **C2R / Insider Trading Report** — The target asking a trader whether to honestly report insider-information use.
- **W6 / Insider Trading Confirm** — The target asking for confirmation of non-use after earlier insider-information use was concealed.
- **J9 / Sandbagging** — The target instructing deliberate underperformance on some tasks and normal behavior on others.
- **K8 / GPT-4o** — The response labeler.
- **Labels -1 / 0 / 1** — Ambiguous, honest, and deceptive responses, respectively; ambiguous responses are removed.
- **Standardization** — Zero-mean, unit-variance transformation using source statistics for both source and targets.
- **PCA / N4D** — The transformation retaining the leading 100 source directions.
- **PC / PCs / Q7** — Principal component or principal components: the source directions retained after N4D.
- **Leading / lower-index components** — Components earlier in the retained ordering, which show stronger decoder alignment.
- **\(S\)** — The subset of directions onto which activations are projected.
- **Projection** — Restricting activations to the selected directions before training and evaluation.
- **Logistic regression / \(\lambda\)** — The classifier used throughout; its stated parameter is \(\lambda=10\).
- **B8Q / full-representation baseline** — A source-trained probe using all standardized activations.
- **Explained variance / variance ranking** — The criterion used to select leading components for the variance control.
- **\(k\) / top-\(k\)** — The selected direction count, or the first \(k\) directions in a ranking.
- **Y4 / top-k PCs by variance** — A source-trained probe using a variance-ranked subset matched to another method’s dimension count.
- **H2 / greedy oracle** — Target-guided greedy direction selection, with probe weights fitted only on source data.
- **CV / 5-fold CV** — Cross-validation using five folds; selection uses non-held-out target folds and evaluation uses the held-out fold.
- **Transfer upper bound** — The role assigned to target-label-guided selection, not an unbiased OOD estimate.
- **F3 / AUROC** — The recorded discrimination measure used to report performance.
- **CI / error bars** — Confidence intervals; error bars denote 95% CIs.
- **T6 / source-PCA probe** — The source-trained ranking probe supplying source coefficient rankings.
- **M9 / target-PCA probe** — The target-trained ranking probe supplying target coefficient rankings.
- **X4 / standardized coefficient** — Absolute probe coefficient multiplied by the corresponding Q7 standard deviation.
- **R5 / OOD score / generalization rating** — C7’s rating of a direction’s usefulness for OOD deception detection.
- **L2 / interp-selected PC probe** — A source-trained probe using directions reaching the observed maximum R5 in both scoring runs.
- **A7 / interp-ablated PC probe** — A source-trained probe using the complement of the L2 directions.
- **P2 / GPT-5-mini** — The model used for direction interpretation.
- **SAE / sparse autoencoder / U9 / SAE decoder features** — The feature dictionary and decoder vectors used to assess direction alignment.
- **Decoder alignment** — Alignment with the best-matching U9 feature or the mean of the top 10 features.
- **Spearman \(\rho\)** — The statistic reported for the relationship between component index and decoder alignment.
- **`andyrdt/saes-llama-3.1-8b-instruct`** — The resource used for decoder-alignment measurements.
- **D3 / Jaccard overlap** — The set measure used to compare selected directions.
- **`token<TAB>activation`** — The prompt’s example format: a token, a tab separator, and an activation.
- **`{positive_examples}` / `{negative_examples}`** — Prompt placeholders for high-positive and high-negative source examples.
- **QA/V8** — The similar task settings named in the interpretation of Q7 8.
- **PR-style wording** — The defensive wording named in the interpretation of Q7 86.
- **AI Safety Camp** — The 11th camp, during which the work was conducted and which supplied experimental compute.

# Probe Generalization as Subspace Selection for OOD Deception Detection

Daniel Yoo  
Adrians Skapars

###### Abstract

*Source-weighted and transferable directions appear to differ in abstraction, making subspace selection central to probe robustness.*

Directions heavily weighted by a source probe—a classifier trained on source data—appear to encode source-specific surface features. Transferable directions appear to encode the same contrast more abstractly, in a form describable in natural language. Probe OOD robustness appears largely determined by subspace selection: choosing which representation directions the probe uses.

**Claims and evidence**
- Source-weighted and transferable directions appear different: Appendix E, Table 4, Q7 7, 8, 11, 12, and 4 have R5 ratings 3, 4, 3, 3, and 6; Table 3, Q7 62, 57, and 86 each have R5 6. Their explanations state the surface-feature and transferability caveats.
- Subspace selection affects transfer: Figures 3–5; Appendix A, “B8Q F3,” “H2 F3 used for recovery comparison,” and “L2 F3” rows report C2R 0.706, 0.929, 0.881 and J9 0.606, 0.832, 0.662.

Machine Learning, ICML

## 1 Introduction

*Linear probes can detect meaningful information but may fail to transfer; the paper studies which representation directions support generalization.*

Linear probes cheaply expose semantically meaningful information in otherwise uninterpretable language-model representations, or internal model values (Alain and Bengio, 2018; Hewitt and Liang, 2019). They detect behaviors and concepts but sometimes fail out of distribution despite strong in-distribution performance (Kirch et al., 2026). Behavioral concepts have linear representations (Marks and Tegmark, 2024; Park et al., 2024; Zou et al., 2025). Probes should therefore have detection capacity, yet appear to learn spurious correlations in practice: correlations that do not transfer robustly (Orgad et al., 2025; Wang et al., 2025). Such unreliability is unacceptable for harmful-interaction monitoring.

Working in source principal-component coordinates (PCs: Q7) avoids privileging the model’s latent dimensions. These coordinates provide data-driven representation structure. Removing leading components improves downstream embedding performance (Mu et al., 2018), while projections onto leading components serve as a probing baseline in Kantamneni et al. (2025). The selection problem is to find dimensions that permit generalizing rather than spurious source-trained classification. Here, N4D denotes the PCA transformation that produces the source basis.

Figure 1: OOD deception-probing pipeline: projection into the source N4D basis, filtering candidate Q7 directions, probe training in the selected low-dimensional subspace.

## 2 Related Work

*Prior work establishes simple concept directions, limits of probe evidence, and natural-language methods for interpreting features.*

#### Linear concept directions in activation space.

*Simple activation directions support readout and intervention, but their stability across contexts remains unresolved.*

True/false representations exhibit cross-dataset linear structure (Marks and Tegmark, 2024). Park et al. (2024) formalize the linear representation hypothesis and connect it to probing and steering, or behavior changes through activation directions. Honesty and emotion admit population-level activation directions that permit readout and intervention (Zou et al., 2025). A single direction mediates refusal (Arditi et al., 2024); an “assistant” axis corresponds to assistant-like behavior (Lu et al., 2026). Such simple features support mechanistic understanding and activation interventions. They leave unresolved the distinction between stable concepts and context-dependent correlations.

#### Linear probes and their limitations.

*Probe accuracy does not by itself establish robust internal representation of a property.*

Linear and lightweight probes have a long interpretability history (Alain and Bengio, 2018; Hewitt and Liang, 2019), including latent truthfulness without direct supervision (Burns et al., 2024) and strategic deception (Goldowsky-Dill et al., 2025). High accuracy need not establish that a model already represents the property. The probe can learn the task or exploit surface correlates (Hewitt and Liang, 2019). Encoded information and merely recoverable information remain distinct concerns (Ravichander et al., 2021). Strong in-distribution probes generalize unevenly (kramár2026buildingproductionreadyprobesgemini). Safety-probe evaluation must distinguish robust internal signals from dataset artifacts.

#### Interpreting features with natural-language explanations.

*Language-model explanations describe activated features and can support interventions.*

Bills et al. (2023) introduced language-model-generated (LLM-generated) neuron explanations from high-activation examples. Automatic sparse-autoencoder feature labeling extends this approach to large feature dictionaries ((Paulo et al., 2025), (Bricken et al., 2023)). Natural-language descriptions also support intervention. Casademunt et al. (2025) automatically identify concepts and ablate them—remove them—during fine-tuning, reducing misaligned behavior.

## 3 Setup

*The experiments use one model, one source dataset, three targets, and linear classifiers on standardized activations.*

The model, Llama-3.1-8B-Instruct (E6), from Grattafiori et al. (2024), supplies layer 15 residual-stream activations. This follows Ying et al. (2026), whose analysis assigns that layer the highest cross-domain discrimination. Each prompt is passed through E6. Activations are averaged over assistant-response tokens only.

The four Goldowsky-Dill et al. (2025) datasets comprise one source, Roleplaying (V8), and three held-out targets: a reporting task, Insider Trading Report (C2R); a confirmation task, Insider Trading Confirm (W6); and an underperformance task, Sandbagging (J9). No other source dataset is used. The response labeler, GPT-4o (K8), OpenAI (2024), assigns honest 0, ambiguous -1, and deceptive 1. Ambiguous responses are removed.

Source activations receive zero-mean, unit-variance standardization. The same source statistics standardize targets. The subsequent PCA transformation, N4D, retains the leading 100 principal components, Q7. Activations are projected onto a direction subset \(S\) before training and evaluation. All classifiers use logistic regression with \(\lambda=10\). The full-representation baseline (B8Q) instead trains directly on all standardized activations.

Interpretability comparability, not superior low-index generalization, determines the retained 100. Alignment with sparse-autoencoder decoder features—decoder vectors called U9 (Bricken et al., 2023)—decreases with index. Lower indices are therefore more amenable to natural-language interpretation. Error bars denote 95% confidence intervals (CIs).

## 4 Experiments

*The experiments compare full representations, target-guided subspaces, coefficient rankings, and interpretation-based selection.*

### 4.1 Setting a baseline

*Target-trained probes separate classes within the retained source subspace even where source-trained transfer is poor.*

W6 transfer is reasonably good, C2R moderate, and J9 poor. Target-trained classification within the retained source subspace separates deceptive from honest responses on every target. Relevant information is present; the source-trained direction fails to generalize.

The recorded discrimination measure is AUROC (F3). Five-fold cross-validation (CV) evaluates target-trained probes across five folds.

Figure 2: Target F3 for source-trained full representations, blue; source-trained leading 100 Q7, orange; target-trained leading 100 source Q7, green, with 5-fold CV.

**Claims and evidence**
- Baseline transfer varies across targets: Appendix A, “B8Q F3” row—W6 0.874, C2R 0.706, J9 0.606.
- The retained source subspace supports target-trained separation on every target: Appendix A, “Target-trained F3, 5-fold CV” row—C2R 0.936, W6 0.950, J9 0.869. Figure 2 compares these probes with source-trained probes.

### 4.2 Showing a transferable subspace exists

*Target-guided selection finds small source-trained subspaces that transfer well, but this procedure is an upper bound rather than an unbiased OOD estimate.*

Small source-trained subspaces nearly match target-trained probes, particularly on C2R and J9. Variance selection—top-k PCs by variance (Y4)—matches the greedy oracle (H2), the target-guided search described below, only on the easy W6 case. It falls far behind on the harder C2R and J9 tasks. Explained variance does not predict usefulness. Here, \(k\) is the selected direction count.

Figure 3: Target F3 for B8Q, blue; Y4 with \(k\) matched to H2, teal; H2, purple.

The search starts with an empty subset. It considers each of the leading 100 source Q7 directions. For each candidate addition, it projects source activations onto that subset and fits source logistic-regression weights. It then adds the direction that most improves target F3.

Target 5-fold CV selects directions on non-held-out target folds and evaluates the selected source-trained probe on the held-out fold. Search stops when no candidate improvement exceeds 0.001. Target labels guide subset selection, never probe-weight fitting. This is a transfer upper bound, not an unbiased OOD estimate.

**Claims and evidence**
- Small selected subspaces transfer well: Appendix A, “Mean H2 subset size, target CV” row—C2R 15, W6 8, J9 14; “Small H2 subset range” row—8–15.
- H2 improves substantially on C2R and J9: Appendix A, “H2 improvement over B8Q, F3 points” row—0.22 and 0.23; “H2 F3 used for recovery comparison” row—0.929 and 0.832. Target-trained F3 is 0.936 and 0.869 in the corresponding row.
- Variance selection matches H2 only on W6 and falls behind on C2R and J9: Figure 3; exact Y4 values are not supplied.

### 4.3 Testing hypotheses for transferable components

*Neither source-weight nor target-separation rankings recover the transferable subspace.*

Neither source-side nor target-side ranking recovers the transferable subspace or a basis transferring better than B8Q. The source-PCA probe (T6), a source-trained ranking probe, transfers about as well as B8Q: source-reliant directions do not generalize. The target-PCA probe (M9), a target-trained ranking probe, does well on C2R, worse than H2 on W6, and below random on J9. Target-class separation and transfer under source-fitted weighting select different directions.

Figure 4: Target F3 for T6, red; M9, pink; H2, purple.

The two hypotheses identify transferable directions as either:
- Directions with individually strong target-class separation, accessible to target-guided selection.
- Directions already heavily weighted by the source probe, compactly rediscovered by selection.

Both rankings use the standardized coefficient (X4): the absolute probe coefficient multiplied by the corresponding Q7 standard deviation. They compute this measure on source-N4D-projected activations.

T6 and M9 supply their respective X4 orderings. Both evaluated probes then train on source data projected onto their top \(k\) ranked directions. Full-target H2 selection supplies a stable set and \(k\), rather than fold-specific selections. Its resulting numbers differ slightly from CV results.

**Claims and evidence**
- T6 performs about like B8Q; M9 does well on C2R, worse than H2 on W6, and below random on J9: Figure 4; exact T6 and M9 F3 values are not supplied.
- Target separation and transfer select different directions: Appendix D, “H2–M9 D3” table—C2R 0.182, W6 0.077, J9 0.167; Table 2 gives the direction sets.
- Full-target selection uses target-specific \(k\): Appendix A, “Full-target H2 \(k\)” row—C2R 13, W6 7, J9 14.

### 4.4 Interpreting the transferable subspace

*Direction descriptions suggest transferable contrasts at a more domain-stable abstraction, but recovery is partial and scores are noisy.*

Transferable subsets differ from directions relied on by source or target classification. Interpretations of H2-selected directions express contrasts at an abstraction that survives domain changes. Only partial recovery occurs.

For each leading source direction, an LLM judge (C7) receives the eight most-activating and eight least-activating source examples. Following Bills et al. (2023) and Casademunt et al. (2025), it generates positive-pole, negative-pole, and whole-direction descriptions. The poles are the opposing sides associated with positive and negative activations. It then assigns a generalization rating, or OOD score (R5), to the direction. Selection retains all directions reaching the observed maximum in both scoring runs. Scores are noisy.

**Claims and evidence**
- H2 interpretations include potentially transferable contrasts and explicit limitations: Appendix E, Table 3, Q7 62, 57, and 86 each have R5 6; Q7 9 and 81 each have R5 3.
- Recovery is partial: Appendix E, “Maximum-rated directions among those shared H2 selections” row—3 out of 5; “Maximum-rated directions among the top T6 selections” row—1 of the top 5.
- Selection requires agreement across noisy runs: Appendix E reports maximum R5 6, 2 scoring runs, and 19 directions reaching that maximum in both runs.

### 4.5 Selecting a subspace by interpreting components

*Interpretation-only selection improves transfer without target data, but the amount of recovery differs sharply by target.*

Language-only selection, without target data, meaningfully improves generalization over B8Q. The interp-selected PC probe (L2) uses the intersection of directions retained across both scoring runs. It exceeds B8Q on all targets and the variance control, widely on C2R and J9. It also exceeds the interp-ablated PC probe (A7), which uses the complement, everywhere.

Recovery varies sharply. C2R substantially closes the baseline–H2 gap; J9 closes it only partially. W6 has little headroom and is uninformative, with every selection method except A7 roughly tied.

Figure 5: Target F3 for L2, red; B8Q, blue; Y4 at the L2 dimension count, teal; A7, brown.

The retained intersection defines L2. Its controls are A7 and Y4, an equal-dimensional variance-ranked subset. All are trained as source probes.

**Claims and evidence**
- L2 improves over B8Q on C2R and J9: Appendix A, “L2 F3” and “B8Q F3” rows—C2R 0.881 versus 0.706; J9 0.662 versus 0.606.
- Recovery differs sharply: Appendix A, “Baseline–H2 gap recovered by L2” row—C2R 78%, J9 25%.
- L2 exceeds B8Q, Y4, and A7 on all targets; W6 methods except A7 are roughly tied: Figure 5. Exact W6 L2, Y4, and A7 values are not supplied.
- L2 and A7 use complementary subsets: Appendix E, retained-direction and complement rows—19 and 81 directions.

## 5 Limitations

*The results have narrow model and dataset scope, and the direction interpretations have not been causally verified.*

Claims remain narrow. The single-model, single-source scope leaves other model families and datasets unresolved. W6’s already-high baseline makes its results uninformative.

J9 is the weakest result and the most out-of-distribution target. Its modest improvement likely reflects distribution shift. C2R’s improvement is more robust.

The pipeline is fully linear: N4D projection followed by logistic regression. This makes transferable-subspace existence perhaps unsurprising. Extension to nonlinear probes or other readout locations remains open.

Statements about what directions “encode” are descriptive. They are based on extreme-activation examples and C7 verbalizations, not causal evidence. High-rated directions have not been ablated during inference to verify their apparent contrast.

**Claims and evidence**
- W6 has a high baseline: Appendix A, “B8Q F3” row—0.874.
- J9 improves modestly, while C2R improves more: Appendix A, “B8Q F3” and “L2 F3” rows—J9 0.606 and 0.662; C2R 0.706 and 0.881. The recovery row reports 25% and 78%, respectively.
- Scope and causal limits are methodological: Section 3 specifies one model and one source; Section 4.4 describes example-based verbal interpretation. No inference-time ablation result is reported.

## 6 Conclusion

*Robust probing requires selecting representations that preserve the intended behavior while filtering spurious correlations.*

Robust behavioral probing requires identifying where a linearly accessible signal is represented most robustly, not merely establishing its accessibility. Full-representation probes may mix general behavioral features with dataset-specific surface cues. This can support high in-distribution accuracy while harming transfer. Sparse directional subspaces may isolate more general behavioral structure. OOD probing therefore becomes representation selection as well as classifier fitting: preserve the intended behavioral feature and filter spurious correlation.

**Claims and evidence**
- Selected subspaces can outperform full representations in transfer: Appendix A, “B8Q F3,” “H2 F3 used for recovery comparison,” and “L2 F3” rows—C2R 0.706, 0.929, 0.881; J9 0.606, 0.832, 0.662.
- Interpretations distinguish potentially general contrasts from surface cues: Appendix E, Tables 3 and 4; these remain descriptive, as stated in Section 5.

## Acknowledgment

*The work was conducted during AI Safety Camp, with compute and discussion support.*

Conducted during the 11th AI Safety Camp, which supplied experimental compute. Prasad Mahadik and Arya Hariharan provided useful discussions.

## Impact Statement

*The authors identify no societal consequence requiring specific highlighting.*

The work aims to advance Machine Learning; many societal consequences are possible, none requiring specific highlighting in the authors’ view.

## Appendix A Dataset Details

*This appendix describes the datasets, label distributions, and recorded numerical observations.*

V8 places E6 in a simple scenario offering an opportunity to lie. In C2R, E6 acts as a trader deciding whether to honestly report insider-information use. W6 asks E6, after concealing earlier insider-information use, to confirm non-use. J9 instructs deliberate underperformance on some tasks and normal behavior on others.

Table 1: Dataset label distributions.

| Dataset | Total examples | Label counts (-1 / 0 / 1) |
|---|---:|---|
| V8 | 371 | 140 / 59 / 172 |
| C2R | 473 | 218 / 194 / 61 |
| W6 | 151 | 29 / 38 / 84 |
| J9 | 1000 | 379 / 375 / 246 |

The recorded discrimination measure, AUROC (F3), supplies the following numerical observations. Target-trained entries are obtained inside the leading-100 source subspace. H2 gains are retained as reported.

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

Decoder alignment measures correspondence with U9 decoder vectors. Spearman \(\rho\) is the reported statistic relating alignment to direction index.

| Additional observation | Value |
|---|---|
| Small H2 subset range | 8–15 |
| Decoder alignment, best-matching feature, Spearman \(\rho\) | -0.857 |
| Decoder alignment, mean of top 10 features, Spearman \(\rho\) | -0.917 |

**Claims and evidence**
- Dataset sizes and labels differ: Table 1, V8—371 and 140 / 59 / 172; C2R—473 and 218 / 194 / 61; W6—151 and 29 / 38 / 84; J9—1000 and 379 / 375 / 246.
- Target-trained F3 exceeds B8Q on every target: numerical-observations table, “Target-trained F3, 5-fold CV” versus “B8Q F3”—C2R 0.936 versus 0.706; W6 0.950 versus 0.874; J9 0.869 versus 0.606.
- H2 gains, recovery-comparison F3, and L2 recovery are recorded as reported: corresponding rows give C2R 0.22, 0.929, 0.881, 78%; J9 0.23, 0.832, 0.662, 25%.
- Subsets are small: subset-size rows give CV means 15, 8, 14; full-target \(k\) 13, 7, 14; the additional-observations table gives range 8–15.
- Decoder alignment decreases with index: additional-observations table, best-matching and mean-top-10 rows—Spearman \(\rho\) -0.857 and -0.917.

## Appendix B LLM Judge Prompt

*This appendix gives the interpretation model and the complete direction-rating prompt.*

Direction interpretation uses GPT-5-mini (P2), OpenAI (2025), with the following prompt. The example format `token<TAB>activation` places a tab between token and activation. The placeholders supply positive and negative examples.

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

*Lower-index directions align more strongly with both individual and grouped decoder features.*

Lower-index directions align more strongly with decoder vectors, the SAE decoder features U9. Alignment is measured using both the best-matching U9 and mean top-feature alignment, with `andyrdt/saes-llama-3.1-8b-instruct` (andyrdt, 2024), the resource used for these measurements.

Figure 6: Lower Q7 exhibit stronger decoder alignment for the best-matching feature and the mean of the top 10 features.

**Claims and evidence**
- Lower Q7 have stronger alignment under both measures: Figure 6; Appendix A, additional-observations table, best-matching-feature row—Spearman \(\rho\) -0.857; mean-of-top-10-features row—Spearman \(\rho\) -0.917.

## Appendix D Q7 Set Comparisons

*Transfer selection and target-separation ranking have low overlap, and their selected direction sets differ.*

H2 and M9 agree on less than a fifth of selections under Jaccard overlap (D3), the set measure used here. This is consistent with transferable space being neither the most source-discriminative nor the most target-discriminative region.

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

**Claims and evidence**
- H2–M9 overlap is less than a fifth on every target: D3 table, C2R 0.182, W6 0.077, J9 0.167.
- Source-weight, transfer-selected, and target-separation sets differ: Table 2, T6 Source row and the H2 and M9 rows for C2R, W6, and J9 give the exact indices.

## Appendix E Q7 Interpretability

*Repeated ratings retain 19 directions, and the explanations show both potentially transferable contrasts and surface-level limitations.*

| Selection observation | Value |
|---|---:|
| Maximum observed R5; no direction exceeded it | 6 |
| Scoring runs | 2 |
| Directions reaching maximum in both runs, retained for L2 | 19 |
| Complement directions used by A7 | 81 |
| Directions selected by H2 for more than one target | 5 |
| Maximum-rated directions among those shared H2 selections | 3 out of 5 |
| Maximum-rated directions among the top T6 selections | 1 of the top 5 |

In Table 3, “PR-style wording” names the defensive wording described for Q7 86. In Table 4, “QA/V8” names the similar task settings described for Q7 8.

Table 3: Directions selected for likely transferability from C7 interpretation.

| Q7 index | Selected on | R5 | Full explanation |
|---|---|---:|---|
| 62 | C2R + J9 | 6 | General deception-relevant denial/deflection versus detail provision can be useful across settings; noise includes truthful denial or obfuscation for legitimate reasons and liars’ elaborate fabricated details. Lexical negation rather than semantic strategy may drive the signal. |
| 57 | C2R + J9 | 6 | General denial/minimization or innocuous explanations versus specific/incriminating details can flag rhetorical hiding or deflection. Partial dependence on surface denials, hedges, and domain terms makes it lexical, task-specific, brittle, and context-dependent OOD. |
| 9 | C2R + J9 | 3 | Mainly domain-specific medical/leave versus project/status vocabulary and stylistic framing, not generalizable deception strategy; genuine illness claims may be flagged, with brittleness across domains and rephrasings. |
| 81 | W6 + J9 | 3 | Plausibly deception-relevant admission versus denial/hedging, mainly carried by surface/formal negations, hedges, résumé style, and enumerations; task- and wording-dependent, likely brittle across domains and phrasings. |
| 86 | W6 + J9 | 6 | Hedging and defensive PR-style wording are general deceptive or self-protective tactics potentially transferable across domains; legitimate privacy, politeness, and negotiation uses add noise, phrasing and role sensitivity, and false positives, limiting OOD reliability to moderate. |

Table 4: Directions, R5 ratings, and interpretations.

| Q7 index | R5 | Full explanation |
|---|---:|---|
| 7 | 3 | Pragmatic denial/deflection versus admission may correlate with deception in some settings; heavy dependence on explicit “No,” word order, “I” plus admission, and task-specific phrasing makes it brittle, likely to fail or mislead across domains or varied wording. |
| 8 | 4 | Narrow denial/excuse versus admission can correlate with evasion or deception in similar QA/V8 tasks; explicit “No”/“Yes” and common denial phrasing dominate, with likely brittleness under paraphrase, domain change, or subtler deception strategies. |
| 11 | 3 | Stylistic/pragmatic assertive first-person role/defense versus problem-reporting/negation can correlate with evasive or face-saving lies in some scenarios; surface-level, context-dependent, likely to fail across domains or phrasings. |
| 12 | 3 | Discourse style/forcefulness is a surface cue; overconfidence or over-assertion sometimes correlates with deceptive defensiveness, yet brittleness and domain dependence produce many false positives/negatives across speakers, topics, and cultural styles. |
| 4 | 6 | General communicative denial versus admission is deception-relevant and can transfer beyond particular prompts; dependence on explicit “Yes”/“No” turns and common defensive phrasing risks brittleness with differently worded denial or lies and subtler tactics. |

**Claims and evidence**
- Maximum R5 is 6 across 2 runs; 19 directions reach it in both runs, leaving 81 for A7: selection-observations table, first four rows.
- Interpretation only partially recovers shared H2 selections: selection-observations table, shared-selection rows—5 shared directions, with 3 out of 5 maximum-rated; top-T6 row—1 of the top 5.
- Shared H2 directions receive different ratings and caveats: Table 3, Q7 62, 57, 9, 81, 86—R5 6, 6, 3, 3, 6. The explanations retain lexical, domain, noise, and false-positive limitations.
- The listed source-ranked directions are mostly rated below the maximum: Table 4, Q7 7, 8, 11, 12, 4—R5 3, 4, 3, 3, 6. Their explanations retain wording dependence and transfer caveats.
