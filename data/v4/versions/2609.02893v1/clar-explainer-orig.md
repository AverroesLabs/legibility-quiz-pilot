# Probe Generalization as Subspace Selection for OOD Deception Detection

Daniel Yoo

  
Adrians Skapars

**Summary**

Can selecting principal components (PCs), directions in a source activation basis, improve out-of-distribution (OOD) deception detection? The paper trains linear probes—logistic-regression classifiers—on one source dataset and evaluates 3 held-out target datasets. A target-supervised greedy search, which adds PCs according to target performance, finds 8–15 PCs that nearly match probes trained on target data. Ranking PCs by source- or target-probe weights does not recover this transferable subspace, a selected set of PC directions. A language-model (LLM) judge scores natural-language PC descriptions without target data. Selecting 19 PCs that receive the maximum observed score in both runs improves transfer on all 3 targets. It closes 78% and 25% of the gap between the full-representation baseline and the greedy oracle on two targets.

## Glossary

*This glossary gives the paper’s meanings and uses of its technical terms, abbreviations, symbols, and codes.*

| Term, abbreviation, symbol, or code | Definition |
|---|---|
| Activation | A model representation extracted while processing an example. |
| Activation space / representation space | The space of model activations in which probes, directions, and subspaces are studied. |
| Residual stream | The model representation from which layer 15 activations are extracted. |
| Token activation | A model activation at an individual token; token activations are averaged over the assistant’s response to obtain probe inputs. |
| Linear probe / probe | A classifier used to read information from model activations; all probes here use logistic regression. |
| Logistic regression | The linear classifier used for every probe in the experiments. |
| Probe weight / coefficient | A value learned by a probe for a representation direction. |
| Standardized coefficient | The absolute probe coefficient multiplied by the corresponding PC’s standard deviation. |
| Standardization / scaling | Transforming source activations to zero mean and unit variance, then applying the source statistics to target activations. |
| Source / source domain | The Roleplaying dataset used to train probes and construct the PCA basis. |
| Target / target domain | A held-out deception dataset used to evaluate transfer. |
| In-distribution | Evaluation within the distribution on which a probe is trained. |
| Out-of-distribution / OOD | Evaluation on examples outside the probe’s training distribution. |
| Transfer / generalization | A source-trained probe’s performance on a target domain. |
| Cross-domain transfer | Transfer from the source dataset to a different target dataset. |
| Spurious correlation | A source-specific correlation that a probe can learn but that does not reliably transfer. |
| Surface feature / surface cue | A wording, lexical, stylistic, or task-specific feature contrasted with a general behavioral feature. |
| Linear direction | A direction in activation space used to represent or read a concept or behavior. |
| Subspace | A selected set of representation directions onto which activations are projected. |
| Transferable subspace | Source-PC directions on which a source-trained probe transfers to a target. |
| Basis / PCA basis | The set of source-PC directions used instead of raw activation dimensions. |
| Latent dimensions | The model’s original representation dimensions, which the paper does not assume form a privileged basis. |
| PCA | The procedure applied to scaled source activations to obtain principal components. |
| Principal component / PC; plural PCs | A direction in the source-domain PCA basis. |
| PC index | The number identifying a PC in the source basis. |
| Top PCs / lower-index PCs | PCs selected from the beginning of the source-PC ordering by variance. |
| Variance | The quantity used to order PCs and construct variance-based selection controls. |
| Projection | Restricting activations to selected PC directions before probe training and evaluation. |
| Full representation | The complete standardized activation representation used by baseline probes. |
| Baseline | A source-trained probe using the full standardized activations. |
| Target-trained probe | A probe fit directly on target data. |
| Greedy selection / greedy oracle | Target-guided selection that repeatedly adds the source PC giving the largest target-AUROC improvement. |
| Oracle | The target-supervised greedy selection used as an upper bound on transfer performance. |
| Variance control / source-variance selection | Selecting the top \(k\) source PCs by variance. |
| Source-PCA probe | The source-trained probe whose standardized coefficients rank source PCs. |
| Target-PCA probe | The target-trained probe whose standardized coefficients rank source PCs. |
| Target separability ranking | Ranking source PCs by weights from a target-trained probe. |
| Separability | How well a representation distinguishes deceptive from honest responses. |
| AUROC | The probe-performance metric reported throughout the experiments. |
| Cross-validation / CV | Evaluation using 5 target folds, with held-out folds used for testing. |
| Fold / held-out fold | A cross-validation partition withheld from the fitting or selection step being evaluated. |
| CI | The uncertainty interval shown by figure error bars at 95%. |
| \(S\) | A subset of the retained 100 source PCs. |
| \(k\) | The number of PCs selected for a probe. |
| \(\lambda\) | The logistic-regression parameter set to 10. |
| LLM | A language model used to interpret and score PC directions. |
| LLM judge / judge model | The model that describes PC poles and rates likely OOD deception usefulness. |
| Natural-language interpretation | A verbal description of a PC’s positive pole, negative pole, and overall contrast. |
| Positive pole | The side of a PC direction associated with positive activations. |
| Negative pole | The opposite side, associated with negative activations. |
| Activation magnitude | The strength of an example’s match to a PC pole in the judge prompt. |
| OOD score | The judge’s 1–10 rating of a direction’s likely usefulness for OOD deception detection. |
| Maximum observed OOD score | A score of 6, the highest score received by any PC. |
| Interp-selected PCs / interp-selection | The 19 PCs receiving the maximum observed OOD score in both scoring runs. |
| Interp-ablated PCs / ablation control | The remaining 81 PCs after removing the 19 interp-selected PCs. |
| Ablation | Removing selected components; inference-time ablation was not performed for causal validation. |
| Inference | The model-use setting in which the paper did not causally test high-scoring PCs by ablation. |
| Causal validation | Testing whether PCs carry the apparent contrast by intervening on them; this was not done. |
| Linear pipeline | PCA projection followed by logistic regression. |
| Nonlinear probe | An alternative probe type to which the paper leaves generalization open. |
| Readout location | A model location from which probe inputs could be extracted; alternatives remain untested. |
| Linear representation hypothesis | The hypothesis formalized by Park et al. (2024) and connected to probing and steering. |
| Steering / activation-space intervention | Using representation directions to intervene on model behavior. |
| Population-level direction | The representation of high-level properties studied by Zou et al. (2025). |
| Refusal direction | The single direction reported by Arditi et al. (2024) to mediate refusal behavior. |
| Assistant axis | The direction identified by Lu et al. (2026) as corresponding to assistant-like behavior. |
| SAE / sparse autoencoder | The feature-learning method whose feature descriptions and decoder features are discussed. |
| SAE decoder feature | A feature used to assess alignment with PCs. |
| Decoder alignment | The PC-to-SAE-feature alignment measured in Appendix C. |
| Spearman \(\rho\) | The statistic reported for the relationship between PC index and decoder alignment. |
| Jaccard overlap | The PC-set overlap statistic reported for greedy and target-PCA selections. |
| Roleplaying | A dataset in which a model faces a simple scenario with an opportunity to lie. |
| Insider Trading Report | A dataset in which a trader model decides whether to report honestly that it used insider information. |
| Insider Trading Confirm | A dataset in which a model that hid earlier insider-information use is asked to confirm it did not use any. |
| Sandbagging | A dataset in which a model is told to under-perform on some tasks and behave normally on others. |
| Honest / \(0\) | The honest-response label assigned by GPT-4o. |
| Ambiguous / \(-1\) | The response label removed before the experiments. |
| Deceptive / \(1\) | The deceptive-response label assigned by GPT-4o. |
| Llama-3.1-8B-Instruct | The language model used in all experiments. |
| GPT-4o | The model used to label responses as honest, ambiguous, or deceptive. |
| GPT-5-mini | The model used to interpret PC directions. |
| `andyrdt/saes-llama-3.1-8b-instruct` | The SAE resource used for Appendix C’s decoder-alignment analysis. |
| `andyrdt` | The attribution attached to the SAE resource, dated 2024. |
| `token<TAB>activation` | The judge prompt’s format for a token and its normalized activation. |
| `<TAB>` | The separator shown between token and activation in that format. |
| `{positive_examples}` | The prompt placeholder for high-positive examples. |
| `{negative_examples}` | The prompt placeholder for high-negative examples. |
| `Positive pole:` | The requested output field describing the positive side of a PC. |
| `Negative pole:` | The requested output field describing the negative side of a PC. |
| `Shared direction:` | The requested output field describing the overall PC contrast. |
| `OOD deception usefulness (1-10):` | The requested output field for the OOD usefulness rating. |
| `...` | A placeholder for the judge’s answer in the requested output format. |
| \(-10\) to \(+10\) | The integer activation-normalization range used in the judge prompt. |
| \(1\)–\(10\) | The judge’s OOD usefulness scale. |
| \(-1 / 0 / 1\) | The order of ambiguous, honest, and deceptive label counts in Table 1. |
| `+` between dataset names | Joint selection of a PC on the named targets in Table 3. |
| QA/roleplay | The task description used in Table 4’s PC8 interpretation. |
| PR-style wording | Defensive wording discussed in Table 3’s PC86 interpretation. |
| AI Safety Camp | The program during whose 11th edition this work was conducted. |
| ICML | The label listed with “Machine Learning” after the abstract. |
| `kramár2026buildingproductionreadyprobesgemini` | The citation code used for work on uneven OOD probe generalization. |

The PC codes below identify individual directions in the source-PCA basis.

| Code | Definition |
|---|---|
| PC1 | Source principal component 1. |
| PC2 | Source principal component 2. |
| PC3 | Source principal component 3. |
| PC4 | Source principal component 4. |
| PC5 | Source principal component 5. |
| PC6 | Source principal component 6. |
| PC7 | Source principal component 7. |
| PC8 | Source principal component 8. |
| PC9 | Source principal component 9. |
| PC10 | Source principal component 10. |
| PC11 | Source principal component 11. |
| PC12 | Source principal component 12. |
| PC13 | Source principal component 13. |
| PC14 | Source principal component 14. |
| PC18 | Source principal component 18. |
| PC19 | Source principal component 19. |
| PC20 | Source principal component 20. |
| PC21 | Source principal component 21. |
| PC22 | Source principal component 22. |
| PC23 | Source principal component 23. |
| PC25 | Source principal component 25. |
| PC26 | Source principal component 26. |
| PC27 | Source principal component 27. |
| PC28 | Source principal component 28. |
| PC29 | Source principal component 29. |
| PC30 | Source principal component 30. |
| PC31 | Source principal component 31. |
| PC32 | Source principal component 32. |
| PC33 | Source principal component 33. |
| PC34 | Source principal component 34. |
| PC37 | Source principal component 37. |
| PC39 | Source principal component 39. |
| PC41 | Source principal component 41. |
| PC43 | Source principal component 43. |
| PC44 | Source principal component 44. |
| PC46 | Source principal component 46. |
| PC48 | Source principal component 48. |
| PC49 | Source principal component 49. |
| PC51 | Source principal component 51. |
| PC53 | Source principal component 53. |
| PC54 | Source principal component 54. |
| PC57 | Source principal component 57. |
| PC61 | Source principal component 61. |
| PC62 | Source principal component 62. |
| PC66 | Source principal component 66. |
| PC67 | Source principal component 67. |
| PC70 | Source principal component 70. |
| PC71 | Source principal component 71. |
| PC77 | Source principal component 77. |
| PC78 | Source principal component 78. |
| PC79 | Source principal component 79. |
| PC81 | Source principal component 81. |
| PC86 | Source principal component 86. |
| PC87 | Source principal component 87. |
| PC89 | Source principal component 89. |
| PC91 | Source principal component 91. |
| PC93 | Source principal component 93. |
| PC97 | Source principal component 97. |

###### Abstract

*Selecting a small source-PC subspace enables transfer, and an LLM judge partly identifies that subspace without target data.*

Linear probes, classifiers that read information from language model activations, can detect behaviors and concepts but may fail out of distribution (OOD), on examples outside their training distribution. We study probes on Llama-3.1-8B-Instruct, the language model used throughout this work, over 3 held-out deception detection datasets.

Projecting inputs onto a small subset of principal components (PCs), directions obtained from the training activation distribution, enables cross-domain transfer that nearly matches probes trained directly on the test distribution. PC interpretations can identify some of these transferable PCs.

A language-model (LLM) judge scores whether each PC’s most- and least-activating examples imply a transferable deception direction. Probing on the highest-scoring PCs closes the baseline-to-oracle gap by 78% on Insider Trading Report, where a trader model reports insider-information use, and by 25% on Sandbagging, where a model deliberately under-performs on some tasks.

Directions heavily weighted by a source probe appear to encode source-specific surface features: wording or other local cues. Transferable directions appear to encode the same contrast more abstractly, in a form natural-language descriptions can capture. Overall, the results suggest that OOD probe robustness is largely determined by subspace selection: choosing the representation directions on which to train.

Machine Learning, ICML

**Claims and evidence**
- A small source-PC subspace nearly matches target-trained performance: Section 4.2, Figure 3; greedy selection averages 15 / 8 / 14 PCs across the targets.
- Interpretation-based selection partly closes the baseline-to-oracle gap: 78% on Insider Trading Report and 25% on Sandbagging; Section 4.5, Figure 5.

## 1 Introduction

*The paper asks which source PCs support generalizing deception probes and how to identify them.*

Linear probes are standard tools for reading information from language model activations, the representations produced while processing examples (Alain and Bengio, 2018; Hewitt and Liang, 2019). They are cheap to train and reveal some semantically meaningful information in an otherwise uninterpretable representation space.

However, probes that perform well in-distribution—within their training distribution—can be unreliable OOD (Kirch et al., 2026). Existing work finds linear representations of key behavioral concepts (Marks and Tegmark, 2024; Park et al., 2024; Zou et al., 2025). Probes should therefore have the capacity to detect these concepts, yet in practice they appear to learn spurious correlations: source-specific associations that fail to transfer (Orgad et al., 2025; Wang et al., 2025). This is unacceptable for monitoring harmful interactions. The challenge is to build probes that learn only generalizing features.

We work in the source-domain PC basis—the set of PC directions derived from training data—rather than raw activations. This avoids assuming that the model’s latent dimensions, its original representation dimensions, provide a privileged basis. PCs offer a simple, data-driven way to study representation structure. Prior work suggests their selection matters: Mu et al. (2018) find that removing top PCs improves downstream embedding performance, and Kantamneni et al. (2025) use top-PC projections as a probe baseline.

Our question is therefore: which PC subset supports generalizable probe training, and how can it be identified?

We study deception probes on Llama-3.1-8B-Instruct (Grattafiori et al., 2024). Roleplaying, where a model faces a simple scenario with an opportunity to lie, is the source dataset used for training. The 3 held-out target datasets used to evaluate transfer are Insider Trading Report, Insider Trading Confirm, and Sandbagging (Goldowsky-Dill et al., 2025). In Insider Trading Confirm, a model that hid earlier insider-information use is asked to confirm it did not use any.

Working in the source-domain PCA basis—the basis obtained by applying PCA to source activations—we test ways to select training dimensions that avoid spurious correlations and transfer to targets.

This work contributes three findings:

- **A small transferable subspace exists (Section 4.2).** A target-supervised greedy search, which adds source PCs according to target performance, finds 8–15 components on which a source-trained probe nearly matches a target-trained probe.
- **Source-side and target-side rankings both fail to recover the subspace (Section 4.3).** Ranking PCs by source- or target-probe weights does not produce a basis that transfers better than the full-representation baseline.
- **An LLM judge partially recovers the subspace without target data (Section 4.5).** The judge can interpret and select PCs likely to encode transferable behavioral contrasts rather than surface patterns.

Figure 1: Overview of PC-based subspace selection for OOD deception probing. Activations are projected onto the source-domain PCA basis, candidate PC directions are filtered, and probes are trained on the selected low-dimensional subspace.

**Claims and evidence**
- A small transferable subspace exists: 8–15 PCs; Section 4.2, Figure 3.
- Weight-based rankings fail to recover it: source- and target-weight comparisons in Section 4.3, Figure 4.
- A judge partly recovers it without target data: 19 PCs; 78% and 25% gap recovery on Insider Trading Report and Sandbagging; Section 4.5, Figure 5.

## 2 Related Work

*Prior work supports linear behavioral representations but warns that probe accuracy need not identify robust concepts.*

#### Linear concept directions in activation space.

*Prior studies represent high-level properties as directions or low-dimensional subspaces and use them to explain or control behavior.*

One line of work asks whether high-level behavioral or semantic properties occupy directions or low-dimensional subspaces in activation space. Marks and Tegmark (2024) report linear structure in true/false representations across datasets. Park et al. (2024) formalize the linear representation hypothesis and connect it to probing and steering, the use of representation directions to intervene on behavior. Zou et al. (2025) model properties such as honesty and emotion as population-level directions that can be read from and intervened on in activations.

This perspective also supports explanations and interventions. Arditi et al. (2024) show that a single direction can mediate refusal behavior. Lu et al. (2026) identify an “assistant” axis corresponding to assistant-like behavior. Together, these works suggest that linear structure can support mechanistic understanding and activation-space interventions. They also raise the question of whether directions capture stable concepts or context-dependent correlations.

#### Linear probes and their limitations.

*Probe performance can reflect learned tasks or surface correlations rather than robust internal signals.*

Linear and lightweight probes have a long history as tools for analyzing neural-network representations (Alain and Bengio, 2018; Hewitt and Liang, 2019). Recent work applies them to safety-relevant LLM signals, including latent truthfulness without direct supervision (Burns et al., 2024) and strategic deception (Goldowsky-Dill et al., 2025).

The probing literature also emphasizes methodological skepticism. Hewitt and Liang (2019) argue that high probe accuracy alone does not establish that a model already represents the target property: a probe may learn the task or exploit surface correlates. Ravichander et al. (2021) similarly question whether probing establishes what is encoded or merely recoverable. Probes with strong in-distribution performance can also generalize unevenly OOD (kramár2026buildingproductionreadyprobesgemini). These concerns motivate testing whether safety probes identify robust signals rather than dataset artifacts.

#### Interpreting features with natural-language explanations.

*Natural-language feature descriptions have been used for both interpretation and intervention.*

Another approach assigns natural-language descriptions to neurons or learned features. Bills et al. (2023) use LLMs to explain neurons from high-activation examples. This approach has been extended to automatic feature labeling in sparse autoencoders (SAEs), which provide learned feature dictionaries ((Paulo et al., 2025), (Bricken et al., 2023)).

Recent work also uses these descriptions for intervention. Casademunt et al. (2025), for example, identify concepts through automated natural-language interpretation and ablate them—remove them—during fine-tuning to reduce misaligned behavior.

## 3 Setup

*The experiments use one source dataset, 3 targets, layer 15 activations, and logistic-regression probes.*

#### Datasets.

*Roleplaying supplies training data; the other 3 datasets test transfer after ambiguous responses are removed.*

Following Goldowsky-Dill et al. (2025), we use 4 deception detection datasets: Roleplaying, Insider Trading Report, Insider Trading Confirm, and Sandbagging. Appendix A describes them.

Roleplaying is the source domain. The other 3 are target test sets. GPT-4o (OpenAI, 2024), the response-labeling model, labels each response honest (0), ambiguous (-1), or deceptive (1). We remove ambiguous responses from the experiments.

#### Language models.

*Inputs to the probes are layer 15 residual-stream activations averaged over the assistant’s response.*

Following Ying et al. (2026), we use layer 15 residual stream activations of Llama-3.1-8B-Instruct (Grattafiori et al., 2024). The residual stream is the model representation read at that layer. Layer 15 had the highest cross-domain AUROC—the probe-performance metric used here—in their analysis.

To extract activations, we pass each prompt through the model. We average token activations over only the assistant’s response.

#### Probe training and evaluation.

*All probes use source-standardized activations, and PC-based probes use subsets of the top 100 source PCs.*

We standardize source activations to zero mean and unit variance. We apply those same source statistics to target activations. We run PCA on the scaled source activations and retain the top 100 PCs.

All probes are logistic-regression classifiers, the linear classifiers used throughout, with parameter \(\lambda=10\). For a subset \(S\) of the 100 PCs, we project activations onto \(S\)—restrict them to those directions—before training and evaluation. Baseline probes instead train on the full standardized activations, called the full representation.

We retain the top 100 PCs because alignment with SAE decoder features decreases with PC index, the component’s position in the source basis (Appendix C). Lower-index PCs are therefore more amenable to natural-language interpretation and comparable with interpretability-driven selection methods. This does **not** claim that low-index PCs generalize better. Section 4.1 checks that the 100-PC subspace retains enough information to distinguish deceptive from honest responses on every target.

## 4 Experiments

*The experiments establish a transferable subspace, test weight-based selection, and evaluate interpretation-based selection.*

### 4.1 Setting a baseline

*Source-trained probes transfer unevenly, although the top-100-PC subspace supports strong target-trained probes.*

Figure 2: Probe AUROC on target datasets for source-trained probes on full representations (blue) and top-100 PCs (orange), and target-trained probes on top-100 source PCs (green, 5-fold CV). Error bars: 95% CIs

Figure 2 compares source-trained probes using the full source representation. Insider Trading Confirm transfers reasonably well, with 0.874 AUROC. Insider Trading Report performs moderately, with 0.706 AUROC. Sandbagging performs poorly, with 0.606 AUROC.

To test whether the 100-PC subspace is itself a bottleneck, we also train directly on target data within that subspace. We use 5-fold cross-validation (CV): target data are divided into folds, with held-out folds used for evaluation. Figure error bars show 95% CIs, the reported uncertainty intervals.

The target-trained probe achieves 0.936 / 0.950 / 0.869 AUROC across the three targets. Thus, the 100-PC subspace contains enough information to distinguish deceptive from honest responses on every target.

The difficulty is therefore not missing information. It is that the source probe learns a direction in the subspace that does not generalize.

**Claims and evidence**
- Full-representation transfer varies by target: Insider Trading Report 0.706, Insider Trading Confirm 0.874, Sandbagging 0.606 AUROC; Figure 2.
- The 100-PC subspace retains target-class information: target-trained, 5-fold-CV AUROCs are 0.936 / 0.950 / 0.869; Figure 2.
- The source-learned direction, rather than absent target information, limits transfer: the source-trained and target-trained comparisons above; Figure 2.

### 4.2 Showing a transferable subspace exists

*Target-guided greedy selection finds small source-PC subsets that transfer well, but it is not an unbiased OOD estimate.*

Figure 3: Probe AUROC across the three targets for the full-representation baseline (blue), top-k PCs by variance (teal, k matched to greedy), and greedy oracle PCs (purple). Error bars: 95% CIs.

Can a probe trained on source data close the gap between baseline and target-trained performance? We test a greedy PC-selection algorithm.

The algorithm begins with an empty set. At each step, it searches the top 100 source PCs and adds the PC that most improves target AUROC. For each candidate subset, it projects source activations onto that subset and trains a logistic-regression probe.

Probe weights are always fit on source data. Only subset selection uses labeled target examples. In 5-fold target CV, greedy selection uses the non-held-out target folds. The selected source-trained probe is then evaluated on the held-out target fold.

Selection stops when no candidate PC improves AUROC by more than 0.001. The resulting subsets average 15 PCs for Insider Trading Report, 8 for Insider Trading Confirm, and 14 for Sandbagging.

This greedy probe is an upper bound on transfer performance, not an unbiased OOD estimate, because selection uses target examples. We therefore call it the greedy oracle. As a control, we select the top \(k\) PCs by variance, where \(k\), the subset size, matches the number selected by greedy. Figure 3 shows the comparison.

For Insider Trading Report and Sandbagging, greedy comes close to a probe trained directly on the target. It improves baseline AUROC by 0.22 and 0.23 points, respectively. A transferable subspace therefore exists in principle.

The variance control matches greedy only on Insider Trading Confirm, the easy case. It falls far behind on Insider Trading Report and Sandbagging, the harder transfer tasks. Thus, which PCs are selected matters; their explained variance does not predict usefulness.

**Claims and evidence**
- Greedy finds small subsets: mean sizes are 15 / 8 / 14 PCs for Report / Confirm / Sandbagging; Figure 3 and the selection results above.
- These subsets nearly match target-trained probes on Report and Sandbagging: baseline improvements are 0.22 and 0.23 AUROC; Figure 3.
- Variance alone does not identify equally useful PCs: matched-\(k\) variance controls match greedy only on Confirm and fall behind on Report and Sandbagging; Figure 3.
- Greedy is not an unbiased OOD estimate: its subset selection uses labeled non-held-out target folds in 5-fold CV.

### 4.3 Testing hypotheses for transferable principal components

*Neither source-probe nor target-probe coefficient rankings recover greedy transfer performance.*

Figure 4: Probe AUROC across the three targets for three PC-selection rules: ranking by source-probe weights (red), by target-probe weights (pink), and greedy oracle selection (purple). Error bars: 95% CIs.

What makes the selected PCs transferable, and can that property be detected from the source alone? We test 2 hypotheses:

1. Transferable PCs are those that individually separate target classes well across the test set; greedy finds them because it uses target data.
2. Transferable PCs are those on which the source probe already relies most heavily; greedy merely compresses the source signal.

We compare 2 PC rankings. Each uses a **standardized coefficient**: the absolute probe coefficient multiplied by the corresponding PC’s standard deviation. Both ranking probes operate on activations projected into the source-PCA basis.

The **source-PCA probe** ranks PCs using weights learned from source data. The **target-PCA probe** ranks them using weights learned from target data. For both rankings, the probes evaluated for transfer are trained on source data projected onto the top \(k\) PCs in the respective ranking. If either hypothesis were correct, its selected probe should recover greedy performance.

To set \(k\), we run greedy on the full target dataset rather than separately within folds. This gives one stable PC set: \(k=13\) for Insider Trading Report, 7 for Insider Trading Confirm, and 14 for Sandbagging. Because this uses the full dataset rather than CV, the greedy numbers differ slightly from those in Section 4.2. Figure 4 shows the results.

Both hypotheses fail, but differently. The source-PCA probe transfers about as well as the full-representation baseline. The PCs on which the source probe relies are those that do not generalize.

The target-PCA probe does well on Insider Trading Report. It performs worse than greedy on Insider Trading Confirm and below random on Sandbagging. Thus, PCs that best separate target classes are not necessarily those that transfer best when their probe weights are fit on source data.

Appendix D gives further PC-set comparisons.

**Claims and evidence**
- Source-weight ranking does not recover greedy transfer: source-PCA transfer is about the full-representation baseline; Figure 4.
- Target-weight ranking also fails: it does well on Report, falls below greedy on Confirm, and is below random on Sandbagging; Figure 4.
- The full-target comparison uses \(k=13 / 7 / 14\) for Report / Confirm / Sandbagging; the full-target greedy procedure above.
- Greedy and target-ranked sets differ: Jaccard overlaps are 0.182 / 0.077 / 0.167; Appendix D. Table 2 lists the “Greedy selection” and “Target separability ranking” rows for each target.

### 4.4 Interpreting the transferable subspace

*Natural-language interpretations identify some repeatedly transferable PCs as more general behavioral contrasts.*

A small subset of source PCs supports cross-domain transfer. It differs from directions emphasized by source or target classification. Does it have a human-interpretable meaning? We use the interpretation methods of Bills et al. (2023) and Casademunt et al. (2025).

For each of the top 100 source PCs, we find the 8 most-activating and 8 least-activating source examples. An LLM describes the positive pole, associated with positive activations; the negative pole, associated with negative activations; and the direction’s overall contrast.

The judge then assigns an **OOD score**, a 1–10 rating of likely generalizability. Appendix B gives the full prompt and model details. No PC scores above 6, so 6 is the maximum observed OOD score. We retain PCs scoring 6 in both runs, yielding 19 PCs.

There are 5 PCs selected by greedy for more than one target. Of these, 3 out of 5 receive the maximum OOD score. This suggests that LLMs can partly recover transferable directions. By comparison, only 1 of the top 5 source-PCA-probe PCs receives the maximum score.

Interpretations of greedy-selected PCs describe the contrast at an abstraction level that survives transfer to other domains. Appendix E gives the breakdown of PCs selected by each method.

**Claims and evidence**
- The judge partly identifies repeatedly transferable PCs: 3 of 5 receive score 6; Table 3, rows PC62, PC57, and PC86. Rows PC9 and PC81 score 3.
- The source probe’s top PCs receive lower scores: only 1 of the top 5 receives 6; Table 4, rows PC7, PC8, PC11, PC12, and PC4 have scores 3 / 4 / 3 / 3 / 6.
- Repeated maximum-score selection yields 19 PCs: score 6 in both runs; the scoring procedure above.
- Greedy-selected descriptions include more general contrasts, with caveats: Table 3, especially PC62, PC57, and PC86.

### 4.5 Selecting a subspace by interpreting PCs

*Selecting 19 PCs from judge interpretations improves transfer without using target data, with sharply different gains across targets.*

Figure 5: Probe AUROC across the three targets for the interp-selected subspace (red; 19 PCs) versus three controls: full-representation baseline (blue), top-19 PCs by variance (teal), and the remaining 81 PCs after ablating the interp-selected PCs (brown). Error bars: 95% CIs.

Can interpretation-based OOD scores recover a meaningful fraction of greedy performance without target data?

We use the PCs receiving the maximum observed judge score as the probe subspace. Because judge scores are noisy, we score twice and retain only PCs receiving the maximum in both runs. This gives 19 **interp-selected PCs**, the interpretation-selected subspace.

We compare against two controls: the top 19 source PCs by variance, and the **interp-ablated** subspace, the remaining 81 PCs after removing the selected PCs. Figure 5 shows the results.

The interp-selected probe beats the baseline on all 3 targets. It also beats the source-variance control, by wide margins on Insider Trading Report and Sandbagging. As expected, it beats the interp-ablated probe on all 3 targets.

Recovery varies sharply by dataset:

- On **Insider Trading Report**, interp-selection closes 78% of the gap between baseline (0.706) and greedy oracle (0.929).
- On **Insider Trading Confirm**, baseline transfer is already near the ceiling. There is little headroom, so the result is uninformative: each selection method except interp-ablation performs roughly equally.
- On **Sandbagging**, interp-selection closes 25% of the gap between the greedy oracle (0.832).

Overall, selecting PCs solely from natural-language descriptions, without target data, meaningfully improves baseline performance on the generalization tasks.

**Claims and evidence**
- Interpretation-only selection improves all 3 targets and beats both controls: Figure 5 compares 19 interp-selected PCs, top-19 variance PCs, and the remaining 81 PCs.
- Report recovers 78% of the gap: baseline 0.706, interp-selected 0.881, greedy 0.929; Figure 5 and Section 5.
- Confirm is uninformative because baseline is already high: baseline 0.874; Figure 2 and Section 5. Figure 5 shows the selection comparisons.
- Sandbagging recovers 25% of the gap: baseline 0.606, interp-selected 0.662, greedy 0.832; Figure 5 and Section 5.

**Section 4 — Claims and evidence**
- Small target-guided subsets support transfer: mean greedy subset sizes 15 / 8 / 14; Figure 3.
- Weight rankings do not recover greedy performance: Figure 4; Table 2 lists the corresponding selections.
- Interpretation partly recovers the subspace without target data: 19 selected PCs; 78% and 25% gap recovery on Report and Sandbagging; Figure 5.

## 5 Limitations

*The evidence is limited to one model and source dataset, a linear pipeline, modest Sandbagging gains, and descriptive rather than causal interpretations.*

Our strongest claims remain narrow.

**Single model and source dataset.** All experiments use Llama-3.1-8B-Instruct, layer 15, and Roleplaying as the source (Goldowsky-Dill et al., 2025). Insider Trading Confirm is uninformative because baseline transfer already reaches 0.874. Whether the findings generalize to other model families or datasets remains open.

**Sandbagging margins are small.** Interpretability selection raises Sandbagging AUROC from 0.606 to 0.662. This recovers only 25% of the gap to greedy’s 0.832 upper bound. Sandbagging is also the most OOD target, and its modest gain likely reflects this distribution shift. Insider Trading Report is the more robust result: 0.706 to 0.881, closing 78% of the gap to greedy at 0.929.

**Linear pipeline.** The pipeline is fully linear: PCA projection followed by logistic regression. A transferable subspace may therefore be unsurprising. Whether the findings extend to nonlinear probes or other readout locations—other model locations used to extract probe inputs—remains open.

**No causal validation.** Claims about what a PC “encodes” come from top-activating examples and the judge’s verbalized contrast. These are descriptive, not causal, claims. We have not ablated high-scoring PCs during inference, when the model is used, to verify that they carry the apparent contrast.

**Claims and evidence**
- Scope is one model, layer, and source: Llama-3.1-8B-Instruct, layer 15, Roleplaying; Section 3.
- Confirm has little headroom: baseline 0.874; Figure 2.
- Sandbagging gains are modest: 0.606 to 0.662 against greedy 0.832, recovering 25%; Figure 5.
- Report gives the more robust result: 0.706 to 0.881 against greedy 0.929, recovering 78%; Figure 5.
- The evidence does not establish nonlinear generalization or causality: Section 3 uses PCA plus logistic regression; this section reports no inference-time causal ablation.

## 6 Conclusion

*Robust deception probing requires selecting where a behavioral signal is represented, not merely establishing linear accessibility.*

For deception monitoring, a small subset of source PCs enables cross-domain transfer that nearly matches target-trained probes. Source-side or target-side separability alone does not identify this subset. An LLM judge’s interpretation score over each PC’s top-activating examples partly recovers it without target data.

Robust probing therefore requires identifying not only whether a behavioral signal is linearly accessible, but where it is most robustly represented. Full-representation probes may combine general behavioral features with dataset-specific surface cues. Such heuristics can support high in-distribution accuracy while harming transfer.

Sparse PC subspaces may better isolate general behavioral structure. This reframes OOD probing as representation selection: the goal is to fit a classifier **and** choose a subspace that preserves the intended behavioral feature while filtering spurious correlation.

**Claims and evidence**
- A small source-PC subspace nearly matches target-trained performance: 8–15 components; Section 4.2, Figure 3.
- Source- and target-side separability do not recover it: Section 4.3, Figure 4; Appendix D reports overlaps of 0.182 / 0.077 / 0.167.
- Interpretation partly recovers it without target data: 19 PCs, with 78% Report and 25% Sandbagging gap recovery; Section 4.5, Figure 5.
- Interpretation remains descriptive rather than causal: Section 5 reports no inference-time ablation of high-scoring PCs.

## Acknowledgment

*The work was supported by AI Safety Camp compute and discussions with the named contributors.*

This work was conducted during the 11th edition of AI Safety Camp, which provided the compute to run our experiments. Thanks to Prasad Mahadik and Arya Hariharan for useful discussions around this work.

## Impact Statement

*The work aims to advance Machine Learning, and the authors identify no societal consequence requiring specific emphasis.*

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none
which we feel must be specifically highlighted here.

## Appendix A Dataset Details

*The appendix describes the 4 deception datasets and reports their full label distributions.*

**Roleplaying:** the model is placed in a simple scenario with an opportunity to lie.

**Insider Trading Report:** acting as a trader, the model decides whether to honestly report that it used insider information.

**Insider Trading Confirm:** after hiding earlier use of insider information, the model is asked to confirm it did not use any.

**Sandbagging:** the model is told to deliberately under-perform on some tasks while behaving normally on others.

In the table, label counts are ordered as ambiguous (-1), honest (0), and deceptive (1).

| Dataset | Total Examples | Label Counts (-1 / 0 / 1) |
|---|---|---|
| Roleplaying | 371 | 140 / 59 / 172 |
| Insider Trading Report | 473 | 218 / 194 / 61 |
| Insider Trading Confirm | 151 | 29 / 38 / 84 |
| Sandbagging | 1000 | 379 / 375 / 246 |

Table 1: Dataset label distributions.

**Claims and evidence**
- Roleplaying contains 371 examples: 140 / 59 / 172 by label; Table 1, “Roleplaying.”
- Insider Trading Report contains 473 examples: 218 / 194 / 61; Table 1, “Insider Trading Report.”
- Insider Trading Confirm contains 151 examples: 29 / 38 / 84; Table 1, “Insider Trading Confirm.”
- Sandbagging contains 1000 examples: 379 / 375 / 246; Table 1, “Sandbagging.”

## Appendix B LLM Judge Prompt

*The prompt asks GPT-5-mini to describe both PC poles and score likely OOD deception usefulness.*

We use GPT-5-mini (OpenAI, 2025), the direction-interpretation model, to interpret directions. In the prompt, `token<TAB>activation` gives the token–activation line format; `<TAB>` separates the fields. `{positive_examples}` and `{negative_examples}` are placeholders for the two example groups. Activations are normalized to integers from -10 to +10. The final four labels are the requested response fields, with `...` marking the answer placeholders.

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

*Lower-index PCs align more strongly with SAE decoder features.*

Decoder alignment is the match between a PC and an SAE decoder feature. Spearman \(\rho\) is the statistic reported for how this alignment varies with PC index. The resource code `andyrdt/saes-llama-3.1-8b-instruct` identifies the SAE used, attributed to andyrdt (2024).

Figure 6: Lower PCs show stronger decoder alignment, both for the best-matching SAE feature and for the mean of the top 10 features (Spearman \(\rho\) = -0.857 and -0.917, respectively). We used andyrdt/saes-llama-3.1-8b-instruct (andyrdt, 2024)

.

**Claims and evidence**
- Lower PCs have stronger decoder alignment: Spearman \(\rho=-0.857\) for the best-matching SAE feature and \(\rho=-0.917\) for the mean of the top 10 features; Figure 6.

## Appendix D PC Set Comparisons

*Greedy and target-PCA selections overlap little, and the table lists PCs selected by each method.*

Jaccard overlap, the reported PC-set overlap statistic, between greedy and target-pca is 0.182 for Insider Trading Report, 0.077 for Insider Trading Confirm, and 0.167 for Sandbagging. The 2 methods agree on less than a fifth of their selections.

Together, these results suggest that the transferable subspace is neither the most source-discriminative nor the most target-discriminative part of representation space.

In Table 2, each `PC` code identifies the corresponding numbered source principal component. “Target separability ranking” means the ranking based on target-probe weights.

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

**Claims and evidence**
- Greedy and target-pca agree on less than a fifth of their selections: overlaps 0.182 / 0.077 / 0.167 for Report / Confirm / Sandbagging; the overlap results above.
- The selected PC sets differ: Table 2, paired “Greedy selection” and “Target separability ranking” rows for each target.
- The transferable subspace is not simply the most source- or target-discriminative space: Table 2’s “Source probe weights,” “Greedy selection,” and “Target separability ranking” rows, together with Figure 4’s transfer comparisons.

## Appendix E PC Interpretability

*The tables give judge scores and full explanations for repeatedly greedy-selected PCs and leading source-probe PCs.*

In Table 3, `+` joins targets that selected the same PC. “PR-style wording” is the defensive wording discussed in the PC86 explanation. In Table 4, “QA/roleplay” is the task description used in the PC8 explanation.

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

**Claims and evidence**
- Three repeatedly greedy-selected PCs receive 6: Table 3, PC62, PC57, and PC86. The other two receive 3: PC9 and PC81.
- Only one of the five leading source-probe PCs receives 6: Table 4, PC4. PC7 / PC8 / PC11 / PC12 receive 3 / 4 / 3 / 3.
- High scores retain substantial caveats: Table 3, PC62, PC57, and PC86 describe noise, lexical dependence, brittleness, or false positives; Table 4, PC4 also warns of surface cues and wording dependence.
