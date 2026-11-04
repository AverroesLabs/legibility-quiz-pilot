# When Is an SAE Feature Interpretable?

A Validation Ladder for EEG Foundation Models

Yucong Cao

Affiliation: Department of Engineering Science

Affiliation: University of Oxford

  
Chenqi Li

Affiliation: Department of Engineering Science

Affiliation: University of Oxford

  
Tingting Zhu

Affiliation: Department of Engineering Science

Affiliation: University of Oxford

###### Abstract

Across 27 settings (three backbones, three EEG datasets, three depths), an
8–12 Hz band-stop changes SAE latent firing $7.3\times$ more than an
equal-width sham notch (95% CI $[6.2,8.7]$, bootstrapped over settings).
Per unit removed spectral energy the ratio is $0.28\times$ (95% CI
$[0.22,0.36]$), exceeding one in none of the 27 settings. Latents selected
on alpha-removal response have mean $r=-0.073$ with clean-EEG relative
alpha power. We give a validation ladder for semantic readings of SAE
latents—response, distortion-controlled response, specificity against broad
fragility, and clean-data correspondence—with task-classifier relevance as a
separate branch.

## 1 Introduction

SAEs decompose dense foundation-model (FM) activations into discrete latents
amenable to naming (Gao et al., 2025; Cunningham et al., 2023; Bricken et al., 2023).
We audit “alpha feature” labels on frozen
CBraMod (Wang et al., 2025), REVE (El Ouahidi et al., 2025), and
DINOv3 (Siméoni et al., 2025) at early, middle, and late depth on three
public EEG tasks. Raw alpha-over-sham firing change is $7.3\times$ in
every one of 27 settings; per unit removed spectral energy it is
$0.28\times$, exceeding one in $0/27$. Latents are chosen by perturbation
response and evaluated on clean signals; the selected units fire when
relative alpha power is lower, not higher.

The ladder (Figure 1; Table 1): Stages 1–3 (sensitivity, distortion-controlled
selectivity, specificity) and Stage 5 (independent observational validation)
yield a representational claim; Stage 4 (readout intervention) is a side
branch, required only for functional relevance to a specified readout.

The individual diagnostics—sensitivity, specificity, intervention,
downstream relevance, external validation—are standard. The contribution is
the identification of a recurrent inferential failure in SAE-based
biosignal interpretation and an operational hierarchy of alternatives to
exclude before physiological labelling. SAE interpretability in language
models targets sparsity, reconstruction, monosemanticity, or causal
editability (Cunningham et al., 2023; Bricken et al., 2023; Marks et al., 2025);
EEG explainability targets input attribution (Schirrmeister et al., 2017).
Our question: when does a perturbation-responsive SAE latent warrant a
physiological label?

DINOv3 tests persistence under a foundation model pretrained in another
modality. Contributions: (i) a forked validation protocol supporting claims
at graded strength; (ii) a 27-setting demonstration that a $7.3\times$
alpha-sensitive signature is largely an input-distortion artifact; (iii)
specialist-like latents neither track clean-EEG alpha in the intended
direction nor are preferentially important to a linear readout.

Table 1: Evidence available after each stage, and the claim that evidence
supports. None of these establish a causal neural mechanism.

|

Evidence

|

Claim supported

|

Responds to alpha removal

|

Alpha-sensitive latent

|

Response survives distortion control and competing perturbations

|

Alpha-selective latent

|

Above, and independently covaries with alpha structure in clean EEG

|

Evidence for an alpha-representing latent

|

Above, and readout intervention changes task behavior

|

Task-relevant alpha representation

Figure 1: Forked validation ladder. Stages 1–3 plus stage 5 lead to a
representational reading; stage 4 to a task-head reading. Neither is
reached here.

## 2 Experimental design

#### Backbones and data.

CBraMod, REVE, and DINOv3 ViT-B/16 (channel-by-time EEG images, prefix
tokens excluded) on three 200 Hz sets: BCI Competition IV
2a (Tangermann et al., 2012; Brunner et al., 2008) (BCI; motor imagery, 22
channels), PhysioNet Mental Arithmetic (Zyma et al., 2019; Goldberger et al., 2000)
(MA; workload, 20 channels), and BCI Competition 2020 Track 3 imagined
speech (Jeong et al., 2022; Lee et al., 2020) (Speech; 64 channels).
Fixed splits, frozen backbones; backbone × dataset × depth gives 27
settings. Licenses and compute: Appendix F.

#### TopK SAEs.

For $x\in\mathbb{R}^{d}$ ,

| | $c=\operatorname{ReLU}\!\left(\operatorname{TopK}_{k}\left((x-b_{\rm dec})W_{\rm enc}+b_{\rm enc}\right)\right),\qquad\hat{x}=cW_{\rm dec}+b_{\rm dec}.$
|

$k=32$; loss MSE + AuxK (Gao et al., 2025). Early/middle/late block tokens,
layer-normalized to $\mathbb{E}\|x\|_{2}=\sqrt{d}$ before mixing. Expansion
$16\times$, or $4\times$/$8\times$ for DINOv3 under limited token support;
27 dictionaries, test FVU $[0.059,0.228]$, $L_{0}=32$. BCI middle of each
backbone retrained over three seeds (Appendix A).

#### Perturbations and distortion control.

Clean test trials paired with perturbed copies; SAEs frozen. Spatial: random
channel or contiguous-window zeroing. Spectral: fourth-order zero-phase
Butterworth band-stop at $\delta,\theta,\alpha,\beta,\gamma$ , high-
$\gamma$ , and 8–12 Hz alpha vs. width-matched 30–34 Hz sham. Recorded:
sparse-code cosine, active-set Jaccard, $|\Delta f_{j}|$. Under the $1/f$
EEG spectrum, alpha/sham removed-energy ratios are 57.2 / 34.6 / 11.8 on
BCI / MA / Speech. Responses are reported raw and divided by
$\|x-\tilde{x}\|_{2}/\|x\|_{2}$ or by removed spectral-energy fraction.

#### Selectivity, probes, and independent validation.

$s_{j,b}=|\Delta f_{j,b}|-\frac{1}{5}\sum_{b^{\prime}\neq b}|\Delta f_{j,b^{\prime}}|$ ;
specialist-like alpha latents: top-quartile $s_{j,\alpha}$, channel-dropout
fragility $\le$ median. Readout contrasts: size-matched general-fragile and
random groups; clean-alpha contrast: activity-matched non-specialists
(nearest clean firing rate and mean activation, without replacement).
Logistic probes on mean-pooled codes, fixed split; importance
$|w_{j}|\mathbb{E}_{\rm train}[c_{j}]$. Group ablation zeroes a size-matched
set (specialist-like, general-fragile, random) and records $\Delta$
balanced accuracy, with the probe frozen (trained on clean codes, ablation
at test) or refit after ablation. Clean activation is correlated with
trial-wise relative alpha power (unused in selection) for 12,041 latents in
each named group.

CIs: percentile 95%, 10,000 bootstrap resamples of settings (27, or 21 for
task analyses), not latents.

## 3 Results

### 3.1 A large raw alpha response collapses under distortion control

Alpha/sham firing change is 5–9 $\times$ per backbone; alpha lowers code
overlap in all 27 settings (Appendix B). In the six-band sweep alpha is the
maximum- $|\Delta f|$ and minimum-Jaccard band in 15/27 settings.
Channel-dropout robustness decreases with depth in all nine
backbone–dataset combinations, most for DINOv3 (Appendix C).

Table 2: Pooled alpha/sham $|\Delta f|$ ratio, 27 settings, raw and
normalized; 95% CIs bootstrapped over settings.

| $|\Delta f|$ ratio, 27 settings | Mean | 95% CI | Settings $>1$

| raw | 7.33 | $[6.18,8.68]$ | 27/27

| per unit time-domain distortion | 1.32 | $[1.12,1.55]$ | 17/27

| per unit removed spectral energy | 0.28 | $[0.22,0.36]$ | 0/27

The removed-energy denominator varies almost fivefold across datasets; the
two normalizations bracket the ratio. Code-overlap ratios: raw 2.77, then
0.51 and 0.11.

### 3.2 Alpha selectivity is not readout importance

Probe balanced accuracy 0.236–0.625; ablations compared within setting.
Selectivity–importance Pearson $r=0.107$ (CI $[0.068,0.146]$), 21 settings
(Figure 2A); $|r|\leq 0.32$ in all nine BCI-middle retrainings.

Frozen-probe $\Delta$BA: specialist-like $+0.0055$ (CI $[-0.007,0.027]$),
size-matched fragile $-0.028$ (CI $[-0.051,-0.003]$), random $-0.019$
(CI $[-0.033,-0.008]$); smaller after retraining. Ordering is seed-dependent
on BCI middle, and fragile latents fire $8.9\times$ more; perturbation-defined
specialists are not preferentially important to this linear readout.
Nonlinear and fine-tuned readouts are untested; Stage 4 lies off the
representational path (Figure 1).

### 3.3 Independent observational validation fails in the intended direction

Top-activating trials: specialists exceed matched fragile exemplars by
$+0.012$ relative alpha power (higher in 12/27). All clean trials: mean
$|r|$ 0.143 (specialists), 0.246 (size-matched fragile; 26/27), 0.097
(random; Figure 2B). Specialist signed $r=-0.073$, negative in 23/27.

Versus activity-matched non-specialists (firing $0.374$ vs. $0.374$):
$|r|$ $0.143$ vs. $0.101$ ($27/27$; $\Delta|r|$ CI $[0.032,0.052]$); sign
unchanged. No alpha-representing latent is supported; “specialist-like”
denotes a perturbation response. Structured spatial–spectral features are
not excluded, only a global frequency-detector reading.

Figure 2: A: per-setting $r$, alpha selectivity vs. activation-weight
importance (REVE: BCI only); means near 0. B: per-setting mean $|r|$ with
relative alpha power, specialists vs. random; signed $r$ negative on
average. 21-setting ablation chart: Appendix D.

## 4 Discussion and limitations

The ladder excludes specific alternatives; it is not a feature finder. The
claim concerns the $7.3\times$ alpha signature, not SAE latents in general.
A positive control on another physiological axis is deferred to a longer
version.

Limits: one SAE seed outside BCI middle; readout groups unmatched on
baseline activity (matched for clean-alpha only); $|\Delta f|$ may include
TopK rank displacement; relative band power tests only a global frequency
reading; relevance defined for a logistic probe on mean-pooled codes;
task-readout products cover 21/27 settings (REVE MA/Speech exports
unavailable); band-stop filtering cannot isolate a biological oscillator,
and the two normalizations bound rather than remove that confound. No
claims on clinical validity or neural causality.

## 5 Conclusion

The $7.3\times$ alpha-over-sham response, present in every setting, fails
distortion control, does not mark latents tracking clean-EEG alpha in the
intended direction, and does not identify units a linear readout depends
on. Semantic names are to be matched to the evidence stage of Table 1.

#### Ethics statement.

Reanalysis of public EEG datasets; no participants recruited, no clinical
predictions; consent and ethics inherited from the original releases.

## Appendix A Additional experimental details

Layer groups (per-layer $\ell_{2}$ normalization): blocks 0–3 / 4–7 / 8–11
for CBraMod and DINOv3 (12 blocks); 0–6 / 7–14 / 15–21 for REVE (22
blocks). DINOv3 width $4\times$ on Mental Arithmetic, $8\times$ on BCI;
otherwise $16\times$. MSE + AuxK, $k=32$. Multi-seed: BCI middle only.

Top-five clean test activations streamed for up to three alpha
specialist-like and matched fragile latents per setting plus smaller beta
and high-gamma sets (298 latents; 1,490 exemplars); Welch spectra give
relative target-band power. All-trial Pearson $r$ is primary, as it does
not condition on the activation extreme.

Activity matching within dictionary: unused non-specialist minimizing
squared $z$ -distance in clean firing rate and mean activation
(standardized on specialists ∪ pool). Firing 0.374 vs. 0.374; activation
0.0060 vs. 0.0062.

## Appendix B Per-backbone raw alpha/sham contrast

Table 3: Mean active-set Jaccard and $|\Delta f|$, alpha (8–12 Hz) vs.
sham (30–34 Hz), over the nine dataset–depth settings per backbone.

| | Jaccard | $|\Delta f|$ | Ratio

| Model | $\alpha$ | sham | $\alpha$ | sham | $\alpha$ /sham

| CBraMod | .541 | .819 | .00338 | .00046 | 7.4

| REVE | .466 | .841 | .00127 | .00014 | 8.9

| DINOv3 | .484 | .782 | .00169 | .00033 | 5.1

## Appendix C Depth-dependent channel-dropout robustness

In all nine model–dataset combinations late-layer active-set overlap is
below early-layer overlap (mean late-minus-early Jaccard $-0.17$ random,
$-0.20$ contiguous). Contiguous $\Delta$ Jaccard: DINOv3 $\approx-0.35$,
CBraMod $\approx-0.19$, REVE $\approx-0.07$. Deeper SAE representations are
less stable to spatial perturbation in EEG-native and cross-modal
backbones alike; vision-FM transfer need not bring matching internal
robustness.

## Appendix D Full 21-setting ablation chart

Figure 3 is the per-setting frozen-probe ablation display summarized by
Figure 2.

Figure 3: Frozen-probe balanced accuracy change after zeroing
size-matched latent groups, 21 settings.

## Appendix E Setting-level bootstrap and activity matching

Percentile 95% CIs, 10,000 bootstrap resamples of settings.

| Quantity | Mean | 95% CI

| raw $\alpha$ /sham $|\Delta f|$ | 7.33 | $[6.18,8.68]$

| time-normalized ratio | 1.32 | $[1.12,1.55]$

| spectral-normalized ratio | 0.28 | $[0.22,0.36]$

| selectivity–importance $r$ (21 settings) | 0.107 | $[0.068,0.146]$

| $\Delta|r|$ specialist $-$ random control | 0.046 | $[0.033,0.059]$

| $\Delta|r|$ specialist $-$ activity-matched | 0.042 | $[0.032,0.052]$

| frozen $\Delta$ BA specialist | $+0.0055$ | $[-0.007,0.027]$

| frozen $\Delta$ BA fragile | $-0.028$ | $[-0.051,-0.003]$

| frozen $\Delta$ BA random control | $-0.019$ | $[-0.033,-0.008]$

## Appendix F Licenses, assets, and compute

•

BCI Competition IV 2a: research use; cite
Tangermann et al. [2012]; Graz description
[Brunner et al., 2008].

•

Mental Arithmetic (EEGMAT): CC BY 4.0; hosted on PhysioNet
[Zyma et al., 2019, Goldberger et al., 2000].

•

BCI Competition 2020 Track 3: competition / research use
[Jeong et al., 2022, Lee et al., 2020].

•

CBraMod weights: as released by the authors; reference
implementation BSD-3-Clause [Wang et al., 2025].

•

REVE-base / positions: as released by the authors
[El Ouahidi et al., 2025].

•

DINOv3 ViT-B/16: DINOv3 License (Meta) [Siméoni et al., 2025].

NVIDIA A100 GPUs, shared academic Slurm cluster. Pipeline: token-level
activation export for 27 settings, 27 TopK SAEs plus 9 multi-seed
BCI-middle retrains, perturbation sweeps, linear probes on 21 settings,
CPU-side analyses; on the order of 300 GPU-hours, excluding unused
exploratory exports and trainings. Anonymized training and analysis code
in supplementary material; checkpoints and raw EEG omitted for size and
available from the cited public sources.

## NeurIPS Paper Checklist

1.

Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction state the empirical scope, and Sections 3–4 distinguish observations from physiological or causal claims.

Guidelines:

•

The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

•

The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

•

The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

•

It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

2.

Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Section 4 discusses SAE seeds, TopK competition, activity matching, linear-probe scope, missing task-analysis settings, and dataset scope.

Guidelines:

•

The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

•

The authors are encouraged to create a separate “Limitations” section in their paper.

•

The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

•

The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

•

The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

•

The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

•

If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

•

While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

3.

Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A]

Justification: The paper reports empirical analyses and makes no theoretical claims.

Guidelines:

•

The answer [N/A] means that the paper does not include theoretical results.

•

All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.

•

All assumptions should be clearly stated or referenced in the statement of any theorems.

•

The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

•

Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

•

Theorems and Lemmas that the proof relies upon should be properly referenced.

4.

Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Section 2 gives models, data splits, preprocessing, SAE construction, perturbations, metrics, and probe protocol; the supplementary code contains exact configurations.

Guidelines:

•

The answer [N/A] means that the paper does not include experiments.

•

If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

•

If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

•

Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

•

While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a)

If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b)

If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c)

If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d)

We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

5.

Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: Datasets and backbone implementations are public and cited. An anonymized code archive with training and analysis scripts is provided as supplementary material; checkpoints and raw EEG are omitted because of size.

Guidelines:

•

The answer [N/A] means that paper does not include experiments requiring code.

•

Please see the NeurIPS code and data submission guidelines (https://neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

•

While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

•

The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https://neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

•

The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

•

The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

•

At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

•

Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

6.

Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: Section 2 reports the split policy, preprocessing, SAE sparsity and expansion, perturbation bands, metrics, and classifier protocol; full training details are in the supplement.

Guidelines:

•

The answer [N/A] means that the paper does not include experiments.

•

The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

•

The full details can be provided either with the code, in appendix, or as supplemental material.

7.

Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Main-text means for alpha/sham ratios, selectivity–importance correlations, clean-alpha $|r|$ contrasts, and group ablations are reported with percentile 95% confidence intervals from 10,000 bootstrap resamples of settings (27 settings, or 21 for task analyses). The resampling unit is the setting, not the latent. Initialization variability is additionally quantified for the BCI middle-layer setting under three SAE seeds.

Guidelines:

•

The answer [N/A] means that the paper does not include experiments.

•

The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

•

The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

•

The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

•

The assumptions made should be given (e.g., Normally distributed errors).

•

It should be clear whether the error bar is the standard deviation or the standard error of the mean.

•

It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

•

For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

•

If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

8.

Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: Appendix F reports NVIDIA A100 GPUs on a shared academic Slurm cluster and an estimate of about 300 GPU-hours for the reported pipeline, excluding unused exploratory runs.

Guidelines:

•

The answer [N/A] means that the paper does not include experiments.

•

The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

•

The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

•

The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

9.

Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The study reanalyzes public research datasets, reports negative findings and limitations, and makes no clinical or individual-level claims.

Guidelines:

•

The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

•

If the authors answer [No] , they should explain the special circumstances that require a deviation from the Code of Ethics.

•

The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

10.

Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Section 4 motivates stricter validation of biosignal interpretations and limits the findings to representation auditing rather than clinical or neural causality.

Guidelines:

•

The answer [N/A] means that there is no societal impact of the work performed.

•

If the authors answer [N/A] or [No] , they should explain why their work has no societal impact or why the paper does not address societal impact.

•

Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

•

The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

•

The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

•

If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

11.

Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: The paper releases no new generative model, personal data, or high-risk dataset; it analyzes existing frozen models and aggregate results.

Guidelines:

•

The answer [N/A] means that the paper poses no such risks.

•

Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

•

Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

•

We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

12.

Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Section 2 cites the original dataset releases. Appendix F lists licenses or terms of use for the three datasets and three backbone checkpoints.

Guidelines:

•

The answer [N/A] means that the paper does not use existing assets.

•

The authors should cite the original paper that produced the code package or dataset.

•

The authors should state which version of the asset is used and, if possible, include a URL.

•

The name of the license (e.g., CC-BY 4.0) should be included for each asset.

•

For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

•

If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

•

For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

•

If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

13.

New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: The SAE training and analysis code records configurations, output schemas, random seeds, and run manifests in the supplementary repository.

Guidelines:

•

The answer [N/A] means that the paper does not release new assets.

•

Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

•

The paper should discuss whether and how consent was obtained from people whose asset is used.

•

At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

14.

Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: This secondary analysis did not recruit participants or run crowdsourcing experiments.

Guidelines:

•

The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

•

Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

•

According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

15.

Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: No new human-subject data were collected. Section 2 and Appendix F cite the original public dataset releases.

Guidelines:

•

The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

•

Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

•

We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

•

For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

16.

Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: LLM assistance was limited to writing and formatting and was not part of the scientific method, so declaration is not required under this question.

Guidelines:

•

The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

•

Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.