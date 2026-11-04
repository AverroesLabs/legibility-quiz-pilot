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

Sparse autoencoders (SAEs) factor dense foundation-model activations into
discrete latents that readily acquire semantic labels. For EEG foundation
models, the canonical case is the latent whose firing shifts under alpha-band
removal. Across 27 backbone–dataset–depth settings (three backbones, three
EEG datasets, three depths), alpha removal changes latent firing $7.3\times$
more than an equal-width sham notch (95% CI $[6.2,8.7]$, bootstrapped over
settings). Normalized by removed spectral energy, the ratio is $0.28\times$
(95% CI $[0.22,0.36]$) and exceeds one in none of the 27 settings. Latents
selected on alpha-removal response are, on clean EEG, weakly anti-correlated
with relative alpha power (mean $r=-0.073$). We propose a validation ladder
for semantic interpretations of SAE latents comprising sensitivity,
distortion-controlled selectivity, specificity against broad fragility, and
observational correspondence on unperturbed data, with readout intervention
as a separate test of task relevance. Perturbation sensitivity alone does not
establish what an SAE latent represents.

## 1 Introduction

Foundation models (FMs) for brain and body signals raise the question of what
their representations encode. SAEs decompose dense activations into discrete
latents amenable to individual naming (Gao et al., 2025; Cunningham et al., 2023; Bricken et al., 2023);
discreteness is not semantic identifiability, and a latent responsive to an
alpha band-stop is routinely termed an “alpha feature” without further test.

The signature in question appears in all 27 settings spanning frozen
CBraMod (Wang et al., 2025), REVE (El Ouahidi et al., 2025), and
DINOv3 (Siméoni et al., 2025) at early, middle, and late depth on three
public EEG tasks: raw alpha-over-sham firing change is $7.3\times$. Under
spectral-energy normalization the ratio is $0.28\times$ and exceeds one in
$0/27$ settings. Selection uses perturbation responses; evaluation uses clean
signals. Latents surviving the subsequent filters fire, on clean EEG, on
trials of lower rather than higher relative alpha power.

The ladder (Figure 1; Table 1) comprises sensitivity, distortion control,
and specificity (Stages 1–3), and independent observational validation
(Stage 5), which jointly bound a representational claim. Readout
intervention (Stage 4) is a side branch, required only for the claim that a
feature is functionally relevant to a specified readout. A latent tracking
alpha perfectly could be irrelevant to imagined speech without thereby
failing to be an alpha feature.

The individual diagnostics—sensitivity, specificity, intervention,
downstream relevance, external validation—are standard; the contribution is
the identification of a recurrent inferential failure in SAE-based biosignal
interpretation and an operational hierarchy of alternatives to be excluded
before physiological semantics are assigned. SAE interpretability work
addresses sparsity, reconstruction, monosemanticity, and causal editability
in language models (Cunningham et al., 2023; Bricken et al., 2023; Marks et al., 2025);
EEG explainability addresses input attribution for predictions
(Schirrmeister et al., 2017). We address when a perturbation-responsive SAE
latent warrants a physiological label.

DINOv3 tests whether the problem is specific to EEG-pretrained
representations or persists for an FM pretrained in another modality.
Contributions: (i) a forked validation protocol supporting claims of
graded strength; (ii) a 27-setting demonstration that a $7.3\times$
alpha-sensitive signature is largely attributable to input distortion; and
(iii) evidence that the residual specialist-like latents neither track
clean-EEG alpha in the intended direction nor are preferentially important
to a linear readout.

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

Figure 1: Forked validation ladder. Stages 1–3 and stage 5 (independent
observational validation) bound a representational reading; stage 4
(readout intervention) bounds task relevance to a specified head. Neither
claim is supported here.

## 2 Experimental design

#### Backbones and data.

EEG-pretrained CBraMod and REVE are compared with DINOv3 ViT-B/16 on
channel-by-time EEG images (prefix tokens excluded). Datasets, all at
200 Hz: BCI Competition IV 2a (Tangermann et al., 2012; Brunner et al., 2008)
(BCI; motor imagery, 22 channels), PhysioNet Mental Arithmetic
(Zyma et al., 2019; Goldberger et al., 2000) (MA; workload, 20 channels),
and BCI Competition 2020 Track 3 imagined speech
(Jeong et al., 2022; Lee et al., 2020) (Speech; 64 channels). Splits fixed,
backbones frozen; backbone × dataset × depth yields 27 settings. Licenses and
compute: Appendix F.

#### TopK SAEs.

For $x\in\mathbb{R}^{d}$ ,

| | $c=\operatorname{ReLU}\!\left(\operatorname{TopK}_{k}\left((x-b_{\rm dec})W_{\rm enc}+b_{\rm enc}\right)\right),\qquad\hat{x}=cW_{\rm dec}+b_{\rm dec}.$
|

$k=32$; loss is reconstruction MSE plus AuxK dead-latent recovery
(Gao et al., 2025). Early, middle, or late token activations are
layer-normalized to $\mathbb{E}\|x\|_{2}=\sqrt{d}$ prior to mixing. Expansion
is $16\times$, reduced to $4\times$ or $8\times$ for DINOv3 under limited
token support; 27 dictionaries, all converged to test FVU in
$[0.059,0.228]$ at $L_{0}=32$. The BCI middle setting of each backbone is
retrained under three seeds (Appendix A).

#### Perturbations and distortion control.

Each clean test trial is paired with a perturbed copy; SAEs remain frozen.
Spatial interventions zero random channels or contiguous windows. Spectral
interventions are fourth-order zero-phase Butterworth band-stops at
$\delta,\theta,\alpha,\beta,\gamma$, and high-$\gamma$, plus an 8–12 Hz
alpha test against a width-matched 30–34 Hz sham. Responses: sparse-code
cosine, active-set Jaccard, and $|\Delta f_{j}|$. Under the $1/f$ EEG
spectrum, alpha/sham removed-energy ratios are 57.2 / 34.6 / 11.8 on
BCI / MA / Speech. SAE responses are divided by time-domain
$\|x-\tilde{x}\|_{2}/\|x\|_{2}$ or by fractional removed spectral energy.

#### Selectivity, probes, and independent validation.

$s_{j,b}=|\Delta f_{j,b}|-\frac{1}{5}\sum_{b^{\prime}\neq b}|\Delta f_{j,b^{\prime}}|$ .
Specialist-like alpha latents: top quartile of alpha selectivity and at or
below median channel-dropout fragility. Readout contrasts are size-matched
general-fragile and random groups; clean-alpha contrasts are
activity-matched non-specialists (nearest in clean firing rate and mean
activation, without replacement). Logistic probes on mean-pooled SAE codes,
fixed split; importance $|w_{j}|\mathbb{E}_{\rm train}[c_{j}]$. Group
ablation zeroes the codes of a size-matched set (specialist-like,
general-fragile, random) and records the change in balanced accuracy, either
with a probe trained on clean codes and ablated at test time only (frozen)
or refit post-ablation (retrained). Observational validation correlates
clean activation with trial-wise relative alpha power, unused in selection,
for 12,041 latents in each named group.

Intervals are percentile 95% CIs from 10,000 bootstrap resamples of
settings (27, or 21 for task analyses), not of latents.

## 3 Results

### 3.1 A large raw alpha response collapses under distortion control

Alpha removal changes firing 5–9 $\times$ more than sham in every backbone
and lowers code overlap in all 27 settings (Appendix B). In the six-band
sweep alpha is the maximum-$|\Delta f|$, minimum-Jaccard band in 15/27
settings. Channel-dropout robustness decreases with depth in all nine
backbone–dataset combinations, most strongly for DINOv3 (Appendix C).

Table 2: Pooled alpha/sham $|\Delta f|$ ratio across 27 settings, raw and
distortion-normalized, with 95% CIs bootstrapped over settings.

| $|\Delta f|$ ratio, 27 settings | Mean | 95% CI | Settings $>1$

| raw | 7.33 | $[6.18,8.68]$ | 27/27

| per unit time-domain distortion | 1.32 | $[1.12,1.55]$ | 17/27

| per unit removed spectral energy | 0.28 | $[0.22,0.36]$ | 0/27

The removed-energy denominator varies almost fivefold across datasets, so
the two normalizations bracket rather than fix the ratio; under either,
alpha sensitivity is not distortion-independent (Table 2). Code-overlap
ratios: raw 2.77, normalized 0.51 and 0.11.

### 3.2 Alpha selectivity is not readout importance

Probe balanced accuracy spans 0.236–0.625; ablations are compared within
settings. Selectivity–importance Pearson $r$ averages $0.107$
(CI $[0.068,0.146]$) over the 21 settings with task-analysis codes
(Figure 2A), with $|r|\leq 0.32$ in all nine BCI-middle retrainings.

Frozen-probe balanced-accuracy change on zeroing: specialist-like
$+0.0055$ (CI $[-0.007,0.027]$), size-matched fragile $-0.028$
(CI $[-0.051,-0.003]$), random $-0.019$ (CI $[-0.033,-0.008]$). Effects
shrink under retraining. Ordering is seed-dependent on the BCI middle
setting and groups are unmatched on baseline activity (fragile latents fire
$8.9\times$ more); the result is directional and restricted to the linear
readout, excluding nonlinear or fine-tuned behavior, and per
Figure 1 bears only on task relevance.

### 3.3 Independent observational validation fails in the intended direction

Top-activating-trial relative alpha power exceeds that of matched fragile
exemplars by $+0.012$ (higher in 12/27 settings). All-trial mean $|r|$ with
relative alpha power: specialists 0.143, size-matched fragile 0.246 (higher
in 26/27 settings), random 0.097 (Figure 2B). Specialist signed $r$ is
$-0.073$, negative in 23/27 settings.

Against activity-matched non-specialists (mean firing $0.374$ vs. $0.374$),
specialists retain $|r|$ of $0.143$ vs. $0.101$ ($27/27$ settings;
$\Delta|r|$ CI $[0.032,0.052]$) with sign unchanged. “Specialist-like”
denotes a perturbation response. Structured spatial–spectral features are
not excluded; only a simple global frequency-detector reading is.

Figure 2: A: per-setting Pearson $r$ between alpha selectivity and
activation-weight importance (REVE task codes BCI only); means near 0.
B: per-setting mean $|r|$ of clean activation with relative alpha power,
specialists vs. random controls; signed $r$ in Section 3.3. The 21-setting
ablation chart is in Appendix D.

## 4 Discussion and limitations

The ladder is a protocol for excluding enumerated alternatives, not a
feature finder (Table 1). The claim is not that no SAE latent is
interpretable, but that the $7.3\times$ alpha signature does not support an
alpha-representing reading. A positive control on another physiological axis,
testing whether the hierarchy can accept a label, is deferred to a longer
version.

Limits: remaining settings use one SAE seed; readout groups are unmatched
on baseline activity (addressed for clean-alpha correspondence, not
ablation); $|\Delta f|$ can include TopK rank displacement; relative band
power tests only a global frequency reading; relevance is defined for a
logistic probe on mean-pooled codes; task-readout products cover 21/27
settings (REVE MA/Speech exports were unavailable); band-stop filtering
cannot isolate a biological oscillator, and the two normalizations bound
that confound rather than remove it. Conclusions concern semantic claims
about fitted SAE latents, not clinical validity or neural causality.

## 5 Conclusion

A $7.3\times$ alpha-over-sham SAE response, present in every setting
studied, fails distortion control, does not mark latents tracking clean-EEG
alpha in the intended direction, and does not identify units on which a
linear readout depends. Semantic names for SAE features should be indexed to
the evidence stage in Table 1.

#### Ethics statement.

This study reanalyzes publicly available EEG datasets, recruits no
participants, and makes no clinical predictions; consent and ethics
procedures are inherited from the original releases.

## Appendix A Additional experimental details

Layer groups mix consecutive transformer blocks after per-layer $\ell_{2}$
normalization: blocks 0–3 / 4–7 / 8–11 for CBraMod and DINOv3 (12
blocks) and 0–6 / 7–14 / 15–21 for REVE (22 blocks). DINOv3 dictionary
width is $4\times$ on Mental Arithmetic and $8\times$ on BCI; all other
settings use $16\times$. Training uses reconstruction MSE plus AuxK with
$k=32$. Multi-seed analyses retrain only the BCI middle setting of each
backbone.

For observational validation we additionally stream, for up to three alpha
specialist-like and matched fragile latents per setting plus smaller beta and
high-gamma sets, each latent’s top-five clean test activations (298 selected
latents; 1,490 exemplars), quantifying relative target-band power via Welch
spectra. The all-trial Pearson $r$ is primary, being unconditioned on the
activation extreme.

Activity-matched controls are drawn from the same dictionary: for each
specialist, the unused non-specialist minimizing squared $z$-distance in clean
firing rate and mean activation (standardized on the union of specialists and
candidate pool). Mean matched firing 0.374 vs. 0.374; mean matched activation
0.0060 vs. 0.0062.

## Appendix B Per-backbone raw alpha/sham contrast

Table 3: Mean active-set Jaccard and mean $|\Delta f|$ under alpha
(8–12 Hz) vs. sham (30–34 Hz), averaged over the nine
dataset–depth settings of each backbone. Raw rows are omitted from the main
table; the normalized 27-setting ratios carry the inferential claim.

| | Jaccard | $|\Delta f|$ | Ratio

| Model | $\alpha$ | sham | $\alpha$ | sham | $\alpha$ /sham

| CBraMod | .541 | .819 | .00338 | .00046 | 7.4

| REVE | .466 | .841 | .00127 | .00014 | 8.9

| DINOv3 | .484 | .782 | .00169 | .00033 | 5.1

## Appendix C Depth-dependent channel-dropout robustness

In all nine model–dataset combinations, late-layer active-set overlap is
below early-layer overlap under both random and contiguous dropout (mean
late-minus-early Jaccard $-0.17$ and $-0.20$). Contiguous $\Delta$ Jaccard
is $\approx-0.35$ for DINOv3, $\approx-0.19$ for CBraMod, and $\approx-0.07$
for REVE. Useful transfer by a vision FM thus need not entail matching
internal robustness. This observation is secondary to the validation
ladder.

## Appendix D Full 21-setting ablation chart

Figure 3 is the per-setting frozen-probe ablation display summarized in
Figure 2.

Figure 3: Frozen-probe balanced accuracy change after zeroing
size-matched latent groups, 21 settings. Seed dependence and the
baseline-activity confound are discussed in Section 3.2.

## Appendix E Setting-level bootstrap and activity matching

Percentile 95% CIs from 10,000 bootstrap resamples of settings.

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

Experiments used NVIDIA A100 GPUs on a shared academic Slurm cluster. The
reported pipeline comprises token-level activation export for 27 settings,
training of 27 TopK SAEs plus 9 multi-seed BCI-middle retrains,
perturbation sweeps, linear probes on 21 settings, and CPU-side analyses,
estimated at on the order of 300 GPU-hours, excluding unused exploratory
exports and trainings. An anonymized code archive with training and
analysis scripts is provided as supplementary material; checkpoints and raw
EEG are omitted for size and are available from the cited public sources.

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