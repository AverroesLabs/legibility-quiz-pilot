## Summary

The paper asks whether cleaning and aggregating noisy labels can improve the consistency of surrogate-loss minimization. It studies repeated labeling, majority voting, ranking aggregation, unrestricted measurable predictors, and restricted model classes. Convex surrogates can fail for single-comparison ranking, while aggregation permits a consistent squared-loss construction. Under weak identifiability and noise conditions, generalized majority voting gives asymptotic consistency and approximately linear risk comparisons. Aggregation also restores consistency in restricted classes that contain an optimal predictor, even when unaggregated surrogate minimization fails. For finite-dimensional multiclass logistic models, small link mis-specification can destroy unaggregated consistency, while majority voting recovers the optimal parameter direction asymptotically. Numerical illustrations examine consistency amplification and mis-specified logistic models. The paper leaves finite-label guarantees, sharper noise dependence, and restricted-class comparison inequalities as open questions.

## Glossary

### Concepts, abbreviations, and codes

- **Aggregation / label cleaning:** Refinement of noisy responses into a replacement response used for surrogate fitting.
- **Aggregated labels:** Labels or other responses produced by an aggregation map.
- **Aggregated surrogate risk / P9S:** \(R_{\varphi,A}(f)=\mathbb E[\varphi(f(X),A(Z))]\).
- **Approximate minimizer / \(\epsilon\)-argmin:** A parameter or function whose objective is at most its infimum plus \(\epsilon\).
- **Asymptotic consistency:** Convergence of task risk to its optimal value.
- **Bayes-optimal classifier:** A classifier minimizing task risk over all measurable functions.
- **Bipartite matching:** The paper's matching problem on two sets of \(N\) vertices, with each of \(N\) edges joining unique vertices on opposite sides.
- **Borel measurable functions:** The unrestricted function class used in the classical surrogate-consistency framework.
- **Calibration function / H8N:** The smallest surrogate excess conditional risk among scores with task excess conditional risk at least \(\epsilon\), with pointwise and uniform forms.
- **Classification-rule error:** The parameter-based error plotted in Figure 2, as specified in its caption.
- **Comparison inequality:** An inequality bounding transformed excess task risk by excess surrogate risk.
- **Cone:** A function class closed under multiplication by every positive scalar.
- **Conditional risk:** Task or surrogate risk evaluated at a score while conditioning on \(X=x\).
- **Convex margin-based binary surrogate:** A surrogate of the form \(\varphi(s,y)=\phi(ys)\), with convex \(\phi\).
- **Decoder:** The map \(\mathsf d\) taking a score vector to a prediction.
- **Efficiency:** The advantage attributed to using all information when statistical models are perfectly specified; no separate formal definition is supplied.
- **Empirical-process theory:** The toolkit cited for analyses based on individual input–response pairs.
- **Error function / V5R:** \(e_m(t)\), the auxiliary quantity in equation (10) used to express likely correctness of aggregation.
- **Excess conditional risk:** Conditional risk minus its infimum over scores.
- **Excess risk / surrogate regret:** Risk minus the corresponding optimal risk.
- **Feasible:** The descriptor used in Lemma 3.1 for a surrogate with the displayed constants.
- **Fenchel biconjugate:** The operation denoted by \(^{**}\), used to construct comparison functions.
- **Finite-dimensional classifier:** A classifier indexed by a finite-dimensional parameter such as \(\theta\) or \(\Theta\).
- **Fisher consistency / Q7D:** The classical surrogate-consistency notion for unrestricted Borel measurable functions, characterized through positive calibration functions.
- **Generalized hinge loss / structured hinge loss:** Other names for the maximum-margin surrogate.
- **Generalized majority vote / J9P:** Empirical task-loss minimization followed by substitution of a class-specific parameter \(a_y\).
- **Gold standard labels:** The crowdsourced labels discussed in the cited label-collection literature.
- **\(H\)-consistency / \(\mathcal H\)-consistency:** Consistency comparisons within a restricted hypothesis class.
- **Hypothesis class:** The set of predictors under consideration, denoted by \(\mathcal F\), \(H\), or \(\mathcal H\).
- **Identifiability gap:** The structured-prediction quantity \(\tau(y)\).
- **Identifying surrogate / W3C:** A surrogate satisfying inequalities (6a) and (6b) for class parameters and score vectors.
- **i.i.d.:** Independent and identically distributed, including conditionally repeated labels where stated.
- **Instantaneous task loss:** A loss \(\ell(\widehat y,Y)\) evaluated against an individual response.
- **\(K\)-nearest-neighbor aggregation:** Empirical task minimization using the response at \(X\) and responses at its \(K\) nearest sampled inputs.
- **Label-space cardinality:** The number of possible responses, written \(\operatorname{card}(\mathcal Y)\) or \(|\mathcal Y|\).
- **Level-\((\xi,\zeta)\) consistency / U6A:** A comparison inequality whose comparison function is at least \(\zeta\epsilon\) for \(\epsilon\geq\xi\).
- **Linear R2V:** A uniform comparison inequality with linear comparison function.
- **Link:** The map \(\sigma\) from linear scores to class probabilities.
- **Logistic loss:** The multiclass fitting loss in equation (15).
- **Majority vote / T4K:** Empirical minimization of zero-one loss over repeated labels.
- **Mammen–Tsybakov noise conditions / N6Y:** Conditions (Nα) and (Mβ), controlling disagreement or small conditional-risk gaps.
- **Maximum-margin surrogate / A3J:** The structured loss in equation (11).
- **Minimal excess conditional risk / F8Q:** \(\Delta(x)\), the smallest conditional excess risk of an incorrect prediction.
- **Mis-specification:** A mismatch involving the fitted loss, link, or model, in the settings studied.
- **Mistaken-edge loss:** The normalized difference between the edge embeddings of two matchings.
- **Multilabeled observations:** Inputs accompanied by several responses.
- **Nonparametric regime:** The regime using unrestricted measurable predictors.
- **Noise condition number / B2Z:** \(\kappa(x)\), the ratio in equation (9).
- **Noise exponent:** \(\alpha\) in (Nα), with the corresponding exponent \(\beta\) in (Mβ).
- **Optimal predictability / K7M:** Existence in the hypothesis class of a predictor agreeing with an optimal conditional decision.
- **Ordering consistency:** The equivalence between maximizing scores and maximizing class probabilities in equation (14).
- **Pair-based / unaggregated procedure:** A procedure using individual \((X,Y)\) responses rather than aggregated responses.
- **Pairwise-comparison ranking:** Ranking from outcomes \((i,j)\), where item \(i\) outranks item \(j\).
- **Parametric regime:** The regime using a restricted, potentially mis-specified function class.
- **Perron vectors:** Alternative ranking quantities mentioned alongside powers of the comparison matrix.
- **Population aggregation:** Transformation of conditional comparison probabilities into an ordering.
- **Repeated labeling:** Independent responses drawn conditionally on the same input.
- **Restricted-class consistency:** Consistency when predictors range over a specified subset of measurable functions.
- **Robustness:** The consistency gains under noisy responses or mis-specification established by the paper's aggregation results.
- **Score:** A real-valued or vector-valued predictor output decoded into a response.
- **Softmax:** The logistic link \(\sigma^{\rm lr}\) used in the paper.
- **Stochastic comparison matrix:** The nonnegative matrix \(C_x\) satisfying \(C_x^T\mathbf1=\mathbf1\).
- **Structured embedding:** A map \(v:\mathcal Y\to\mathbb R^d\) representing response structure.
- **Surrogate loss:** The easier loss \(\varphi\) used in place of the task loss.
- **Surrogate-risk consistency:** Convergence of surrogate risk to its optimum implying convergence of task risk to its optimum.
- **Task loss / task risk:** The criterion \(\ell\) and its expected value \(R\), which define the prediction objective.
- **Truncated quadratic loss / D4X:** The binary margin loss \(\phi(\delta)=\max\{1-\delta,0\}^2\).
- **Uniform comparison inequality / R2V:** A nondecreasing positive comparison function bounding transformed excess task risk by excess surrogate risk.
- **Well-specification:** The setting in which the statistical model is perfectly specified.
- **Zero-one loss:** The loss indicating an incorrect prediction.

### Symbols and notation

- **\(X,x,\mathcal X\):** Random input, input value, and input space.
- **\(Y,y,\widehat y,\mathcal Y\):** Random response, response value, predicted response, and response space.
- **\((X_i,Y_i)_{i=1}^n\):** A sample of \(n\) input–response pairs.
- **\(n\):** Number of sampled observations.
- **\(m\), \(m(n)\):** Repeated-label or aggregation count, possibly depending on \(n\).
- **\(Y_1,\ldots,Y_m\):** Repeated responses conditionally independent given an input.
- **\(P\), \(\mathbb P\):** The data distribution and probabilities under the indicated distribution.
- **\(P(\cdot\mid X)\), \(P_{Y\mid X=x}\):** Conditional response distribution given the input.
- **\(\mathbb E\), \(\mathbb E_P\):** Expectation under the indicated distribution.
- **\(\mathbb R,\mathbb R_+\):** Real and nonnegative-real spaces used for scores, losses, and constants.
- **\(d\):** Score dimension, or input dimension where specified.
- **\(k\), \([k]\):** Number of classes, or ranking length where specified, and the index set \(\{1,\ldots,k\}\).
- **\(f,\mathcal F\):** A score function and its hypothesis class.
- **\(f_n,f_m\):** Sequences of predictors indexed by sample size or aggregation count.
- **\(f^\star\):** An optimal predictor used in the noise condition.
- **\(\mathsf d,\widehat y=\mathsf d\circ f(x)\):** Decoder and decoded prediction.
- **\(\circ\):** Composition used to apply a decoder to a score.
- **\(\mathsf{sgn}\):** The sign decoder used for binary classification.
- **\(s,s'\):** Score vectors.
- **\(\ell\):** Task loss, evaluated against an individual response or a conditional distribution.
- **\(R(f),R(s\mid x)\):** Task risk and conditional task risk.
- **\(R^\star\):** Infimum of task risk over the stated class.
- **\(\varphi\):** Surrogate loss.
- **\(\phi\):** Binary margin loss, or the loss notation used in Proposition 4.
- **\(R_\varphi(f),R_\varphi(s\mid x)\):** Surrogate risk and conditional surrogate risk.
- **\(R_\varphi^\star\):** Infimum of surrogate risk over the stated class.
- **\(Z,\mathcal Z\):** Replacement response and its space; \(Z\) also denotes a Gaussian vector in Proposition 5.
- **\(\mathcal A,A\):** Aggregation space and aggregation map.
- **\(A_m,A_{m,K}\):** Aggregators indexed by repeated-label count, or by count and neighbor count.
- **\(R_{\varphi,A},R_{\varphi,A}^\star\):** Aggregated surrogate risk and its infimum.
- **\(\operatorname{dist}\):** The input-space metric used for nearest-neighbor aggregation.
- **\(K\):** Number of sampled neighbors included with the response at \(X\).
- **\(X_{(l)},Y_{(l)}\):** Inputs ordered by distance from \(X\), and their associated responses.
- **\(\delta_\ell,\delta_\varphi,\delta_{\varphi,A}\):** Conditional excess task, surrogate, and aggregated-surrogate risks.
- **\(\epsilon,\varepsilon\):** Positive risk level, approximation tolerance, or perturbation size, as specified; the paper writes these using \(\epsilon\).
- **\(\psi(\epsilon,x),\overline\psi(\epsilon)\):** Pointwise and uniform calibration functions.
- **\(\psi\):** A comparison function, including the stated Fenchel biconjugate.
- **\(\overline\psi_A(\epsilon,x),\overline\psi_A(\epsilon),\psi_A\):** Aggregated calibration quantities in equation (5) and their biconjugate construction.
- **\(^{**}\):** Fenchel biconjugation.
- **\(i,j,l\):** Indices for observations, classes, comparisons, or edges, as specified.
- **\(p_{ij}\):** Conditional probability of the ranking response \((i,j)\).
- **\(C_x,C\):** Conditional ranking-comparison matrix and a generic such matrix.
- **\(\mathbf1,\mathbf0\):** All-ones and zero vectors of the required dimension.
- **\(e_i\):** Coordinate vector used to compare ranking items \(i\) and \(j\).
- **\(p\) in \(C_x^p\):** Power of the comparison matrix.
- **\(q\):** Vector-valued ranking-aggregation response.
- **\(\star\):** Aggregation output used when some comparison column has no observations.
- **\(m_{ij},m_j\):** Comparison-count quantities defined in the ranking aggregation construction.
- **\(b(x)\):** Integrable upper bound on conditional task excess risk.
- **\(a_y,s_y\):** Class-specific aggregation parameter and score vector for identifying surrogates.
- **\(y^\star(x)\):** Set of conditional task-loss minimizers, or unique maximizing class where specified.
- **\(M\):** A finite random stabilization count, a positive threshold, or a matrix argument, depending on context.
- **\(\mathsf C_{\varphi,1},\mathsf C_{\varphi,2}\):** Positive constants in the identifying-surrogate inequalities.
- **\(\xi,\zeta\):** Risk threshold and linear-comparison slope.
- **\(\xi_m,\xi_{m,k}\):** Aggregation-dependent risk thresholds.
- **\(\Delta(x)\):** Minimal excess conditional risk of an incorrect prediction.
- **\(\kappa(x)\):** Noise condition number.
- **\(\mathsf C_{\mathsf{MT}}\):** Positive constant in the Mammen–Tsybakov noise conditions, allowed to change.
- **\(\alpha,\beta\):** Noise exponents; \(\alpha\) also denotes the score-shrinkage factor in the numerical link.
- **(Nα), (Mβ):** The two displayed noise conditions.
- **\(e_m(t)\):** Error function at aggregation count \(m\) and condition-number argument \(t\).
- **\(\delta\):** Positive margin parameter, or the argument of the truncated quadratic loss.
- **\(\partial\phi(0),\phi'(0)\):** Subdifferential and derivative at zero used in binary-surrogate assumptions.
- **\(N\):** Number of vertices on each side of the matching problem.
- **\(G=(V,E),V_1,V_2\):** Matching graph, its vertices and edges, and its two vertex sets.
- **\(u,v\) as vertex indices:** Endpoints used in the edge-index embedding.
- **\(v(y),v(G)\):** Structured or matching embedding.
- **\(\mathcal S(y)\):** Scores strictly decoding to class \(y\) in the structured construction.
- **\(\tau(y)\):** Structured identifiability gap.
- **\(\hat y,y',y_+,y_-\):** Alternative response indices used in losses and comparisons.
- **\(\theta,\theta^\star,\theta_\varphi,\theta_m\):** Model parameter, optimal direction, surrogate minimizer, and aggregated approximate minimizer.
- **\(f_\theta\):** The linear predictor \(x\mapsto\langle\theta,x\rangle\).
- **\(\epsilon_m\):** Approximation tolerance tending to zero with aggregation count.
- **\(\angle(\theta,\theta^\star),\cos\angle(\theta,\theta^\star)\):** Angle and normalized inner-product comparison of parameter directions.
- **\(H,\mathcal H\):** Restricted hypothesis classes in the cited consistency work.
- **\(g\):** An objective in the approximate-minimizer definition, or a measurable binary score function.
- **\(\Theta,\Theta^\star,\Theta_m\):** Multiclass parameter matrix, true matrix, and fitted population minimizer.
- **\(\theta_i,\theta_k=\mathbf0\):** Class parameter columns and the zero reference-class parameter.
- **\(\Sigma,I,I_d,I_{k-1}\):** Covariance and identity matrices used in Gaussian settings.
- **\(\mathsf N(0,I)\):** The centered Gaussian distribution with the indicated identity covariance.
- **\(\mathsf{Cat}(p)\):** Categorical response distribution with probability vector \(p\).
- **\(p_\Theta,p_{\Theta,m}\):** Conditional class-probability vectors before and after aggregation.
- **\(\sigma,\sigma_k\):** Link to the first \(k-1\) probabilities and the remaining class probability.
- **\(t,t_i,t_k=0\):** Link-score vector, its coordinates, and the reference score.
- **\(\sigma^{\rm lr}\):** Multiclass logistic link.
- **\(\rho_m\):** Majority-vote distribution from \(m\) responses.
- **\(Y_i^+,Y_m^+\):** Majority-vote responses in the data and population-loss notation.
- **\(L_m,L_{m,\epsilon},L_{m,\sigma}\):** Population logistic objectives under the stated aggregation and link.
- **\(\nabla,\nabla_\Theta,\nabla_{\theta_i}\):** Derivatives used in the stated first-order conditions.
- **\(DM\):** The linear map in Proposition 5 whose invertibility is assumed.
- **\(\epsilon_0,\mathcal T_\epsilon\):** Perturbation bound and link-perturbation set.
- **\(\sigma^\epsilon\):** The perturbed logistic link in Section 4.3.
- **\(U^\star,T^\star,T_m\):** Orthogonal factor, nonsingular factor, and fitted factor in Theorem 3.
- **\(\widetilde\sigma,r\):** Piecewise corrupted link and its radius in the numerical experiment.
- **\(\psi_m\):** Aggregated comparison curve used in Figure 1.
- **\(T=100\):** Number of numerical trials for each aggregation count.
- **\(1\{\cdot\}\):** Indicator used in the paper's losses, counts, embeddings, and piecewise links.
- **\(\operatorname{argmin},\operatorname{argmax},\inf,\max,\min\):** Optimizer sets and extrema over the stated arguments.
- **\(\operatorname{card}(\mathcal Y),|\mathcal Y|\):** Number of responses in \(\mathcal Y\).
- **\(\langle\cdot,\cdot\rangle,{}^T,{}^\top\):** Inner product and transpose notation.
- **\(\|\cdot\|,\|\cdot\|_1,\|\cdot\|_2\):** Norms used for parameter directions, embeddings, and distances.
- **\((\cdot)_+\):** Positive-part notation in the numerical comparison curve.
- **\(\log,\exp,e^t\):** Logarithm and exponential notation used in losses and bounds.
- **\(O(\cdot),o_m(1)\):** Order notation for aggregation requirements and quantities tending to zero with \(m\).
- **\(N!\):** The matching-label cardinality.
- **\(\to,\uparrow,\Longrightarrow,\Longleftrightarrow\):** Convergence, increasing limit, implication, and equivalence notation.

# Some Robustness Properties of Label Cleaning

Chen Cheng

John Duchi

###### Abstract

*Aggregation strengthens surrogate consistency and protects consistency against some loss mis-specification.*

Aggregated labels—responses refined from noisy signals—permit robustness unavailable without cleaning. Minimizing an easier, typically convex surrogate loss gives stronger guarantees for task-risk consistency, meaning convergence to optimal task risk, than procedures using raw labels. Finite-dimensional classifiers also remain consistent under loss mis-specification that defeats standard uncertainty-modeling approaches. Perfectly specified statistical models favor using all information for efficiency. Yet even slight loss mis-specification destroys standard consistency. Aggregation retains convergence to optimal classifiers. Collection, fitting, and prediction can therefore jointly gain robustness through noisy-signal refinement.

## 1 Introduction

*The paper asks when aggregation is necessary for consistent prediction and how it strengthens consistency.*

Aggregation permits consistent optimal prediction through surrogate loss rather than task loss when unaggregated estimation cannot. This includes finite-dimensional statistical problems. Linear-estimator constructions for ranking and binary and multiclass classification separate procedures using only \((X,Y)\), an input–response pair, from aggregation procedures. Nonparametric results combine aggregation counts and noise conditions. They permit approximately linear comparison inequalities—bounds relating excess task risk to excess surrogate risk—under weak surrogate conditions. They also extend consistency beyond unrestricted measurable hypothesis classes, the sets of predictors being considered.

The conventional supervised pipeline collects \((X_i,Y_i)_{i=1}^n\), a sample of \(n\) input–response pairs, and fits a model [17]. This description suppresses the cleaning, filtering, repeated crowdsourced labeling, and denoising required by modern datasets [9, 22, 39]. Work on crowdsourced “gold standard” labels [8, 48, 47, 46, 37] studies this machinery. Here the concern is its fundamental, necessary improvement of fitted-model consistency.

Important contributions to surrogate-risk consistency theory date to the 2000s [52, 24, 40]. Bartlett, Jordan and McAuliffe [4] characterize convex-surrogate consistency for binary zero-one classification, whose loss indicates an incorrect prediction. Later settings include multi-label classification [50, 43, 16, 49, 1], ranking [13, 14, 36], structured prediction [34, 6, 32], ordinal regression [35], restricted hypothesis classes [2, 3, 28, 29], and general theory [40]. Pair-based analyses admit empirical-process theory [45, 5, 21, 4], but not the aggregation machinery of contemporary dataset creation.

The questions are whether aggregation is needed at all, whether pair-based surrogate consistency suffices, and in what sense aggregation improves consistency.

## 2 Preliminaries

*The paper defines task and surrogate risks for unrestricted and potentially mis-specified function classes.*

Two regimes are considered. The first uses an unrestricted class \(\mathcal F\) of Borel measurable functions and the classical notion **Q7D: Fisher consistency**. The second uses a parametric, potentially mis-specified \(\mathcal F\). Aggregation improves consistency in both.

Let \((X_i,Y_i)_{i=1}^n\sim P\) be independent and identically distributed (i.i.d.) on input and output spaces \(\mathcal X\times\mathcal Y\). Here \(P\) is the data distribution. A score function \(f:\mathcal X\to\mathbb R^d\), with score dimension \(d\geq1\), is decoded by \(\mathsf d:\mathbb R^d\to\mathcal Y\). Thus the prediction is \(\widehat y=\mathsf d\circ f(x)\), where \(\circ\) denotes composition.

The distribution-dependent task risk is

\[
R(f)\coloneqq
\mathbb E_P\!\left[\ell(\mathsf d\circ f(X),P(\cdot\mid X))\right].
\tag{2}
\]

Here \(\mathbb E_P\) denotes expectation, \(\ell\) is the task loss, and \(P(\cdot\mid X)\) is the conditional response distribution. Its minimizers frequently coincide with the original problem (2). Dependence on the conditional distribution allows greater sophistication. For \(\mathcal Y=\{1,\ldots,k\}\), where \(k\) is the number of classes, and
\[
\ell(\widehat y,P)=\sum_y P(Y=y)1\{\widehat y\neq y\},
\]
the criterion is the standard zero-one error rate. The notation \(1\{\cdot\}\) indicates whether its condition holds.

The instantaneous formulation evaluates a loss against one response. For \(\ell:\mathcal Y\times\mathcal Y\to\mathbb R_+\), where \(\mathbb R_+\) denotes nonnegative reals, it minimizes over \(f\in\mathcal F\)
\[
R(f):=\mathbb E_P[\ell(\mathsf d\circ f(X),Y)].
\tag{1}
\]
Binary classification uses \(d=1\), the sign decoder \(\mathsf d(s)=\mathsf{sgn}(s)\), and
\[
\ell(y,y')=1\{yy'\leq0\},\qquad
R(f)=\mathbb P(Yf(X)\leq0).
\]
Here \(s\) is a score and \(\mathbb P\) denotes probability.

Task losses may be nonsmooth, nonconvex, and uninformative. The zero-one landscape is flat almost everywhere, precluding even practical first-order optimization. An easier surrogate loss \(\varphi:\mathbb R^d\times\mathcal Y\to\mathbb R\) replaces it:
\[
R_\varphi(f):=\mathbb E_P[\varphi(f(X),Y)].
\]
This surrogate risk must be consistent with the task criterion.

### 2.1 Label aggregation

*Aggregation replaces an individual response by a response constructed from repeated labels or nearby observations.*

The quantity **P9S: aggregated surrogate risk** is
\[
R_{\varphi,A}(f):=\mathbb E[\varphi(f(X),A(Z))].
\tag{3}
\]
Its minimization is assessed against actual task-risk minimization. Here \(Z\in\mathcal Z\) is a replacement response, \(\mathcal A\) is the aggregation space, \(A:\mathcal Z\to\mathcal A\) is the aggregation map, and \(\varphi:\mathbb R^d\times\mathcal A\to\mathbb R_+\) is a nonnegative surrogate. The response is no longer an ordinary \((X_i,Y_i)\) response.

Repeated labeling uses
\[
Z=(Y_1,\ldots,Y_m)\in\mathcal Y^m,\qquad
Y_i\mid X=x\stackrel{\rm iid}{\sim}P_{Y\mid X=x},
\]
where \(m\) is the repetition count. Empirical task minimization gives
\[
A(Z)=A(\{Y_1,\ldots,Y_m\})
=\mathop{\rm argmin}_{y\in\mathcal Y}
\sum_{l=1}^m\ell(y,Y_l).
\]
The notation \(\operatorname{argmin}\) denotes the minimizing response or responses. For \(\ell(\widehat y,y)=1\{y\neq\widehat y\}\), this is **T4K: majority vote**. General \(\ell\) permits abstract procedures. The notation \(A_m(Z)\) records the repetition count. Mao et al. [28] use a similar multilabeled structure for multilabel losses rather than for the aggregation pipeline.

A rough \(K\)-nearest-neighbor construction uses the response \(Y\) at \(X\) and an additional independent sample
\[
(X_i,Y_i)_{i=1}^m\stackrel{\rm iid}{\sim}P,
\qquad Z=(Y,(X_i,Y_i)_{i=1}^m).
\]
For an input-space metric \(\operatorname{dist}:\mathcal X\times\mathcal X\to\mathbb R_+\), order the sampled inputs by distance:
\[
\operatorname{dist}(X,X_{(1)})\leq\cdots\leq
\operatorname{dist}(X,X_{(m)}),\qquad X_{(0)}=X.
\]
Here \(Y_{(l)}\) is the response associated with \(X_{(l)}\). For neighbor count \(K\geq0\), take
\[
A_{m,K}(Z)\coloneqq
\mathop{\rm argmin}_{y\in\mathcal Y}
\sum_{l=0}^K\ell(y,Y_{(l)}).
\]
The population-level repeated-sampling treatment also extends to aggregation from one sample \((X_i,Y_i)_{i=1}^n\) (Appendix C).

## 3 Surrogate consistency

*Calibration functions characterize classical consistency, while later results show failures without aggregation and stronger guarantees with it.*

A calibration function measures the smallest surrogate excess conditional risk compatible with a specified task excess conditional risk. Below, \(\epsilon>0\) is that specified risk level, \(\psi(\epsilon,x)\) is pointwise in \(x\), and \(\overline\psi\) is its uniform form. The optimal risks are \(R^\star\) and \(R_\varphi^\star\).

###### Corollary 3.1.

Positivity of \(\psi(\epsilon,x)\) for every \(x\in\mathcal X\) and \(\epsilon>0\) is equivalent to \(\varphi\) being Q7D for \(\ell\). If \(\psi\) is the Fenchel biconjugate of \(\overline\psi\), then
\[
\overline\psi(\epsilon)>0\quad\Longleftrightarrow\quad
\psi(\epsilon)>0,
\qquad
\psi(R(f)-R^\star)\leq R_\varphi(f)-R_\varphi^\star
\]
for every measurable \(f\).

The standard framework [40] lets \(\mathcal F\) contain every Borel measurable \(f:\mathcal X\to\mathbb R^d\). For all distributions \(P\), it considers two requirements [40, 4, 52]:

(i) Every sequence \(f_n\in\mathcal F\) with
\[
R_\varphi(f_n)\to R_\varphi^\star
\]
also satisfies \(R(f_n)\to R^\star\). Thus surrogate-risk convergence implies task-risk convergence.

(ii) A nondecreasing \(\psi:\mathbb R_+\to\mathbb R_+\), positive at every \(\epsilon>0\), satisfies
\[
\psi(R(f)-R^\star)\leq R_\varphi(f)-R_\varphi^\star
\quad\text{for every }f\in\mathcal F.
\]

Requirement (ii), **R2V: uniform comparison inequality**, is stronger and need not hold. It coincides with (i) for convex margin-based binary surrogates [4], namely surrogates depending convexly on the signed score.

For score \(s\in\mathbb R^d\), conditional risks and their excesses are
\[
\begin{aligned}
R(s\mid x)&\coloneqq
\ell(\mathsf d\circ s,P(Y\in\cdot\mid X=x)),\\
R_\varphi(s\mid x)&:=
\mathbb E[\varphi(s,Y)\mid X=x],\\
\delta_\ell(s,x)&\coloneqq
R(s\mid x)-\inf_{s'\in\mathbb R^d}R(s'\mid x),\\
\delta_\varphi(s,x)&\coloneqq
R_\varphi(s\mid x)-\inf_{s'\in\mathbb R^d}R_\varphi(s'\mid x),\\
R^\star&:=\inf_{f\in\mathcal F}R(f),\qquad
R_\varphi^\star:=\inf_{f\in\mathcal F}R_\varphi(f).
\end{aligned}
\]
Thus \(\delta_\ell\) and \(\delta_\varphi\) subtract the best conditional risk over scores; \(R^\star\) and \(R_\varphi^\star\) are risk infima over \(\mathcal F\).

The canonical excess-risk object is **H8N: calibration function**. Its pointwise and uniform forms are
\[
\psi(\epsilon,x)\coloneqq
\inf_{s\in\mathbb R^d}
\{\delta_\varphi(s,x)\mid\delta_\ell(s,x)\geq\epsilon\},
\qquad
\overline\psi(\epsilon)\coloneqq
\inf_{x\in\mathcal X}\psi(\epsilon,x).
\]
The equivalence and inequality in Corollary 3.1 follow from Zhang [51, Prop. 25] and Steinwart [40, Thm. 2.8 and Lemma 2.9].

A Q7D surrogate for a general task would make single pairs sufficient for consistent estimation. It would simplify collection and admit the full empirical-process toolkit. Such pair-only results are generally impossible.

### 3.1 Q7D failure without aggregation: ranking

*Convex losses fail Fisher consistency for single-comparison ranking, while aggregated ordering frequencies permit a consistent squared loss.*

###### Proposition 1.

For ranking with \(k\geq3\) outcomes, every \(\varphi:\mathbb R^k\times\mathcal Y\to\mathbb R\) convex in its first argument fails Q7D (Appendix A.1).

###### Proposition 2.

For score \(s\) and vector-valued response \(q\) in \(\mathbb R^k\), set
\[
\varphi(s,q)=\|s-q\|_2^2,\qquad
\varphi(s,\star)=0.
\]
Here \(\|\cdot\|_2\) is the norm used for squared-error fitting, and \(\star\) is the aggregation output specified below. With that aggregation and \(m\geq k\), \(\varphi\) is Q7D for ranking risk (proof: Appendix A.2).

The response space expands from \(\mathcal Y\) to \(\mathbb R^k\). This expansion is unavailable to the convex single-comparison formulation. Regression on ordering frequencies yields the positive result.

Pairwise-comparison ranking [20, 15, 14, 30] uses
\[
\mathcal Y=\{(i,j):i\neq j,\ 1\leq i,j\leq k\},
\]
where \((i,j)\) means item \(i\) outranks item \(j\). For each \(x\), population aggregation transforms
\[
p_{ij}=P(Y=(i,j)\mid x)
\]
into an ordering. A simple comparison scheme [20, cf.] defines
\[
(C_x)_{ii}=0,\qquad
(C_x)_{ij}=\frac{p_{ij}}{\sum_{l\neq j}p_{lj}}
\quad(i\neq j),\qquad
0/0=1/(k-1).
\]
Thus \(C_x\in\mathbb R_+^{k\times k}\) is a stochastic comparison matrix:
\[
C_x^T\mathbf1=\mathbf1.
\]
Here \(T\) denotes transpose and \(\mathbf1\) is the all-ones vector. The vector \(C_x\mathbf1\) measures each item's preference frequency. Higher matrix powers \(C_x^p\mathbf1\), Perron vectors [20], and more sophisticated schemes admit similar negative results.

Absorbing decoding into the loss gives
\[
\ell(s,C)\coloneqq
\max_{i<j}1\!\left\{
(s_i-s_j)(e_i-e_j)^TC\mathbf1\leq0,\ 
(e_i-e_j)^TC\mathbf1\neq0
\right\}.
\]
Here \(C\) is a comparison matrix and \(e_i\) is the coordinate vector for item \(i\). The loss penalizes disagreement between score and matrix orderings:
\[
R(f)\coloneqq
\mathbb P\!\left(f(X)\text{ and }C_X\mathbf1
\text{ order differently}\right).
\tag{4}
\]
Restricting scores to \(s^T\mathbf1=0\) is minor, familiar in multiclass classification [51, 43], and natural when only ordering is decoded.

For multiple independent comparisons \(Z=(Y_1,\ldots,Y_m)\) conditional on \(X\), define the comparison-count quantities
\[
m_{ij}=\sum_{y\in\mathcal Y^m}1\{y=(i,j)\},
\qquad
m_j=\sum_{i=1}^k m_{ij}.
\]
With \([k]=\{1,\ldots,k\}\), use
\[
A(Z)=
\begin{cases}
\star,&m_j=0\text{ for some }j\in[k],\\[2mm]
\left(
\frac{m_{i1}}{m_1}+\frac{m_{i2}}{m_2}+\cdots+
\frac{m_{ik}}{m_k}
\right)_{i\in[k]},
&m_j>0\text{ for all }j\in[k].
\end{cases}
\]
Regression acts directly on non-\(\star\) outputs.

**Claims and evidence**

- **Convex single-comparison surrogates fail Q7D:** Proposition 1 gives the condition \(k\geq3\) and applies to every surrogate convex in its first argument.
- **Aggregation permits Q7D squared loss:** Proposition 2 uses \(\|s-q\|_2^2\), loss \(0\) at \(\star\), and \(m\geq k\); the aggregation rule above supplies vector-valued ordering frequencies.
- **The task measures ordering disagreement:** Equation (4) gives its exact probability criterion. No numerical table is reported.

### 3.2 Aggregation and stronger surrogate consistency

*Aggregation extends the classical framework and can make nominally inconsistent losses consistent.*

Aggregation both extends ordinary surrogate-risk theory and upgrades nominally inconsistent losses.

#### 3.2.1 Basic extensions

*Aggregated conditional risks give a calibration characterization and comparison inequality under an integrable task-risk bound.*

###### Proposition 3.

Suppose an integrable upper bound \(b:\mathcal X\to\mathbb R_+\) satisfies
\[
\int b(x)\,dP(x)<\infty,\qquad
\delta_\ell(f(x),x)\leq b(x).
\]
Then \(\varphi\) is Q7D for the task risk exactly when
\[
\overline\psi_A(\epsilon,x)>0
\quad\text{for every }x\in\mathcal X,\ \epsilon>0.
\]
With \(\psi_A=(\overline\psi_A)^{**}\), where \(^{**}\) denotes Fenchel biconjugation,
\[
\psi_A(R(f)-R^\star)
\leq R_{\varphi,A}(f)-R_{\varphi,A}^\star.
\]
Here \(R_{\varphi,A}^\star\) is the infimum of aggregated surrogate risk.

For arbitrary \(A:\mathcal Z\to\mathcal A\), the aggregated conditional risk and excess risk are
\[
\begin{aligned}
R_{\varphi,A}(s\mid x)&\coloneqq
\mathbb E[\varphi(s,A(Z))\mid X=x],\\
\delta_{\varphi,A}(s,x)&:=
R_{\varphi,A}(s\mid x)-
\inf_{s\in\mathbb R^d}R_{\varphi,A}(s\mid x).
\end{aligned}
\]
The calibration quantities are
\[
\overline\psi_A(\epsilon,x)\coloneqq
\inf_{s\in\mathbb R^d}
\{\delta_{\varphi,A}(s,x)\mid\delta_\ell(s,x)\geq\epsilon\},
\qquad
\overline\psi_A(\epsilon)\coloneqq
\inf_{s\in\mathbb R^d}\overline\psi_A(\epsilon,x).
\tag{5}
\]
Appropriate measurability conditions are tacit and noncentral. This near-corollary of Steinwart [40, Thm. 2.8] preserves classical guarantees without supplying a particular improvement from aggregation. Appendix B.1 gives a proof for completeness.

**Claims and evidence**

- **Positive aggregated calibration characterizes Q7D:** Proposition 3 requires positivity for every \(x\) and every \(\epsilon>0\), under the displayed integrable bound \(b\).
- **A biconjugate gives a risk comparison:** Proposition 3 states the exact inequality with \(\psi_A=(\overline\psi_A)^{**}\); equation (5) specifies the calibration quantities. No numerical table is reported.

#### 3.2.2 W3C and consistency

*An identifying surrogate becomes asymptotically consistent under generalized majority voting as the repetition count increases.*

Here **W3C: Identifying surrogate** means the class-identification property in Definition 3.1.

###### Corollary 3.2.

For \(m=m(n)\to\infty\) and W3C \(\varphi\),
\[
R_{\varphi,A_m}(f_n)-R_{\varphi,A_m}^\star\to0
\quad\Longrightarrow\quad R(f_n)-R^\star\to0.
\]

The regime has finite label-space cardinality
\[
\operatorname{card}(\mathcal Y)=k<\infty
\]
and instantaneous task loss
\[
R(f)=\mathbb E[\ell(f(X),Y)].
\]
The ranking construction belongs to this regime. There, \(k\) is ranking length and \(\operatorname{card}(\mathcal Y)=k(k-1)\). Henceforth \(k\) means label-space cardinality, with this slight notation abuse.

For class-specific parameters \(\{a_y\}_{y\in\mathcal Y}\), empirical task minimization followed by parameter substitution defines **J9P: generalized majority vote**. With arbitrary tie-breaking,
\[
A_m(y_1,\ldots,y_m)\coloneqq a_{\hat y},
\qquad
\hat y=\mathop{\rm argmin}_{y\in\mathcal Y}
\sum_{i=1}^m\ell(y,y_i).
\tag{7}
\]
For i.i.d. \(Y_i\) and finite \(\mathcal Y\), a random stabilization count \(M<\infty\) necessarily exists such that every \(m\geq M\) satisfies
\[
\mathop{\rm argmin}_{y\in\mathcal Y}
\left\{\sum_{i=1}^m\ell(y,Y_i)\right\}
\subset
y^\star(x)\coloneqq
\mathop{\rm argmin}_{y\in\mathcal Y}
\mathbb E[\ell(y,Y)\mid X=x].
\]
Thus \(y^\star(x)\) is the set of conditional task-loss minimizers. This strategy supplies asymptotic consistency under minimal identifiability for any such task loss.

###### Definition 3.1.

A surrogate \(\varphi:\mathbb R^d\times\mathcal A\to\mathbb R\) has W3C, with constants
\[
0<\mathsf C_{\varphi,1}\leq\mathsf C_{\varphi,2}<\infty,
\]
when parameters \(\{a_y\}_{y\in\mathcal Y}\subset\mathcal A\) and score vectors \(\{s_y\}_{y\in\mathcal Y}\subset\mathbb R^d\) exist with \(\mathsf d(s_y)=y\) and, for all \(y\neq y'\),
\[
\varphi(s_y,a_y)+\mathsf C_{\varphi,1}
\leq\inf_{\mathsf d\circ s\neq y}\varphi(s,a_y),
\tag{6a}
\]
\[
\varphi(s_y,a_{y'})-\mathsf C_{\varphi,2}
\leq\inf_{s\in\mathbb R^d}\varphi(s,a_{y'}).
\tag{6b}
\]
The first inequality makes a minimizer at \(a_y\) identify \(y\). A finite lower bound on \(\varphi(\cdot,a)\) supplies finite \(\mathsf C_{\varphi,2}\) for the second inequality. W3C imposes neither convexity, unaggregated consistency with \(\mathcal A=\mathcal Y\) and \(Z=Y\), nor a decoder requirement such as pointwise \(\operatorname{argmax}\), which denotes a maximizing response.

Binary hinge loss,
\[
\varphi(s,a)=\max\{1-sa,0\},
\qquad \mathcal A=\mathcal Y=\{\pm1\},
\]
uses \(a_y=s_y=y\). It satisfies
\[
\varphi(s_1,a_1)=\varphi(s_{-1},a_{-1})=0,\qquad
\inf_{sa\leq0}\varphi(s,a)=1,
\]
\[
\varphi(s_1,a_{-1})=\varphi(s_{-1},a_1)=2.
\]
Its W3C constants are \((1,2)\).

**Claims and evidence**

- **Growing aggregation gives consistency for W3C losses:** Corollary 3.2 assumes \(m(n)\to\infty\) and states the displayed implication from vanishing aggregated-surrogate excess risk to vanishing task excess risk.
- **Generalized majority vote eventually chooses task-optimal classes:** The stabilization statement gives a random \(M<\infty\) and the optimizer-set inclusion for every \(m\geq M\).
- **Binary hinge loss is identifying:** Its displayed losses are \(0\), \(1\), and \(2\), giving W3C constants \((1,2)\). No numerical table is reported.

#### 3.2.3 W3C and consistency amplification

*Identifiability and noise conditions yield linear comparisons above an aggregation-dependent threshold.*

The theorem uses the noise condition (Nα), defined below. Its exponent is \(\alpha\), its positive constant is \(\mathsf C_{\mathsf{MT}}\), the noise condition number is \(\kappa(x)\), and the auxiliary error function is \(e_m(t)\). Their full formulas appear in (Nα), (9), and (10).

###### Theorem 1.

Let \(\varphi\) be \((\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})\)-W3C with parameters \(\{a_y\}_{y\in\mathcal Y}\), and let \(A_m\) be J9P. Assume \(0\leq\ell\leq1\) and (Nα), with \(\alpha\in[0,1]\). For any threshold \(M>0\) and \(f\in\mathcal F\) satisfying
\[
R(f)-R^\star\geq
2\mathbb P(\kappa(X)>M)+
(4\mathsf C_{\mathsf{MT}}e_m(M))^{\frac1{1-\alpha}},
\]
one has
\[
R(f)-R^\star\leq
\frac{16}{\mathsf C_{\varphi,1}}
\bigl(R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star\bigr).
\]

The corresponding risk threshold \(\xi\) and linear slope \(\zeta\) are
\[
\xi=2\mathbb P(\kappa(X)>M)+
(4\mathsf C_{\mathsf{MT}}e_m(M))^{\frac1{1-\alpha}},
\qquad
\zeta=\frac{\mathsf C_{\varphi,1}}{16}.
\]
Define the optimized threshold
\[
\xi_m\coloneqq
\inf_M\left\{
2\mathbb P(\kappa(X)>M)+
(4\mathsf C_{\mathsf{MT}}e_m(M))^{\frac1{1-\alpha}}
\right\}.
\]
Then \(\xi_m\to0\): \(\mathbb P(\kappa(X)>M)\to0\) as \(M\uparrow\infty\), and \(e_m(M)\to0\) with \(m\) for fixed \(M\). Taking \(\alpha=0\) yields asymptotic consistency.

The incorrect-prediction gap \(\Delta(x)\) is defined in (8). Using the crude bound \(\kappa(x)\leq1/\Delta(x)\), choosing \(M=1\) for \(\operatorname{card}(\mathcal Y)=k=2\), and optimizing \(M\) otherwise lower bounds the promised \(\xi_{m,k}\) by setting
\[
\xi_{m,k}\coloneqq
\begin{cases}
\left(
\frac{32\mathsf C_{\mathsf{MT}}^2}{m}
\log\!\left(
\frac{8(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})}
{\mathsf C_{\varphi,1}}
\right)
\right)^{\frac1{2(1-\alpha)}},
&k=2,\\[3mm]
4\cdot\left(
\frac{32\mathsf C_{\mathsf{MT}}^4}{m}
\log\!\left(
\frac{4k(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})}
{\mathsf C_{\varphi,1}}
\right)
\right)^{\frac{\alpha}{2(1-\alpha^2)}},
&\text{otherwise}.
\end{cases}
\]
Here \(\log\) is the logarithm used in the bound. Appendix B.4 gives the algebraic substitutions and manipulations.

###### Corollary 3.3.

Under Theorem 1's conditions, every \(f\) with \(R(f)-R^\star\geq\xi_{m,k}\) satisfies
\[
R(f)-R^\star\leq
\frac{16}{\mathsf C_{\varphi,1}}
\bigl(R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star\bigr).
\]

For fixed \(f\), the comparison requires fewer aggregated samples as \(\alpha\) approaches \(1\). At \(\alpha=1\),
\[
m\geq
32\max\{\mathsf C_{\mathsf{MT}}^2,\mathsf C_{\mathsf{MT}}^4\}
\log\!\left(
\frac{4k(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})}
{\mathsf C_{\varphi,1}}
\right)
=O(\log k)
\]
gives \(\xi_{m,k}=0\) and linear R2V. The order notation describes the displayed logarithmic aggregation requirement. Lower-noise problems require less aggregation for stronger consistency.

Low noise permits a comparison written as
\[
\psi(R(f)-R^\star)\leq R_\varphi(f)-R^\star,
\]
with \(\psi\) linear over some positive-\(\epsilon\) range. Linear growth is the strongest possible [34, 32]. Strict comparison requirements may exclude practically convenient inconsistent surrogates [23, 34, 32]. Following Osokin et al. [34], the paper uses a positive-range relaxation.

###### Definition 3.2.

The surrogate and aggregator have **U6A: level-\((\xi,\zeta)\) consistency** when some \(\psi\) satisfies
\[
\psi(\epsilon)\geq\zeta\epsilon\quad(\epsilon\geq\xi),
\qquad
\psi(R(f)-R^\star)\leq
R_{\varphi,A}(f)-R_{\varphi,A}^\star.
\]
Only risks above \(\xi\) require linear growth. This retains a “good enough” criterion. Minimal assumptions permit levels \((o_m(1),\zeta)\), where the first component tends to zero with \(m\), even for a non-Q7D surrogate.

The quantity **F8Q: minimal excess conditional risk** is
\[
\Delta(x)\coloneqq
\min_{\mathsf d(s)\notin y^\star(x)}
\delta_\ell(s,x).
\tag{8}
\]
This incorrect-prediction gap equals
\[
|2P(Y=1\mid X=x)-1|
\]
in binary classification. Estimation is expected to be harder nearer zero.

Adapting classical assumptions [25], also Bartlett et al. [4], the pair **N6Y: Mammen-Tsybakov noise conditions** uses a positive constant \(\mathsf C_{\mathsf{MT}}\) that may change:
\[
\mathbb P(\mathsf d\circ f\neq\mathsf d\circ f^\star)
\leq
\mathsf C_{\mathsf{MT}}(R(f)-R^\star)^\alpha
\quad\text{for all measurable }f,
\tag{Nα}
\]
\[
\mathbb P(\Delta(X)\leq\epsilon)
\leq(\mathsf C_{\mathsf{MT}}\epsilon)^\beta
\quad(\epsilon>0).
\tag{Mβ}
\]
Here \(f^\star\) is the optimal predictor used for the disagreement comparison, \(\alpha\in[0,1]\) is the noise exponent, and \(\beta\in[0,\infty]\). Both conditions hold trivially at \(\alpha=\beta=0\). As in the binary result [4, Thm. 3], they are equivalent under
\[
\beta=\alpha/(1-\alpha)
\]
(Appendix B.2).

Mao et al. [29] use exactly the same noise statistic for multiclass \(H\)-consistency, a comparison within hypothesis class \(H\). They impose surrogate and hypothesis-class structure for a global convex lower bound. With minimal structure, the paper instead uses **B2Z: noise condition number**:
\[
\kappa(x)\coloneqq
\frac{\max_{\mathsf d(s)\neq y^\star(x)}\delta_\ell(s,x)}
{\min_{\mathsf d(s)\neq y^\star(x)}\delta_\ell(s,x)}.
\tag{9}
\]
For \(\mathsf d(s)\neq y^\star(x)\),
\[
\Delta(x)\geq\delta_\ell(s,x)/\kappa(x).
\]
Binary classification has \(\kappa(x)=1\) provided \(\mathbb P(\Delta(X)>0)=1\).

The auxiliary quantity **V5R: error function** is
\[
e_m(t)\coloneqq
t\sqrt{\frac2m
\log\!\left(
\frac{4k(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})}
{\mathsf C_{\varphi,1}}
\right)}.
\tag{10}
\]
At \(\kappa(x)=t\), values much smaller than one express likely correctness of \(A_m\).

Appendix B.3 proves Theorem 1. Its coarse bounds reflect weak identifiability. Tighter \(\ell\)–\(\varphi\) relationships, making J9P correct or sufficiently correct for \(\varphi\), should strengthen the bounds. Near-consistency without aggregation could link smaller initial inconsistency to better \(m\)-dependence. These refinements are not pursued.

**Claims and evidence**

- **Aggregation gives a linear comparison above a threshold:** Theorem 1 gives factor \(16/\mathsf C_{\varphi,1}\), threshold \(2\mathbb P(\kappa(X)>M)+(4\mathsf C_{\mathsf{MT}}e_m(M))^{1/(1-\alpha)}\), and slope \(\mathsf C_{\varphi,1}/16\).
- **The optimized threshold vanishes:** The displayed definition of \(\xi_m\) is followed by \(\mathbb P(\kappa(X)>M)\to0\) and \(e_m(M)\to0\); \(\alpha=0\) yields asymptotic consistency.
- **Explicit thresholds depend on class count and noise:** The \(\xi_{m,k}\) display gives coefficient \(32\), logarithmic factor \(8\) for \(k=2\), and outer factor \(4\) with logarithmic factor \(4k\) otherwise. Corollary 3.3 gives the same comparison factor \(16/\mathsf C_{\varphi,1}\).
- **At \(\alpha=1\), finite aggregation gives zero threshold:** The displayed requirement has coefficient \(32\), is \(O(\log k)\), and gives \(\xi_{m,k}=0\).
- **The two noise conditions are equivalent under the stated exponent relation:** (Nα), (Mβ), and \(\beta=\alpha/(1-\alpha)\); both hold at \(\alpha=\beta=0\).
- **Binary noise condition number is \(1\):** Equation (9) and the stated condition \(\mathbb P(\Delta(X)>0)=1\). No numerical table is reported.

**Claims and evidence for Section 3.2**

- **Aggregation preserves classical calibration guarantees:** Proposition 3 and equation (5).
- **Identifiability suffices for asymptotic consistency:** Corollary 3.2, with \(m(n)\to\infty\).
- **Aggregation strengthens comparisons:** Theorem 1 and Corollary 3.3 give factor \(16/\mathsf C_{\varphi,1}\); at \(\alpha=1\), the displayed \(O(\log k)\) count gives threshold \(0\).

### 3.3 Surrogate consistency with T4K

*Majority voting yields explicit consistency guarantees for binary, matching, and structured-prediction surrogates.*

The instances below assume noise exponent \(\alpha\in[0,1]\). This loses no generality because \(\alpha=0\) always satisfies (Nα). Appendix B.5 proves each instance.

#### 3.3.1 Binary classification, nonsmooth surrogate

*The binary margin-loss assumptions give explicit identifying constants and an aggregation-dependent linear comparison.*

###### Lemma 3.1.

For every positive margin parameter \(\delta>0\), \(\varphi\) is \((\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})\)-feasible with
\[
\mathsf C_{\varphi,1}=\phi(0)-\phi(\delta)>0,
\qquad
\mathsf C_{\varphi,2}=\phi(-\delta).
\]
Consequently, for \(k=2\) and \(f:\mathcal X\to\mathbb R\),
\[
R(f)-R^\star\geq
\left(
\frac{32\mathsf C_{\mathsf{MT}}^2}{m}
\log\!\left(
\frac{8(\phi(-\delta)+\phi(0)-\phi(\delta))}
{\phi(0)-\phi(\delta)}
\right)
\right)^{\frac1{2(1-\alpha)}}
\]
implies
\[
R(f)-R^\star\leq
\frac{16}{\phi(0)-\phi(\delta)}
\bigl(R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star\bigr).
\]

Here
\[
\varphi(f(x),y)=\phi(yf(x)),
\]
where \(\phi\) is convex,
\[
\partial\phi(0)\subset(-\infty,0),\qquad
\lim_{t\to\infty}\phi(t)=0.
\]
The symbol \(\partial\phi(0)\) denotes the subdifferential used in this assumption. For convex binary margin surrogates, Q7D and R2V over all measurable functions hold exactly when the derivative satisfies \(\phi'(0)<0\) [4]. Aggregation can make an otherwise inconsistent \(\phi\) consistent. This is a somewhat trivial thought experiment, not a surrogate one would use.

**Claims and evidence**

- **The margin loss has explicit feasible constants:** Lemma 3.1 gives \(\mathsf C_{\varphi,1}=\phi(0)-\phi(\delta)>0\) and \(\mathsf C_{\varphi,2}=\phi(-\delta)\) for every \(\delta>0\).
- **The aggregated comparison is linear above the displayed threshold:** Lemma 3.1 uses threshold coefficient \(32\), logarithmic factor \(8\), exponent \(1/[2(1-\alpha)]\), and comparison factor \(16/(\phi(0)-\phi(\delta))\).
- **Classical convex-margin consistency has an exact derivative condition:** The cited result [4] requires \(\phi'(0)<0\). No numerical table is reported.

#### 3.3.2 Bipartite matching

*The matching surrogate is identifying despite factorially many labels, and aggregation yields an explicit consistency bound.*

###### Lemma 3.2.

On \(2N\) vertices, where \(N\) is the number on each side, the matching surrogate is W3C with
\[
\mathsf C_{\varphi,1}=\frac1N,\qquad
\mathsf C_{\varphi,2}=2.
\]
Despite factorial label cardinality, aggregation yields consistency once its count exceeds \(O(N\log N)\). For \(k\geq3\), measurable \(f:\mathcal X\to\mathbb R^d\) obeying
\[
R(f)-R^\star\geq
4\cdot\left(
\frac{32\mathsf C_{\mathsf{MT}}^4}{m}
\log(4k(2N+1))
\right)^{\frac{\alpha}{2(1-\alpha^2)}}
\]
satisfy
\[
R(f)-R^\star\leq
16N\bigl(R_{\varphi,A_n}(f)-R_{\varphi,A_n}^\star\bigr).
\]

For structured response spaces, which are typically large, an embedding
\[
v:\mathcal Y\to\mathbb R^d
\]
represents structural information [33]. Use decoder
\[
\mathsf d(s)=\operatorname{argmax}_{y\in\mathcal Y}\langle s,v(y)\rangle,
\]
where \(\langle\cdot,\cdot\rangle\) denotes an inner product. For \(\ell:\mathcal Y\times\mathcal Y\to\mathbb R_+\) with \(\ell(y,y)=0\), **A3J: maximum-margin surrogate**, also called generalized hinge loss or structured hinge loss, is
\[
\varphi(s,y)=
\max_{\hat y\in\mathcal Y}
\left(\ell(\hat y,y)+
\langle v(\hat y)-v(y),s\rangle\right).
\tag{11}
\]
It is generally inconsistent except for certain low-noise cases [34, 32]. It originates in [42, 44, 19].

For matching, the graph \(G=(V,E)\) has vertex and edge sets \(V,E\), with
\[
V=V_1\cup V_2,\qquad
V_1=\{1,\ldots,N\},\qquad
V_2=\{N+1,\ldots,2N\}.
\]
Each of the \(N\) edges joins exactly one unique node on each side. All matchings form \(\mathcal Y\), so
\[
k=\operatorname{card}(\mathcal Y)=N!.
\]
The edge-index embedding and mistaken-edge loss are
\[
v(G)\coloneqq
(1\{(u,v)\in E\})_{u\in V_1,v\in V_2}
\in\mathbb R^{N^2},\qquad d=N^2,
\]
\[
\ell(y_1,y_2)\coloneqq
\frac1{2N}\|v(y_1)-v(y_2)\|_1
=\frac1{2N}\|v(y_1)-v(y_2)\|_2^2.
\]
Here \(u,v\) index edge endpoints, and \(\|\cdot\|_1\) and \(\|\cdot\|_2\) are the norms used to compare embeddings.

**Claims and evidence**

- **The matching surrogate is W3C:** Lemma 3.2 gives constants \(1/N\) and \(2\) on \(2N\) vertices.
- **Aggregation handles factorial label cardinality:** The construction has \(k=N!\) and \(d=N^2\); Lemma 3.2 states the requirement \(O(N\log N)\).
- **The task comparison has an explicit threshold and factor:** Lemma 3.2 gives outer threshold factor \(4\), coefficient \(32\), logarithm \(\log(4k(2N+1))\), and comparison factor \(16N\). No numerical table is reported.

#### 3.3.3 Structured prediction

*A general structured decoder gives identifying constants and a logarithmic-in-label-count aggregation requirement.*

###### Lemma 3.3.

A3J is \((\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})\)-W3C with
\[
\mathsf C_{\varphi,1}=
\min_{\hat y\neq y}\ell(\hat y,y),
\qquad
\mathsf C_{\varphi,2}=
\max_{y\in\mathcal Y}\tau(y)+1.
\]
Here \(\tau(y)\) is the identifiability gap defined below. In particular,
\[
v(y)\in\{0,1\}^d,\qquad
\ell(\hat y,y)=\frac1{2d}\|v(\hat y)-v(y)\|_1
\]
give \(\tau(y)=1\) for every \(y\) and \(\mathsf C_{\varphi,2}=2\).

Nontrivial consistency holds once
\[
m\geq\log\operatorname{card}(\mathcal Y),
\]
as in the binary matching case. With \(0\leq\ell(\cdot,\cdot)\leq1\), every measurable \(f:\mathcal X\to\mathbb R^d\) satisfying
\[
R(f)-R^\star\geq
\left(
\frac{32\mathsf C_{\mathsf{MT}}^4}{m}
\log\!\left(
4|\mathcal Y|
\left(1+
\frac{\max_{\hat y\in\mathcal Y}\tau(y)+1}
{\min_{\hat y\neq y}\ell(\hat y,y)}
\right)
\right)
\right)^{\frac{\alpha}{2(1-\alpha^2)}}
\]
also satisfies
\[
R(f)-R^\star\leq
\frac8{\min_{\hat y\neq y}\ell(\hat y,y)}
\bigl(R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star\bigr).
\]
The notation \(|\mathcal Y|\) also denotes label-space cardinality.

The general structured decoder is assumed capable of picking every class. For each \(y\), the strict-decoding set
\[
\mathcal S(y)\coloneqq
\{s:\langle v(y),s\rangle>
\langle v(\hat y),s\rangle
\text{ for }\hat y\neq y\}
\]
is nonempty. Its identifiability gap is
\[
\tau(y)\coloneqq
\inf_{s\in\mathcal S(y)}
\max_{y_+,y_-\neq y}
\frac{\ell(y_+,y)}
{\langle v(y)-v(y_+),s\rangle}
\cdot
\frac{\langle v(y)-v(y_-),s\rangle}
{\ell(y_-,y)}.
\]
Here \(y_+\) and \(y_-\) are alternative responses in the displayed maximum.

**Claims and evidence**

- **The structured hinge surrogate is identifying:** Lemma 3.3 gives \(\mathsf C_{\varphi,1}=\min_{\hat y\neq y}\ell(\hat y,y)\) and \(\mathsf C_{\varphi,2}=\max_y\tau(y)+1\).
- **Binary embeddings give a fixed second constant:** For \(v(y)\in\{0,1\}^d\) and loss normalized by \(1/(2d)\), Lemma 3.3 gives \(\tau(y)=1\) and \(\mathsf C_{\varphi,2}=2\).
- **Aggregation gives nontrivial and linear comparisons:** The section states \(m\geq\log\operatorname{card}(\mathcal Y)\), the displayed threshold with coefficient \(32\), and comparison factor \(8/\min_{\hat y\neq y}\ell(\hat y,y)\). No numerical table is reported.

**Claims and evidence for Section 3.3**

- **Majority voting gives explicit guarantees in three settings:** Lemma 3.1 gives binary constants \(\phi(0)-\phi(\delta)\) and \(\phi(-\delta)\); Lemma 3.2 gives matching constants \((1/N,2)\); Lemma 3.3 gives structured constants \(\min_{\hat y\neq y}\ell(\hat y,y)\) and \(\max_y\tau(y)+1\).

**Claims and evidence for Section 3**

- **Classical consistency has a calibration characterization:** Corollary 3.1 gives positivity and the excess-risk comparison.
- **Aggregation separates ranking consistency from convex pair-only fitting:** Propositions 1 and 2 give failure for \(k\geq3\) and success for \(m\geq k\), respectively.
- **Aggregation amplifies consistency:** Theorem 1 gives slope \(\mathsf C_{\varphi,1}/16\); Corollary 3.2 gives asymptotic consistency; Lemmas 3.1–3.3 give the application-specific constants above.

## 4 Robustness and consistency for models

*The paper turns from unrestricted predictors to restricted classes and shows how aggregation protects consistency under model restrictions.*

A restricted, often parametric \(\mathcal F\) now replaces unrestricted measurable functions. Well-specification makes consistency straightforward, but it is brittle and unrealistic. Surrogate consistency on subsets of measurable functions is a long-standing topic [12, 31].

Recent \(H\)-consistency bounds connect excess target risk and surrogate regret within a regret hypothesis class \(H\), across diverse settings [2, 3, 26, 28, 29]. These results are primarily surrogate-centric. The aggregation contribution is orthogonal and depends less on a particular class and surrogate.

### 4.1 Binary consistency failure in finite dimensions

*An optimal linear classifier can belong to the class while unaggregated surrogate minimizers are nearly orthogonal to it; aggregation restores asymptotic consistency.*

Here **K7M: optimal predictability** means that the class contains a predictor agreeing with an optimal conditional classification decision.

###### Proposition 4.

For every \(\epsilon>0\) and nonzero optimal-direction vector \(\theta^\star\in\mathbb R^d\), some distribution \(P\) on \((X,Y)\) satisfies K7M along \(\theta^\star\), yet every surrogate-minimizing parameter
\[
\theta_\varphi\in
\mathop{\rm argmin}_\theta R_\varphi(f_\theta)
=
\mathop{\rm argmin}_\theta
\mathbb E[\phi(f_\theta(X),Y)]
\]
has
\[
R(f_{\theta_\varphi})>R(f_{\theta^\star}),
\qquad
|\cos\angle(\theta_\varphi,\theta^\star)|
=
\frac{|\langle\theta_\varphi,\theta^\star\rangle|}
{\|\theta_\varphi\|_2\|\theta^\star\|_2}
\leq\epsilon.
\]
Here \(f_\theta\) is the linear predictor defined below, and the displayed normalized inner product measures its parameter angle. This essential orthogonality occurs for any nonnegative loss \(\phi\). Proof: Appendix D.1.

###### Corollary 4.1.

As a corollary to Theorem 2, if \(P\) satisfies K7M along \(\theta^\star\) and approximation tolerances \(\epsilon_m\to0\) as \(m\to\infty\), the aggregated approximate minimizers satisfy
\[
R(f_{\theta_m})\to R(f_{\theta^\star}),
\qquad
\cos\angle(\theta_m,\theta^\star)\to1.
\]

The binary setting is
\[
\mathcal Y=\{\pm1\},\qquad
\ell(\mathsf d(s),y)=1\{ys\leq0\},
\]
with margin surrogate \(\varphi(s,y)=\phi(sy)\), convex \(\phi:\mathbb R\to\mathbb R_+\), input space \(\mathcal X=\mathbb R^d\), and linear class
\[
\mathcal F=
\{f_\theta:f_\theta(x)=\langle\theta,x\rangle\}_{\theta\in\mathbb R^d}.
\]
Even including an optimal classifier does not preserve classical consistency.

K7M requires an \(f\in\mathcal F\) satisfying
\[
\mathsf{sgn}(f(x))=
\mathsf{sgn}(P(Y=1\mid X=x)-1/2).
\tag{12}
\]
For a vector \(\theta^\star\) realizing this condition,
\[
\mathsf{sgn}(\langle\theta^\star,x\rangle)
=
\mathsf{sgn}(P(Y=1\mid x)-\tfrac12).
\]
Then \(f_{\theta^\star}\) minimizes \(R\) over every measurable function, and \(P\) has K7M along \(\theta^\star\).

For an objective \(g\), approximate minimizers are defined by
\[
\operatorname{\epsilon\textup{-argmin}}g
=
\operatorname{\epsilon\textup{-argmin}}_\theta g(\theta)
\coloneqq
\{\theta:g(\theta)\leq\inf_\theta g(\theta)+\epsilon\}.
\]
With independent repeated labels \(Z=(Y_1,\ldots,Y_m)\) and T4K \(A_m\), the relevant parameter sequence is
\[
\theta_m\in
\operatorname{\epsilon_m\textup{-argmin}}_\theta
R_{\varphi,A_m}(f_\theta)
=
\operatorname{\epsilon_m\textup{-argmin}}_\theta
\mathbb E[
\phi(A_m(Y_1,\ldots,Y_m)f_\theta(X))].
\]
Thus a linear restriction can make unaggregated surrogate minimization arbitrarily incorrect, while aggregation retains asymptotic consistency.

**Claims and evidence**

- **An optimal linear classifier does not ensure unaggregated consistency:** Proposition 4 gives strictly greater task risk and \(|\cos\angle(\theta_\varphi,\theta^\star)|\leq\epsilon\) for every \(\epsilon>0\), despite K7M.
- **Aggregation restores risk and directional convergence:** Corollary 4.1 assumes \(\epsilon_m\to0\) and gives task-risk convergence plus \(\cos\angle(\theta_m,\theta^\star)\to1\).
- **Optimal predictability means the optimal decision is represented:** Equation (12) gives the sign equality, with threshold \(1/2\). No numerical table is reported.

### 4.2 Aggregation and restricted classes

*Majority voting gives consistency in cones containing an optimal predictor under the stated surrogate and uniqueness conditions.*

A cone is a function class closed under multiplication by every positive scalar.

###### Theorem 2.

Let a cone \(\mathcal F\) satisfy K7M for \(P\), and assume
\[
P(\Delta(X)>0)=1.
\]
For \(\epsilon_m\geq0\), \(\epsilon_m\to0\), every sequence
\[
f_m\in
\operatorname{\epsilon_m\textup{-argmin}}_{f\in\mathcal F}
R_{\varphi,A_m}(f)
=
\operatorname{\epsilon_m\textup{-argmin}}_{f\in\mathcal F}
\mathbb E[
\varphi(f(X),A_m(Y_1,\ldots,Y_m))]
\]
satisfies \(R(f_m)\to R^\star\). Proof: Appendix D.2.

Multiclass labels are \(\mathcal Y=\{1,\ldots,k\}\), with zero-one loss
\[
\ell(y,y')=1\{y\neq y'\},
\qquad
\mathsf d(s)=\operatorname{argmax}_{y\in[k]}s_y.
\]
The linear-cone conditions are
\[
f(x)^T\mathbf1=0,\qquad
f\in\mathcal F,\ t>0\ \Longrightarrow\ tf\in\mathcal F.
\]
Here \(t\) is a positive scaling factor. Normalization does not affect decoding.

The nonnegative surrogate
\[
\varphi:\mathbb R^k\times[k]\to\mathbb R_+
\]
is Q7D and satisfies
\[
\varphi(s,y)\to0
\quad\text{if}\quad
s_y-s_j\to+\infty\text{ for every }j\neq y.
\tag{13}
\]
The response space is \(\mathcal Z=\mathcal Y^m\), aggregation is T4K, and the population maximizing class
\[
y^\star(x)=\operatorname{argmax}_yP(Y=y\mid x)
\]
is unique.

Multiclass K7M requires an \(f\in\mathcal F\), with \(f:\mathcal X\to\mathbb R^k\), such that
\[
\operatorname{argmax}_y f_y(x)
\in\operatorname{argmax}_yP(Y=y\mid X=x)
\quad\text{for every }x.
\]
The broad cone result requires aggregation. Without it, even binary subclasses containing an optimal predictor can have arbitrarily uninformative surrogates.

Examples of Q7D losses meeting (13) include multiclass logistic loss,
\[
\varphi(s,y)=\log\left(\sum_{j=1}^k e^{s_j-s_y}\right),
\]
and
\[
\varphi(s,y)=\sum_{i\neq y}\phi(s_y-s_i)
\]
for convex, nonincreasing \(\phi\) with \(\phi'(0)<0\) and \(\inf_t\phi(t)=0\). Zhang [51, Thm. 8] establishes consistency of these losses over
\[
\mathcal F=\{f:\mathcal X\to\mathbb R^k:\mathbf1^Tf=0\}.
\]
Here \(e^{(\cdot)}\) is exponential notation.

Binary margin predictors correspond to
\[
f(x)=(g(x),-g(x))
\]
for measurable score function \(g\). For \(\mathcal X=\mathbb R^d\), linear predictors
\[
f(x)=(\langle\theta_1,x\rangle,\ldots,\langle\theta_k,x\rangle),
\qquad \sum_{i=1}^k\theta_i=0,
\]
form a convex cone. Such examples typically require strong structure, such as convexity, without aggregation. The minimal limiting condition permits aggregated asymptotic consistency.

**Claims and evidence**

- **Aggregation gives consistency in the stated cones:** Theorem 2 assumes K7M, \(P(\Delta(X)>0)=1\), and \(\epsilon_m\to0\), and concludes \(R(f_m)\to R^\star\).
- **The surrogate must vanish as every correct-class margin diverges:** Equation (13) gives the exact limit \(0\).
- **The listed losses meet the consistency framework:** The displayed logistic loss and pairwise sum use \(\phi'(0)<0\) and \(\inf_t\phi(t)=0\); Zhang [51, Thm. 8] supplies the cited unrestricted consistency result. No numerical table is reported.

### 4.3 Finite-dimensional multiclass classification

*Small link mis-specification can destroy unaggregated logistic consistency, while majority voting recovers the optimal parameter direction asymptotically.*

Here \(\Theta^\star\) is the true multiclass parameter matrix and \(\Theta_m\) is the population logistic minimizer after aggregation. The logistic link is \(\sigma^{\rm lr}\), defined below.

###### Proposition 5.

Let \(k\geq3\), and let the covariance matrix be \(\Sigma=I\). For a Gaussian vector
\[
Z\sim\mathsf N(0,I_{k-1}),
\]
where \(I_{k-1}\) is the identity matrix, assume invertibility of the map
\[
M\mapsto DM\coloneqq
\mathbb E\!\left[
Z\bigl(
\nabla\sigma^{\rm lr}({\Theta^\star}^TZ)MZ
\bigr)^T
\right].
\]
Here \(M\) is the matrix argument, \(DM\) is the displayed linear map, and \(\nabla\) denotes differentiation. Some \(\epsilon_0>0\) then has the following property. Every \(\epsilon\in(0,\epsilon_0)\) admits a perturbation set \(\mathcal T_\epsilon\) of Lebesgue measure at most \(\epsilon\) with
\[
\Theta_1(\epsilon)/\|\Theta_1(\epsilon)\|
\neq
\Theta^\star/\|\Theta^\star\|.
\]
The norm \(\|\cdot\|\) is the parameter norm used in these normalized directions.

Proposition 5 proof: Appendix E.1.

###### Theorem 3.

Suppose
\[
\Theta^\star=U^\star T^\star,
\]
with orthogonal \(U^\star\in\mathbb R^{d\times(k-1)}\) and nonsingular \(T^\star\in\mathbb R^{(k-1)\times(k-1)}\). For each \(m\), there exists \(T_m\in\mathbb R^{(k-1)\times(k-1)}\) with
\[
\Theta_m=U^\star T_m,
\]
and
\[
\|T_m\|\to\infty,\qquad
T_m/\|T_m\|\to T^\star/\|T^\star\|
\quad(m\to\infty).
\]

Theorem 3 proof: Appendix E.2.

Pair-based surrogate risks can be inconsistent even with K7M and a linear model correct except on an \(\epsilon\)-fraction of data. T4K recovers the optimal linear predictor asymptotically. For \(k\geq3\), arbitrarily slight true-link mis-specification defeats risk consistency. Aggregation needs no true-link assumption beyond ordering consistency, defined in (14).

Here
\[
\mathcal X=\mathbb R^d,\qquad \mathcal Y=[k],\qquad
\Theta=[\theta_1~\cdots~\theta_{k-1}]
\in\mathbb R^{d\times(k-1)}.
\]
The conditional response law is
\[
Y\mid X=x\sim\mathsf{Cat}(p_\Theta(x)),
\]
where \(\mathsf{Cat}\) denotes a categorical distribution and \(p_\Theta(x)\) is its probability vector:
\[
p_\Theta(x)\in\mathbb R_+^k,\qquad
\mathbf1^\top p_\Theta(x)=1.
\]
Specifically,
\[
p_\Theta(x)=
\begin{bmatrix}
\sigma(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)\\
1-\mathbf1^\top
\sigma(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)
\end{bmatrix}.
\]
The link
\[
\sigma:\mathbb R^{k-1}\to\mathbb R_+^{k-1},
\qquad \mathbf1^\top\sigma\leq1,
\]
maps scores to the first \(k-1\) probabilities. For score vector \(t\), put \(t_k=0\) and
\[
\sigma_k(t)=1-\mathbf1^\top\sigma(t).
\]
The imposed **ordering consistency** condition is
\[
t_y=\max_{1\leq i\leq k}t_i
\quad\Longleftrightarrow\quad
\sigma_y(t)=\max_{1\leq i\leq k}\sigma_i(t).
\tag{14}
\]
Multiclass logistic regression uses the link, also called softmax here,
\[
\sigma_i^{\rm lr}(t)=
\frac{e^{t_i}}{1+e^{t_1}+\cdots+e^{t_{k-1}}}.
\]

The multilabeled dataset is
\[
\{(X_i;(Y_{1i},\cdots,Y_{im}))\}_{i=1}^n,
\]
with
\[
Y_{ij}\mid X_i\stackrel{\rm iid}{\sim}
\mathsf{Cat}(p_{\Theta^\star}(X_i)).
\]
The arbitrarily tie-broken majority-vote label
\[
Y_i^+=A_m(Y_{i1},\cdots,Y_{im})
\]
obeys
\[
Y_i^+\mid X_i\sim\mathsf{Cat}(p_{\Theta^\star,m}(X_i)).
\]
Let \(\rho_m(t)\) be the T4K distribution for \(m\) items with initial probabilities \(\sigma(t)\in\mathbb R_+^{k-1}\). Then
\[
p_{\Theta,m}(x)=
\begin{bmatrix}
\rho_m(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)\\
1-\mathbf1^\top
\rho_m(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)
\end{bmatrix}.
\]
The aggregated link \(\rho_m\) satisfies ordering consistency.

The fitted logistic loss, with reference-class parameter \(\theta_k=\mathbf0\), the zero vector, is
\[
\varphi(\Theta^\top x,y)=
-\langle\theta_y,x\rangle+
\log\left(1+\sum_{i=1}^{k-1}
\exp(\langle\theta_i,x\rangle)\right).
\tag{15}
\]
Here \(\exp\) denotes the exponential. For the population objective
\[
L_m(\Theta)=\mathbb E[\varphi(\Theta^\top X,Y_m^+)],
\]
where \(Y_m^+\) denotes the aggregated response, differentiation gives
\[
\nabla_{\theta_i}\varphi(\Theta^\top x,y)=
-x(1\{y=i\}-\sigma_i^{\rm lr}(\Theta^\top x)).
\]
Thus the minimizer \(\Theta_m\) satisfies
\[
\nabla_\Theta L_m(\Theta_m)=
\mathbb E\!\left[
X\left(
\sigma^{\rm lr}(\Theta_m^\top X)
-\rho_m({\Theta^\star}^\top X)
\right)^\top
\right]=0.
\]

Well-specified unaggregated models have standard statistical consistency and efficiency. For \(m\geq1\), \(k=2\), and \(X\sim\mathsf N(0,I_d)\), Cheng et al. [7, Prop. 3] obtain
\[
\Theta_m=t_m\Theta^\star
\quad\text{for some }t_m>0
\]
even with a mis-specified link. The direction \(\Theta^\star/\|\Theta^\star\|_2\) therefore suffices for consistent binary classification at every aggregation level.

For fixed \(\mathcal T_\epsilon\subset\mathbb R^{k-1}\) of Lebesgue measure \(\epsilon\), set
\[
\sigma^\epsilon(t)=
\sigma^{\rm lr}(t)1\{t\notin\mathcal T_\epsilon\}
+\frac1k\mathbf1\,1\{t\in\mathcal T_\epsilon\}.
\]
The conditional distribution on \(Y\in\{1,\ldots,k\}\) samples \(Y\sim\sigma^{\rm lr}(t)\) if \(t\in\mathcal T_\epsilon\) and uniformly otherwise. The link satisfies ordering consistency and K7M.

For \(\epsilon>0\), define
\[
L_{m,\epsilon}(\Theta)\coloneqq
\mathbb E[
\mathbb E_{\sigma^\epsilon}[
\varphi(\Theta^\top X,Y_m^+)\mid X]],
\qquad
\Theta_m(\epsilon)=
\operatorname{argmin}_\Theta L_{m,\epsilon}(\Theta).
\]
This is the population logistic loss for \(m\)-label T4K when
\[
Y\mid X=x\sim\sigma^\epsilon({\Theta^\star}^\top x).
\]
Although \(\Theta_1(0)=\Theta^\star\), arbitrarily small positive corruption destroys unaggregated consistency.

For general ordering-consistent \(\sigma\), Theorem 3 uses
\[
L_{m,\sigma}(\Theta)=
\mathbb E[
\mathbb E_\sigma[
\varphi(\Theta^\top X,Y_m^+)\mid X]],
\qquad
\Theta_m=\operatorname{argmin}L_{m,\sigma}(\Theta).
\]
These definitions give robustness under mis-specification whenever some link describes the \(X\)–\(Y\) relationship. Without aggregation, multiclass methods are generally inconsistent.

**Claims and evidence**

- **Arbitrarily small perturbation can change the unaggregated parameter direction:** Proposition 5 applies for \(k\geq3\), \(\Sigma=I\), and invertible \(M\mapsto DM\). For every \(\epsilon\in(0,\epsilon_0)\), it gives a set of measure at most \(\epsilon\) and unequal normalized directions.
- **Aggregation recovers the optimal direction asymptotically:** Theorem 3 gives \(\Theta_m=U^\star T_m\), \(\|T_m\|\to\infty\), and \(T_m/\|T_m\|\to T^\star/\|T^\star\|\).
- **The binary Gaussian case preserves direction at every aggregation level:** The cited [7, Prop. 3] gives \(\Theta_m=t_m\Theta^\star\), \(t_m>0\), for \(m\geq1\), \(k=2\), and \(X\sim\mathsf N(0,I_d)\).
- **Uncorrupted and corrupted unaggregated fits differ:** The section states \(\Theta_1(0)=\Theta^\star\); Proposition 5 gives directional failure for arbitrarily small positive \(\epsilon\). No numerical table is reported.

**Claims and evidence for Section 4**

- **Restricted classes can defeat unaggregated surrogate consistency despite containing an optimal predictor:** Proposition 4 gives excess task risk and absolute cosine at most any \(\epsilon>0\).
- **Aggregation restores consistency in the stated classes:** Theorem 2 gives \(R(f_m)\to R^\star\); Corollary 4.1 additionally gives cosine convergence to \(1\).
- **Multiclass link perturbations and aggregation separate:** Proposition 5 gives failure for \(k\geq3\); Theorem 3 gives normalized-factor convergence as \(m\to\infty\).

## 5 Numerical illustrations

*Two experiments illustrate nonparametric consistency amplification and finite-dimensional consistency under link mis-specification.*

Two experiments corroborate the theory: nonparametric amplification and finite-dimensional multiclass asymptotic consistency under link mis-specification.

##### Surrogate consistency amplification for D4X.

*The comparison curves move from quadratic toward linear growth as aggregation increases, while the general-theorem curves remain conservative.*

The aggregated curve changes from quadratic toward linear as \(m\to\infty\), upgrading consistency. The general-theorem curves are conservative.

The binary margin surrogate \(\varphi(f(x),y)=\phi(yf(x))\) uses **D4X: truncated quadratic loss**:
\[
\phi(\delta)=\max\{1-\delta,0\}^2.
\]
Its exact unaggregated H8N is
\[
\psi(\epsilon)=\epsilon^2
\]
[4, Example 2]. Taking \(\delta=1\) in the binary lemma and imposing no low-noise condition, so \(\alpha=0\) and \(\mathsf C_{\mathsf{MT}}=1\), gives the aggregated comparison curve
\[
\begin{aligned}
\psi_m(\epsilon)
&=
\frac{\phi(0)-\phi(\delta)}{16}
\left(
\epsilon-
\sqrt{\frac{32}{m}
\log\!\left(
\frac{8(\phi(-\delta)+\phi(0)-\phi(\delta))}
{\phi(0)-\phi(\delta)}
\right)}
\right)_+\\
&=\frac1{16}
\left(\epsilon-\sqrt{\frac{32}{m}\log40}\right)_+.
\end{aligned}
\]
The subscript \(+\) is positive-part notation. The parallel comparisons are
\[
\psi(R(f)-R^\star)\leq R_\varphi(f)-R_\varphi^\star,
\qquad
\psi_m(R(f)-R^\star)\leq
R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star.
\]

Figure 1: D4X comparison curves for unaggregated \(\psi\) (optimal) and T4K \(\psi_m\) (suboptimal).

| Figure | Aggregated-label counts |
|---|---|
| 1 | \(m\in\{2^{16},2^{20},2^{24},2^{28},2^{32}\}\) |

**Claims and evidence**

- **The unaggregated comparison is exactly quadratic:** The displayed \(\psi(\epsilon)=\epsilon^2\), cited to [4, Example 2].
- **The aggregated comparison approaches linear growth:** The display gives \(\psi_m(\epsilon)=\frac1{16}(\epsilon-\sqrt{\frac{32}{m}\log40})_+\).
- **Figure 1 compares increasing aggregation levels:** The Figure 1 table, row “1,” lists \(m\in\{2^{16},2^{20},2^{24},2^{28},2^{32}\}\). Its caption identifies the unaggregated curve as optimal and aggregated curves as suboptimal.

##### Asymptotic consistency for a mis-specified logistic model.

*Synthetic data from a corrupted link illustrate asymptotic consistency under fixed-observation and fixed-total-label designs.*

Synthetic data from a corrupted link exhibit asymptotic consistency as \(m\to\infty\). The fitting loss is the finite-dimensional logistic loss above. The corrupted link \(\widetilde\sigma\) agrees with softmax near the origin and shrinks scores outside radius \(r\):
\[
\widetilde\sigma(t)=
\sigma^{\rm lr}(t)1\{\|t\|\leq r\}
+\sigma^{\rm lr}(\alpha t)1\{\|t\|>r\}.
\]
Here \(r\) is the radius and \(\alpha\) is the score-shrinkage factor.

| (i) Experiment with constant dataset size \(n\) | (ii) Experiment with constant label size \(nm\) |
|---|---|
| For each \(m\), \(n=10{,}000\) multilabeled observations \((X_i,(Y_{i1},\ldots,Y_{im}))\) sampled from \(\widetilde\sigma\) | For each \(m\), \(n=640{,}000/m\); total labels constant across \(m\) |

Figure 2: Mis-specified logistic-model illustrations, left panel (i), right panel (ii); plotted classification-rule error is \(\left\|\|T_m\|-T^\star/\|T^\star\|\right\|\) against label count.

| Setting | Value |
|---|---|
| Dimension | \(d=2\) |
| Classes | \(k=3\) |
| True parameters | \(\theta_1^\star=(0,1)^\top,\quad\theta_2^\star=(1,1)^\top\) |
| Corrupted-link parameters | \(r=3,\quad\alpha=0.1\) |
| Label counts | \(m\in\{1,4,16,64,256,1024\}\) |
| Error bars | \(95\%\) |
| Trials for each \(m\) | \(T=100\) |

Here \(T\) denotes the number of trials for each \(m\).

**Claims and evidence**

- **The corrupted-link experiments exhibit asymptotic consistency:** Figure 2 is the reported illustration; the paper supplies no numerical error values in its tables.
- **Both observation-budget and label-budget designs are used:** The experiment-design table's sole data row gives \(n=10{,}000\) in column (i) and \(n=640{,}000/m\) in column (ii).
- **The experiment uses a three-class, two-dimensional model:** The settings table, rows “Dimension” and “Classes,” gives \(d=2\) and \(k=3\); “True parameters” gives \((0,1)^\top\) and \((1,1)^\top\).
- **The corruption and aggregation levels are specified:** The settings table, rows “Corrupted-link parameters” and “Label counts,” gives \(r=3\), \(\alpha=0.1\), and \(m\in\{1,4,16,64,256,1024\}\).
- **Repeated-trial uncertainty is displayed:** The settings table, rows “Error bars” and “Trials for each \(m\),” gives \(95\%\) and \(T=100\).

**Claims and evidence for Section 5**

- **The illustrations cover amplification and mis-specified-model consistency:** Figure 1 and its row “1” give five aggregation counts from \(2^{16}\) through \(2^{32}\); Figure 2 and the settings-table row “Label counts” give \(1,4,16,64,256,1024\). The exact comparison formula is displayed above; numerical Figure 2 errors are not tabulated.

## 6 Discussion and future work

*The paper contrasts efficiency with robustness and identifies open questions about finite aggregation, noise, mis-specification, and restricted classes.*

Cheng, Asi and Duchi [7] discuss the collection debate. Using all unaggregated labels appears statistically more efficient when models can represent all uncertainty. Otherwise, cleaning appears more robust. Dorner and Hardt [11] favor more noisy rather than cleaned labels for validation comparisons of binary classifiers. The present results establish fundamental limitations of uncleaned supervised-label information and robustness gains from cleaning.

##### Finite \(m\) and fundamental limits.

*Many results rely on infinitely growing aggregation, while a joint finite-count theory remains open.*

Many consistency results require \(m\to\infty\). Reliance on this limit remains unsatisfying. Ranking admits explicit finite-\(m\) guarantees.

A theory depending jointly on \((\ell,\varphi)\) and available count \(m\) remains open. Second- or third-order item comparisons, including powers \(C_x^p\) suggested by Keener [20], may require increasing counts. Precisely separating problems that require cleaning and aggregation from those that do not is a central challenge and a natural, important direction.

##### Fundamental limits of B2Z.

*The noise condition number can remain large beyond binary classification, motivating closer task–surrogate connections and sharper limits.*

Beyond binary classification, \(\kappa(X)\) may remain large even with low N6Y noise, \(\alpha\approx1\). Binary problems trivially have \(\kappa(X)=1\). Minimal surrogate assumptions produce this behavior. Closer task–surrogate connections could better represent difficulty. Explicit failure constructions and sharper fundamental limits could improve understanding of predictive-model fitting.

##### Mis-specified models.

*The model results require an optimal classifier in the hypothesis class, and going beyond that restriction needs new approaches.*

The mis-specified-model results require K7M: a Bayes-optimal classifier, meaning a classifier minimizing risk over all measurable functions, must belong to \(\mathcal F\). Classical surrogate consistency can fail even there, while aggregation succeeds.

Beyond this restriction, Nguyen et al. [31] for binary classification and Duchi et al. [12] for multiclass classification show that jointly inferring predictor \(f\) and a representation of \(x\) forces a task-loss-dependent surrogate form. Their decision rules remain infinitely powerful. New approaches are needed.

##### Restricted-class comparison with aggregation.

*Extending aggregation comparisons to restricted classes could connect the paper's nonparametric and parametric advantages.*

Nonparametric comparison inequalities mitigate surrogate-induced regret, the excess surrogate risk. Parametric results establish statistical consistency. Recent \(H\)-consistency work [2, 28, 27, 29] suggests extending aggregation comparisons to restricted classes.

Mao et al. [27] establish rate-optimal quadratic \(\psi\) growth for smooth convex loss. Aggregation-based upgrades in \(\mathcal H\)-consistency remain of interest, where \(\mathcal H\) denotes the restricted hypothesis class. An extension could unify parametric and nonparametric advantages and characterize how hypothesis-class “size” affects aggregation.
