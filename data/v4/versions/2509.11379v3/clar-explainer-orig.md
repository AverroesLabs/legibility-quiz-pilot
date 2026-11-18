**Summary.** The paper asks whether aggregating noisy labels improves the consistency of learned predictors. It studies surrogate losses, which replace task losses during fitting, both over all measurable predictors and over restricted model classes. It develops calibration inequalities for aggregated labels and analyzes ranking, classification, and structured prediction. For ranking with \(k\geq3\) items, no convex surrogate using individual comparisons is Fisher consistent, but an aggregated squared-loss procedure is consistent when \(m\geq k\). More generally, identifying surrogates become asymptotically consistent as the number \(m\) of aggregated labels grows. Under noise conditions, aggregation gives approximately linear comparison inequalities. In restricted classes, aggregation can restore consistency when ordinary surrogate minimization fails, including under link mis-specification. Two numerical illustrations support these conclusions. Many guarantees require \(m\to\infty\).

**Glossary.** Definitions below restate the paper’s terminology and notation. Symbols with several uses are listed together.

- **Supervised learning:** learning predictions from features and labels.
- **Data collection pipeline / data analysis pipeline:** the process from collecting and cleaning data through model fitting and prediction.
- **Label cleaning / label aggregation / data aggregation:** combining noisy label information into an aggregated response.
- **Aggregated labels:** label information distilled from noisy responses.
- **Raw labels / non-aggregated labels:** individual labels used without aggregation.
- **Crowdsourcing:** collecting multiple labels and denoising them.
- **Gold standard labels:** the labels sought by the crowdsourcing data-cleaning procedures discussed in the introduction.
- **Robustness:** the consistency advantages that label cleaning provides in the settings studied.
- **Consistency / risk consistency:** convergence to optimal task risk when surrogate risk is minimized.
- **Surrogate risk consistency:** consistency obtained when a surrogate loss replaces a task loss.
- **Task loss \(\ell\):** the loss for the desired prediction task.
- **Surrogate loss \(\varphi\):** a loss minimized in place of the task loss.
- **Zero-one loss / \(0\)-\(1\) loss / mis-classification error:** the loss indicating an incorrect prediction.
- **Convex surrogate:** a surrogate convex in its score argument.
- **Margin-based surrogate:** a loss of the form \(\varphi(s,y)=\phi(sy)\).
- **\(\phi\):** the one-variable function defining a margin-based surrogate.
- **Nonsmooth / nonconvex / flat loss landscape:** properties of task losses that obstruct practical optimization in the paper’s discussion.
- **First-order optimization:** the practical optimization approach mentioned as obstructed by the flat \(0\)-\(1\) loss.
- **Statistical efficiency:** the advantage attributed to using all information in perfectly specified models.
- **Well-specified / perfectly specified model:** a model able to represent the label uncertainty being modeled.
- **Mis-specified model / loss / link:** a model, loss, or link that does not match the specified data-generating relationship.
- **Nonparametric setting:** the setting where the hypothesis class contains all measurable functions.
- **Parametric / finite-dimensional setting:** the restricted model settings studied in Section 4.
- **Hypothesis class \(\mathcal F\):** the collection of scoring functions over which risk is minimized.
- **Restricted hypothesis class:** a class containing only a subset of measurable functions.
- **Borel measurable functions / Borel functions:** the functions included in the classical Fisher-consistency hypothesis class.
- **Empirical processes:** the statistical theory available for learning from paired observations \((X,Y)\).
- **\(H\)-consistency / \(\mathcal H\)-consistency:** bounds linking excess target risk to surrogate regret within a restricted hypothesis class \(H\).
- **Regret / excess risk:** risk above the corresponding minimum.
- **Binary classification:** classification with \(\mathcal Y=\{\pm1\}\).
- **Multiclass classification / multiclassification:** classification with \(\mathcal Y=\{1,\ldots,k\}\).
- **Multi-label classification:** the classification setting cited in the introduction’s review of surrogate consistency.
- **Ranking / rank aggregation:** transforming pairwise-comparison probabilities into an ordering of items.
- **Structured prediction:** prediction over a structured output space encoded by an embedding.
- **Ordinal regression:** the surrogate-consistency setting cited in the introduction.
- **Linear estimator / linear predictor / linear functional:** a predictor built from inner products between parameters and inputs.
- **\(\mathcal X\), \(x\), \(X\):** input space, an input, and a random input.
- **Features \(X_i\):** the inputs in the collected sample.
- **\(\mathcal Y\), \(y\), \(Y\):** output space, an output or class, and a random label.
- **\(Y_i,Y_l,Y_{ij},Y_{(l)}\):** sample, repeated, or distance-ordered labels, as specified where used.
- **\(n\):** the number of collected observations.
- **\(m\):** the number of labels aggregated, or the number of additional examples in Example 2.
- **\(d\):** score dimension, or input dimension in the finite-dimensional settings.
- **\(k\), \([k]\):** the number of ranking items or label-space elements, and \(\{1,\ldots,k\}\).
- **\(i,j,l\):** indices for observations, labels, items, or coordinates.
- **\(P\):** the data distribution.
- **\(P(\cdot\mid X)\), \(P_{Y\mid X=x}\):** the conditional label distribution.
- **\(\mathbb E,\mathbb E_P\):** expectation, with the subscript identifying the distribution when shown.
- **\(\mathbb P\):** probability.
- **i.i.d. / iid:** independently and identically distributed.
- **\(\mathbb R,\mathbb R_+\):** the real and nonnegative-real score or loss spaces used in the paper.
- **\(f,f_n,f_m\):** a scoring function or a sequence of scoring functions.
- **\(s,s'\):** score vectors.
- **Decoder \(\mathsf d\):** the map from scores to final predictions.
- **\(\widehat y,\hat y\):** a decoded prediction or an empirical loss-minimizing label.
- **\(\circ\):** composition, as in the prediction \(\mathsf d\circ f(x)\).
- **\(\mathsf{sgn}\):** the sign decoder used for binary classification.
- **\(1\{\cdot\}\):** the indicator appearing in losses, counts, and piecewise definitions.
- **\(R(f)\):** task risk.
- **\(R_\varphi(f)\):** surrogate risk without aggregation.
- **\(R_{\varphi,A}(f)\):** surrogate risk after aggregation by \(A\).
- **\(R(s\mid x)\):** conditional task risk.
- **\(R_\varphi(s\mid x)\), \(R_{\varphi,A}(s\mid x)\):** conditional surrogate risks without and with aggregation.
- **\(R^\star,R_\varphi^\star,R_{\varphi,A}^\star\):** the corresponding infimal risks over \(\mathcal F\).
- **\(\delta_\ell,\delta_\varphi,\delta_{\varphi,A}\):** pointwise excess task, surrogate, and aggregated-surrogate risks.
- **\(\star\) as a superscript:** notation for an optimum or target parameter.
- **\(\inf,\min,\max\):** the risk and score extrema used in the definitions and bounds.
- **\(\argmin,\argmax\):** the sets of minimizing and maximizing arguments.
- **Fisher consistency (i):** convergence of surrogate risk to its infimum implies convergence of task risk to its infimum.
- **Uniform comparison inequality (ii):** a positive nondecreasing function of excess task risk is bounded by excess surrogate risk.
- **\(\psi\):** a comparison function, or the pointwise calibration function before biconjugation.
- **\(\overline\psi\):** the uniform calibration function.
- **\(\overline\psi_A(\epsilon,x),\overline\psi_A(\epsilon)\):** the calibration functions defined for aggregated responses.
- **\(\psi_A\):** the Fenchel biconjugate of \(\overline\psi_A\).
- **Calibration function:** the smallest excess surrogate risk subject to a specified excess task risk.
- **Fenchel biconjugate / \(^{**}\):** the operation used to construct the comparison function from the uniform calibration function.
- **\(\epsilon,\delta\):** positive risk tolerances, approximation levels, or scalar arguments, as specified locally.
- **\(\mathcal Z,Z\):** the abstract response space and response that replace an individual label.
- **\(\mathcal A\):** the aggregation space.
- **\(A:\mathcal Z\to\mathcal A\):** an aggregation function.
- **Repeated sampling:** collecting conditionally independent labels for the same input.
- **Majority vote \(A_m\):** the empirical loss-minimizing label, or its identifying parameter under generalized majority vote.
- **Generalized majority vote:** the aggregator in equation (7).
- **\(K\)-nearest neighbors:** aggregation over the \(K\) closest additional inputs, together with the original input in Example 2.
- **\(\operatorname{dist}\):** the distance metric on \(\mathcal X\).
- **\(X_{(1)},\ldots,X_{(m)}\):** additional inputs ordered by distance from \(X\).
- **\(X_{(0)}\):** the original input \(X\).
- **\(A_{m,K}\):** the empirical loss minimizer over labels indexed \(0,\ldots,K\).
- **\(p_{ij}\):** the conditional probability that item \(i\) is preferred to item \(j\).
- **\(C_x\):** the normalized transition matrix built from the \(p_{ij}\).
- **Stochastic matrix:** the matrix \(C_x\) satisfying \(C_x^T\mathbf1=\mathbf1\).
- **\(\mathbf1,\mathbf0\):** the all-one and zero vectors used in normalizations and parameter conventions.
- **\(C_x\mathbf1\):** the vector used to rank items by how often they are preferred.
- **\(C_x^p\), \(p\):** higher powers of the comparison matrix and their exponent.
- **Perron vectors:** an alternative ranking construction mentioned alongside higher powers of \(C_x\).
- **\(e_i\):** the coordinate vectors used to compare entries of \(C\mathbf1\).
- **\(m_{ij},m_j\):** the comparison counts and their column totals defined in Section 3.1.
- **\(\star\) as a response:** the aggregation output when some \(m_j=0\).
- **\(q\):** the response vector used in the ranking squared-loss surrogate.
- **Squared loss:** \(\varphi(s,q)=\|s-q\|_2^2\) in Proposition 2.
- **\(b(x)\):** the integrable bound on pointwise excess task risk in Proposition 3.
- **\(\operatorname{card}(\mathcal Y),|\mathcal Y|\):** label-space cardinality.
- **Identifying surrogate / identifiable surrogate:** a surrogate satisfying Definition 3.1.
- **\(\mathsf C_{\varphi,1},\mathsf C_{\varphi,2}\):** the positive identification gap and finite cross-label loss bound in (6a)–(6b).
- **\(a_y,s_y\):** aggregation parameters and score vectors identifying class \(y\).
- **Binary hinge loss:** \(\varphi(s,a)=\max\{1-sa,0\}\).
- **\(a\):** an aggregation-space argument of a surrogate.
- **\(y^\star(x)\):** the set of conditional task-loss minimizers, or the unique most probable class where specified.
- **\(M\):** an eventual sample threshold in Section 3.2.2 or a condition-number threshold in Theorem 1.
- **\(m(n)\):** an aggregation count growing with \(n\).
- **Level-\((\xi,\zeta)\) consistency:** a comparison inequality whose function is at least \(\zeta\epsilon\) for \(\epsilon\geq\xi\).
- **\(\xi,\xi_m,\xi_{m,k}\):** excess-task-risk thresholds for approximately linear comparison.
- **\(\zeta\):** the linear comparison slope.
- **\(o_m(1)\):** a quantity tending to zero as \(m\) grows.
- **Low noise:** the setting quantified by the Mammen–Tsybakov conditions.
- **Mammen–Tsybakov conditions:** conditions (Nα) and (Mβ), relating prediction disagreement or small conditional gaps to risk.
- **\(\Delta(x)\):** the smallest conditional excess risk of an incorrect prediction.
- **\(\mathsf C_{\mathsf{MT}}\):** the positive constant in the Mammen–Tsybakov conditions.
- **Noise exponent \(\alpha\):** the exponent in condition (Nα), with \(\alpha\in[0,1]\).
- **\(\beta\):** the exponent in condition (Mβ), with \(\beta=\alpha/(1-\alpha)\).
- **(Nα):** the bound on prediction disagreement by \(\mathsf C_{\mathsf{MT}}(R(f)-R^\star)^\alpha\).
- **(Mβ):** the bound \(\mathbb P(\Delta(X)\leq\epsilon)\leq(\mathsf C_{\mathsf{MT}}\epsilon)^\beta\).
- **Noise condition number \(\kappa(x)\):** the ratio of largest to smallest incorrect-prediction conditional excess risk in (9).
- **Error function \(e_m(t)\):** the function in (10) used to quantify majority-vote accuracy.
- **\(t\):** a scalar argument or scale, a condition-number value, or a vector of link scores, depending on context.
- **\(O(\cdot)\):** the order notation used for aggregation counts.
- **\(\partial\phi(0)\):** the subgradient set of \(\phi\) at zero.
- **\(\phi'(0)\):** the derivative condition used in binary surrogate consistency.
- **Feasible:** the descriptor used for the constants in Lemma 3.1.
- **Embedding \(v:\mathcal Y\to\mathbb R^d\):** the map encoding structural information about outputs.
- **\(\langle\cdot,\cdot\rangle\):** the inner products defining linear scores and structured decoders.
- **Maximum-margin / max-margin / generalized hinge / structured hinge surrogate:** the loss in (11).
- **Bipartite matching:** a graph whose \(N\) edges uniquely connect the two sets of \(N\) vertices.
- **\(G=(V,E)\):** a graph with vertex set \(V\) and edge set \(E\).
- **\(V_1,V_2\):** the left and right vertex sets.
- **\(N\):** the size of each vertex set in bipartite matching.
- **\(N!\):** the label-space size for all bipartite matchings.
- **\(u,v\) as vertices:** the endpoints indexing the edge embedding.
- **\(\|\cdot\|_1,\|\cdot\|_2,\|\cdot\|\):** the norms appearing in losses, parameter normalization, and reported errors.
- **\(\mathcal S(y)\):** the scores that strictly select class \(y\) under the embedding decoder.
- **Identifiability gap \(\tau(y)\):** the loss-to-score-gap ratio expression in Section 3.3.3.
- **\(y_+,y_-\):** the alternative outputs in the definition of \(\tau(y)\).
- **Optimal predictability:** existence of a predictor in \(\mathcal F\) that makes Bayes-optimal class predictions.
- **Bayes-optimal classifier:** the classifier selecting a most probable conditional label in the classification settings.
- **\(\theta,\theta^\star,\theta_\varphi,\theta_m\):** a linear parameter, an optimally predicting target, a surrogate-risk minimizer, and aggregated-risk approximate minimizers.
- **\(f_\theta\):** the linear predictor \(x\mapsto\langle\theta,x\rangle\).
- **\(\cos\angle(\theta,\theta^\star)\):** the normalized inner product comparing parameter directions.
- **\(\epsilon\textup{-argmin}\):** parameters whose objective value is within \(\epsilon\) of its infimum.
- **\(g\):** an objective in the approximate-minimizer definition, or a scalar predictor in the binary-to-multiclass construction.
- **\(\epsilon_m\):** the approximate-minimization tolerance tending to zero.
- **Linear cone / cone:** a hypothesis class closed under positive scaling, with the score normalization stated in Section 4.2.
- **Limiting loss condition (13):** surrogate loss tends to zero when the correct score exceeds all others without bound.
- **Multiclass logistic loss / logistic loss:** the losses displayed in Section 4.2 and equation (15).
- **\(\exp,e^{(\cdot)},\log\):** the exponential and logarithm operations used in logistic losses and bounds.
- **\(\Theta=[\theta_1\cdots\theta_{k-1}]\):** the multiclass linear-parameter matrix.
- **\(\Theta^\star,\Theta_m,\Theta_m(\epsilon)\):** the true parameter matrix and the population logistic-risk minimizers.
- **\(\mathsf{Cat}\):** the categorical label distribution.
- **\(p_\Theta(x)\):** the conditional class-probability vector determined by \(\Theta\) and a link.
- **Link \(\sigma\):** the function mapping \(k-1\) linear scores to the first \(k-1\) class probabilities.
- **\(t_k,\sigma_k(t)\):** the baseline score \(0\) and remaining class probability \(1-\mathbf1^\top\sigma(t)\).
- **Link consistency (14):** equivalence between maximal scores and maximal linked probabilities.
- **\(\sigma^{\textup{lr}},\textup{lr}\):** the logistic-regression link and its label in the notation.
- **\(\rho_m\):** the first \(k-1\) probabilities of majority vote on \(m\) labels.
- **\(Y_i^+,Y_m^+\):** majority-vote labels.
- **\(p_{\Theta,m}(x)\):** the conditional majority-vote probability vector.
- **\(L_m,L_{m,\epsilon},L_{m,\sigma}\):** population logistic losses with majority vote and the specified link.
- **\(\nabla_{\theta_i},\nabla_\Theta,\nabla\sigma^{\textup{lr}}\):** the derivatives appearing in logistic-risk first-order conditions and Proposition 5.
- **\(\mathsf N(0,I_d),\mathsf N(0,I_{k-1})\):** the centered normal input distributions stated in Section 4.3.
- **\(I,I_d,I_{k-1},\Sigma\):** the identity matrices and the matrix set equal to \(I\) in Proposition 5.
- **\(\mathcal T_\epsilon\):** the set of link-score values on which the logistic link is modified.
- **Lebesgue measure \(\epsilon\):** the size restriction imposed on \(\mathcal T_\epsilon\).
- **\(\sigma^\epsilon\):** the modified logistic link displayed before Proposition 5.
- **\(\epsilon_0\):** the positive upper limit for the perturbations in Proposition 5.
- **\(D,M\) in Proposition 5:** the linear map \(M\mapsto DM\) and its matrix argument.
- **\(U^\star,T^\star,T_m\):** the orthogonal factor, nonsingular target factor, and fitted factors in Theorem 3.
- **Truncated quadratic loss:** \(\phi(\delta)=\max\{1-\delta,0\}^2\).
- **\(\psi_m\):** the majority-vote comparison function used in the first numerical illustration.
- **\((\cdot)_+\):** the truncation at zero used in \(\psi_m\).
- **\(\tilde\sigma\):** the piecewise corrupted logistic link in the second numerical illustration.
- **Softmax:** the logistic link matched near the origin in that illustration.
- **\(r\):** the radius separating the two pieces of \(\tilde\sigma\).
- **\(\alpha=0.1\) in Figure 2:** the score-shrinkage parameter of the corrupted link.
- **Synthetic dataset:** observations sampled from the corrupted link in the numerical illustration.
- **\(nm\):** the total label count held constant in Figure 2(ii).
- **\(T=100\):** the number of trials for each aggregation count in Figure 2.
- **\(95\%\) error bar:** the reported uncertainty display over those trials.
- **Validation setting:** the comparison of binary classifiers discussed through Dorner and Hardt [11].
- **Data representation:** the representation for \(x\) jointly inferred with a predictor in the future-work discussion.
- **Sec., Section; Eq.; Fig.; Def.; Prop.; Thm.:** section, equation, figure, definition, proposition, and theorem references.
- **e.g.; i.e.; cf.; vs.; et al.:** example, restatement, comparison, contrast, and coauthor shorthand used in the paper.

# Some Robustness Properties of Label Cleaning

*Label cleaning provides robustness properties for surrogate-risk minimization and restricted predictive models.*

Chen Cheng

  
John Duchi

###### Abstract

*Aggregating noisy labels can give consistency guarantees that are unavailable from raw labels.*

Learning procedures that use aggregated labels—label information distilled from noisy responses—have robustness properties that are impossible without data cleaning, or label aggregation.

This robustness appears in several ways. Risk consistency concerns the standard machine-learning approach of minimizing a surrogate loss, typically convex, instead of the desired task loss, such as zero-one mis-classification error. Procedures using label aggregation obtain stronger consistency guarantees than are possible using raw labels.

Classical statistical settings with perfectly specified models suggest that incorporating all available information, including uncertainty in labels, is statistically efficient. Yet consistency fails for “standard” approaches as soon as a loss to be minimized is even slightly mis-specified. Procedures using aggregated information still converge to optimal classifiers. These results show how considering the full data analysis pipeline—from collection through model fitting to prediction—can yield a more robust methodology by refining noisy signals.

**Claims and evidence**
- Aggregation gives consistency unavailable from individual labels: Propositions 1–2 establish failure for convex ranking surrogates when \(k\geq3\), but consistency after aggregation when \(m\geq k\).
- Aggregation can restore consistency under model restrictions and mis-specification: Corollary 4.1 and Theorems 2–3 give convergence as \(m\to\infty\); Propositions 4–5 give failures without aggregation.

## 1 Introduction

*The paper asks whether label aggregation is necessary for consistency and how it improves consistency.*

Consider the data collection pipeline in supervised learning, where predictions are learned from features and labels. A simple description says that we collect pairs \((X_i,Y_i)_{i=1}^n\), with features \(X_i\), labels \(Y_i\), and \(n\) observations, then fit a model [17]. This hides the complexity of modern datasets [9, 22, 39]. They require substantial cleaning and filtering, often with crowdsourcing: collecting multiple labels and then denoising them.

The crowdsourcing community has studied this process intensively, especially for obtaining “gold standard” labels [8, 48, 47, 46, 37]. We take a complementary view. We study how data aggregation fundamentally and necessarily improves the consistency of fitted models.

The paper argues that label cleaning, or aggregating labels, provides robustness unavailable without aggregation. There are two aspects.

First, aggregation improves consistency of estimation. When minimizing a surrogate loss, such as multiclass logistic loss, instead of a task loss, such as zero-one error, aggregated-label procedures can achieve consistent and optimal prediction in the limit when procedures without aggregation cannot.

Second, aggregation can give consistent classifiers even in finite-dimensional statistical problems where standard methods fail.

Important work on surrogate risk consistency dates to the 2000s [52, 24, 40]. Bartlett, Jordan and McAuliffe [4] characterized when fitting a model with a convex surrogate is consistent for binary classification under zero-one error. Later work studied multi-label classification [50, 43, 16, 49, 1], ranking [13, 14, 36], structured prediction [34, 6, 32], ordinal regression [35], restricted hypothesis classes [2, 3, 28, 29], and general theory [40].

These analyses use the standard supervised-learning setting of pairs \((X,Y)\). This allows the full use of empirical-process theory, the statistical theory for such sampled observations [45, 5, 21, 4]. But the analyses do not address the aggregation machinery now common in dataset creation.

This raises the paper’s main questions. Do we need to aggregate at all? If consistency is possible without aggregation, perhaps we should rely on the mature theory for \((X,Y)\) pairs. If we do aggregate, does aggregation help consistency, and in what sense?

We give examples in ranking and in binary and multiclass classification with linear estimators. In these examples, estimators using only \((X,Y)\) pairs necessarily fail, while aggregation methods are consistent. We develop notions and theory for surrogate consistency with aggregation.

In fully nonparametric settings, where the hypothesis class contains all measurable functions, we show how the number of aggregated samples combines with noise conditions to improve consistency. Under weak conditions on the surrogate, aggregated-label losses admit approximate linear comparison inequalities: linear bounds relating excess task risk to excess surrogate risk. We also show classification results over restricted hypothesis classes \(\mathcal F\), the collections of predictors being fitted. Aggregation guarantees consistency in cases where it can fail without aggregation, unlike conventional theory that takes \(\mathcal F\) to contain all measurable functions.

**Claims and evidence**
- Aggregation can make otherwise impossible ranking consistency possible: Propositions 1–2, with \(k\geq3\) and \(m\geq k\), respectively.
- Weak surrogate assumptions permit approximately linear comparison: Theorem 1 gives slope \(\mathsf C_{\varphi,1}/16\) above its stated risk threshold.
- Aggregation can restore consistency over restricted classes: Corollary 4.1 and Theorem 2 give convergence as \(m\to\infty\).

## 2 Preliminaries

*The paper defines task and surrogate risks, then identifies the two consistency settings it studies.*

We first review classical surrogate risk minimization. Let \(\mathcal X\) be the input space and \(\mathcal Y\) the output space. The data \((X_i,Y_i)_{i=1}^n\in\mathcal X\times\mathcal Y\) are drawn independently and identically distributed (i.i.d.) from a distribution \(P\).

A scoring function \(f:\mathcal X\to\mathbb R^d\) maps input \(x\) to score \(s\in\mathbb R^d\), where \(d\geq1\) is the score dimension. A decoder \(\mathsf d:\mathbb R^d\to\mathcal Y\) maps scores to the final prediction
\(\widehat y=\mathsf d\circ f(x)\), where \(\circ\) denotes composition.

For a nonnegative task loss \(\ell:\mathcal Y\times\mathcal Y\to\mathbb R_+\) and hypothesis class \(\mathcal F\), the goal is to minimize the task risk
\[
R(f):=\mathbb E_P\!\left[\ell(\mathsf d\circ f(X),Y)\right].
\tag{1}
\]
Here \(\mathbb E_P\) denotes expectation under \(P\).

In binary classification, with two labels, \(d=1\), the decoder is the sign function \(\mathsf d(s)=\mathsf{sgn}(s)\), and
\[
\ell(y,y')=1\{yy'\leq0\},
\qquad
R(f)=\mathbb P(Yf(X)\leq0).
\]
The notation \(1\{\cdot\}\) is an indicator, and \(\mathbb P\) denotes probability.

Minimizing \(R(f)\) is difficult because the task loss may be nonsmooth, nonconvex, and uninformative. In particular, the \(0\)-\(1\) loss landscape is flat almost everywhere. This makes even practical methods such as first-order optimization impossible.

We consider a more sophisticated version of (1). Instead of depending only on the instantaneous label \(Y\), the loss may depend on the conditional label distribution \(P(Y\in\cdot\mid X)\):
\[
R(f)\coloneqq
\mathbb E_P\!\left[
\ell(\mathsf d\circ f(X),P(\cdot\mid X))
\right].
\tag{2}
\]
Its minimizers frequently coincide with the original problem (2), but this formulation allows more sophistication. For example, in multiclass classification, \(Y\in\{1,\ldots,k\}\), where \(k\) is the number of classes. Taking
\[
\ell(\widehat y,P)=\sum_yP(Y=y)1\{\widehat y\neq y\}
\]
makes the risk equal to the standard \(0\)-\(1\) error rate.

Rather than minimize (2) directly, we use an easier-to-optimize surrogate
\(\varphi:\mathbb R^d\times\mathcal Y\to\mathbb R\). Its surrogate risk is
\[
R_\varphi(f):=\mathbb E_P[\varphi(f(X),Y)].
\]

This requires a consistency relationship with task problem (2). We study two settings:

(i) **Fisher consistency:** the classical setting where \(\mathcal F\) contains all Borel measurable functions.

(ii) **Statistical settings with restricted models:** \(\mathcal F\) is parametric but may be mis-specified.

Our main message is that aggregation improves consistency in both settings.

### 2.1 Label aggregation

*Aggregation replaces an individual label with a function of a richer response, including repeated labels or neighboring examples.*

Replace the output \(Y\) by an abstract response \(Z\in\mathcal Z\), where \(\mathcal Z\) is its response space. If we collect \(m\) noisy labels for each input \(X\), then
\[
Z=(Y_1,\ldots,Y_m)\in\mathcal Y^m.
\]
Mao et al. [28] consider a similar multilabeled data structure for multilabel losses. Our focus is the aggregation pipeline.

Let \(\mathcal A\) be an aggregation space and \(A:\mathcal Z\to\mathcal A\) an aggregation function, such as majority vote. Let
\(\varphi:\mathbb R^d\times\mathcal A\to\mathbb R_+\) be a surrogate on that space. Define the aggregated surrogate risk
\[
R_{\varphi,A}(f):=
\mathbb E[\varphi(f(X),A(Z))].
\tag{3}
\]
We ask when minimizing (3) suffices to minimize task risk (2).

**Example 1 (Majority vote):**

In the repeated-sampling regime, we collect conditionally independent labels:
\[
Z=(Y_1,\cdots,Y_m),
\qquad
Y_i\mid X=x\stackrel{\rm iid}{\sim}P_{Y\mid X=x}.
\]
Define \(A_m(Z)\) as the empirical loss minimizer:
\[
A(Z)=A(\{Y_1,\ldots,Y_m\})
=\argmin_{y\in\mathcal Y}\sum_{l=1}^m\ell(y,Y_l).
\]
Here \(\argmin\) denotes minimizing arguments, and \(l\) indexes labels.

For \(\ell(\widehat y,y)=1\{y\neq\widehat y\}\), this is exactly majority vote. The general form permits more abstract procedures. \(\Diamond\)

We can also roughly capture \(K\)-nearest-neighbor aggregation.

**Example 2 (\(K\)-nearest neighbors):**

Suppose an input \(X\) comes with label \(Y\) and an additional i.i.d. sample
\[
(X_i,Y_i)_{i=1}^m\stackrel{\rm iid}{\sim}P.
\]
Here \(m\) is the number of additional examples, and
\[
Z=(Y,(X_i,Y_i)_{i=1}^m).
\]
Let \(\operatorname{dist}:\mathcal X\times\mathcal X\to\mathbb R_+\) be a distance metric. Order the additional inputs by distance from \(X\):
\[
\operatorname{dist}(X,X_{(1)})
\leq\cdots\leq
\operatorname{dist}(X,X_{(m)}),
\]
and set \(X_{(0)}=X\). For \(K\geq0\), aggregate the \(K\) nearest neighbors, for example by
\[
A_{m,K}(Z)\coloneqq
\argmin_{y\in\mathcal Y}\sum_{l=0}^K\ell(y,Y_{(l)}).
\]
The labels \(Y_{(l)}\) correspond to the distance-ordered inputs.

Appendix C uses the coming results to move beyond this population-level setting and study aggregation from one sample \((X_i,Y_i)_{i=1}^n\). \(\Diamond\)

## 3 Surrogate consistency

*Calibration functions characterize classical consistency, but individual labels do not always permit a consistent surrogate.*

The standard framework [40] takes \(\mathcal F\) to contain all Borel measurable functions \(f:\mathcal X\to\mathbb R^d\). Define the conditional task and surrogate risks, for score \(s\), by
\[
R(s\mid x)\coloneqq
\ell(\mathsf d\circ s,P(Y\in\cdot\mid X=x)),
\qquad
R_\varphi(s\mid x):=
\mathbb E[\varphi(s,Y)\mid X=x].
\]
Their pointwise excess risks—the amounts above their conditional infima—are
\[
\delta_\ell(s,x)\coloneqq
R(s\mid x)-\inf_{s'\in\mathbb R^d}R(s'\mid x),
\]
\[
\delta_\varphi(s,x)\coloneqq
R_\varphi(s\mid x)-\inf_{s'\in\mathbb R^d}R_\varphi(s'\mid x).
\]
Define the minimal risks
\[
R^\star:=\inf_{f\in\mathcal F}R(f),
\qquad
R_\varphi^\star:=\inf_{f\in\mathcal F}R_\varphi(f).
\]

Following [40, 4, 52], consistency requires at least (i), and preferably the stronger quantitative property (ii), for every data distribution \(P\):

(i) **Fisher consistency.** For every sequence \(f_n\in\mathcal F\),
\[
R_\varphi(f_n)\to R_\varphi^\star
\quad\Longrightarrow\quad
R(f_n)\to R^\star.
\]

(ii) **Uniform comparison inequality.** There is a nondecreasing function
\(\psi:\mathbb R_+\to\mathbb R_+\) such that, for every \(f\in\mathcal F\),
\[
\psi(R(f)-R^\star)\leq R_\varphi(f)-R_\varphi^\star,
\]
and \(\psi(\epsilon)>0\) for every \(\epsilon>0\).

For binary classification with a convex margin-based surrogate, these notions coincide [4].

The stronger uniform guarantee need not always hold. Calibration functions provide a canonical construction from excess risks:
\[
\psi(\epsilon,x)\coloneqq
\inf_{s\in\mathbb R^d}
\{\delta_\varphi(s,x)\mid\delta_\ell(s,x)\geq\epsilon\},
\qquad
\overline\psi(\epsilon)\coloneqq
\inf_{x\in\mathcal X}\psi(\epsilon,x).
\]
Here \(\epsilon\) is an excess-task-risk level. The first function is pointwise; the second is uniform.

The following result follows from these functions [51, Prop. 25; 40, Thm. 2.8 and Lemma 2.9].

###### Corollary 3.1.

*Positive calibration characterizes Fisher consistency and yields a comparison inequality through biconjugation.*

The surrogate \(\varphi\) is Fisher consistent (i) for \(\ell\) if and only if
\(\psi(\epsilon,x)>0\) for every \(x\in\mathcal X\) and \(\epsilon>0\).

Let \(\psi\) be the Fenchel biconjugate of \(\overline\psi\), the operation used here to construct the comparison function. Then
\[
\overline\psi(\epsilon)>0
\quad\Longleftrightarrow\quad
\psi(\epsilon)>0,
\]
and, for all measurable \(f\),
\[
\psi(R(f)-R^\star)\leq R_\varphi(f)-R_\varphi^\star.
\]

For general task problem (2), we would like at least a Fisher-consistent surrogate. Minimizing \(\mathbb E[\varphi(f(X),Y)]\) would then minimize \(R(f)\). Such a result would let us use paired observations \((X,Y)\), rather than tuples \((X,Y_1,\ldots,Y_m)\), together with empirical-process theory and related tools [45, 5, 21, 4]. Data collection would also require only single pairs.

Unfortunately, these results are generally impossible. The next example shows why a theory of aggregation, developed in Section 3.2, is necessary.

### 3.1 Fisher consistency failure without label aggregation: ranking

*Convex surrogates fail for the stated ranking task with individual comparisons, but an aggregated squared loss is consistent when \(m\geq k\).*

Consider ranking \(k\) items from pairwise comparisons [20, 15, 14, 30]. The label space is
\[
\mathcal Y=\{(i,j):i\neq j,\ 1\leq i,j\leq k\},
\]
where \((i,j)\) means that item \(i\) ranks above item \(j\).

For each input \(x\), population rank aggregation transforms
\[
p_{ij}=P(Y=(i,j)\mid x)
\]
into a ranking. We use a simple comparison-based scheme [20, cf.]. Similar negative results hold for more sophisticated schemes.

Define the normalized transition matrix \(C_x\in\mathbb R_+^{k\times k}\) by
\[
(C_x)_{ii}=0,
\qquad
(C_x)_{ij}
=\frac{p_{ij}}{\sum_{l\neq j}p_{lj}}
\quad(i\neq j).
\]
Use \(0/0=1/(k-1)\), so \(C_x\) is stochastic:
\[
C_x^T\mathbf1=\mathbf1.
\]
Here \(\mathbf1\) is the all-one vector, and \(T\) denotes transpose.

Rank items by \(C_x\mathbf1\in\mathbb R_+^k\), which measures how often each item is preferred to others. One can instead use higher powers \(C_x^p\mathbf1\), where \(p\) is the power, or Perron vectors [20]. Similar results hold in those cases.

Incorporating the decoder into the loss, define
\[
\ell(s,C)\coloneqq
\max_{i<j}
1\!\left\{
(s_i-s_j)(e_i-e_j)^TC\mathbf1\leq0,\
(e_i-e_j)^TC\mathbf1\neq0
\right\},
\]
where \(e_i\) are the coordinate vectors. This penalizes mis-ordered scores. The population task risk is
\[
R(f)\coloneqq
\mathbb P\!\left(
f(X)\text{ and }C_X\mathbf1\text{ order differently}
\right).
\tag{4}
\]

Now consider a convex surrogate
\(\varphi:\mathbb R^k\times\mathcal Y\to\mathbb R\).
Restrict scores to \(s^T\mathbf1=0\). This minor restriction is familiar from multiclass classification [51, 43] and is natural because ranking requires only the order of score entries.

No convex Fisher-consistent surrogate exists for (4); see Appendix A.1.

###### Proposition 1.

*For \(k\geq3\), convexity in the score argument rules out Fisher consistency for this ranking risk.*

For ranking risk (4) over \(k\geq3\) outcomes, any
\(\varphi:\mathbb R^k\times\mathcal Y\to\mathbb R\) convex in its first argument is not Fisher consistent.

Aggregation nevertheless gives consistency as soon as the number \(m\) of collected comparisons satisfies \(m\geq k\). The method regresses predicted scores \(f(x)\) on frequencies of label orderings.

Assume multiple independent pairwise comparisons \(Z=(Y_1,\ldots,Y_m)\) conditional on \(X\). Let
\[
m_{ij}=\sum_{y\in\mathcal Y^m}1\{y=(i,j)\},
\qquad
m_j=\sum_{i=1}^k m_{ij}.
\]
Define
\[
A(Z)=
\begin{cases}
\star,
&\text{if }m_j=0\text{ for some }j\text{ in }[k],\\[2mm]
\left(
\frac{m_{i1}}{m_1}+\frac{m_{i2}}{m_2}+\cdots+
\frac{m_{ik}}{m_k}
\right)_{i\in[k]},
&\text{otherwise, i.e. if }m_j>0\text{ for all }j\in[k].
\end{cases}
\]
Here \([k]=\{1,\ldots,k\}\), and \(\star\) marks a response with a missing column count.

Regressing directly on \(A(Z)\) when \(A(Z)\neq\star\) is consistent. Appendix A.2 proves the next result.

###### Proposition 2.

*Squared-loss regression on the aggregated response is Fisher consistent when \(m\geq k\).*

For score and response vectors \(s,q\in\mathbb R^k\), define the squared-loss surrogate
\[
\varphi(s,q)=\|s-q\|_2^2,
\qquad
\varphi(s,\star)=0.
\]
Then, if \(m\geq k\), \(\varphi\) is Fisher consistent for ranking risk (4). Here \(\|\cdot\|_2\) is the norm used in the squared loss.

The difference between Propositions 1 and 2 is the response-space expansion. Aggregation changes the response space from \(\mathcal Y\) to \(\mathbb R^k\).

**Claims and evidence**
- Individual comparisons admit no convex Fisher-consistent surrogate: Proposition 1, \(k\geq3\), task risk (4).
- Aggregated comparisons admit a consistent squared loss: Proposition 2, \(m\geq k\), \(\varphi(s,q)=\|s-q\|_2^2\) and \(\varphi(s,\star)=0\).
- The response space expands under aggregation: the displayed \(A(Z)\) takes vector values in \(\mathbb R^k\) when every \(m_j>0\).

### 3.2 Label aggregation obtains stronger surrogate consistency

*Aggregation preserves classical calibration theory and can also make otherwise inconsistent surrogates consistent.*

The ranking example suggests broader benefits of aggregation. We give two types of results. First, we extend standard surrogate-risk consistency to aggregation. Second, we show how aggregation can “upgrade” nominally inconsistent losses into consistent ones, as Proposition 2 suggests.

#### 3.2.1 Basic extensions of surrogate consistency

*Aggregated risks have calibration functions and consistency guarantees analogous to the classical ones.*

For any aggregation method \(A:\mathcal Z\to\mathcal A\), define the conditional aggregated surrogate risk
\[
R_{\varphi,A}(s\mid x)\coloneqq
\mathbb E[\varphi(s,A(Z))\mid X=x].
\]
Its pointwise excess risk is
\[
\delta_{\varphi,A}(s,x):=
R_{\varphi,A}(s\mid x)
-\inf_{s\in\mathbb R^d}R_{\varphi,A}(s\mid x).
\]
Define the pointwise and uniform calibration functions:
\[
\overline\psi_A(\epsilon,x)\coloneqq
\inf_{s\in\mathbb R^d}
\{\delta_{\varphi,A}(s,x)\mid\delta_\ell(s,x)\geq\epsilon\},
\qquad
\overline\psi_A(\epsilon)\coloneqq
\inf_{s\in\mathbb R^d}\overline\psi_A(\epsilon,x).
\tag{5}
\]

A result like Corollary 3.1 follows under appropriate measurability conditions. We leave those conditions tacit because they are not central here. The next result is essentially a corollary of Steinwart [40, Thm. 2.8]. Appendix B.1 includes a proof.

###### Proposition 3.

*An integrable task-risk bound permits the aggregated calibration characterization and comparison inequality.*

Assume there is a function \(b:\mathcal X\to\mathbb R_+\) bounding pointwise excess task risk:
\[
\int b(x)\,dP(x)<\infty,
\qquad
\delta_\ell(f(x),x)\leq b(x).
\]
Then \(\varphi\) is Fisher consistent (i) for task risk (2) if and only if
\[
\overline\psi_A(\epsilon,x)>0
\quad\text{for every }x\in\mathcal X,\ \epsilon>0.
\]
If
\[
\psi_A=(\overline\psi_A)^{**},
\]
where \(^{**}\) denotes the Fenchel biconjugate, then
\[
\psi_A(R(f)-R^\star)
\leq R_{\varphi,A}(f)-R_{\varphi,A}^\star.
\]
The quantity \(R_{\varphi,A}^\star\) is the infimal aggregated surrogate risk.

Aggregation therefore preserves classical consistency guarantees. This result alone does not give improved consistency. We turn to improvement next.

**Claims and evidence**
- Classical calibration guarantees extend to aggregation: Proposition 3, under \(\int b(x)\,dP(x)<\infty\) and \(\delta_\ell(f(x),x)\leq b(x)\).
- The resulting comparison is quantitative: Proposition 3 gives \(\psi_A(R(f)-R^\star)\leq R_{\varphi,A}(f)-R_{\varphi,A}^\star\).

#### 3.2.2 Identifying surrogates and consistency

*An identifying surrogate combined with generalized majority vote becomes consistent as the aggregation count grows.*

Under essentially minimal surrogate conditions, a generic aggregation strategy guarantees consistency asymptotically in the number of observations \(y\). Here the task minimizes \(\ell(f(x),y)\):
\[
R(f)=\mathbb E[\ell(f(X),Y)].
\]
Assume the label-space cardinality, or number of elements, is
\(\operatorname{card}(\mathcal Y)=k<\infty\).[^1] We impose a minimal identifiability assumption.

[^1]: We emphasize that the example discussed in Sec. 3.1 belongs to this regime, where \(k\) denotes the ranking length and \(\operatorname{card}(\mathcal Y)=k(k-1)\). With a slight abuse of notation, we will henceforth use \(k\) to denote the cardinality of the label space.

###### Definition 3.1 (Identifying surrogate).

*An identifying surrogate separates each class from incorrect decoded predictions using finite constants.*

A surrogate \(\varphi:\mathbb R^d\times\mathcal A\to\mathbb R\) is
\((\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})\)-identifying for \(\mathcal Y\), where
\[
0<\mathsf C_{\varphi,1}\leq\mathsf C_{\varphi,2}<\infty,
\]
if there are aggregation parameters
\(\{a_y\}_{y\in\mathcal Y}\subset\mathcal A\) and score vectors
\(\{s_y\}_{y\in\mathcal Y}\subset\mathbb R^d\), with
\(\mathsf d(s_y)=y\), such that, for every \(y\neq y'\),
\[
\varphi(s_y,a_y)+\mathsf C_{\varphi,1}
\leq
\inf_{\mathsf d\circ s\neq y}\varphi(s,a_y),
\tag{6a}
\]
\[
\varphi(s_y,a_{y'})-\mathsf C_{\varphi,2}
\leq
\inf_{s\in\mathbb R^d}\varphi(s,a_{y'}).
\tag{6b}
\]

Inequality (6a) says that each class has a parameter \(a_y\) for which minimizing \(\varphi(\cdot,a_y)\) identifies \(y\). A finite \(\mathsf C_{\varphi,2}\) exists in (6b) if \(\varphi(\cdot,a)\) has a finite lower bound.

The definition requires neither convexity of \(\varphi(\cdot,a)\) nor consistency without aggregation, when \(\mathcal A=\mathcal Y\) and \(Z=Y\). It also does not impose a requirement on the decoder such as coincidence with the pointwise \(\argmax\), the maximizing-argument operator.

**Example 3:**

For binary hinge loss,
\[
\varphi(s,a)=\max\{1-sa,0\},
\qquad
\mathcal A=\mathcal Y=\{\pm1\},
\]
take \(a_y=s_y=y\) for \(y\in\{-1,1\}\). Then
\[
\varphi(s_1,a_1)=\varphi(s_{-1},a_{-1})=0,
\qquad
\inf_{sa\leq0}\varphi(s,a)=1.
\]
Also,
\[
\varphi(s_1,a_{-1})=\varphi(s_{-1},a_1)=2.
\]
Thus hinge loss is \((1,2)\)-identifying. \(\Diamond\)

Section 3.3 gives further examples. Given identifying parameters \(\{a_y\}_{y\in\mathcal Y}\), consider generalized majority vote:
\[
A_m(y_1,\ldots,y_m)\coloneqq a_{\hat y},
\qquad
\hat y=\argmin_{y\in\mathcal Y}\sum_{i=1}^m\ell(y,y_i).
\tag{7}
\]
Break ties arbitrarily.

Since \(\mathcal Y\) is finite, for i.i.d. \(Y_i\) there is necessarily a random finite threshold \(M<\infty\) such that, when \(m\geq M\),
\[
\argmin_{y\in\mathcal Y}
\left\{\sum_{i=1}^m\ell(y,Y_i)\right\}
\subset
y^\star(x)\coloneqq
\argmin_{y\in\mathcal Y}\mathbb E[\ell(y,Y)\mid X=x].
\]
Here \(y^\star(x)\) is the set of conditionally optimal labels.

This suggests consistency as \(m\to\infty\). The following corollary of the coming results gives that guarantee.

###### Corollary 3.2.

*Vanishing aggregated surrogate regret implies vanishing task regret when \(m\) grows without bound.*

Let \(m=m(n)\to\infty\), and let \(\varphi\) be identifying under Definition 3.1. Then
\[
R_{\varphi,A_m}(f_n)-R_{\varphi,A_m}^\star\to0
\quad\Longrightarrow\quad
R(f_n)-R^\star\to0.
\]

**Claims and evidence**
- Hinge loss meets the identification assumption: Example 3 gives own-label loss \(0\), incorrect-decoding infimum \(1\), and cross-label loss \(2\), hence constants \((1,2)\).
- Generalized majority vote eventually selects only optimal labels: the displayed inclusion holds for \(m\geq M\), with random \(M<\infty\).
- Identifying surrogates become asymptotically consistent: Corollary 3.2, \(m(n)\to\infty\) and aggregated surrogate excess risk \(\to0\).

#### 3.2.3 Identifying surrogates and consistency amplification

*Noise conditions quantify how aggregation yields approximately linear comparison inequalities.*

With low label noise, aggregation strategy (7) improves the comparison inequality
\[
\psi(R(f)-R^\star)\leq R_\varphi(f)-R^\star
\]
by making \(\psi\) linear over some range of \(\epsilon>0\). Linear growth is the strongest comparison possible [34, 32].

Strict comparison inequalities, such as Proposition 3, can be too narrow because inconsistent surrogates can still be practically convenient [23, 34, 32]. Following Osokin et al. [34], we use level-\((\xi,\zeta)\) consistency. Here \(\xi\) is a risk threshold and \(\zeta\) is a linear slope. The comparison function need grow linearly only for \(\epsilon\geq\xi\), capturing a “good enough” risk.

###### Definition 3.2.

*Level-\((\xi,\zeta)\) consistency requires a linear comparison only above excess-risk threshold \(\xi\).*

A surrogate \(\varphi\) and aggregator \(A\) yield level-\((\xi,\zeta)\) consistency if a function \(\psi\) satisfies
\[
\psi(\epsilon)\geq\zeta\epsilon
\quad\text{for }\epsilon\geq\xi,
\qquad
\psi(R(f)-R^\star)\leq
R_{\varphi,A}(f)-R_{\varphi,A}^\star.
\]

Under minimal assumptions, aggregation (7) achieves level-\((o_m(1),\zeta)\) consistency even for a Fisher-inconsistent surrogate. The notation \(o_m(1)\) denotes a quantity tending to zero as \(m\) grows.

We quantify noise by adapting the Mammen–Tsybakov conditions [25]; see also Bartlett et al. [4]. Define
\[
\Delta(x)\coloneqq
\min_{\mathsf d(s)\notin y^\star(x)}\delta_\ell(s,x),
\tag{8}
\]
the smallest conditional excess risk of an incorrect prediction. For binary classification,
\[
\Delta(x)=|2P(Y=1\mid X=x)-1|.
\]
Consistent estimation is expected to be harder when \(\Delta(x)\) is closer to \(0\).

With a positive constant \(\mathsf C_{\mathsf{MT}}\), which may change, the conditions are
\[
\mathbb P(\mathsf d\circ f\neq\mathsf d\circ f^\star)
\leq
\mathsf C_{\mathsf{MT}}(R(f)-R^\star)^\alpha
\quad\text{for all measurable }f,
\tag{Nα}
\]
and
\[
\mathbb P(\Delta(X)\leq\epsilon)
\leq(\mathsf C_{\mathsf{MT}}\epsilon)^\beta
\quad\text{for }\epsilon>0.
\tag{Mβ}
\]
Condition (Nα) has noise exponent \(\alpha\). Here
\(\alpha\in[0,1]\) and \(\beta\in[0,\infty]\). Both conditions hold trivially with \(\alpha=\beta=0\). As in binary classification [4, Thm. 3], they are equivalent under
\[
\beta=\frac{\alpha}{1-\alpha}.
\]
See Appendix B.2.

Mao et al. [29] use the same noise statistic for multiclass \(H\)-consistency: bounds connecting excess target risk and surrogate regret within a hypothesis class \(H\). Their structural assumptions on the surrogate and class permit a global convex lower bound. Our generic setting has minimal structural constraints, requiring the noise condition number
\[
\kappa(x)\coloneqq
\frac{\max_{\mathsf d(s)\neq y^\star(x)}\delta_\ell(s,x)}
{\min_{\mathsf d(s)\neq y^\star(x)}\delta_\ell(s,x)}.
\tag{9}
\]
It connects \(\Delta(x)\) to pointwise excess risk:
\[
\Delta(x)\geq\frac{\delta_\ell(s,x)}{\kappa(x)}
\quad\text{whenever }\mathsf d(s)\neq y^\star(x).
\]
For binary classification, \(\kappa(x)=1\) provided
\(\mathbb P(\Delta(X)>0)=1\).

These quantities let us analyze generalized majority vote with any identifying surrogate. Define
\[
e_m(t)\coloneqq
t\sqrt{\frac{2}{m}
\log\left(
\frac{4k(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})}
{\mathsf C_{\varphi,1}}
\right)}.
\tag{10}
\]
This error function roughly says that, when \(\kappa(x)=t\), majority vote is likely correct if \(m\) is large enough that \(e_m(t)\ll1\).

The next theorem gives a near-linear calibration function. Appendix B.3 proves it.

###### Theorem 1.

*An identifying surrogate has a linear excess-risk comparison above a threshold determined by aggregation, noise, and the condition-number tail.*

Let \(\varphi\) be
\((\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})\)-identifying with parameters \(\{a_y\}_{y\in\mathcal Y}\), and let \(A_m\) be aggregator (7). Assume \(0\leq\ell\leq1\) and condition (Nα) with \(\alpha\in[0,1]\).

For any condition-number threshold \(M>0\) and \(f\in\mathcal F\) satisfying
\[
R(f)-R^\star\geq
2\mathbb P(\kappa(X)>M)
+(4\mathsf C_{\mathsf{MT}}e_m(M))^{\frac1{1-\alpha}},
\]
we have
\[
R(f)-R^\star
\leq
\frac{16}{\mathsf C_{\varphi,1}}
\left(R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star\right).
\]

Thus the theorem gives level-\((\xi,\zeta)\) consistency with
\[
\xi=
2\mathbb P(\kappa(X)>M)
+(4\mathsf C_{\mathsf{MT}}e_m(M))^{\frac1{1-\alpha}},
\qquad
\zeta=\frac{\mathsf C_{\varphi,1}}{16}.
\]

It also proves Corollary 3.2. Define
\[
\xi_m\coloneqq
\inf_M
\left\{
2\mathbb P(\kappa(X)>M)
+(4\mathsf C_{\mathsf{MT}}e_m(M))^{\frac1{1-\alpha}}
\right\}.
\]
Then \(\xi_m\to0\) as \(m\to\infty\): the tail probability tends to \(0\) as \(M\uparrow\infty\), while \(e_m(M)\to0\) for each fixed \(M\). Taking \(\alpha=0\) gives Corollary 3.2.

Theorem 1 is somewhat gross because the identification conditions are weak. A tighter connection between \(\ell\) and \(\varphi\) might give stronger bounds by making majority vote more likely to be correct, or correct enough for \(\varphi\). For example, a surrogate nearly consistent without aggregation might permit stronger bounds in \(m\). We do not pursue this.

For a more concrete bound, optimize over \(M\) using
\[
\kappa(x)\leq1/\Delta(x).
\]
Taking \(M=1\) when \(\operatorname{card}(\mathcal Y)=k=2\), and optimizing \(M\) for \(k\geq3\), we may lower bound the threshold \(\xi_{m,k}\) in Theorem 1’s level-\((\xi_{m,k},\zeta)\) consistency by setting
\[
\xi_{m,k}\coloneqq
\begin{cases}
\displaystyle
\left(
\frac{32\mathsf C_{\mathsf{MT}}^2}{m}
\log\left(
\frac{8(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})}
{\mathsf C_{\varphi,1}}
\right)
\right)^{\frac1{2(1-\alpha)}},
&\text{if }k=2,\\[4mm]
\displaystyle
4\cdot
\left(
\frac{32\mathsf C_{\mathsf{MT}}^4}{m}
\log\left(
\frac{4k(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})}
{\mathsf C_{\varphi,1}}
\right)
\right)^{\frac{\alpha}{2(1-\alpha^2)}},
&\text{otherwise}.
\end{cases}
\]
The algebra is in Appendix B.4.

###### Corollary 3.3.

*The explicit threshold \(\xi_{m,k}\) suffices for the same linear comparison.*

Under Theorem 1’s conditions, any \(f\) with
\(R(f)-R^\star\geq\xi_{m,k}\) satisfies
\[
R(f)-R^\star
\leq
\frac{16}{\mathsf C_{\varphi,1}}
\left(R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star\right).
\]

Corollaries 3.2–3.3 support the robustness of cleaning: minimal surrogate assumptions still permit consistency with aggregation.

As \(\alpha\) approaches \(1\), the aggregation count required for a fixed \(f\) decreases. In particular, if \(\alpha=1\), then whenever
\[
m\geq
32\max\{\mathsf C_{\mathsf{MT}}^2,\mathsf C_{\mathsf{MT}}^4\}
\log\left(
\frac{4k(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})}
{\mathsf C_{\varphi,1}}
\right)
=O(\log k),
\]
we have \(\xi_{m,k}=0\). Here \(O(\log k)\) is the stated order of aggregation count. This gives uniform comparison (ii) with a linear function. An “easier,” lower-noise problem needs less aggregation for stronger consistency.

**Claims and evidence**
- The two noise conditions are equivalent: (Nα), (Mβ), and \(\beta=\alpha/(1-\alpha)\); both hold at \(\alpha=\beta=0\).
- Binary classification has condition number \(1\): (9), assuming \(\mathbb P(\Delta(X)>0)=1\).
- Aggregation gives a near-linear comparison: Theorem 1 gives factor \(16/\mathsf C_{\varphi,1}\) above \(2\mathbb P(\kappa(X)>M)+(4\mathsf C_{\mathsf{MT}}e_m(M))^{1/(1-\alpha)}\).
- The threshold vanishes asymptotically: the displayed \(\xi_m\to0\) argument gives Corollary 3.2 at \(\alpha=0\).
- Explicit thresholds give the same comparison: Corollary 3.3 and the two displayed rows defining \(\xi_{m,k}\), for \(k=2\) and otherwise.
- At \(\alpha=1\), finite aggregation gives uniform linear comparison: the displayed bound \(m\geq32\max\{\mathsf C_{\mathsf{MT}}^2,\mathsf C_{\mathsf{MT}}^4\}\log(4k(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})/\mathsf C_{\varphi,1})=O(\log k)\) gives \(\xi_{m,k}=0\).

**Claims and evidence for Section 3.2**
- Calibration extends to aggregation: Proposition 3 and (5).
- Identification plus aggregation yields asymptotic consistency: Corollary 3.2, \(m(n)\to\infty\).
- Noise conditions yield approximately linear comparison: Theorem 1 and Corollary 3.3, with factor \(16/\mathsf C_{\varphi,1}\) and the thresholds stated above.

### 3.3 Surrogate consistency examples with majority vote

*Binary classification, bipartite matching, and structured prediction instantiate the aggregation bounds.*

We give examples applying Theorem 1 and Corollary 3.3. Throughout, assume \(P\) has noise exponent \(\alpha\in[0,1]\). This loses no generality because (Nα) always holds with \(\alpha=0\). Appendix B.5 contains all proofs for this section.

#### 3.3.1 Binary classification with a nonsmooth surrogate

*A convex binary surrogate satisfying the stated subgradient conditions admits identification constants and an aggregated comparison bound.*

Consider a margin-based surrogate
\[
\varphi(f(x),y)=\phi(yf(x)),
\]
where \(\phi\) is convex. Bartlett et al. [4] show consistency if and only if
\(\phi'(0)<0\), where the prime denotes the derivative.

We give a somewhat trivial example of robustness: even if \(\phi\) is inconsistent without aggregation, aggregation can make it consistent. One would never use such a surrogate; this is a thought experiment.

Assume
\[
\partial\phi(0)\subset(-\infty,0),
\qquad
\lim_{t\to\infty}\phi(t)=0,
\]
with \(\phi\) convex. Here \(\partial\phi(0)\) is the subgradient set at \(0\).

###### Lemma 3.1.

*Every positive choice of \(\delta\) gives the stated feasibility constants.*

For any \(\delta>0\), \(\varphi\) is
\((\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})\)-feasible with
\[
\mathsf C_{\varphi,1}=\phi(0)-\phi(\delta)>0,
\qquad
\mathsf C_{\varphi,2}=\phi(-\delta).
\]

Apply Corollary 3.3 with \(k=2\). If \(f:\mathcal X\to\mathbb R\) satisfies
\[
R(f)-R^\star\geq
\left(
\frac{32\mathsf C_{\mathsf{MT}}^2}{m}
\log\left(
\frac{8(\phi(-\delta)+\phi(0)-\phi(\delta))}
{\phi(0)-\phi(\delta)}
\right)
\right)^{\frac1{2(1-\alpha)}},
\]
then
\[
R(f)-R^\star
\leq
\frac{16}{\phi(0)-\phi(\delta)}
\left(R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star\right).
\]

**Claims and evidence**
- The surrogate has positive feasibility gap: Lemma 3.1 gives \(\mathsf C_{\varphi,1}=\phi(0)-\phi(\delta)>0\) and \(\mathsf C_{\varphi,2}=\phi(-\delta)\), for every \(\delta>0\).
- Aggregation gives a linear comparison above the stated binary threshold: the two displayed inequalities, with coefficient \(16/(\phi(0)-\phi(\delta))\), obtained from Corollary 3.3 at \(k=2\).

#### 3.3.2 Bipartite matching

*The structured hinge loss for bipartite matching is identifying and therefore supports aggregation-based consistency.*

In structured prediction [33], an embedding map
\(v:\mathcal Y\to\mathbb R^d\) encodes structural information about outputs in a typically large structured space \(\mathcal Y\).

Use decoder
\[
\mathsf d(s)=\argmax_{y\in\mathcal Y}\langle s,v(y)\rangle,
\]
where \(\langle\cdot,\cdot\rangle\) is the inner product. For nonnegative loss \(\ell\) with \(\ell(y,y)=0\), the maximum-margin surrogate, also called the generalized hinge loss [42, 44, 19], is
\[
\varphi(s,y)=
\max_{\hat y\in\mathcal Y}
\left(
\ell(\hat y,y)+\langle v(\hat y)-v(y),s\rangle
\right).
\tag{11}
\]
This loss is typically inconsistent except in some low-noise cases [34, 32].

First consider bipartite matching. A matching is a graph \(G=(V,E)\), with vertex set
\[
V=V_1\cup V_2,\qquad
V_1=\{1,\ldots,N\},\qquad
V_2=\{N+1,\ldots,2N\}.
\]
The \(N\) edges in \(E\) each connect exactly one unique node from each side. The label space contains all bipartite matchings between \(V_1\) and \(V_2\), so
\[
k=\operatorname{card}(\mathcal Y)=N!.
\]

For graph \(G\), use the edge-indexing embedding
\[
v(G)\coloneqq
\left(1\{(u,v)\in E\}\right)_{u\in V_1,v\in V_2}
\in\mathbb R^{N^2},
\]
where \(u,v\) index edge endpoints. Thus \(d=N^2\).

The task loss counts mistaken edges:
\[
\ell(y_1,y_2)\coloneqq
\frac1{2N}\|v(y_1)-v(y_2)\|_1
=
\frac1{2N}\|v(y_1)-v(y_2)\|_2^2.
\]
Here \(\|\cdot\|_1\) is the norm used in the edge-counting expression.

###### Lemma 3.2.

*For matching on \(2N\) vertices, the identification constants are \(1/N\) and \(2\).*

For bipartite matching on \(2N\) vertices, the structured hinge surrogate (11) is
\((\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})\)-identifying under Definition 3.1, with
\[
\mathsf C_{\varphi,1}=\frac1N,
\qquad
\mathsf C_{\varphi,2}=2.
\]

Even though \(k=N!\), aggregation-based methods can therefore yield consistency through the structured hinge loss once \(m\) exceeds \(O(N\log N)\).

Substituting these constants into Corollary 3.3 for \(k\geq3\), every measurable \(f:\mathcal X\to\mathbb R^d\) satisfying
\[
R(f)-R^\star\geq
4\cdot
\left(
\frac{32\mathsf C_{\mathsf{MT}}^4}{m}
\log(4k(2N+1))
\right)^{\frac{\alpha}{2(1-\alpha^2)}}
\]
obeys
\[
R(f)-R^\star
\leq
16N\left(R_{\varphi,A_n}(f)-R_{\varphi,A_n}^\star\right).
\]

**Claims and evidence**
- The output space has \(N!\) elements and embedding dimension \(N^2\): the displayed matching cardinality and embedding.
- The structured hinge is identifying: Lemma 3.2, \(\mathsf C_{\varphi,1}=1/N\), \(\mathsf C_{\varphi,2}=2\).
- Aggregation can yield consistency beyond \(O(N\log N)\) labels: Lemma 3.2 and Corollary 3.3; the displayed specialization has threshold \(4(32\mathsf C_{\mathsf{MT}}^4\log(4k(2N+1))/m)^{\alpha/(2(1-\alpha^2))}\) and comparison factor \(16N\).

#### 3.3.3 Structured prediction

*An embedding-based identifiability gap gives constants for the general structured hinge loss.*

Return to general structured prediction. Suppose the decoder can select every class: for each \(y\in\mathcal Y\), the set of scores strictly selecting \(y\),
\[
\mathcal S(y)\coloneqq
\{s:\langle v(y),s\rangle>
\langle v(\hat y),s\rangle
\text{ for }\hat y\neq y\},
\]
is nonempty.

Define the identifiability gap
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
The outputs \(y_+,y_-\) index alternatives in this expression.

###### Lemma 3.3.

*The structured hinge has constants determined by the smallest task loss over distinct labels and the largest identifiability gap.*

For structured prediction, max-margin surrogate (11) is
\((\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})\)-identifiable with
\[
\mathsf C_{\varphi,1}=\min_{\hat y\neq y}\ell(\hat y,y),
\qquad
\mathsf C_{\varphi,2}=\max_{y\in\mathcal Y}\tau(y)+1.
\]
In particular, if
\[
v(y)\in\{0,1\}^d,
\qquad
\ell(\hat y,y)=\frac1{2d}\|v(\hat y)-v(y)\|_1,
\]
then \(\tau(y)=1\) for every \(y\), and \(\mathsf C_{\varphi,2}=2\).

As in the binary matching case, nontrivial consistency guarantees hold once
\[
m\geq\log\operatorname{card}(\mathcal Y).
\]
Since \(0\leq\ell(\cdot,\cdot)\leq1\), Corollary 3.3 gives, for every measurable \(f:\mathcal X\to\mathbb R^d\),
\[
R(f)-R^\star\geq
\left(
\frac{32\mathsf C_{\mathsf{MT}}^4}{m}
\log\left(
4|\mathcal Y|
\left(
1+
\frac{\max_{\hat y\in\mathcal Y}\tau(y)+1}
{\min_{\hat y\neq y}\ell(\hat y,y)}
\right)
\right)
\right)^{\frac{\alpha}{2(1-\alpha^2)}}
\]
implies
\[
R(f)-R^\star
\leq
\frac8{\min_{\hat y\neq y}\ell(\hat y,y)}
\left(R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star\right).
\]
Here \(|\mathcal Y|\) also denotes label-space cardinality.

**Claims and evidence**
- The structured hinge is identifiable: Lemma 3.3 gives \(\mathsf C_{\varphi,1}=\min_{\hat y\neq y}\ell(\hat y,y)\) and \(\mathsf C_{\varphi,2}=\max_y\tau(y)+1\).
- Binary-valued embeddings have a simpler gap: Lemma 3.3 gives \(\tau(y)=1\) and \(\mathsf C_{\varphi,2}=2\) for its stated loss.
- Nontrivial guarantees hold once \(m\geq\log\operatorname{card}(\mathcal Y)\): the displayed Corollary 3.3 specialization gives the stated threshold and factor \(8/\min_{\hat y\neq y}\ell(\hat y,y)\).

**Claims and evidence for Section 3.3**
- The general bounds apply to the stated binary, matching, and structured examples: Lemmas 3.1–3.3 give the constants listed in each subsection.
- The resulting comparisons are linear above explicit thresholds: the final inequalities in Sections 3.3.1–3.3.3 give factors \(16/(\phi(0)-\phi(\delta))\), \(16N\), and \(8/\min_{\hat y\neq y}\ell(\hat y,y)\), respectively.

**Claims and evidence for Section 3**
- Individual-label convex ranking consistency is impossible, but aggregated consistency is possible: Propositions 1–2, \(k\geq3\) and \(m\geq k\).
- Aggregation preserves calibration theory and strengthens consistency: Proposition 3, Corollaries 3.2–3.3, and Theorem 1 with factor \(16/\mathsf C_{\varphi,1}\).
- The theory covers the concrete surrogate examples: Lemmas 3.1–3.3 and their displayed comparison inequalities.

## 4 Robustness and consistency for models

*Aggregation can restore consistency when optimization is restricted to a hypothesis class.*

Section 3 used classical surrogate-consistency theory, where \(\mathcal F\) contains all measurable functions. It showed that sufficient aggregation can upgrade an inconsistent surrogate to level-\((\xi,\zeta)\) consistency.

We now consider optimization over restricted, often parametric classes \(\mathcal F\). Consistency is not an issue in a well-specified model, but that is an unrealistic and brittle assumption. Quantifying surrogate consistency when the class contains only some measurable functions is a long-standing challenge [12, 31].

Recent work develops \(H\)-consistency bounds linking excess target risk to surrogate regret within a regret hypothesis class \(H\), across various settings [2, 3, 26, 28, 29]. These results are mainly surrogate-centric. Our contribution is orthogonal: it depends less on the specific class and surrogate and instead studies how aggregation improves consistency itself.

We show that aggregation permits consistent estimation in settings where consistency otherwise may fail.

### 4.1 Consistency failure for binary classification in
finite dimensions

*Linear-class surrogate minimizers can point almost orthogonally to an optimal direction, while aggregated minimizers recover consistency asymptotically.*

Consider binary classification with
\[
\mathcal Y=\{\pm1\},
\qquad
\ell(\mathsf d(s),y)=1\{ys\leq0\}.
\]
Use a margin-based surrogate
\[
\varphi(s,y)=\phi(sy),
\]
with convex \(\phi:\mathbb R\to\mathbb R_+\). Over all measurable functions, it has both Fisher (i) and uniform (ii) consistency if and only if \(\phi'(0)<0\) [4].

Restricting the class changes this. Classical consistency fails even when \(\mathcal F\) contains an optimal classifier. We say \(P\) is **optimally predictable using \(\mathcal F\)** when a predictor in the class satisfies
\[
\mathsf{sgn}(f(x))
=
\mathsf{sgn}(P(Y=1\mid X=x)-1/2).
\tag{12}
\]

Let \(\mathcal X=\mathbb R^d\), now using \(d\) for input dimension, and take the linear class
\[
\mathcal F=
\{f_\theta\mid f_\theta(x)=\langle\theta,x\rangle\}_{\theta\in\mathbb R^d}.
\]
If \(P\) is optimally predictable using this class, some target parameter \(\theta^\star\) satisfies
\[
\mathsf{sgn}(\langle\theta^\star,x\rangle)
=
\mathsf{sgn}(P(Y=1\mid x)-\tfrac12).
\]
Then \(f_{\theta^\star}\) minimizes \(R(f)\) over all measurable functions. We call this optimal predictability along \(\theta^\star\).

A margin-based surrogate that is Fisher consistent in the classical setting need not remain consistent here. More strongly, for any nonnegative loss \(\phi\), some distribution \(P\) has a surrogate minimizer
\[
\theta_\varphi=
\argmin_\theta\mathbb E_P[\varphi(f_\theta(X),Y)]
\]
essentially orthogonal to \(\theta^\star\).

###### Proposition 4.

*For an optimally predictable distribution, every surrogate minimizer can have higher task risk and an arbitrarily small absolute directional cosine.*

For any \(\epsilon>0\) and nonzero \(\theta^\star\in\mathbb R^d\), there is a distribution \(P\) on \((X,Y)\), optimally predictable along \(\theta^\star\), such that every
\[
\theta_\varphi\in
\argmin_\theta R_\varphi(f_\theta)
=
\argmin_\theta
\mathbb E[\phi(f_\theta(X),Y)]
\]
satisfies
\[
R(f_{\theta_\varphi})>R(f_{\theta^\star}),
\]
and
\[
|\cos\angle(\theta_\varphi,\theta^\star)|
=
\frac{|\langle\theta_\varphi,\theta^\star\rangle|}
{\|\theta_\varphi\|_2\|\theta^\star\|_2}
\leq\epsilon.
\]
The cosine compares normalized parameter directions.

Appendix D.1 proves the result.

Aggregation can circumvent this failure. For any objective \(g\), define its approximate minimizers by
\[
\epsilon\textup{-argmin}\,g
=
\epsilon\textup{-argmin}_\theta g(\theta)
\coloneqq
\{\theta\mid g(\theta)\leq\inf_\theta g(\theta)+\epsilon\}.
\]
Thus \(\epsilon\) is the allowed objective gap.

Suppose \(Z=(Y_1,\ldots,Y_m)\) consists of independent samples, and \(A_m(Z)\) is majority vote. For tolerances \(\epsilon_m\), take
\[
\theta_m\in
\epsilon_m\textup{-argmin}_\theta R_{\varphi,A_m}(f_\theta)
=
\epsilon_m\textup{-argmin}_\theta
\mathbb E[
\phi(A_m(Y_1,\ldots,Y_m)f_\theta(X))
].
\]
As a corollary of Theorem 2 below, these predictors are asymptotically consistent as \(m\to\infty\).

###### Corollary 4.1.

*Vanishing optimization tolerance and increasing aggregation recover optimal risk and direction.*

Let \(P\) be optimally predictable along \(\theta^\star\). If
\(\epsilon_m\to0\) as \(m\to\infty\), then
\[
R(f_{\theta_m})\to R(f_{\theta^\star}),
\qquad
\cos\angle(\theta_m,\theta^\star)\to1.
\]

Without aggregation, Proposition 4 makes surrogate minimization over linear predictors essentially arbitrarily incorrect. With aggregation, consistency is retained.

**Claims and evidence**
- Linear-class surrogate minimization can fail despite optimal predictability: Proposition 4 gives \(R(f_{\theta_\varphi})>R(f_{\theta^\star})\) and \(|\cos\angle(\theta_\varphi,\theta^\star)|\leq\epsilon\) for any \(\epsilon>0\).
- Aggregation restores asymptotic risk and directional consistency: Corollary 4.1 gives \(R(f_{\theta_m})\to R(f_{\theta^\star})\) and \(\cos\angle(\theta_m,\theta^\star)\to1\), with \(\epsilon_m\to0\) and \(m\to\infty\).

### 4.2 Aggregation, consistency, and restricted hypothesis classes

*For an optimally predicting cone, majority-vote surrogate minimization is asymptotically consistent under the stated conditions.*

Proposition 4 shows how strongly classical consistency depends on including all measurable functions. Consider multiclass classification with
\(\mathcal Y=\{1,\ldots,k\}\). Let \(\mathcal F\) form a linear cone with
\[
f(x)^T\mathbf1=0,
\qquad
tf\in\mathcal F\quad\text{for }t>0\text{ if }f\in\mathcal F.
\]
Here a cone is closed under positive scaling.

Use zero-one loss
\[
\ell(y,y')=1\{y\neq y'\},
\qquad
\mathsf d(s)=\argmax_{y\in[k]}s_y.
\]
The zero-sum normalization does not affect this decoder.

Assume the nonnegative surrogate
\(\varphi:\mathbb R^k\times[k]\to\mathbb R_+\) is Fisher consistent and satisfies the limiting loss condition
\[
\varphi(s,y)\to0
\quad\text{if}\quad
s_y-s_j\to+\infty
\text{ for every }j\neq y.
\tag{13}
\]
Thus the loss vanishes when the correct score exceeds every other score without bound.

Examples include the multiclass logistic loss
\[
\varphi(s,y)=
\log\left(\sum_{j=1}^ke^{s_j-s_y}\right)
\]
and losses of the form
\[
\varphi(s,y)=\sum_{i\neq y}\phi(s_y-s_i),
\]
where \(\phi\) is convex and nonincreasing,
\(\phi'(0)<0\), and \(\inf_t\phi(t)=0\).
Zhang [51, Thm. 8] shows consistency of such losses over
\[
\mathcal F=
\{f:\mathcal X\to\mathbb R^k\mid\mathbf1^Tf=0\}.
\]

The binary setting in Section 4.1 fits this framework by taking
\(f(x)=(g(x),-g(x))\), where \(g\) is measurable.

For parametric models on \(\mathcal X=\mathbb R^d\), the class of linear functions
\[
f(x)=
(\langle\theta_1,x\rangle,\ldots,\langle\theta_k,x\rangle),
\qquad
\sum_{i=1}^k\theta_i=0,
\]
is a convex cone. These examples typically require strong structural assumptions, such as convexity, for consistency without aggregation. Aggregation gives asymptotic consistency under the minimal limiting loss condition (13).

Extending (12), say \(\mathcal F\) **optimally predicts \(P\)** if some \(f\in\mathcal F\) satisfies
\[
\argmax_y f_y(x)
\in
\argmax_y P(Y=y\mid X=x)
\quad\text{for all }x.
\]
This means the class contains a Bayes-optimal classifier, one selecting a most probable conditional label.

When \(\mathcal Z=\mathcal Y^m\), use majority vote \(A_m\). With a unique most probable class
\[
y^\star(x)=\argmax_yP(Y=y\mid x),
\]
surrogate minimization is consistent whenever \(\mathcal F\) optimally predicts \(P\).

###### Theorem 2.

*Approximate aggregated-risk minimizers over an optimally predicting cone converge to optimal task risk.*

Let \(\mathcal F\) be a cone that optimally predicts \(P\), and assume the minimal excess risk (8) satisfies
\[
P(\Delta(X)>0)=1.
\]
Let \(\epsilon_m\geq0\) and \(\epsilon_m\to0\). For every sequence
\[
f_m\in
\epsilon_m\textup{-argmin}_{f\in\mathcal F}R_{\varphi,A_m}(f)
=
\epsilon_m\textup{-argmin}_{f\in\mathcal F}
\mathbb E[
\varphi(f(X),A_m(Y_1,\ldots,Y_m))
],
\]
we have
\[
R(f_m)\to R^\star.
\]

Appendix D.2 proves the theorem.

Thus aggregation gives asymptotic consistency as \(m\to\infty\) for a broad class of surrogate problems over cones. In contrast, Proposition 4 shows that even binary classification can fail over subclasses containing an optimal predictor, with an arbitrarily uninformative surrogate.

**Claims and evidence**
- Familiar multiclass losses satisfy the stated framework: the displayed logistic and pairwise-margin losses satisfy Fisher consistency and (13) under the listed \(\phi\) conditions; Zhang [51, Thm. 8] supplies the cited consistency result.
- Aggregated minimization is consistent over an optimally predicting cone: Theorem 2 gives \(R(f_m)\to R^\star\), assuming \(P(\Delta(X)>0)=1\) and \(\epsilon_m\to0\).
- Optimal predictors alone do not ensure unaggregated consistency: Proposition 4 gives higher task risk and directional cosine at most any \(\epsilon>0\).

### 4.3 On finite-dimensional multiclass classification

*Small link mis-specification can defeat unaggregated logistic fitting, while majority vote recovers the optimal parameter direction asymptotically.*

Finally, consider multiclass linear prediction with
\(\mathcal X=\mathbb R^d\) and \(\mathcal Y=[k]\), allowing model mis-specification. These results specialize and refine Proposition 4 and Theorem 2.

Even when the problem is optimally predictable and the linear model is well-specified except on an \(\epsilon\)-fraction of data, surrogate risks based only on \((X,Y)\) pairs are inconsistent. Majority-vote methods recover the optimal linear predictor.

Let
\[
\Theta=[\theta_1\ \cdots\ \theta_{k-1}]
\in\mathbb R^{d\times(k-1)}.
\]
Labels follow the categorical distribution
\[
Y\mid X=x\sim\mathsf{Cat}(p_\Theta(x)),
\]
where \(p_\Theta(x)\in\mathbb R_+^k\) is the class-probability vector,
\(\mathbf1^\top p_\Theta(x)=1\), and
\[
p_\Theta(x)=
\begin{bmatrix}
\sigma(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)\\
1-\mathbf1^\top
\sigma(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)
\end{bmatrix}.
\]
The link
\(\sigma:\mathbb R^{k-1}\to\mathbb R_+^{k-1}\) maps scores to the first \(k-1\) probabilities and satisfies \(\mathbf1^\top\sigma\leq1\).

For score vector \(t\in\mathbb R^{k-1}\), set
\[
t_k=0,\qquad
\sigma_k(t)=1-\mathbf1^\top\sigma(t).
\]
Assume link consistency:
\[
t_y=\max_{1\leq i\leq k}t_i
\quad\text{if and only if}\quad
\sigma_y(t)=\max_{1\leq i\leq k}\sigma_i(t).
\tag{14}
\]

The logistic-regression link, denoted by the superscript \(\textup{lr}\), is
\[
\sigma_i^{\textup{lr}}(t)=
\frac{e^{t_i}}{1+e^{t_1}+\cdots+e^{t_{k-1}}}.
\]

Let the multilabeled dataset be
\[
\{(X_i;(Y_{1i},\cdots,Y_{im}))\}_{i=1}^n,
\]
with repeated sampling
\[
Y_{ij}\mid X_i
\stackrel{\rm iid}{\sim}
\mathsf{Cat}(p_{\Theta^\star}(X_i)).
\]
Let
\[
Y_i^+=A_m(Y_{i1},\cdots,Y_{im})
\]
be majority vote, breaking ties arbitrarily.

Then
\[
Y_i^+\mid X_i\sim
\mathsf{Cat}(p_{\Theta^\star,m}(X_i)).
\]
If \(\rho_m(t)\) denotes the majority-vote distribution on \(m\) items with initial probabilities \(\sigma(t)\in\mathbb R_+^{k-1}\), then
\[
p_{\Theta,m}(x)=
\begin{bmatrix}
\rho_m(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)\\
1-\mathbf1^\top
\rho_m(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)
\end{bmatrix}.
\]
It is evident that \(\rho_m\) satisfies (14).

Fit logistic regression with loss
\[
\varphi(\Theta^\top x,y)=
-\langle\theta_y,x\rangle+
\log\left(
1+\sum_{i=1}^{k-1}\exp(\langle\theta_i,x\rangle)
\right),
\tag{15}
\]
using the convention \(\theta_k=\mathbf0\), the zero vector.

Define population logistic risk with \(m\)-majority vote:
\[
L_m(\Theta)=
\mathbb E[\varphi(\Theta^\top X,Y_m^+)].
\]
Its derivative with respect to \(\theta_i\) is
\[
\nabla_{\theta_i}\varphi(\Theta^\top x,y)=
-x\left(1\{y=i\}-\sigma_i^{\textup{lr}}(\Theta^\top x)\right).
\]
Thus a minimizer \(\Theta_m\) satisfies the first-order condition
\[
\nabla_\Theta L_m(\Theta_m)=
\mathbb E\left[
X\left(
\sigma^{\textup{lr}}(\Theta_m^\top X)
-\rho_m({\Theta^\star}^\top X)
\right)^\top
\right]=0.
\]

Standard statistical results guarantee consistency and efficiency for a well-specified model without aggregation. For \(m\geq1\) and \(k=2\), Cheng et al. [7, Prop. 3] show that some \(t_m>0\) gives
\[
\Theta_m=t_m\Theta^\star
\]
when \(X\sim\mathsf N(0,I_d)\), a centered normal input with identity matrix \(I_d\), even under link mis-specification.

Hence binary classification remains consistent for any aggregation level \(m\), despite an incorrect link, because the direction
\(\Theta^\star/\|\Theta^\star\|_2\) determines consistency.

For \(k\geq3\), slight link mis-specification instead causes risk-consistency failure. Fix a set
\(\mathcal T_\epsilon\subset\mathbb R^{k-1}\) of Lebesgue measure \(\epsilon\), and define
\[
\sigma^\epsilon(t)=
\sigma^{\textup{lr}}(t)1\{t\notin\mathcal T_\epsilon\}
+\frac1k\mathbf1\cdot1\{t\in\mathcal T_\epsilon\}.
\]
This defines a conditional distribution on \(Y\in\{1,\ldots,k\}\) that samples
\(Y\sim\sigma^{\textup{lr}}(t)\) if \(t\in\mathcal T_\epsilon\), and uniformly otherwise. Clearly \(\sigma^\epsilon\) satisfies (14) and is optimally predictable under (12).

For \(\epsilon>0\), define
\[
L_{m,\epsilon}(\Theta)\coloneqq
\mathbb E[
\mathbb E_{\sigma^\epsilon}[
\varphi(\Theta^\top X,Y_m^+)\mid X
]
].
\]
This is the population logistic loss with \(m\)-majority vote when
\[
Y\mid X=x\sim\sigma^\epsilon({\Theta^\star}^\top x).
\]
Let
\[
\Theta_m(\epsilon)=\argmin_\Theta L_{m,\epsilon}(\Theta).
\]
Evidently \(\Theta_1(0)=\Theta^\star\). Yet arbitrarily small positive \(\epsilon\) can cause inconsistency without aggregation.

###### Proposition 5.

*An arbitrarily small modified-link set can change the normalized unaggregated logistic minimizer.*

Let \(k\geq3\) and \(\Sigma=I\), where \(I\) is the identity matrix. For
\(Z\sim\mathsf N(0,I_{k-1})\), assume the linear mapping
\[
M\mapsto DM\coloneqq
\mathbb E\left[
Z\left(
\nabla\sigma^{\textup{lr}}({\Theta^\star}^\top Z)MZ
\right)^\top
\right]
\]
is invertible. Here \(M\) is the matrix argument and \(D\) denotes the displayed linear map.

Then there is an \(\epsilon_0>0\) such that, for any
\(\epsilon\in(0,\epsilon_0)\), some set \(\mathcal T_\epsilon\) has Lebesgue measure at most \(\epsilon\) and satisfies
\[
\Theta_1(\epsilon)/\|\Theta_1(\epsilon)\|
\neq
\Theta^\star/\|\Theta^\star\|.
\]

Appendix E.1 proves the proposition.

Majority vote addresses this inconsistency as \(m\to\infty\), without assumptions on the true link beyond (14). Define
\[
L_{m,\sigma}(\Theta)=
\mathbb E[
\mathbb E_\sigma[
\varphi(\Theta^\top X,Y_m^+)\mid X
]
],
\qquad
\Theta_m=\argmin L_{m,\sigma}(\Theta).
\]

###### Theorem 3.

*The fitted parameter remains in the target subspace and its factor converges in normalized direction as aggregation grows.*

Let
\[
\Theta^\star=U^\star T^\star,
\]
where \(U^\star\in\mathbb R^{d\times(k-1)}\) is orthogonal and
\(T^\star\in\mathbb R^{(k-1)\times(k-1)}\) is nonsingular. Then some
\(T_m\in\mathbb R^{(k-1)\times(k-1)}\) satisfies
\[
\Theta_m=U^\star T_m
\quad\text{for every }m.
\]
As \(m\to\infty\),
\[
\|T_m\|\to\infty,
\qquad
T_m/\|T_m\|\to T^\star/\|T^\star\|.
\]

Appendix E.2 proves the theorem.

This gives further evidence for aggregation’s robustness: asymptotic consistency is possible under model mis-specification so long as a link describes the relationship between \(X\) and \(Y\). The contrast is striking for \(k\geq3\), where Proposition 5 shows that methods without aggregation are generally inconsistent.

**Claims and evidence**
- Binary Gaussian-input fitting preserves the target direction despite a mis-specified link: Cheng et al. [7, Prop. 3], \(k=2\), \(m\geq1\), \(X\sim\mathsf N(0,I_d)\), and \(\Theta_m=t_m\Theta^\star\) with \(t_m>0\).
- Small multiclass link changes can defeat unaggregated fitting: Proposition 5, \(k\geq3\), \(\epsilon\in(0,\epsilon_0)\), and a set of measure at most \(\epsilon\), with unequal normalized parameters.
- Aggregation recovers normalized direction: Theorem 3 gives \(\Theta_m=U^\star T_m\) for every \(m\), \(\|T_m\|\to\infty\), and \(T_m/\|T_m\|\to T^\star/\|T^\star\|\).

**Claims and evidence for Section 4**
- Restricted classes can invalidate ordinary surrogate consistency despite containing an optimal predictor: Proposition 4, risk strictly above optimum and absolute directional cosine at most any \(\epsilon>0\).
- Aggregation restores asymptotic consistency in the stated restricted classes: Corollary 4.1 and Theorem 2, with optimization tolerance \(\epsilon_m\to0\).
- Aggregation is robust to the stated multiclass link mis-specification: Proposition 5 supplies the unaggregated failure for \(k\geq3\); Theorem 3 supplies normalized-direction convergence as \(m\to\infty\).

## 5 Numerical illustrations

*Two experiments illustrate consistency amplification and recovery under link mis-specification.*

We give two experiments supporting the theory: (i) a nonparametric example where aggregation amplifies surrogate consistency, and (ii) a finite-dimensional multiclass example where it recovers asymptotic consistency under link mis-specification.

##### Surrogate consistency amplification for truncated quadratic loss.

*The majority-vote comparison bound approaches linear behavior for the truncated quadratic surrogate.*

For binary classification, use the margin-based surrogate
\[
\varphi(f(x),y)=\phi(yf(x)),
\qquad
\phi(\delta)=\max\{1-\delta,0\}^2.
\]
This \(\phi\) is the truncated quadratic loss. Bartlett et al. [4, Example 2] give the exact calibration function in Corollary 3.1:
\[
\psi(\epsilon)=\epsilon^2.
\]

With majority vote over \(m\) labels, apply Lemma 3.1 with \(\delta=1\). Without a low-noise condition, meaning
\(\alpha=0,\mathsf C_{\mathsf{MT}}=1\), take
\[
\begin{aligned}
\psi_m(\epsilon)
&=
\frac{\phi(0)-\phi(\delta)}{16}
\left(
\epsilon-
\sqrt{\frac{32}{m}
\log\left(
\frac{8(\phi(-\delta)+\phi(0)-\phi(\delta))}
{\phi(0)-\phi(\delta)}
\right)}
\right)_+\\
&=
\frac1{16}
\left(
\epsilon-\sqrt{\frac{32}{m}\log40}
\right)_+.
\end{aligned}
\]
Here \((\cdot)_+\) is truncation at zero. The non-aggregated and aggregated comparisons are
\[
\psi(R(f)-R^\star)\leq R_\varphi(f)-R_\varphi^\star,
\qquad
\psi_m(R(f)-R^\star)\leq
R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star.
\]

Numerically, Figure 1 shows the aggregated curve transitioning from quadratic toward linear behavior as \(m\to\infty\). This illustrates upgraded surrogate consistency. The functions \(\psi_m\) are generally conservative because they come from the general Theorem 1.

Figure 1: Numerical illustrations for truncated quadratic surrogate \(\phi(\delta)=\max\{1-\delta,0\}^{2}\), showing comparison inequalities for non-aggregated data \(\psi\) (optimal) vs. majority vote aggregation \(\psi_m\) (suboptimal) when \(m\in\{2^{16},2^{20},2^{24},2^{28},2^{32}\}\).

**Claims and evidence**
- The unaggregated calibration is quadratic: Bartlett et al. [4, Example 2], \(\psi(\epsilon)=\epsilon^2\).
- The aggregation bound approaches linear behavior: the displayed \(\psi_m(\epsilon)=\frac1{16}(\epsilon-\sqrt{(32/m)\log40})_+\) and Figure 1, with \(m\in\{2^{16},2^{20},2^{24},2^{28},2^{32}\}\).
- The aggregated bound is conservative: Figure 1 labels \(\psi_m\) “suboptimal”; it is obtained from Theorem 1.

##### Asymptotic consistency for mis-specified logistic model.

*The corrupted-link experiment illustrates asymptotic consistency as the number of aggregated labels grows.*

Use Section 4.3’s logistic regression with \(k=3\) and \(d=2\), and logistic loss (15). Define a corrupted link
\[
\tilde\sigma(t)=
\sigma^{\textup{lr}}(t)1\{\|t\|\leq r\}
+
\sigma^{\textup{lr}}(\alpha t)1\{\|t\|>r\}.
\]
This piecewise link matches softmax, the logistic link, near the origin and shrinks scores outside radius \(r\). Here \(\alpha\) is the score-shrinkage parameter, rather than the noise exponent.

Figure 2 gives numerical results from synthetic data sampled from this link. It shows asymptotic consistency as \(m\to\infty\).

| (i) Experiment with constant dataset size \(n\) | (ii) Experiment with constant label size \(nm\). |
|---|---|

Figure 2: Numerical illustrations for mis-specified logistic models when \(d=2\) and \(k=3\). The true parameters are \(\theta_1^\star=(0,1)^\top,\theta_2^\star=(1,1)^\top\). The corrupted link parameters are \(r=3\) and \(\alpha=0.1\). The plot shows the error of classification rule \(\left\|\|T_m\|-T^\star/\|T^\star\|\right\|\) vs. number of labels \(m\in\{1,4,16,64,256,1024\}\). (i) In the left panel, for each value of \(m\), the synthetic dataset consists of \(n=10{,}000\) multilabeled observations \((X_i,(Y_{i1},\ldots,Y_{im}))\) sampled from \(\tilde\sigma\). (ii) In the right panel, for each \(m\), we set \(n=640{,}000/m\) so that the total number of labels remains constant across different values of \(m\). We report a \(95\%\) error bar on \(T=100\) trials for each \(m\).

**Claims and evidence**
- The experiment illustrates asymptotic consistency under a corrupted link: Figure 2, \(d=2\), \(k=3\), \(r=3\), \(\alpha=0.1\), and \(m\in\{1,4,16,64,256,1024\}\).
- Both fixed-observation and fixed-total-label designs are shown: Figure 2(i), \(n=10{,}000\); Figure 2(ii), \(n=640{,}000/m\).
- Uncertainty is reported across repeated trials: Figure 2 caption, \(95\%\) error bars and \(T=100\) trials for each \(m\).

**Claims and evidence for Section 5**
- Aggregation amplifies the illustrated surrogate comparison: Figure 1 contrasts \(\psi(\epsilon)=\epsilon^2\) with the displayed \(\psi_m\) at \(m\in\{2^{16},2^{20},2^{24},2^{28},2^{32}\}\).
- Aggregation illustrates recovery under link mis-specification: Figure 2 uses \(d=2\), \(k=3\), \(r=3\), \(\alpha=0.1\), and the six aggregation counts listed above. Its caption reports designs and error bars, but no numerical error values.

## 6 Discussion and future work

*The paper places its robustness results in the data-cleaning debate and identifies finite-aggregation, noise, model, and restricted-class questions.*

Whether and how to clean data has motivated much discussion about dataset collection. Cheng, Asi and Duchi [7] describe an apparent pattern: using all non-aggregated labels improves statistical efficiency when models can represent all uncertainty, while cleaning appears more robust otherwise.

Similarly, Dorner and Hardt [11] argue that, when validating and comparing binary classifiers, more noisy labels are preferable to cleaned variants.

This paper contributes evidence of fundamental limits to using uncleaned, unaggregated label information in supervised learning. It also shows robustness improvements from cleaning. Many questions remain.

##### Finite \(m\) results and fundamental limits.

*Many guarantees require infinitely growing aggregation, leaving finite-label consistency theory as a central question.*

Many results require \(m\), the number of aggregated labels, to tend to infinity. Although the paper aims to show ways aggregation improves robustness, reliance on this asymptotic setting may be unsatisfying.

Ranking in Section 3.1 has explicit guarantees at finite \(m\). Extending such results is a natural and, we believe, important direction. A theory depending jointly on the loss pair \((\ell,\varphi)\) and available label count \(m\) would be useful.

For example, do second- or third-order ranking comparisons—such as powers \(C_x^p\), suggested by Keener [20]—require larger \(m\)? A central challenge is to identify precisely which problems require label cleaning and aggregation and which do not.

##### Fundamental limits of the noise condition number.

*The condition number can remain large beyond binary classification, even at low noise, motivating sharper loss-dependent analysis.*

We use the noise condition number \(\kappa(X)\) in (9) to characterize multiclass comparison inequalities. This hints at the difficulty beyond binary classification, where \(\kappa(X)=1\) trivially.

Beyond binary classification, the condition number may be large even when the Mammen–Tsybakov noise level (Nα) is low, meaning \(\alpha\approx1\). This follows from our minimal surrogate assumptions.

It would help to identify connections between \(\ell\) and \(\varphi\) that capture problem difficulty more closely. Explicit failure constructions could also clarify fundamental limits of predictive-model fitting.

##### Behavior in mis-specified models.

*The model results require an optimal classifier in the class, and extending beyond that restriction remains open.*

The mis-specified-model results, especially Section 4.3, require optimal predictability (12): a Bayes-optimal classifier must lie in \(\mathcal F\).

Classical surrogate consistency can fail even then, while aggregated-label methods can succeed. Going beyond this restriction is a fruitful direction.

Nguyen et al. [31], followed by Duchi et al. [12], identify one direction. In binary and multiclass classification, respectively, jointly inferring predictor \(f\) and a data representation for \(x\) requires surrogates of a particular form determined by task loss \(\ell\). Those results still require infinitely powerful decision rules \(f\). New approaches are needed.

##### Restricted class comparison inequality with label aggregation.

*Extending aggregation-based comparison bounds to restricted classes could connect the parametric and nonparametric results.*

This paper studies how aggregation mitigates surrogate-induced regret through nonparametric comparison inequalities, and it establishes statistical consistency in parametric settings.

Recent \(H\)-consistency work [2, 28, 27, 29] motivates extending these techniques to comparison inequalities over restricted classes. Mao et al. [27] show that quadratic growth of \(\psi\) is rate optimal for smooth convex loss. It would be interesting to use aggregation to upgrade consistency in \(\mathcal H\)-consistency.

Such an extension could unify aggregation’s advantages across parametric and nonparametric settings. It could also clarify how the “size” of \(\mathcal F\) affects aggregation’s effectiveness.

**Claims and evidence**
- Some consistency guarantees already hold at finite aggregation: Proposition 2 gives \(m\geq k\) for ranking; Section 3.2.3 gives uniform linear comparison at \(\alpha=1\) under its displayed \(O(\log k)\) bound.
- Many model guarantees remain asymptotic: Corollary 4.1 and Theorems 2–3 take \(m\to\infty\).
- Optimal predictability does not prevent ordinary surrogate failure, but aggregation can succeed: Proposition 4 contrasts with Corollary 4.1 and Theorem 2; Proposition 5 contrasts with Theorem 3.
- The stated noise and model limitations remain: Section 3.2.3 uses condition number (9); Section 4.3 requires optimal predictability (12).
