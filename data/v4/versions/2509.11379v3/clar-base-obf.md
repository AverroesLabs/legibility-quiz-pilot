# Some Robustness Properties of Label Cleaning

Chen Cheng

John Duchi

###### Abstract

Aggregated labels combine information from responses, including noisy responses. This cleaning permits robustness that is unavailable without aggregation. In particular, minimizing a surrogate loss—an easier loss used in place of the task loss—can give stronger guarantees that task risk converges to its optimal value. The surrogate is typically convex. Aggregation also preserves consistency for finite-dimensional classifiers when the fitted loss is mis-specified, meaning that it does not correctly describe the data-generating relationship. Standard approaches that model uncertainty can fail under this mis-specification.

When a statistical model is perfectly specified, incorporating all information favors statistical efficiency. Yet even slight loss mis-specification destroys standard consistency. Aggregation retains convergence to optimal classifiers. These results show how collection, fitting, and prediction can jointly gain robustness by refining noisy signals.

## 1 Introduction

Aggregation permits consistent optimal prediction under a surrogate loss rather than the task loss, even when estimation from unaggregated labels cannot. This distinction also occurs in finite-dimensional statistical problems.

Write \(X\) for an input and \(Y\) for its response or label. Constructions using linear estimators in ranking, binary classification, and multiclass classification separate procedures restricted to individual \((X,Y)\) pairs from procedures that aggregate responses. Nonparametric results, which do not restrict predictors to a finite-dimensional model, combine conditions on the aggregation count and noise. They give approximately linear comparison inequalities between excess task risk and excess surrogate risk under weak surrogate assumptions. They also extend consistency beyond classes containing every measurable predictor.

The conventional supervised-learning pipeline collects \(n\) input–label pairs, \((X_i,Y_i)_{i=1}^n\), and fits a model [17]. Here \(i\) indexes observations. This description suppresses the cleaning, filtering, repeated crowdsourced labeling, and denoising required by modern datasets [9, 22, 39]. Work on crowdsourced “gold standard” labels [8, 48, 47, 46, 37] studies this machinery. The concern here is its fundamental and necessary improvement of fitted-model consistency: convergence to an optimal prediction rule.

Important contributions to surrogate-risk consistency theory date to the 2000s [52, 24, 40]. Bartlett, Jordan and McAuliffe [4] characterize consistency of convex surrogates for binary zero-one classification, where the task loss records whether a prediction is wrong. Later work treats multi-label classification [50, 43, 16, 49, 1], ranking [13, 14, 36], structured prediction [34, 6, 32], ordinal regression [35], restricted hypothesis classes [2, 3, 28, 29], and general theory [40]. A hypothesis class is the set of predictors available to the fitting procedure.

Analyses based on individual pairs admit empirical-process theory, which controls differences between sample averages and population expectations [45, 5, 21, 4]. They do not include the aggregation machinery used to create contemporary datasets. The questions here are whether aggregation is needed, whether pair-based surrogate consistency suffices, and in what sense aggregation improves consistency.

## 2 Preliminaries

The paper retains two regimes. The first uses an unrestricted class \(\mathcal F\) of Borel measurable predictors and the classical notion of **Fisher consistency**, labeled **Q7D**. Borel measurability means measurability for the Borel sets of the spaces involved. The second uses a parametric, potentially mis-specified class \(\mathcal F\). Aggregation improves consistency in both regimes.

Throughout, \(\mathbb R\) denotes the real numbers, and \(\mathbb R_+\) denotes the nonnegative real numbers. The symbols \(\mathbb E\) and \(\mathbb P\) denote expectation and probability. A subscript \(P\) identifies the distribution used. The indicator \(1\{B\}\) equals \(1\) when statement \(B\) holds and \(0\) otherwise. The notation \(\inf\) denotes an infimum; \(\operatorname{argmin}\) and \(\operatorname{argmax}\) denote minimizing and maximizing arguments, respectively. They can be sets unless a selection rule is specified.

Let \((X_i,Y_i)_{i=1}^n\sim P\) be independent and identically distributed, abbreviated **i.i.d.**, on the input and output spaces \(\mathcal X\times\mathcal Y\). A score function is
\[
f:\mathcal X\to\mathbb R^d,\qquad d\geq1,
\]
where \(d\) is the number of score coordinates. A decoder
\(\mathsf d:\mathbb R^d\to\mathcal Y\) converts scores into predictions:
\(\widehat y=\mathsf d\circ f(x)=\mathsf d(f(x))\).

The task criterion may depend on the conditional distribution of the response:
\[
R(f)\coloneqq
\mathbb E_P\!\left[\ell(\mathsf d\circ f(X),P(\cdot\mid X))\right].
\tag{2}
\]
Here \(\coloneqq\) means “is defined as,” \(R(f)\) is the task risk, \(\ell\) is the task loss, and \(P(\cdot\mid X)\) is the conditional distribution of \(Y\) given \(X\).

Its minimizers frequently coincide with the original problem (2). Allowing dependence on the conditional distribution permits more sophisticated criteria. For example, let \(\mathcal Y=\{1,\ldots,k\}\), where \(k\) is the number of labels, and set
\[
\ell(\widehat y,P)=\sum_y P(Y=y)1\{\widehat y\neq y\}.
\]
Then \(R(f)\) is the standard zero-one error rate.

The instantaneous formulation instead uses a loss
\(\ell:\mathcal Y\times\mathcal Y\to\mathbb R_+\) on a prediction and one observed label. It minimizes, over \(f\in\mathcal F\),
\[
R(f):=\mathbb E_P[\ell(\mathsf d\circ f(X),Y)].
\tag{1}
\]
In binary classification, \(d=1\), the decoder is the sign function
\(\mathsf d(s)=\mathsf{sgn}(s)\), and
\[
\ell(y,y')=1\{yy'\leq0\},\qquad
R(f)=\mathbb P(Yf(X)\leq0).
\]

Task losses may be nonsmooth, nonconvex, and uninformative for optimization. The zero-one loss is flat almost everywhere, preventing even practical first-order optimization based on derivatives. An easier surrogate
\(\varphi:\mathbb R^d\times\mathcal Y\to\mathbb R\) replaces it:
\[
R_\varphi(f):=\mathbb E_P[\varphi(f(X),Y)].
\]
This replacement requires consistency with the task criterion.

### 2.1 Label aggregation

Let \(Z\in\mathcal Z\) be the information used in place of an ordinary response. Let \(\mathcal A\) be the aggregation-output space, and let
\(A:\mathcal Z\to\mathcal A\) be the aggregation map. For a nonnegative surrogate
\(\varphi:\mathbb R^d\times\mathcal A\to\mathbb R_+\), define the **aggregated surrogate risk**, labeled **P9S**, by
\[
R_{\varphi,A}(f):=\mathbb E[\varphi(f(X),A(Z))].
\tag{3}
\]
The question is whether minimizing this quantity also minimizes the actual task risk. The response used for fitting is \(A(Z)\), rather than an ordinary response from an \((X_i,Y_i)\) pair.

For repeated labeling,
\[
Z=(Y_1,\ldots,Y_m)\in\mathcal Y^m,
\qquad
Y_i\mid X=x\stackrel{\rm iid}{\sim}P_{Y\mid X=x},
\]
where \(m\) is the number of labels for the input. Aggregation can minimize the empirical task loss:
\[
A(Z)=A(\{Y_1,\ldots,Y_m\})
=\mathop{\rm argmin}_{y\in\mathcal Y}
\sum_{l=1}^m\ell(y,Y_l).
\]
For \(\ell(\widehat y,y)=1\{y\neq\widehat y\}\), this is **majority vote**, labeled **T4K**. General \(\ell\) allows more abstract aggregation procedures. The notation \(A_m(Z)\) records the repetition count. Mao et al. [28] use a similar multilabeled structure for multilabel losses, rather than for this aggregation pipeline.

A rough \(K\)-nearest-neighbor construction uses nearby inputs instead of repeated responses at exactly the same input. It supplies \(X\) with its label \(Y\) and an additional independent sample
\((X_i,Y_i)_{i=1}^m\stackrel{\rm iid}{\sim}P\). Thus
\(Z=(Y,(X_i,Y_i)_{i=1}^m)\). For a metric
\(\operatorname{dist}:\mathcal X\times\mathcal X\to\mathbb R_+\), order the additional inputs by distance:
\[
\operatorname{dist}(X,X_{(1)})\leq\cdots\leq
\operatorname{dist}(X,X_{(m)}),\qquad X_{(0)}=X.
\]
Parenthesized indices denote this ordering; the corresponding label at index \(0\) is \(Y\). For \(K\geq0\), use
\[
A_{m,K}(Z)\coloneqq
\mathop{\rm argmin}_{y\in\mathcal Y}
\sum_{l=0}^K\ell(y,Y_{(l)}).
\]
The population-level repeated-sampling treatment also extends to aggregation from one sample \((X_i,Y_i)_{i=1}^n\) (Appendix C).

## 3 Surrogate consistency

The standard framework [40] takes \(\mathcal F\) to contain all Borel measurable functions \(f:\mathcal X\to\mathbb R^d\). Define the optimal task and surrogate risks by
\[
R^\star:=\inf_{f\in\mathcal F}R(f),\qquad
R_\varphi^\star:=\inf_{f\in\mathcal F}R_\varphi(f).
\]
The differences \(R(f)-R^\star\) and
\(R_\varphi(f)-R_\varphi^\star\) are the corresponding excess risks.

For a score vector \(s\in\mathbb R^d\) and input \(x\), define conditional risks and conditional excess risks:
\[
\begin{aligned}
R(s\mid x)&\coloneqq
\ell(\mathsf d\circ s,P(Y\in\cdot\mid X=x)),\\
R_\varphi(s\mid x)&:=\mathbb E[\varphi(s,Y)\mid X=x],\\
\delta_\ell(s,x)&\coloneqq
R(s\mid x)-\inf_{s'\in\mathbb R^d}R(s'\mid x),\\
\delta_\varphi(s,x)&\coloneqq
R_\varphi(s\mid x)-\inf_{s'\in\mathbb R^d}R_\varphi(s'\mid x).
\end{aligned}
\]
Here \(\mathsf d\circ s\) means the prediction \(\mathsf d(s)\), and \(s'\) is a candidate score vector.

The **calibration function**, labeled **H8N**, measures the least surrogate excess risk compatible with a specified task excess risk. Its pointwise and uniform forms are
\[
\psi(\epsilon,x)\coloneqq
\inf_{s\in\mathbb R^d}
\{\delta_\varphi(s,x)\mid\delta_\ell(s,x)\geq\epsilon\},
\qquad
\overline\psi(\epsilon)\coloneqq
\inf_{x\in\mathcal X}\psi(\epsilon,x).
\]
The number \(\epsilon>0\) is a task-excess-risk threshold. Below, the same letter \(\psi\), without an \(x\) argument, also denotes a comparison function obtained from \(\overline\psi\).

###### Corollary 3.1.

Positivity of \(\psi(\epsilon,x)\) for every \(x\in\mathcal X\) and \(\epsilon>0\) is equivalent to \(\varphi\) being Fisher consistent (Q7D) for \(\ell\). If \(\psi\) is the Fenchel biconjugate of \(\overline\psi\), meaning its closed convex lower envelope, then
\[
\overline\psi(\epsilon)>0\quad\Longleftrightarrow\quad
\psi(\epsilon)>0,
\qquad
\psi(R(f)-R^\star)\leq R_\varphi(f)-R_\varphi^\star
\]
for every measurable \(f\).

For all distributions \(P\), the standard framework considers two requirements [40, 4, 52]:

(i) Every sequence \(f_n\in\mathcal F\) with
\[
R_\varphi(f_n)\to R_\varphi^\star
\]
also satisfies \(R(f_n)\to R^\star\). This is the consistency requirement: approaching optimal surrogate risk must approach optimal task risk.

(ii) There is a nondecreasing function
\(\psi:\mathbb R_+\to\mathbb R_+\), positive at every \(\epsilon>0\), such that
\[
\psi(R(f)-R^\star)\leq R_\varphi(f)-R_\varphi^\star
\quad\text{for every }f\in\mathcal F.
\]

Requirement (ii) is the **uniform comparison inequality**, labeled **R2V**. It is stronger than (i) and need not hold. For convex margin-based binary surrogates, the two requirements coincide [4]. A margin-based binary surrogate depends on the product of the label and the score.

The equivalence and inequality in Corollary 3.1 follow from Zhang [51, Prop. 25] and Steinwart [40, Thm. 2.8 and Lemma 2.9].

A Fisher-consistent surrogate for a general task would make single input–label pairs sufficient for consistent estimation. It would simplify collection and admit the full empirical-process toolkit. Such pair-only results are generally impossible.

### 3.1 Q7D failure without aggregation: ranking

Consider pairwise-comparison ranking [20, 15, 14, 30] with \(k\) items. Its response space is
\[
\mathcal Y=\{(i,j):i\neq j,\ 1\leq i,j\leq k\},
\]
where \((i,j)\) says that item \(i\) outranks item \(j\).

For each input \(x\), write
\(p_{ij}=P(Y=(i,j)\mid x)\). Population aggregation transforms these comparison probabilities into an ordering. A simple comparison scheme [20, cf.] defines
\[
(C_x)_{ii}=0,\qquad
(C_x)_{ij}=\frac{p_{ij}}{\sum_{l\neq j}p_{lj}}
\quad(i\neq j),\qquad
0/0=1/(k-1).
\]
The matrix \(C_x\in\mathbb R_+^{k\times k}\) is stochastic:
\(C_x^T\mathbf1=\mathbf1\). Here \(T\) denotes transpose, and \(\mathbf1\) is the all-ones vector of the required dimension. Thus each column sums to one. The vector \(C_x\mathbf1\) measures each item's preference frequency.

Higher matrix powers \(C_x^p\mathbf1\), where \(p\) is the power, Perron vectors [20], and more sophisticated schemes admit similar negative results. A Perron vector is a leading eigenvector of the nonnegative comparison matrix.

Absorb decoding into the loss:
\[
\ell(s,C)\coloneqq
\max_{i<j}1\!\left\{
(s_i-s_j)(e_i-e_j)^TC\mathbf1\leq0,\ 
(e_i-e_j)^TC\mathbf1\neq0
\right\}.
\]
Here \(s\in\mathbb R^k\) is the score vector, \(C\) is a comparison matrix, and \(e_i\) is the \(i\)th coordinate vector. The loss penalizes disagreement between the score ordering and the matrix ordering. The resulting risk is
\[
R(f)\coloneqq
\mathbb P\!\left(f(X)\text{ and }C_X\mathbf1
\text{ order differently}\right).
\tag{4}
\]
Restricting scores to \(s^T\mathbf1=0\) is a minor restriction, familiar in multiclass classification [51, 43]. It is natural when decoding uses only the ordering.

###### Proposition 1.

For this ranking task with \(k\geq3\) outcomes, every
\(\varphi:\mathbb R^k\times\mathcal Y\to\mathbb R\) that is convex in its first argument fails Fisher consistency (Q7D). See Appendix A.1.

For the aggregation used in the next proposition, take multiple independent comparisons
\(Z=(Y_1,\ldots,Y_m)\) conditional on \(X\). Define \(m_{ij}\) and \(m_j\) by
\[
m_{ij}=\sum_{y\in\mathcal Y^m}1\{y=(i,j)\},
\qquad m_j=\sum_{i=1}^k m_{ij}.
\]
Write \([k]=\{1,\ldots,k\}\). Let \(\star\) be a special aggregation output, and use
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

###### Proposition 2.

For \(s,q\in\mathbb R^k\), set
\[
\varphi(s,q)=\|s-q\|_2^2,\qquad \varphi(s,\star)=0,
\]
where \(\|\cdot\|_2\) is the Euclidean norm. With the aggregation above and \(m\geq k\), \(\varphi\) is Fisher consistent (Q7D) for the ranking risk. See Appendix A.2.

The response space has expanded from \(\mathcal Y\) to \(\mathbb R^k\). This expansion is unavailable to the convex single-comparison formulation and separates Propositions 1 and 2. Regression on ordering frequencies gives the positive result. It acts directly on non-\(\star\) outputs; the loss at \(\star\) is zero.

### 3.2 Aggregation and stronger surrogate consistency

Aggregation extends ordinary surrogate-risk theory. It also upgrades losses that are inconsistent without aggregation.

#### 3.2.1 Basic extensions

For an arbitrary aggregation map \(A:\mathcal Z\to\mathcal A\), define
\[
\begin{aligned}
R_{\varphi,A}(s\mid x)&\coloneqq
\mathbb E[\varphi(s,A(Z))\mid X=x],\\
\delta_{\varphi,A}(s,x)&:=
R_{\varphi,A}(s\mid x)-
\inf_{s\in\mathbb R^d}R_{\varphi,A}(s\mid x).
\end{aligned}
\]
These are the conditional aggregated surrogate risk and its excess above the conditional infimum. Let
\(R_{\varphi,A}^\star=\inf_{f\in\mathcal F}R_{\varphi,A}(f)\).
The corresponding calibration quantities are
\[
\overline\psi_A(\epsilon,x)\coloneqq
\inf_{s\in\mathbb R^d}
\{\delta_{\varphi,A}(s,x)\mid\delta_\ell(s,x)\geq\epsilon\},
\qquad
\overline\psi_A(\epsilon)\coloneqq
\inf_{s\in\mathbb R^d}\overline\psi_A(\epsilon,x).
\tag{5}
\]

###### Proposition 3.

Suppose an integrable function \(b:\mathcal X\to\mathbb R_+\) satisfies
\[
\int b(x)\,dP(x)<\infty,\qquad
\delta_\ell(f(x),x)\leq b(x).
\]
Thus \(b\) is an upper bound on the displayed conditional task excess risk and has finite expectation. Then \(\varphi\) is Fisher consistent (Q7D) for the task risk exactly when
\(\overline\psi_A(\epsilon,x)>0\) for every \(x\in\mathcal X\) and \(\epsilon>0\). With
\(\psi_A=(\overline\psi_A)^{**}\), where \(^{**}\) denotes the Fenchel biconjugate,
\[
\psi_A(R(f)-R^\star)
\leq R_{\varphi,A}(f)-R_{\varphi,A}^\star.
\]

Appropriate measurability conditions are tacit and noncentral. Proposition 3 is nearly a corollary of Steinwart [40, Thm. 2.8]; Appendix B.1 gives a proof for completeness. It preserves classical guarantees but does not itself identify a particular improvement from aggregation.

#### 3.2.2 W3C and consistency

The property **W3C** means **Identifying surrogate**. Its formal inequalities appear in Definition 3.1 below. It requires aggregation parameters that make different labels distinguishable through the surrogate.

The setting here has a finite label space:
\(\operatorname{card}(\mathcal Y)=k<\infty\), where
\(\operatorname{card}\) denotes the number of elements. The task uses an instantaneous loss,
\[
R(f)=\mathbb E[\ell(f(X),Y)].
\]
The ranking construction belongs to this regime. There, \(k\) was the ranking length and
\(\operatorname{card}(\mathcal Y)=k(k-1)\). From now on, \(k\) denotes label-space cardinality, with this slight abuse of notation.

For parameters \(\{a_y\}_{y\in\mathcal Y}\), define **generalized majority vote**, labeled **J9P**, by empirical task minimization followed by parameter substitution:
\[
A_m(y_1,\ldots,y_m)\coloneqq a_{\hat y},
\qquad
\hat y=\mathop{\rm argmin}_{y\in\mathcal Y}
\sum_{i=1}^m\ell(y,y_i).
\tag{7}
\]
Ties are broken arbitrarily. Each \(a_y\) belongs to the aggregation space \(\mathcal A\).

###### Corollary 3.2.

For \(m=m(n)\to\infty\), where the repetition count grows with sample size \(n\), and an identifying surrogate \(\varphi\) satisfying W3C,
\[
R_{\varphi,A_m}(f_n)-R_{\varphi,A_m}^\star\to0
\quad\Longrightarrow\quad R(f_n)-R^\star\to0.
\]

For i.i.d. \(Y_i\) and finite \(\mathcal Y\), a finite random index \(M<\infty\) necessarily exists such that every \(m\geq M\) satisfies
\[
\mathop{\rm argmin}_{y\in\mathcal Y}
\left\{\sum_{i=1}^m\ell(y,Y_i)\right\}
\subset
y^\star(x)\coloneqq
\mathop{\rm argmin}_{y\in\mathcal Y}
\mathbb E[\ell(y,Y)\mid X=x].
\]
Here \(y^\star(x)\) is the set of conditionally optimal labels. Because there are finitely many candidate labels, empirical loss averages eventually exclude every label whose expected loss is strictly larger than the minimum. This strategy gives asymptotic consistency under minimal identifiability for any such task loss.

###### Definition 3.1.

A surrogate \(\varphi:\mathbb R^d\times\mathcal A\to\mathbb R\) is an **Identifying surrogate (W3C)** with constants
\(0<\mathsf C_{\varphi,1}\leq\mathsf C_{\varphi,2}<\infty\) if there exist parameters
\(\{a_y\}_{y\in\mathcal Y}\subset\mathcal A\) and score vectors
\(\{s_y\}_{y\in\mathcal Y}\subset\mathbb R^d\) such that
\(\mathsf d(s_y)=y\) and, for all \(y\neq y'\),
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

Inequality (6a) requires a positive surrogate-loss gap between \(s_y\) and every score decoded as a label other than \(y\), when the aggregation parameter is \(a_y\). It makes a minimizer at \(a_y\) identify \(y\). A finite lower bound on \(\varphi(\cdot,a)\) supplies a finite \(\mathsf C_{\varphi,2}\) for (6b).

The definition does not require convexity. It does not require unaggregated consistency with \(\mathcal A=\mathcal Y\) and \(Z=Y\). It also does not impose a decoder such as pointwise \(\operatorname{argmax}\).

For example, the binary hinge surrogate is
\(\varphi(s,a)=\max\{1-sa,0\}\), with
\(\mathcal A=\mathcal Y=\{\pm1\}\). Choose \(a_y=s_y=y\). Then
\[
\varphi(s_1,a_1)=\varphi(s_{-1},a_{-1})=0,\qquad
\inf_{sa\leq0}\varphi(s,a)=1,
\]
\[
\varphi(s_1,a_{-1})=\varphi(s_{-1},a_1)=2.
\]
Its W3C constants are \((1,2)\).

#### 3.2.3 W3C and consistency amplification

The next theorem uses several quantities defined fully below. The exponent
\(\alpha\in[0,1]\) controls the noise condition (Nα), and
\(\mathsf C_{\mathsf{MT}}>0\) is its constant. The quantity \(\kappa(x)\) is a ratio of the largest to smallest conditional excess risks of incorrect predictions, defined in (9). The function \(e_m(t)\), defined in (10), measures aggregation error at repetition count \(m\) and ratio \(t\). The number \(M>0\) is a cutoff for \(\kappa(X)\).

###### Theorem 1.

Let \(\varphi\) be
\((\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})\)-W3C with parameters
\(\{a_y\}_{y\in\mathcal Y}\), and let \(A_m\) be generalized majority vote (J9P). Assume \(0\leq\ell\leq1\) and condition (Nα), with \(\alpha\in[0,1]\). For any \(M>0\) and \(f\in\mathcal F\) satisfying
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

Thus the corresponding threshold \(\xi\) and linear-comparison slope \(\zeta\) are
\[
\xi=2\mathbb P(\kappa(X)>M)+
(4\mathsf C_{\mathsf{MT}}e_m(M))^{\frac1{1-\alpha}},
\qquad
\zeta=\frac{\mathsf C_{\varphi,1}}{16}.
\]
Define the best displayed threshold over cutoffs by
\[
\xi_m\coloneqq
\inf_M\left\{
2\mathbb P(\kappa(X)>M)+
(4\mathsf C_{\mathsf{MT}}e_m(M))^{\frac1{1-\alpha}}
\right\}.
\]
Then \(\xi_m\to0\). Indeed,
\(\mathbb P(\kappa(X)>M)\to0\) as \(M\uparrow\infty\), while
\(e_m(M)\to0\) as \(m\to\infty\) for fixed \(M\). One can first choose a large cutoff and then increase the aggregation count. Taking \(\alpha=0\) gives asymptotic consistency.

The quantity \(\Delta(x)\), defined in (8) below, is the smallest positive conditional task-risk gap for an incorrect prediction. The crude bound
\(\kappa(x)\leq1/\Delta(x)\), choosing \(M=1\) when
\(\operatorname{card}(\mathcal Y)=k=2\) and optimizing \(M\) otherwise, lower bounds the promised \(\xi_{m,k}\) by setting
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
Here \(\log\) is the natural logarithm. Appendix B.4 gives the algebraic substitutions and manipulations.

###### Corollary 3.3.

Under Theorem 1's conditions, every \(f\) with
\(R(f)-R^\star\geq\xi_{m,k}\) satisfies
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
gives \(\xi_{m,k}=0\) and a linear uniform comparison inequality (R2V). The notation \(O(\log k)\) states that the required count grows at most on the order of \(\log k\). Thus lower-noise problems require less aggregation for stronger consistency.

Low noise permits a comparison written as
\[
\psi(R(f)-R^\star)\leq R_\varphi(f)-R^\star,
\]
with \(\psi\) linear over some positive-\(\epsilon\) range. Linear growth is the strongest possible [34, 32]. Strict comparison requirements can exclude practically convenient inconsistent surrogates [23, 34, 32]. Following Osokin et al. [34], the paper therefore uses a relaxation that requires linear growth only above a threshold.

###### Definition 3.2.

The surrogate and aggregator have **level-\((\xi,\zeta)\) consistency**, labeled **U6A**, if some function \(\psi\) satisfies
\[
\psi(\epsilon)\geq\zeta\epsilon\quad(\epsilon\geq\xi),
\qquad
\psi(R(f)-R^\star)\leq
R_{\varphi,A}(f)-R_{\varphi,A}^\star.
\]
The threshold is \(\xi\), and \(\zeta\) is the required linear slope. Only excess risks above the threshold require linear growth. This retains a “good enough” criterion. Minimal assumptions permit levels \((o_m(1),\zeta)\), where \(o_m(1)\) denotes a quantity tending to zero as \(m\to\infty\), even for a surrogate that is not Fisher consistent.

The **minimal excess conditional risk**, labeled **F8Q**, is the incorrect-prediction gap
\[
\Delta(x)\coloneqq
\min_{\mathsf d(s)\notin y^\star(x)}
\delta_\ell(s,x).
\tag{8}
\]
In binary classification it equals
\(|2P(Y=1\mid X=x)-1|\). Estimation is expected to be harder when this gap is nearer zero.

Adapting classical assumptions [25], also Bartlett et al. [4], the **Mammen–Tsybakov noise conditions**, labeled **N6Y**, are
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
Here \(f^\star\) is an optimal task predictor. The probability in (Nα) measures disagreement between the decoded predictions of \(f\) and \(f^\star\). The positive constant \(\mathsf C_{\mathsf{MT}}\) may change. The noise exponents satisfy \(\alpha\in[0,1]\) and \(\beta\in[0,\infty]\). Both conditions hold trivially at \(\alpha=\beta=0\). As in the binary result [4, Thm. 3], they are equivalent under
\(\beta=\alpha/(1-\alpha)\); see Appendix B.2.

Mao et al. [29] use exactly the same noise statistic for multiclass \(H\)-consistency, meaning consistency relative to a hypothesis class \(H\). They impose structure on both the surrogate and the hypothesis class to obtain a global convex lower bound.

With minimal structure, the paper instead requires the **noise condition number**, labeled **B2Z**:
\[
\kappa(x)\coloneqq
\frac{\max_{\mathsf d(s)\neq y^\star(x)}\delta_\ell(s,x)}
{\min_{\mathsf d(s)\neq y^\star(x)}\delta_\ell(s,x)}.
\tag{9}
\]
It compares the largest and smallest conditional excess risks among incorrect predictions. For \(\mathsf d(s)\neq y^\star(x)\),
\[
\Delta(x)\geq\delta_\ell(s,x)/\kappa(x).
\]
Binary classification has \(\kappa(x)=1\), provided
\(\mathbb P(\Delta(X)>0)=1\).

The auxiliary **error function**, labeled **V5R**, is
\[
e_m(t)\coloneqq
t\sqrt{\frac2m
\log\!\left(
\frac{4k(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})}
{\mathsf C_{\varphi,1}}
\right)}.
\tag{10}
\]
It is used with \(\kappa(x)=t\). When it is much smaller than one, it expresses likely correctness of \(A_m\).

Appendix B.3 proves Theorem 1. Its coarse bounds reflect the weak identifiability assumptions. Tighter relationships between \(\ell\) and \(\varphi\), ensuring that generalized majority vote is correct or sufficiently correct for \(\varphi\), should strengthen the bounds. Near-consistency without aggregation could connect smaller initial inconsistency to better dependence on \(m\). These refinements are not pursued.

### 3.3 Surrogate consistency with T4K

The following instances use majority vote (T4K) and assume a noise exponent \(\alpha\in[0,1]\). This loses no generality because \(\alpha=0\) always satisfies (Nα). Appendix B.5 gives the proofs.

#### 3.3.1 Binary classification, nonsmooth surrogate

Use a binary margin surrogate
\[
\varphi(f(x),y)=\phi(yf(x)).
\]
Assume that \(\phi\) is convex,
\[
\partial\phi(0)\subset(-\infty,0),
\qquad
\lim_{t\to\infty}\phi(t)=0.
\]
Here \(\partial\phi(0)\) is the set of subgradients, or supporting-line slopes, at zero. The parameter \(\delta>0\) below selects a score magnitude.

###### Lemma 3.1.

For every \(\delta>0\), \(\varphi\) is
\((\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})\)-feasible, with feasibility expressed by the identifying-surrogate constants
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

For convex binary margin surrogates, Fisher consistency (Q7D) and a uniform comparison inequality (R2V) over all measurable functions hold exactly when \(\phi'(0)<0\) [4]. Here \(\phi'(0)\) denotes the derivative at zero. Aggregation can make an otherwise inconsistent \(\phi\) consistent. This is a somewhat trivial thought experiment, not a surrogate one would use.

#### 3.3.2 Bipartite matching

For typically large structured output spaces, an embedding
\(v:\mathcal Y\to\mathbb R^d\) represents structural information [33]. With
\[
\mathsf d(s)=\operatorname{argmax}_{y\in\mathcal Y}\langle s,v(y)\rangle
\]
and a task loss \(\ell:\mathcal Y\times\mathcal Y\to\mathbb R_+\) satisfying \(\ell(y,y)=0\), define the **maximum-margin surrogate**, labeled **A3J**:
\[
\varphi(s,y)=
\max_{\hat y\in\mathcal Y}
\left(\ell(\hat y,y)+
\langle v(\hat y)-v(y),s\rangle\right).
\tag{11}
\]
The notation \(\langle a,b\rangle\) denotes the usual inner product of vectors. This loss is also called the generalized hinge loss or structured hinge loss. It originates in [42, 44, 19]. It is generally inconsistent, except in certain low-noise cases [34, 32].

For the matching instance, let \(G=(V,E)\) be a graph with vertex set \(V\) and edge set \(E\). Its two sides are
\[
V=V_1\cup V_2,\qquad
V_1=\{1,\ldots,N\},\qquad
V_2=\{N+1,\ldots,2N\}.
\]
Each of the \(N\) edges joins exactly one unique node on each side. The output space \(\mathcal Y\) contains all such matchings, so
\(k=\operatorname{card}(\mathcal Y)=N!\), where \(N!\) is the factorial of \(N\).

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
Here \(\|\cdot\|_1\) is the sum of absolute coordinate values. The embedding records which edges are present.

###### Lemma 3.2.

On \(2N\) vertices, the matching surrogate is W3C with
\[
\mathsf C_{\varphi,1}=\frac1N,\qquad
\mathsf C_{\varphi,2}=2.
\]
Despite the factorial label cardinality \(k=N!\), aggregation gives consistency once its count exceeds \(O(N\log N)\). For \(k\geq3\), every measurable \(f:\mathcal X\to\mathbb R^d\) obeying
\[
R(f)-R^\star\geq
4\cdot\left(
\frac{32\mathsf C_{\mathsf{MT}}^4}{m}
\log(4k(2N+1))
\right)^{\frac{\alpha}{2(1-\alpha^2)}}
\]
satisfies
\[
R(f)-R^\star\leq
16N\bigl(R_{\varphi,A_n}(f)-R_{\varphi,A_n}^\star\bigr).
\]

#### 3.3.3 Structured prediction

Use the maximum-margin surrogate (A3J) from (11). Assume that the structured decoder can pick every class. Specifically, for each \(y\), the set
\[
\mathcal S(y)\coloneqq
\{s:\langle v(y),s\rangle>
\langle v(\hat y),s\rangle
\text{ for }\hat y\neq y\}
\]
is nonempty. Thus \(\mathcal S(y)\) contains scores that strictly prefer \(y\).

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
The labels \(y_+\) and \(y_-\) range over alternatives to \(y\). This quantity compares task-loss gaps with score gaps.

###### Lemma 3.3.

The maximum-margin surrogate A3J is
\((\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})\)-W3C with
\[
\mathsf C_{\varphi,1}=
\min_{\hat y\neq y}\ell(\hat y,y),
\qquad
\mathsf C_{\varphi,2}=
\max_{y\in\mathcal Y}\tau(y)+1.
\]
In particular,
\[
v(y)\in\{0,1\}^d,\qquad
\ell(\hat y,y)=\frac1{2d}\|v(\hat y)-v(y)\|_1
\]
give \(\tau(y)=1\) for every \(y\) and
\(\mathsf C_{\varphi,2}=2\).

Nontrivial consistency holds once
\(m\geq\log\operatorname{card}(\mathcal Y)\), as in the binary matching case. With \(0\leq\ell(\cdot,\cdot)\leq1\), every measurable
\(f:\mathcal X\to\mathbb R^d\) satisfying
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
Here \(|\mathcal Y|\) is another notation for the label-space cardinality.

## 4 Robustness and consistency for models

Now restrict \(\mathcal F\), often to a parametric class, instead of allowing all measurable functions. Correct model specification makes consistency straightforward. It is also brittle and unrealistic. Surrogate consistency on subsets of measurable functions is a long-standing topic [12, 31].

Recent \(H\)-consistency bounds connect excess target risk and surrogate regret within a hypothesis class \(H\) in diverse settings [2, 3, 26, 28, 29]. Surrogate regret is excess surrogate risk relative to the class under consideration. These results are primarily surrogate-centric. The contribution from aggregation is orthogonal and depends less on a particular class and surrogate.

### 4.1 Binary consistency failure in finite dimensions

The binary setting uses
\[
\mathcal Y=\{\pm1\},\qquad
\ell(\mathsf d(s),y)=1\{ys\leq0\},
\qquad
\varphi(s,y)=\phi(sy),
\]
where \(\phi:\mathbb R\to\mathbb R_+\) is convex. Inputs lie in
\(\mathcal X=\mathbb R^d\), and predictors are linear:
\[
\mathcal F=
\{f_\theta:f_\theta(x)=\langle\theta,x\rangle\}_{\theta\in\mathbb R^d}.
\]
The vector \(\theta\) is the model parameter.

Even including an optimal classifier does not preserve classical consistency. The inclusion requirement, **optimal predictability**, is labeled **K7M**. It means that some \(f\in\mathcal F\) satisfies
\[
\mathsf{sgn}(f(x))=
\mathsf{sgn}(P(Y=1\mid X=x)-1/2).
\tag{12}
\]
If a vector \(\theta^\star\) realizes this requirement, then
\[
\mathsf{sgn}(\langle\theta^\star,x\rangle)
=
\mathsf{sgn}(P(Y=1\mid x)-\tfrac12).
\]
The function \(f_{\theta^\star}\) minimizes \(R\) over all measurable functions. In this case, \(P\) has K7M along \(\theta^\star\).

###### Proposition 4.

For every \(\epsilon>0\) and nonzero
\(\theta^\star\in\mathbb R^d\), some distribution \(P\) on \((X,Y)\) satisfies optimal predictability (K7M) along \(\theta^\star\), yet every surrogate-risk minimizer
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
The notation \(\angle(a,b)\) denotes the angle between vectors \(a\) and \(b\); the displayed ratio defines its absolute cosine. This essential orthogonality occurs for any nonnegative loss \(\phi\). See Appendix D.1.

To state the aggregation result, define approximate minimizers. For an objective \(g\) and tolerance \(\epsilon\geq0\),
\[
\operatorname{\epsilon\textup{-argmin}}g
=
\operatorname{\epsilon\textup{-argmin}}_\theta g(\theta)
\coloneqq
\{\theta:g(\theta)\leq\inf_\theta g(\theta)+\epsilon\}.
\]
With independent repeated labels
\(Z=(Y_1,\ldots,Y_m)\) and majority vote \(A_m\), consider
\[
\theta_m\in
\operatorname{\epsilon_m\textup{-argmin}}_\theta
R_{\varphi,A_m}(f_\theta)
=
\operatorname{\epsilon_m\textup{-argmin}}_\theta
\mathbb E[
\phi(A_m(Y_1,\ldots,Y_m)f_\theta(X))].
\]
Here \(\epsilon_m\) is the optimization tolerance at aggregation count \(m\).

###### Corollary 4.1.

As a corollary to Theorem 2, if \(P\) satisfies K7M along \(\theta^\star\) and
\(\epsilon_m\to0\) as \(m\to\infty\), the aggregated approximate minimizers satisfy
\[
R(f_{\theta_m})\to R(f_{\theta^\star}),
\qquad
\cos\angle(\theta_m,\theta^\star)\to1.
\]

Thus a linear restriction can make unaggregated surrogate minimization arbitrarily incorrect, as Proposition 4 shows. Aggregation retains asymptotic consistency, as Corollary 4.1 states.

### 4.2 Aggregation and restricted classes

For multiclass classification, take
\[
\mathcal Y=\{1,\ldots,k\},\qquad
\ell(y,y')=1\{y\neq y'\},\qquad
\mathsf d(s)=\operatorname{argmax}_{y\in[k]}s_y.
\]
The class \(\mathcal F\) satisfies the linear-cone conditions
\[
f(x)^T\mathbf1=0,\qquad
f\in\mathcal F,\ t>0\ \Longrightarrow\ tf\in\mathcal F.
\]
A cone is closed under multiplication by positive scalars. The zero-sum normalization does not affect decoding.

Assume that the nonnegative surrogate
\(\varphi:\mathbb R^k\times[k]\to\mathbb R_+\) is Fisher consistent (Q7D) and satisfies
\[
\varphi(s,y)\to0
\quad\text{if}\quad
s_y-s_j\to+\infty\text{ for every }j\neq y.
\tag{13}
\]
Thus its loss tends to zero when the correct class score exceeds every other score by a diverging margin.

The response space is \(\mathcal Z=\mathcal Y^m\), and aggregation is majority vote (T4K). Assume that the population maximizer
\[
y^\star(x)=\operatorname{argmax}_yP(Y=y\mid x)
\]
is unique.

Multiclass optimal predictability (K7M) requires some
\(f\in\mathcal F\), \(f:\mathcal X\to\mathbb R^k\), such that
\[
\operatorname{argmax}_y f_y(x)
\in\operatorname{argmax}_yP(Y=y\mid X=x)
\quad\text{for every }x.
\]

###### Theorem 2.

Let a cone \(\mathcal F\) satisfy K7M for \(P\), and assume
\(P(\Delta(X)>0)=1\). For tolerances \(\epsilon_m\geq0\) with
\(\epsilon_m\to0\), every sequence
\[
f_m\in
\operatorname{\epsilon_m\textup{-argmin}}_{f\in\mathcal F}
R_{\varphi,A_m}(f)
=
\operatorname{\epsilon_m\textup{-argmin}}_{f\in\mathcal F}
\mathbb E[
\varphi(f(X),A_m(Y_1,\ldots,Y_m))]
\]
satisfies \(R(f_m)\to R^\star\). See Appendix D.2.

This broad cone result requires aggregation. Without it, even binary subclasses containing an optimal predictor can have arbitrarily uninformative surrogates, as Proposition 4 shows.

Examples of Fisher-consistent losses meeting (13) are the multiclass logistic loss
\[
\varphi(s,y)=\log\left(\sum_{j=1}^k e^{s_j-s_y}\right)
\]
and losses of the form
\[
\varphi(s,y)=\sum_{i\neq y}\phi(s_y-s_i),
\]
where \(\phi\) is convex and nonincreasing, with
\(\phi'(0)<0\) and \(\inf_t\phi(t)=0\). Zhang [51, Thm. 8] establishes consistency of these losses over
\[
\mathcal F=\{f:\mathcal X\to\mathbb R^k:\mathbf1^Tf=0\}.
\]

Binary margin predictors correspond to
\(f(x)=(g(x),-g(x))\), where \(g\) is measurable. For
\(\mathcal X=\mathbb R^d\), linear functions
\[
f(x)=(\langle\theta_1,x\rangle,\ldots,\langle\theta_k,x\rangle),
\qquad
\sum_{i=1}^k\theta_i=0,
\]
form a convex cone. A convex cone is also closed under convex combinations. Without aggregation, such examples typically require strong structure such as convexity. With aggregation, the minimal limiting condition permits asymptotic consistency.

### 4.3 Finite-dimensional multiclass classification

The setup uses inputs \(\mathcal X=\mathbb R^d\), labels
\(\mathcal Y=[k]\), and a parameter matrix
\[
\Theta=[\theta_1~\cdots~\theta_{k-1}]
\in\mathbb R^{d\times(k-1)}.
\]
The conditional response distribution is
\(Y\mid X=x\sim\mathsf{Cat}(p_\Theta(x))\). Here
\(\mathsf{Cat}(p)\) is the categorical distribution with label-probability vector \(p\). The vector
\(p_\Theta(x)\in\mathbb R_+^k\) satisfies
\(\mathbf1^\top p_\Theta(x)=1\) and is defined by
\[
p_\Theta(x)=
\begin{bmatrix}
\sigma(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)\\
1-\mathbf1^\top
\sigma(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)
\end{bmatrix}.
\]
The link function
\(\sigma:\mathbb R^{k-1}\to\mathbb R_+^{k-1}\) maps scores to the first \(k-1\) class probabilities and satisfies
\(\mathbf1^\top\sigma\leq1\).

For a score vector \(t=(t_1,\ldots,t_{k-1})\), put \(t_k=0\) and
\(\sigma_k(t)=1-\mathbf1^\top\sigma(t)\). The required **ordering consistency** is
\[
t_y=\max_{1\leq i\leq k}t_i
\quad\Longleftrightarrow\quad
\sigma_y(t)=\max_{1\leq i\leq k}\sigma_i(t).
\tag{14}
\]
Thus the largest scores and largest probabilities correspond to the same labels.

Multiclass logistic regression uses the link
\[
\sigma_i^{\rm lr}(t)=
\frac{e^{t_i}}{1+e^{t_1}+\cdots+e^{t_{k-1}}}.
\]
The superscript \({\rm lr}\) identifies the logistic-regression link.

The multilabeled dataset is
\[
\{(X_i;(Y_{1i},\cdots,Y_{im}))\}_{i=1}^n,
\qquad
Y_{ij}\mid X_i\stackrel{\rm iid}{\sim}
\mathsf{Cat}(p_{\Theta^\star}(X_i)),
\]
where \(\Theta^\star\) is the true parameter matrix. With arbitrary tie-breaking, the majority-vote label
\(Y_i^+=A_m(Y_{i1},\cdots,Y_{im})\) obeys
\[
Y_i^+\mid X_i\sim\mathsf{Cat}(p_{\Theta^\star,m}(X_i)).
\]
Let \(\rho_m(t)\) denote the first \(k-1\) probabilities of the majority-vote distribution from \(m\) labels whose initial probabilities are
\(\sigma(t)\in\mathbb R_+^{k-1}\). Then
\[
p_{\Theta,m}(x)=
\begin{bmatrix}
\rho_m(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)\\
1-\mathbf1^\top
\rho_m(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)
\end{bmatrix},
\]
and \(\rho_m\) satisfies ordering consistency.

The fitted logistic loss, with \(\theta_k=\mathbf0\), is
\[
\varphi(\Theta^\top x,y)=
-\langle\theta_y,x\rangle+
\log\left(1+\sum_{i=1}^{k-1}
\exp(\langle\theta_i,x\rangle)\right).
\tag{15}
\]
The zero vector is denoted by \(\mathbf0\). Let \(Y_m^+\) denote a majority-vote response formed from \(m\) labels, and define
\(L_m(\Theta)=\mathbb E[\varphi(\Theta^\top X,Y_m^+)]\).
Since
\[
\nabla_{\theta_i}\varphi(\Theta^\top x,y)=
-x(1\{y=i\}-\sigma_i^{\rm lr}(\Theta^\top x)),
\]
its minimizer \(\Theta_m\) satisfies
\[
\nabla_\Theta L_m(\Theta_m)=
\mathbb E\!\left[
X\left(
\sigma^{\rm lr}(\Theta_m^\top X)
-\rho_m({\Theta^\star}^\top X)
\right)^\top
\right]=0.
\]
The symbol \(\nabla\) denotes the gradient, or the matrix of first derivatives for a vector-valued function.

For the perturbation result below, take a fixed set
\(\mathcal T_\epsilon\subset\mathbb R^{k-1}\) of Lebesgue measure \(\epsilon\), where Lebesgue measure is ordinary volume. Set
\[
\sigma^\epsilon(t)=
\sigma^{\rm lr}(t)1\{t\notin\mathcal T_\epsilon\}
+\frac1k\mathbf1\,1\{t\in\mathcal T_\epsilon\}.
\]
The conditional distribution on \(Y\in\{1,\ldots,k\}\) samples
\(Y\sim\sigma^{\rm lr}(t)\) if \(t\in\mathcal T_\epsilon\) and uniformly otherwise. The link satisfies ordering consistency and K7M.

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
The inner expectation uses the conditional label distribution specified by \(\sigma^\epsilon\). Thus \(L_{m,\epsilon}\) is the population logistic loss for \(m\)-label majority vote when
\(Y\mid X=x\sim\sigma^\epsilon({\Theta^\star}^\top x)\).

###### Proposition 5.

Let \(k\geq3\) and \(\Sigma=I\), where \(I\) is the identity matrix and \(\Sigma\) is the matrix set equal to it in this statement. Let
\(Z\sim\mathsf N(0,I_{k-1})\), the centered normal distribution with identity covariance in \(k-1\) dimensions. Assume that the linear map on matrices \(M\),
\[
M\mapsto DM\coloneqq
\mathbb E\!\left[
Z\bigl(
\nabla\sigma^{\rm lr}({\Theta^\star}^TZ)MZ
\bigr)^T
\right],
\]
is invertible. Here \(DM\) denotes the displayed image of \(M\).

Then some \(\epsilon_0>0\) has the following property. Every
\(\epsilon\in(0,\epsilon_0)\) admits a set \(\mathcal T_\epsilon\) of Lebesgue measure at most \(\epsilon\) such that
\[
\Theta_1(\epsilon)/\|\Theta_1(\epsilon)\|
\neq
\Theta^\star/\|\Theta^\star\|.
\]
The notation \(\|\cdot\|\) denotes the norm used for these matrices.

Proposition 5 is proved in Appendix E.1.

For a general ordering-consistent link \(\sigma\), define
\[
L_{m,\sigma}(\Theta)=
\mathbb E[
\mathbb E_\sigma[
\varphi(\Theta^\top X,Y_m^+)\mid X]],
\qquad
\Theta_m=\operatorname{argmin}L_{m,\sigma}(\Theta).
\]

###### Theorem 3.

Suppose
\(\Theta^\star=U^\star T^\star\), where
\(U^\star\in\mathbb R^{d\times(k-1)}\) is orthogonal, meaning that its columns are orthonormal, and
\(T^\star\in\mathbb R^{(k-1)\times(k-1)}\) is nonsingular, meaning invertible. For each \(m\), there is a matrix
\(T_m\in\mathbb R^{(k-1)\times(k-1)}\) such that
\(\Theta_m=U^\star T_m\), and
\[
\|T_m\|\to\infty,\qquad
T_m/\|T_m\|\to T^\star/\|T^\star\|
\quad(m\to\infty).
\]

Theorem 3 is proved in Appendix E.2.

Pair-based surrogate risks can be inconsistent even with K7M and a linear model that is correct except on an \(\epsilon\)-fraction of data. Majority vote recovers the optimal linear predictor asymptotically. Proposition 5 gives the failure under perturbation, and Theorem 3 gives asymptotic recovery with aggregation. For \(k\geq3\), arbitrarily slight mis-specification of the true link defeats risk consistency. Aggregation requires no true-link assumption beyond ordering consistency.

Well-specified unaggregated models have standard statistical consistency and efficiency. For \(m\geq1\), \(k=2\), and
\(X\sim\mathsf N(0,I_d)\), Cheng et al. [7, Prop. 3] obtain
\(\Theta_m=t_m\Theta^\star\) for some \(t_m>0\), even with a mis-specified link. The direction
\(\Theta^\star/\|\Theta^\star\|_2\) therefore suffices for consistent binary classification at every aggregation level.

Although \(\Theta_1(0)=\Theta^\star\), arbitrarily small positive corruption destroys unaggregated consistency, as Proposition 5 states. For general ordering-consistent \(\sigma\), Theorem 3 gives robustness under mis-specification whenever some link describes the \(X\)–\(Y\) relationship. Without aggregation, multiclass methods are generally inconsistent.

## 5 Numerical illustrations

Two experiments corroborate the theory. The first illustrates nonparametric consistency amplification. The second illustrates finite-dimensional multiclass asymptotic consistency under link mis-specification.

##### Surrogate consistency amplification for D4X.

The **truncated quadratic loss**, labeled **D4X**, is the binary margin loss
\[
\varphi(f(x),y)=\phi(yf(x)),\qquad
\phi(\delta)=\max\{1-\delta,0\}^2.
\]
Its exact unaggregated calibration function (H8N) is
\(\psi(\epsilon)=\epsilon^2\) [4, Example 2].

Take \(\delta=1\) in Lemma 3.1. Impose no low-noise condition:
\(\alpha=0\) and \(\mathsf C_{\mathsf{MT}}=1\). The resulting comparison function is
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
Here \((u)_+=\max\{u,0\}\) is the positive part of \(u\).

The two comparisons are
\[
\psi(R(f)-R^\star)\leq R_\varphi(f)-R_\varphi^\star,
\qquad
\psi_m(R(f)-R^\star)\leq
R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star.
\]
The aggregated curve changes from quadratic toward linear as \(m\to\infty\), upgrading consistency. Figure 1 compares these curves for the counts listed below. The curves from the general theorem are conservative.

Figure 1: D4X comparison curves for unaggregated \(\psi\) (optimal) and T4K \(\psi_m\) (suboptimal).

| Figure | Aggregated-label counts |
|---|---|
| 1 | \(m\in\{2^{16},2^{20},2^{24},2^{28},2^{32}\}\) |

##### Asymptotic consistency for a mis-specified logistic model.

Synthetic data from a corrupted link exhibit asymptotic consistency as \(m\to\infty\). The fitted logistic loss is (15). The piecewise link agrees with the logistic, or softmax, link near the origin and shrinks scores outside radius \(r\):
\[
\widetilde\sigma(t)=
\sigma^{\rm lr}(t)1\{\|t\|\leq r\}
+\sigma^{\rm lr}(\alpha t)1\{\|t\|>r\}.
\]
Here \(r\) is the radius, and \(\alpha\) is the score-shrinkage factor, rather than the noise exponent used earlier.

The experiments vary the number \(m\) of labels per input in two ways:

| (i) Experiment with constant dataset size \(n\) | (ii) Experiment with constant label size \(nm\) |
|---|---|
| For each \(m\), \(n=10{,}000\) multilabeled observations \((X_i,(Y_{i1},\ldots,Y_{im}))\) sampled from \(\widetilde\sigma\) | For each \(m\), \(n=640{,}000/m\); total labels constant across \(m\) |

Figure 2: Mis-specified logistic-model illustrations, left panel (i), right panel (ii); plotted classification-rule error is \(\left\|\|T_m\|-T^\star/\|T^\star\|\right\|\) against label count.

The settings and repetition counts supporting Figure 2 are:

| Setting | Value |
|---|---|
| Dimension | \(d=2\) |
| Classes | \(k=3\) |
| True parameters | \(\theta_1^\star=(0,1)^\top,\quad\theta_2^\star=(1,1)^\top\) |
| Corrupted-link parameters | \(r=3,\quad\alpha=0.1\) |
| Label counts | \(m\in\{1,4,16,64,256,1024\}\) |
| Error bars | \(95\%\) |
| Trials for each \(m\) | \(T=100\) |

Here \(T\) denotes the number of trials, not matrix transpose.

## 6 Discussion and future work

Cheng, Asi and Duchi [7] discuss the data-collection debate. Using all unaggregated labels appears statistically more efficient when models can represent all uncertainty. Otherwise, cleaning appears more robust. Dorner and Hardt [11] favor more noisy labels rather than cleaned labels for validation comparisons of binary classifiers. The present results establish fundamental limitations of uncleaned supervised-label information and robustness gains from cleaning.

##### Finite \(m\) and fundamental limits.

Many consistency results require \(m\to\infty\). Reliance on this limit remains unsatisfying. Ranking admits explicit finite-\(m\) guarantees: Proposition 2 requires \(m\geq k\).

A theory depending jointly on the task loss \(\ell\), surrogate loss \(\varphi\), and available aggregation count \(m\) remains open. Second- or third-order item comparisons, including the powers \(C_x^p\) suggested by Keener [20], may require increasing counts. Precisely separating problems that require cleaning and aggregation from those that do not is a central challenge and a natural, important direction.

##### Fundamental limits of B2Z.

Beyond binary classification, the noise condition number \(\kappa(X)\) (B2Z) may remain large even with low Mammen–Tsybakov noise (N6Y), meaning \(\alpha\approx1\). Binary problems trivially have \(\kappa(X)=1\). The minimal surrogate assumptions produce this behavior.

Closer connections between the task and surrogate losses could better represent difficulty. Explicit failure constructions and sharper fundamental limits could improve understanding of predictive-model fitting.

##### Mis-specified models.

The mis-specified-model results require optimal predictability (K7M): the class \(\mathcal F\) contains a Bayes-optimal classifier, meaning a classifier that minimizes task risk over all measurable predictors. Classical surrogate consistency can fail even under this requirement, while aggregation succeeds.

Beyond this restriction, Nguyen et al. [31] for binary classification and Duchi et al. [12] for multiclass classification show that jointly inferring a predictor \(f\) and a representation of \(x\) forces a task-loss-dependent surrogate form. Their decision rules remain infinitely powerful. New approaches are needed.

##### Restricted-class comparison with aggregation.

Nonparametric comparison inequalities mitigate regret caused by using a surrogate. The parametric results establish statistical consistency. Recent \(H\)-consistency work [2, 28, 27, 29] suggests extending aggregation comparisons to restricted classes.

Mao et al. [27] establish rate-optimal quadratic growth of \(\psi\) for smooth convex loss. Aggregation-based upgrades in \(\mathcal H\)-consistency, where \(\mathcal H\) denotes the hypothesis class, remain of interest. An extension could unify parametric and nonparametric advantages and characterize how hypothesis-class “size” affects aggregation.
