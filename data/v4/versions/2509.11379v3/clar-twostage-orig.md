# Some Robustness Properties of Label Cleaning

Chen Cheng

  
John Duchi

###### Abstract

We show that learning procedures based on aggregated labels—label information distilled from noisy responses—have robustness properties that are impossible without label cleaning.

The first property concerns surrogate risk consistency. A task loss measures the error we want to minimize, such as zero-one misclassification error. A surrogate loss is a replacement loss used because it is easier to optimize. It is typically convex: its value between two points does not exceed the corresponding linear interpolation of its endpoint values. Consistency asks whether minimizing surrogate risk also minimizes task risk. Procedures using aggregated labels obtain stronger consistency guarantees than are possible with raw labels. The ranking results in Propositions 1–2 and the comparison bounds in Theorem 1 demonstrate this difference.

The second property concerns model mis-specification. A perfectly specified model matches the true data-generating model. Classical scenarios suggest that incorporating all available information, including label uncertainty, is statistically efficient: it uses the information to estimate parameters precisely. But standard approaches can lose consistency when the loss being minimized is even slightly mis-specified. Procedures using aggregated information still converge to optimal classifiers; Section 4 gives the corresponding results. These results show how considering the full pipeline—from collection through model fitting to prediction—can produce more robust methods by refining noisy signals.

## 1 Introduction

Consider data collection for supervised learning, where features are used to predict labels. The simple description is that we collect pairs \((X_i,Y_i)_{i=1}^n\), fit a model, and make predictions [17]. Here \(n\) is the number of observations, \(i\) indexes observations, \(X_i\) contains the features, and \(Y_i\) is the label.

This description hides much of the work required by modern datasets [9, 22, 39]. Dataset creation involves substantial cleaning and filtering. It often involves crowdsourcing—collecting responses from multiple annotators—and then denoising their labels. The crowdsourcing community has studied this process extensively, especially for obtaining “gold standard,” or reference-quality, labels [8, 48, 47, 46, 37]. We take a complementary view. We study how aggregation fundamentally and necessarily improves the consistency of fitted models.

Our argument is that label cleaning, or aggregating labels together, provides robustness that is impossible without aggregation. This robustness has two forms.

First, aggregation improves consistency when we minimize a surrogate loss instead of a task loss. For example, the surrogate may be multiclass logistic loss, a logarithmic loss for choosing among several classes, while the task loss is zero-one error, which records whether a prediction is wrong. Aggregated labels can yield consistent and optimal prediction in the limit when raw labels cannot. Propositions 1–2 give a ranking example.

Second, aggregation can yield consistent classifiers even in finite-dimensional statistical problems, where a model is described by finitely many parameters. Standard methods can fail in these problems. Proposition 4 and Corollary 4.1 give a binary-classification comparison; Proposition 5 and Theorem 3 give a multiclass comparison.

Important work on surrogate risk consistency dates to the 2000s [52, 24, 40]. Bartlett, Jordan and McAuliffe [4] characterized when fitting a convex surrogate is consistent for binary classification under zero-one error. Later work studied multilabel classification, where an output can contain several labels [50, 43, 16, 49, 1]; ranking, which orders items [13, 14, 36]; structured prediction, which predicts outputs with internal structure [34, 6, 32]; ordinal regression, which predicts ordered categories [35]; restricted hypothesis classes [2, 3, 28, 29]; and general theory [40]. A hypothesis class is the collection of prediction functions available to the learner.

These analyses use the standard setting of feature-label pairs \((X,Y)\). This permits the use of empirical-process theory, which studies how sample averages relate to population averages [45, 5, 21, 4]. But it does not address the aggregation machinery now common in dataset creation.

This raises several questions. Do we need to aggregate at all? If raw labels permit surrogate consistency, perhaps we should rely on the mature theory for \((X,Y)\) pairs. If we do aggregate, does aggregation improve consistency? In what sense? These questions motivate this paper.

We give examples in ranking and in binary and multiclass classification with linear estimators. In these examples, estimators using only \((X,Y)\) pairs necessarily fail, while aggregation methods are consistent. We also develop notions and theory for surrogate consistency with aggregation.

In fully nonparametric settings, where the prediction function is not restricted to a finite-dimensional model, we show how the number of aggregated labels combines with label-noise conditions to improve consistency. Theorem 1 gives this relationship. Under weak conditions on the surrogate, aggregated-label losses admit approximate linear comparison inequalities: above a threshold, task excess risk is bounded by a constant times surrogate excess risk.

Conventional risk-consistency theory takes the hypothesis class \(\mathcal F\) to contain all measurable scoring functions, meaning functions for which the probabilities and expectations used below are defined. We also study restricted classes. In classification, aggregation can guarantee consistency over such classes when fitting raw labels can fail; Section 4 develops these results.

## 2 Preliminaries

We first review classical surrogate risk minimization. Let \(\mathcal X\) be the input space and \(\mathcal Y\) the output space. The observations \((X_i,Y_i)_{i=1}^n\) are drawn independently and identically distributed, abbreviated i.i.d., from a probability distribution \(P\) on \(\mathcal X\times\mathcal Y\).

We use the following notation. The sets \(\mathbb R\) and \(\mathbb R_+\) are the real and nonnegative real numbers. The notation \(\mathbb R^d\) denotes real vectors with \(d\) coordinates. Expectations and probabilities are written \(\mathbb E\) and \(\mathbb P\); a subscript \(P\) identifies the distribution used. A vertical bar denotes conditioning, as in \(P(Y\in\cdot\mid X=x)\), the distribution of the label when the input is \(x\). The notation \(1\{B\}\) is the indicator of a condition \(B\): it is one when \(B\) holds and zero otherwise. The operators \(\inf\), \(\argmin\), and \(\argmax\) denote the greatest lower bound, the set of minimizers, and the set of maximizers, respectively. A chosen minimizer or maximizer is used when a single output is required.

A scoring function \(f:\mathcal X\to\mathbb R^d\) maps an input \(x\) to a score vector \(s=f(x)\), where \(d\geq1\) is the score dimension. A decoder \(\mathsf d:\mathbb R^d\to\mathcal Y\) turns scores into a prediction:
\[
\widehat y=\mathsf d\circ f(x).
\]
Here \(\circ\) denotes function composition, so this means \(\mathsf d(f(x))\).

Let \(\ell:\mathcal Y\times\mathcal Y\to\mathbb R_+\) be a task loss and \(\mathcal F\) a hypothesis class. The goal is to minimize the task risk, or expected task loss, over \(f\in\mathcal F\):
\[
R(f):=\mathbb E_P\left[\ell(\mathsf d\circ f(X),Y)\right].
\tag{1}
\]

For binary classification, \(d=1\), \(\mathsf d(s)=\mathsf{sgn}(s)\), and
\[
\ell(y,y')=1\{yy'\leq0\}.
\]
The sign function \(\mathsf{sgn}\) identifies the sign of a scalar. This choice gives
\[
R(f)=\mathbb P(Yf(X)\leq0).
\]

Minimizing \(R(f)\) is difficult because the task loss can be nonsmooth, nonconvex, and uninformative. In particular, the zero-one loss is flat almost everywhere. This makes even practical optimization methods, such as first-order methods that use derivatives, impossible.

We allow a more general version of (1). Instead of depending only on the observed label \(Y\), the loss may depend on the full conditional label distribution \(P(Y\in\cdot\mid X)\). We therefore consider
\[
R(f)\coloneqq
\mathbb E_P\left[
\ell(\mathsf d\circ f(X),P(\cdot\mid X))
\right].
\tag{2}
\]
The symbol \(\coloneqq\), like \(:=\), introduces a definition. The minimizers frequently coincide with those of the original problem (2), while this formulation permits more sophisticated tasks.

For example, in multiclass classification \(Y\in\{1,\ldots,k\}\), where \(k\) is the number of classes. Taking
\[
\ell(\widehat y,P)=
\sum_yP(Y=y)1\{\widehat y\neq y\}
\]
makes (2) equal to the standard zero-one error rate.

Rather than minimize (2) directly, we use an easier-to-optimize surrogate loss
\(\varphi:\mathbb R^d\times\mathcal Y\to\mathbb R\). Its surrogate risk is
\[
R_\varphi(f):=\mathbb E_P[\varphi(f(X),Y)].
\]
This replacement is sensible only if minimizing surrogate risk is consistent with minimizing the task risk (2).

We study two settings:

(i) **Classical Fisher consistency.** The class \(\mathcal F\) contains all Borel functions: functions measurable with respect to the Borel sets generated by open sets. Section 3 states the consistency requirement precisely.

(ii) **Parametric, possibly mis-specified models.** The class \(\mathcal F\) is described by a parameter vector but may not correctly represent the true model.

Our main message is that aggregation improves consistency in both settings.

### 2.1 Label aggregation

Replace the single output \(Y\) by a more abstract observation \(Z\in\mathcal Z\), where \(\mathcal Z\) is its observation space. If we collect \(m\) noisy labels for each input, then
\[
Z=(Y_1,\ldots,Y_m)\in\mathcal Y^m.
\]
Here \(\mathcal Y^m\) is the space of \(m\)-tuples of labels. Mao et al. [28] consider a similar multilabeled data structure for multilabel losses. Our focus is the aggregation pipeline.

Let \(\mathcal A\) be an aggregation space, and let \(A:\mathcal Z\to\mathcal A\) be an aggregator, such as majority vote. A surrogate \(\varphi:\mathbb R^d\times\mathcal A\to\mathbb R_+\) takes scores and aggregated responses as inputs. Define the aggregated surrogate risk by
\[
R_{\varphi,A}(f):=
\mathbb E[\varphi(f(X),A(Z))].
\tag{3}
\]
We ask when minimizing (3) is sufficient to minimize the task risk (2).

**Example 1 (Majority vote):**

In repeated sampling, the data at an input are
\[
Z=(Y_1,\ldots,Y_m),\qquad
Y_i\mid X=x\stackrel{\mathrm{iid}}{\sim}P_{Y\mid X=x}.
\]
The symbol \(\sim\) means “is distributed according to.” Define \(A_m(Z)\) as the empirical task-loss minimizer:
\[
A(Z)=A(\{Y_1,\ldots,Y_m\})
=\argmin_{y\in\mathcal Y}\sum_{l=1}^m\ell(y,Y_l).
\]
“Empirical” means that the loss is summed over the observed labels rather than averaged over their population distribution. When
\(\ell(\widehat y,y)=1\{y\neq\widehat y\}\), this is exactly majority vote. The general form permits other procedures. \(\Diamond\)

We can also roughly represent aggregation using nearby examples.

**Example 2 (\(K\)-nearest neighbors):**

Suppose an example \(X\) comes with a label \(Y\) and \(m\) additional i.i.d. observations:
\[
(X_i,Y_i)_{i=1}^m\stackrel{\mathrm{iid}}{\sim}P,\qquad
Z=(Y,(X_i,Y_i)_{i=1}^m).
\]
Here \(m\) counts additional examples, rather than repeated labels at the same input.

Let \(\operatorname{dist}:\mathcal X\times\mathcal X\to\mathbb R_+\) be a distance metric. Order the additional inputs by distance from \(X\):
\[
\operatorname{dist}(X,X_{(1)})
\leq\cdots\leq
\operatorname{dist}(X,X_{(m)}),
\]
and let \(X_{(0)}=X\). The label \(Y_{(l)}\) is paired with \(X_{(l)}\), with \(Y_{(0)}=Y\). For \(K\geq0\), the \(K\)-nearest-neighbor rule uses the \(K\) closest additional inputs. One possible aggregator is
\[
A_{m,K}(Z)\coloneqq
\argmin_{y\in\mathcal Y}
\sum_{l=0}^K\ell(y,Y_{(l)}).
\]
Appendix C applies the coming results beyond this population-level setup, to aggregation from a single sample \((X_i,Y_i)_{i=1}^n\). \(\Diamond\)

## 3 Surrogate consistency

The standard framework [40] takes \(\mathcal F\) to be all Borel measurable functions \(f:\mathcal X\to\mathbb R^d\). For a score \(s\in\mathbb R^d\) at input \(x\), define the conditional task and surrogate risks:
\[
R(s\mid x)\coloneqq
\ell(\mathsf d\circ s,P(Y\in\cdot\mid X=x)),
\qquad
R_\varphi(s\mid x):=
\mathbb E[\varphi(s,Y)\mid X=x].
\]

The pointwise excess risks measure the difference between a score’s conditional risk and the smallest possible conditional risk:
\[
\delta_\ell(s,x)\coloneqq
R(s\mid x)-\inf_{s'\in\mathbb R^d}R(s'\mid x),
\]
\[
\delta_\varphi(s,x)\coloneqq
R_\varphi(s\mid x)-\inf_{s'\in\mathbb R^d}R_\varphi(s'\mid x).
\]
The superscript \(\star\) marks a minimal population risk:
\[
R^\star:=\inf_{f\in\mathcal F}R(f),
\qquad
R_\varphi^\star:=\inf_{f\in\mathcal F}R_\varphi(f).
\]

Following the standard approach [40, 4, 52], we require at least Fisher consistency and, if possible, a quantitative uniform comparison inequality. Both requirements below apply to all data distributions \(P\).

(i) **Fisher consistency:** For every sequence \(f_n\in\mathcal F\),
\[
R_\varphi(f_n)\to R_\varphi^\star
\quad\Longrightarrow\quad
R(f_n)\to R^\star.
\]
The arrow \(\to\) denotes convergence. Thus approaching the smallest surrogate risk must also approach the smallest task risk.

(ii) **Uniform comparison inequality:** There is a nondecreasing function
\(\psi:\mathbb R_+\to\mathbb R_+\), positive at every \(\epsilon>0\), such that
\[
\psi(R(f)-R^\star)
\leq R_\varphi(f)-R_\varphi^\star
\quad\text{for every }f\in\mathcal F.
\]
The function \(\psi\) quantitatively compares task and surrogate excess risks. “Nondecreasing” means its value cannot decrease when its argument increases.

For binary classification with a convex, margin-based surrogate, these two consistency notions coincide [4]. A margin-based loss depends on a score through the signed margin \(yf(x)\).

The stronger uniform guarantee need not always hold. Calibration functions provide a standard construction:
\[
\psi(\epsilon,x)\coloneqq
\inf_{s\in\mathbb R^d}
\left\{\delta_\varphi(s,x)\mid
\delta_\ell(s,x)\geq\epsilon\right\},
\qquad
\overline\psi(\epsilon)\coloneqq
\inf_{x\in\mathcal X}\psi(\epsilon,x).
\]
The pointwise function asks for the smallest surrogate excess among scores whose task excess is at least \(\epsilon\). The uniform function takes the smallest pointwise value over inputs.

The following result connects calibration, consistency, and comparison inequalities [51, Proposition 25; 40, Theorem 2.8 and Lemma 2.9].

###### Corollary 3.1.

The surrogate \(\varphi\) is Fisher consistent (i) for \(\ell\) if and only if
\[
\psi(\epsilon,x)>0
\quad\text{for all }x\in\mathcal X,\ \epsilon>0.
\]
Let \(\psi\) be the Fenchel biconjugate of \(\overline\psi\): the function obtained by applying the convex-conjugate operation twice, giving its lower-semicontinuous convex envelope. Then
\(\overline\psi(\epsilon)>0\) if and only if \(\psi(\epsilon)>0\), and every measurable \(f\) satisfies
\[
\psi(R(f)-R^\star)
\leq R_\varphi(f)-R_\varphi^\star.
\]

For the general task (2), we would like at least a Fisher-consistent surrogate. Then minimizing
\(R_\varphi(f)=\mathbb E[\varphi(f(X),Y)]\) would minimize \(R(f)\). If this were possible using only pairs \((X,Y)\), rather than tuples \((X,Y_1,\ldots,Y_m)\), we could use the full theory of empirical processes and related statistical tools [45, 5, 21, 4]. Collection would also be simpler: one label per example would suffice for consistent estimation.

Unfortunately, such results are generally impossible. The next ranking example demonstrates this failure and motivates the aggregation theory in Section 3.2.

### 3.1 Fisher consistency failure without label aggregation: ranking

Consider ranking \(k\) items from pairwise comparisons [20, 15, 14, 30]. The label space is
\[
\mathcal Y=\{(i,j):i\neq j,\ 1\leq i,j\leq k\}.
\]
A label \((i,j)\) means that item \(i\) ranks above item \(j\). At each input \(x\), population rank aggregation converts the probabilities
\[
p_{ij}=P(Y=(i,j)\mid x)
\]
into a ranking.

We use a simple comparison-based scheme [20]. Similar negative results hold for more sophisticated schemes. Define a normalized transition matrix \(C_x\in\mathbb R_+^{k\times k}\) by
\[
(C_x)_{ii}=0,\qquad
(C_x)_{ij}=
\frac{p_{ij}}{\sum_{l\neq j}p_{lj}}
\quad\text{for }i\neq j.
\]
Set \(0/0=1/(k-1)\). Then \(C_x\) is stochastic in the sense that its columns sum to one:
\[
C_x^T\mathbf1=\mathbf1.
\]
Here \(T\) denotes transpose, and \(\mathbf1\) is the vector of ones. We rank items using \(C_x\mathbf1\in\mathbb R_+^k\), which measures how often each item is preferred to others.

One can instead use higher matrix powers \(C_x^p\mathbf1\), where \(p\) is the power, or Perron vectors, the leading eigenvectors of the transition matrix [20]. Similar results hold for these choices.

We incorporate decoding directly into the task loss:
\[
\ell(s,C)\coloneqq
\max_{i<j}
1\left\{
(s_i-s_j)(e_i-e_j)^TC\mathbf1\leq0,\ 
(e_i-e_j)^TC\mathbf1\neq0
\right\}.
\]
The vector \(e_i\) is the coordinate vector with one in position \(i\) and zeros elsewhere. The loss penalizes scores \(s\) that misorder any pair ordered by \(C\mathbf1\). The population task risk is
\[
R(f)\coloneqq
\mathbb P\left(
f(X)\text{ and }C_X\mathbf1\text{ order differently}
\right).
\tag{4}
\]

Now consider a convex surrogate
\(\varphi:\mathbb R^k\times\mathcal Y\to\mathbb R\). Restrict scores to
\(s^T\mathbf1=0\), so their coordinates sum to zero. This familiar restriction is minor [51, 43]: decoding a ranking requires only the ordering of the scores. No convex surrogate is Fisher consistent for (4).

###### Proposition 1.

For the ranking task (4) with \(k\geq3\) outcomes, every surrogate
\(\varphi:\mathbb R^k\times\mathcal Y\to\mathbb R\) that is convex in its first argument is not Fisher consistent.

The proof is in Appendix A.1.

Aggregation restores consistency as soon as the number \(m\) of collected comparisons satisfies \(m\geq k\). The idea is to regress predicted scores \(f(x)\) on label-ordering frequencies.

Assume that \(Z=(Y_1,\ldots,Y_m)\) contains independent pairwise comparisons conditional on \(X\). Let
\[
m_{ij}=\sum_{y\in\mathcal Y^m}1\{y=(i,j)\},
\qquad
m_j=\sum_{i=1}^km_{ij}.
\]
Write \([k]=\{1,\ldots,k\}\). Define
\[
A(Z)=
\begin{cases}
\star,
&\text{if }m_j=0\text{ for some }j\in[k],\\[3pt]
\left(
\frac{m_{i1}}{m_1}+\frac{m_{i2}}{m_2}
+\cdots+\frac{m_{ik}}{m_k}
\right)_{i\in[k]},
&\text{otherwise, that is, if }m_j>0
\text{ for all }j\in[k].
\end{cases}
\]
The response \(\star\) marks the case in which a denominator is zero. Regressing on \(A(Z)\) when \(A(Z)\neq\star\) yields consistency.

###### Proposition 2.

Define
\[
\varphi(s,q)=\|s-q\|_2^2
\quad\text{for }s,q\in\mathbb R^k,
\qquad
\varphi(s,\star)=0.
\]
Here \(q\) is the aggregated response vector, and \(\|\cdot\|_2\) is the Euclidean norm, the square root of the sum of squared coordinates. If \(m\geq k\), then \(\varphi\) is Fisher consistent for the ranking risk (4).

The proof is in Appendix A.2. Aggregation expands the response space from \(\mathcal Y\) to \(\mathbb R^k\). This expansion explains why Proposition 1 gives an impossibility result while Proposition 2 guarantees consistency.

### 3.2 Label aggregation obtains stronger surrogate consistency

The ranking example suggests broader benefits from aggregation. We give two types of results. The first extends standard surrogate-risk consistency to aggregated observations. The second shows how aggregation can upgrade a nominally inconsistent loss into a consistent one, as in Proposition 2.

#### 3.2.1 Basic extensions of surrogate consistency

Let \(A:\mathcal Z\to\mathcal A\) be any aggregation method. Its conditional surrogate risk is
\[
R_{\varphi,A}(s\mid x)\coloneqq
\mathbb E[\varphi(s,A(Z))\mid X=x].
\]
Define the corresponding pointwise excess risk:
\[
\delta_{\varphi,A}(s,x):=
R_{\varphi,A}(s\mid x)
-\inf_{s\in\mathbb R^d}R_{\varphi,A}(s\mid x).
\]
The pointwise and uniform calibration functions are
\[
\overline\psi_A(\epsilon,x)\coloneqq
\inf_{s\in\mathbb R^d}
\left\{
\delta_{\varphi,A}(s,x)\mid
\delta_\ell(s,x)\geq\epsilon
\right\},
\qquad
\overline\psi_A(\epsilon)\coloneqq
\inf_{s\in\mathbb R^d}\overline\psi_A(\epsilon,x).
\tag{5}
\]

Under appropriate measurability conditions, this construction gives an analogue of Corollary 3.1. We leave these conditions tacit because they are not central to our results. The next result follows essentially from Steinwart [40, Theorem 2.8]; Appendix B.1 includes a proof.

###### Proposition 3.

Assume there is a nonnegative function \(b:\mathcal X\to\mathbb R_+\) such that
\[
\int b(x)\,dP(x)<\infty,
\qquad
\delta_\ell(f(x),x)\leq b(x).
\]
The integral condition says that the bound \(b\) has finite expectation. Then \(\varphi\) is Fisher consistent (i) for the task risk (2) if and only if
\[
\overline\psi_A(\epsilon,x)>0
\quad\text{for all }x\in\mathcal X,\ \epsilon>0.
\]
Also, if
\[
\psi_A=(\overline\psi_A)^{**}
\]
is the Fenchel biconjugate, with \(**\) denoting the twice-applied conjugate operation, then
\[
\psi_A(R(f)-R^\star)
\leq R_{\varphi,A}(f)-R_{\varphi,A}^\star.
\]
Here \(R_{\varphi,A}^\star\) denotes the infimum of aggregated surrogate risk over \(\mathcal F\).

This result preserves classical consistency guarantees under aggregation. It does not itself show that aggregation improves consistency. We turn to that question next.

#### 3.2.2 Identifying surrogates and consistency

Under essentially minimal surrogate conditions, a generic aggregator gives consistency as the number of observed labels grows. We consider tasks that minimize \(\ell(f(x),y)\), with
\[
R(f)=\mathbb E[\ell(f(X),Y)].
\]
Assume that the label space is finite:
\[
\operatorname{card}(\mathcal Y)=k<\infty.
\]
The operator \(\operatorname{card}\) denotes the number of elements in a set.[^1] We impose a minimal identifiability condition on the surrogate.

[^1]: The example in Section 3.1 belongs to this regime. There, \(k\) denotes ranking length and \(\operatorname{card}(\mathcal Y)=k(k-1)\). With a slight abuse of notation, we henceforth use \(k\) for the cardinality of the label space.

###### Definition 3.1 (Identifying surrogate).

A surrogate \(\varphi:\mathbb R^d\times\mathcal A\to\mathbb R\) is
\((\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})\)-identifying for \(\mathcal Y\), where
\[
0<\mathsf C_{\varphi,1}\leq
\mathsf C_{\varphi,2}<\infty,
\]
if there are aggregated responses
\(\{a_y\}_{y\in\mathcal Y}\subset\mathcal A\) and score vectors
\(\{s_y\}_{y\in\mathcal Y}\subset\mathbb R^d\) such that
\(\mathsf d(s_y)=y\) and, for every \(y\neq y'\),
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

Condition (6a) gives each class \(y\) a response \(a_y\) that identifies it: every score decoded to another class has a loss at least \(\mathsf C_{\varphi,1}\) above the loss at \(s_y\). Condition (6b) bounds how far the loss at \(s_y\) can be above the minimum for another response \(a_{y'}\). A finite \(\mathsf C_{\varphi,2}\) exists for (6b) if \(\varphi(\cdot,a)\) has a finite lower bound.

Definition 3.1 requires neither convexity nor consistency without aggregation, meaning the case \(\mathcal A=\mathcal Y\) and \(Z=Y\). It also imposes no requirement that the decoder \(\mathsf d\) coincide with the pointwise \(\argmax\) operator.

**Example 3:**

Consider binary hinge loss:
\[
\varphi(s,a)=\max\{1-sa,0\},
\qquad
\mathcal A=\mathcal Y=\{\pm1\}.
\]
For \(y\in\{-1,1\}\), take \(a_y=s_y=y\). Then
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

Section 3.3 gives further examples. Given an identifying surrogate and its responses \(\{a_y\}_{y\in\mathcal Y}\), use generalized majority vote:
\[
A_m(y_1,\ldots,y_m)\coloneqq a_{\widehat y},
\qquad
\widehat y=
\argmin_{y\in\mathcal Y}\sum_{i=1}^m\ell(y,y_i),
\tag{7}
\]
with ties broken arbitrarily.

Because \(\mathcal Y\) is finite, i.i.d. labels give the following eventual-correctness property. There is a random \(M<\infty\) such that, for \(m\geq M\),
\[
\argmin_{y\in\mathcal Y}
\left\{\sum_{i=1}^m\ell(y,Y_i)\right\}
\subset
y^\star(x)\coloneqq
\argmin_{y\in\mathcal Y}
\mathbb E[\ell(y,Y)\mid X=x].
\]
The set \(y^\star(x)\) contains the conditionally optimal labels. Thus, as \(m\to\infty\), we expect \(\varphi(\cdot,A_m)\) to become consistent. The coming results imply this formally.

###### Corollary 3.2.

Let \(m=m(n)\to\infty\), and let \(\varphi\) be identifying as in Definition 3.1. Then
\[
R_{\varphi,A_m}(f_n)-R_{\varphi,A_m}^\star\to0
\quad\Longrightarrow\quad
R(f_n)-R^\star\to0.
\]

#### 3.2.3 Identifying surrogates and consistency amplification

When labels have low noise, aggregator (7) improves the comparison inequality
\[
\psi(R(f)-R^\star)\leq R_\varphi(f)-R^\star
\]
by making \(\psi\) linear over a range of \(\epsilon>0\). Linear growth is the strongest comparison inequality possible [34, 32].

Strict comparison inequalities, such as those in Proposition 3, can be too narrow. Inconsistent surrogates may still be convenient in practice [23, 34, 32]. Following Osokin et al. [34], we therefore allow a “good enough” notion: linear growth is required only above a task-excess threshold.

###### Definition 3.2.

The surrogate \(\varphi\) and aggregator \(A\) yield **level-\((\xi,\zeta)\) consistency** if there is a function \(\psi\) satisfying
\[
\psi(\epsilon)\geq\zeta\epsilon
\quad\text{for }\epsilon\geq\xi,
\]
and
\[
\psi(R(f)-R^\star)
\leq
R_{\varphi,A}(f)-R_{\varphi,A}^\star.
\]
Here \(\xi\) is the threshold above which the linear comparison applies, and \(\zeta\) is its slope lower bound.

We show that aggregation (7) achieves level-\((o_m(1),\zeta)\) consistency under minimal assumptions, even for Fisher-inconsistent surrogates. The notation \(o_m(1)\) means a quantity tending to zero as \(m\to\infty\).

We first quantify label noise using Mammen–Tsybakov conditions [25; see also 4]. Define
\[
\Delta(x)\coloneqq
\min_{\mathsf d(s)\notin y^\star(x)}
\delta_\ell(s,x).
\tag{8}
\]
This is the smallest conditional excess risk from an incorrect prediction. In binary classification with \(\mathcal Y=\{\pm1\}\),
\[
\Delta(x)=|2P(Y=1\mid X=x)-1|,
\]
where \(|\cdot|\) denotes absolute value. We expect consistent estimation to be harder when \(\Delta(x)\) is closer to zero.

Let \(f^\star\) denote an optimal scoring function. We use the following Mammen–Tsybakov conditions; the positive constant \(\mathsf C_{\mathsf{MT}}\), whose subscript abbreviates Mammen–Tsybakov, may change:
\[
\mathbb P(\mathsf d\circ f\neq\mathsf d\circ f^\star)
\leq
\mathsf C_{\mathsf{MT}}
(R(f)-R^\star)^\alpha
\quad\text{for all measurable }f.
\tag{Nα}
\]
We call \(\alpha\) the **noise exponent**. An alternative condition is
\[
\mathbb P(\Delta(X)\leq\epsilon)
\leq
(\mathsf C_{\mathsf{MT}}\epsilon)^\beta
\quad\text{for }\epsilon>0.
\tag{Mβ}
\]
The ranges are \(\alpha\in[0,1]\) and \(\beta\in[0,\infty]\). Both conditions hold trivially with \(\alpha=\beta=0\). As in binary classification [4, Theorem 3], they are equivalent through
\[
\beta=\frac{\alpha}{1-\alpha}.
\]
Appendix B.2 gives the argument.

Mao et al. [29] use the same noise statistic for multiclass \(H\)-consistency. Here \(H\)-consistency means relating target excess risk to surrogate excess risk within a specified hypothesis class \(H\). Their structural assumptions on the surrogate and class permit a global convex lower bound. Our setup imposes minimal structural constraints, so we introduce a **noise condition number**:
\[
\kappa(x)\coloneqq
\frac{
\max_{\mathsf d(s)\neq y^\star(x)}\delta_\ell(s,x)
}{
\min_{\mathsf d(s)\neq y^\star(x)}\delta_\ell(s,x)
}.
\tag{9}
\]
It connects \(\Delta(x)\) to pointwise excess risk:
\[
\Delta(x)\geq\frac{\delta_\ell(s,x)}{\kappa(x)}
\quad\text{whenever }\mathsf d(s)\neq y^\star(x).
\]
This permits a finer analysis. In binary classification, \(\kappa(x)=1\) so long as
\(\mathbb P(\Delta(X)>0)=1\).

Together, \(\Delta(x)\) and \(\kappa(x)\) quantify the consistency obtained by generalized majority vote (7) with any identifiable surrogate from Definition 3.1. Define the error function
\[
e_m(t)\coloneqq
t\sqrt{
\frac{2}{m}
\log\left(
\frac{4k(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})}
{\mathsf C_{\varphi,1}}
\right)
}.
\tag{10}
\]
Here \(\log\) is the natural logarithm, and \(t\) is a possible value of the condition number. Roughly, when \(\kappa(x)=t\), majority vote is likely correct if \(m\) makes \(e_m(t)\ll1\), where \(\ll1\) means much smaller than one.

###### Theorem 1.

Let \(\varphi\) be
\((\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})\)-identifying, with responses \(\{a_y\}_{y\in\mathcal Y}\). Let \(A_m\) be aggregator (7). Assume
\[
0\leq\ell\leq1
\]
and that \(P\) satisfies (Nα) with \(\alpha\in[0,1]\).

For every cutoff \(M>0\) and \(f\in\mathcal F\) satisfying
\[
R(f)-R^\star\geq
2\mathbb P(\kappa(X)>M)
+
(4\mathsf C_{\mathsf{MT}}e_m(M))^{\frac1{1-\alpha}},
\]
we have
\[
R(f)-R^\star
\leq
\frac{16}{\mathsf C_{\varphi,1}}
\left(
R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star
\right).
\]

Appendix B.3 proves the theorem. In Definition 3.2’s terms, this is level-\((\xi,\zeta)\) consistency with
\[
\xi=
2\mathbb P(\kappa(X)>M)
+
(4\mathsf C_{\mathsf{MT}}e_m(M))^{\frac1{1-\alpha}},
\qquad
\zeta=\frac{\mathsf C_{\varphi,1}}{16}.
\]

The theorem also proves Corollary 3.2. Define
\[
\xi_m\coloneqq
\inf_M\left\{
2\mathbb P(\kappa(X)>M)
+
(4\mathsf C_{\mathsf{MT}}e_m(M))^{\frac1{1-\alpha}}
\right\}.
\]
Then \(\xi_m\to0\) as \(m\to\infty\). To see the connection, first take \(M\) large: the tail probability
\(\mathbb P(\kappa(X)>M)\) tends to zero as \(M\uparrow\infty\), where \(\uparrow\) denotes increase toward the limit. For any fixed \(M\), \(e_m(M)\to0\) as \(m\) grows. Taking \(\alpha=0\) therefore gives Corollary 3.2.

Theorem 1 is coarse because Definition 3.1 is weak. A closer connection between task loss and surrogate loss could improve the bound. For example, generalized majority vote might be more likely to be correct, or correct enough for \(\varphi\). If \(\varphi\) is nearly consistent without aggregation, we might expect smaller raw-label inconsistency to give stronger bounds in \(m\). We do not pursue these details.

For a more concrete bound, use
\[
\kappa(x)\leq1/\Delta(x).
\]
Take \(M=1\) when \(\operatorname{card}(\mathcal Y)=k=2\), and optimize \(M\) when \(k\geq3\). This lets us lower bound the threshold \(\xi_{m,k}\) for the level-\((\xi_{m,k},\zeta)\) consistency promised by Theorem 1. Set
\[
\xi_{m,k}\coloneqq
\begin{cases}
\left(
\frac{32\mathsf C_{\mathsf{MT}}^2}{m}
\log\left(
\frac{8(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})}
{\mathsf C_{\varphi,1}}
\right)
\right)^{\frac1{2(1-\alpha)}},
&\text{if }k=2,\\[8pt]
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
The algebraic substitutions are in Appendix B.4.

###### Corollary 3.3.

Under Theorem 1’s conditions, every \(f\) satisfying
\[
R(f)-R^\star\geq\xi_{m,k}
\]
also satisfies
\[
R(f)-R^\star
\leq
\frac{16}{\mathsf C_{\varphi,1}}
\left(
R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star
\right).
\]

Corollaries 3.2–3.3 show robustness from label cleaning: minimal surrogate assumptions still permit consistency after aggregation. As \(\alpha\) approaches \(1\), the label count \(m\) required for the comparison inequality at a fixed \(f\) shrinks. In particular, if \(\alpha=1\), then
\[
m\geq
32\max\{\mathsf C_{\mathsf{MT}}^2,\mathsf C_{\mathsf{MT}}^4\}
\log\left(
\frac{4k(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})}
{\mathsf C_{\varphi,1}}
\right)
=O(\log k)
\]
gives \(\xi_{m,k}=0\). The notation \(O(\log k)\) means growth bounded by a constant multiple of \(\log k\). We then obtain the uniform comparison inequality (ii) with a linear comparison function.

Thus the learning problem’s noise level affects how much aggregation consistency requires. An easier problem needs less aggregation to obtain stronger consistency.

### 3.3 Surrogate consistency examples with majority vote

We now instantiate Theorem 1 and Corollary 3.3. Throughout, assume \(P\) has noise exponent \(\alpha\in[0,1]\). This loses no generality because (Nα) always holds with \(\alpha=0\). Proofs of all results in this section are in Appendix B.5.

#### 3.3.1 Binary classification with a nonsmooth surrogate

Consider binary classification with
\[
\varphi(f(x),y)=\phi(yf(x)),
\]
where \(\phi\) is convex. Bartlett et al. [4] show that this surrogate is consistent if and only if \(\phi'(0)<0\), where the prime denotes a derivative.

We give a somewhat trivial illustration: a loss inconsistent without aggregation can become consistent with it. One would never use such a surrogate, so this is a thought experiment.

Assume
\[
\partial\phi(0)\subset(-\infty,0),
\qquad
\lim_{t\to\infty}\phi(t)=0,
\]
and that \(\phi\) is convex. The subgradient set \(\partial\phi(0)\) contains the slopes of supporting lines to \(\phi\) at zero; it permits slopes to be described even when an ordinary derivative does not exist.

###### Lemma 3.1.

For every \(\delta>0\), the surrogate \(\varphi\) is
\((\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})\)-feasible, with the following constants:
\[
\mathsf C_{\varphi,1}
=\phi(0)-\phi(\delta)>0,
\qquad
\mathsf C_{\varphi,2}=\phi(-\delta).
\]
Here \(\delta\) is a positive margin used to choose the constants.

Corollary 3.3 applies with \(k=2\). Thus, for \(f:\mathcal X\to\mathbb R\), the condition
\[
R(f)-R^\star
\geq
\left(
\frac{32\mathsf C_{\mathsf{MT}}^2}{m}
\log\left(
\frac{8(\phi(-\delta)+\phi(0)-\phi(\delta))}
{\phi(0)-\phi(\delta)}
\right)
\right)^{\frac1{2(1-\alpha)}}
\]
implies
\[
R(f)-R^\star
\leq
\frac{16}{\phi(0)-\phi(\delta)}
\left(
R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star
\right).
\]

#### 3.3.2 Bipartite matching

In structured prediction [33], an embedding
\(v:\mathcal Y\to\mathbb R^d\) represents the structure of each output \(y\) as a vector. The structured output space \(\mathcal Y\) is typically large. Use the decoder
\[
\mathsf d(s)=\argmax_{y\in\mathcal Y}\langle s,v(y)\rangle,
\]
where \(\langle\cdot,\cdot\rangle\) is the inner product, the sum of coordinatewise products.

For a task loss \(\ell:\mathcal Y\times\mathcal Y\to\mathbb R_+\) with \(\ell(y,y)=0\), the maximum-margin surrogate, also called generalized hinge loss [42, 44, 19], is
\[
\varphi(s,y)=
\max_{\widehat y\in\mathcal Y}
\left(
\ell(\widehat y,y)
+\langle v(\widehat y)-v(y),s\rangle
\right).
\tag{11}
\]
This loss is typically inconsistent, except in certain low-noise cases [34, 32].

We first study bipartite matching. A matching is a graph \(G=(V,E)\), where \(V\) is the vertex set and \(E\) the edge set. The vertices divide into
\[
V=V_1\cup V_2,\qquad
V_1=\{1,\ldots,N\},\qquad
V_2=\{N+1,\ldots,2N\}.
\]
Each of the \(N\) edges joins one unique vertex in \(V_1\) to one unique vertex in \(V_2\). Let \(\mathcal Y\) contain all such matchings. Then
\[
k=\operatorname{card}(\mathcal Y)=N!,
\]
where \(N!\) is the product of the integers from \(1\) through \(N\).

Embed a graph by its edge indicators:
\[
v(G)\coloneqq
\left(1\{(u,v)\in E\}\right)_{u\in V_1,v\in V_2}
\in\mathbb R^{N^2}.
\]
Here \(u,v\) index vertices; \(v(G)\) denotes the embedding. Its dimension is \(d=N^2\).

The task loss counts mistaken edges:
\[
\ell(y_1,y_2)\coloneqq
\frac1{2N}\|v(y_1)-v(y_2)\|_1
=
\frac1{2N}\|v(y_1)-v(y_2)\|_2^2.
\]
The norm \(\|\cdot\|_1\) is the sum of absolute coordinate values.

###### Lemma 3.2.

For bipartite matching on \(2N\) vertices, the structured hinge surrogate (11) is
\((\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})\)-identifying as in Definition 3.1, with
\[
\mathsf C_{\varphi,1}=\frac1N,
\qquad
\mathsf C_{\varphi,2}=2.
\]

Even though there are \(k=N!\) labels, this result allows aggregation-based consistency using structured hinge loss once \(m\) exceeds \(O(N\log N)\).

For example, substitute the lemma’s constants into Corollary 3.3 for \(k\geq3\). Every measurable \(f:\mathcal X\to\mathbb R^d\) satisfying
\[
R(f)-R^\star
\geq
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
16N\left(
R_{\varphi,A_n}(f)-R_{\varphi,A_n}^\star
\right).
\]

#### 3.3.3 Structured prediction

Return to general structured prediction. Assume the decoder can select every label. More precisely, for each \(y\in\mathcal Y\), the score set
\[
\mathcal S(y)\coloneqq
\left\{
s:
\langle v(y),s\rangle>
\langle v(\widehat y),s\rangle
\text{ for }\widehat y\neq y
\right\}
\]
is nonempty. These are the scores that strictly select \(y\).

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
Here \(y_+\) and \(y_-\) range over labels other than \(y\). The quantity \(\tau(y)\) compares loss-to-score-gap ratios for those alternatives.

###### Lemma 3.3.

For structured prediction, the max-margin surrogate (11) is
\((\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})\)-identifiable with
\[
\mathsf C_{\varphi,1}
=\min_{\widehat y\neq y}\ell(\widehat y,y),
\qquad
\mathsf C_{\varphi,2}
=\max_{y\in\mathcal Y}\tau(y)+1.
\]
In particular, if
\[
v(y)\in\{0,1\}^d,
\qquad
\ell(\widehat y,y)=
\frac1{2d}\|v(\widehat y)-v(y)\|_1,
\]
then \(\tau(y)=1\) for all \(y\), and
\(\mathsf C_{\varphi,2}=2\). The notation \(\{0,1\}^d\) means vectors whose coordinates are all zero or one.

As in binary matching, nontrivial consistency guarantees hold once
\[
m\geq\log\operatorname{card}(\mathcal Y).
\]
Since \(0\leq\ell(\cdot,\cdot)\leq1\), Corollary 3.3 applies. For every measurable \(f:\mathcal X\to\mathbb R^d\), the condition
\[
R(f)-R^\star
\geq
\left(
\frac{32\mathsf C_{\mathsf{MT}}^4}{m}
\log\left(
4|\mathcal Y|
\left(
1+
\frac{\max_{\widehat y\in\mathcal Y}\tau(y)+1}
{\min_{\widehat y\neq y}\ell(\widehat y,y)}
\right)
\right)
\right)^{\frac{\alpha}{2(1-\alpha^2)}}
\]
implies
\[
R(f)-R^\star
\leq
\frac8{\min_{\widehat y\neq y}\ell(\widehat y,y)}
\left(
R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star
\right).
\]
Here \(|\mathcal Y|\) is another notation for the number of labels.

## 4 Robustness and consistency for models

Section 3 builds on classical surrogate-risk theory, which takes \(\mathcal F\) to contain all measurable functions. Its results show that aggregation can upgrade an inconsistent raw-label surrogate to level-\((\xi,\zeta)\) consistency.

We now restrict the hypothesis class, often to a parametric class. Consistency is not an issue for a well-specified model, but assuming exact specification is unrealistic and brittle. Quantifying surrogate consistency over only a subset of measurable functions is a long-standing challenge [12, 31].

Recent \(H\)-consistency bounds link target excess risk to surrogate regret within a hypothesis class \(H\) [2, 3, 26, 28, 29]. Regret here means risk above the best risk available in the specified class. These results are primarily surrogate-centric. Our contribution is orthogonal: it depends less on a particular class or surrogate and instead studies how aggregation improves consistency itself.

We show that aggregation gives consistent estimates in settings where consistency can otherwise fail.

### 4.1 Consistency failure for binary classification in finite dimensions

Consider binary classification with
\[
\mathcal Y=\{\pm1\},
\qquad
\ell(\mathsf d(s),y)=1\{ys\leq0\}.
\]
Let
\[
\varphi(s,y)=\phi(sy),
\]
where \(\phi:\mathbb R\to\mathbb R_+\) is convex. Over all measurable functions, this surrogate has Fisher consistency (i) and uniform consistency (ii) if and only if \(\phi'(0)<0\) [4].

We now restrict the class. Classical consistency can fail even when the class contains an optimal classifier. Say \(P\) is **optimally predictable using \(\mathcal F\)** when a function \(f\in\mathcal F\) satisfies
\[
\mathsf{sgn}(f(x))
=
\mathsf{sgn}(P(Y=1\mid X=x)-1/2).
\tag{12}
\]

Let \(\mathcal X=\mathbb R^d\), where \(d\) now denotes feature dimension, and consider linear scores:
\[
\mathcal F=
\{f_\theta:f_\theta(x)=\langle\theta,x\rangle\}_{\theta\in\mathbb R^d}.
\]
The parameter \(\theta\) is a vector of coefficients. If this class optimally predicts \(P\), there is a vector \(\theta^\star\) such that
\[
\mathsf{sgn}(\langle\theta^\star,x\rangle)
=
\mathsf{sgn}(P(Y=1\mid x)-\tfrac12).
\]
Then \(f_{\theta^\star}\) minimizes task risk over all measurable functions. We call \(P\) **optimally predictable along \(\theta^\star\)**.

One might expect a classically Fisher-consistent margin surrogate to remain consistent here. It need not. More strongly, for any nonnegative loss \(\phi\), there is a distribution \(P\) for which the surrogate minimizer
\[
\theta_\varphi=
\argmin_\theta
\mathbb E_P[\varphi(f_\theta(X),Y)]
\]
is essentially orthogonal to \(\theta^\star\).

###### Proposition 4.

For every \(\epsilon>0\) and nonzero \(\theta^\star\in\mathbb R^d\), there is a distribution \(P\) of \((X,Y)\), optimally predictable along \(\theta^\star\), such that every
\[
\theta_\varphi\in
\argmin_\theta R_\varphi(f_\theta)
=
\argmin_\theta
\mathbb E[\phi(f_\theta(X),Y)]
\]
satisfies
\[
R(f_{\theta_\varphi})>R(f_{\theta^\star})
\]
and
\[
|\cos\angle(\theta_\varphi,\theta^\star)|
=
\frac{|\langle\theta_\varphi,\theta^\star\rangle|}
{\|\theta_\varphi\|_2\|\theta^\star\|_2}
\leq\epsilon.
\]
Here \(\angle\) is the angle between two vectors and \(\cos\) its cosine. The displayed ratio measures how closely their directions align.

The proof is in Appendix D.1.

Aggregation can avoid this failure. To allow inexact optimization, define the set of \(\epsilon\)-approximate minimizers of an objective \(g\):
\[
\epsilon\textup{-argmin}\,g
=
\epsilon\textup{-argmin}_\theta g(\theta)
\coloneqq
\left\{
\theta:
g(\theta)\leq\inf_\theta g(\theta)+\epsilon
\right\}.
\]
Thus \(\epsilon\) is an allowed gap above the smallest objective value.

Suppose collection gives independent labels \(Z=(Y_1,\ldots,Y_m)\), and \(A_m(Z)\) is their majority vote. For an error sequence \(\epsilon_m\), take
\[
\theta_m\in
\epsilon_m\textup{-argmin}_\theta
R_{\varphi,A_m}(f_\theta)
=
\epsilon_m\textup{-argmin}_\theta
\mathbb E\left[
\phi(A_m(Y_1,\ldots,Y_m)f_\theta(X))
\right].
\]
The coming Theorem 2 gives asymptotic consistency as \(m\to\infty\).

###### Corollary 4.1.

Let \(P\) be optimally predictable along \(\theta^\star\). If
\(\epsilon_m\to0\) as \(m\to\infty\), then
\[
R(f_{\theta_m})\to R(f_{\theta^\star}),
\qquad
\cos\angle(\theta_m,\theta^\star)\to1.
\]

Thus raw-label surrogate minimization over linear predictors can be essentially arbitrarily incorrect, as Proposition 4 shows. Aggregation retains consistency.

### 4.2 Aggregation, consistency, and restricted hypothesis classes

Proposition 4 shows how strongly ordinary surrogate consistency depends on having all measurable functions available.

Now consider multiclass classification with \(\mathcal Y=\{1,\ldots,k\}\). Let \(\mathcal F\) be a linear cone satisfying
\[
f(x)^T\mathbf1=0,
\qquad
tf\in\mathcal F
\quad\text{for }t>0\text{ whenever }f\in\mathcal F.
\]
The cone property means that positively rescaling an available function leaves it in the class.

Use zero-one loss and the largest-score decoder:
\[
\ell(y,y')=1\{y\neq y'\},
\qquad
\mathsf d(s)=\argmax_{y\in[k]}s_y.
\]
The zero-sum normalization is immaterial for this decoder.

Assume that \(\varphi:\mathbb R^k\times[k]\to\mathbb R_+\) is Fisher consistent (i) and has the following limiting-loss property:
\[
\varphi(s,y)\to0
\quad\text{if}\quad
s_y-s_j\to+\infty
\text{ for every }j\neq y.
\tag{13}
\]
Thus the loss tends to zero when the correct class’s score exceeds every other score by a diverging margin.

Familiar examples include multiclass logistic loss,
\[
\varphi(s,y)=
\log\left(\sum_{j=1}^ke^{s_j-s_y}\right),
\]
where \(e^t=\exp(t)\) is the exponential function, and losses of the form
\[
\varphi(s,y)=\sum_{i\neq y}\phi(s_y-s_i),
\]
where \(\phi\) is convex and non-increasing, with
\[
\phi'(0)<0,
\qquad
\inf_t\phi(t)=0.
\]
Non-increasing means that its value cannot increase as its argument increases. Zhang [51, Theorem 8] shows that these losses are consistent over
\[
\mathcal F=
\{f:\mathcal X\to\mathbb R^k:\mathbf1^Tf=0\}.
\]

The binary margin-based setting in Section 4.1 is included by taking
\(f(x)=(g(x),-g(x))\) for a measurable scalar function \(g\). A parametric example is the class of linear functions
\[
f(x)=
(\langle\theta_1,x\rangle,\ldots,\langle\theta_k,x\rangle),
\qquad
\sum_{i=1}^k\theta_i=0.
\]
This is a convex cone: it is also closed under convex combinations, or weighted averages with nonnegative weights summing to one.

Without aggregation, such examples typically need strong structural assumptions, including convexity, for consistency. With aggregation, the minimal limiting condition (13) permits asymptotic consistency.

Extend optimal predictability (12) to multiclass problems. We say that \(\mathcal F\) **can optimally predict \(P\)** if some \(f\in\mathcal F\) satisfies
\[
\argmax_y f_y(x)
\in
\argmax_yP(Y=y\mid X=x)
\quad\text{for every }x.
\]

Suppose \(\mathcal Z=\mathcal Y^m\), aggregation is majority vote \(A_m\), and there is a unique conditionally optimal label
\[
y^\star(x)=\argmax_yP(Y=y\mid x).
\]
Then surrogate minimization is consistent whenever \(\mathcal F\) can optimally predict \(P\).

###### Theorem 2.

Let \(\mathcal F\) be a cone that optimally predicts \(P\). Assume the minimal excess risk (8) satisfies
\[
P(\Delta(X)>0)=1.
\]
Let \(\epsilon_m\geq0\) and \(\epsilon_m\to0\). For any sequence
\[
f_m\in
\epsilon_m\textup{-argmin}_{f\in\mathcal F}
R_{\varphi,A_m}(f)
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

Appendix D.2 gives the proof.

Theorem 2 provides asymptotic consistency as \(m\to\infty\) for a broad class of surrogate problems whose hypothesis class is a linear cone. Proposition 4 gives the contrast: even for binary classification, a restricted class containing the optimal predictor can have an arbitrarily uninformative raw-label surrogate.

### 4.3 On finite-dimensional multiclass classification

We finish the technical results with multiclass linear prediction under possible model mis-specification. Let \(\mathcal X=\mathbb R^d\) and \(\mathcal Y=[k]\). The results give a more specialized view than Proposition 4 and Theorem 2.

Even when the problem is optimally predictable and the linear model is well-specified except on an \(\epsilon\)-fraction of the data, surrogate fitting from \((X,Y)\) pairs is inconsistent. Majority vote, however, recovers the optimal linear predictor. Proposition 5 and Theorem 3 give these results.

Let the parameter matrix be
\[
\Theta=[\theta_1~\cdots~\theta_{k-1}]
\in\mathbb R^{d\times(k-1)}.
\]
Assume
\[
Y\mid X=x\sim\mathsf{Cat}(p_\Theta(x)),
\]
where \(\mathsf{Cat}\) denotes a categorical distribution: it assigns the probabilities in the vector \(p_\Theta(x)\) to the \(k\) labels. Thus \(p_\Theta(x)\in\mathbb R_+^k\) and
\(\mathbf1^\top p_\Theta(x)=1\).

A link function \(\sigma\) maps the linear scores to probabilities:
\[
p_\Theta(x)=
\begin{bmatrix}
\sigma(\langle\theta_1,x\rangle,\ldots,
\langle\theta_{k-1},x\rangle)\\
1-\mathbf1^\top
\sigma(\langle\theta_1,x\rangle,\ldots,
\langle\theta_{k-1},x\rangle)
\end{bmatrix},
\]
where
\[
\sigma:\mathbb R^{k-1}\to\mathbb R_+^{k-1},
\qquad
\mathbf1^\top\sigma\leq1.
\]
The last class receives the remaining probability.

Assume the link preserves the maximizing classes. Specifically, for a score vector \(t\in\mathbb R^{k-1}\), set \(t_k=0\) and
\(\sigma_k(t)=1-\mathbf1^\top\sigma(t)\), and require
\[
t_y=\max_{1\leq i\leq k}t_i
\quad\Longleftrightarrow\quad
\sigma_y(t)=\max_{1\leq i\leq k}\sigma_i(t).
\tag{14}
\]
A standard example is the multiclass logistic-regression link:
\[
\sigma_i^{\textup{lr}}(t)=
\frac{e^{t_i}}
{1+e^{t_1}+\cdots+e^{t_{k-1}}}.
\]
The superscript \(\textup{lr}\) abbreviates logistic regression. This normalized-exponential link is also called softmax when the baseline score is included.

Let the multilabeled dataset be
\[
\{(X_i;(Y_{1i},\ldots,Y_{im}))\}_{i=1}^n,
\]
with repeated sampling
\[
Y_{ij}\mid X_i
\stackrel{\mathrm{iid}}{\sim}
\mathsf{Cat}(p_{\Theta^\star}(X_i)).
\]
The matrix \(\Theta^\star\) is the true parameter. Let
\[
Y_i^+=A_m(Y_{i1},\ldots,Y_{im})
\]
be majority vote, with arbitrary tie-breaking. Then
\[
Y_i^+\mid X_i
\sim\mathsf{Cat}(p_{\Theta^\star,m}(X_i)).
\]

Let \(\rho_m(t)\) denote the majority-vote probability vector for the first \(k-1\) classes when \(m\) labels are drawn using initial probabilities \(\sigma(t)\). Then
\[
p_{\Theta,m}(x)=
\begin{bmatrix}
\rho_m(\langle\theta_1,x\rangle,\ldots,
\langle\theta_{k-1},x\rangle)\\
1-\mathbf1^\top
\rho_m(\langle\theta_1,x\rangle,\ldots,
\langle\theta_{k-1},x\rangle)
\end{bmatrix}.
\]
The link \(\rho_m\) satisfies the consistency condition (14).

Fit logistic regression using
\[
\varphi(\Theta^\top x,y)=
-\langle\theta_y,x\rangle+
\log\left(
1+\sum_{i=1}^{k-1}
\exp(\langle\theta_i,x\rangle)
\right),
\tag{15}
\]
with baseline parameter \(\theta_k=\mathbf0\), the zero vector. Define the population loss for \(m\)-label majority vote:
\[
L_m(\Theta)=
\mathbb E[\varphi(\Theta^\top X,Y_m^+)].
\]

Writing \(\nabla_{\theta_i}\) for the gradient, or vector of partial derivatives with respect to \(\theta_i\), gives
\[
\nabla_{\theta_i}\varphi(\Theta^\top x,y)
=
-x\left(
1\{y=i\}-\sigma_i^{\textup{lr}}(\Theta^\top x)
\right).
\]
Consequently, a minimizer \(\Theta_m\) of \(L_m\) satisfies
\[
\nabla_\Theta L_m(\Theta_m)
=
\mathbb E\left[
X\left(
\sigma^{\textup{lr}}(\Theta_m^\top X)
-\rho_m({\Theta^\star}^\top X)
\right)^\top
\right]
=0.
\]

Standard statistical results guarantee consistency and efficiency for well-specified models without aggregation. In the binary case, Cheng et al. [7, Proposition 3] show more: if \(m\geq1\), \(k=2\), and
\[
X\sim\mathsf N(0,I_d),
\]
then \(\Theta_m=t_m\Theta^\star\) for some \(t_m>0\), even with a mis-specified link. Here \(\mathsf N(0,I_d)\) denotes a normal distribution with zero mean and identity covariance matrix \(I_d\). The identity matrix has ones on its diagonal and zeros elsewhere.

Thus binary classification remains consistent for every aggregation level \(m\), despite an incorrect link. The direction
\(\Theta^\star/\|\Theta^\star\|_2\) determines classification consistency.

For \(k\geq3\), even slight link mis-specification causes risk-consistency failure. Fix a set
\(\mathcal T_\epsilon\subset\mathbb R^{k-1}\) with Lebesgue measure \(\epsilon\), where Lebesgue measure is ordinary volume in this space. Consider
\[
\sigma^\epsilon(t)=
\sigma^{\textup{lr}}(t)1\{t\notin\mathcal T_\epsilon\}
+\frac1k\mathbf1\cdot1\{t\in\mathcal T_\epsilon\}.
\]
This defines a conditional distribution on \(Y\in\{1,\ldots,k\}\) that samples
\(Y\sim\sigma^{\textup{lr}}(t)\) if \(t\in\mathcal T_\epsilon\), and samples uniformly otherwise. Clearly, \(\sigma^\epsilon\) satisfies (14) and is optimally predictable as in (12).

For \(\epsilon>0\), define
\[
L_{m,\epsilon}(\Theta)\coloneqq
\mathbb E\left[
\mathbb E_{\sigma^\epsilon}
[\varphi(\Theta^\top X,Y_m^+)\mid X]
\right].
\]
This is the population logistic loss based on \(m\)-label majority vote when
\[
Y\mid X=x\sim
\sigma^\epsilon({\Theta^\star}^\top x).
\]
The subscript on the inner expectation specifies the true link used to generate labels. Let
\[
\Theta_m(\epsilon)=\argmin_\Theta L_{m,\epsilon}(\Theta).
\]
Although \(\Theta_1(0)=\Theta^\star\), arbitrarily small positive perturbations can destroy consistency without aggregation.

###### Proposition 5.

Let \(k\geq3\) and set \(\Sigma=I\), with \(I\) denoting the identity matrix. For
\[
Z\sim\mathsf N(0,I_{k-1}),
\]
assume that the linear mapping on matrices \(M\),
\[
M\mapsto DM\coloneqq
\mathbb E\left[
Z\left(
\nabla\sigma^{\textup{lr}}({\Theta^\star}^\top Z)MZ
\right)^\top
\right],
\]
is invertible, meaning that it has an inverse. Here \(Z\) is a Gaussian vector, distinct from the aggregate observation used earlier, and
\(\nabla\sigma^{\textup{lr}}\) is the matrix of derivatives of the link.

Then there is \(\epsilon_0>0\) such that, for every
\(\epsilon\in(0,\epsilon_0)\), some set \(\mathcal T_\epsilon\) has Lebesgue measure at most \(\epsilon\) and satisfies
\[
\frac{\Theta_1(\epsilon)}{\|\Theta_1(\epsilon)\|}
\neq
\frac{\Theta^\star}{\|\Theta^\star\|}.
\]
An unsubscripted \(\|\cdot\|\) denotes a norm; no particular norm is specified here.

The proof is in Appendix E.1.

Majority vote addresses this inconsistency as \(m\to\infty\). It requires no assumption on the true link \(\sigma\) beyond (14). Define
\[
L_{m,\sigma}(\Theta)=
\mathbb E\left[
\mathbb E_\sigma[
\varphi(\Theta^\top X,Y_m^+)\mid X]
\right],
\qquad
\Theta_m=\argmin L_{m,\sigma}(\Theta).
\]

###### Theorem 3.

Let
\[
\Theta^\star=U^\star T^\star,
\]
where \(U^\star\in\mathbb R^{d\times(k-1)}\) is orthogonal, meaning its columns are orthonormal, and
\(T^\star\in\mathbb R^{(k-1)\times(k-1)}\) is nonsingular, meaning invertible.

For every \(m\), there is
\(T_m\in\mathbb R^{(k-1)\times(k-1)}\) such that
\[
\Theta_m=U^\star T_m.
\]
As \(m\to\infty\),
\[
\|T_m\|\to\infty,
\qquad
T_m/\|T_m\|\to T^\star/\|T^\star\|.
\]

Appendix E.2 gives the proof.

Theorem 3 provides further evidence of aggregation’s robustness. It gives asymptotic consistency in mis-specified models so long as a link describes the relationship between \(X\) and \(Y\). The contrast is striking for \(k\geq3\): Proposition 5 shows that methods without aggregation are generally inconsistent.

## 5 Numerical illustrations

We give two experiments that corroborate the theory. The first is a nonparametric example of surrogate-consistency amplification. The second is a finite-dimensional multiclass example in which aggregation recovers asymptotic consistency under link mis-specification.

##### Surrogate consistency amplification for truncated quadratic loss.

In binary classification, consider
\[
\varphi(f(x),y)=\phi(yf(x)),
\qquad
\phi(\delta)=\max\{1-\delta,0\}^2.
\]
The function \(\phi\) is the truncated quadratic loss. Bartlett et al. [4, Example 2] give the exact calibration function in Corollary 3.1:
\[
\psi(\epsilon)=\epsilon^2.
\]

For majority vote with \(m\) labels, apply Lemma 3.1 with \(\delta=1\). Without a low-noise condition, take
\(\alpha=0\) and \(\mathsf C_{\mathsf{MT}}=1\). Define
\[
\begin{aligned}
\psi_m(\epsilon)
&=
\frac{\phi(0)-\phi(\delta)}{16}
\left(
\epsilon-
\sqrt{
\frac{32}{m}
\log\left(
\frac{8(\phi(-\delta)+\phi(0)-\phi(\delta))}
{\phi(0)-\phi(\delta)}
\right)
}
\right)_+\\
&=
\frac1{16}
\left(
\epsilon-\sqrt{\frac{32}{m}\log40}
\right)_+.
\end{aligned}
\]
The positive-part operation is \(u_+=\max\{u,0\}\).

We then have parallel comparison inequalities:
\[
\psi(R(f)-R^\star)
\leq
R_\varphi(f)-R_\varphi^\star,
\]
\[
\psi_m(R(f)-R^\star)
\leq
R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star.
\]

Numerically, the aggregated curve transitions from quadratic toward linear behavior as \(m\to\infty\); see Figure 1. This shows that aggregation upgrades surrogate consistency. The functions \(\psi_m\) are nevertheless generally conservative because they come from the general Theorem 1.

**Figure 1:** Numerical illustrations for truncated quadratic surrogate \(\phi(\delta)=\max\{1-\delta,0\}^{2}\), showing comparison inequalities for non-aggregated data \(\psi\) (optimal) vs. majority vote aggregation \(\psi_m\) (suboptimal) when \(m\in\{2^{16},2^{20},2^{24},2^{28},2^{32}\}\).

##### Asymptotic consistency for mis-specified logistic model.

Consider the logistic-regression setting of Section 4.3 with \(k=3\) and \(d=2\). Use loss (15), with the corrupted logistic link
\[
\widetilde\sigma(t)=
\sigma^{\textup{lr}}(t)1\{\|t\|\leq r\}
+
\sigma^{\textup{lr}}(\alpha t)1\{\|t\|>r\}.
\]
This piecewise link equals softmax near the origin and shrinks scores outside radius \(r\). Here \(\alpha\) is the link’s shrinkage parameter, not the noise exponent used earlier.

Figure 2 reports results for synthetic data from this corrupted link. They show asymptotic consistency as \(m\to\infty\).

| (i) Experiment with constant dataset size \(n\) | (ii) Experiment with constant label size \(nm\). |
|---|---|

**Figure 2:** Numerical illustrations for mis-specified logistic models when \(d=2\) and \(k=3\). The true parameters are \(\theta_1^\star=(0,1)^\top,\theta_2^\star=(1,1)^\top\). The corrupted link parameters are \(r=3\) and \(\alpha=0.1\). The plot shows the error of classification rule \(\left\|\|T_m\|-T^\star/\|T^\star\|\right\|\) vs. number of labels \(m\in\{1,4,16,64,256,1024\}\). (i) In the left panel, for each value of \(m\), the synthetic dataset consists of \(n=10{,}000\) multilabeled observations \((X_i,(Y_{i1},\ldots,Y_{im}))\) sampled from \(\widetilde\sigma\). (ii) In the right panel, for each \(m\), we set \(n=640{,}000/m\) so that the total number of labels remains constant across different values of \(m\). We report a \(95\%\) error bar on \(T=100\) trials for each \(m\). Here \(T\) counts experimental trials, distinct from the matrices \(T_m\) and \(T^\star\).

## 6 Discussion and future work

Whether and how to clean data is a central question in dataset collection. Cheng, Asi and Duchi [7] discuss the apparent tradeoff: using all unaggregated labels improves statistical efficiency when models can represent all uncertainty, but cleaning appears more robust otherwise.

In a related validation setting, Dorner and Hardt [11] argue that comparing binary classifiers is better served by more noisy labels than by cleaned labels. Our paper contributes evidence of fundamental limits to using uncleaned, unaggregated labels for supervised learning, together with robustness improvements from cleaning. The ranking comparison in Propositions 1–2, the amplification bound in Theorem 1, and the model results in Section 4 provide this evidence. Many questions remain.

##### Finite \(m\) results and fundamental limits.

Many of our consistency results require the number \(m\) of aggregated labels to tend to infinity. Although our purpose is to identify ways aggregation improves robustness, reliance on this asymptotic setting may be unsatisfying.

The ranking result in Section 3.1 gives explicit finite-\(m\) consistency: Proposition 2 requires \(m\geq k\). Developing finite-label guarantees further is a natural and important direction.

It would be useful to develop surrogate-consistency theory that depends jointly on the loss pair \((\ell,\varphi)\) and the available label count \(m\). For ranking, do second- or third-order comparisons require more labels? Examples include the matrix powers \(C_x^p\) suggested by Keener [20]. A central challenge is to identify precisely which problems require cleaning and aggregation and which do not.

##### Fundamental limits of the noise condition number.

Our multiclass comparison inequalities use the noise condition number \(\kappa(X)\) from (9). It indicates difficulties beyond binary classification, where \(\kappa(X)=1\).

In multiclass problems, the condition number can remain large even when Mammen–Tsybakov noise is low, meaning \(\alpha\approx1\). This follows from our minimal surrogate assumptions. Closer connections between task loss \(\ell\) and surrogate \(\varphi\) could describe problem difficulty more accurately.

Explicit failure constructions would also help delineate fundamental limits and clarify predictive-model fitting.

##### Behavior in mis-specified models.

Our mis-specification results, especially Section 4.3, require optimal predictability (12). In other words, a Bayes-optimal classifier—a classifier minimizing population task risk—must belong to \(\mathcal F\).

Classical surrogate consistency can fail even under this requirement, while aggregation can succeed. Moving beyond optimal predictability remains a promising direction.

Nguyen et al. [31] and then Duchi et al. [12] give one related direction for binary and multiclass classification, respectively. When a predictor \(f\) and an input representation are inferred jointly, the surrogate \(\varphi\) must have a particular form determined by task loss \(\ell\). But these results still use infinitely powerful decision rules \(f\). New approaches are therefore needed.

##### Restricted class comparison inequality with label aggregation.

Our paper studies how aggregation reduces regret from surrogate losses through nonparametric comparison inequalities. It also establishes statistical consistency in parametric settings.

Recent \(H\)-consistency results [2, 28, 27, 29] motivate extending these techniques to comparison inequalities over restricted classes. Mao et al. [27] show that, for smooth convex loss, quadratic growth of \(\psi\) is rate-optimal: the comparison rate cannot be improved in that setting. It would be interesting to use aggregation to upgrade consistency in the setting of \(\mathcal H\)-consistency, another notation for consistency relative to a hypothesis class.

Such an extension could unify the benefits of aggregation across parametric and nonparametric settings. It could also clarify how the “size,” or richness, of \(\mathcal F\) affects the effectiveness of aggregation.
