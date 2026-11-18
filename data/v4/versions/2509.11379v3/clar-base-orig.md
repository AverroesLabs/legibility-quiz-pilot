# Some Robustness Properties of Label Cleaning

Chen Cheng

  
John Duchi

###### Abstract

We show that learning procedures based on aggregated labels have robustness properties that are impossible without data cleaning. Aggregation combines noisy responses into label information used for learning.

This robustness takes several forms. First, consider risk consistency: whether minimizing an easier surrogate loss also minimizes the desired task loss. A task loss might count classification errors. A surrogate is often convex: its value at a weighted average is no greater than the corresponding weighted average of its values. Procedures using aggregated labels obtain stronger consistency guarantees than are possible with raw labels.

Second, consider statistical models that are perfectly specified, meaning that they correctly describe the data distribution. Classical theory suggests that using all available information, including label uncertainty, is statistically efficient. But “standard” approaches lose consistency as soon as the loss being minimized is even slightly mis-specified. Procedures using aggregated information still converge to optimal classifiers. These results show how considering the full pipeline—from data collection through model fitting to prediction—can produce a more robust method by refining noisy signals.

## 1 Introduction

Consider data collection for supervised learning, where a model learns to predict labels from features. The simple account is that we collect feature-label pairs $(X_i,Y_i)_{i=1}^{n}$, fit a model, and use it [17]. Here, $n$ is the number of examples, $i$ indexes examples, $X_i$ contains the observed features, and $Y_i$ is the corresponding label.

This account hides the complexity of modern datasets [9, 22, 39]. Their construction requires substantial cleaning and filtering. It often involves crowdsourcing several labels for each example and then removing noise from those labels. The crowdsourcing community has studied this process intensively, especially for obtaining “gold standard” labels [8, 48, 47, 46, 37]. We take a complementary view. We study how data aggregation fundamentally and necessarily improves the consistency of fitted models.

Our argument is that label cleaning—combining labels—provides robustness that cannot be achieved without aggregation. This robustness has two forms.

First, aggregation improves estimation consistency. Suppose we minimize a surrogate loss, such as multiclass logistic loss, instead of a task loss, such as zero-one error. Zero-one error counts an incorrect prediction as one and a correct prediction as zero. Aggregated labels can yield consistent, optimal prediction in the limit even when this is impossible without aggregation.

Second, aggregation can yield consistent classifiers in finite-dimensional statistical problems, where models have finitely many parameters, even when standard methods fail.

Important contributions to surrogate risk consistency date to the 2000s [52, 24, 40]. Bartlett, Jordan and McAuliffe [4] characterized when fitting a convex surrogate yields consistency for binary classification under zero-one error. Subsequent work has studied multi-label classification [50, 43, 16, 49, 1], ranking [13, 14, 36], structured prediction [34, 6, 32], ordinal regression [35], restricted hypothesis classes [2, 3, 28, 29], and general theory [40]. Multi-label classification allows several labels for an example. Ranking predicts an ordering. Structured prediction predicts objects with internal structure. Ordinal regression predicts ordered categories. A hypothesis class is the collection of prediction functions available to the learner.

These analyses use the standard supervised-learning setup of pairs $(X,Y)$. This lets us use the statistical theory of empirical processes, which studies fluctuations of sample averages over classes of functions [45, 5, 21, 4]. But the analyses do not address the aggregation machinery now common in dataset creation.

This raises our main questions. Must we aggregate labels at all? If surrogate consistency is possible without aggregation, perhaps we should rely on existing theory for $(X,Y)$ pairs. If we do aggregate, does aggregation improve consistency? In what sense?

We give concrete examples in ranking and in binary and multiclass classification with linear estimators. In these examples, estimators using only $(X,Y)$ pairs necessarily fail, but aggregation methods are consistent. We develop definitions and theory for surrogate consistency with aggregation.

In fully nonparametric settings, where prediction functions are not restricted to a finite-dimensional model, we show how the number of aggregated samples interacts with label-noise conditions. Under weak conditions on the surrogate, aggregated labels also yield approximate linear comparison inequalities: inequalities that bound excess task risk by a constant times excess surrogate risk, above an approximation threshold.

Conventional risk consistency theory takes the hypothesis class $\mathcal F$ to contain all measurable functions—functions compatible with the measurable sets used to define probabilities. We also study restricted classes. In classification, aggregation can guarantee consistency over such classes even when consistency fails without aggregation.

## 2 Preliminaries

We first review classical surrogate risk minimization. Let $\mathcal X$ be the input space and $\mathcal Y$ the output space. The observations $(X_i,Y_i)_{i=1}^{n}$ are drawn independently from the same probability distribution $P$ on $\mathcal X\times\mathcal Y$. We abbreviate “independent and identically distributed” as i.i.d.

A scoring function $f:\mathcal X\to\mathbb R^d$ maps an input $x$ to a score vector $s\in\mathbb R^d$. Here, $\mathbb R$ denotes the real numbers, and $d\geq1$ is the number of score coordinates. A decoder $\mathsf d:\mathbb R^d\to\mathcal Y$ converts scores into a prediction:
\[
\widehat y=\mathsf d\circ f(x).
\]
The symbol $\circ$ denotes function composition, so this means $\widehat y=\mathsf d(f(x))$.

Let $\ell:\mathcal Y\times\mathcal Y\to\mathbb R_+$ be a task loss, where $\mathbb R_+$ denotes the nonnegative real numbers. For a hypothesis class $\mathcal F$, the goal is to minimize the task risk
\[
R(f):=\mathbb E_P\!\left[\ell(\mathsf d\circ f(X),Y)\right],
\qquad f\in\mathcal F.
\tag{1}
\]
Here, $\mathbb E_P$ is expectation under $P$, and $:=$ defines the expression on its left.

For binary classification, $d=1$, $\mathsf d(s)=\mathsf{sgn}(s)$, and
\[
\ell(y,y')=1\{yy'\leq0\}.
\]
The sign function $\mathsf{sgn}$ returns the sign of its argument. The indicator $1\{B\}$ equals one when statement $B$ holds and zero otherwise. Thus,
\[
R(f)=\mathbb P(Yf(X)\leq0),
\]
where $\mathbb P$ denotes probability.

Minimizing $R(f)$ is difficult because $\ell$ can be nonsmooth, nonconvex, and uninformative for optimization. In particular, the $0$-$1$ loss is flat almost everywhere. This makes even practical first-order optimization, which uses derivatives or gradients, impossible.

We consider a more general version of (1). Instead of depending only on the observed label $Y$, the task loss may depend on the conditional label distribution $P(Y\in\cdot\mid X)$. The dot stands for the set whose conditional probability is being evaluated. We study
\[
R(f)\coloneqq
\mathbb E_P\!\left[
\ell(\mathsf d\circ f(X),P(\cdot\mid X))
\right].
\tag{2}
\]
Its minimizers frequently coincide with those of the original problem (2), but this formulation allows more sophisticated losses.

For example, in multiclass classification, $Y\in\{1,\ldots,k\}$, where $k$ is the number of classes. Taking
\[
\ell(\widehat y,P)
=\sum_y P(Y=y)1\{\widehat y\neq y\}
\]
makes the risk equal to the standard $0$-$1$ error rate. The sum runs over labels $y$.

Rather than minimizing (2) directly, we use an easier-to-optimize surrogate loss
$\varphi:\mathbb R^d\times\mathcal Y\to\mathbb R$. Its risk is
\[
R_\varphi(f):=\mathbb E_P\!\left[\varphi(f(X),Y)\right].
\]

This substitution requires a consistency guarantee linking surrogate minimization to task minimization. We study two settings, formalized below:

(i) The classical setting of Fisher consistency, where $\mathcal F$ contains all Borel functions. Borel functions are measurable with respect to the Borel sets of the relevant spaces.

(ii) Statistical settings where $\mathcal F$ is parametric, meaning that functions are indexed by parameters, but the model may be mis-specified: it may not correctly describe the data.

Our main message is that aggregation improves consistency in both settings.

### 2.1 Label aggregation

Replace the single output $Y$ with a more general observation $Z\in\mathcal Z$, where $\mathcal Z$ is its possible-observation space. For example, if we collect $m$ noisy labels for each input $X$, then
\[
Z=(Y_1,\ldots,Y_m)\in\mathcal Y^m.
\]
Here, $m$ is the number of collected labels, and $\mathcal Y^m$ is the space of length-$m$ label sequences. Mao et al. [28] consider a similar multiply labeled data structure for multilabel losses. We instead focus on the aggregation pipeline.

Let $\mathcal A$ be an aggregation-output space, and let $A:\mathcal Z\to\mathcal A$ be an aggregation function, such as majority vote. Let
$\varphi:\mathbb R^d\times\mathcal A\to\mathbb R_+$
be a surrogate defined on that space. Define the aggregated surrogate risk by
\[
R_{\varphi,A}(f):=
\mathbb E\!\left[\varphi(f(X),A(Z))\right].
\tag{3}
\]
We ask when minimizing (3) is enough to minimize the task risk (2).

**Example 1 (Majority vote):**

In repeated sampling, we observe $Z=(Y_1,\ldots,Y_m)$, with
\[
Y_i\mid X=x\stackrel{\rm iid}{\sim}P_{Y\mid X=x}.
\]
This notation means that, conditional on $X=x$, the labels are independent draws from the conditional label distribution.

Define $A_m(Z)$ as the empirical loss minimizer
\[
A(Z)=A(\{Y_1,\ldots,Y_m\})
=\mathop{\rm argmin}_{y\in\mathcal Y}
\sum_{l=1}^{m}\ell(y,Y_l).
\]
The notation $\argmin$ denotes a minimizing argument or the set of minimizing arguments, as used in each expression. The index $l$ runs over collected labels.

For $\ell(\widehat y,y)=1\{y\neq\widehat y\}$, this is exactly majority vote. The more general formula allows other aggregation procedures. $\Diamond$

We can also roughly represent $K$-nearest-neighbor aggregation, which combines labels from inputs closest to the input being predicted.

**Example 2 ($K$-nearest neighbors):**

Suppose an input $X$ comes with its label $Y$ and an additional sample
\[
(X_i,Y_i)_{i=1}^{m}\stackrel{\rm iid}{\sim}P.
\]
Thus,
\[
Z=(Y,(X_i,Y_i)_{i=1}^{m}),
\]
where $m$ is the number of additional examples.

Let $\operatorname{dist}:\mathcal X\times\mathcal X\to\mathbb R_+$ be a distance metric. Order the additional inputs by distance from $X$:
\[
\operatorname{dist}(X,X_{(1)})
\leq\cdots\leq
\operatorname{dist}(X,X_{(m)}).
\]
Parenthesized indices indicate this ordering; $Y_{(l)}$ is the corresponding label. Set $X_{(0)}=X$, with corresponding label $Y_{(0)}=Y$.

For $K\geq0$, aggregate the $K$ nearest neighbors, together with the original example, by
\[
A_{m,K}(Z)\coloneqq
\mathop{\rm argmin}_{y\in\mathcal Y}
\sum_{l=0}^{K}\ell(y,Y_{(l)}).
\]

Appendix C uses the results below to move beyond this population-level setup, defined through draws from $P$, and address aggregation within a single sample $(X_i,Y_i)_{i=1}^{n}$. $\Diamond$

## 3 Surrogate consistency

The standard framework [40] takes $\mathcal F$ to contain all Borel measurable functions $f:\mathcal X\to\mathbb R^d$.

For a fixed input $x$ and score $s\in\mathbb R^d$, define the conditional task and surrogate risks:
\[
R(s\mid x)\coloneqq
\ell(\mathsf d\circ s,P(Y\in\cdot\mid X=x)),
\qquad
R_\varphi(s\mid x):=
\mathbb E[\varphi(s,Y)\mid X=x].
\]
Here, $\mathsf d\circ s$ denotes decoding the score $s$.

Define their pointwise excess risks—the amount above the best conditional risk—as
\[
\delta_\ell(s,x)\coloneqq
R(s\mid x)-\inf_{s'\in\mathbb R^d}R(s'\mid x),
\qquad
\delta_\varphi(s,x)\coloneqq
R_\varphi(s\mid x)-\inf_{s'\in\mathbb R^d}R_\varphi(s'\mid x).
\]
The infimum $\inf$ is the greatest lower bound, whether or not attained. The vector $s'$ ranges over alternative scores.

The minimal overall risks are
\[
R^\star:=\inf_{f\in\mathcal F}R(f),
\qquad
R_\varphi^\star:=\inf_{f\in\mathcal F}R_\varphi(f).
\]
A superscript $\star$ marks an optimal value or object.

Following standard usage [40, 4, 52], we seek at least (i) Fisher consistency and, if possible, (ii) a stronger quantitative, uniform comparison inequality. These conditions must hold for every data distribution $P$:

(i) For every sequence $f_n\in\mathcal F$,
\[
R_\varphi(f_n)\to R_\varphi^\star
\quad\Longrightarrow\quad
R(f_n)\to R^\star.
\]
Thus, approaching the optimal surrogate risk must imply approaching the optimal task risk.

(ii) There is a non-decreasing comparison function
$\psi:\mathbb R_+\to\mathbb R_+$ such that, for every $f\in\mathcal F$,
\[
\psi(R(f)-R^\star)
\leq R_\varphi(f)-R_\varphi^\star,
\]
with $\psi(\epsilon)>0$ for every $\epsilon>0$. Here, $\epsilon$ is a positive excess-risk level. “Non-decreasing” means that increasing the argument cannot decrease the function value.

For binary classification with a convex, margin-based surrogate, the two notions coincide [4]. A margin-based loss depends on the product of the binary label and the score.

The stronger uniform guarantee (ii) need not always hold. Calibration functions provide a canonical way to investigate it:
\[
\psi(\epsilon,x)\coloneqq
\inf_{s\in\mathbb R^d}
\left\{\delta_\varphi(s,x)\mid
\delta_\ell(s,x)\geq\epsilon\right\},
\qquad
\overline\psi(\epsilon)\coloneqq
\inf_{x\in\mathcal X}\psi(\epsilon,x).
\]
The first function measures the least surrogate excess risk among scores with task excess risk at least $\epsilon$. The second takes the smallest such value across inputs.

Consistency and comparison inequalities follow from these functions [51, Prop. 25; 40, Thm. 2.8 and Lemma 2.9].

###### Corollary 3.1.

The surrogate $\varphi$ is Fisher consistent (i) for $\ell$ if and only if
$\psi(\epsilon,x)>0$ for every $x\in\mathcal X$ and $\epsilon>0$.

Let $\psi$ be the Fenchel biconjugate of $\overline\psi$, the convex lower-semicontinuous envelope obtained by taking its convex conjugate twice. Then
$\overline\psi(\epsilon)>0$ if and only if $\psi(\epsilon)>0$. For every measurable $f$,
\[
\psi(R(f)-R^\star)
\leq R_\varphi(f)-R_\varphi^\star.
\]

For the general task (2), we would like at least a Fisher-consistent surrogate. Then minimizing
$R_\varphi(f)=\mathbb E[\varphi(f(X),Y)]$
would imply minimizing $R(f)$.

If this were possible using only pairs $(X,Y)$, rather than tuples $(X,Y_1,\ldots,Y_m)$, we could apply the full theory of empirical processes and related statistical tools [45, 5, 21, 4]. Data collection would also be simpler: one label per input would suffice for consistent estimation.

Such guarantees are generally impossible. The next extended example demonstrates this and motivates the aggregation theory in Sec. 3.2.

### 3.1 Fisher consistency failure without label aggregation: ranking

Consider ranking $k$ items from pairwise comparisons [20, 15, 14, 30]. The label space is
\[
\mathcal Y=\{(i,j):i\neq j,\ 1\leq i,j\leq k\}.
\]
A label $(i,j)$ means that item $i$ ranks above item $j$.

For each input $x$, define
\[
p_{ij}=P(Y=(i,j)\mid x).
\]
The population rank-aggregation problem converts these probabilities into a ranking of the $k$ items.

Many aggregation rules are possible. We use a simple comparison-based rule [20, cf.]. Similar negative results hold for more sophisticated rules.

Define the normalized transition matrix $C_x\in\mathbb R_+^{k\times k}$ by $(C_x)_{ii}=0$ and
\[
(C_x)_{ij}
=\frac{p_{ij}}{\sum_{l\neq j}p_{lj}}
\qquad\text{for }i\neq j.
\]
Here, $\mathbb R_+^{k\times k}$ is the space of nonnegative $k$-by-$k$ matrices. Set $0/0=1/(k-1)$. Then $C_x$ is stochastic in the sense that
\[
C_x^T\mathbf1=\mathbf1.
\]
The superscript $T$ denotes transpose, and $\mathbf1$ is the vector of all ones. This equality says that each column sums to one.

Rank the items by the entries of $C_x\mathbf1\in\mathbb R_+^k$. This vector measures how often an item is preferred to others. One could instead use higher matrix powers $C_x^p\mathbf1$, where $p$ is the power, or Perron vectors, eigenvectors associated with the leading eigenvalue [20]. Similar results hold for these alternatives.

Absorb the decoder $\mathsf d$ into the task loss and define
\[
\ell(s,C)\coloneqq
\max_{i<j}
1\!\left\{
(s_i-s_j)(e_i-e_j)^TC\mathbf1\leq0,\ 
(e_i-e_j)^TC\mathbf1\neq0
\right\}.
\]
Here, $s_i$ is score coordinate $i$, and $e_i$ is the vector with one in coordinate $i$ and zero elsewhere. The loss penalizes a score ordering that disagrees with the ordering induced by $C$. The population task risk (2) is
\[
R(f)\coloneqq
\mathbb P\!\left(
f(X)\text{ and }C_X\mathbf1\text{ order differently}
\right).
\tag{4}
\]

Now consider a convex surrogate
$\varphi:\mathbb R^k\times\mathcal Y\to\mathbb R$.
Restrict scores to $s^T\mathbf1=0$, so their coordinates sum to zero. This is a minor restriction familiar from multiclass classification [51, 43]. It is natural because decoding a ranking uses only the ordering of the coordinates.

There is no convex Fisher-consistent surrogate for (4), as established in Appendix A.1.

###### Proposition 1.

Consider ranking with task risk (4) over $k\geq3$ outcomes. If
$\varphi:\mathbb R^k\times\mathcal Y\to\mathbb R$
is convex in its first argument, it is not Fisher consistent.

Aggregation nevertheless yields consistency once the number $m$ of collected comparisons satisfies $m\geq k$. The idea is to regress predicted scores $f(x)$ on frequencies of label orderings—that is, fit scores to those frequency-derived responses.

Assume that $Z=(Y_1,\ldots,Y_m)$ consists of independent pairwise comparisons conditional on $X$. Let
\[
m_{ij}=\sum_{y\in\mathcal Y^m}1\{y=(i,j)\},
\qquad
m_j=\sum_{i=1}^{k}m_{ij}.
\]
Define the aggregation
\[
A(Z)=
\begin{cases}
\star,
&\text{if }m_j=0\text{ for some }j\text{ in }[k],\\[2mm]
\left(
\frac{m_{i1}}{m_1}
+\frac{m_{i2}}{m_2}
+\cdots+
\frac{m_{ik}}{m_k}
\right)_{i\in[k]},
&\text{otherwise, i.e. if }m_j>0\text{ for all }j\in[k].
\end{cases}
\]
Here, $[k]=\{1,\ldots,k\}$, and $\star$ is a special output used when a required denominator is zero.

Regressing directly on $A(Z)$ whenever $A(Z)\neq\star$ yields consistency. Appendix A.2 proves the next result.

###### Proposition 2.

Define
\[
\varphi(s,q)=\|s-q\|_2^2
\qquad(s,q\in\mathbb R^k),
\qquad
\varphi(s,\star)=0.
\]
The Euclidean norm $\|v\|_2$ is the square root of the sum of squared coordinates of $v$. If $m\geq k$, then $\varphi$ is Fisher consistent for the ranking risk (4).

The difference between Propositions 1 and 2 is the response space. Aggregating several labels expands it from $\mathcal Y$ to $\mathbb R^k$.

### 3.2 Label aggregation obtains stronger surrogate consistency

The ranking example suggests that aggregation can improve surrogate consistency. We now study this interaction more generally.

We give two types of results. The first extends standard surrogate-risk consistency to aggregated responses. The second shows how aggregation can “upgrade” an otherwise inconsistent loss into a consistent one, as in Proposition 2.

#### 3.2.1 Basic extensions of surrogate consistency

We first extend calibration functions and the comparison inequalities of Corollary 3.1.

For an arbitrary aggregator $A:\mathcal Z\to\mathcal A$, define the conditional aggregated surrogate risk
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
Define the pointwise and uniform calibration functions by
\[
\overline\psi_A(\epsilon,x)\coloneqq
\inf_{s\in\mathbb R^d}
\left\{\delta_{\varphi,A}(s,x)\mid
\delta_\ell(s,x)\geq\epsilon\right\},
\qquad
\overline\psi_A(\epsilon)\coloneqq
\inf_{s\in\mathbb R^d}\overline\psi_A(\epsilon,x).
\tag{5}
\]

Under appropriate measurability conditions, an analogue of Corollary 3.1 follows. We leave these conditions implicit because they are not central to our results. The next proposition follows essentially from Steinwart [40, Thm. 2.8]. Appendix B.1 includes a proof.

###### Proposition 3.

Assume there is a function $b:\mathcal X\to\mathbb R_+$ such that
\[
\int b(x)\,dP(x)<\infty,
\qquad
\delta_\ell(f(x),x)\leq b(x).
\]
The integral condition says that the bound $b$ has finite expectation.

The surrogate $\varphi$ is Fisher consistent (i) for task risk (2) if and only if
$\overline\psi_A(\epsilon,x)>0$
for every $x\in\mathcal X$ and $\epsilon>0$.

Additionally, let
$\psi_A=(\overline\psi_A)^{**}$,
where $^{**}$ denotes the Fenchel biconjugate. Then
\[
\psi_A(R(f)-R^\star)
\leq R_{\varphi,A}(f)-R_{\varphi,A}^\star.
\]
Here, $R_{\varphi,A}^\star=\inf_{f\in\mathcal F}R_{\varphi,A}(f)$ is the minimal aggregated surrogate risk.

This result preserves classical consistency guarantees under aggregation. It does not yet show that aggregation improves consistency. We turn to that question next.

#### 3.2.2 Identifying surrogates and consistency

Under essentially minimal conditions on the surrogate, a generic aggregation strategy guarantees asymptotic consistency as the number of observed labels grows. The task loss here seeks to minimize $\ell(f(x),y)$, with
\[
R(f)=\mathbb E[\ell(f(X),Y)].
\]
We assume
$\operatorname{card}(\mathcal Y)=k<\infty$,
where $\operatorname{card}$ denotes the number of elements in a set.[^1] We impose a minimal identifiability condition on the surrogate.

[^1]: The example in Sec. 3.1 belongs to this regime. There, $k$ denotes the ranking length and $\operatorname{card}(\mathcal Y)=k(k-1)$. With a slight abuse of notation, we henceforth use $k$ for the cardinality of the label space.

###### Definition 3.1 (Identifying surrogate).

A surrogate
$\varphi:\mathbb R^d\times\mathcal A\to\mathbb R$
is $(\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})$-identifying for $\mathcal Y$, with
\[
0<\mathsf C_{\varphi,1}\leq\mathsf C_{\varphi,2}<\infty,
\]
if there are aggregation outputs
$\{a_y\}_{y\in\mathcal Y}\subset\mathcal A$
and score vectors
$\{s_y\}_{y\in\mathcal Y}\subset\mathbb R^d$
such that $\mathsf d(s_y)=y$ and, for every $y\neq y'$,
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

Inequality (6a) says that each class $y$ has an aggregation output $a_y$ for which minimizing $\varphi(\cdot,a_y)$ identifies $y$. The positive constant $\mathsf C_{\varphi,1}$ separates the selected score from scores decoded incorrectly. A finite $\mathsf C_{\varphi,2}$ in (6b) exists if $\varphi(\cdot,a)$ has a finite lower bound.

The definition does not require convexity. It also does not require consistency without aggregation, where $\mathcal A=\mathcal Y$ and $Z=Y$. Nor do we require a particular decoder, such as the pointwise $\argmax$ rule, which selects a largest score coordinate.

**Example 3:**

Consider the binary hinge loss
\[
\varphi(s,a)=\max\{1-sa,0\},
\qquad
\mathcal A=\mathcal Y=\{\pm1\}.
\]
For $y\in\{-1,1\}$, choose $a_y=s_y=y$. Then
\[
\varphi(s_1,a_1)=\varphi(s_{-1},a_{-1})=0,
\qquad
\inf_{sa\leq0}\varphi(s,a)=1.
\]
Also,
\[
\varphi(s_1,a_{-1})=\varphi(s_{-1},a_1)=2.
\]
Thus, hinge loss is $(1,2)$-identifying. $\Diamond$

Section 3.3 gives further examples and studies how aggregation amplifies consistency.

For an identifying surrogate with outputs $\{a_y\}_{y\in\mathcal Y}$, consider generalized majority vote:
\[
A_m(y_1,\ldots,y_m)\coloneqq a_{\hat y},
\qquad
\hat y=\mathop{\rm argmin}_{y\in\mathcal Y}
\sum_{i=1}^{m}\ell(y,y_i).
\tag{7}
\]
Break ties arbitrarily.

Because $\mathcal Y$ is finite, i.i.d. labels have the following property as $m\to\infty$. There is a random finite $M$ such that, for $m\geq M$,
\[
\mathop{\rm argmin}_{y\in\mathcal Y}
\left\{\sum_{i=1}^{m}\ell(y,Y_i)\right\}
\subset
y^\star(x)
\coloneqq
\mathop{\rm argmin}_{y\in\mathcal Y}
\mathbb E[\ell(y,Y)\mid X=x].
\]
The set $y^\star(x)$ contains the conditionally optimal predictions. Thus, eventually every empirical minimizing label belongs to that set.

This suggests that $\varphi(\cdot,A_m)$ should become consistent as $m\to\infty$. The following corollary of the results below gives that guarantee.

###### Corollary 3.2.

Let $m=m(n)\to\infty$, so the number of aggregated labels grows with the sequence index $n$. Let $\varphi$ be identifying under Definition 3.1. Then
\[
R_{\varphi,A_m}(f_n)-R_{\varphi,A_m}^\star\to0
\quad\Longrightarrow\quad
R(f_n)-R^\star\to0.
\]

#### 3.2.3 Identifying surrogates and consistency amplification

When label noise is low, aggregation strategy (7) improves the comparison inequality
\[
\psi(R(f)-R^\star)\leq R_\varphi(f)-R^\star
\]
by making $\psi$ linear over a range of positive excess risks. Linear growth is the strongest possible comparison inequality [34, 32].

Strict comparison inequalities, such as Proposition 3, can also be too restrictive. Inconsistent surrogates may still be convenient in practice [23, 34, 32]. Following Osokin et al. [34], we therefore introduce $(\xi,\zeta)$ consistency. The parameter $\xi$ is an excess-risk threshold; $\zeta$ is the required linear growth rate above that threshold.

###### Definition 3.2.

A surrogate $\varphi$ and aggregator $A$ yield level-$(\xi,\zeta)$ consistency if there is a function $\psi$ such that
\[
\psi(\epsilon)\geq\zeta\epsilon
\qquad\text{for }\epsilon\geq\xi,
\]
and
\[
\psi(R(f)-R^\star)
\leq R_{\varphi,A}(f)-R_{\varphi,A}^\star.
\]

Under minimal assumptions, aggregation (7) achieves level-$(o_m(1),\zeta)$ consistency even for a Fisher-inconsistent surrogate. The notation $o_m(1)$ denotes a quantity tending to zero as $m\to\infty$.

We quantify label noise using an adaptation of the Mammen-Tsybakov conditions [25; see also 4]. Define
\[
\Delta(x)\coloneqq
\min_{\mathsf d(s)\notin y^\star(x)}
\delta_\ell(s,x).
\tag{8}
\]
This is the smallest conditional excess risk of an incorrect prediction.

For binary classification with $\mathcal Y=\{\pm1\}$,
\[
\Delta(x)=|2P(Y=1\mid X=x)-1|,
\]
where $|\cdot|$ denotes absolute value. Consistent estimation is expected to be harder when $\Delta(x)$ is closer to $0$.

The Mammen-Tsybakov conditions are
\[
\mathbb P(\mathsf d\circ f\neq\mathsf d\circ f^\star)
\leq
\mathsf C_{\mathsf{MT}}
\bigl(R(f)-R^\star\bigr)^\alpha
\qquad
\text{for all measurable }f,
\tag{Nα}
\]
and
\[
\mathbb P(\Delta(X)\leq\epsilon)
\leq
(\mathsf C_{\mathsf{MT}}\epsilon)^\beta
\qquad\text{for }\epsilon>0.
\tag{Mβ}
\]
Here, $f^\star$ is an optimal prediction function. The constant
$\mathsf C_{\mathsf{MT}}>0$ may change between conditions. We call $\alpha$ the noise exponent in (Nα).

The ranges are $\alpha\in[0,1]$ and $\beta\in[0,\infty]$. Both conditions hold trivially with $\alpha=\beta=0$. As in binary classification [4, Thm. 3], they are equivalent under
\[
\beta=\frac{\alpha}{1-\alpha}.
\]
Appendix B.2 gives the details.

Mao et al. [29] use the same noise statistic for $H$-consistency in multiclass classification. Here, $H$ denotes a restricted hypothesis class, and $H$-consistency compares task and surrogate performance within that class. Their structural assumptions on the surrogate and class allow a global convex lower bound. Our setup imposes minimal structural constraints. We therefore introduce the noise condition number
\[
\kappa(x)\coloneqq
\frac{
\max_{\mathsf d(s)\neq y^\star(x)}\delta_\ell(s,x)
}{
\min_{\mathsf d(s)\neq y^\star(x)}\delta_\ell(s,x)
}.
\tag{9}
\]
It compares the largest and smallest conditional excess risks among incorrect predictions. It connects $\Delta(x)$ to pointwise excess risk through
\[
\Delta(x)\geq\frac{\delta_\ell(s,x)}{\kappa(x)}
\qquad
\text{for }\mathsf d(s)\neq y^\star(x).
\]
This permits a more detailed analysis. In binary classification, $\kappa(x)=1$ provided
$\mathbb P(\Delta(X)>0)=1$.

Together, $\Delta(x)$ and $\kappa(x)$ describe how generalized majority vote (7), combined with any identifying surrogate, achieves level-$(\xi,\zeta)$ consistency.

Define the error function
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
Here, $t$ is a possible condition-number value, and $\log$ is the natural logarithm. Roughly, when $\kappa(x)=t$, majority vote is likely correct if $m$ is large enough that $e_m(t)\ll1$. The symbol $\ll$ means “much smaller than.”

The next theorem gives a nearly linear calibration function. Appendix B.3 proves it.

###### Theorem 1.

Let $\varphi$ be
$(\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})$-identifying, with outputs $\{a_y\}_{y\in\mathcal Y}$. Let $A_m$ be the majority-vote aggregator (7).

Assume $0\leq\ell\leq1$, and assume $P$ satisfies (Nα) with $\alpha\in[0,1]$. For any $M>0$ and $f\in\mathcal F$ satisfying
\[
R(f)-R^\star
\geq
2\mathbb P(\kappa(X)>M)
+
\bigl(4\mathsf C_{\mathsf{MT}}e_m(M)\bigr)^{\frac{1}{1-\alpha}},
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

Thus, under the theorem’s assumptions, aggregation gives level-$(\xi,\zeta)$ consistency with
\[
\xi=
2\mathbb P(\kappa(X)>M)
+
\bigl(4\mathsf C_{\mathsf{MT}}e_m(M)\bigr)^{\frac{1}{1-\alpha}},
\qquad
\zeta=\frac{\mathsf C_{\varphi,1}}{16}.
\]

Theorem 1 also proves Corollary 3.2. Define
\[
\xi_m\coloneqq
\inf_M
\left\{
2\mathbb P(\kappa(X)>M)
+
\bigl(4\mathsf C_{\mathsf{MT}}e_m(M)\bigr)^{\frac{1}{1-\alpha}}
\right\}.
\]
Then $\xi_m\to0$ as $m\to\infty$. To see the two steps, first
$\mathbb P(\kappa(X)>M)\to0$ as $M\uparrow\infty$; second, for every fixed $M$, $e_m(M)\to0$ as $m$ grows. Taking $\alpha=0$ then yields Corollary 3.2.

Theorem 1 is a coarse result because Definition 3.1 is so weak. A tighter connection between $\ell$ and $\varphi$ could make majority vote more likely to be correct, or correct enough for $\varphi$, and yield stronger bounds. For example, if $\varphi$ is nearly consistent without aggregation, smaller initial inconsistency might lead to stronger bounds as a function of $m$. We do not pursue these details.

For a more concrete bound, optimize over $M$ using
\[
\kappa(x)\leq\frac{1}{\Delta(x)}.
\]
Take $M=1$ for $\operatorname{card}(\mathcal Y)=k=2$, and optimize $M$ for $k\geq3$. We may lower bound $\xi_{m,k}$ in the level-$(\xi_{m,k},\zeta)$ consistency promised by Theorem 1 by setting
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
\right)^{\frac{1}{2(1-\alpha)}},
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
Appendix B.4 gives the algebraic substitutions and manipulations leading to the next result.

###### Corollary 3.3.

Under Theorem 1’s assumptions, every $f$ satisfying
$R(f)-R^\star\geq\xi_{m,k}$
obeys
\[
R(f)-R^\star
\leq
\frac{16}{\mathsf C_{\varphi,1}}
\left(
R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star
\right).
\]

Corollaries 3.2 and 3.3 support the robustness of label cleaning: aggregation can yield consistency under minimal surrogate assumptions.

As $\alpha$ approaches $1$ in Corollary 3.3, fewer aggregated labels $m$ are required for the comparison inequality to hold for a fixed $f$. In particular, if $\alpha=1$ and
\[
m\geq
32\max\{\mathsf C_{\mathsf{MT}}^2,\mathsf C_{\mathsf{MT}}^4\}
\cdot
\log\left(
\frac{4k(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})}
{\mathsf C_{\varphi,1}}
\right)
=O(\log k),
\]
then $\xi_{m,k}=0$. The notation $O(\log k)$ means growth bounded by a constant multiple of $\log k$. This gives the uniform comparison inequality (ii) with a linear comparison function.

Thus, the problem’s noise level affects how much aggregation is needed. An “easier” problem needs less aggregation to obtain stronger consistency.

### 3.3 Surrogate consistency examples with majority vote

We now instantiate Theorem 1 and Corollary 3.3 in several examples. Throughout, assume that $P$ has noise exponent $\alpha\in[0,1]$. This loses no generality because (Nα) always holds with $\alpha=0$.

Appendix B.5 contains the proofs for all results in this section.

#### 3.3.1 Binary classification with a nonsmooth surrogate

Consider binary classification with the margin-based surrogate
\[
\varphi(f(x),y)=\phi(yf(x)),
\]
where $\phi$ is convex. The scalar function $\phi$ converts the margin $yf(x)$ into loss. Bartlett et al. [4] show that $\varphi$ is consistent if and only if $\phi'(0)<0$, where the prime denotes a derivative.

We give a somewhat trivial example in which aggregation makes an inconsistent $\phi$ consistent. One would never use such a surrogate; this is a thought experiment.

Assume that $\phi$ is convex,
\[
\partial\phi(0)\subset(-\infty,0),
\qquad
\lim_{t\to\infty}\phi(t)=0.
\]
The set $\partial\phi(0)$ contains the subgradients at zero: slopes of supporting lines to the convex function there.

###### Lemma 3.1.

For every $\delta>0$, the surrogate $\varphi$ is
$(\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})$-feasible with
\[
\mathsf C_{\varphi,1}
=\phi(0)-\phi(\delta)>0,
\qquad
\mathsf C_{\varphi,2}=\phi(-\delta).
\]

Applying Corollary 3.3 with these constants and $k=2$ gives the following. If $f:\mathcal X\to\mathbb R$ satisfies
\[
R(f)-R^\star
\geq
\left(
\frac{32\mathsf C_{\mathsf{MT}}^2}{m}
\log\left(
\frac{8(\phi(-\delta)+\phi(0)-\phi(\delta))}
{\phi(0)-\phi(\delta)}
\right)
\right)^{\frac{1}{2(1-\alpha)}},
\]
then
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
$v:\mathcal Y\to\mathbb R^d$
represents structural information about each output $y$. The structured output space $\mathcal Y$ is typically large.

Use the decoder
\[
\mathsf d(s)=
\mathop{\rm argmax}_{y\in\mathcal Y}
\langle s,v(y)\rangle,
\]
where $\langle\cdot,\cdot\rangle$ is the inner product, the sum of coordinatewise products. For a loss
$\ell:\mathcal Y\times\mathcal Y\to\mathbb R_+$
with $\ell(y,y)=0$, the maximum-margin surrogate, or generalized hinge loss [42, 44, 19], is
\[
\varphi(s,y)=
\max_{\hat y\in\mathcal Y}
\left(
\ell(\hat y,y)
+\langle v(\hat y)-v(y),s\rangle
\right).
\tag{11}
\]
This loss is typically inconsistent, except in certain low-noise cases [34, 32].

We first specialize to bipartite matching. A matching is a graph $G=(V,E)$, where $V$ is the vertex set and $E$ the edge set. The vertices split into
\[
V=V_1\cup V_2,
\qquad
V_1=\{1,\ldots,N\},
\qquad
V_2=\{N+1,\ldots,2N\}.
\]
Here, $N$ is the number of vertices on each side. Each of the $N$ edges connects one unique vertex in $V_1$ to one unique vertex in $V_2$.

Let $\mathcal Y$ contain all such matchings. Then
\[
k=\operatorname{card}(\mathcal Y)=N!,
\]
where $N!$ is the factorial of $N$.

For a graph $G$, define its edge-indicator embedding
\[
v(G)\coloneqq
\left(1\{(u,v)\in E\}\right)_{u\in V_1,v\in V_2}
\in\mathbb R^{N^2}.
\]
The coordinates index possible edges, so $d=N^2$.

The task loss counts mistaken edges:
\[
\ell(y_1,y_2)\coloneqq
\frac{1}{2N}\|v(y_1)-v(y_2)\|_1
=
\frac{1}{2N}\|v(y_1)-v(y_2)\|_2^2.
\]
The norm $\|\cdot\|_1$ is the sum of absolute coordinate values.

The structured hinge loss (11) is identifying in this problem.

###### Lemma 3.2.

For bipartite matching on $2N$ vertices, the structured hinge surrogate (11) is
$(\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})$-identifying under Definition 3.1, with
\[
\mathsf C_{\varphi,1}=\frac1N,
\qquad
\mathsf C_{\varphi,2}=2.
\]

Thus, even though there are $k=N!$ possible outputs, aggregation can yield consistency through structured hinge loss once $m$ exceeds $O(N\log N)$.

For one specialization, substitute Lemma 3.2’s constants into Corollary 3.3 for $k\geq3$. Every measurable $f:\mathcal X\to\mathbb R^d$ satisfying
\[
R(f)-R^\star
\geq
4\cdot
\left(
\frac{32\mathsf C_{\mathsf{MT}}^4}{m}
\log\left(4k(2N+1)\right)
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

Return to the general structured-prediction setup at the start of Sec. 3.3.2. Assume the decoder can select every class. Specifically, for each $y\in\mathcal Y$, assume the score set
\[
\mathcal S(y)\coloneqq
\left\{
s:
\langle v(y),s\rangle>
\langle v(\hat y),s\rangle
\text{ for }\hat y\neq y
\right\}
\]
is nonempty. These are the scores that strictly select $y$.

Define the identifiability gap for class $y$ by
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
Here, $y_+$ and $y_-$ range over alternative outputs.

###### Lemma 3.3.

For structured prediction, the max-margin surrogate (11) is
$(\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})$-identifiable with
\[
\mathsf C_{\varphi,1}
=\min_{\hat y\neq y}\ell(\hat y,y),
\qquad
\mathsf C_{\varphi,2}
=\max_{y\in\mathcal Y}\tau(y)+1.
\]

In particular, if
\[
v(y)\in\{0,1\}^d,
\qquad
\ell(\hat y,y)=
\frac{1}{2d}\|v(\hat y)-v(y)\|_1,
\]
then $\tau(y)=1$ for every $y$ and $\mathsf C_{\varphi,2}=2$. The set $\{0,1\}^d$ consists of length-$d$ vectors with binary coordinates.

As in the binary matching case, nontrivial consistency guarantees hold once
$m\geq\log\operatorname{card}(\mathcal Y)$.
Since $0\leq\ell(\cdot,\cdot)\leq1$, Corollary 3.3 applies. For every measurable $f:\mathcal X\to\mathbb R^d$,
\[
R(f)-R^\star
\geq
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
\frac{8}{\min_{\hat y\neq y}\ell(\hat y,y)}
\left(
R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star
\right).
\]
Here, $|\mathcal Y|$ is another notation for the number of outputs.

## 4 Robustness and consistency for models

Section 3 builds on classical surrogate consistency theory, where $\mathcal F$ contains all measurable functions. Its results show that aggregation can upgrade an inconsistent surrogate to level-$(\xi,\zeta)$ consistency under Definition 3.2.

We now restrict $\mathcal F$, often to a parametric class, and ask what happens when optimization is confined to that class. Consistency is not an issue for a well-specified model, but such a brittle condition is unrealistic. Quantifying surrogate consistency over only a subset of measurable functions is a long-standing challenge [12, 31].

Recent work develops $H$-consistency bounds linking excess target risk to surrogate regret within a hypothesis class $H$ [2, 3, 26, 28, 29]. “Regret” here means risk above the relevant optimum. These results focus primarily on the surrogate. Our contribution is orthogonal: rather than relying as heavily on a particular class or surrogate, we study how aggregation improves consistency itself.

We show that aggregation permits consistent estimation in settings where consistency may otherwise fail.

### 4.1 Consistency failure for binary classification in finite dimensions

Binary classification shows how restricting the hypothesis class changes the problem. Take
\[
\mathcal Y=\{\pm1\},
\qquad
\ell(\mathsf d(s),y)=1\{ys\leq0\}.
\]
Use a margin-based surrogate
\[
\varphi(s,y)=\phi(sy),
\]
with convex $\phi:\mathbb R\to\mathbb R_+$. Over all measurable functions, $\varphi$ has both Fisher consistency (i) and uniform consistency (ii) if and only if $\phi'(0)<0$ [4].

Now restrict the class. Classical consistency can fail even when an optimal classifier belongs to $\mathcal F$. Say that $P$ is optimally predictable using $\mathcal F$ when a function in the class satisfies
\[
\mathsf{sgn}(f(x))
=
\mathsf{sgn}\!\left(P(Y=1\mid X=x)-1/2\right).
\tag{12}
\]
This means that the function’s sign selects the most probable binary label.

Let $\mathcal X=\mathbb R^d$, and consider linear scoring functions
\[
\mathcal F=
\{f_\theta\mid f_\theta(x)=\langle\theta,x\rangle\}_{\theta\in\mathbb R^d}.
\]
The vector $\theta$ contains the model parameters.

If $\mathcal F$ optimally predicts $P$, there is a vector $\theta^\star$ such that
\[
\mathsf{sgn}(\langle\theta^\star,x\rangle)
=
\mathsf{sgn}\!\left(P(Y=1\mid x)-\tfrac12\right).
\]
Then $f_{\theta^\star}$ minimizes $R(f)$ over all measurable functions. We call this optimal predictability along $\theta^\star$.

One might expect a margin-based surrogate that is Fisher consistent in the unrestricted setting to remain consistent here. It does not. For any nonnegative loss $\phi$, there is a distribution $P$ for which
\[
\theta_\varphi=
\mathop{\rm argmin}_\theta
\mathbb E_P[\varphi(f_\theta(X),Y)]
\]
is essentially orthogonal to $\theta^\star$: their directions have arbitrarily small inner product after normalization.

###### Proposition 4.

For every $\epsilon>0$ and nonzero $\theta^\star\in\mathbb R^d$, there is a distribution $P$ on $(X,Y)$, optimally predictable along $\theta^\star$, such that every
\[
\theta_\varphi\in
\mathop{\rm argmin}_\theta R_\varphi(f_\theta)
=
\mathop{\rm argmin}_\theta
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
Here, $\angle$ is the angle between two vectors, and $\cos$ is its cosine.

Appendix D.1 gives the proof.

Aggregation can avoid the inconsistency in Proposition 4. Define approximate minimizers by
\[
\mathop{\epsilon\textup{-argmin}}g
=
\mathop{\epsilon\textup{-argmin}}_\theta g(\theta)
\coloneqq
\left\{
\theta\mid
g(\theta)\leq\inf_\theta g(\theta)+\epsilon
\right\}.
\]
Thus, an $\epsilon$-approximate minimizer has objective value at most $\epsilon$ above the infimum.

Suppose we collect independent labels $Z=(Y_1,\ldots,Y_m)$ and let $A_m(Z)$ be their majority vote. For an approximation-error sequence $\epsilon_m$, take
\[
\theta_m\in
\mathop{\epsilon_m\textup{-argmin}}_\theta
R_{\varphi,A_m}(f_\theta)
=
\mathop{\epsilon_m\textup{-argmin}}_\theta
\mathbb E\!\left[
\phi\!\left(
A_m(Y_1,\ldots,Y_m)f_\theta(X)
\right)
\right].
\]
Theorem 2 below gives the following asymptotic consistency result as $m\to\infty$.

###### Corollary 4.1.

Let $P$ be optimally predictable along $\theta^\star$. If $\epsilon_m\to0$ as $m\to\infty$, then
\[
R(f_{\theta_m})\to R(f_{\theta^\star}),
\qquad
\cos\angle(\theta_m,\theta^\star)\to1.
\]

Thus, without aggregation, surrogate minimization over linear predictors can be essentially arbitrarily incorrect, as Proposition 4 shows. With aggregation, Corollary 4.1 retains consistency.

### 4.2 Aggregation, consistency, and restricted hypothesis classes

Proposition 4 shows that surrogate consistency depends fundamentally on allowing all measurable functions.

Now consider multiclass classification with $\mathcal Y=\{1,\ldots,k\}$. Let $\mathcal F$ be a linear cone satisfying
\[
f(x)^T\mathbf1=0,
\qquad
tf\in\mathcal F
\quad\text{for }t>0\text{ whenever }f\in\mathcal F.
\]
The cone property here means closure under multiplication by positive scalars.

Use zero-one loss
\[
\ell(y,y')=1\{y\neq y'\}
\]
and decoder
\[
\mathsf d(s)=\mathop{\rm argmax}_{y\in[k]}s_y.
\]
Because this decoder depends only on relative scores, normalizing their sum to zero is immaterial.

Assume that
$\varphi:\mathbb R^k\times[k]\to\mathbb R_+$
is Fisher consistent (i) and satisfies the limiting loss condition
\[
\varphi(s,y)\to0
\quad\text{if}\quad
s_y-s_j\to+\infty
\text{ for every }j\neq y.
\tag{13}
\]
Thus, loss vanishes when the score for the correct class exceeds every other score by an increasingly large amount.

Many familiar surrogates are Fisher consistent and satisfy (13). These include multiclass logistic loss,
\[
\varphi(s,y)
=
\log\left(\sum_{j=1}^{k}e^{s_j-s_y}\right),
\]
where $e^u$ is the exponential function, and losses of the form
\[
\varphi(s,y)=\sum_{i\neq y}\phi(s_y-s_i),
\]
where $\phi$ is convex and non-increasing, $\phi'(0)<0$, and
$\inf_t\phi(t)=0$.

Zhang [51, Thm. 8] proves that each such loss is consistent over
\[
\mathcal F=
\{f:\mathcal X\to\mathbb R^k\mid\mathbf1^Tf=0\}.
\]

The binary setup in Sec. 4.1 fits this framework by taking
$f(x)=(g(x),-g(x))$
for a measurable scalar function $g$. In a parametric setting with $\mathcal X=\mathbb R^d$, the class of linear functions
\[
f(x)=
(\langle\theta_1,x\rangle,\ldots,\langle\theta_k,x\rangle),
\qquad
\sum_{i=1}^{k}\theta_i=0,
\]
is a convex cone. Convexity means that weighted averages of functions in the class remain in the class.

Such examples typically require strong structural assumptions, such as convexity, for consistency without aggregation. With aggregation, we will obtain asymptotic consistency under the minimal limiting loss condition (13).

Extend optimal predictability (12) to multiclass classification. Say that $\mathcal F$ optimally predicts $P$ if there is an $f\in\mathcal F$, $f:\mathcal X\to\mathbb R^k$, such that
\[
\mathop{\rm argmax}_y f_y(x)
\in
\mathop{\rm argmax}_y P(Y=y\mid X=x)
\qquad\text{for every }x.
\]
That is, the selected class is conditionally most probable.

The next theorem considers $\mathcal Z=\mathcal Y^m$, majority-vote aggregation $A_m$, and a unique
\[
y^\star(x)=\mathop{\rm argmax}_yP(Y=y\mid x).
\]
It gives consistency whenever $\mathcal F$ optimally predicts $P$.

###### Theorem 2.

Let $\mathcal F$ be a cone that optimally predicts $P$. Assume the minimal excess risk (8) satisfies
\[
P(\Delta(X)>0)=1.
\]
Let $\epsilon_m\geq0$ with $\epsilon_m\to0$. For every sequence
\[
f_m\in
\mathop{\epsilon_m\textup{-argmin}}_{f\in\mathcal F}
R_{\varphi,A_m}(f)
=
\mathop{\epsilon_m\textup{-argmin}}_{f\in\mathcal F}
\mathbb E\!\left[
\varphi(f(X),A_m(Y_1,\ldots,Y_m))
\right],
\]
we have
\[
R(f_m)\to R^\star.
\]

Appendix D.2 proves the theorem.

Theorem 2 gives asymptotic consistency, as $m\to\infty$, for a broad class of surrogate problems over a linear cone. Proposition 4 gives the contrast: even binary classification can lose consistency over subclasses containing an optimal predictor, and the surrogate can be arbitrarily uninformative.

### 4.3 On finite-dimensional multiclass classification

We now specialize to multiclass linear prediction with
$\mathcal X=\mathbb R^d$ and $\mathcal Y=[k]$,
allowing a mis-specified predictive model.

These results refine Proposition 4 and Theorem 2. Even when the problem is optimally predictable and the linear model is well specified except on an $\epsilon$-fraction of data, risks based only on $(X,Y)$ pairs are inconsistent. Majority-vote methods, however, recover the optimal linear predictor.

Let
\[
\Theta=[\theta_1\ \cdots\ \theta_{k-1}]
\in\mathbb R^{d\times(k-1)}.
\]
Its columns are the class-specific parameter vectors. Suppose
\[
Y\mid X=x\sim\mathsf{Cat}(p_\Theta(x)),
\]
where $\mathsf{Cat}$ is the categorical distribution: it selects class $y$ with probability given by coordinate $y$ of its probability vector.

The vector $p_\Theta(x)\in\mathbb R_+^k$ satisfies
$\mathbf1^\top p_\Theta(x)=1$
and has the form
\[
p_\Theta(x)=
\begin{bmatrix}
\sigma(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)\\
1-\mathbf1^\top
\sigma(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)
\end{bmatrix}.
\]
Here, $\top$ also denotes transpose. The link function
$\sigma:\mathbb R^{k-1}\to\mathbb R_+^{k-1}$
maps linear scores to the first $k-1$ class probabilities, with
$\mathbf1^\top\sigma\leq1$. The last probability is the remaining mass.

Assume the following link consistency condition. For $t\in\mathbb R^{k-1}$, set
$t_k=0$ and
$\sigma_k(t)=1-\mathbf1^\top\sigma(t)$.
Then
\[
t_y=\max_{1\leq i\leq k}t_i
\quad\text{if and only if}\quad
\sigma_y(t)=\max_{1\leq i\leq k}\sigma_i(t).
\tag{14}
\]
Thus, the maximizing scores and maximizing probabilities agree.

For multiclass logistic regression, the standard link is
\[
\sigma_i^{\textup{lr}}(t)
=
\frac{e^{t_i}}
{1+e^{t_1}+\cdots+e^{t_{k-1}}}.
\]
The superscript $\textup{lr}$ denotes the logistic-regression link.

Consider the multiply labeled dataset
\[
\{(X_i;(Y_{1i},\cdots,Y_{im}))\}_{i=1}^{n}
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
be majority vote, with ties broken arbitrarily. Then
\[
Y_i^+\mid X_i
\sim\mathsf{Cat}(p_{\Theta^\star,m}(X_i)).
\]

Let $\rho_m(t)$ give the first $k-1$ probabilities of majority vote on $m$ labels whose initial probabilities are $\sigma(t)\in\mathbb R_+^{k-1}$. Then
\[
p_{\Theta,m}(x)=
\begin{bmatrix}
\rho_m(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)\\
1-\mathbf1^\top
\rho_m(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)
\end{bmatrix}.
\]
The link $\rho_m$ also satisfies (14).

Fit logistic regression using
\[
\varphi(\Theta^\top x,y)
=
-\langle\theta_y,x\rangle
+
\log\left(
1+\sum_{i=1}^{k-1}\exp(\langle\theta_i,x\rangle)
\right),
\tag{15}
\]
with $\theta_k=\mathbf0$, the zero vector. Here, $\exp(u)=e^u$.

Define the population logistic loss based on $m$-label majority vote by
\[
L_m(\Theta)
=
\mathbb E[\varphi(\Theta^\top X,Y_m^+)].
\]
The gradient with respect to $\theta_i$ is
\[
\nabla_{\theta_i}\varphi(\Theta^\top x,y)
=
-x\left(
1\{y=i\}-\sigma_i^{\textup{lr}}(\Theta^\top x)
\right).
\]
The symbol $\nabla$ denotes the vector or matrix of partial derivatives with respect to its subscript.

Consequently, a minimizer $\Theta_m$ of $L_m$ satisfies
\[
\nabla_\Theta L_m(\Theta_m)
=
\mathbb E\!\left[
X\left(
\sigma^{\textup{lr}}(\Theta_m^\top X)
-\rho_m({\Theta^\star}^\top X)
\right)^\top
\right]
=0.
\]

Standard statistical results guarantee consistency and efficiency for a well-specified model without aggregation. For $m\geq1$ and $k=2$, Cheng et al. [7, Prop. 3] show that, even with a mis-specified link, there is a $t_m>0$ such that
\[
\Theta_m=t_m\Theta^\star
\]
when
\[
X\sim\mathsf N(0,I_d).
\]
Here, $\mathsf N(0,I_d)$ is the normal distribution with zero mean and identity covariance matrix $I_d$.

Thus, binary classification remains consistent despite an incorrect link, regardless of $m$. The direction
$\Theta^\star/\|\Theta^\star\|_2$
determines classification consistency.

For $k\geq3$, however, even slight link mis-specification causes risk consistency to fail. Fix a set
$\mathcal T_\epsilon\subset\mathbb R^{k-1}$
of Lebesgue measure $\epsilon$, where Lebesgue measure is ordinary volume. Define
\[
\sigma^\epsilon(t)
=
\sigma^{\textup{lr}}(t)1\{t\notin\mathcal T_\epsilon\}
+
\frac1k\mathbf1\cdot1\{t\in\mathcal T_\epsilon\}.
\]
This defines a conditional distribution on $Y\in\{1,\ldots,k\}$ that samples
$Y\sim\sigma^{\textup{lr}}(t)$ if $t\in\mathcal T_\epsilon$ and uniformly otherwise. The link $\sigma^\epsilon$ satisfies (14) and is optimally predictable under (12).

For $\epsilon>0$, define
\[
L_{m,\epsilon}(\Theta)
\coloneqq
\mathbb E\!\left[
\mathbb E_{\sigma^\epsilon}
[\varphi(\Theta^\top X,Y_m^+)\mid X]
\right].
\]
This is the population logistic loss based on $m$-label majority vote when
$Y\mid X=x\sim\sigma^\epsilon({\Theta^\star}^\top x)$.
The subscript $\sigma^\epsilon$ indicates that labels are sampled using that link.

Let
\[
\Theta_m(\epsilon)
=
\mathop{\rm argmin}_\Theta L_{m,\epsilon}(\Theta).
\]
Although $\Theta_1(0)=\Theta^\star$, the next result shows failure of consistency without aggregation for arbitrarily small positive $\epsilon$.

###### Proposition 5.

Let $k\geq3$ and $\Sigma=I$, where $\Sigma$ is the covariance matrix and $I$ is the identity matrix. Assume that, for
$Z\sim\mathsf N(0,I_{k-1})$, the linear mapping
\[
M\mapsto DM
\coloneqq
\mathbb E\!\left[
Z\left(
\nabla\sigma^{\textup{lr}}({\Theta^\star}^\top Z)MZ
\right)^\top
\right]
\]
is invertible. Here, $M$ is a matrix argument, $D$ names the linear operator, and $\nabla\sigma^{\textup{lr}}$ is the matrix of partial derivatives of the link.

Then there is an $\epsilon_0>0$ such that every
$\epsilon\in(0,\epsilon_0)$ admits a set $\mathcal T_\epsilon$ of Lebesgue measure at most $\epsilon$ for which
\[
\Theta_1(\epsilon)/\|\Theta_1(\epsilon)\|
\neq
\Theta^\star/\|\Theta^\star\|.
\]
The notation $\|\cdot\|$ denotes the matrix norm used to normalize these parameter matrices.

Appendix E.1 gives the proof.

Majority vote can address this inconsistency as $m\to\infty$. The true link $\sigma$ need only satisfy (14). Define
\[
L_{m,\sigma}(\Theta)
=
\mathbb E\!\left[
\mathbb E_\sigma
[\varphi(\Theta^\top X,Y_m^+)\mid X]
\right],
\qquad
\Theta_m=\mathop{\rm argmin}L_{m,\sigma}(\Theta).
\]

###### Theorem 3.

Suppose
\[
\Theta^\star=U^\star T^\star,
\]
where $U^\star\in\mathbb R^{d\times(k-1)}$ is orthogonal, meaning that its columns are orthonormal, and
$T^\star\in\mathbb R^{(k-1)\times(k-1)}$
is nonsingular, meaning invertible.

Then, for every $m$, there is a matrix
$T_m\in\mathbb R^{(k-1)\times(k-1)}$
such that
\[
\Theta_m=U^\star T_m.
\]
As $m\to\infty$,
\[
\|T_m\|\to\infty,
\qquad
T_m/\|T_m\|\to T^\star/\|T^\star\|.
\]

Appendix E.2 proves the theorem.

Theorem 3 provides further evidence for aggregation’s robustness. It yields asymptotic consistency in mis-specified models as long as a link function describes the relationship between $X$ and $Y$. For $k\geq3$, this contrasts with the general inconsistency of methods without aggregation highlighted by Proposition 5.

## 5 Numerical illustrations

We give two experiments supporting the theory: (i) a nonparametric example in which aggregation amplifies surrogate consistency, and (ii) a finite-dimensional multiclass example in which aggregation recovers asymptotic consistency under link mis-specification.

##### Surrogate consistency amplification for truncated quadratic loss.

For binary classification, take the margin-based surrogate
\[
\varphi(f(x),y)=\phi(yf(x)),
\qquad
\phi(\delta)=\max\{1-\delta,0\}^2.
\]
This is the truncated quadratic loss. Bartlett et al. [4, Example 2] give its exact calibration function in Corollary 3.1:
\[
\psi(\epsilon)=\epsilon^2.
\]

With majority vote on $m$ labels, apply Lemma 3.1 with $\delta=1$. Without a low-noise assumption, take
$\alpha=0$ and $\mathsf C_{\mathsf{MT}}=1$. Define
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
Here, $(u)_+=\max\{u,0\}$ is the positive part.

The non-aggregated and majority-vote risks satisfy the parallel inequalities
\[
\psi(R(f)-R^\star)
\leq R_\varphi(f)-R_\varphi^\star,
\qquad
\psi_m(R(f)-R^\star)
\leq R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star.
\]

Numerically, the aggregated curve transitions from quadratic toward linear behavior as $m\to\infty$; see Figure 1. This shows aggregation upgrading surrogate consistency. The functions $\psi_m$ are generally conservative because they come from the general bound in Theorem 1.

Figure 1: Numerical illustrations for truncated quadratic surrogate $\phi(\delta)=\max\{1-\delta,0\}^{2}$, showing comparison inequalities for non-aggregated data $\psi$ (optimal) vs. majority vote aggregation $\psi_{m}$ (suboptimal) when $m\in\{2^{16},2^{20},2^{24},2^{28},2^{32}\}$.

##### Asymptotic consistency for mis-specified logistic model.

For the second experiment, use the logistic-regression setup of Sec. 4.3 with $k=3$ and $d=2$. Let $\varphi$ be the logistic loss (15), and corrupt the logistic link as follows:
\[
\tilde\sigma(t)
=
\sigma^{\textup{lr}}(t)1\{\|t\|\leq r\}
+
\sigma^{\textup{lr}}(\alpha t)1\{\|t\|>r\}.
\]
Here, $\tilde\sigma$ names the corrupted link, $r$ is a radius, and $\alpha$ is the score-shrinkage factor in this experiment, rather than the noise exponent above. The vector norm $\|t\|$ measures distance from the origin.

This piecewise link equals the logistic, or softmax, link near the origin and shrinks scores outside radius $r$. Figure 2 reports results on synthetic data generated from this link. They show asymptotic consistency as $m\to\infty$.

| (i) Experiment with constant dataset size $n$ | (ii) Experiment with constant label size $nm$ |
|---|---|

Figure 2: Numerical illustrations for mis-specified logistic models when $d=2$ and $k=3$. The true parameters are $\theta_{1}^{\star}=(0,1)^{\top},\theta_{2}^{\star}=(1,1)^{\top}$. The corrupted link parameters are $r=3$ and $\alpha=0.1$. The plot shows the error of classification rule $\left\|\|T_{m}\|-T^{\star}/\|T^{\star}\|\right\|$ vs. number of labels $m\in\{1,4,16,64,256,1024\}$. (i) In the left panel, for each value of $m$, the synthetic dataset consists of $n=10{,}000$ multilabeled observations $(X_{i},(Y_{i1},\ldots,Y_{im}))$ sampled from $\tilde{\sigma}$. (ii) In the right panel, for each $m$, we set $n=640{,}000/m$ so that the total number of labels remains constant across different values of $m$. We report a $95\%$ error bar on $T=100$ trials for each $m$.

Here, $nm$ is the total label count, and $T$ in the caption denotes the number of trials, not the parameter matrices $T_m$ or $T^\star$.

## 6 Discussion and future work

Whether and how to clean data has motivated much discussion about dataset collection. Cheng, Asi and Duchi [7] discuss an apparent pattern: using all non-aggregated labels gives better statistical efficiency when the model can represent all label uncertainty, but cleaning appears more robust otherwise.

Dorner and Hardt [11] make a related argument for validation, the evaluation used to compare binary classifiers. In that setting, they argue for more noisy labels rather than cleaned labels.

Our paper adds evidence for fundamental limits of uncleaned, non-aggregated labels in supervised learning. It also identifies robustness improvements from cleaning. Many questions remain.

##### Finite $m$ results and fundamental limits.

Many of our consistency results take the number $m$ of aggregated labels to infinity. This highlights ways aggregation improves robustness, but relying on an asymptotic setting may be unsatisfying.

For ranking, Sec. 3.1 gives explicit consistency guarantees at finite $m$. Extending such results is a natural and, we believe, important direction.

It would be useful to develop a surrogate consistency theory depending jointly on the loss pair $(\ell,\varphi)$ and the available label count $m$. For example, in Sec. 3.1, would second- or third-order item comparisons require larger $m$? Such comparisons include matrix powers $C_x^p$, as suggested by Keener [20].

A central challenge is to identify precisely which problems require label cleaning and aggregation and which do not.

##### Fundamental limits of the noise condition number.

Our multiclass comparison inequalities rely on the noise condition number $\kappa(X)$ in (9). This points to a difficulty absent in binary classification, where $\kappa(X)=1$.

Beyond binary classification, $\kappa(X)$ can be large even when the Mammen-Tsybakov noise level is low, meaning $\alpha\approx1$ in (Nα). The symbol $\approx$ means “approximately equal.”

This reflects our minimal assumptions on the surrogate. It would be useful to identify relationships between $\ell$ and $\varphi$ that better capture problem difficulty. Explicit failure constructions could also clarify fundamental limits and provide further insight into fitting predictive models.

##### Behavior in mis-specified models.

Our mis-specified-model results, especially Sec. 4.3, require optimal predictability (12). That is, $\mathcal F$ must contain a Bayes-optimal classifier, a classifier minimizing task risk over all measurable prediction rules.

Classical surrogate consistency can fail even in this setting, while aggregated-label methods can succeed. Moving beyond this restriction is a fruitful direction.

Nguyen et al. [31], followed by Duchi et al. [12], identify one possible direction. For binary and multiclass classification, respectively, jointly learning a predictor $f$ and a representation of the input $x$ requires surrogates of a particular form determined by $\ell$. Those results still rely on infinitely powerful decision rules $f$, so new approaches are needed.

##### Restricted class comparison inequality with label aggregation.

We study how aggregation reduces surrogate-induced regret through nonparametric comparison inequalities. We also establish statistical consistency in parametric settings.

Recent work on $H$-consistency [2, 28, 27, 29] motivates extending these methods to comparison inequalities over restricted hypothesis classes. Mao et al. [27] show that, for smooth convex loss, quadratic growth of $\psi$ is rate optimal: its growth order cannot be improved in that setting. It would be interesting to use aggregation to upgrade consistency in the context of $\mathcal H$-consistency, where $\mathcal H$ denotes the restricted hypothesis class.

Such an extension could unify aggregation’s benefits across parametric and nonparametric settings. It could also clarify how the “size” of $\mathcal F$ affects aggregation’s effectiveness.
