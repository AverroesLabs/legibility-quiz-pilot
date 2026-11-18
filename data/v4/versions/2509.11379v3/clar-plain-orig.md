# Some Robustness Properties of Label Cleaning

Chen Cheng

  
John Duchi

###### Abstract

We show that learning methods that use aggregated labels—label information combined from noisy responses—have robustness properties that methods without data cleaning cannot achieve. This robustness takes several forms.

First, aggregation strengthens risk consistency. Risk is the expected loss of a prediction method. Risk consistency means that minimizing a surrogate loss, which replaces the desired task loss with a typically convex loss that is usually easier to optimize, also minimizes the task loss. Methods that aggregate labels achieve stronger consistency guarantees than methods that use raw labels can achieve.

Second, aggregation protects against model mis-specification, where a model does not exactly describe the data. Classical statistical results for perfectly specified models suggest that using all available information, including label uncertainty, gives statistically efficient estimates. Yet standard methods lose consistency when the loss to be minimized is even slightly mis-specified. Methods that use aggregated information still converge to optimal classifiers. These results show how considering the full data analysis pipeline—from data collection through model fitting to prediction—can make methods more robust by refining noisy signals.

## 1 Introduction

Consider the data collection pipeline for supervised learning, where a model learns from inputs with labels. A simple account says that we collect feature-label pairs $(X_i,Y_i)_{i=1}^{n}$, fit a model, and proceed [17]. This account hides the complexity of modern datasets [9, 22, 39]. These datasets require substantial cleaning and filtering. They often require crowdsourcing several labels for each example and then removing noise from those labels. The crowdsourcing community has studied this cleaning process closely, especially for obtaining “gold standard” labels [8, 48, 47, 46, 37]. We take a complementary view. We study how data aggregation fundamentally and necessarily improves the consistency of fitted models.

This paper argues that label cleaning, or combining labels, provides robustness that cannot be achieved without aggregation. This robustness has two forms. First, aggregation improves estimation consistency. When we minimize a surrogate loss, such as multiclass logistic loss, instead of a task loss, such as zero-one error, aggregated labels can support consistent and optimal prediction in the limit when raw labels cannot. Second, even in finite-dimensional statistical problems, aggregation can produce consistent classifiers when standard methods fail.

Important contributions to surrogate risk consistency theory date to the 2000s [52, 24, 40]. Bartlett, Jordan and McAuliffe [4] characterized when fitting a model with a convex surrogate yields consistency for binary classification under zero-one error. A convex loss has a shape that supports convex optimization. Later work has studied surrogate risk consistency for multi-label classification [50, 43, 16, 49, 1], ranking [13, 14, 36], structured prediction [34, 6, 32], ordinal regression [35], restricted hypothesis classes [2, 3, 28, 29], and general theory [40]. A hypothesis class is the set of prediction functions a method may use.

These analyses assume the standard supervised learning setting of pairs $(X,Y)$. This lets us use the full statistical theory of empirical processes, which studies how sample averages differ from their population counterparts [45, 5, 21, 4]. But these analyses do not address the aggregation methods now common in dataset creation.

This raises several questions about consistency and aggregation. Do we need to aggregate at all? If surrogate consistency holds without aggregation, perhaps we should rely on the mature theory for $(X,Y)$ pairs. If we do aggregate, does aggregation improve consistency, and in what sense? These questions motivate this paper.

We give concrete examples in ranking and in binary and multiclass classification with linear estimators. In these examples, estimators that use only $(X,Y)$ pairs necessarily fail, but label aggregation yields consistency. We develop new definitions and theory for surrogate consistency with aggregation.

In fully nonparametric settings, which do not restrict prediction functions to a finite-dimensional model, we show how the number of aggregated samples and noise conditions combine to improve consistency. Aggregation also gives surrogate risk consistency under only weak conditions on the surrogate loss. In the field’s terminology, losses that use aggregated labels admit approximate linear comparison inequalities. These inequalities bound excess task risk by excess surrogate risk, with an approximately linear relationship.

Conventional risk consistency theory requires a hypothesis class $\mathcal{F}$ that contains all measurable functions. We also give classification results in which aggregation guarantees consistency over restricted hypothesis classes, although consistency may fail without aggregation.

## 2 Preliminaries

We first review classical surrogate risk minimization. Let $\mathcal{X}$ be the input space and $\mathcal{Y}$ the output space. Draw the data $(X_i,Y_i)_{i=1}^{n}\in\mathcal{X}\times\mathcal{Y}$ independently from the same distribution $P$; we use “i.i.d.” for independently and identically distributed.

We learn a scoring function $f:\mathcal{X}\to\mathbb{R}^{d}$. It maps an input $x\in\mathcal{X}$ to a score $s\in\mathbb{R}^{d}$ for some $d\geq 1$. A decoder $\mathsf{d}:\mathbb{R}^{d}\to\mathcal{Y}$ converts scores into the final prediction,
$\widehat{y}=\mathsf{d}\circ f(x)$. Here, $\circ$ denotes function composition.

Given a loss $\ell:\mathcal{Y}\times\mathcal{Y}\to\mathbb{R}_{+}$ and a hypothesis class $\mathcal{F}$, we seek to minimize the task risk over $f\in\mathcal{F}$:

$$
R(f):=\mathbb{E}_{P}\left[\ell(\mathsf{d}\circ f(X),Y)\right].
\tag{1}
$$

Here, $\mathbb{R}_{+}$ denotes the nonnegative real numbers, and $\mathbb{E}_{P}$ denotes expectation under $P$.

For example, in binary classification, $d=1$,
$\mathsf{d}(s)=\mathsf{sgn}(s)$, and
$\ell(y,y')=1\{yy'\leq 0\}$. The function $\mathsf{sgn}$ gives the sign, and $1\{\cdot\}$ is the indicator, equal to one when its condition holds and zero otherwise. Thus,
$R(f)=\mathbb{P}(Yf(X)\leq 0)$, where $\mathbb{P}$ denotes probability.

Minimizing $R(f)$ is difficult because the task loss $\ell$ may be nonsmooth, nonconvex, and uninformative. In particular, zero-one loss is flat almost everywhere. This makes even practical methods, such as first-order optimization methods that use gradients, impossible.

We consider a slightly more general version of problem (1). Instead of requiring $\ell$ to depend only on the observed label $Y$, we allow it to depend on the conditional label distribution $P(Y\in\cdot\mid X)$. We therefore study

$$
R(f)\coloneqq
\mathbb{E}_{P}\left[
\ell(\mathsf{d}\circ f(X),P(\cdot\mid X))
\right].
\tag{2}
$$

Its minimizers often coincide with those of the original problem (2), but this form allows more complex losses. For example, in multiclass classification, $Y\in\{1,\ldots,k\}$. Taking
$\ell(\widehat{y},P)=\sum_y P(Y=y)1\{\widehat{y}\neq y\}$
makes the risk equal to the standard zero-one error rate.

Rather than optimize the task loss $\ell$, we use a surrogate
$\varphi:\mathbb{R}^{d}\times\mathcal{Y}\to\mathbb{R}$
that is easier to optimize. Instead of directly minimizing risk (2), we minimize the surrogate risk

$$
R_{\varphi}(f):=\mathbb{E}_{P}\left[\varphi(f(X),Y)\right].
$$

For this approach to make sense, we need some form of consistency with the task problem (2). We focus on two settings, which we formalize later:

(i) The classical case of Fisher consistency, where $\mathcal{F}$ contains all Borel functions, meaning functions measurable with respect to the Borel sets.

(ii) Statistical settings where the hypothesis class $\mathcal{F}$ is parametric, meaning that parameters specify its functions, but may be mis-specified.

Our main message is that label aggregation improves consistency in both settings. This demonstrates the robustness of label cleaning.

### 2.1 Label aggregation

Instead of observing pairs $(X_i,Y_i)$, suppose we replace the output $Y$ with a more general variable $Z\in\mathcal{Z}$. For example, when we collect $m$ noisy labels for each input $X$, we take
$Z=(Y_1,\ldots,Y_m)\in\mathcal{Y}^{m}$.
Mao et al. [28] consider a similar data structure with several labels for multilabel losses. We instead focus on the label aggregation pipeline.

Let $\mathcal{A}$ be an aggregation space, and let
$A:\mathcal{Z}\to\mathcal{A}$ combine the observations, for example by majority vote. Let
$\varphi:\mathbb{R}^{d}\times\mathcal{A}\to\mathbb{R}_{+}$
be a surrogate loss on this aggregation space. Define the aggregated surrogate risk as

$$
R_{\varphi,A}(f):=
\mathbb{E}\left[\varphi(f(X),A(Z))\right].
\tag{3}
$$

We ask when minimizing the surrogate problem (3) suffices to minimize the actual task risk (2). Two examples make this setting more concrete.

**Example 1 (Majority vote):**

In the repeated sampling setting, we collect
$Z=(Y_1,\cdots,Y_m)$, where

$$
Y_i\mid X=x\stackrel{\mathrm{iid}}{\sim}P_{Y\mid X=x}.
$$

Define $A_m(Z)$ as the empirical minimizer, which minimizes the loss summed over observed labels:

$$
A(Z)=A(\{Y_1,\ldots,Y_m\})
=
\mathop{\rm argmin}_{y\in\mathcal{Y}}
\sum_{l=1}^{m}\ell(y,Y_l).
$$

Here, $\mathop{\rm argmin}$ denotes the set of minimizing choices. When
$\ell(\widehat{y},y)=1\{y\neq\widehat{y}\}$, this is exactly majority vote. The more general form allows other procedures. $\Diamond$

We can also roughly describe $K$-nearest neighbor aggregation methods, which combine labels from nearby inputs.

**Example 2 ( $K$ -nearest neighbors):**

Consider a repeated sampling setting in which an input $X$ comes with a label $Y$ and an additional independent sample

$$
(X_i,Y_i)_{i=1}^{m}\stackrel{\mathrm{iid}}{\sim}P.
$$

Here, $m$ is the number of additional examples, and
$Z=(Y,(X_i,Y_i)_{i=1}^{m})$.
Let $\mathop{\rm dist}:\mathcal{X}\times\mathcal{X}\to\mathbb{R}_{+}$ be a distance metric on $\mathcal{X}$. Order the additional inputs by distance from $X$:

$$
\mathop{\rm dist}(X,X_{(1)})
\leq\ldots\leq
\mathop{\rm dist}(X,X_{(m)}),
$$

and set $X_{(0)}=X$. For $K\geq 0$, we can aggregate the $K$ nearest neighbors of $X$, for example by choosing

$$
A_{m,K}(Z)\coloneqq
\mathop{\rm argmin}_{y\in\mathcal{Y}}
\sum_{l=0}^{K}\ell(y,Y_{(l)}).
$$

Appendix C uses the results in the following sections to move beyond this population-level setting and study aggregation from a single sample $(X_i,Y_i)_{i=1}^{n}$. $\Diamond$

## 3 Surrogate consistency

The standard surrogate consistency framework [40] assumes that $\mathcal{F}$ contains all Borel measurable functions
$f:\mathcal{X}\to\mathbb{R}^{d}$.
In the general setting of Section 2, define the conditional task risk and conditional surrogate risk at score $s\in\mathbb{R}^{d}$ by

$$
R(s\mid x)\coloneqq
\ell(\mathsf{d}\circ s,P(Y\in\cdot\mid X=x))
\quad\text{and}\quad
R_{\varphi}(s\mid x):=
\mathbb{E}\left[\varphi(s,Y)\mid X=x\right].
$$

Define the pointwise excess risks, which measure how much a score’s conditional risk exceeds the best possible conditional risk:

$$
\delta_{\ell}(s,x)\coloneqq
R(s\mid x)-\inf_{s'\in\mathbb{R}^{d}}R(s'\mid x),
\qquad
\delta_{\varphi}(s,x)\coloneqq
R_{\varphi}(s\mid x)
-\inf_{s'\in\mathbb{R}^{d}}R_{\varphi}(s'\mid x).
$$

Also define the minimal risks
$R^{\star}:=\inf_{f\in\mathcal{F}}R(f)$ and
$R_{\varphi}^{\star}:=\inf_{f\in\mathcal{F}}R_{\varphi}(f)$.
The symbol $\inf$ denotes an infimum, or greatest lower bound.

Following the standard approach [40, 4, 52], we require at least (i) Fisher consistency and, if possible, (ii) a stronger, quantitative uniform comparison inequality. These conditions require the following for every data distribution $P$:

(i) For any sequence $f_n\in\mathcal{F}$,
$R_{\varphi}(f_n)\to R_{\varphi}^{\star}$
implies
$R(f_n)\to R^{\star}$.

(ii) There is a non-decreasing function
$\psi:\mathbb{R}_{+}\to\mathbb{R}_{+}$ such that

$$
\psi(R(f)-R^{\star})
\leq R_{\varphi}(f)-R_{\varphi}^{\star}
\quad\text{for all }f\in\mathcal{F},
$$

where $\psi(\epsilon)>0$ for every $\epsilon>0$.

For binary classification with a convex, margin-based surrogate $\varphi$, these two notions coincide [4]. A margin-based loss depends on the score multiplied by the label. The stronger uniform guarantee (ii) need not always hold. Calibration functions provide a standard construction from excess risk:

$$
\psi(\epsilon,x)\coloneqq
\inf_{s\in\mathbb{R}^{d}}
\left\{
\delta_{\varphi}(s,x)
\mid \delta_{\ell}(s,x)\geq\epsilon
\right\},
\qquad
\overline{\psi}(\epsilon)\coloneqq
\inf_{x\in\mathcal{X}}\psi(\epsilon,x).
$$

These functions measure the least excess surrogate risk associated with at least a given excess task risk. They yield consistency and comparison inequalities; see Zhang [51, Prop. 25] and Steinwart [40, Thm. 2.8 and Lemma 2.9].

###### Corollary 3.1.

The surrogate $\varphi$ is Fisher consistent (i) for $\ell$ if and only if
$\psi(\epsilon,x)>0$ for every $x\in\mathcal{X}$ and $\epsilon>0$.
Let $\psi$ be the Fenchel biconjugate of $\overline{\psi}$, a standard operation that constructs a convex lower bound. Then
$\overline{\psi}(\epsilon)>0$ if and only if $\psi(\epsilon)>0$, and for every measurable $f$,

$$
\psi(R(f)-R^{\star})
\leq R_{\varphi}(f)-R_{\varphi}^{\star}.
$$

For the general risk minimization problem (2), we would like at least a Fisher consistent surrogate for $\ell$. Minimizing
$R_{\varphi}(f)=\mathbb{E}[\varphi(f(X),Y)]$
would then imply minimizing $R(f)$.

If such a result held using paired observations $(X,Y)$ rather than tuples
$(X,Y_1,\ldots,Y_m)$, we could apply the full theory of empirical processes and related statistical tools [45, 5, 21, 4]. Data collection would also be simpler: consistent estimation would require only single pairs $(X,Y)$. Unfortunately, such results are generally impossible. The extended example below shows why we need the aggregation theory developed in Section 3.2.

### 3.1 Fisher consistency failure without label aggregation: ranking

Consider ranking $k$ items from pairwise comparisons [20, 15, 14, 30]. The label space contains all comparisons:

$$
\mathcal{Y}=\{(i,j):i\neq j,\ 1\leq i,j\leq k\}.
$$

The label $(i,j)$ means that item $i$ ranks above item $j$. For each input $x$, the population rank aggregation problem converts the probabilities
$p_{ij}=P(Y=(i,j)\mid x)$ into a ranking of the $k$ items.

Many aggregation methods are possible. We use a simple comparison-based method [20, cf.]. Similar negative results hold for more complex methods. Define the normalized transition matrix
$C_x\in\mathbb{R}_{+}^{k\times k}$ by $(C_x)_{ii}=0$ and

$$
(C_x)_{ij}
=
\frac{p_{ij}}{\sum_{l\neq j}p_{lj}}
\quad\text{for }i\neq j.
$$

Set $0/0=1/(k-1)$ so that $C_x$ is stochastic: its columns sum to one, or
$C_x^{T}\mathbf{1}=\mathbf{1}$.
Here, $\mathbf{1}$ is the vector of ones, and the superscript $T$ denotes transpose.

Rank the items by $C_x\mathbf{1}\in\mathbb{R}_{+}^{k}$, which measures how often each item is preferred to others. One could instead use higher powers $C_x^{p}\mathbf{1}$ or Perron vectors, the eigenvectors used for this ranking construction [20]. Similar results hold in those cases.

Incorporating the decoder $\mathsf{d}$ into the task loss $\ell$ without writing it separately, define

$$
\ell(s,C)\coloneqq
\max_{i<j}
1\!\left\{
(s_i-s_j)(e_i-e_j)^TC\mathbf{1}\leq 0,\
(e_i-e_j)^TC\mathbf{1}\neq 0
\right\}.
$$

Here, $e_i$ is the coordinate vector for item $i$. This loss penalizes scores $s$ that order items differently from $C$. The population task risk (2) is therefore

$$
R(f)\coloneqq
\mathbb{P}\left(
f(X)\text{ and }C_X\mathbf{1}\text{ order differently}
\right).
\tag{4}
$$

Now consider a convex surrogate
$\varphi:\mathbb{R}^{k}\times\mathcal{Y}\to\mathbb{R}$.
We restrict scores to $s^T\mathbf{1}=0$. This minor restriction is familiar from multiclass classification [51, 43]. It is natural because decoding a ranking requires only the order of the scores $s_i$.

Unfortunately, no convex Fisher consistent surrogate exists for problem (4); see Appendix A.1.

###### Proposition 1.

Consider the ranking problem with task risk (4) over $k\geq 3$ outcomes. If
$\varphi:\mathbb{R}^{k}\times\mathcal{Y}\to\mathbb{R}$
is convex in its first argument, it is not Fisher consistent.

Yet a fairly direct argument gives consistency with aggregation as soon as the number $m$ of collected comparisons satisfies $m\geq k$. The idea is to regress the predicted scores $f(x)$ on frequencies of label orderings.

Assume that $Z=(Y_1,\ldots,Y_m)$ contains pairwise comparisons that are independent conditional on $X$. Let

$$
m_{ij}=\sum_{y\in\mathcal{Y}^{m}}1\{y=(i,j)\},
\qquad
m_j=\sum_{i=1}^{k}m_{ij},
$$

and define

$$
A(Z)=
\begin{cases}
\star,
&\text{if }m_j=0\text{ for some }j\text{ in }[k],\\[3pt]
\left(
\frac{m_{i1}}{m_1}
+\frac{m_{i2}}{m_2}
+\cdots+
\frac{m_{ik}}{m_k}
\right)_{i\in[k]},
&\text{otherwise, that is, if }m_j>0
\text{ for all }j\in[k].
\end{cases}
$$

Here, $[k]=\{1,\ldots,k\}$, and $\star$ marks an aggregation that lacks a required comparison count. Regressing directly on $A(Z)$ when $A(Z)\neq\star$ gives consistency. Appendix A.2 proves the next proposition.

###### Proposition 2.

Define
$\varphi(s,q)=\|s-q\|_2^2$
for $s,q\in\mathbb{R}^{k}$, where $\|\cdot\|_2$ is the Euclidean norm, and define
$\varphi(s,\star)=0$.
If $m\geq k$, then $\varphi$ is Fisher consistent for the ranking risk (4).

The difference between Proposition 1, which rules out consistency, and Proposition 2, which ensures it, comes from the response space. Aggregating several labels expands that space from $\mathcal{Y}$ to $\mathbb{R}^{k}$.

### 3.2 Label aggregation obtains stronger surrogate consistency

The ranking example suggests that label aggregation can improve consistency. We now ask how aggregation affects surrogate consistency more generally.

We give two results. The first extends standard surrogate risk consistency in a basic way. The second shows how aggregation can turn losses that would otherwise be inconsistent into consistent losses, as Proposition 2 suggests.

#### 3.2.1 Basic extensions of surrogate consistency

We first extend standard calibration functions and the classical comparison inequalities in Corollary 3.1. For any aggregation method
$A:\mathcal{Z}\to\mathcal{A}$, define the conditional aggregated surrogate risk

$$
R_{\varphi,A}(s\mid x)\coloneqq
\mathbb{E}\left[\varphi(s,A(Z))\mid X=x\right].
$$

As in the setting without aggregation, the pointwise excess risk

$$
\delta_{\varphi,A}(s,x):=
R_{\varphi,A}(s\mid x)
-\inf_{s\in\mathbb{R}^{d}}R_{\varphi,A}(s\mid x)
$$

defines the pointwise and uniform calibration functions

$$
\overline{\psi}_{A}(\epsilon,x)\coloneqq
\inf_{s\in\mathbb{R}^{d}}
\left\{
\delta_{\varphi,A}(s,x)
\mid \delta_{\ell}(s,x)\geq\epsilon
\right\},
\qquad
\overline{\psi}_{A}(\epsilon)\coloneqq
\inf_{s\in\mathbb{R}^{d}}
\overline{\psi}_{A}(\epsilon,x).
\tag{5}
$$

A consistency result similar to Corollary 3.1 follows under suitable measurability conditions. We leave those conditions implicit because they are not central to our results. The next result follows essentially as a corollary of Steinwart [40, Thm. 2.8]. Appendix B.1 includes a proof for completeness.

###### Proposition 3.

Assume that there is a function
$b:\mathcal{X}\to\mathbb{R}_{+}$ with
$\int b(x)dP(x)<\infty$ such that
$\delta_{\ell}(f(x),x)\leq b(x)$.
The surrogate $\varphi$ is Fisher consistent (i) for task risk (2) if and only if
$\overline{\psi}_{A}(\epsilon,x)>0$
for every $x\in\mathcal{X}$ and $\epsilon>0$.

Also, if
$\psi_A=(\overline{\psi}_A)^{**}$
is the Fenchel biconjugate of $\overline{\psi}_A$, then

$$
\psi_A(R(f)-R^{\star})
\leq R_{\varphi,A}(f)-R_{\varphi,A}^{\star}.
$$

This result preserves the classical consistency guarantees under aggregation. It does not give a specific improvement in consistency. We turn to that question next.

#### 3.2.2 Identifying surrogates and consistency

Under essentially minimal conditions on the surrogate, a general aggregation strategy guarantees consistency as the number of observations $y$ grows. This applies to any task loss that seeks to minimize $\ell(f(x),y)$, so that
$R(f)=\mathbb{E}[\ell(f(X),Y)]$.

Assume that
$\mathop{\rm card}(\mathcal{Y})=k<\infty$,
where $\mathop{\rm card}$ denotes the number of elements in a set.[^1]
We impose a minimal identifiability assumption on the surrogate loss.

[^1]: The example in Section 3.1 belongs to this setting. There, $k$ denotes the ranking length and $\mathop{\rm card}(\mathcal{Y})=k(k-1)$. With a slight abuse of notation, we use $k$ from here onward for the cardinality of the label space.

###### Definition 3.1 (Identifying surrogate).

A surrogate
$\varphi:\mathbb{R}^{d}\times\mathcal{A}\to\mathbb{R}$
is $(\mathsf{C}_{\varphi,1},\mathsf{C}_{\varphi,2})$-identifying for $\mathcal{Y}$, with

$$
0<\mathsf{C}_{\varphi,1}
\leq\mathsf{C}_{\varphi,2}<\infty,
$$

if there are
$\{a_y\}_{y\in\mathcal{Y}}\subset\mathcal{A}$
and vectors
$\{s_y\}_{y\in\mathcal{Y}}\subset\mathbb{R}^{d}$
such that $\mathsf{d}(s_y)=y$ and, for every $y\neq y'$,

$$
\varphi(s_y,a_y)+\mathsf{C}_{\varphi,1}
\leq
\inf_{\mathsf{d}\circ s\neq y}\varphi(s,a_y),
\tag{6a}
$$

$$
\varphi(s_y,a_{y'})-\mathsf{C}_{\varphi,2}
\leq
\inf_{s\in\mathbb{R}^{d}}\varphi(s,a_{y'}).
\tag{6b}
$$

Inequality (6a) says that each class $y\in\mathcal{Y}$ has a parameter
$a_y\in\mathcal{A}$ for which minimizing $\varphi(\cdot,a_y)$ identifies $y$. A finite $\mathsf{C}_{\varphi,2}$ exists for (6b) if $\varphi(\cdot,a)$ has a finite lower bound.

Definition 3.1 does not require $\varphi(\cdot,a)$ to be convex. It also does not require consistency when $\mathcal{A}=\mathcal{Y}$ and $Z=Y$, that is, without aggregation. Nor do we impose a requirement on the decoder $\mathsf{d}$, such as requiring it to equal the pointwise $\mathop{\rm argmax}$ operator, which selects a maximizing class.

**Example 3:**

Consider the binary hinge loss

$$
\varphi(s,a)=\max\{1-sa,0\},
\qquad
\mathcal{A}=\mathcal{Y}=\{\pm1\}.
$$

For $y\in\{-1,1\}$, set $a_y=s_y=y$. Then
$\varphi(s_1,a_1)=\varphi(s_{-1},a_{-1})=0$, while

$$
\inf_{sa\leq 0}\varphi(s,a)=1.
$$

Also,
$\varphi(s_1,a_{-1})=\varphi(s_{-1},a_1)=2$.
Thus, hinge loss is $(1,2)$-identifying. $\Diamond$

Section 3.3 gives further examples when we study how aggregation amplifies consistency. For an identifying surrogate with parameters $\{a_y\}_{y\in\mathcal{Y}}$, consider the simple aggregation strategy of generalized majority vote:

$$
A_m(y_1,\ldots,y_m)\coloneqq a_{\hat y},
\qquad
\hat y=
\mathop{\rm argmin}_{y\in\mathcal{Y}}
\sum_{i=1}^{m}\ell(y,y_i).
\tag{7}
$$

Break ties arbitrarily. Because $\mathcal{Y}$ is finite, if the $Y_i$ are i.i.d., then as $m\to\infty$ there necessarily exists a random $M<\infty$ such that $m\geq M$ implies

$$
\mathop{\rm argmin}_{y\in\mathcal{Y}}
\left\{\sum_{i=1}^{m}\ell(y,Y_i)\right\}
\subset
y^{\star}(x)
\coloneqq
\mathop{\rm argmin}_{y\in\mathcal{Y}}
\mathbb{E}\left[\ell(y,Y)\mid X=x\right].
$$

This suggests that the surrogate $\varphi(\cdot,A_m)$ should become consistent as $m\to\infty$. The next corollary of the results below gives this asymptotic consistency.

###### Corollary 3.2.

Let $m=m(n)\to\infty$, and let $\varphi$ be identifying in the sense of Definition 3.1. Then

$$
R_{\varphi,A_m}(f_n)-R_{\varphi,A_m}^{\star}\to 0
\quad\text{implies}\quad
R(f_n)-R^{\star}\to 0.
$$

#### 3.2.3 Identifying surrogates and consistency amplification

When labels have low noise, aggregation strategy (7) gives an explicitly improved comparison inequality

$$
\psi(R(f)-R^{\star})
\leq R_{\varphi}(f)-R^{\star}.
$$

The improvement is that $\psi$ is linear over some range of $\epsilon>0$. Linear growth gives the strongest possible comparison inequality [34, 32].

More generally, strict comparison inequalities such as those in Proposition 3 can be too restrictive. In practice, using inconsistent surrogates can still be convenient [23, 34, 32]. Following Osokin et al. [34], we therefore introduce $(\xi,\zeta)$ consistency. This requires $\psi$ to grow linearly only for $\epsilon\geq\xi$, so that the surrogate captures a “good enough” level of risk.

###### Definition 3.2.

The surrogate loss $\varphi$ and aggregator $A$ yield level-$(\xi,\zeta)$ consistency if there is a function $\psi$ such that
$\psi(\epsilon)\geq\zeta\epsilon$ for $\epsilon\geq\xi$ and

$$
\psi(R(f)-R^{\star})
\leq
R_{\varphi,A}(f)-R_{\varphi,A}^{\star}.
$$

We show that, under minimal assumptions, aggregation (7) can achieve level-$(o_m(1),\zeta)$ consistency even when the surrogate is Fisher inconsistent. Here, $o_m(1)$ denotes a quantity that tends to zero as $m$ grows.

We introduce a quantitative noise condition based on the classical Mammen-Tsybakov conditions [25]; see also Bartlett et al. [4]. Define

$$
\Delta(x)\coloneqq
\min_{\mathsf{d}(s)\notin y^{\star}(x)}
\delta_{\ell}(s,x).
\tag{8}
$$

This is the smallest excess conditional risk of an incorrect prediction. For binary classification with $\mathcal{Y}=\{\pm1\}$,

$$
\Delta(x)=|2P(Y=1\mid X=x)-1|.
$$

More generally, we expect consistent estimation to become harder as $\Delta(x)$ approaches zero.

We write the Mammen-Tsybakov conditions as follows, allowing the constant $\mathsf{C}_{\mathsf{MT}}>0$ to change:

$$
\mathbb{P}\left(
\mathsf{d}\circ f\neq\mathsf{d}\circ f^{\star}
\right)
\leq
\mathsf{C}_{\mathsf{MT}}
\left(R(f)-R^{\star}\right)^{\alpha}
\quad\text{for every measurable }f,
\tag{Nα}
$$

and

$$
\mathbb{P}(\Delta(X)\leq\epsilon)
\leq
(\mathsf{C}_{\mathsf{MT}}\epsilon)^{\beta}
\quad\text{for }\epsilon>0.
\tag{Mβ}
$$

We call $\alpha$ in condition (Nα) the noise exponent. Here,
$\alpha\in[0,1]$ and $\beta\in[0,\infty]$.
Both conditions always hold trivially with $\alpha=\beta=0$. As in the binary case [4, Thm. 3], they are equivalent under the transformation
$\beta=\frac{\alpha}{1-\alpha}$.
See Appendix B.2.

Mao et al. [29] recently used exactly the same noise statistic for $H$-consistency in multiclass classification, where consistency is measured within a specified hypothesis class $H$. Their structural assumptions on the surrogate and hypothesis class allow a global convex lower bound. We instead use a general setting with minimal structural constraints. This requires the following noise condition number, which compares the largest and smallest excess losses of incorrect predictions:

$$
\kappa(x)\coloneqq
\frac{
\max_{\mathsf{d}(s)\neq y^{\star}(x)}
\delta_{\ell}(s,x)
}{
\min_{\mathsf{d}(s)\neq y^{\star}(x)}
\delta_{\ell}(s,x)
}.
\tag{9}
$$

It connects $\Delta(x)$ to pointwise excess risk through

$$
\Delta(x)\geq
\frac{\delta_{\ell}(s,x)}{\kappa(x)}
\quad\text{for every }s
\text{ such that }\mathsf{d}(s)\neq y^{\star}(x).
$$

This permits a more detailed analysis. In binary classification,
$\kappa(x)=1$ as long as $\mathbb{P}(\Delta(X)>0)=1$.

The statistic $\Delta(x)$ and condition number $\kappa(x)$ let us show how generalized majority vote (7), used with any identifying surrogate from Definition 3.1, achieves level-$(\xi,\zeta)$ consistency. Define the error function

$$
e_m(t)\coloneqq
t\sqrt{
\frac{2}{m}
\log\left(
\frac{
4k(\mathsf{C}_{\varphi,1}+\mathsf{C}_{\varphi,2})
}{
\mathsf{C}_{\varphi,1}
}
\right)
}.
\tag{10}
$$

Here, $\log$ denotes the logarithm. Roughly, if $\kappa(x)=t$, majority vote $A_m$ is likely to be correct when $m$ is large enough that $e_m(t)\ll1$.

The next theorem gives a nearly linear calibration function. Appendix B.3 proves it.

###### Theorem 1.

Let the surrogate loss $\varphi$ be
$(\mathsf{C}_{\varphi,1},\mathsf{C}_{\varphi,2})$-identifying with parameters
$\{a_y\}_{y\in\mathcal{Y}}$.
Let $A_m$ be the majority vote aggregator (7). Assume that the task loss satisfies $0\leq\ell\leq1$ and that $P$ satisfies condition (Nα) with noise exponent $\alpha\in[0,1]$.

Then, for any $M>0$ and $f\in\mathcal{F}$ such that

$$
R(f)-R^{\star}
\geq
2\mathbb{P}(\kappa(X)>M)
+
\left(4\mathsf{C}_{\mathsf{MT}}e_m(M)\right)^{\frac{1}{1-\alpha}},
$$

we have

$$
R(f)-R^{\star}
\leq
\frac{16}{\mathsf{C}_{\varphi,1}}
\cdot
\left(
R_{\varphi,A_m}(f)-R_{\varphi,A_m}^{\star}
\right).
$$

Equivalently, under these conditions, $\varphi$ with aggregation gives level-$(\xi,\zeta)$ consistency from Definition 3.2, with

$$
\xi=
2\mathbb{P}(\kappa(X)>M)
+
\left(4\mathsf{C}_{\mathsf{MT}}e_m(M)\right)^{\frac{1}{1-\alpha}},
\qquad
\zeta=\frac{\mathsf{C}_{\varphi,1}}{16}.
$$

Theorem 1 also immediately proves Corollary 3.2, the asymptotic consistency guarantee. Define

$$
\xi_m\coloneqq
\inf_M
\left\{
2\mathbb{P}(\kappa(X)>M)
+
\left(4\mathsf{C}_{\mathsf{MT}}e_m(M)\right)^{\frac{1}{1-\alpha}}
\right\}.
$$

Then $\xi_m\to0$ as $m\to\infty$. This follows because
$\mathbb{P}(\kappa(X)>M)\to0$ as $M\uparrow\infty$, while
$e_m(M)\to0$ as $m$ grows for any fixed $M$. Taking $\alpha=0$ gives Corollary 3.2 directly.

Theorem 1 gives a fairly coarse result because Definition 3.1 imposes such weak identifiability conditions. A closer connection between the task loss $\ell$ and surrogate $\varphi$ could yield stronger bounds. For example, that connection might make generalized majority vote (7) more likely to be correct, or at least correct enough for $\varphi$. If $\varphi$ is nearly consistent without aggregation, we might expect its smaller inconsistency to imply stronger bounds in $m$ with majority vote. We do not pursue those details here.

For a more concrete bound, we optimize over $M$ in Theorem 1 using the coarse estimate
$\kappa(x)\leq1/\Delta(x)$.
Take $M=1$ when $\mathop{\rm card}(\mathcal{Y})=k=2$, and optimize $M$ when $k\geq3$. We may then lower bound $\xi_{m,k}$ in the level-$(\xi_{m,k},\zeta)$ consistency promised by Theorem 1, setting

$$
\xi_{m,k}\coloneqq
\begin{cases}
\left(
\frac{32\mathsf{C}_{\mathsf{MT}}^{2}}{m}
\log\left(
\frac{
8(\mathsf{C}_{\varphi,1}+\mathsf{C}_{\varphi,2})
}{
\mathsf{C}_{\varphi,1}
}
\right)
\right)^{\frac{1}{2(1-\alpha)}},
&\text{if }k=2,\\[6pt]
4\cdot
\left(
\frac{32\mathsf{C}_{\mathsf{MT}}^{4}}{m}
\log\left(
\frac{
4k(\mathsf{C}_{\varphi,1}+\mathsf{C}_{\varphi,2})
}{
\mathsf{C}_{\varphi,1}
}
\right)
\right)^{\frac{\alpha}{2(1-\alpha^{2})}},
&\text{otherwise}.
\end{cases}
$$

Algebraic substitutions and rearrangements give the following corollary; see Appendix B.4.

###### Corollary 3.3.

Under the conditions of Theorem 1, for any $f$ such that
$R(f)-R^{\star}\geq\xi_{m,k}$,

$$
R(f)-R^{\star}
\leq
\frac{16}{\mathsf{C}_{\varphi,1}}
\cdot
\left(
R_{\varphi,A_m}(f)-R_{\varphi,A_m}^{\star}
\right).
$$

This result and Corollary 3.2 support the robustness of label cleaning. Aggregation can yield consistency under minimal assumptions on the surrogate.

As the noise exponent $\alpha$ approaches $1$ in Corollary 3.3, the sample size $m$ needed for the comparison inequality to hold for a fixed scoring function $f$ decreases. In particular, if $\alpha=1$, then whenever

$$
m\geq
32\max\{
\mathsf{C}_{\mathsf{MT}}^{2},
\mathsf{C}_{\mathsf{MT}}^{4}
\}
\cdot
\log\left(
\frac{
4k(\mathsf{C}_{\varphi,1}+\mathsf{C}_{\varphi,2})
}{
\mathsf{C}_{\varphi,1}
}
\right)
=
O(\log k),
$$

we have $\xi_{m,k}=0$. Here, $O(\log k)$ indicates growth bounded by a constant multiple of $\log k$. We therefore obtain the uniform comparison inequality (ii) with a linear comparison function. The learning problem’s noise level affects how much aggregation consistency requires: an easier problem needs less aggregation to achieve stronger consistency.

### 3.3 Surrogate consistency examples with majority vote

We give several examples, with varying levels of detail, that apply Theorem 1 and Corollary 3.3. Throughout, assume that $P$ has noise exponent $\alpha\in[0,1]$. This loses no generality because condition (Nα) always holds with $\alpha=0$. Appendix B.5 gives the proofs for this section.

#### 3.3.1 Binary classification with a nonsmooth surrogate

Consider binary classification with a margin-based surrogate
$\varphi(f(x),y)=\phi(yf(x))$, where $\phi$ is convex. Bartlett et al. [4] show that $\varphi$ is consistent if and only if
$\phi'(0)<0$, where the prime denotes a derivative.

We give a somewhat trivial example of aggregation’s robustness: even if $\phi$ is inconsistent without aggregation, aggregation can make it consistent. One would never use such a surrogate, so this example should be viewed as a thought experiment.

Assume that the subgradient set
$\partial\phi(0)\subset(-\infty,0)$.
A subgradient gives a supporting slope for a convex function, including at points where a derivative may not exist. Also assume that $\phi$ is convex and
$\lim_{t\to\infty}\phi(t)=0$.

###### Lemma 3.1.

For any $\delta>0$, $\varphi$ is
$(\mathsf{C}_{\varphi,1},\mathsf{C}_{\varphi,2})$-feasible with

$$
\mathsf{C}_{\varphi,1}
=
\phi(0)-\phi(\delta)>0,
\qquad
\mathsf{C}_{\varphi,2}
=
\phi(-\delta).
$$

Corollary 3.3 therefore applies with $k=2$. Thus, if
$f:\mathcal{X}\to\mathbb{R}$ satisfies

$$
R(f)-R^{\star}
\geq
\left(
\frac{32\mathsf{C}_{\mathsf{MT}}^{2}}{m}
\log\left(
\frac{
8(\phi(-\delta)+\phi(0)-\phi(\delta))
}{
\phi(0)-\phi(\delta)
}
\right)
\right)^{\frac{1}{2(1-\alpha)}},
$$

then

$$
R(f)-R^{\star}
\leq
\frac{16}{\phi(0)-\phi(\delta)}
\left(
R_{\varphi,A_m}(f)-R_{\varphi,A_m}^{\star}
\right).
$$

#### 3.3.2 Bipartite matching

In general structured prediction problems [33], labels have internal structure and usually form a large space $\mathcal{Y}$. An embedding map
$v:\mathcal{Y}\to\mathbb{R}^{d}$
represents that structure as a vector. Use the decoder

$$
\mathsf{d}(s)=
\mathop{\rm argmax}_{y\in\mathcal{Y}}
\langle s,v(y)\rangle,
$$

where $\langle\cdot,\cdot\rangle$ denotes the inner product. For a loss
$\ell:\mathcal{Y}\times\mathcal{Y}\to\mathbb{R}_{+}$
with $\ell(y,y)=0$, the maximum-margin surrogate, or generalized hinge loss [42, 44, 19], is

$$
\varphi(s,y)=
\max_{\hat y\in\mathcal{Y}}
\left(
\ell(\hat y,y)
+
\langle v(\hat y)-v(y),s\rangle
\right).
\tag{11}
$$

This loss is usually inconsistent except in certain low-noise settings [34, 32].

Before addressing structured prediction generally, consider bipartite matching. A bipartite matching is a graph $G=(V,E)$ whose vertices split into left and right sets,

$$
V=V_1\cup V_2,
\qquad
V_1=\{1,\ldots,N\},
\qquad
V_2=\{N+1,\ldots,2N\}.
$$

Its $N$ edges each connect exactly one unique node in $V_1$ to one unique node in $V_2$. Let $\mathcal{Y}$ contain all bipartite matchings between $V_1$ and $V_2$. Then
$k=\mathop{\rm card}(\mathcal{Y})=N!$.

For any graph $G$, the embedding

$$
v(G)\coloneqq
\left(
1\{(u,v)\in E\}
\right)_{u\in V_1,v\in V_2}
\in\mathbb{R}^{N^{2}}
$$

indexes its edges, so $d=N^{2}$. The task loss counts mistaken edges:

$$
\ell(y_1,y_2)\coloneqq
\frac{1}{2N}\|v(y_1)-v(y_2)\|_1
=
\frac{1}{2N}\|v(y_1)-v(y_2)\|_2^2.
$$

Here, $\|\cdot\|_1$ sums the absolute values of the coordinates. In this setting, the maximum-margin, or structured hinge loss, surrogate (11) is identifying.

###### Lemma 3.2.

For bipartite matching on $2N$ vertices, the structured hinge loss surrogate $\varphi$ in (11) is
$(\mathsf{C}_{\varphi,1},\mathsf{C}_{\varphi,2})$-identifying in the sense of Definition 3.1, with

$$
\mathsf{C}_{\varphi,1}=\frac{1}{N},
\qquad
\mathsf{C}_{\varphi,2}=2.
$$

The key consequence is that, even when
$k=\mathop{\rm card}(\mathcal{Y})=N!$,
aggregation can yield consistency through structured hinge loss once the number $m$ of aggregated labels exceeds $O(N\log N)$.

For one specialization, substitute these constants into Corollary 3.3 for $k\geq3$. For every measurable
$f:\mathcal{X}\to\mathbb{R}^{d}$ such that

$$
R(f)-R^{\star}
\geq
4\cdot
\left(
\frac{32\mathsf{C}_{\mathsf{MT}}^{4}}{m}
\log\left(4k(2N+1)\right)
\right)^{\frac{\alpha}{2(1-\alpha^{2})}},
$$

we have

$$
R(f)-R^{\star}
\leq
16N\left(
R_{\varphi,A_n}(f)-R_{\varphi,A_n}^{\star}
\right).
$$

#### 3.3.3 Structured prediction

Return to the general structured prediction setting introduced in the preceding subsection. Suppose the decoder $\mathsf{d}$ can select any class $y\in\mathcal{Y}$. Specifically, assume that for each $y$, the set of scores selecting it,

$$
\mathcal{S}(y)\coloneqq
\left\{
s:
\langle v(y),s\rangle
>
\langle v(\hat y),s\rangle
\text{ for }\hat y\neq y
\right\},
$$

is nonempty. Define the identifiability gap for each $y\in\mathcal{Y}$ as

$$
\tau(y)\coloneqq
\inf_{s\in\mathcal{S}(y)}
\max_{y_+,y_-\neq y}
\frac{
\ell(y_+,y)
}{
\langle v(y)-v(y_+),s\rangle
}
\cdot
\frac{
\langle v(y)-v(y_-),s\rangle
}{
\ell(y_-,y)
}.
$$

This quantity gives the following identifiability guarantee.

###### Lemma 3.3.

For structured prediction, the maximum-margin surrogate $\varphi$ in (11) is
$(\mathsf{C}_{\varphi,1},\mathsf{C}_{\varphi,2})$-identifiable with

$$
\mathsf{C}_{\varphi,1}
=
\min_{\hat y\neq y}\ell(\hat y,y),
\qquad
\mathsf{C}_{\varphi,2}
=
\max_{y\in\mathcal{Y}}\tau(y)+1.
$$

In particular, if $v(y)\in\{0,1\}^{d}$ and

$$
\ell(\hat y,y)
=
\frac{1}{2d}\|v(\hat y)-v(y)\|_1,
$$

then $\tau(y)=1$ for every $y$ and
$\mathsf{C}_{\varphi,2}=2$.

As in the binary matching case, nontrivial consistency guarantees hold once
$m\geq\log\mathop{\rm card}(\mathcal{Y})$.
Because $0\leq\ell(\cdot,\cdot)\leq1$, Corollary 3.3 applies. For every measurable
$f:\mathcal{X}\to\mathbb{R}^{d}$,

$$
R(f)-R^{\star}
\geq
\left(
\frac{32\mathsf{C}_{\mathsf{MT}}^{4}}{m}
\log\left(
4|\mathcal{Y}|
\left(
1+
\frac{
\max_{\hat y\in\mathcal{Y}}\tau(y)+1
}{
\min_{\hat y\neq y}\ell(\hat y,y)
}
\right)
\right)
\right)^{\frac{\alpha}{2(1-\alpha^{2})}}
$$

implies

$$
R(f)-R^{\star}
\leq
\frac{8}{\min_{\hat y\neq y}\ell(\hat y,y)}
\left(
R_{\varphi,A_m}(f)-R_{\varphi,A_m}^{\star}
\right).
$$

## 4 Robustness and consistency for models

The previous section builds on classical surrogate risk consistency theory, which assumes that $\mathcal{F}$ contains all measurable functions. Its results show that aggregation can strengthen consistency. Even if a surrogate $\varphi$ is inconsistent for paired, non-aggregated data $(X,Y)$, sufficient aggregation can give level-$(\xi,\zeta)$ consistency from Definition 3.2.

We now study consistency when optimization uses a restricted, often parametric, hypothesis class $\mathcal{F}$. Consistency with such a class poses no issue in a well-specified model, but assuming perfect specification is unrealistic and brittle. This leads to the long-standing problem of quantifying surrogate consistency when the hypothesis class contains only some measurable functions [12, 31].

Recent work develops $H$-consistency bounds that link excess target risk to surrogate regret—the excess surrogate risk—within a regret hypothesis class $H$, across several settings [2, 3, 26, 28, 29]. Those results focus mainly on surrogates. Our contribution takes a different direction. It depends less on the particular class and surrogate and instead studies how aggregation improves consistency itself. We show that aggregating labels allows consistent estimates in settings where consistency might otherwise fail.

### 4.1 Consistency failure for binary classification in
finite dimensions

To see how restricting the hypothesis class can substantially change even a well-understood problem, consider binary classification. Let
$\mathcal{Y}=\{\pm1\}$ and use zero-one error,

$$
\ell(\mathsf{d}(s),y)=1\{ys\leq0\}.
$$

Take a margin-based surrogate
$\varphi(s,y)=\phi(sy)$, where
$\phi:\mathbb{R}\to\mathbb{R}_{+}$ is convex. As discussed above, when $\mathcal{F}$ contains all measurable functions, $\varphi$ achieves both Fisher consistency (i) and uniform consistency (ii) if and only if $\phi'(0)<0$ [4].

Now restrict the hypothesis class. We show that classical consistency fails even when $\mathcal{F}$ contains optimal classifiers. In particular, this can happen when $P$ is optimally predictable using $\mathcal{F}$, meaning that

$$
\mathsf{sgn}(f(x))
=
\mathsf{sgn}(P(Y=1\mid X=x)-1/2).
\tag{12}
$$

Let $\mathcal{X}=\mathbb{R}^{d}$, and take the class of linear functions

$$
\mathcal{F}
=
\{f_{\theta}\mid
f_{\theta}(x)=\langle\theta,x\rangle\}_{\theta\in\mathbb{R}^{d}}.
$$

When $P$ is optimally predictable using $\mathcal{F}$, there is a vector $\theta^{\star}$ such that

$$
\mathsf{sgn}(\langle\theta^{\star},x\rangle)
=
\mathsf{sgn}\left(P(Y=1\mid x)-\frac12\right).
$$

Then $f_{\theta^{\star}}$ minimizes $R(f)$ over all measurable functions. We call this optimal predictability along $\theta^{\star}$.

One might expect a margin-based surrogate that is Fisher consistent in the classical setting to remain consistent here. It does not. More strongly, for any nonnegative loss $\phi$, there is a distribution $P$ on $(X,Y)$ such that

$$
\theta_{\varphi}
=
\mathop{\rm argmin}_{\theta}
\mathbb{E}_{P}[\varphi(f_{\theta}(X),Y)]
$$

is essentially orthogonal to $\theta^{\star}$, meaning that the angle is nearly a right angle.

###### Proposition 4.

For any $\epsilon>0$ and nonzero
$\theta^{\star}\in\mathbb{R}^{d}$, there is a distribution $P$ on $(X,Y)$ that is optimally predictable along $\theta^{\star}$ such that every

$$
\theta_{\varphi}
\in
\mathop{\rm argmin}_{\theta}R_{\varphi}(f_{\theta})
=
\mathop{\rm argmin}_{\theta}
\mathbb{E}\left[\phi(f_{\theta}(X),Y)\right]
$$

satisfies
$R(f_{\theta_{\varphi}})>R(f_{\theta^{\star}})$ and

$$
|\cos\angle(\theta_{\varphi},\theta^{\star})|
=
\frac{
|\langle\theta_{\varphi},\theta^{\star}\rangle|
}{
\|\theta_{\varphi}\|_2\|\theta^{\star}\|_2
}
\leq\epsilon.
$$

Appendix D.1 gives the proof.

Aggregation offers one way around this inconsistency. To state the result, define approximate minimizers by

$$
\mathop{\epsilon\textup{-argmin}}g
=
\mathop{\epsilon\textup{-argmin}}_{\theta}g(\theta)
\coloneqq
\left\{
\theta:
g(\theta)\leq\inf_{\theta}g(\theta)+\epsilon
\right\}.
$$

Suppose data collection gives independent labels
$Z=(Y_1,\ldots,Y_m)$, and let $A_m(Z)$ be their majority vote. For a sequence $\epsilon_m$, choose

$$
\theta_m
\in
\mathop{\epsilon_m\textup{-argmin}}_{\theta}
R_{\varphi,A_m}(f_{\theta})
=
\mathop{\epsilon_m\textup{-argmin}}_{\theta}
\mathbb{E}\left[
\phi(A_m(Y_1,\ldots,Y_m)f_{\theta}(X))
\right].
$$

As a corollary of Theorem 2 below, the functions $f_{\theta_m}$ are asymptotically consistent as $m\to\infty$.

###### Corollary 4.1.

Let $P$ be optimally predictable along $\theta^{\star}$. If
$\epsilon_m\to0$ as $m\to\infty$, then

$$
R(f_{\theta_m})\to R(f_{\theta^{\star}})
\quad\text{and}\quad
\cos\angle(\theta_m,\theta^{\star})\to1.
$$

Thus, without aggregation, surrogate risk minimization over linear predictors can be essentially arbitrarily incorrect by Proposition 4. With aggregation, we retain consistency.

### 4.2 Aggregation, consistency, and restricted hypothesis classes

Proposition 4 shows that surrogate risk consistency depends fundamentally on $\mathcal{F}$ containing all measurable functions. We now consider multiclass classification with
$\mathcal{Y}=\{1,\ldots,k\}$, where $\mathcal{F}$ forms a linear cone satisfying

$$
f(x)^T\mathbf{1}=0,
\qquad
tf\in\mathcal{F}\text{ for }t>0
\text{ if }f\in\mathcal{F}.
$$

The cone condition means that positive rescaling keeps a function in the class.

Use zero-one loss
$\ell(y,y')=1\{y\neq y'\}$ and decoder
$\mathsf{d}(s)=\mathop{\rm argmax}_{y\in[k]}s_y$.
The normalization $f(x)^T\mathbf{1}=0$ does not affect these predictions.

Assume that the surrogate
$\varphi:\mathbb{R}^{k}\times[k]\to\mathbb{R}_{+}$
is Fisher consistent (i) and satisfies the limiting loss condition

$$
\varphi(s,y)\to0
\quad\text{if}\quad
s_y-s_j\to+\infty
\text{ for every }j\neq y.
\tag{13}
$$

Many familiar surrogates are Fisher consistent and satisfy (13). They include multiclass logistic loss,

$$
\varphi(s,y)
=
\log\left(\sum_{j=1}^{k}e^{s_j-s_y}\right),
$$

and any loss of the form

$$
\varphi(s,y)
=
\sum_{i\neq y}\phi(s_y-s_i)
$$

where $\phi$ is convex and non-increasing,
$\phi'(0)<0$, and $\inf_t\phi(t)=0$.
Zhang [51, Thm. 8] shows that any such loss is consistent over

$$
\mathcal{F}
=
\{f:\mathcal{X}\to\mathbb{R}^{k}\mid\mathbf{1}^{T}f=0\}.
$$

The margin-based binary setting in Section 4.1 fits this setting by taking
$f(x)=(g(x),-g(x))$ for a measurable $g$.

In a parametric setting with $\mathcal{X}=\mathbb{R}^{d}$, the class of linear functions

$$
f(x)=
(\langle\theta_1,x\rangle,\ldots,\langle\theta_k,x\rangle),
\qquad
\sum_{i=1}^{k}\theta_i=0,
$$

also forms a convex cone. These examples typically require strong structural assumptions, such as convexity, for consistency without aggregation. As we show next, aggregation gives asymptotic consistency with the minimal limiting loss condition (13).

Extend optimal predictability from (12) to multiple classes. We say that $\mathcal{F}$ can optimally predict $P$ if there is a function
$f\in\mathcal{F}$, $f:\mathcal{X}\to\mathbb{R}^{k}$, such that

$$
\mathop{\rm argmax}_{y}f_y(x)
\in
\mathop{\rm argmax}_{y}P(Y=y\mid X=x)
\quad\text{for every }x.
$$

The next theorem assumes $\mathcal{Z}=\mathcal{Y}^{m}$ and uses majority vote $A_m$. It shows that, when
$y^{\star}(x)=\mathop{\rm argmax}_{y}P(Y=y\mid x)$
is unique, surrogate risk minimization is consistent whenever $\mathcal{F}$ can optimally predict $P$.

###### Theorem 2.

Let $\mathcal{F}$ be a cone that optimally predicts $P$, and assume that the minimal excess risk (8) satisfies
$P(\Delta(X)>0)=1$.
Let $\epsilon_m\geq0$ satisfy $\epsilon_m\to0$.

Then, for any sequence

$$
f_m
\in
\mathop{\epsilon_m\textup{-argmin}}_{f\in\mathcal{F}}
R_{\varphi,A_m}(f)
=
\mathop{\epsilon_m\textup{-argmin}}_{f\in\mathcal{F}}
\mathbb{E}\left[
\varphi(f(X),A_m(Y_1,\ldots,Y_m))
\right],
$$

we have $R(f_m)\to R^{\star}$.

Appendix D.2 gives a proof.

Theorem 2 shows that aggregation gives asymptotic consistency as $m\to\infty$ for a broad class of surrogate problems whose hypothesis class forms a linear cone. In contrast, Proposition 4 shows that consistency can fail even for binary classification over subclasses $\mathcal{F}$ that include the optimal predictor. In that setting, the surrogate can be arbitrarily uninformative.

### 4.3 On finite-dimensional multiclass classification

Our final technical results concern multiclass classification with
$\mathcal{X}=\mathbb{R}^{d}$, $\mathcal{Y}=[k]$, and linear predictors, when the predictive model may be mis-specified. These results give a more detailed and specialized view than Proposition 4 and Theorem 2.

We show that surrogate risks based only on $(X,Y)$ pairs are inconsistent even when the problem is optimally predictable and the linear model is well-specified on all but an $\epsilon$ fraction of the data. Majority vote methods, however, recover the optimal linear predictor.

Let

$$
\Theta=[\theta_1~\cdots~\theta_{k-1}]
\in\mathbb{R}^{d\times k-1}.
$$

Assume that labels follow the categorical distribution
$Y\mid X=x\sim\mathsf{Cat}(p_{\Theta}(x))$,
which assigns probabilities to the $k$ classes. Here,
$p_{\Theta}(x)\in\mathbb{R}_{+}^{k}$,
$\mathbf{1}^{\top}p_{\Theta}(x)=1$, and

$$
p_{\Theta}(x)=
\begin{bmatrix}
\sigma(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)\\
1-\mathbf{1}^{\top}
\sigma(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)
\end{bmatrix}.
$$

The link function
$\sigma:\mathbb{R}^{k-1}\to\mathbb{R}_{+}^{k-1}$
converts scores to probabilities and satisfies
$\mathbf{1}^{\top}\sigma\leq1$.

Assume that $\sigma$ satisfies the following consistency condition. For
$t\in\mathbb{R}^{k-1}$, set $t_k=0$ and
$\sigma_k(t)=1-\mathbf{1}^{\top}\sigma(t)$. Then

$$
t_y=\max_{1\leq i\leq k}t_i
\quad\text{if and only if}\quad
\sigma_y(t)=\max_{1\leq i\leq k}\sigma_i(t).
\tag{14}
$$

A standard example is multiclass logistic regression, where

$$
\sigma_i^{\textup{lr}}(t)
=
\frac{e^{t_i}}{1+e^{t_1}+\cdots+e^{t_{k-1}}}.
$$

The superscript “lr” identifies the logistic regression link.

Let the dataset with repeated labels be

$$
\{(X_i;(Y_{1i},\cdots,Y_{im}))\}_{i=1}^{n},
$$

with repeated sampling

$$
Y_{ij}\mid X_i
\stackrel{\mathrm{iid}}{\sim}
\mathsf{Cat}(p_{\Theta^{\star}}(X_i)).
$$

Let
$Y_i^{+}=A_m(Y_{i1},\cdots,Y_{im})$
be the majority vote, with ties broken arbitrarily. Then

$$
Y_i^{+}\mid X_i
\sim
\mathsf{Cat}(p_{\Theta^{\star},m}(X_i)).
$$

If $\rho_m(t)$ denotes the distribution of majority vote over $m$ items with initial probabilities
$\sigma(t)\in\mathbb{R}_{+}^{k-1}$, then

$$
p_{\Theta,m}(x)=
\begin{bmatrix}
\rho_m(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)\\
1-\mathbf{1}^{\top}
\rho_m(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)
\end{bmatrix}.
$$

Clearly, $\rho_m$ satisfies link consistency (14). Consider fitting logistic regression with loss

$$
\varphi(\Theta^{\top}x,y)
=
-\langle\theta_y,x\rangle
+
\log\left(
1+\sum_{i=1}^{k-1}\exp(\langle\theta_i,x\rangle)
\right),
\tag{15}
$$

using the convention $\theta_k=\mathbf{0}$.
Let

$$
L_m(\Theta)
=
\mathbb{E}[\varphi(\Theta^{\top}X,Y_m^{+})]
$$

be the logistic loss with majority vote over $m$ labels. The gradient of $\varphi$ with respect to $\theta_i$, meaning its vector of partial derivatives, is

$$
\nabla_{\theta_i}\varphi(\Theta^{\top}x,y)
=
-x\left(
1\{y=i\}-\sigma_i^{\textup{lr}}(\Theta^{\top}x)
\right).
$$

Thus, a minimizer $\Theta_m$ of $L_m$ satisfies

$$
\nabla_{\Theta}L_m(\Theta_m)
=
\mathbb{E}\left[
X\left(
\sigma^{\textup{lr}}(\Theta_m^{\top}X)
-
\rho_m({\Theta^{\star}}^{\top}X)
\right)^{\top}
\right]
=
0.
$$

Standard statistical results guarantee consistency and efficiency when the model is well-specified without aggregation. For $m\geq1$ and $k=2$, Cheng et al. [7, Prop. 3] show that there is a $t_m>0$ such that
$\Theta_m=t_m\Theta^{\star}$
if $X\sim\mathsf{N}(0,I_d)$, even with a mis-specified link. Here, $\mathsf{N}(0,I_d)$ is the normal distribution with zero mean and identity covariance matrix.

This means that binary classification can remain consistent with an incorrect link, regardless of the aggregation level $m$. The direction
$\Theta^{\star}/\|\Theta^{\star}\|_2$
determines consistency.

But when $k\geq3$, risk consistency fails as soon as the true link is slightly mis-specified. Fix a set
$\mathcal{T}_{\epsilon}\subset\mathbb{R}^{k-1}$
with Lebesgue measure $\epsilon$, meaning its ordinary volume is $\epsilon$, and consider

$$
\sigma^{\epsilon}(t)
=
\sigma^{\textup{lr}}(t)1\{t\notin\mathcal{T}_{\epsilon}\}
+
\frac{1}{k}\mathbf{1}\cdot
1\{t\in\mathcal{T}_{\epsilon}\}.
$$

This defines a distribution on $Y\in\{1,\ldots,k\}$ conditional on
$t\in\mathbb{R}^{k-1}$ that samples
$Y\sim\sigma^{\textup{lr}}(t)$
if $t\in\mathcal{T}_{\epsilon}$ and samples uniformly otherwise. Clearly, $\sigma^{\epsilon}$ satisfies link consistency (14) and is optimally predictable in the sense of (12).

For $\epsilon>0$, define

$$
L_{m,\epsilon}(\Theta)\coloneqq
\mathbb{E}\left[
\mathbb{E}_{\sigma^{\epsilon}}
[\varphi(\Theta^{\top}X,Y_m^{+})\mid X]
\right].
$$

This is the population logistic loss based on majority vote over $m$ labels when
$Y\mid X=x\sim\sigma^{\epsilon}({\Theta^{\star}}^{\top}x)$.
Let
$\Theta_m(\epsilon)=\mathop{\rm argmin}_{\Theta}L_{m,\epsilon}(\Theta)$.
Clearly, $\Theta_1(0)=\Theta^{\star}$.
Yet the next result shows that consistency fails without aggregation for arbitrarily small $\epsilon>0$.

###### Proposition 5.

Let $k\geq3$ and $\Sigma=I$. Assume that, for
$Z\sim\mathsf{N}(0,I_{k-1})$, the linear mapping

$$
M\mapsto DM
\coloneqq
\mathbb{E}\left[
Z\left(
\nabla\sigma^{\textup{lr}}({\Theta^{\star}}^{\top}Z)MZ
\right)^{\top}
\right]
$$

is invertible. Then there is an $\epsilon_0>0$ such that, for every
$\epsilon\in(0,\epsilon_0)$, there is a set $\mathcal{T}_{\epsilon}$ with Lebesgue measure at most $\epsilon$ for which

$$
\Theta_1(\epsilon)/\|\Theta_1(\epsilon)\|
\neq
\Theta^{\star}/\|\Theta^{\star}\|.
$$

Appendix E.1 gives the proof.

Majority vote can address this inconsistency as $m\to\infty$. It needs no assumption on the true link $\sigma$ other than consistency condition (14). Define

$$
L_{m,\sigma}(\Theta)
=
\mathbb{E}\left[
\mathbb{E}_{\sigma}
[\varphi(\Theta^{\top}X,Y_m^{+})\mid X]
\right],
\qquad
\Theta_m=\mathop{\rm argmin}L_{m,\sigma}(\Theta).
$$

We then have the following result.

###### Theorem 3.

Let $\Theta^{\star}$ have the decomposition
$\Theta^{\star}=U^{\star}T^{\star}$, where
$U^{\star}\in\mathbb{R}^{d\times(k-1)}$ is orthogonal and
$T^{\star}\in\mathbb{R}^{(k-1)\times(k-1)}$ is nonsingular, meaning invertible.

Then there is a matrix
$T_m\in\mathbb{R}^{(k-1)\times(k-1)}$
such that $\Theta_m=U^{\star}T_m$ for every $m$. As $m\to\infty$,

$$
\|T_m\|\to\infty
\quad\text{and}\quad
T_m/\|T_m\|
\to
T^{\star}/\|T^{\star}\|.
$$

Appendix E.2 gives a proof.

Theorem 3 provides more evidence for the robustness of label aggregation. It gives asymptotic consistency even in mis-specified models, as long as some link function describes the relationship between $X$ and $Y$. This robustness is striking when $k\geq3$: Proposition 5 shows that methods without aggregation are generally inconsistent.

## 5 Numerical illustrations

We present two experiments that support the theory. The first is a nonparametric example in which aggregation amplifies surrogate consistency. The second is a finite-dimensional multiclass example in which aggregation restores asymptotic consistency under link mis-specification.

##### Surrogate consistency amplification for truncated quadratic loss.

For binary classification, consider the classical margin-based surrogate
$\varphi(f(x),y)=\phi(yf(x))$, where

$$
\phi(\delta)=\max\{1-\delta,0\}^{2}
$$

is the truncated quadratic loss. Bartlett et al. [4, Example 2] show that the calibration function in Corollary 3.1 is exactly
$\psi(\epsilon)=\epsilon^{2}$.

With majority vote over $m$ labels, we can apply Lemma 3.1 using $\delta=1$. With no low-noise condition, so that
$\alpha=0$ and $\mathsf{C}_{\mathsf{MT}}=1$, take

$$
\begin{aligned}
\psi_m(\epsilon)
&=
\frac{\phi(0)-\phi(\delta)}{16}
\left(
\epsilon-
\sqrt{
\frac{32}{m}
\log\left(
\frac{
8(\phi(-\delta)+\phi(0)-\phi(\delta))
}{
\phi(0)-\phi(\delta)
}
\right)
}
\right)_{+}\\
&=
\frac{1}{16}
\left(
\epsilon-\sqrt{\frac{32}{m}\log40}
\right)_{+}.
\end{aligned}
$$

Here, $(\cdot)_{+}$ keeps the positive part and replaces negative values with zero. We obtain parallel comparison inequalities for data without aggregation and for majority vote aggregation:

$$
\psi(R(f)-R^{\star})
\leq R_{\varphi}(f)-R_{\varphi}^{\star},
\qquad
\psi_m(R(f)-R^{\star})
\leq R_{\varphi,A_m}(f)-R_{\varphi,A_m}^{\star}.
$$

The numerical results in Figure 1 show the aggregated curve shifting from quadratic toward linear behavior as $m\to\infty$. This shows how label aggregation strengthens surrogate consistency. Still, the functions $\psi_m$ are generally conservative because they come from the general result in Theorem 1.

Figure 1: Numerical illustrations for the truncated quadratic surrogate $\phi(\delta)=\max\{1-\delta,0\}^{2}$. The figure compares comparison inequalities for non-aggregated data, $\psi$ (optimal), with majority vote aggregation, $\psi_m$ (suboptimal), for $m\in\{2^{16},2^{20},2^{24},2^{28},2^{32}\}$.

##### Asymptotic consistency for mis-specified logistic model.

For the second example, consider the logistic regression setting of Section 4.3 with $k=3$ and $d=2$. Use logistic loss $\varphi$ from equation (15) and the corrupted logistic link

$$
\tilde{\sigma}(t)
=
\sigma^{\textup{lr}}(t)1\{\|t\|\leq r\}
+
\sigma^{\textup{lr}}(\alpha t)1\{\|t\|>r\}.
$$

This piecewise link matches softmax, the logistic probability map, near the origin and shrinks scores outside radius $r$. Figure 2 gives numerical results from a synthetic dataset generated with this corrupted link. The results show asymptotic consistency as $m\to\infty$.

| (i) Experiment with constant dataset size $n$ | (ii) Experiment with constant label size $nm$ . |
|---|---|

Figure 2: Numerical illustrations for mis-specified logistic models with $d=2$ and $k=3$. The true parameters are $\theta_1^{\star}=(0,1)^{\top}$ and $\theta_2^{\star}=(1,1)^{\top}$. The corrupted link parameters are $r=3$ and $\alpha=0.1$. The plot shows the classification rule error $\left\|\|T_m\|-T^{\star}/\|T^{\star}\|\right\|$ against the number of labels $m\in\{1,4,16,64,256,1024\}$. (i) In the left panel, for each $m$, the synthetic dataset contains $n=10{,}000$ observations with repeated labels, $(X_i,(Y_{i1},\ldots,Y_{im}))$, sampled from $\tilde{\sigma}$. (ii) In the right panel, for each $m$, we set $n=640{,}000/m$ so that the total label count stays constant across values of $m$. We report a $95\%$ error bar over $T=100$ trials for each $m$.

## 6 Discussion and future work

Whether and how to clean data has driven much of the research discussion about dataset collection. Cheng, Asi and Duchi [7] discuss these issues. They highlight an apparent pattern: using all available, non-aggregated labels gives better statistical efficiency when models can fully represent uncertainty. Otherwise, data cleaning appears more robust.

Similarly, Dorner and Hardt [11] argue that, when comparing binary classifiers in a validation setting, using more noisy labels is better than using cleaned versions. Our paper contributes evidence of fundamental limits to using uncleaned, non-aggregated labels in supervised learning. It also shows robustness gains from label cleaning. Many questions remain.

##### Finite $m$ results and fundamental limits.

Many of our consistency results require the number $m$ of aggregated labels to tend to infinity. Our purpose is partly to show how aggregation can improve robustness, but relying on this asymptotic setting may be unsatisfying.

For ranking in Section 3.1, we give explicit consistency guarantees for finite $m$. Extending these results is one of the most natural and, we believe, important directions for future work.

It would be useful to develop surrogate consistency theory that depends on both the loss pair $(\ell,\varphi)$ and the available label count $m$. For example, in the ranking setting of Section 3.1, suppose we want second- or third-order item comparisons, such as powers $C_x^{p}$ proposed by Keener [20]. Must we increase the label count $m$? A central challenge is to distinguish precisely which problems require label cleaning and aggregation and which do not.

##### Fundamental limits of the noise condition number.

We use the noise condition number $\kappa(X)$ from (9) to characterize comparison inequalities for multiclass classification. It indicates the added difficulty beyond binary classification, where $\kappa(X)=1$ trivially.

Beyond binary classification, this condition number can remain large even when the Mammen-Tsybakov noise level (Nα) is low, meaning $\alpha\approx1$. This follows from our minimal assumptions on the surrogate. It would help to identify relationships between task loss $\ell$ and surrogate $\varphi$ that capture problem difficulty more closely.

Constructing explicit failure cases would also clarify the fundamental limits and give more insight into fitting predictive models.

##### Behavior in mis-specified models.

Our results for mis-specified models, especially in Section 4.3, require optimal predictability (12). That is, $\mathcal{F}$ must contain a Bayes-optimal classifier, a classifier that minimizes risk over all measurable prediction rules.

Classical surrogate consistency results provably fail even in this setting, while methods that aggregate labels can succeed. Moving beyond these restricted settings seems useful and interesting.

Nguyen et al. [31], followed by Duchi et al. [12], identify one direction. For binary and multiclass classification, respectively, they show that jointly learning a predictor $f$ and a representation of the input $x$ requires surrogates $\varphi$ of a particular form determined by the task loss $\ell$. These results still rely on infinitely powerful decision rules $f$, so new approaches are needed.

##### Restricted class comparison inequality with label aggregation.

Our paper studies how label aggregation reduces surrogate-induced regret through comparison inequalities in nonparametric settings. It also establishes statistical consistency in parametric settings.

Recent work on $H$-consistency [2, 28, 27, 29] motivates extending these techniques to comparison inequalities for restricted hypothesis classes. In particular, Mao et al. [27] show that quadratic growth of $\psi$ is rate optimal for smooth convex loss. It would be interesting to use aggregation to strengthen consistency in the $\mathcal{H}$-consistency setting.

Such an extension could provide a unified framework connecting aggregation’s benefits across parametric and nonparametric settings. It could also clarify how the “size” of the hypothesis class $\mathcal{F}$ affects aggregation’s effectiveness.
