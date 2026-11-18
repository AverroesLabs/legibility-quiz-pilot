# Some Robustness Properties of Label Cleaning

Chen Cheng

  
John Duchi

###### Abstract

We demonstrate that learning procedures that rely on aggregated labels—for example, label information distilled from noisy responses—enjoy robustness properties impossible without data cleaning. This robustness appears in two ways.

First, consider risk consistency: the standard machine-learning approach minimizes a surrogate loss, typically convex, in place of a desired task loss, such as the zero-one mis-classification error. Procedures using label aggregation obtain stronger consistency guarantees than are possible using raw labels.

Second, classical statistical scenarios involving perfectly-specified models suggest that incorporating all possible information, including modeling uncertainty in labels, is statistically efficient. However, consistency fails for “standard” approaches as soon as the loss to be minimized is even slightly mis-specified. In contrast, procedures leveraging aggregated information still converge to optimal classifiers. Thus, incorporating a fuller view of the data analysis pipeline—from collection to model-fitting to prediction time—can yield a more robust methodology by refining noisy signals.

## 1 Introduction

Consider the data collection pipeline in a supervised learning problem. Naively, we say that we collect $n$ pairs $(X_i,Y_i)_{i=1}^{n}$, where $X_i$ contains features and $Y_i$ is a label, fit a model, and away we go [17]. However, this description belies the complexity of modern datasets [9, 22, 39]. Modern datasets require substantial data cleaning and filtering, often including crowdsourcing multiple labels and then denoising those labels.

The crowdsourcing community has intensively studied such data cleaning, especially for obtaining “gold standard” labels [8, 48, 47, 46, 37]. We take a complementary view: we investigate the ways in which data aggregation fundamentally and necessarily improves the consistency of the models we fit.

In this sense, the paper argues that label cleaning, or aggregating labels together, provides robustness that is impossible without label aggregation. This robustness has two faces:

- **Consistency of estimation using surrogate losses.** When minimizing a surrogate loss, such as the multiclass logistic loss, instead of a task loss, such as the zero-one error, procedures using aggregated labels can achieve consistent and optimal prediction in the limit even when consistent prediction is impossible without aggregation.
- **Consistency in finite-dimensional statistical problems.** Label aggregation can also provide consistent classifiers in finite-dimensional problems where standard methods fail.

Important contributions to surrogate risk consistency theory trace to the 2000s [52, 24, 40]. In particular, Bartlett, Jordan and McAuliffe [4] characterized when fitting a model using a convex surrogate is consistent for binary classification under zero-one error. Subsequent work on surrogate risk consistency includes multi-label classification [50, 43, 16, 49, 1], ranking [13, 14, 36], structured prediction [34, 6, 32], ordinal regression [35], restricted hypothesis classes [2, 3, 28, 29], and general theory [40].

On the one hand, these analyses consider the standard supervised-learning setting of data pairs $(X,Y)$, so they allow us to exploit the statistical theory of empirical processes [45, 5, 21, 4]. On the other hand, these analyses do not address the data aggregation machinery now common in modern dataset creation.

It is therefore natural to ask how consistency interacts with data aggregation. To begin with, do we need to aggregate at all? If surrogate consistency is achievable without aggregation, perhaps we should rely on our mature theoretical understanding of processes involving $(X,Y)$ pairs. If we do aggregate, does aggregation help consistency, and in what sense? These questions motivate the paper.

To further establish the importance of studying consistency with aggregated labels, we give concrete examples in ranking and in binary and multiclass classification with linear estimators. In these examples, estimators using only pairs $(X,Y)$ necessarily fail, whereas label aggregation methods yield consistency.

We develop new notions and theory for surrogate consistency with data aggregation. In fully nonparametric scenarios, we show how the number of aggregated samples combines with noise conditions to improve consistency. Aggregation also allows us to demonstrate surrogate risk consistency under only weak conditions on the surrogate loss. In the language of the field, losses using aggregated labels admit approximate linear comparison inequalities.

Finally, conventional risk consistency theory requires a hypothesis class $\mathcal F$ containing all measurable functions. In contrast, we show classification results in which label aggregation guarantees consistency even over restricted hypothesis classes, although consistency may fail over the same classes without aggregation.

## 2 Preliminaries

We first review classical surrogate risk minimization. Let $\mathcal X$ be the input space and $\mathcal Y$ the output space. The data pairs $(X_i,Y_i)_{i=1}^{n}$ take values in $\mathcal X\times\mathcal Y$ and are drawn independently and identically distributed (i.i.d.) from a distribution $P$.

We learn a scoring function $f:\mathcal X\to\mathbb R^d$, where $d\geq 1$. The scoring function maps an input $x$ to a score $s=f(x)\in\mathbb R^d$. A decoder $\mathsf d:\mathbb R^d\to\mathcal Y$ then determines the final prediction:
\[
\widehat y=\mathsf d\circ f(x)=\mathsf d(f(x)).
\]
Here, $\circ$ denotes function composition.

Given a nonnegative task loss $\ell:\mathcal Y\times\mathcal Y\to\mathbb R_+$ and a hypothesis class $\mathcal F$ of scoring functions, the goal is to minimize the task risk over $f\in\mathcal F$:
\[
R(f):=\mathbb E_P\!\left[\ell(\mathsf d\circ f(X),Y)\right].
\tag{1}
\]
Here, $\mathbb E_P$ denotes expectation under $P$.

For example, in binary classification, $d=1$, $\mathsf d(s)=\mathsf{sgn}(s)$, where $\mathsf{sgn}$ is the sign function, and
\[
\ell(y,y')=1\!\left\{yy'\leq 0\right\},
\]
where $1\{\cdot\}$ denotes an indicator. Consequently,
\[
R(f)=\mathbb P(Yf(X)\leq 0).
\]

Minimizing $R(f)$ is challenging because the task loss $\ell$ can be nonsmooth, nonconvex, and uninformative. In particular, the zero-one, or $0$-$1$, loss has a landscape that is flat almost everywhere. This flatness makes even practical optimization methods, such as first-order methods, impossible.

We consider a slightly more sophisticated version of problem (1). Instead of allowing $\ell$ to depend only on the instantaneous label $Y$, we allow $\ell$ to depend on the conditional label distribution $P(Y\in\cdot\mid X)$. Thus, we investigate
\[
R(f)\coloneqq
\mathbb E_P\!\left[
\ell(\mathsf d\circ f(X),P(\cdot\mid X))
\right].
\tag{2}
\]
The minimizers of this risk frequently coincide with those of the original problem (2), while the distribution-dependent loss allows more sophistication. For example, in multiclass classification with $Y\in\{1,\ldots,k\}$, taking
\[
\ell(\widehat y,P)
=\sum_y P(Y=y)1\{\widehat y\neq y\}
\]
makes the risk coincide with the standard $0$-$1$ error rate.

Because directly minimizing the task risk is difficult, we consider an easier-to-optimize surrogate loss
\[
\varphi:\mathbb R^d\times\mathcal Y\to\mathbb R.
\]
Rather than minimizing the task risk (2) directly, we minimize the surrogate risk
\[
R_\varphi(f):=
\mathbb E_P\!\left[\varphi(f(X),Y)\right].
\]

For surrogate minimization to be sensible, we must establish some form of consistency with the task problem (2). We study two scenarios, which we formalize below:

(i) The “classical” case of Fisher consistency, where $\mathcal F$ contains all Borel functions.

(ii) Statistical scenarios where $\mathcal F$ is parametric but may be mis-specified.

Our main message is that label aggregation improves consistency in both scenarios, thereby demonstrating the robustness of label cleaning.

### 2.1 Label aggregation

Instead of obtaining pairs $(X_i,Y_i)$, suppose that we replace the output $Y$ with a more abstract variable $Z\in\mathcal Z$. For example, if we collect $m$ noisy labels for each example $X$, we take
\[
Z=(Y_1,\ldots,Y_m)\in\mathcal Y^m.
\]
Mao et al. [28] consider a similar multilabeled data structure for multilabel losses; in contrast, we focus on the label aggregation pipeline.

Let $\mathcal A$ be an abstract aggregation space, and let
\[
A:\mathcal Z\to\mathcal A
\]
be an aggregating function, such as majority vote. Let
\[
\varphi:\mathbb R^d\times\mathcal A\to\mathbb R_+
\]
be a surrogate loss defined on the aggregation space. We define the aggregated surrogate risk as
\[
R_{\varphi,A}(f):=
\mathbb E\!\left[\varphi(f(X),A(Z))\right].
\tag{3}
\]
The consistency question is therefore whether minimizing the aggregated surrogate risk (3) is sufficient to minimize the actual task risk (2). The following two examples make the aggregation setup concrete.

**Example 1 (Majority vote):**

In the repeated sampling regime, the collected labels are
\[
Z=(Y_1,\ldots,Y_m),
\qquad
Y_i\mid X=x
\stackrel{\mathrm{iid}}{\sim}
P_{Y\mid X=x}.
\]
Define $A_m(Z)$ to be the empirical minimizer
\[
A(Z)=A(\{Y_1,\ldots,Y_m\})
=
\mathop{\rm argmin}_{y\in\mathcal Y}
\sum_{l=1}^{m}\ell(y,Y_l).
\]
When $\ell(\widehat y,y)=1\{y\neq\widehat y\}$, this empirical minimization is exactly majority vote. The more general form permits more abstract aggregation procedures. $\Diamond$

We can also roughly capture $K$-nearest neighbor aggregation.

**Example 2 ($K$-nearest neighbors):**

Consider an abstract repeated sampling scenario in which an example $X$ comes with a label $Y$ and an additional i.i.d. sample
\[
(X_i,Y_i)_{i=1}^{m}\stackrel{\mathrm{iid}}{\sim}P.
\]
Here, $m$ is the number of additional examples, and
\[
Z=(Y,(X_i,Y_i)_{i=1}^{m}).
\]

Let $\mathop{\rm dist}:\mathcal X\times\mathcal X\to\mathbb R_+$ be a distance metric. Order the input sample by distance from $X$:
\[
\mathop{\rm dist}(X,X_{(1)})
\leq\cdots\leq
\mathop{\rm dist}(X,X_{(m)}),
\]
and let $X_{(0)}=X$. The labels $Y_{(l)}$ correspond to the ordered inputs, with $Y_{(0)}=Y$.

For $K\geq 0$, we can aggregate the $K$-nearest neighbors of $X$, for example by choosing
\[
A_{m,K}(Z)\coloneqq
\mathop{\rm argmin}_{y\in\mathcal Y}
\sum_{l=0}^{K}\ell(y,Y_{(l)}).
\]
In Appendix C, we use the results of the coming sections to move beyond this population-level scenario and address aggregation from a single sample $(X_i,Y_i)_{i=1}^{n}$. $\Diamond$

## 3 Surrogate consistency

The standard surrogate consistency framework [40] assumes that $\mathcal F$ consists of all Borel measurable functions $f:\mathcal X\to\mathbb R^d$. In the abstract setting of Section 2, define the conditional task risk and conditional surrogate risk, respectively, for $s\in\mathbb R^d$:
\[
R(s\mid x)\coloneqq
\ell(\mathsf d\circ s,P(Y\in\cdot\mid X=x)),
\qquad
R_\varphi(s\mid x):=
\mathbb E\!\left[\varphi(s,Y)\mid X=x\right].
\]
Here, $\mathsf d\circ s$ denotes decoding the score $s$.

Subtracting the best conditional risk gives the pointwise excess risks
\[
\delta_\ell(s,x)\coloneqq
R(s\mid x)-
\inf_{s'\in\mathbb R^d}R(s'\mid x),
\]
\[
\delta_\varphi(s,x)\coloneqq
R_\varphi(s\mid x)-
\inf_{s'\in\mathbb R^d}R_\varphi(s'\mid x).
\]
We also define the minimal population risks
\[
R^\star:=\inf_{f\in\mathcal F}R(f),
\qquad
R_\varphi^\star:=\inf_{f\in\mathcal F}R_\varphi(f).
\]

Following the standard framework [40, 4, 52], we require at least (i) Fisher consistency and, when possible, the stronger quantitative guarantee (ii), a uniform comparison inequality. Specifically, for all data distributions $P$:

(i) **Fisher consistency.** For any sequence $f_n\in\mathcal F$,
\[
R_\varphi(f_n)\to R_\varphi^\star
\quad\Longrightarrow\quad
R(f_n)\to R^\star.
\]

(ii) **Uniform comparison inequality.** There is a non-decreasing function $\psi:\mathbb R_+\to\mathbb R_+$ such that
\[
\psi(R(f)-R^\star)
\leq R_\varphi(f)-R_\varphi^\star
\qquad\text{for all }f\in\mathcal F,
\]
and $\psi(\epsilon)>0$ for every $\epsilon>0$.

For binary classification with a convex, margin-based surrogate $\varphi$, the two consistency notions coincide [4]. In general, however, the stronger uniform guarantee (ii) need not hold.

Calibration functions provide a canonical construction based on excess risk. Define the pointwise calibration function and its uniform version by
\[
\psi(\epsilon,x)\coloneqq
\inf_{s\in\mathbb R^d}
\left\{
\delta_\varphi(s,x)
\mid
\delta_\ell(s,x)\geq\epsilon
\right\},
\qquad
\overline\psi(\epsilon)\coloneqq
\inf_{x\in\mathcal X}\psi(\epsilon,x).
\]
These calibration functions yield consistency and comparison inequalities, as follows from Zhang [51, Prop. 25] and Steinwart [40, Thm. 2.8 and Lemma 2.9].

###### Corollary 3.1.

The surrogate $\varphi$ is Fisher consistent (i) for $\ell$ if and only if
\[
\psi(\epsilon,x)>0
\qquad
\text{for all }x\in\mathcal X,\ \epsilon>0.
\]
Let $\psi$ be the Fenchel biconjugate of $\overline\psi$. Then
\[
\overline\psi(\epsilon)>0
\quad\Longleftrightarrow\quad
\psi(\epsilon)>0,
\]
and, for every measurable $f$,
\[
\psi(R(f)-R^\star)
\leq R_\varphi(f)-R_\varphi^\star.
\]

For the general task risk (2), we would like at least a Fisher consistent surrogate for $\ell$. Such consistency would mean that minimizing
\[
R_\varphi(f)=\mathbb E[\varphi(f(X),Y)]
\]
also minimizes $R(f)$.

If this guarantee held using only paired observations $(X,Y)$ rather than tuples $(X,Y_1,\ldots,Y_m)$, we could apply the full theory of empirical processes and related statistical tools [45, 5, 21, 4]. Data collection would also be simpler because consistent estimation would require only single pairs $(X,Y)$. Unfortunately, such guarantees are generally impossible. The extended ranking example below demonstrates this impossibility and motivates the aggregation theory in Section 3.2.

### 3.1 Fisher consistency failure without label aggregation: ranking

Consider ranking $k$ items using pairwise comparison data [20, 15, 14, 30]. The label space contains all ordered pairwise comparisons:
\[
\mathcal Y=\{(i,j):i\neq j,\ 1\leq i,j\leq k\}.
\]
A label $(i,j)$ means that item $i$ has a higher ranking than item $j$.

For each input $x$, the population rank aggregation problem transforms the comparison probabilities
\[
p_{ij}=P(Y=(i,j)\mid x)
\]
into a ranking of the $k$ items. Although many aggregation schemes are possible, we consider a simple comparison-based scheme [20, cf.]. Similar negative results hold for more sophisticated schemes.

Define the normalized transition matrix $C_x\in\mathbb R_+^{k\times k}$ by $(C_x)_{ii}=0$ and
\[
(C_x)_{ij}
=
\frac{p_{ij}}{\sum_{l\neq j}p_{lj}}
\qquad\text{for }i\neq j.
\]
We set $0/0=1/(k-1)$, so $C_x$ is stochastic:
\[
C_x^T\mathbf 1=\mathbf 1.
\]
Here, $T$ denotes transpose, and $\mathbf 1$ is the all-ones vector of the appropriate dimension.

We rank the items using $C_x\mathbf 1\in\mathbb R_+^k$. This vector measures how often each item is preferred to others. One may instead use higher powers $C_x^p\mathbf 1$ or Perron vectors [20]; similar results to those below hold for these choices.

Tacitly incorporating the decoder $\mathsf d$ into the task loss, define
\[
\ell(s,C)\coloneqq
\max_{i<j}
1\!\left\{
(s_i-s_j)(e_i-e_j)^TC\mathbf 1\leq 0,\
(e_i-e_j)^TC\mathbf 1\neq 0
\right\},
\]
where $e_i$ is the $i$-th coordinate vector. Thus, the loss penalizes scores $s$ that mis-order a pair whose entries in $C\mathbf 1$ differ. The population task risk (2) is therefore
\[
R(f)\coloneqq
\mathbb P\!\left(
f(X)\text{ and }C_X\mathbf 1
\text{ order differently}
\right).
\tag{4}
\]

Now consider a surrogate
\[
\varphi:\mathbb R^k\times\mathcal Y\to\mathbb R
\]
that is convex in its score argument. We restrict scores to $s^T\mathbf 1=0$. This minor restriction is familiar from multiclass classification [51, 43] and is natural here because ranking requires only the ordering of the coordinates $s_i$.

Unfortunately, no convex Fisher consistent surrogate exists for task risk (4), as shown in Appendix A.1.

###### Proposition 1.

Consider the ranking problem with task risk (4) over $k\geq 3$ outcomes. If $\varphi:\mathbb R^k\times\mathcal Y\to\mathbb R$ is convex in its first argument, then $\varphi$ is not Fisher consistent.

In contrast, a reasonably straightforward argument yields consistency with aggregation as soon as the number $m$ of collected comparisons satisfies $m\geq k$. The idea is to regress predicted scores $f(x)$ on frequencies of label orderings.

Assume that
\[
Z=(Y_1,\ldots,Y_m)
\]
contains multiple independent pairwise comparisons conditioned on $X$. Let
\[
m_{ij}=\sum_{y\in\mathcal Y^m}1\!\left\{y=(i,j)\right\},
\qquad
m_j=\sum_{i=1}^{k}m_{ij}.
\]
Writing $[k]=\{1,\ldots,k\}$, define the aggregation
\[
A(Z)=
\begin{cases}
\star,
&
\text{if }m_j=0\text{ for some }j\in[k],
\\[3pt]
\left(
\frac{m_{i1}}{m_1}+
\frac{m_{i2}}{m_2}+\cdots+
\frac{m_{ik}}{m_k}
\right)_{i\in[k]},
&
\text{otherwise, i.e., if }m_j>0
\text{ for every }j\in[k].
\end{cases}
\]
The symbol $\star$ is the aggregation output used when a denominator count is zero.

Regressing directly on $A(Z)$ whenever $A(Z)\neq\star$ yields consistency. The following proposition, proved in Appendix A.2, makes this statement precise.

###### Proposition 2.

Define
\[
\varphi(s,q)=\|s-q\|_2^2
\qquad\text{for }s,q\in\mathbb R^k,
\qquad
\varphi(s,\star)=0,
\]
where $\|\cdot\|_2$ denotes the Euclidean norm. If $m\geq k$, then $\varphi$ is Fisher consistent for ranking risk (4).

The distinction between Proposition 1, which rules out consistency, and Proposition 2, which guarantees consistency, is the response space. Aggregating multiple labels expands the response space from $\mathcal Y$ to $\mathbb R^k$.

### 3.2 Label aggregation obtains stronger surrogate consistency

The ranking example suggests that aggregation can improve surrogate consistency. We now ask how this improvement extends beyond ranking. We present two kinds of results: a basic extension of standard surrogate-risk consistency, and a result showing how aggregation can “upgrade” nominally inconsistent losses into consistent losses, as Proposition 2 suggests.

#### 3.2.1 Basic extensions of surrogate consistency

We first generalize the standard calibration functions and thereby extend the classical comparison inequalities in Corollary 3.1.

For an arbitrary aggregation method $A:\mathcal Z\to\mathcal A$, define the conditional aggregated surrogate risk
\[
R_{\varphi,A}(s\mid x)\coloneqq
\mathbb E\!\left[\varphi(s,A(Z))\mid X=x\right].
\]
As in the non-aggregated case, define the pointwise excess surrogate risk
\[
\delta_{\varphi,A}(s,x):=
R_{\varphi,A}(s\mid x)-
\inf_{s\in\mathbb R^d}R_{\varphi,A}(s\mid x).
\]
This excess risk gives the pointwise and uniform calibration functions
\[
\overline\psi_A(\epsilon,x)\coloneqq
\inf_{s\in\mathbb R^d}
\left\{
\delta_{\varphi,A}(s,x)
\mid
\delta_\ell(s,x)\geq\epsilon
\right\},
\qquad
\overline\psi_A(\epsilon)\coloneqq
\inf_{s\in\mathbb R^d}\overline\psi_A(\epsilon,x).
\tag{5}
\]

Under appropriate measurability conditions, a consistency result analogous to Corollary 3.1 follows. We leave these measurability conditions tacit because they are not central to our results. The following proposition is essentially a corollary of Steinwart [40, Thm. 2.8]; Appendix B.1 includes a proof for completeness.

###### Proposition 3.

Assume there is a function $b:\mathcal X\to\mathbb R_+$ satisfying
\[
\int b(x)\,dP(x)<\infty,
\qquad
\delta_\ell(f(x),x)\leq b(x).
\]
The surrogate $\varphi$ is Fisher consistent (i) for task risk (2) if and only if
\[
\overline\psi_A(\epsilon,x)>0
\qquad
\text{for all }x\in\mathcal X,\ \epsilon>0.
\]
Additionally, let
\[
\psi_A=(\overline\psi_A)^{**}
\]
be the Fenchel biconjugate of $\overline\psi_A$, where ${}^{**}$ denotes biconjugation. Writing
\[
R_{\varphi,A}^\star
=\inf_{f\in\mathcal F}R_{\varphi,A}(f),
\]
we have
\[
\psi_A(R(f)-R^\star)
\leq R_{\varphi,A}(f)-R_{\varphi,A}^\star.
\]

Proposition 3 captures the classical consistency guarantees: aggregation does not make those guarantees fall apart. However, Proposition 3 does not itself guarantee improved consistency. We now turn to such improvements.

#### 3.2.2 Identifying surrogates and consistency

Under essentially minimal conditions on the surrogate, a generic aggregation strategy guarantees consistency asymptotically in the number of observed labels $y$. The strategy applies to any task loss seeking to minimize $\ell(f(x),y)$, that is,
\[
R(f)=\mathbb E[\ell(f(X),Y)].
\]

Assume the label space is finite:
\[
\mathop{\rm card}(\mathcal Y)=k<\infty,
\]
where $\mathop{\rm card}$ denotes cardinality.[^1] We impose the following minimal identifiability assumption on the surrogate.

[^1]: The example in Section 3.1 belongs to this regime. In that section, $k$ denotes the ranking length and $\mathop{\rm card}(\mathcal Y)=k(k-1)$. With a slight abuse of notation, henceforth $k$ denotes the cardinality of the label space.

###### Definition 3.1 (Identifying surrogate).

A surrogate $\varphi:\mathbb R^d\times\mathcal A\to\mathbb R$ is **$(\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})$-identifying for $\mathcal Y$**, with
\[
0<\mathsf C_{\varphi,1}
\leq\mathsf C_{\varphi,2}<\infty,
\]
if there are aggregation parameters
\[
\{a_y\}_{y\in\mathcal Y}\subset\mathcal A
\]
and score vectors
\[
\{s_y\}_{y\in\mathcal Y}\subset\mathbb R^d
\]
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

Inequality (6a) says that each class $y$ has an aggregation parameter $a_y$ for which the minimizer of $\varphi(\cdot,a_y)$ identifies $y$. A finite $\mathsf C_{\varphi,2}$ satisfying (6b) exists if $\varphi(\cdot,a)$ has a finite lower bound.

Importantly, Definition 3.1 does not require $\varphi(\cdot,a)$ to be convex. Nor does Definition 3.1 require consistency when $\mathcal A=\mathcal Y$ and $Z=Y$, that is, without aggregation. The definition also does not impose a particular form on the decoder $\mathsf d$, such as requiring the decoder to coincide with the pointwise $\mathop{\rm argmax}$ operator.

**Example 3:**

Consider the binary hinge loss
\[
\varphi(s,a)=\max\{1-sa,0\},
\qquad
\mathcal A=\mathcal Y=\{\pm1\}.
\]
For $y\in\{-1,1\}$, take $a_y=s_y=y$. Then
\[
\varphi(s_1,a_1)=\varphi(s_{-1},a_{-1})=0,
\qquad
\inf_{sa\leq 0}\varphi(s,a)=1.
\]
Thus, the correct-class gap in (6a) is $1$. Also,
\[
\varphi(s_1,a_{-1})=\varphi(s_{-1},a_1)=2,
\]
so the hinge loss is $(1,2)$-identifying. $\Diamond$

Section 3.3 gives further concrete examples when studying consistency amplification. For an identifying surrogate with parameters $\{a_y\}_{y\in\mathcal Y}$, consider the naive aggregation strategy of generalized majority vote:
\[
A_m(y_1,\ldots,y_m)\coloneqq a_{\widehat y},
\qquad
\widehat y=
\mathop{\rm argmin}_{y\in\mathcal Y}
\sum_{i=1}^{m}\ell(y,y_i),
\tag{7}
\]
with ties broken arbitrarily.

For each $x$, let
\[
y^\star(x)\coloneqq
\mathop{\rm argmin}_{y\in\mathcal Y}
\mathbb E[\ell(y,Y)\mid X=x]
\]
be the set of conditionally optimal labels. Because $\mathcal Y$ is finite, whenever the $Y_i$ are i.i.d., there necessarily exists a random $M<\infty$ such that every $m\geq M$ satisfies
\[
\mathop{\rm argmin}_{y\in\mathcal Y}
\left\{\sum_{i=1}^{m}\ell(y,Y_i)\right\}
\subset y^\star(x).
\]
This eventual inclusion suggests that the surrogate $\varphi(\cdot,A_m)$ should become consistent as $m\to\infty$. The following corollary of our later results establishes this asymptotic consistency.

###### Corollary 3.2.

Let $m=m(n)\to\infty$, and let $\varphi$ be identifying in the sense of Definition 3.1. Then
\[
R_{\varphi,A_m}(f_n)-R_{\varphi,A_m}^\star\to 0
\quad\Longrightarrow\quad
R(f_n)-R^\star\to 0.
\]

#### 3.2.3 Identifying surrogates and consistency amplification

When label noise is low, aggregation strategy (7) provides an explicitly improved comparison inequality
\[
\psi(R(f)-R^\star)\leq R_\varphi(f)-R^\star.
\]
The improvement is that $\psi$ is linear over some range of $\epsilon>0$, and linear growth is the strongest possible comparison inequality [34, 32].

Strict comparison inequalities, such as those in Proposition 3, can be too narrow because inconsistent surrogates can still be practically convenient [23, 34, 32]. Following Osokin et al. [34], we therefore introduce $(\xi,\zeta)$ consistency. This notion requires linear growth only when the excess task risk is at least $\xi$, so the surrogate captures a “good enough” risk.

###### Definition 3.2.

The surrogate loss $\varphi$ and aggregator $A$ yield **level-$(\xi,\zeta)$ consistency** if there exists a function $\psi$ such that
\[
\psi(\epsilon)\geq\zeta\epsilon
\qquad\text{for }\epsilon\geq\xi,
\]
and
\[
\psi(R(f)-R^\star)
\leq R_{\varphi,A}(f)-R_{\varphi,A}^\star.
\]

We show that, under minimal assumptions, aggregation (7) achieves level-$(o_m(1),\zeta)$ consistency even for a Fisher inconsistent surrogate. Here, $o_m(1)$ denotes a quantity tending to zero as $m\to\infty$.

To quantify label noise, we adapt the classical Mammen-Tsybakov noise conditions [25]; see also Bartlett et al. [4]. Define
\[
\Delta(x)\coloneqq
\min_{\mathsf d(s)\notin y^\star(x)}
\delta_\ell(s,x).
\tag{8}
\]
Thus, $\Delta(x)$ is the minimal excess conditional risk incurred by an incorrect prediction.

For binary classification with $\mathcal Y=\{\pm1\}$,
\[
\Delta(x)=|2P(Y=1\mid X=x)-1|.
\]
More generally, we expect consistent estimation to become harder as $\Delta(x)$ approaches $0$.

Let $f^\star$ denote an optimal task predictor. The Mammen-Tsybakov conditions can be stated as
\[
\mathbb P\!\left(
\mathsf d\circ f\neq\mathsf d\circ f^\star
\right)
\leq
\mathsf C_{\mathsf{MT}}
\left(R(f)-R^\star\right)^\alpha
\quad
\text{for all measurable }f,
\tag{Nα}
\]
and
\[
\mathbb P(\Delta(X)\leq\epsilon)
\leq
(\mathsf C_{\mathsf{MT}}\epsilon)^\beta
\qquad
\text{for }\epsilon>0.
\tag{Mβ}
\]
Here, $\mathsf C_{\mathsf{MT}}>0$ is a constant that may change, $\alpha\in[0,1]$, and $\beta\in[0,\infty]$. We call $\alpha$ in condition (Nα) the noise exponent.

Both conditions hold trivially with $\alpha=\beta=0$. Moreover, as in binary classification [4, Thm. 3], conditions (Nα) and (Mβ) are equivalent under
\[
\beta=\frac{\alpha}{1-\alpha}.
\]
Appendix B.2 establishes this equivalence.

Mao et al. [29] recently considered the same noise statistic for $H$-consistency in multiclassification, where $H$ denotes a restricted hypothesis class. Their structural assumptions on the surrogate and hypothesis class enable a global convex lower bound. In contrast, our generic setup imposes minimal structural constraints, so we introduce the noise condition number
\[
\kappa(x)\coloneqq
\frac{
\max_{\mathsf d(s)\neq y^\star(x)}
\delta_\ell(s,x)
}{
\min_{\mathsf d(s)\neq y^\star(x)}
\delta_\ell(s,x)
}.
\tag{9}
\]
The condition number connects $\Delta(x)$ to the pointwise excess risk:
\[
\Delta(x)\geq
\frac{\delta_\ell(s,x)}{\kappa(x)}
\qquad
\text{for every }s
\text{ such that }\mathsf d(s)\neq y^\star(x).
\]
This relation permits a more fine-grained analysis. In binary classification, $\kappa(x)=1$ so long as $\mathbb P(\Delta(X)>0)=1$.

Together, $\Delta(x)$ and $\kappa(x)$ allow us to show that generalized majority vote (7), applied to any identifying surrogate from Definition 3.1, achieves level-$(\xi,\zeta)$ consistency. Define the error function
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
Roughly, when $\kappa(x)=t$, the error function captures that majority vote $A_m$ is likely to be correct if $m$ is large enough to make $e_m(t)\ll1$.

The next theorem gives a near-linear calibration function. Appendix B.3 contains the proof.

###### Theorem 1.

Let $\varphi$ be $(\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})$-identifying with parameters $\{a_y\}_{y\in\mathcal Y}$, and let $A_m$ be the majority vote aggregator (7). Assume
\[
0\leq\ell\leq1
\]
and that $P$ satisfies condition (Nα) with noise exponent $\alpha\in[0,1]$.

Then, for any $M>0$ and $f\in\mathcal F$ satisfying
\[
R(f)-R^\star
\geq
2\mathbb P(\kappa(X)>M)
+
\left(
4\mathsf C_{\mathsf{MT}}e_m(M)
\right)^{\frac{1}{1-\alpha}},
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

Equivalently, under Theorem 1's conditions, aggregation gives level-$(\xi,\zeta)$ consistency in Definition 3.2 with
\[
\xi=
2\mathbb P(\kappa(X)>M)
+
\left(
4\mathsf C_{\mathsf{MT}}e_m(M)
\right)^{\frac{1}{1-\alpha}},
\qquad
\zeta=\frac{\mathsf C_{\varphi,1}}{16}.
\]

Theorem 1 also immediately proves the asymptotic consistency in Corollary 3.2. To see why, define
\[
\xi_m\coloneqq
\inf_M
\left\{
2\mathbb P(\kappa(X)>M)
+
\left(
4\mathsf C_{\mathsf{MT}}e_m(M)
\right)^{\frac{1}{1-\alpha}}
\right\}.
\]
Then $\xi_m\to0$ as $m\to\infty$: the first term can be made small because
\[
\mathbb P(\kappa(X)>M)\to0
\qquad\text{as }M\uparrow\infty,
\]
and, for any fixed $M$, the second term becomes small because $e_m(M)\to0$ as $m$ grows. Taking $\alpha=0$ therefore gives Corollary 3.2.

Theorem 1 is a somewhat gross result because Definition 3.1 imposes such weak identifiability conditions. A tighter connection between the task loss $\ell$ and surrogate $\varphi$ could make the naive majority vote (7) more likely to be correct, or at least correct enough for $\varphi$, so we would expect stronger bounds. For example, if $\varphi$ is nearly consistent without aggregation, we might expect smaller inconsistency without aggregation to imply stronger majority-vote bounds as a function of $m$. We do not pursue these details here.

For a more concrete bound, we optimize over $M$ in Theorem 1 using the crude inequality
\[
\kappa(x)\leq\frac{1}{\Delta(x)}.
\]
We take $M=1$ when $\mathop{\rm card}(\mathcal Y)=k=2$ and optimize $M$ when $k\geq3$. This gives a lower bound on the level threshold $\xi_{m,k}$ promised by Theorem 1, setting
\[
\xi_{m,k}\coloneqq
\begin{cases}
\left(
\frac{32\mathsf C_{\mathsf{MT}}^2}{m}
\log\left(
\frac{8(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})}
{\mathsf C_{\varphi,1}}
\right)
\right)^{\frac{1}{2(1-\alpha)}},
&\text{if }k=2,
\\[7pt]
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
The algebraic substitutions and manipulations in Appendix B.4 yield the following corollary.

###### Corollary 3.3.

Under the conditions of Theorem 1, every $f$ satisfying
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

Corollaries 3.2 and 3.3 provide evidence for the robustness of label cleaning: even under minimal surrogate assumptions, aggregation can yield consistency. In Corollary 3.3, as the noise exponent $\alpha$ approaches $1$, the number $m$ of labels needed for the comparison inequality to hold for a fixed score function $f$ decreases.

In particular, if $\alpha=1$, then whenever
\[
m\geq
32\max\{
\mathsf C_{\mathsf{MT}}^2,
\mathsf C_{\mathsf{MT}}^4
\}
\cdot
\log\left(
\frac{4k(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})}
{\mathsf C_{\varphi,1}}
\right)
=O(\log k),
\]
we have $\xi_{m,k}=0$. Here, $O(\log k)$ expresses logarithmic order in $k$. Thus, the bound becomes the uniform comparison inequality (ii) with a linear comparison function.

The learning problem's noise level therefore affects how much aggregation consistency requires: an “easier” problem needs less aggregation to obtain stronger consistency.

### 3.3 Surrogate consistency examples with majority vote

We now give examples, at varying levels of concreteness, that instantiate Theorem 1 and Corollary 3.3. Throughout this section, assume $P$ has noise exponent $\alpha\in[0,1]$. This assumption loses no generality because condition (Nα) always holds with $\alpha=0$. Appendix B.5 contains the proofs of all results in this section.

#### 3.3.1 Binary classification with a nonsmooth surrogate

Consider binary classification with a margin-based surrogate
\[
\varphi(f(x),y)=\phi(yf(x)),
\]
where $\phi$ is convex. Bartlett et al. [4] show that $\varphi$ is consistent if and only if $\phi'(0)<0$.

We use a somewhat trivial example to illustrate aggregation's robustness: even if $\phi$ is inconsistent without aggregation, aggregation can make $\phi$ consistent. One would never use such a surrogate, so the example should be viewed as a thought experiment.

Assume $\phi$ is convex, the subgradient set at zero satisfies
\[
\partial\phi(0)\subset(-\infty,0),
\]
and
\[
\lim_{t\to\infty}\phi(t)=0.
\]

###### Lemma 3.1.

For any $\delta>0$, the surrogate $\varphi$ is $(\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})$-feasible with
\[
\mathsf C_{\varphi,1}
=
\phi(0)-\phi(\delta)>0,
\qquad
\mathsf C_{\varphi,2}
=
\phi(-\delta).
\]

Consequently, Corollary 3.3 applies with $k=2$. Specifically, if $f:\mathcal X\to\mathbb R$ satisfies
\[
R(f)-R^\star
\geq
\left(
\frac{32\mathsf C_{\mathsf{MT}}^2}{m}
\log\left(
\frac{
8(\phi(-\delta)+\phi(0)-\phi(\delta))
}{
\phi(0)-\phi(\delta)
}
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

In general structured prediction [33], the label space $\mathcal Y$ is a typically large “structured” space. An embedding map
\[
v:\mathcal Y\to\mathbb R^d
\]
encodes structural information about each label. Use the decoder
\[
\mathsf d(s)=
\mathop{\rm argmax}_{y\in\mathcal Y}
\langle s,v(y)\rangle,
\]
where $\langle\cdot,\cdot\rangle$ is the inner product.

For a loss $\ell:\mathcal Y\times\mathcal Y\to\mathbb R_+$ satisfying $\ell(y,y)=0$, the maximum-margin surrogate, also called the generalized hinge loss [42, 44, 19], is
\[
\varphi(s,y)=
\max_{\widehat y\in\mathcal Y}
\left(
\ell(\widehat y,y)
+
\langle v(\widehat y)-v(y),s\rangle
\right).
\tag{11}
\]
This surrogate is typically inconsistent, except in certain low-noise cases [34, 32].

Before treating general structured prediction, consider bipartite matching. A bipartite matching is a graph $G=(V,E)$ whose vertices partition into
\[
V=V_1\cup V_2,
\qquad
V_1=\{1,\ldots,N\},
\qquad
V_2=\{N+1,\ldots,2N\}.
\]
The graph has $N$ edges, each connecting exactly one unique node in $V_1$ to one unique node in $V_2$.

Let $\mathcal Y$ be the collection of all bipartite matchings between $V_1$ and $V_2$. Then
\[
k=\mathop{\rm card}(\mathcal Y)=N!.
\]
For a graph $G$, the edge-indexing embedding is
\[
v(G)\coloneqq
\left(
1\!\left\{(u,v)\in E\right\}
\right)_{u\in V_1,\ v\in V_2}
\in\mathbb R^{N^2},
\]
so the score dimension is $d=N^2$.

The task loss counts mistaken edges:
\[
\ell(y_1,y_2)\coloneqq
\frac{1}{2N}
\|v(y_1)-v(y_2)\|_1
=
\frac{1}{2N}
\|v(y_1)-v(y_2)\|_2^2,
\]
where $\|\cdot\|_1$ is the sum of absolute coordinate values.

For this problem, the structured hinge surrogate (11) is identifying.

###### Lemma 3.2.

For bipartite matching on $2N$ vertices, the structured hinge surrogate $\varphi$ in (11) is $(\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})$-identifying in the sense of Definition 3.1, with
\[
\mathsf C_{\varphi,1}=\frac1N,
\qquad
\mathsf C_{\varphi,2}=2.
\]

The important consequence is that, even though $k=N!$, aggregation-based methods using the structured hinge loss can yield consistency once the number $m$ of aggregated labels exceeds $O(N\log N)$.

For one specialization, substitute the constants from Lemma 3.2 into Corollary 3.3 for $k\geq3$. Every measurable $f:\mathcal X\to\mathbb R^d$ satisfying
\[
R(f)-R^\star
\geq
4\cdot
\left(
\frac{32\mathsf C_{\mathsf{MT}}^4}{m}
\log\left(4k(2N+1)\right)
\right)^{\frac{\alpha}{2(1-\alpha^2)}}
\]
then obeys
\[
R(f)-R^\star
\leq
16N
\left(
R_{\varphi,A_n}(f)-R_{\varphi,A_n}^\star
\right).
\]

#### 3.3.3 Structured prediction

Return to the general structured prediction setup introduced at the beginning of Section 3.3.2. Suppose the decoder can pick every class $y\in\mathcal Y$. More precisely, suppose that for each $y$, the set of scores selecting $y$,
\[
\mathcal S(y)\coloneqq
\left\{
s:
\langle v(y),s\rangle
>
\langle v(\widehat y),s\rangle
\text{ for }\widehat y\neq y
\right\},
\]
is nonempty.

For each $y\in\mathcal Y$, define the identifiability gap
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
This gap gives the following identifiability guarantee.

###### Lemma 3.3.

For structured prediction, the max-margin surrogate $\varphi$ in (11) is $(\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})$-identifiable with
\[
\mathsf C_{\varphi,1}
=
\min_{\widehat y\neq y}\ell(\widehat y,y),
\qquad
\mathsf C_{\varphi,2}
=
\max_{y\in\mathcal Y}\tau(y)+1.
\]

In particular, if
\[
v(y)\in\{0,1\}^d,
\qquad
\ell(\widehat y,y)
=
\frac{1}{2d}
\|v(\widehat y)-v(y)\|_1,
\]
then $\tau(y)=1$ for every $y$ and $\mathsf C_{\varphi,2}=2$.

As in the binary matching case, nontrivial consistency guarantees therefore hold once
\[
m\geq\log\mathop{\rm card}(\mathcal Y).
\]
Because $0\leq\ell(\cdot,\cdot)\leq1$, Corollary 3.3 applies. In particular, for every measurable $f:\mathcal X\to\mathbb R^d$,
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
\frac{8}{\min_{\widehat y\neq y}\ell(\widehat y,y)}
\left(
R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star
\right).
\]
Here, $|\mathcal Y|$ also denotes the cardinality of $\mathcal Y$.

## 4 Robustness and consistency for models

Section 3 builds on classical surrogate risk consistency theory, which assumes $\mathcal F$ contains all measurable functions. The results show that aggregation can “upgrade” consistency: even when $\varphi$ is inconsistent for non-aggregated pairs $(X,Y)$, sufficient aggregation can provide level-$(\xi,\zeta)$ consistency in Definition 3.2.

We now take a different view by considering optimization over a restricted, often parametric, hypothesis class $\mathcal F$. Consistency over such a class is no issue for a well-specified model, but assuming well-specification is unrealistically brittle. This motivates the long-standing challenge of quantifying surrogate consistency when $\mathcal F$ contains only a subset of measurable functions [12, 31].

Recent work develops $H$-consistency bounds linking excess target risk to surrogate regret within a restricted hypothesis class $H$, across various setups [2, 3, 26, 28, 29]. Those results are primarily surrogate-centric. Our contribution is orthogonal: rather than hinging on a specific class and surrogate, we study how aggregation improves consistency itself. We show that label aggregation permits consistent estimates in scenarios where consistency might otherwise fail.

### 4.1 Consistency failure for binary classification in finite dimensions

Binary classification illustrates how restricting the hypothesis class substantially changes even a well-understood problem. Let
\[
\mathcal Y=\{\pm1\},
\qquad
\ell(\mathsf d(s),y)=1\!\left\{ys\leq0\right\}.
\]
Consider a margin-based surrogate
\[
\varphi(s,y)=\phi(sy),
\]
where $\phi:\mathbb R\to\mathbb R_+$ is convex. As discussed above, if $\mathcal F$ contains all measurable functions, then $\varphi$ achieves both Fisher consistency (i) and uniform consistency (ii) if and only if $\phi'(0)<0$ [4].

We now restrict $\mathcal F$. Even in this simple setting, classical consistency fails when optimal classifiers lie in $\mathcal F$. Specifically, say that $P$ is **optimally predictable using $\mathcal F$** when a function $f\in\mathcal F$ satisfies
\[
\mathsf{sgn}(f(x))
=
\mathsf{sgn}\left(P(Y=1\mid X=x)-1/2\right).
\tag{12}
\]

Let $\mathcal X=\mathbb R^d$, and take the linear hypothesis class
\[
\mathcal F=
\left\{
f_\theta:
f_\theta(x)=\langle\theta,x\rangle
\right\}_{\theta\in\mathbb R^d}.
\]
If $P$ is optimally predictable using this class, there exists $\theta^\star$ such that
\[
\mathsf{sgn}(\langle\theta^\star,x\rangle)
=
\mathsf{sgn}\left(P(Y=1\mid x)-\frac12\right).
\]
Consequently, $f_{\theta^\star}$ minimizes $R(f)$ over all measurable functions. We then say that $P$ is **optimally predictable along $\theta^\star$**.

One might expect that a margin-based surrogate that is Fisher consistent over all measurable functions would remain consistent in this setting. However, this expectation fails. Indeed, for any nonnegative loss $\phi$, there is a distribution $P$ for which the surrogate minimizer
\[
\theta_\varphi=
\mathop{\rm argmin}_{\theta}
\mathbb E_P[\varphi(f_\theta(X),Y)]
\]
is essentially orthogonal to $\theta^\star$.

###### Proposition 4.

For any $\epsilon>0$ and nonzero $\theta^\star\in\mathbb R^d$, there exists an $(X,Y)$ distribution $P$, optimally predictable along $\theta^\star$, such that every
\[
\theta_\varphi
\in
\mathop{\rm argmin}_{\theta}R_\varphi(f_\theta)
=
\mathop{\rm argmin}_{\theta}
\mathbb E\!\left[\phi(f_\theta(X),Y)\right]
\]
satisfies
\[
R(f_{\theta_\varphi})
>
R(f_{\theta^\star}),
\]
and
\[
|\cos\angle(\theta_\varphi,\theta^\star)|
=
\frac{
|\langle\theta_\varphi,\theta^\star\rangle|
}{
\|\theta_\varphi\|_2\|\theta^\star\|_2
}
\leq\epsilon.
\]
Here, $\angle(\theta_\varphi,\theta^\star)$ is the angle between the two parameter vectors.

Appendix D.1 contains the proof.

Label aggregation provides one way around the inconsistency in Proposition 4. To state the aggregation result, define the set of $\epsilon$-approximate minimizers of a function $g$:
\[
\mathop{\epsilon\textup{-argmin}}g
=
\mathop{\epsilon\textup{-argmin}}_{\theta}g(\theta)
\coloneqq
\left\{
\theta:
g(\theta)\leq
\inf_{\theta}g(\theta)+\epsilon
\right\}.
\]

Suppose data collection provides independent labels
\[
Z=(Y_1,\ldots,Y_m),
\]
and let $A_m(Z)$ be majority vote. For a sequence $\epsilon_m$, take
\[
\theta_m
\in
\mathop{\epsilon_m\textup{-argmin}}_{\theta}
R_{\varphi,A_m}(f_\theta)
=
\mathop{\epsilon_m\textup{-argmin}}_{\theta}
\mathbb E\!\left[
\phi\!\left(
A_m(Y_1,\ldots,Y_m)f_\theta(X)
\right)
\right].
\]
The coming Theorem 2 gives the following corollary: the predictors $f_{\theta_m}$ are asymptotically consistent as $m\to\infty$.

###### Corollary 4.1.

Let $P$ be optimally predictable along $\theta^\star$. If $\epsilon_m\to0$ as $m\to\infty$, then
\[
R(f_{\theta_m})\to R(f_{\theta^\star}),
\qquad
\cos\angle(\theta_m,\theta^\star)\to1.
\]

Thus, Proposition 4 shows that surrogate risk minimization without aggregation can be essentially arbitrarily incorrect over linear predictors. In contrast, Corollary 4.1 shows that aggregation retains consistency.

### 4.2 Aggregation, consistency, and restricted hypothesis classes

Proposition 4 shows that surrogate risk consistency depends fundamentally on $\mathcal F$ containing all measurable functions. We now consider multiclass classification with
\[
\mathcal Y=\{1,\ldots,k\},
\]
and a hypothesis class $\mathcal F$ forming a linear cone with
\[
f(x)^T\mathbf 1=0,
\qquad
tf\in\mathcal F
\quad\text{whenever }t>0,\ f\in\mathcal F.
\]
Thus, the class is closed under positive scaling, and the score vectors are normalized to sum to zero.

Use the zero-one loss and maximum-score decoder:
\[
\ell(y,y')=1\!\left\{y\neq y'\right\},
\qquad
\mathsf d(s)=\mathop{\rm argmax}_{y\in[k]}s_y.
\]
Because this decoder depends only on the largest score, the normalization $f(x)^T\mathbf 1=0$ is immaterial.

Assume the surrogate
\[
\varphi:\mathbb R^k\times[k]\to\mathbb R_+
\]
is Fisher consistent (i) and satisfies the limiting loss condition
\[
\varphi(s,y)\to0
\quad\text{if}\quad
s_y-s_j\to+\infty
\text{ for every }j\neq y.
\tag{13}
\]

Many familiar losses are Fisher consistent and satisfy (13). Examples include multiclass logistic loss,
\[
\varphi(s,y)=
\log\left(\sum_{j=1}^{k}e^{s_j-s_y}\right),
\]
and every loss of the form
\[
\varphi(s,y)=
\sum_{i\neq y}\phi(s_y-s_i),
\]
where $\phi$ is convex and non-increasing, $\phi'(0)<0$, and $\inf_t\phi(t)=0$. Zhang [51, Thm. 8] shows that any such loss is consistent over
\[
\mathcal F=
\left\{
f:\mathcal X\to\mathbb R^k:
\mathbf 1^Tf=0
\right\}.
\]

The margin-based binary setting in Section 4.1 fits this formulation by taking
\[
f(x)=(g(x),-g(x))
\]
for a measurable function $g$. For a parametric example with $\mathcal X=\mathbb R^d$, the class of linear functions
\[
f(x)=
(\langle\theta_1,x\rangle,\ldots,
\langle\theta_k,x\rangle),
\qquad
\sum_{i=1}^{k}\theta_i=0,
\]
is a convex cone. However, consistency without aggregation in these examples typically requires strong structural assumptions such as convexity. In contrast, aggregation will give asymptotic consistency under the minimal limiting loss condition (13).

Extending optimal predictability (12), say that **$\mathcal F$ can optimally predict $P$** if some $f\in\mathcal F$, with $f:\mathcal X\to\mathbb R^k$, satisfies
\[
\mathop{\rm argmax}_{y}f_y(x)
\in
\mathop{\rm argmax}_{y}P(Y=y\mid X=x)
\qquad\text{for every }x.
\]

Suppose $\mathcal Z=\mathcal Y^m$, we aggregate by majority vote $A_m$, and the most probable label
\[
y^\star(x)=
\mathop{\rm argmax}_{y}P(Y=y\mid x)
\]
is unique. The next theorem shows that surrogate risk minimization is then consistent whenever $\mathcal F$ can optimally predict $P$.

###### Theorem 2.

Let $\mathcal F$ be a cone that optimally predicts $P$, and assume the minimal excess risk (8) satisfies
\[
P(\Delta(X)>0)=1.
\]
Let $\epsilon_m\geq0$ with $\epsilon_m\to0$. Then every sequence
\[
f_m
\in
\mathop{\epsilon_m\textup{-argmin}}_{f\in\mathcal F}
R_{\varphi,A_m}(f)
=
\mathop{\epsilon_m\textup{-argmin}}_{f\in\mathcal F}
\mathbb E\!\left[
\varphi(f(X),A_m(Y_1,\ldots,Y_m))
\right]
\]
satisfies
\[
R(f_m)\to R^\star.
\]

Appendix D.2 contains the proof.

Thus, Theorem 2 gives asymptotic consistency as $m\to\infty$ for a broad class of surrogate problems whose hypothesis class forms a linear cone. In contrast, Proposition 4 shows that, even in binary classification, consistency can fail over a subclass containing the optimal predictor, and the surrogate can be arbitrarily uninformative.

### 4.3 On finite-dimensional multiclass classification

Our final technical results concern multiclass classification with
\[
\mathcal X=\mathbb R^d,
\qquad
\mathcal Y=[k],
\]
using linear predictors whose predictive model may be mis-specified. These results give a more nuanced and specialized view than Proposition 4's negative result and Theorem 2's consistency guarantee.

We show that even when the problem is optimally predictable and the linear model is well-specified except on an $\epsilon$-fraction of the data, surrogate risks based only on pairs $(X,Y)$ are inconsistent. In contrast, majority-vote methods recover the optimal linear predictor.

Let the parameter matrix be
\[
\Theta=
[\theta_1\ \cdots\ \theta_{k-1}]
\in\mathbb R^{d\times(k-1)}.
\]
Conditional on $X=x$, labels follow the categorical distribution
\[
Y\mid X=x\sim\mathsf{Cat}(p_\Theta(x)),
\]
where $\mathsf{Cat}$ denotes a categorical distribution and
\[
p_\Theta(x)\in\mathbb R_+^k,
\qquad
\mathbf 1^\top p_\Theta(x)=1.
\]
The probability vector is
\[
p_\Theta(x)=
\begin{bmatrix}
\sigma(
\langle\theta_1,x\rangle,\ldots,
\langle\theta_{k-1},x\rangle)
\\
1-\mathbf 1^\top
\sigma(
\langle\theta_1,x\rangle,\ldots,
\langle\theta_{k-1},x\rangle)
\end{bmatrix}.
\]
Here, the link function
\[
\sigma:\mathbb R^{k-1}\to\mathbb R_+^{k-1}
\]
satisfies $\mathbf 1^\top\sigma\leq1$.

Assume the link preserves the largest score. Specifically, for $t\in\mathbb R^{k-1}$, define
\[
t_k=0,
\qquad
\sigma_k(t)=1-\mathbf 1^\top\sigma(t),
\]
and require
\[
t_y=\max_{1\leq i\leq k}t_i
\quad\Longleftrightarrow\quad
\sigma_y(t)=\max_{1\leq i\leq k}\sigma_i(t).
\tag{14}
\]
We call (14) the link consistency condition.

A standard example is the multiclass logistic regression link, denoted by the superscript $\mathrm{lr}$:
\[
\sigma_i^{\mathrm{lr}}(t)
=
\frac{e^{t_i}}
{1+e^{t_1}+\cdots+e^{t_{k-1}}}.
\]

Let the multilabeled dataset be
\[
\{(X_i;(Y_{1i},\ldots,Y_{im}))\}_{i=1}^{n},
\]
with repeated sampling
\[
Y_{ij}\mid X_i
\stackrel{\mathrm{iid}}{\sim}
\mathsf{Cat}(p_{\Theta^\star}(X_i)).
\]
Define the majority-vote label, with ties broken arbitrarily, by
\[
Y_i^+=A_m(Y_{i1},\ldots,Y_{im}).
\]
Then
\[
Y_i^+\mid X_i
\sim
\mathsf{Cat}(p_{\Theta^\star,m}(X_i)).
\]

To describe the aggregated probabilities, let $\rho_m(t)$ denote the distribution of majority vote on $m$ items with initial probabilities $\sigma(t)\in\mathbb R_+^{k-1}$. Then
\[
p_{\Theta,m}(x)=
\begin{bmatrix}
\rho_m(
\langle\theta_1,x\rangle,\ldots,
\langle\theta_{k-1},x\rangle)
\\
1-\mathbf 1^\top
\rho_m(
\langle\theta_1,x\rangle,\ldots,
\langle\theta_{k-1},x\rangle)
\end{bmatrix}.
\]
It is evident that $\rho_m$ satisfies link consistency (14).

Consider fitting logistic regression with loss
\[
\varphi(\Theta^\top x,y)
=
-\langle\theta_y,x\rangle
+
\log\left(
1+\sum_{i=1}^{k-1}
\exp(\langle\theta_i,x\rangle)
\right),
\tag{15}
\]
using the convention $\theta_k=\mathbf 0$, where $\mathbf 0$ is the zero vector. Define the population logistic loss with $m$-majority vote by
\[
L_m(\Theta)
=
\mathbb E[
\varphi(\Theta^\top X,Y_m^+)
].
\]

The gradient of the individual loss is
\[
\nabla_{\theta_i}\varphi(\Theta^\top x,y)
=
-x\left(
1\!\left\{y=i\right\}
-
\sigma_i^{\mathrm{lr}}(\Theta^\top x)
\right).
\]
Consequently, a minimizer $\Theta_m$ of $L_m$ satisfies
\[
\nabla_\Theta L_m(\Theta_m)
=
\mathbb E\!\left[
X\left(
\sigma^{\mathrm{lr}}(\Theta_m^\top X)
-
\rho_m({\Theta^\star}^\top X)
\right)^\top
\right]
=0.
\]

Standard statistical results guarantee both consistency and efficiency when the model is well-specified without aggregation. Moreover, for $m\geq1$ and $k=2$, Cheng et al. [7, Prop. 3] show that if
\[
X\sim\mathsf N(0,I_d),
\]
then some $t_m>0$ satisfies
\[
\Theta_m=t_m\Theta^\star
\]
even when the link is mis-specified. Here, $\mathsf N(0,I_d)$ is the normal distribution with zero mean and identity covariance matrix $I_d$.

Thus, in binary classification, an incorrect link can still yield consistent classification at every aggregation level $m$, because consistency depends on the direction
\[
\Theta^\star/\|\Theta^\star\|_2.
\]
However, once $k\geq3$, even slight link mis-specification causes risk consistency to fail.

To construct such a mis-specification, fix a set
\[
\mathcal T_\epsilon\subset\mathbb R^{k-1}
\]
with Lebesgue measure $\epsilon$, and define
\[
\sigma^\epsilon(t)
=
\sigma^{\mathrm{lr}}(t)
1\{t\notin\mathcal T_\epsilon\}
+
\frac1k\mathbf 1\,
1\!\left\{t\in\mathcal T_\epsilon\right\}.
\]
This defines a distribution on $Y\in\{1,\ldots,k\}$ conditional on $t\in\mathbb R^{k-1}$ that samples
\[
Y\sim\sigma^{\mathrm{lr}}(t)
\quad\text{if }t\in\mathcal T_\epsilon,
\]
and samples uniformly otherwise.

Clearly, $\sigma^\epsilon$ satisfies link consistency (14) and is optimally predictable in the sense of (12). For $\epsilon>0$, define
\[
L_{m,\epsilon}(\Theta)\coloneqq
\mathbb E\!\left[
\mathbb E_{\sigma^\epsilon}
[
\varphi(\Theta^\top X,Y_m^+)
\mid X
]
\right].
\]
This is the population logistic loss based on $m$-majority vote when
\[
Y\mid X=x
\sim
\sigma^\epsilon({\Theta^\star}^\top x).
\]
Let
\[
\Theta_m(\epsilon)
=
\mathop{\rm argmin}_{\Theta}L_{m,\epsilon}(\Theta).
\]
Although $\Theta_1(0)=\Theta^\star$, the next proposition shows failure of consistency without aggregation for arbitrarily small $\epsilon>0$.

###### Proposition 5.

Let $k\geq3$ and $\Sigma=I$, where $\Sigma$ is the covariance matrix and $I$ is the identity matrix. For
\[
Z\sim\mathsf N(0,I_{k-1}),
\]
assume the linear mapping
\[
M\mapsto DM
\coloneqq
\mathbb E\!\left[
Z\left(
\nabla\sigma^{\mathrm{lr}}({\Theta^\star}^\top Z)
MZ
\right)^\top
\right]
\]
is invertible. Here, $\nabla\sigma^{\mathrm{lr}}$ denotes the derivative matrix of the logistic link, and $D$ denotes the displayed linear mapping.

Then there exists $\epsilon_0>0$ such that, for every $\epsilon\in(0,\epsilon_0)$, there is a set $\mathcal T_\epsilon$ of Lebesgue measure at most $\epsilon$ for which
\[
\Theta_1(\epsilon)/\|\Theta_1(\epsilon)\|
\neq
\Theta^\star/\|\Theta^\star\|.
\]

Appendix E.1 contains the proof.

In contrast, majority vote addresses this inconsistency as $m\to\infty$ without assumptions on the true link $\sigma$ beyond link consistency (14). Define
\[
L_{m,\sigma}(\Theta)
=
\mathbb E\!\left[
\mathbb E_\sigma[
\varphi(\Theta^\top X,Y_m^+)
\mid X
]
\right],
\qquad
\Theta_m=\mathop{\rm argmin}L_{m,\sigma}(\Theta).
\]
The following theorem describes the aggregated minimizers.

###### Theorem 3.

Let $\Theta^\star$ have the decomposition
\[
\Theta^\star=U^\star T^\star,
\]
where
\[
U^\star\in\mathbb R^{d\times(k-1)}
\]
is orthogonal and
\[
T^\star\in\mathbb R^{(k-1)\times(k-1)}
\]
is nonsingular. Then, for every $m$, there exists
\[
T_m\in\mathbb R^{(k-1)\times(k-1)}
\]
such that
\[
\Theta_m=U^\star T_m.
\]
As $m\to\infty$,
\[
\|T_m\|\to\infty,
\qquad
T_m/\|T_m\|
\to
T^\star/\|T^\star\|.
\]
The notation $\|\cdot\|$ denotes the matrix norm used in these normalization statements.

Appendix E.2 contains the proof.

Theorem 3 gives further evidence for aggregation's robustness: aggregation provides asymptotic consistency even in mis-specified models, so long as a link function describes the relationship between $X$ and $Y$. The contrast is striking for $k\geq3$, because Proposition 5 shows that methods without aggregation are generally inconsistent.

## 5 Numerical illustrations

We provide two experiments that corroborate the theory: (i) a non-parametric example where aggregation amplifies surrogate consistency, and (ii) a finite-dimensional multiclass example where aggregation recovers asymptotic consistency under link mis-specification.

##### Surrogate consistency amplification for truncated quadratic loss.

For binary classification, consider the margin-based surrogate
\[
\varphi(f(x),y)=\phi(yf(x)),
\qquad
\phi(\delta)=\max\{1-\delta,0\}^2.
\]
This $\phi$ is the truncated quadratic loss. Bartlett et al. [4, Example 2] show that the calibration function in Corollary 3.1 is exactly
\[
\psi(\epsilon)=\epsilon^2.
\]

For majority-vote aggregation using $m$ labels, apply Lemma 3.1 with $\delta=1$. With no low-noise condition, that is, with $\alpha=0$ and $\mathsf C_{\mathsf{MT}}=1$, take
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
\frac{
8(\phi(-\delta)+\phi(0)-\phi(\delta))
}{
\phi(0)-\phi(\delta)
}
\right)
}
\right)_+
\\
&=
\frac1{16}
\left(
\epsilon-
\sqrt{\frac{32}{m}\log40}
\right)_+.
\end{aligned}
\]
Here, $(u)_+=\max\{u,0\}$ denotes the positive part.

We then have parallel comparison inequalities for non-aggregated and majority-vote data:
\[
\psi(R(f)-R^\star)
\leq R_\varphi(f)-R_\varphi^\star,
\]
\[
\psi_m(R(f)-R^\star)
\leq R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star.
\]

Numerically, Figure 1 shows the aggregated curve transitioning from quadratic toward linear behavior as $m\to\infty$, thereby illustrating the upgrade in surrogate consistency. However, the functions $\psi_m$ are generally conservative because they come from the general bound in Theorem 1.

Figure 1: Numerical illustrations for truncated quadratic surrogate $\phi(\delta)=\max\{1-\delta,0\}^{2}$ , showing comparison inequalities for non-aggregated data $\psi$ (optimal) vs. majority vote aggregation $\psi_{m}$ (suboptimal) when $m\in\{2^{16},2^{20},2^{24},2^{28},2^{32}\}$ .

##### Asymptotic consistency for mis-specified logistic model.

For the second experiment, consider the logistic regression setting in Section 4.3 with $k=3$ and $d=2$. Use the logistic loss $\varphi$ in (15) and the corrupted logistic link
\[
\widetilde\sigma(t)
=
\sigma^{\mathrm{lr}}(t)
1\{\|t\|\leq r\}
+
\sigma^{\mathrm{lr}}(\alpha t)
1\{\|t\|>r\}.
\]
Here, $r$ is the radius separating the two pieces of the link, and $\alpha$ in this experiment is the score-shrinkage parameter. The piecewise link matches softmax near the origin and shrinks scores outside radius $r$.

Figure 2 reports the numerical results using a synthetic dataset from the corrupted link. These results show asymptotic consistency as $m\to\infty$.

| (i) Experiment with constant dataset size $n$ | (ii) Experiment with constant label size $nm$ . |
|---|---|

Figure 2: Numerical illustrations for mis-specified logistic models when $d=2$ and $k=3$ . The true parameters are $\theta_{1}^{\star}=(0,1)^{\top},\theta_{2}^{\star}=(1,1)^{\top}$ . The corrupted link parameters are $r=3$ and $\alpha=0.1$ . The plot shows the error of classification rule $\left\|\|T_{m}\|-T^{\star}/\|T^{\star}\|\right\|$ vs. number of labels $m\in\{1,4,16,64,256,1024\}$ . (i) In the left panel, for each value of $m$ , the synthetic dataset consists of $n=10{,}000$ multilabeled observations $(X_{i},(Y_{i1},\ldots,Y_{im}))$ sampled from $\tilde{\sigma}$ . (ii) In the right panel, for each $m$ , we set $n=640{,}000/m$ so that the total number of labels remains constant across different values of $m$ . We report a $95\%$ error bar on $T=100$ trials for each $m$ .

## 6 Discussion and future work

Whether and how to clean data has animated much of the research discussion around dataset collection. Cheng, Asi and Duchi [7] discuss an apparent phenomenon: using non-aggregated data, including all available labels, improves statistical efficiency when models can fully represent uncertainty, whereas data cleaning appears more robust otherwise. In a related validation setting, Dorner and Hardt [11] argue that comparing binary classifiers is better done with more noisy labels rather than cleaned variants.

This paper contributes to that dialogue by providing evidence both for fundamental limits of un-cleaned, un-aggregated label information in supervised learning and for robustness improvements from label cleaning. Nonetheless, many questions remain.

##### Finite $m$ results and fundamental limits.

Many of our consistency results rely on taking the number $m$ of aggregated labels to infinity. Although the paper's purpose is partly to highlight ways aggregation improves robustness, relying on this asymptotic setting may be unsatisfying.

For ranking in Section 3.1, we give explicit consistency guarantees at finite $m$. Developing finite-$m$ results further is one of the most natural and, we believe, important directions for future work.

A surrogate consistency theory depending jointly on the loss pair $(\ell,\varphi)$ and available label count $m$ would be interesting. For example, in Section 3.1's ranking problem, do second- or third-order comparisons, such as the powers $C_x^p$ suggested by Keener [20], require increasing label counts $m$? Precisely separating problems that require label cleaning and aggregation from problems that do not is a central challenge.

##### Fundamental limits of the noise condition number.

Our work uses the noise condition number $\kappa(X)$ in (9) to characterize multiclassification comparison inequalities. This reliance hints at the difficulty beyond binary classification, where $\kappa(X)=1$ trivially.

Beyond binary classification, $\kappa(X)$ can be large even when the Mammen-Tsybakov noise level in condition (Nα) is low, meaning $\alpha\approx1$. This difficulty follows from our minimal surrogate assumptions. It would therefore be useful to identify connections between $\ell$ and $\varphi$ that capture problem difficulty more closely. Constructing explicit failure modes to delineate fundamental limits more precisely would also provide insight into fitting predictive models.

##### Behavior in mis-specified models.

Our mis-specified-model results, especially those in Section 4.3, require optimal predictability (12): a Bayes-optimal classifier, meaning a classifier attaining the optimal task risk, must lie in $\mathcal F$.

Classical surrogate consistency results provably fail even under this requirement, whereas aggregated-label methods can succeed. Nevertheless, moving beyond optimal predictability is a fruitful direction.

Nguyen et al. [31], followed by Duchi et al. [12], identify one possible direction. For binary and multiclass classification, respectively, they show that jointly inferring a predictor $f$ and a data representation for $x$ requires surrogates $\varphi$ of a particular form depending on the task loss $\ell$. However, those results still rely on infinitely powerful decision rules $f$, so new approaches are needed.

##### Restricted class comparison inequality with label aggregation.

Our paper studies how aggregation mitigates surrogate-induced regret through comparison inequalities in the non-parametric setting, and establishes statistical consistency in the parametric setting. Recent developments in $H$-consistency [2, 28, 27, 29] motivate extending these techniques to comparison inequalities under restricted hypothesis classes.

In particular, Mao et al. [27] show that, for smooth convex loss, quadratic growth of $\psi$ is rate optimal. It would therefore be interesting to apply aggregation to upgrade consistency in the context of $\mathcal H$-consistency, where $\mathcal H$ denotes the hypothesis class.

Such an extension could unify the advantages of aggregation across parametric and non-parametric settings. The extension could also clarify how the “size” of the hypothesis class $\mathcal F$ affects aggregation's effectiveness.
