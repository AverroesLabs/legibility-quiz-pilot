# Some Robustness Properties of Label Cleaning

Chen Cheng

John Duchi

###### Abstract

Aggregated labels combine information from responses, including noisy ones. They provide robustness that is not available without label cleaning. Minimizing a surrogate loss, usually a convex one, gives stronger guarantees of consistency for the task risk than methods that use raw labels. Here, consistency means convergence to the best task risk. Finite-dimensional classifiers also remain consistent when the loss is mis-specified in ways that defeat standard methods for modeling uncertainty.

When a statistical model is exactly correct, using all available information improves efficiency. Yet even slight loss mis-specification destroys standard consistency. Aggregation preserves convergence to optimal classifiers. Refining noisy signals can therefore make data collection, model fitting, and prediction jointly robust.

## 1 Introduction

Aggregation can make optimal prediction consistent when a method minimizes a surrogate loss rather than the task loss. Estimation without aggregation cannot always do this, even in finite-dimensional statistical problems. We construct linear estimators for ranking and for binary and multiclass classification. These constructions separate methods that use only $(X,Y)$ pairs from methods that aggregate labels.

Our nonparametric results connect the number of aggregated labels with noise conditions. Under weak conditions on the surrogate, they give approximately linear comparison inequalities, which relate excess task risk to excess surrogate risk. They also extend consistency beyond classes containing all measurable prediction functions.

The conventional supervised-learning pipeline collects $(X_i,Y_i)_{i=1}^n$ and fits a model [17]. This account leaves out the cleaning, filtering, repeated crowdsourced labeling, and denoising that modern datasets require [9, 22, 39]. Research on crowdsourced “gold standard” labels [8, 48, 47, 46, 37] studies this machinery. We instead ask how it fundamentally and necessarily improves the consistency of fitted models.

Important work on surrogate-risk consistency began in the 2000s [52, 24, 40]. Bartlett, Jordan and McAuliffe [4] characterize consistency for convex surrogates in binary zero-one classification. Later work covers multi-label classification [50, 43, 16, 49, 1], ranking [13, 14, 36], structured prediction [34, 6, 32], ordinal regression [35], restricted hypothesis classes [2, 3, 28, 29], and general theory [40].

Analyses based on individual pairs support empirical-process theory, which studies sample-based estimates of population quantities [45, 5, 21, 4]. They do not cover the aggregation machinery used to create contemporary datasets. We ask whether aggregation is needed, whether pair-based surrogate consistency is enough, and in what sense aggregation improves consistency.

## 2 Preliminaries

We study two settings. The first uses an unrestricted Borel class $\mathcal F$: all score functions measurable with respect to the Borel sets. It uses the classical notion of Fisher consistency, denoted Q7D. This notion connects optimal surrogate risk to optimal task risk. The second uses a parametric, possibly mis-specified class $\mathcal F$. Aggregation improves consistency in both settings.

Let $(X_i,Y_i)_{i=1}^n\sim P$ be independent and identically distributed observations on the input and output spaces $\mathcal X\times\mathcal Y$. A score function $f:\mathcal X\to\mathbb R^d$, with $d\geq1$, assigns a vector of scores to each input. A decoder $\mathsf d:\mathbb R^d\to\mathcal Y$ converts scores to a prediction:
$\widehat y=\mathsf d\circ f(x)$.
The circle denotes function composition. The distribution-dependent task criterion is

\[
R(f)\coloneqq
\mathbb E_P\!\left[\ell(\mathsf d\circ f(X),P(\cdot\mid X))\right].
\tag{2}
\]

Here $\mathbb E_P$ denotes expectation under $P$, and $P(\cdot\mid X)$ is the conditional output distribution given the input. The symbol $\coloneqq$ defines a quantity. The minimizers of this criterion often coincide with those of the original problem (2). Allowing the loss to depend on the conditional distribution supports more sophisticated criteria.

For $\mathcal Y=\{1,\ldots,k\}$ and
\[
\ell(\widehat y,P)=\sum_y P(Y=y)1\{\widehat y\neq y\},
\]
the criterion is the standard zero-one error rate. The indicator $1\{\cdot\}$ equals one when its condition holds and zero otherwise.

The instantaneous formulation uses a loss
$\ell:\mathcal Y\times\mathcal Y\to\mathbb R_+$, where $\mathbb R_+$ denotes the nonnegative real numbers. It minimizes, over $f\in\mathcal F$,
\[
R(f):=\mathbb E_P[\ell(\mathsf d\circ f(X),Y)].
\tag{1}
\]
For binary classification, $d=1$,
$\mathsf d(s)=\mathsf{sgn}(s)$, and
$\ell(y,y')=1\{yy'\leq0\}$.
The sign function $\mathsf{sgn}$ converts a score to its sign. Thus
$R(f)=\mathbb P(Yf(X)\leq0)$, where $\mathbb P$ denotes probability.

Task losses can be nonsmooth, nonconvex, and uninformative for optimization. The zero-one loss is flat almost everywhere, which prevents practical first-order optimization based on derivatives. We replace it with an easier surrogate loss
$\varphi:\mathbb R^d\times\mathcal Y\to\mathbb R$:
\[
R_\varphi(f):=\mathbb E_P[\varphi(f(X),Y)].
\]
This replacement must remain consistent with the task criterion.

### 2.1 Label aggregation

We assess minimization of
\[
R_{\varphi,A}(f):=\mathbb E[\varphi(f(X),A(Z))]
\tag{3}
\]
against minimization of the actual task risk. Here $Z\in\mathcal Z$ replaces the ordinary response in an $(X_i,Y_i)$ pair. The aggregation map $A:\mathcal Z\to\mathcal A$ takes values in the aggregation space $\mathcal A$. Together with the nonnegative surrogate
$\varphi:\mathbb R^d\times\mathcal A\to\mathbb R_+$, these objects define the **aggregated surrogate risk**, denoted P9S.

Repeated labeling uses $Z=(Y_1,\ldots,Y_m)\in\mathcal Y^m$, where
$Y_i\mid X=x\stackrel{\rm iid}{\sim}P_{Y\mid X=x}$.
Thus the labels are independent and identically distributed conditional on the input. Empirical task minimization gives
\[
A(Z)=A(\{Y_1,\ldots,Y_m\})
=\mathop{\rm argmin}_{y\in\mathcal Y}
\sum_{l=1}^m\ell(y,Y_l).
\]
The notation $\operatorname{argmin}$ selects a minimizing argument. For
$\ell(\widehat y,y)=1\{y\neq\widehat y\}$, this rule is exactly **majority vote**, denoted T4K. General losses $\ell$ allow more abstract aggregation rules. We write $A_m(Z)$ to record the number of repetitions. Mao et al. [28] use a similar multiple-label structure for multilabel losses, rather than for an aggregation pipeline.

A rough $K$-nearest-neighbor construction instead pairs $X$ with $Y$ and an additional independent sample
$(X_i,Y_i)_{i=1}^m\stackrel{\rm iid}{\sim}P$.
Then $Z=(Y,(X_i,Y_i)_{i=1}^m)$. A metric
$\operatorname{dist}:\mathcal X\times\mathcal X\to\mathbb R_+$
measures distance between inputs. Order the sample inputs by distance:
\[
\operatorname{dist}(X,X_{(1)})\leq\cdots\leq
\operatorname{dist}(X,X_{(m)}),\qquad X_{(0)}=X.
\]
For $K\geq0$, take
\[
A_{m,K}(Z)\coloneqq
\mathop{\rm argmin}_{y\in\mathcal Y}
\sum_{l=0}^K\ell(y,Y_{(l)}).
\]
The population-level treatment based on repeated sampling also extends to aggregation from one sample $(X_i,Y_i)_{i=1}^n$; see Appendix C.

## 3 Surrogate consistency

###### Corollary 3.1.

The condition $\psi(\epsilon,x)>0$ for every $x\in\mathcal X$ and $\epsilon>0$ is equivalent to $\varphi$ being Fisher consistent, or Q7D, for $\ell$. If $\psi$ is the Fenchel biconjugate of $\overline\psi$, its closed convex lower envelope, then
\[
\overline\psi(\epsilon)>0\quad\Longleftrightarrow\quad
\psi(\epsilon)>0,
\qquad
\psi(R(f)-R^\star)\leq R_\varphi(f)-R_\varphi^\star
\]
for every measurable $f$.

The standard framework [40] lets $\mathcal F$ contain all Borel measurable functions $f:\mathcal X\to\mathbb R^d$. It considers two requirements for all distributions $P$ [40, 4, 52]:

(i) Every sequence $f_n\in\mathcal F$ with
$R_\varphi(f_n)\to R_\varphi^\star$ satisfies
$R(f_n)\to R^\star$.

(ii) A nondecreasing function $\psi:\mathbb R_+\to\mathbb R_+$, positive at every $\epsilon>0$, satisfies
\[
\psi(R(f)-R^\star)\leq R_\varphi(f)-R_\varphi^\star
\quad\text{for every }f\in\mathcal F.
\]

The second requirement is a **uniform comparison inequality**, denoted R2V. It quantitatively bounds excess task risk through excess surrogate risk. It is stronger than (i) and need not hold. For convex margin-based binary surrogates, the two requirements coincide [4].

For a score vector $s\in\mathbb R^d$, define the conditional risks, their excesses above the best conditional risk, and the best overall risks by
\[
\begin{aligned}
R(s\mid x)&\coloneqq
\ell(\mathsf d\circ s,P(Y\in\cdot\mid X=x)),\\
R_\varphi(s\mid x)&:=\mathbb E[\varphi(s,Y)\mid X=x],\\
\delta_\ell(s,x)&\coloneqq
R(s\mid x)-\inf_{s'\in\mathbb R^d}R(s'\mid x),\\
\delta_\varphi(s,x)&\coloneqq
R_\varphi(s\mid x)-\inf_{s'\in\mathbb R^d}R_\varphi(s'\mid x),\\
R^\star&:=\inf_{f\in\mathcal F}R(f),\qquad
R_\varphi^\star:=\inf_{f\in\mathcal F}R_\varphi(f).
\end{aligned}
\]
The infimum, written $\inf$, is the greatest lower bound and need not be attained.

The standard construction uses a **calibration function**, denoted H8N. This function measures the least excess surrogate risk required by a specified excess task risk. Its pointwise and uniform forms are
\[
\psi(\epsilon,x)\coloneqq
\inf_{s\in\mathbb R^d}
\{\delta_\varphi(s,x)\mid\delta_\ell(s,x)\geq\epsilon\},
\qquad
\overline\psi(\epsilon)\coloneqq
\inf_{x\in\mathcal X}\psi(\epsilon,x).
\]
The equivalence and inequality in Corollary 3.1 follow from Zhang [51, Prop. 25] and Steinwart [40, Thm. 2.8 and Lemma 2.9].

A Fisher-consistent surrogate for the general task would make individual pairs sufficient for consistent estimation. It would simplify collection and allow the full empirical-process toolkit. In general, however, such pair-only results are impossible.

### 3.1 Q7D failure without aggregation: ranking

###### Proposition 1.

For ranking with $k\geq3$ outcomes, every surrogate
$\varphi:\mathbb R^k\times\mathcal Y\to\mathbb R$
that is convex in its first argument fails Fisher consistency, or Q7D. See Appendix A.1.

###### Proposition 2.

For $s,q\in\mathbb R^k$, set
\[
\varphi(s,q)=\|s-q\|_2^2,\qquad \varphi(s,\star)=0.
\]
Here $\|\cdot\|_2$ is the Euclidean norm, and $\star$ is the special aggregation output defined below. With the aggregation rule below and $m\geq k$, $\varphi$ is Fisher consistent for the ranking risk. Appendix A.2 gives the proof.

Expanding the response space from $\mathcal Y$ to $\mathbb R^k$ separates these conclusions. The convex formulation based on a single comparison cannot make this expansion. Regression on ordering frequencies gives the positive result.

Pairwise-comparison ranking [20, 15, 14, 30] uses
\[
\mathcal Y=\{(i,j):i\neq j,\ 1\leq i,j\leq k\},
\]
where $(i,j)$ means that item $i$ outranks item $j$. For each input $x$, population aggregation converts the probabilities
$p_{ij}=P(Y=(i,j)\mid x)$ into an ordering. A simple comparison scheme [20, cf.] uses
\[
(C_x)_{ii}=0,\qquad
(C_x)_{ij}=\frac{p_{ij}}{\sum_{l\neq j}p_{lj}}
\quad(i\neq j),\qquad
0/0=1/(k-1).
\]
Thus $C_x\in\mathbb R_+^{k\times k}$ is stochastic:
$C_x^T\mathbf1=\mathbf1$.
Here $T$ denotes transpose, and $\mathbf1$ is the vector of ones. The vector $C_x\mathbf1$ measures each item's preference frequency. Higher powers $C_x^p\mathbf1$, Perron vectors, which are leading eigenvectors [20], and more sophisticated schemes admit similar negative results.

We include decoding in the loss:
\[
\ell(s,C)\coloneqq
\max_{i<j}1\!\left\{
(s_i-s_j)(e_i-e_j)^TC\mathbf1\leq0,\ 
(e_i-e_j)^TC\mathbf1\neq0
\right\}.
\]
The vector $e_i$ is the $i$th coordinate vector. This loss penalizes disagreement between the score ordering and the matrix ordering. Its risk is
\[
R(f)\coloneqq
\mathbb P\!\left(f(X)\text{ and }C_X\mathbf1
\text{ order differently}\right).
\tag{4}
\]
Restricting scores to $s^T\mathbf1=0$ is a minor constraint familiar from multiclass classification [51, 43]. It is natural when decoding uses only the ordering.

For multiple comparisons
$Z=(Y_1,\ldots,Y_m)$ that are independent conditional on $X$, put
\[
m_{ij}=\sum_{y\in\mathcal Y^m}1\{y=(i,j)\},
\qquad m_j=\sum_{i=1}^k m_{ij},
\]
and use
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
The notation $[k]$ means $\{1,\ldots,k\}$. Regression acts directly on outputs other than $\star$.

### 3.2 Aggregation and stronger surrogate consistency

Aggregation extends ordinary surrogate-risk theory. It also improves losses that are nominally inconsistent.

#### 3.2.1 Basic extensions

###### Proposition 3.

Suppose an integrable function $b:\mathcal X\to\mathbb R_+$ satisfies
\[
\int b(x)\,dP(x)<\infty,\qquad
\delta_\ell(f(x),x)\leq b(x).
\]
Then $\varphi$ is Fisher consistent for the task risk exactly when
$\overline\psi_A(\epsilon,x)>0$ for every
$x\in\mathcal X$ and $\epsilon>0$. With
$\psi_A=(\overline\psi_A)^{**}$, where the double star denotes the Fenchel biconjugate,
\[
\psi_A(R(f)-R^\star)
\leq R_{\varphi,A}(f)-R_{\varphi,A}^\star.
\]

For any aggregation map $A:\mathcal Z\to\mathcal A$, the required conditional quantities are
\[
\begin{aligned}
R_{\varphi,A}(s\mid x)&\coloneqq
\mathbb E[\varphi(s,A(Z))\mid X=x],\\
\delta_{\varphi,A}(s,x)&:=
R_{\varphi,A}(s\mid x)-
\inf_{s\in\mathbb R^d}R_{\varphi,A}(s\mid x),
\end{aligned}
\]
and
\[
\overline\psi_A(\epsilon,x)\coloneqq
\inf_{s\in\mathbb R^d}
\{\delta_{\varphi,A}(s,x)\mid\delta_\ell(s,x)\geq\epsilon\},
\qquad
\overline\psi_A(\epsilon)\coloneqq
\inf_{s\in\mathbb R^d}\overline\psi_A(\epsilon,x).
\tag{5}
\]
We leave the appropriate measurability conditions implicit because they are not central here. This result is almost a corollary of Steinwart [40, Thm. 2.8]. Appendix B.1 gives a proof for completeness. The result preserves classical guarantees but does not identify a specific improvement from aggregation.

#### 3.2.2 W3C and consistency

###### Corollary 3.2.

For $m=m(n)\to\infty$ and an identifying surrogate $\varphi$, denoted W3C and defined below,
\[
R_{\varphi,A_m}(f_n)-R_{\varphi,A_m}^\star\to0
\quad\Longrightarrow\quad R(f_n)-R^\star\to0.
\]

We consider a finite label space,
$\operatorname{card}(\mathcal Y)=k<\infty$, where $\operatorname{card}$ denotes the number of elements. The instantaneous task risk is
$R(f)=\mathbb E[\ell(f(X),Y)]$.
The ranking construction fits this setting. In that construction, $k$ is the ranking length and
$\operatorname{card}(\mathcal Y)=k(k-1)$.
From now on, $k$ denotes the number of labels. This slightly abuses the notation.

Given parameters $\{a_y\}_{y\in\mathcal Y}$, first minimize the empirical task loss and then substitute the corresponding parameter. We call this rule **generalized majority vote**, denoted J9P. With arbitrary tie-breaking, it is
\[
A_m(y_1,\ldots,y_m)\coloneqq a_{\hat y},
\qquad
\hat y=\mathop{\rm argmin}_{y\in\mathcal Y}
\sum_{i=1}^m\ell(y,y_i).
\tag{7}
\]
For independent and identically distributed $Y_i$ and finite $\mathcal Y$, a random $M<\infty$ necessarily exists such that every $m\geq M$ satisfies
\[
\mathop{\rm argmin}_{y\in\mathcal Y}
\left\{\sum_{i=1}^m\ell(y,Y_i)\right\}
\subset
y^\star(x)\coloneqq
\mathop{\rm argmin}_{y\in\mathcal Y}
\mathbb E[\ell(y,Y)\mid X=x].
\]
For any such task loss, this strategy gives asymptotic consistency under minimal identifiability: the surrogate need only distinguish the relevant labels.

###### Definition 3.1.

A surrogate $\varphi:\mathbb R^d\times\mathcal A\to\mathbb R$ is an **Identifying surrogate**, or W3C, with constants
$0<\mathsf C_{\varphi,1}\leq\mathsf C_{\varphi,2}<\infty$
if parameters $\{a_y\}_{y\in\mathcal Y}\subset\mathcal A$ and score vectors
$\{s_y\}_{y\in\mathcal Y}\subset\mathbb R^d$ exist such that
$\mathsf d(s_y)=y$ and, for all $y\neq y'$,
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
The first inequality makes a minimizer at parameter $a_y$ identify label $y$. A finite lower bound on $\varphi(\cdot,a)$ provides a finite $\mathsf C_{\varphi,2}$ for the second inequality. The definition does not require convexity or unaggregated consistency with $\mathcal A=\mathcal Y$ and $Z=Y$. It also imposes no decoder rule such as pointwise $\operatorname{argmax}$, which selects a maximizing argument.

The binary hinge loss
$\varphi(s,a)=\max\{1-sa,0\}$
on $\mathcal A=\mathcal Y=\{\pm1\}$ uses $a_y=s_y=y$:
\[
\varphi(s_1,a_1)=\varphi(s_{-1},a_{-1})=0,\qquad
\inf_{sa\leq0}\varphi(s,a)=1,
\]
\[
\varphi(s_1,a_{-1})=\varphi(s_{-1},a_1)=2.
\]
Its identifying-surrogate constants are $(1,2)$.

#### 3.2.3 W3C and consistency amplification

###### Theorem 1.

Let $\varphi$ be an identifying surrogate with constants
$(\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})$
and parameters $\{a_y\}_{y\in\mathcal Y}$. Let $A_m$ be the generalized majority-vote rule. Assume $0\leq\ell\leq1$ and noise condition (Nα), defined below, with $\alpha\in[0,1]$. For any $M>0$ and $f\in\mathcal F$ satisfying
\[
R(f)-R^\star\geq
2\mathbb P(\kappa(X)>M)+
(4\mathsf C_{\mathsf{MT}}e_m(M))^{\frac1{1-\alpha}},
\]
we have
\[
R(f)-R^\star\leq
\frac{16}{\mathsf C_{\varphi,1}}
\bigl(R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star\bigr).
\]
The noise condition number $\kappa$ and error function $e_m$ are defined below.

In particular, the corresponding threshold and slope are
\[
\xi=2\mathbb P(\kappa(X)>M)+
(4\mathsf C_{\mathsf{MT}}e_m(M))^{\frac1{1-\alpha}},
\qquad
\zeta=\frac{\mathsf C_{\varphi,1}}{16}.
\]
Writing
\[
\xi_m\coloneqq
\inf_M\left\{
2\mathbb P(\kappa(X)>M)+
(4\mathsf C_{\mathsf{MT}}e_m(M))^{\frac1{1-\alpha}}
\right\},
\]
gives $\xi_m\to0$. Indeed,
$\mathbb P(\kappa(X)>M)\to0$ as $M\uparrow\infty$, and
$e_m(M)\to0$ as $m$ increases for fixed $M$. Taking $\alpha=0$ gives asymptotic consistency.

Use the crude bound $\kappa(x)\leq1/\Delta(x)$, where $\Delta$ is defined below. Choose $M=1$ for
$\operatorname{card}(\mathcal Y)=k=2$, and optimize $M$ otherwise. This lower bounds the promised $\xi_{m,k}$ by setting
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
Appendix B.4 gives the algebraic substitutions and manipulations.

###### Corollary 3.3.

Under the theorem's conditions, every $f$ with
$R(f)-R^\star\geq\xi_{m,k}$ satisfies
\[
R(f)-R^\star\leq
\frac{16}{\mathsf C_{\varphi,1}}
\bigl(R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star\bigr).
\]

For a fixed $f$, the comparison requires fewer aggregated samples as $\alpha$ approaches $1$. At $\alpha=1$,
\[
m\geq
32\max\{\mathsf C_{\mathsf{MT}}^2,\mathsf C_{\mathsf{MT}}^4\}
\log\!\left(
\frac{4k(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})}
{\mathsf C_{\varphi,1}}
\right)
=O(\log k)
\]
gives $\xi_{m,k}=0$ and a linear uniform comparison inequality. The notation $O(\log k)$ describes growth bounded by a constant times $\log k$. Lower-noise problems need less aggregation to obtain stronger consistency.

Low noise allows a comparison of the form
$\psi(R(f)-R^\star)\leq R_\varphi(f)-R^\star$,
with $\psi$ linear over some range of positive $\epsilon$. Linear growth is the strongest possible [34, 32]. Strict comparison requirements can rule out practically convenient but inconsistent surrogates [23, 34, 32]. Following Osokin et al. [34], we relax the requirement to a positive range.

###### Definition 3.2.

The surrogate and aggregator have **level-$(\xi,\zeta)$ consistency**, denoted U6A, if some function $\psi$ satisfies both
\[
\psi(\epsilon)\geq\zeta\epsilon\quad(\epsilon\geq\xi),
\qquad
\psi(R(f)-R^\star)\leq
R_{\varphi,A}(f)-R_{\varphi,A}^\star.
\]
Only excess risks above the threshold must have linear growth. This retains a “good enough” criterion. Minimal assumptions allow levels $(o_m(1),\zeta)$ even for a surrogate that is not Fisher consistent. Here $o_m(1)$ denotes a quantity that tends to zero as $m$ increases.

The quantity
\[
\Delta(x)\coloneqq
\min_{\mathsf d(s)\notin y^\star(x)}
\delta_\ell(s,x)
\tag{8}
\]
is the **minimal excess conditional risk**, denoted F8Q. It is the gap associated with an incorrect prediction. In binary classification, it equals
$|2P(Y=1\mid X=x)-1|$.
Estimation should be harder when this gap is closer to zero.

We adapt classical assumptions [25]; see also Bartlett et al. [4]. The **Mammen-Tsybakov noise conditions**, denoted N6Y, use a positive constant $\mathsf C_{\mathsf{MT}}$ whose value may change:
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
These conditions control prediction disagreement and the probability of a small conditional-risk gap. Here $\alpha\in[0,1]$ is the noise exponent, and $\beta\in[0,\infty]$. Both conditions hold trivially at $\alpha=\beta=0$. As in the binary result [4, Thm. 3], they are equivalent when
$\beta=\alpha/(1-\alpha)$; see Appendix B.2.

Mao et al. [29] use exactly the same noise statistic for multiclass $H$-consistency, which compares risks within a hypothesis class $H$. They impose structure on the surrogate and hypothesis class to obtain a global convex lower bound. Under minimal structure, we instead need the **noise condition number**, denoted B2Z:
\[
\kappa(x)\coloneqq
\frac{\max_{\mathsf d(s)\neq y^\star(x)}\delta_\ell(s,x)}
{\min_{\mathsf d(s)\neq y^\star(x)}\delta_\ell(s,x)}.
\tag{9}
\]
This ratio compares the largest and smallest excess conditional risks of incorrect predictions. For $\mathsf d(s)\neq y^\star(x)$,
$\Delta(x)\geq\delta_\ell(s,x)/\kappa(x)$.
Binary classification has $\kappa(x)=1$ provided
$\mathbb P(\Delta(X)>0)=1$.

The auxiliary **error function**, denoted V5R, is
\[
e_m(t)\coloneqq
t\sqrt{\frac2m
\log\!\left(
\frac{4k(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})}
{\mathsf C_{\varphi,1}}
\right)}.
\tag{10}
\]
With $\kappa(x)=t$, an error-function value much smaller than one expresses the likely correctness of $A_m$.

Appendix B.3 proves Theorem 1. Its coarse bounds reflect the weak identifiability assumptions. Closer relationships between $\ell$ and $\varphi$ should strengthen the bounds by making generalized majority vote correct, or sufficiently correct, for $\varphi$. Near-consistency without aggregation could also connect smaller initial inconsistency with better dependence on $m$. We do not pursue these refinements.

### 3.3 Surrogate consistency with T4K

The examples below use majority vote and assume a noise exponent $\alpha\in[0,1]$. This imposes no loss of generality because $\alpha=0$ always satisfies (Nα). Appendix B.5 proves each result.

#### 3.3.1 Binary classification, nonsmooth surrogate

###### Lemma 3.1.

For every $\delta>0$, $\varphi$ is
$(\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})$-feasible with
\[
\mathsf C_{\varphi,1}=\phi(0)-\phi(\delta)>0,
\qquad
\mathsf C_{\varphi,2}=\phi(-\delta).
\]
Consequently, for $k=2$ and $f:\mathcal X\to\mathbb R$,
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

Here $\varphi(f(x),y)=\phi(yf(x))$, where $\phi$ is convex,
$\partial\phi(0)\subset(-\infty,0)$, and
$\lim_{t\to\infty}\phi(t)=0$.
The subdifferential $\partial\phi(0)$ is the set of supporting slopes at zero. The product $yf(x)$ is the binary margin.

For convex binary margin surrogates, Fisher consistency and a uniform comparison inequality over all measurable functions hold exactly when $\phi'(0)<0$ [4]. Aggregation can make an otherwise inconsistent $\phi$ consistent. This is a somewhat trivial thought experiment, not a surrogate one would use.

#### 3.3.2 Bipartite matching

###### Lemma 3.2.

On $2N$ vertices, the matching surrogate is an identifying surrogate with
\[
\mathsf C_{\varphi,1}=\frac1N,\qquad
\mathsf C_{\varphi,2}=2.
\]
Although the number of labels grows factorially, aggregation gives consistency once the count exceeds $O(N\log N)$. For $k\geq3$, every measurable
$f:\mathcal X\to\mathbb R^d$ satisfying
\[
R(f)-R^\star\geq
4\cdot\left(
\frac{32\mathsf C_{\mathsf{MT}}^4}{m}
\log(4k(2N+1))
\right)^{\frac{\alpha}{2(1-\alpha^2)}}
\]
also satisfies
\[
R(f)-R^\star\leq
16N\bigl(R_{\varphi,A_n}(f)-R_{\varphi,A_n}^\star\bigr).
\]

Structured output spaces are usually large. An embedding
$v:\mathcal Y\to\mathbb R^d$ maps each output to a vector that represents its structure [33]. Let
$\mathsf d(s)=\operatorname{argmax}_{y\in\mathcal Y}\langle s,v(y)\rangle$,
where $\langle\cdot,\cdot\rangle$ is the inner product. For a loss
$\ell:\mathcal Y\times\mathcal Y\to\mathbb R_+$
with $\ell(y,y)=0$, the **maximum-margin surrogate**, denoted A3J, is
\[
\varphi(s,y)=
\max_{\hat y\in\mathcal Y}
\left(\ell(\hat y,y)+
\langle v(\hat y)-v(y),s\rangle\right).
\tag{11}
\]
It is also called the generalized hinge loss or structured hinge loss. It compares the correct output score with competing output scores after adding their task losses. It is generally inconsistent except in certain low-noise cases [34, 32]. It originates in [42, 44, 19].

For the matching example, $G=(V,E)$ has vertex set
$V=V_1\cup V_2$, with
$V_1=\{1,\ldots,N\}$ and
$V_2=\{N+1,\ldots,2N\}$.
Each of the $N$ edges joins exactly one unique node on each side. The label space $\mathcal Y$ contains all matchings, so
$k=\operatorname{card}(\mathcal Y)=N!$.
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
The norm $\|\cdot\|_1$ sums absolute coordinate values.

#### 3.3.3 Structured prediction

###### Lemma 3.3.

The maximum-margin surrogate is an identifying surrogate with
\[
\mathsf C_{\varphi,1}=
\min_{\hat y\neq y}\ell(\hat y,y),
\qquad
\mathsf C_{\varphi,2}=
\max_{y\in\mathcal Y}\tau(y)+1.
\]
In particular, $v(y)\in\{0,1\}^d$ and
$\ell(\hat y,y)=\frac1{2d}\|v(\hat y)-v(y)\|_1$
give $\tau(y)=1$ for every $y$ and
$\mathsf C_{\varphi,2}=2$.

Nontrivial consistency holds once
$m\geq\log\operatorname{card}(\mathcal Y)$,
as in the binary matching case. If $0\leq\ell(\cdot,\cdot)\leq1$, every measurable
$f:\mathcal X\to\mathbb R^d$ satisfying
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
Here $|\mathcal Y|$ also denotes the number of labels.

We assume that the general structured decoder can select every class. That is, for each $y$,
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
This quantity compares task-loss gaps with the score gaps that distinguish $y$ from other classes.

## 4 Robustness and consistency for models

We now replace all measurable functions with a restricted, often parametric class $\mathcal F$. A well-specified model makes consistency straightforward, but this assumption is brittle and unrealistic. Surrogate consistency on subsets of measurable functions is a long-standing topic [12, 31].

Recent $H$-consistency bounds connect excess target risk with surrogate regret, or excess surrogate risk, within a regret hypothesis class $H$ in several settings [2, 3, 26, 28, 29]. Those results focus mainly on the surrogate. Our aggregation contribution addresses a separate issue and depends less on a particular class or surrogate.

### 4.1 Binary consistency failure in finite dimensions

###### Proposition 4.

For every $\epsilon>0$ and nonzero
$\theta^\star\in\mathbb R^d$, some distribution $P$ on $(X,Y)$ satisfies optimal predictability, denoted K7M and defined below, along $\theta^\star$. Yet every
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
Thus the fitted and optimal parameter vectors can be essentially orthogonal. This occurs for any nonnegative loss $\phi$. Appendix D.1 gives the proof.

###### Corollary 4.1.

As a corollary to Theorem 2, if $P$ satisfies optimal predictability along $\theta^\star$ and $\epsilon_m\to0$ as $m\to\infty$, the aggregated approximate minimizers satisfy
\[
R(f_{\theta_m})\to R(f_{\theta^\star}),
\qquad
\cos\angle(\theta_m,\theta^\star)\to1.
\]

The binary setting uses $\mathcal Y=\{\pm1\}$ and
$\ell(\mathsf d(s),y)=1\{ys\leq0\}$.
The margin surrogate is
$\varphi(s,y)=\phi(sy)$
for a convex function $\phi:\mathbb R\to\mathbb R_+$.
The input space is $\mathcal X=\mathbb R^d$, and
\[
\mathcal F=
\{f_\theta:f_\theta(x)=\langle\theta,x\rangle\}_{\theta\in\mathbb R^d}.
\]
Classical consistency can fail even when this class contains an optimal classifier.

The inclusion requirement, **optimal predictability**, or K7M, means that some $f\in\mathcal F$ satisfies
\[
\mathsf{sgn}(f(x))=
\mathsf{sgn}(P(Y=1\mid X=x)-1/2).
\tag{12}
\]
For a vector $\theta^\star$ that realizes this requirement,
\[
\mathsf{sgn}(\langle\theta^\star,x\rangle)
=
\mathsf{sgn}(P(Y=1\mid x)-\tfrac12).
\]
Then $f_{\theta^\star}$ minimizes $R$ over every measurable function, and we say that $P$ has optimal predictability along $\theta^\star$.

An approximate minimizer lies within $\epsilon$ of the infimum:
\[
\operatorname{\epsilon\textup{-argmin}}g
=
\operatorname{\epsilon\textup{-argmin}}_\theta g(\theta)
\coloneqq
\{\theta:g(\theta)\leq\inf_\theta g(\theta)+\epsilon\}.
\]
With independent repeated labels
$Z=(Y_1,\ldots,Y_m)$
and majority vote $A_m$, we study the sequence
\[
\theta_m\in
\operatorname{\epsilon_m\textup{-argmin}}_\theta
R_{\varphi,A_m}(f_\theta)
=
\operatorname{\epsilon_m\textup{-argmin}}_\theta
\mathbb E[
\phi(A_m(Y_1,\ldots,Y_m)f_\theta(X))].
\]
Restricting the model to linear functions can therefore make unaggregated surrogate minimization arbitrarily incorrect. Aggregation preserves asymptotic consistency.

### 4.2 Aggregation and restricted classes

###### Theorem 2.

Let a cone $\mathcal F$ satisfy optimal predictability for $P$, and assume
$P(\Delta(X)>0)=1$.
A cone is a class closed under positive scaling, as specified below. For
$\epsilon_m\geq0$ with $\epsilon_m\to0$, every sequence
\[
f_m\in
\operatorname{\epsilon_m\textup{-argmin}}_{f\in\mathcal F}
R_{\varphi,A_m}(f)
=
\operatorname{\epsilon_m\textup{-argmin}}_{f\in\mathcal F}
\mathbb E[
\varphi(f(X),A_m(Y_1,\ldots,Y_m))]
\]
satisfies $R(f_m)\to R^\star$. Appendix D.2 gives the proof.

The multiclass labels are
$\mathcal Y=\{1,\ldots,k\}$.
The loss is zero-one:
$\ell(y,y')=1\{y\neq y'\}$.
The decoder is
$\mathsf d(s)=\operatorname{argmax}_{y\in[k]}s_y$.
The linear-cone conditions are
\[
f(x)^T\mathbf1=0,\qquad
f\in\mathcal F,\ t>0\ \Longrightarrow\ tf\in\mathcal F.
\]
The normalization does not affect decoding. The nonnegative surrogate
$\varphi:\mathbb R^k\times[k]\to\mathbb R_+$
is Fisher consistent and satisfies
\[
\varphi(s,y)\to0
\quad\text{if}\quad
s_y-s_j\to+\infty\text{ for every }j\neq y.
\tag{13}
\]
The response space is $\mathcal Z=\mathcal Y^m$, and aggregation uses majority vote. The population maximizer
$y^\star(x)=\operatorname{argmax}_yP(Y=y\mid x)$
is unique.

Multiclass optimal predictability requires some
$f\in\mathcal F$, with $f:\mathcal X\to\mathbb R^k$, such that
\[
\operatorname{argmax}_y f_y(x)
\in\operatorname{argmax}_yP(Y=y\mid X=x)
\quad\text{for every }x.
\]
The broad result for cones requires aggregation. Without it, even binary subclasses that contain an optimal predictor can have arbitrarily uninformative surrogates.

Examples of Fisher-consistent losses that meet the limiting condition include multiclass logistic loss,
\[
\varphi(s,y)=\log\left(\sum_{j=1}^k e^{s_j-s_y}\right),
\]
and
\[
\varphi(s,y)=\sum_{i\neq y}\phi(s_y-s_i),
\]
where $\phi$ is convex and nonincreasing, with
$\phi'(0)<0$ and $\inf_t\phi(t)=0$.
Zhang [51, Thm. 8] proves consistency of these losses over
$\mathcal F=\{f:\mathcal X\to\mathbb R^k:\mathbf1^Tf=0\}$.

Binary margin predictors correspond to
$f(x)=(g(x),-g(x))$
for measurable $g$. For $\mathcal X=\mathbb R^d$, linear functions
$f(x)=(\langle\theta_1,x\rangle,\ldots,\langle\theta_k,x\rangle)$
with $\sum_{i=1}^k\theta_i=0$ form a convex cone. Without aggregation, such examples typically require strong structure, such as convexity. With aggregation, the minimal limiting condition allows asymptotic consistency.

### 4.3 Finite-dimensional multiclass classification

###### Proposition 5.

Let $k\geq3$ and $\Sigma=I$, where $I$ is the identity matrix. For
$Z\sim\mathsf N(0,I_{k-1})$, a centered normal vector with identity covariance, assume that the map
\[
M\mapsto DM\coloneqq
\mathbb E\!\left[
Z\bigl(
\nabla\sigma^{\rm lr}({\Theta^\star}^TZ)MZ
\bigr)^T
\right]
\]
is invertible. Here $\nabla\sigma^{\rm lr}$ denotes the derivative matrix of the logistic link defined below. Then some $\epsilon_0>0$ has the following property: every
$\epsilon\in(0,\epsilon_0)$
admits a set $\mathcal T_\epsilon$ of Lebesgue measure, or ordinary volume, at most $\epsilon$ with
\[
\Theta_1(\epsilon)/\|\Theta_1(\epsilon)\|
\neq
\Theta^\star/\|\Theta^\star\|.
\]

Appendix E.1 proves Proposition 5.

###### Theorem 3.

Suppose $\Theta^\star=U^\star T^\star$, where
$U^\star\in\mathbb R^{d\times(k-1)}$ is orthogonal and
$T^\star\in\mathbb R^{(k-1)\times(k-1)}$ is nonsingular, or invertible. For each $m$, there is a matrix
$T_m\in\mathbb R^{(k-1)\times(k-1)}$
with $\Theta_m=U^\star T_m$, and
\[
\|T_m\|\to\infty,\qquad
T_m/\|T_m\|\to T^\star/\|T^\star\|
\quad(m\to\infty).
\]

Appendix E.2 proves Theorem 3.

Pair-based surrogate risks can be inconsistent even under optimal predictability and a linear model that is correct except on an $\epsilon$-fraction of the data. Majority vote recovers the optimal linear predictor asymptotically. For $k\geq3$, an arbitrarily slight mis-specification of the true link defeats risk consistency. Aggregation needs no assumption on the true link beyond consistency of its ordering.

Here $\mathcal X=\mathbb R^d$, $\mathcal Y=[k]$, and
$\Theta=[\theta_1~\cdots~\theta_{k-1}]\in\mathbb R^{d\times(k-1)}$.
The conditional distribution is
$Y\mid X=x\sim\mathsf{Cat}(p_\Theta(x))$,
where $\mathsf{Cat}$ denotes the categorical distribution with the given class probabilities. The vector $p_\Theta(x)\in\mathbb R_+^k$ satisfies
$\mathbf1^\top p_\Theta(x)=1$ and
\[
p_\Theta(x)=
\begin{bmatrix}
\sigma(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)\\
1-\mathbf1^\top
\sigma(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)
\end{bmatrix}.
\]
The link $\sigma:\mathbb R^{k-1}\to\mathbb R_+^{k-1}$ maps scores to the first $k-1$ class probabilities and satisfies
$\mathbf1^\top\sigma\leq1$.
Set $t_k=0$ and
$\sigma_k(t)=1-\mathbf1^\top\sigma(t)$.
We impose the ordering condition
\[
t_y=\max_{1\leq i\leq k}t_i
\quad\Longleftrightarrow\quad
\sigma_y(t)=\max_{1\leq i\leq k}\sigma_i(t).
\tag{14}
\]
Multiclass logistic regression uses the link
\[
\sigma_i^{\rm lr}(t)=
\frac{e^{t_i}}{1+e^{t_1}+\cdots+e^{t_{k-1}}}.
\]

The dataset with repeated labels is
$\{(X_i;(Y_{1i},\cdots,Y_{im}))\}_{i=1}^n$, with
\[
Y_{ij}\mid X_i\stackrel{\rm iid}{\sim}
\mathsf{Cat}(p_{\Theta^\star}(X_i)).
\]
Majority vote, with arbitrary tie-breaking, gives
$Y_i^+=A_m(Y_{i1},\cdots,Y_{im})$,
where
$Y_i^+\mid X_i\sim\mathsf{Cat}(p_{\Theta^\star,m}(X_i))$.
Let $\rho_m(t)$ be the majority-vote distribution for $m$ items whose initial probabilities are
$\sigma(t)\in\mathbb R_+^{k-1}$.
Then
\[
p_{\Theta,m}(x)=
\begin{bmatrix}
\rho_m(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)\\
1-\mathbf1^\top
\rho_m(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)
\end{bmatrix},
\]
and $\rho_m$ satisfies ordering consistency.

With $\theta_k=\mathbf0$, the fitted logistic loss is
\[
\varphi(\Theta^\top x,y)=
-\langle\theta_y,x\rangle+
\log\left(1+\sum_{i=1}^{k-1}
\exp(\langle\theta_i,x\rangle)\right).
\tag{15}
\]
For $L_m(\Theta)=\mathbb E[\varphi(\Theta^\top X,Y_m^+)]$, its gradient with respect to $\theta_i$ is
\[
\nabla_{\theta_i}\varphi(\Theta^\top x,y)=
-x(1\{y=i\}-\sigma_i^{\rm lr}(\Theta^\top x)).
\]
Thus its minimizer $\Theta_m$ satisfies
\[
\nabla_\Theta L_m(\Theta_m)=
\mathbb E\!\left[
X\left(
\sigma^{\rm lr}(\Theta_m^\top X)
-\rho_m({\Theta^\star}^\top X)
\right)^\top
\right]=0.
\]

Well-specified unaggregated models have standard statistical consistency and efficiency. For $m\geq1$, $k=2$, and
$X\sim\mathsf N(0,I_d)$,
Cheng et al. [7, Prop. 3] obtain
$\Theta_m=t_m\Theta^\star$
for some $t_m>0$, even with a mis-specified link. The direction
$\Theta^\star/\|\Theta^\star\|_2$
therefore suffices for consistent binary classification at every aggregation level.

For a fixed set
$\mathcal T_\epsilon\subset\mathbb R^{k-1}$
of Lebesgue measure $\epsilon$, set
\[
\sigma^\epsilon(t)=
\sigma^{\rm lr}(t)1\{t\notin\mathcal T_\epsilon\}
+\frac1k\mathbf1\,1\{t\in\mathcal T_\epsilon\}.
\]
The conditional distribution on $Y\in\{1,\ldots,k\}$ samples
$Y\sim\sigma^{\rm lr}(t)$ if $t\in\mathcal T_\epsilon$ and samples uniformly otherwise. The link satisfies ordering consistency and optimal predictability. For $\epsilon>0$, define
\[
L_{m,\epsilon}(\Theta)\coloneqq
\mathbb E[
\mathbb E_{\sigma^\epsilon}[
\varphi(\Theta^\top X,Y_m^+)\mid X]],
\qquad
\Theta_m(\epsilon)=
\operatorname{argmin}_\Theta L_{m,\epsilon}(\Theta).
\]
This is the population logistic loss for $m$-label majority vote when
$Y\mid X=x\sim\sigma^\epsilon({\Theta^\star}^\top x)$.
Although $\Theta_1(0)=\Theta^\star$, arbitrarily small positive corruption destroys unaggregated consistency.

For a general ordering-consistent link $\sigma$, the theorem uses
\[
L_{m,\sigma}(\Theta)=
\mathbb E[
\mathbb E_\sigma[
\varphi(\Theta^\top X,Y_m^+)\mid X]],
\qquad
\Theta_m=\operatorname{argmin}L_{m,\sigma}(\Theta).
\]
These definitions give robustness to mis-specification whenever some link describes the relationship between $X$ and $Y$. Without aggregation, multiclass methods are generally inconsistent.

## 5 Numerical illustrations

Two experiments support the theory. One illustrates nonparametric consistency amplification. The other illustrates finite-dimensional multiclass asymptotic consistency under link mis-specification.

##### Surrogate consistency amplification for D4X.

As $m\to\infty$, the aggregated curve moves from quadratic toward linear growth, improving consistency. The curves from the general theorem are conservative.

The binary margin surrogate
$\varphi(f(x),y)=\phi(yf(x))$
uses the **truncated quadratic loss**, denoted D4X:
$\phi(\delta)=\max\{1-\delta,0\}^2$.
Its exact unaggregated calibration function is
$\psi(\epsilon)=\epsilon^2$
[4, Example 2]. Take $\delta=1$ in the binary lemma and impose no low-noise condition, so that
$\alpha=0$ and $\mathsf C_{\mathsf{MT}}=1$.
This gives
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
The positive-part notation $(u)_+$ means $\max\{u,0\}$. The two comparisons are
\[
\psi(R(f)-R^\star)\leq R_\varphi(f)-R_\varphi^\star,
\qquad
\psi_m(R(f)-R^\star)\leq
R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star.
\]

Figure 1: D4X comparison curves for unaggregated $\psi$ (optimal) and T4K $\psi_m$ (suboptimal).

| Figure | Aggregated-label counts |
|---|---|
| 1 | $m\in\{2^{16},2^{20},2^{24},2^{28},2^{32}\}$ |

##### Asymptotic consistency for a mis-specified logistic model.

Synthetic data from a corrupted link show asymptotic consistency as $m\to\infty$. We use the finite-dimensional logistic loss specified above. The piecewise link agrees with softmax, the logistic probability map, near the origin. Outside radius $r$, it shrinks the scores:
\[
\widetilde\sigma(t)=
\sigma^{\rm lr}(t)1\{\|t\|\leq r\}
+\sigma^{\rm lr}(\alpha t)1\{\|t\|>r\}.
\]

| (i) Experiment with constant dataset size $n$ | (ii) Experiment with constant label size $nm$ |
|---|---|
| For each $m$, $n=10{,}000$ multilabeled observations $(X_i,(Y_{i1},\ldots,Y_{im}))$ sampled from $\widetilde\sigma$ | For each $m$, $n=640{,}000/m$; total labels constant across $m$ |

Figure 2: Mis-specified logistic-model illustrations, left panel (i), right panel (ii); plotted classification-rule error is $\left\|\|T_m\|-T^\star/\|T^\star\|\right\|$ against label count.

| Setting | Value |
|---|---|
| Dimension | $d=2$ |
| Classes | $k=3$ |
| True parameters | $\theta_1^\star=(0,1)^\top,\quad\theta_2^\star=(1,1)^\top$ |
| Corrupted-link parameters | $r=3,\quad\alpha=0.1$ |
| Label counts | $m\in\{1,4,16,64,256,1024\}$ |
| Error bars | $95\%$ |
| Trials for each $m$ | $T=100$ |

## 6 Discussion and future work

Cheng, Asi and Duchi [7] discuss the debate over label collection. Using all unaggregated labels appears statistically more efficient when models can represent all uncertainty. Otherwise, cleaning appears more robust. Dorner and Hardt [11] favor more noisy labels over cleaned labels when comparing binary classifiers on validation data. Our results establish fundamental limits of uncleaned supervised-label information and robustness gains from cleaning.

##### Finite $m$ and fundamental limits.

Many consistency results require $m\to\infty$. This reliance on a limit remains unsatisfying. Ranking allows explicit guarantees for finite $m$.

A theory that jointly accounts for $(\ell,\varphi)$ and the available count $m$ remains open. Second- or third-order item comparisons may require increasing counts. These include the powers $C_x^p$ suggested by Keener [20]. Precisely separating problems that require cleaning and aggregation from those that do not is a central challenge and a natural, important direction.

##### Fundamental limits of B2Z.

Beyond binary classification, the noise condition number $\kappa(X)$ can remain large even under low Mammen-Tsybakov noise, with $\alpha\approx1$. Binary problems trivially have $\kappa(X)=1$. The minimal surrogate assumptions produce this behavior.

Closer connections between the task and surrogate could better represent problem difficulty. Explicit failure constructions and sharper fundamental limits could improve our understanding of predictive-model fitting.

##### Mis-specified models.

Our mis-specified-model results require optimal predictability: $\mathcal F$ must contain a Bayes-optimal classifier, one that minimizes the task risk over all measurable classifiers. Classical surrogate consistency can fail even under this requirement, while aggregation succeeds.

Beyond this restriction, Nguyen et al. [31] for binary classification and Duchi et al. [12] for multiclass classification study joint inference of a predictor $f$ and a representation of $x$. They show that this forces the surrogate to have a form that depends on the task loss. Their decision rules remain infinitely powerful. New approaches are needed.

##### Restricted-class comparison with aggregation.

Nonparametric comparison inequalities reduce regret caused by the surrogate. Parametric results establish statistical consistency. Recent work on $H$-consistency [2, 28, 27, 29] suggests extending aggregation comparisons to restricted classes.

Mao et al. [27] establish rate-optimal quadratic growth of $\psi$ for smooth convex loss. Aggregation-based improvements in $\mathcal H$-consistency remain of interest. An extension could unify the parametric and nonparametric benefits and describe how the “size” of the hypothesis class affects aggregation.
