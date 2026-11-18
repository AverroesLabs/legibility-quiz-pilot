# Some Robustness Properties of Label Cleaning

Chen Cheng

John Duchi

###### Abstract

Aggregated labels, including information distilled from noisy responses, permit robustness unavailable without cleaning: minimization of a typically convex surrogate loss obtains stronger task-risk consistency guarantees than raw-label procedures; finite-dimensional classifiers remain consistent under loss mis-specification that defeats standard uncertainty-modeling approaches. Perfectly specified statistical models favor incorporating all information for efficiency, yet even slight loss mis-specification destroys standard consistency. Aggregation retains convergence to optimal classifiers; collection, fitting, and prediction jointly admit robustness through noisy-signal refinement.

## 1 Introduction

Aggregation permits consistent optimal prediction under surrogate rather than task loss when unaggregated estimation cannot, including finite-dimensional statistical problems. Linear-estimator constructions in ranking and binary and multiclass classification separate procedures restricted to $(X,Y)$ pairs from aggregation procedures; nonparametric results combine aggregation count and noise conditions, permit approximately linear comparison inequalities under weak surrogate conditions, and extend consistency beyond unrestricted measurable hypothesis classes.

The conventional supervised pipeline collects $(X_i,Y_i)_{i=1}^n$, fits a model [17], and suppresses the cleaning, filtering, repeated crowdsourced labeling, and denoising required by modern datasets [9, 22, 39]. Work on crowdsourced “gold standard” labels [8, 48, 47, 46, 37] concerns this machinery; the present concern is its fundamental, necessary improvement of fitted-model consistency.

Important contributions to surrogate-risk consistency theory date to the 2000s [52, 24, 40]; Bartlett, Jordan and McAuliffe [4] characterize convex-surrogate consistency for binary zero-one classification. Subsequent settings include multi-label classification [50, 43, 16, 49, 1], ranking [13, 14, 36], structured prediction [34, 6, 32], ordinal regression [35], restricted hypothesis classes [2, 3, 28, 29], and general theory [40]. Pair-based analyses admit empirical-process theory [45, 5, 21, 4], not the aggregation machinery of contemporary dataset creation. Whether aggregation is needed at all, whether pair-based surrogate consistency suffices, and in what sense aggregation improves consistency delimit the questions considered.

## 2 Preliminaries

Two regimes are retained: unrestricted Borel $\mathcal F$, with the classical notion (Q7D: Fisher consistency), and parametric, potentially mis-specified $\mathcal F$, with aggregation improving consistency in each.

For i.i.d. $(X_i,Y_i)_{i=1}^n\sim P$ on input and output spaces $\mathcal X\times\mathcal Y$, scores $f:\mathcal X\to\mathbb R^d$, $d\geq1$, are decoded by $\mathsf d:\mathbb R^d\to\mathcal Y$, giving $\widehat y=\mathsf d\circ f(x)$. The distribution-dependent task criterion is

\[
R(f)\coloneqq
\mathbb E_P\!\left[\ell(\mathsf d\circ f(X),P(\cdot\mid X))\right].
\tag{2}
\]

Its minimizers frequently coincide with the original problem (2); conditional-distribution dependence allows greater sophistication. With $\mathcal Y=\{1,\ldots,k\}$ and
\[
\ell(\widehat y,P)=\sum_y P(Y=y)1\{\widehat y\neq y\},
\]
this is the standard zero-one error rate.

The instantaneous formulation, for $\ell:\mathcal Y\times\mathcal Y\to\mathbb R_+$, minimizes over $f\in\mathcal F$
\[
R(f):=\mathbb E_P[\ell(\mathsf d\circ f(X),Y)].
\tag{1}
\]
Binary classification has $d=1$, $\mathsf d(s)=\mathsf{sgn}(s)$,
$\ell(y,y')=1\{yy'\leq0\}$, and $R(f)=\mathbb P(Yf(X)\leq0)$. Task losses may be nonsmooth, nonconvex, and uninformative; the zero-one landscape is flat almost everywhere, precluding even practical first-order optimization. An easier surrogate $\varphi:\mathbb R^d\times\mathcal Y\to\mathbb R$ replaces it:
\[
R_\varphi(f):=\mathbb E_P[\varphi(f(X),Y)].
\]
Such replacement requires consistency with the task criterion.

### 2.1 Label aggregation

Minimization of
\[
R_{\varphi,A}(f):=\mathbb E[\varphi(f(X),A(Z))]
\tag{3}
\]
is assessed against actual task-risk minimization, where the response replacement $Z\in\mathcal Z$, aggregation space $\mathcal A$, map $A:\mathcal Z\to\mathcal A$, and nonnegative surrogate $\varphi:\mathbb R^d\times\mathcal A\to\mathbb R_+$ specify the quantity (P9S: aggregated surrogate risk), rather than an ordinary $(X_i,Y_i)$ response.

Repeated labeling uses $Z=(Y_1,\ldots,Y_m)\in\mathcal Y^m$, with $Y_i\mid X=x\stackrel{\rm iid}{\sim}P_{Y\mid X=x}$, and empirical task minimization
\[
A(Z)=A(\{Y_1,\ldots,Y_m\})
=\mathop{\rm argmin}_{y\in\mathcal Y}
\sum_{l=1}^m\ell(y,Y_l).
\]
For $\ell(\widehat y,y)=1\{y\neq\widehat y\}$ this is precisely the voting procedure (T4K: majority vote), whereas general $\ell$ permits abstract procedures and the notation $A_m(Z)$ records the repetition count. Mao et al. [28] use a similar multilabeled structure for multilabel losses, rather than the aggregation pipeline.

A rough $K$-nearest-neighbor construction instead supplies $X$ with $Y$ and an additional independent draw $(X_i,Y_i)_{i=1}^m\stackrel{\rm iid}{\sim}P$, so $Z=(Y,(X_i,Y_i)_{i=1}^m)$. For a metric $\operatorname{dist}:\mathcal X\times\mathcal X\to\mathbb R_+$, order
\[
\operatorname{dist}(X,X_{(1)})\leq\cdots\leq
\operatorname{dist}(X,X_{(m)}),\qquad X_{(0)}=X,
\]
and, for $K\geq0$, take
\[
A_{m,K}(Z)\coloneqq
\mathop{\rm argmin}_{y\in\mathcal Y}
\sum_{l=0}^K\ell(y,Y_{(l)}).
\]
The population-level repeated-sampling treatment also extends to aggregation from one sample $(X_i,Y_i)_{i=1}^n$ (Appendix C).

## 3 Surrogate consistency

###### Corollary 3.1.

Positivity of $\psi(\epsilon,x)$ for all $x\in\mathcal X$ and $\epsilon>0$ is equivalent to $\varphi$ being Q7D for $\ell$. If $\psi$ is the Fenchel biconjugate of $\overline\psi$, then
\[
\overline\psi(\epsilon)>0\quad\Longleftrightarrow\quad
\psi(\epsilon)>0,
\qquad
\psi(R(f)-R^\star)\leq R_\varphi(f)-R_\varphi^\star
\]
for every measurable $f$.

The standard framework [40] takes $\mathcal F$ to contain all Borel measurable $f:\mathcal X\to\mathbb R^d$. For all distributions $P$, its two requirements [40, 4, 52] are:

(i) Every sequence $f_n\in\mathcal F$ with
$R_\varphi(f_n)\to R_\varphi^\star$ satisfies $R(f_n)\to R^\star$.

(ii) A nondecreasing $\psi:\mathbb R_+\to\mathbb R_+$, positive at every $\epsilon>0$, satisfies
\[
\psi(R(f)-R^\star)\leq R_\varphi(f)-R_\varphi^\star
\quad\text{for every }f\in\mathcal F.
\]

The latter quantitative requirement (R2V: uniform comparison inequality) is stronger, need not hold, and coincides with (i) for convex margin-based binary surrogates [4].

With $s\in\mathbb R^d$, the conditional and excess quantities are
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
Canonical construction uses the excess-risk object (H8N: calibration function), with pointwise and uniform forms
\[
\psi(\epsilon,x)\coloneqq
\inf_{s\in\mathbb R^d}
\{\delta_\varphi(s,x)\mid\delta_\ell(s,x)\geq\epsilon\},
\qquad
\overline\psi(\epsilon)\coloneqq
\inf_{x\in\mathcal X}\psi(\epsilon,x).
\]
The equivalence and inequality follow from Zhang [51, Prop. 25] and Steinwart [40, Thm. 2.8 and Lemma 2.9].

A Q7D surrogate for the general task would make single pairs sufficient for consistent estimation, simplify collection, and admit the full empirical-process toolkit. Such pair-only results are generally impossible.

### 3.1 Q7D failure without aggregation: ranking

###### Proposition 1.

For the ranking task with $k\geq3$ outcomes, every $\varphi:\mathbb R^k\times\mathcal Y\to\mathbb R$ convex in its first argument fails Q7D (Appendix A.1).

###### Proposition 2.

For $s,q\in\mathbb R^k$, set
\[
\varphi(s,q)=\|s-q\|_2^2,\qquad \varphi(s,\star)=0.
\]
With the aggregation specified below and $m\geq k$, $\varphi$ is Q7D for the ranking risk (proof: Appendix A.2).

The response-space expansion from $\mathcal Y$ to $\mathbb R^k$, not available to the convex single-comparison formulation, separates these conclusions; regression on ordering frequencies yields the positive result.

Pairwise-comparison ranking [20, 15, 14, 30] has
\[
\mathcal Y=\{(i,j):i\neq j,\ 1\leq i,j\leq k\},
\]
where $(i,j)$ says item $i$ outranks item $j$. For each $x$, population aggregation transforms $p_{ij}=P(Y=(i,j)\mid x)$ into an ordering. A simple comparison scheme [20, cf.] uses
\[
(C_x)_{ii}=0,\qquad
(C_x)_{ij}=\frac{p_{ij}}{\sum_{l\neq j}p_{lj}}
\quad(i\neq j),\qquad
0/0=1/(k-1).
\]
Thus $C_x\in\mathbb R_+^{k\times k}$ is stochastic, $C_x^T\mathbf1=\mathbf1$; $C_x\mathbf1$ measures preference frequency for each item. Higher powers $C_x^p\mathbf1$, Perron vectors [20], and more sophisticated schemes admit similar negative results.

Absorbing decoding into the loss gives
\[
\ell(s,C)\coloneqq
\max_{i<j}1\!\left\{
(s_i-s_j)(e_i-e_j)^TC\mathbf1\leq0,\ 
(e_i-e_j)^TC\mathbf1\neq0
\right\},
\]
penalizing disagreement between the score and matrix orderings, with
\[
R(f)\coloneqq
\mathbb P\!\left(f(X)\text{ and }C_X\mathbf1
\text{ order differently}\right).
\tag{4}
\]
Restriction to $s^T\mathbf1=0$ is minor, familiar in multiclass classification [51, 43], and natural when only ordering is decoded.

For multiple independent comparisons $Z=(Y_1,\ldots,Y_m)$ conditional on $X$, put
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
Regression acts directly on non-$\star$ outputs.

### 3.2 Aggregation and stronger surrogate consistency

Aggregation both extends ordinary surrogate-risk theory and upgrades nominally inconsistent losses.

#### 3.2.1 Basic extensions

###### Proposition 3.

Suppose an integrable $b:\mathcal X\to\mathbb R_+$ satisfies
\[
\int b(x)\,dP(x)<\infty,\qquad
\delta_\ell(f(x),x)\leq b(x).
\]
Then $\varphi$ is Q7D for the task risk exactly when
$\overline\psi_A(\epsilon,x)>0$ for every $x\in\mathcal X$, $\epsilon>0$; with
$\psi_A=(\overline\psi_A)^{**}$,
\[
\psi_A(R(f)-R^\star)
\leq R_{\varphi,A}(f)-R_{\varphi,A}^\star.
\]

For arbitrary $A:\mathcal Z\to\mathcal A$, the required objects are
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
Appropriate measurability conditions are tacit, noncentral; this near-corollary of Steinwart [40, Thm. 2.8], with a completeness proof in Appendix B.1, preserves classical guarantees without supplying a particular improvement from aggregation.

#### 3.2.2 W3C and consistency

###### Corollary 3.2.

For $m=m(n)\to\infty$ and W3C $\varphi$,
\[
R_{\varphi,A_m}(f_n)-R_{\varphi,A_m}^\star\to0
\quad\Longrightarrow\quad R(f_n)-R^\star\to0.
\]

The regime is finite $\operatorname{card}(\mathcal Y)=k<\infty$ and instantaneous task loss
$R(f)=\mathbb E[\ell(f(X),Y)]$. The ranking construction belongs to this regime: there its $k$ is ranking length, $\operatorname{card}(\mathcal Y)=k(k-1)$; henceforth $k$ denotes label-space cardinality, with this slight notation abuse.

For parameters $\{a_y\}_{y\in\mathcal Y}$, empirical task minimization followed by parameter substitution defines the rule (J9P: generalized majority vote), with arbitrary tie-breaking and formula
\[
A_m(y_1,\ldots,y_m)\coloneqq a_{\hat y},
\qquad
\hat y=\mathop{\rm argmin}_{y\in\mathcal Y}
\sum_{i=1}^m\ell(y,y_i).
\tag{7}
\]
For i.i.d. $Y_i$ and finite $\mathcal Y$, a random $M<\infty$ necessarily exists such that every $m\geq M$ satisfies
\[
\mathop{\rm argmin}_{y\in\mathcal Y}
\left\{\sum_{i=1}^m\ell(y,Y_i)\right\}
\subset
y^\star(x)\coloneqq
\mathop{\rm argmin}_{y\in\mathcal Y}
\mathbb E[\ell(y,Y)\mid X=x].
\]
The strategy supplies asymptotic consistency under minimal identifiability, for any such task loss.

###### Definition 3.1.

A surrogate $\varphi:\mathbb R^d\times\mathcal A\to\mathbb R$ has the property (W3C: Identifying surrogate), with constants $0<\mathsf C_{\varphi,1}\leq\mathsf C_{\varphi,2}<\infty$, when parameters $\{a_y\}_{y\in\mathcal Y}\subset\mathcal A$ and vectors $\{s_y\}_{y\in\mathcal Y}\subset\mathbb R^d$ exist with $\mathsf d(s_y)=y$ and, for all $y\neq y'$,
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
The first inequality makes a minimizer at parameter $a_y$ identify $y$; a finite lower bound on $\varphi(\cdot,a)$ supplies finite $\mathsf C_{\varphi,2}$ for the second. Neither convexity, unaggregated consistency with $\mathcal A=\mathcal Y$, $Z=Y$, nor a decoder requirement such as pointwise $\operatorname{argmax}$ is imposed.

Binary hinge $\varphi(s,a)=\max\{1-sa,0\}$ on $\mathcal A=\mathcal Y=\{\pm1\}$ uses $a_y=s_y=y$:
\[
\varphi(s_1,a_1)=\varphi(s_{-1},a_{-1})=0,\qquad
\inf_{sa\leq0}\varphi(s,a)=1,
\]
\[
\varphi(s_1,a_{-1})=\varphi(s_{-1},a_1)=2.
\]
Its W3C constants are $(1,2)$.

#### 3.2.3 W3C and consistency amplification

###### Theorem 1.

Let $\varphi$ be $(\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})$-W3C with parameters $\{a_y\}_{y\in\mathcal Y}$, and let $A_m$ be the J9P rule. Assume $0\leq\ell\leq1$ and condition (Nα), with $\alpha\in[0,1]$. For any $M>0$ and $f\in\mathcal F$ satisfying
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
gives $\xi_m\to0$: $\mathbb P(\kappa(X)>M)\to0$ as $M\uparrow\infty$, and $e_m(M)\to0$ with $m$ for fixed $M$. Taking $\alpha=0$ yields asymptotic consistency.

The crude bound $\kappa(x)\leq1/\Delta(x)$, choosing $M=1$ for $\operatorname{card}(\mathcal Y)=k=2$ and optimizing $M$ otherwise, lower bounds the promised $\xi_{m,k}$ by setting
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

Algebraic substitutions and manipulations: Appendix B.4.

###### Corollary 3.3.

Under the theorem’s conditions, every $f$ with $R(f)-R^\star\geq\xi_{m,k}$ satisfies
\[
R(f)-R^\star\leq
\frac{16}{\mathsf C_{\varphi,1}}
\bigl(R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star\bigr).
\]

The comparison requires fewer aggregated samples for fixed $f$ as $\alpha$ approaches $1$; at $\alpha=1$,
\[
m\geq
32\max\{\mathsf C_{\mathsf{MT}}^2,\mathsf C_{\mathsf{MT}}^4\}
\log\!\left(
\frac{4k(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})}
{\mathsf C_{\varphi,1}}
\right)
=O(\log k)
\]
gives $\xi_{m,k}=0$ and linear R2V. Lower-noise problems require less aggregation for stronger consistency.

Low noise permits a comparison written as
$\psi(R(f)-R^\star)\leq R_\varphi(f)-R^\star$, with $\psi$ linear over some positive-$\epsilon$ range; linear growth is the strongest possible [34, 32]. Strict comparison requirements may exclude practically convenient inconsistent surrogates [23, 34, 32]. Following Osokin et al. [34], a positive-range relaxation is used.

###### Definition 3.2.

The surrogate and aggregator have the property (U6A: level-$(\xi,\zeta)$ consistency) when some $\psi$ satisfies both
\[
\psi(\epsilon)\geq\zeta\epsilon\quad(\epsilon\geq\xi),
\qquad
\psi(R(f)-R^\star)\leq
R_{\varphi,A}(f)-R_{\varphi,A}^\star.
\]
Thus only risks above the threshold require linear growth, retaining a “good enough” criterion; minimal assumptions permit levels $(o_m(1),\zeta)$ even for a non-Q7D surrogate.

The quantity
\[
\Delta(x)\coloneqq
\min_{\mathsf d(s)\notin y^\star(x)}
\delta_\ell(s,x)
\tag{8}
\]
is the incorrect-prediction gap (F8Q: minimal excess conditional risk), equaling $|2P(Y=1\mid X=x)-1|$ in binary classification, with estimation expected harder nearer zero. Adapting classical assumptions [25], also Bartlett et al. [4], the pair (N6Y: Mammen-Tsybakov noise conditions) uses a positive constant $\mathsf C_{\mathsf{MT}}$ that may change:
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
Here $\alpha\in[0,1]$ is the noise exponent and $\beta\in[0,\infty]$; both hold trivially at $\alpha=\beta=0$, and, as in the binary result [4, Thm. 3], are equivalent under $\beta=\alpha/(1-\alpha)$ (Appendix B.2).

Mao et al. [29] use exactly the same noise statistic for multiclass $H$-consistency, imposing surrogate and hypothesis-class structure for a global convex lower bound. Minimal structure instead requires the ratio (B2Z: noise condition number), defined by
\[
\kappa(x)\coloneqq
\frac{\max_{\mathsf d(s)\neq y^\star(x)}\delta_\ell(s,x)}
{\min_{\mathsf d(s)\neq y^\star(x)}\delta_\ell(s,x)}.
\tag{9}
\]
For $\mathsf d(s)\neq y^\star(x)$,
$\Delta(x)\geq\delta_\ell(s,x)/\kappa(x)$; binary classification has $\kappa(x)=1$ provided $\mathbb P(\Delta(X)>0)=1$.

The auxiliary quantity (V5R: error function), used with $\kappa(x)=t$ to express likely correctness of $A_m$ when it is much smaller than one, is
\[
e_m(t)\coloneqq
t\sqrt{\frac2m
\log\!\left(
\frac{4k(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})}
{\mathsf C_{\varphi,1}}
\right)}.
\tag{10}
\]
Theorem 1 is proved in Appendix B.3; its coarse bounds reflect weak identifiability. Tighter $\ell$–$\varphi$ relationships, making J9P correct or sufficiently correct for $\varphi$, should strengthen bounds; near-consistency without aggregation could link smaller initial inconsistency to better $m$-dependence. These refinements are not pursued.

### 3.3 Surrogate consistency with T4K

The instances below assume noise exponent $\alpha\in[0,1]$, without loss of generality since $\alpha=0$ always satisfies (Nα); proofs for each are in Appendix B.5.

#### 3.3.1 Binary classification, nonsmooth surrogate

###### Lemma 3.1.

For every $\delta>0$, $\varphi$ is $(\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})$-feasible with
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

Here $\varphi(f(x),y)=\phi(yf(x))$, $\phi$ is convex,
$\partial\phi(0)\subset(-\infty,0)$, and
$\lim_{t\to\infty}\phi(t)=0$. For convex binary margin surrogates, Q7D and R2V over all measurable functions hold exactly when $\phi'(0)<0$ [4]. Aggregation can make an otherwise inconsistent $\phi$ consistent; this is a somewhat trivial thought experiment, not a surrogate one would use.

#### 3.3.2 Bipartite matching

###### Lemma 3.2.

On $2N$ vertices, the matching surrogate is W3C with
\[
\mathsf C_{\varphi,1}=\frac1N,\qquad
\mathsf C_{\varphi,2}=2.
\]
Despite factorial label cardinality, aggregation yields consistency once its count exceeds $O(N\log N)$. For $k\geq3$, measurable $f:\mathcal X\to\mathbb R^d$ obeying
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

For structured spaces, typically large, an embedding $v:\mathcal Y\to\mathbb R^d$ represents structural information [33]. With
$\mathsf d(s)=\operatorname{argmax}_{y\in\mathcal Y}\langle s,v(y)\rangle$ and $\ell:\mathcal Y\times\mathcal Y\to\mathbb R_+$ satisfying $\ell(y,y)=0$, the loss (A3J: maximum-margin surrogate, also generalized hinge loss and structured hinge loss) is
\[
\varphi(s,y)=
\max_{\hat y\in\mathcal Y}
\left(\ell(\hat y,y)+
\langle v(\hat y)-v(y),s\rangle\right).
\tag{11}
\]
It is generally inconsistent except for certain low-noise cases [34, 32], and originates in [42, 44, 19].

For the matching instance, $G=(V,E)$ has $V=V_1\cup V_2$,
$V_1=\{1,\ldots,N\}$, $V_2=\{N+1,\ldots,2N\}$; each of the $N$ edges joins exactly one unique node in each side. All matchings form $\mathcal Y$, so $k=\operatorname{card}(\mathcal Y)=N!$. The edge-index embedding and mistaken-edge loss are
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

#### 3.3.3 Structured prediction

###### Lemma 3.3.

A3J is $(\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})$-W3C with
\[
\mathsf C_{\varphi,1}=
\min_{\hat y\neq y}\ell(\hat y,y),
\qquad
\mathsf C_{\varphi,2}=
\max_{y\in\mathcal Y}\tau(y)+1.
\]
In particular, $v(y)\in\{0,1\}^d$ and
$\ell(\hat y,y)=\frac1{2d}\|v(\hat y)-v(y)\|_1$
give $\tau(y)=1$ for every $y$ and $\mathsf C_{\varphi,2}=2$.

Nontrivial consistency holds once $m\geq\log\operatorname{card}(\mathcal Y)$, as in the binary matching case. With $0\leq\ell(\cdot,\cdot)\leq1$, every measurable $f:\mathcal X\to\mathbb R^d$ satisfying
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

The general structured decoder is assumed capable of picking every class: for each $y$,
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

## 4 Robustness and consistency for models

Restricted, often parametric $\mathcal F$ replaces unrestricted measurable functions. Well-specification makes consistency straightforward but is brittle and unrealistic; surrogate consistency on subsets of measurable functions is long-standing [12, 31]. Recent $H$-consistency bounds connect excess target risk and surrogate regret within a regret hypothesis class $H$ in diverse settings [2, 3, 26, 28, 29]. Those results are primarily surrogate-centric; the aggregation contribution is orthogonal, less dependent on a particular class and surrogate.

### 4.1 Binary consistency failure in finite dimensions

###### Proposition 4.

For every $\epsilon>0$ and nonzero $\theta^\star\in\mathbb R^d$, some distribution $P$ on $(X,Y)$ satisfies K7M along $\theta^\star$, yet every
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
This essential orthogonality occurs for any nonnegative loss $\phi$. Proof: Appendix D.1.

###### Corollary 4.1.

As a corollary to Theorem 2, if $P$ satisfies K7M along $\theta^\star$ and $\epsilon_m\to0$ as $m\to\infty$, the aggregated approximate minimizers satisfy
\[
R(f_{\theta_m})\to R(f_{\theta^\star}),
\qquad
\cos\angle(\theta_m,\theta^\star)\to1.
\]

The binary setting uses $\mathcal Y=\{\pm1\}$,
$\ell(\mathsf d(s),y)=1\{ys\leq0\}$, margin surrogate
$\varphi(s,y)=\phi(sy)$ for convex $\phi:\mathbb R\to\mathbb R_+$,
$\mathcal X=\mathbb R^d$, and
\[
\mathcal F=
\{f_\theta:f_\theta(x)=\langle\theta,x\rangle\}_{\theta\in\mathbb R^d}.
\]
Even inclusion of an optimal classifier does not preserve classical consistency. The inclusion requirement (K7M: optimal predictability) means that an $f\in\mathcal F$ satisfies
\[
\mathsf{sgn}(f(x))=
\mathsf{sgn}(P(Y=1\mid X=x)-1/2).
\tag{12}
\]
For a vector $\theta^\star$ realizing it,
\[
\mathsf{sgn}(\langle\theta^\star,x\rangle)
=
\mathsf{sgn}(P(Y=1\mid x)-\tfrac12);
\]
$f_{\theta^\star}$ minimizes $R$ over every measurable function, and $P$ has K7M along $\theta^\star$.

Approximate minimizers mean
\[
\operatorname{\epsilon\textup{-argmin}}g
=
\operatorname{\epsilon\textup{-argmin}}_\theta g(\theta)
\coloneqq
\{\theta:g(\theta)\leq\inf_\theta g(\theta)+\epsilon\}.
\]
With independent repeated labels $Z=(Y_1,\ldots,Y_m)$ and T4K $A_m$, the relevant sequence is
\[
\theta_m\in
\operatorname{\epsilon_m\textup{-argmin}}_\theta
R_{\varphi,A_m}(f_\theta)
=
\operatorname{\epsilon_m\textup{-argmin}}_\theta
\mathbb E[
\phi(A_m(Y_1,\ldots,Y_m)f_\theta(X))].
\]
Thus linear restriction can make unaggregated surrogate minimization arbitrarily incorrect; aggregation retains asymptotic consistency.

### 4.2 Aggregation and restricted classes

###### Theorem 2.

Let a cone $\mathcal F$ satisfy K7M for $P$, and assume
$P(\Delta(X)>0)=1$. For $\epsilon_m\geq0$, $\epsilon_m\to0$, every sequence
\[
f_m\in
\operatorname{\epsilon_m\textup{-argmin}}_{f\in\mathcal F}
R_{\varphi,A_m}(f)
=
\operatorname{\epsilon_m\textup{-argmin}}_{f\in\mathcal F}
\mathbb E[
\varphi(f(X),A_m(Y_1,\ldots,Y_m))]
\]
satisfies $R(f_m)\to R^\star$. Proof: Appendix D.2.

Multiclass labels are $\mathcal Y=\{1,\ldots,k\}$, with zero-one
$\ell(y,y')=1\{y\neq y'\}$ and
$\mathsf d(s)=\operatorname{argmax}_{y\in[k]}s_y$. The linear-cone conditions are
\[
f(x)^T\mathbf1=0,\qquad
f\in\mathcal F,\ t>0\ \Longrightarrow\ tf\in\mathcal F;
\]
normalization does not affect decoding. The nonnegative surrogate
$\varphi:\mathbb R^k\times[k]\to\mathbb R_+$ is Q7D and satisfies
\[
\varphi(s,y)\to0
\quad\text{if}\quad
s_y-s_j\to+\infty\text{ for every }j\neq y.
\tag{13}
\]
The response space is $\mathcal Z=\mathcal Y^m$, aggregation is T4K, and the population maximizer
$y^\star(x)=\operatorname{argmax}_yP(Y=y\mid x)$ is unique.

Multiclass K7M requires an $f\in\mathcal F$, $f:\mathcal X\to\mathbb R^k$, with
\[
\operatorname{argmax}_y f_y(x)
\in\operatorname{argmax}_yP(Y=y\mid X=x)
\quad\text{for every }x.
\]
The broad cone result requires aggregation; even optimal-predictor-containing binary subclasses can otherwise have arbitrarily uninformative surrogates.

Examples of Q7D losses meeting the limiting condition are multiclass logistic
\[
\varphi(s,y)=\log\left(\sum_{j=1}^k e^{s_j-s_y}\right)
\]
and
\[
\varphi(s,y)=\sum_{i\neq y}\phi(s_y-s_i)
\]
for convex, nonincreasing $\phi$ with $\phi'(0)<0$ and $\inf_t\phi(t)=0$. Zhang [51, Thm. 8] establishes consistency of these losses over
$\mathcal F=\{f:\mathcal X\to\mathbb R^k:\mathbf1^Tf=0\}$.
Binary margin predictors correspond to $f(x)=(g(x),-g(x))$ for measurable $g$. For $\mathcal X=\mathbb R^d$, linear
$f(x)=(\langle\theta_1,x\rangle,\ldots,\langle\theta_k,x\rangle)$ with
$\sum_{i=1}^k\theta_i=0$ form a convex cone. Such examples typically require strong structure such as convexity without aggregation; the minimal limiting condition permits aggregated asymptotic consistency.

### 4.3 Finite-dimensional multiclass classification

###### Proposition 5.

Let $k\geq3$, $\Sigma=I$. For $Z\sim\mathsf N(0,I_{k-1})$, assume invertibility of
\[
M\mapsto DM\coloneqq
\mathbb E\!\left[
Z\bigl(
\nabla\sigma^{\rm lr}({\Theta^\star}^TZ)MZ
\bigr)^T
\right].
\]
Some $\epsilon_0>0$ then has the property that every
$\epsilon\in(0,\epsilon_0)$ admits $\mathcal T_\epsilon$ of Lebesgue measure at most $\epsilon$ with
\[
\Theta_1(\epsilon)/\|\Theta_1(\epsilon)\|
\neq
\Theta^\star/\|\Theta^\star\|.
\]

Proposition 5 proof: Appendix E.1.

###### Theorem 3.

Suppose $\Theta^\star=U^\star T^\star$, with orthogonal
$U^\star\in\mathbb R^{d\times(k-1)}$ and nonsingular
$T^\star\in\mathbb R^{(k-1)\times(k-1)}$. For each $m$ there exists
$T_m\in\mathbb R^{(k-1)\times(k-1)}$ with $\Theta_m=U^\star T_m$, and
\[
\|T_m\|\to\infty,\qquad
T_m/\|T_m\|\to T^\star/\|T^\star\|
\quad(m\to\infty).
\]

Theorem 3 proof: Appendix E.2.

Pair-based surrogate risks can be inconsistent even with K7M and a linear model correct except on an $\epsilon$-fraction of data. T4K recovers the optimal linear predictor asymptotically. For $k\geq3$, arbitrarily slight true-link mis-specification defeats risk consistency; aggregation requires no true-link assumption beyond ordering consistency.

Here $\mathcal X=\mathbb R^d$, $\mathcal Y=[k]$,
$\Theta=[\theta_1~\cdots~\theta_{k-1}]\in\mathbb R^{d\times(k-1)}$, and
$Y\mid X=x\sim\mathsf{Cat}(p_\Theta(x))$, where
$p_\Theta(x)\in\mathbb R_+^k$, $\mathbf1^\top p_\Theta(x)=1$, and
\[
p_\Theta(x)=
\begin{bmatrix}
\sigma(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)\\
1-\mathbf1^\top
\sigma(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)
\end{bmatrix}.
\]
The link $\sigma:\mathbb R^{k-1}\to\mathbb R_+^{k-1}$ has
$\mathbf1^\top\sigma\leq1$; putting $t_k=0$ and
$\sigma_k(t)=1-\mathbf1^\top\sigma(t)$ imposes
\[
t_y=\max_{1\leq i\leq k}t_i
\quad\Longleftrightarrow\quad
\sigma_y(t)=\max_{1\leq i\leq k}\sigma_i(t).
\tag{14}
\]
Multiclass logistic regression uses
\[
\sigma_i^{\rm lr}(t)=
\frac{e^{t_i}}{1+e^{t_1}+\cdots+e^{t_{k-1}}}.
\]

The multilabeled dataset is
$\{(X_i;(Y_{1i},\cdots,Y_{im}))\}_{i=1}^n$, with
\[
Y_{ij}\mid X_i\stackrel{\rm iid}{\sim}
\mathsf{Cat}(p_{\Theta^\star}(X_i)).
\]
Arbitrarily tie-broken $Y_i^+=A_m(Y_{i1},\cdots,Y_{im})$ obeys
$Y_i^+\mid X_i\sim\mathsf{Cat}(p_{\Theta^\star,m}(X_i))$. If $\rho_m(t)$ is the T4K distribution for $m$ items with initial probabilities
$\sigma(t)\in\mathbb R_+^{k-1}$, then
\[
p_{\Theta,m}(x)=
\begin{bmatrix}
\rho_m(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)\\
1-\mathbf1^\top
\rho_m(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)
\end{bmatrix},
\]
and $\rho_m$ satisfies ordering consistency.

Fitted logistic loss, with $\theta_k=\mathbf0$, is
\[
\varphi(\Theta^\top x,y)=
-\langle\theta_y,x\rangle+
\log\left(1+\sum_{i=1}^{k-1}
\exp(\langle\theta_i,x\rangle)\right).
\tag{15}
\]
For $L_m(\Theta)=\mathbb E[\varphi(\Theta^\top X,Y_m^+)]$,
\[
\nabla_{\theta_i}\varphi(\Theta^\top x,y)=
-x(1\{y=i\}-\sigma_i^{\rm lr}(\Theta^\top x)),
\]
so its minimizer $\Theta_m$ satisfies
\[
\nabla_\Theta L_m(\Theta_m)=
\mathbb E\!\left[
X\left(
\sigma^{\rm lr}(\Theta_m^\top X)
-\rho_m({\Theta^\star}^\top X)
\right)^\top
\right]=0.
\]
Well-specified unaggregated models have standard statistical consistency and efficiency. For $m\geq1$, $k=2$, and $X\sim\mathsf N(0,I_d)$, Cheng et al. [7, Prop. 3] obtain $\Theta_m=t_m\Theta^\star$ for some $t_m>0$ even with a mis-specified link: the direction
$\Theta^\star/\|\Theta^\star\|_2$ suffices for consistent binary classification at every aggregation level.

For a fixed $\mathcal T_\epsilon\subset\mathbb R^{k-1}$ of Lebesgue measure $\epsilon$, set
\[
\sigma^\epsilon(t)=
\sigma^{\rm lr}(t)1\{t\notin\mathcal T_\epsilon\}
+\frac1k\mathbf1\,1\{t\in\mathcal T_\epsilon\}.
\]
The conditional distribution on $Y\in\{1,\ldots,k\}$ samples
$Y\sim\sigma^{\rm lr}(t)$ if $t\in\mathcal T_\epsilon$ and uniformly otherwise. The link satisfies ordering consistency and K7M. For $\epsilon>0$, define
\[
L_{m,\epsilon}(\Theta)\coloneqq
\mathbb E[
\mathbb E_{\sigma^\epsilon}[
\varphi(\Theta^\top X,Y_m^+)\mid X]],
\qquad
\Theta_m(\epsilon)=
\operatorname{argmin}_\Theta L_{m,\epsilon}(\Theta),
\]
the population logistic loss for $m$-label T4K when
$Y\mid X=x\sim\sigma^\epsilon({\Theta^\star}^\top x)$.
Although $\Theta_1(0)=\Theta^\star$, arbitrarily small positive corruption destroys unaggregated consistency.

For general ordering-consistent $\sigma$, the theorem’s definitions are
\[
L_{m,\sigma}(\Theta)=
\mathbb E[
\mathbb E_\sigma[
\varphi(\Theta^\top X,Y_m^+)\mid X]],
\qquad
\Theta_m=\operatorname{argmin}L_{m,\sigma}(\Theta).
\]
These give robustness under mis-specification whenever some link describes the $X$–$Y$ relationship; without aggregation, multiclass methods are generally inconsistent.

## 5 Numerical illustrations

Two experiments corroborate the theory: nonparametric amplification and finite-dimensional multiclass asymptotic consistency under link mis-specification.

##### Surrogate consistency amplification for D4X.

The aggregated curve changes from quadratic toward linear as $m\to\infty$, upgrading consistency; the general-theorem curves are conservative.

Binary margin $\varphi(f(x),y)=\phi(yf(x))$ uses the loss (D4X: truncated quadratic loss), defined by $\phi(\delta)=\max\{1-\delta,0\}^2$, whose exact unaggregated H8N is $\psi(\epsilon)=\epsilon^2$ [4, Example 2]. Taking $\delta=1$ in the binary lemma and no low-noise condition, $\alpha=0$, $\mathsf C_{\mathsf{MT}}=1$, gives
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
The parallel comparisons are
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

Synthetic data from a corrupted link exhibit asymptotic consistency as $m\to\infty$. Logistic loss is the finite-dimensional loss specified above; the piecewise link agrees with softmax near the origin and shrinks scores outside radius $r$:
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

Cheng, Asi and Duchi [7] discuss the collection debate: using all unaggregated labels appears statistically more efficient when models can represent all uncertainty; otherwise cleaning appears more robust. Dorner and Hardt [11] favor more noisy rather than cleaned labels for validation comparisons of binary classifiers. The present results establish fundamental limitations of uncleaned supervised-label information and robustness gains from cleaning.

##### Finite $m$ and fundamental limits.

Many consistency results require $m\to\infty$; reliance on this limit remains unsatisfying. Ranking admits explicit finite-$m$ guarantees. A theory depending jointly on $(\ell,\varphi)$ and available count $m$ remains open: second- or third-order item comparisons, including powers $C_x^p$ suggested by Keener [20], may require increasing counts. Precisely separating problems requiring cleaning and aggregation from those not requiring them is a central challenge and a natural, important direction.

##### Fundamental limits of B2Z.

Beyond binary classification, $\kappa(X)$ may remain large even with low N6Y noise, $\alpha\approx1$; binary problems trivially have $\kappa(X)=1$. Minimal surrogate assumptions produce this behavior. Closer task–surrogate connections could better represent difficulty; explicit failure constructions and sharper fundamental limits could improve understanding of predictive-model fitting.

##### Mis-specified models.

The mis-specified-model results require K7M, a Bayes-optimal classifier in $\mathcal F$. Classical surrogate consistency can fail even there while aggregation succeeds. Beyond this restriction, Nguyen et al. [31] for binary classification and Duchi et al. [12] for multiclass classification show that joint inference of predictor $f$ and representation of $x$ forces a task-loss-dependent surrogate form. Their decision rules remain infinitely powerful; new approaches are needed.

##### Restricted-class comparison with aggregation.

Nonparametric comparison inequalities mitigate surrogate-induced regret; parametric results establish statistical consistency. Recent $H$-consistency work [2, 28, 27, 29] suggests extending aggregation comparisons to restricted classes. Mao et al. [27] establish rate-optimal quadratic $\psi$ growth for smooth convex loss; aggregation-based upgrades in $\mathcal H$-consistency remain of interest. An extension could unify parametric and nonparametric advantages and characterize how hypothesis-class “size” affects aggregation.
