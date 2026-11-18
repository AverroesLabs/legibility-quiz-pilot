# Some Robustness Properties of Label Cleaning

Chen Cheng

John Duchi

###### Abstract

Aggregated labels, including information distilled from noisy responses, permit robustness that is unavailable without cleaning. In particular, minimizing a typically convex surrogate loss with aggregated labels gives stronger task-risk consistency guarantees than procedures using raw labels. Aggregation also allows finite-dimensional classifiers to remain consistent under loss mis-specification that defeats standard uncertainty-modeling approaches. Perfectly specified statistical models favor incorporating all information for efficiency. However, even slight loss mis-specification destroys standard consistency, whereas aggregation retains convergence to optimal classifiers. Thus collection, fitting, and prediction jointly admit robustness through refinement of noisy signals.

## 1 Introduction

Aggregation permits consistent optimal prediction under a surrogate loss rather than the task loss in settings where unaggregated estimation cannot do so, including finite-dimensional statistical problems. Constructions involving linear estimators in ranking, binary classification, and multiclass classification separate procedures restricted to input–label pairs $(X,Y)$ from procedures that aggregate labels. The nonparametric results combine conditions on aggregation count and noise, permit approximately linear comparison inequalities under weak surrogate conditions, and extend consistency beyond unrestricted classes of measurable hypotheses.

The conventional supervised pipeline collects a dataset $(X_i,Y_i)_{i=1}^n$ and fits a model [17]. However, this description suppresses the cleaning, filtering, repeated crowdsourced labeling, and denoising required by modern datasets [9, 22, 39]. Work on crowdsourced “gold standard” labels [8, 48, 47, 46, 37] concerns this machinery. The present concern is the fundamental, necessary improvement in fitted-model consistency that this machinery can provide.

Important contributions to surrogate-risk consistency theory date to the 2000s [52, 24, 40]. In particular, Bartlett, Jordan and McAuliffe [4] characterize convex-surrogate consistency for binary zero-one classification. Subsequent settings include multi-label classification [50, 43, 16, 49, 1], ranking [13, 14, 36], structured prediction [34, 6, 32], ordinal regression [35], restricted hypothesis classes [2, 3, 28, 29], and general theory [40]. Analyses based on input–label pairs admit empirical-process theory [45, 5, 21, 4], but do not include the aggregation machinery of contemporary dataset creation. The questions considered here are therefore whether aggregation is needed at all, whether pair-based surrogate consistency suffices, and in what sense aggregation improves consistency.

## 2 Preliminaries

We retain two regimes for the hypothesis class $\mathcal F$: an unrestricted class of Borel measurable functions, with the classical notion of **Fisher consistency**, denoted Q7D, and a parametric, potentially mis-specified class. Aggregation improves consistency in each regime.

Let $(X_i,Y_i)_{i=1}^n$ be independent and identically distributed, abbreviated i.i.d., with distribution $P$ on the input and output spaces $\mathcal X\times\mathcal Y$. A score function is a map $f:\mathcal X\to\mathbb R^d$, where $d\geq1$. A decoder $\mathsf d:\mathbb R^d\to\mathcal Y$ turns a score into a prediction:
\[
\widehat y=\mathsf d\circ f(x).
\]
Here $\mathsf d\circ f$ denotes composition of the decoder and score function.

The distribution-dependent task criterion is
\[
R(f)\coloneqq
\mathbb E_P\!\left[\ell(\mathsf d\circ f(X),P(\cdot\mid X))\right].
\tag{2}
\]
Here $P(\cdot\mid X)$ is the conditional output distribution, and $\ell$ evaluates a prediction against that conditional distribution. The minimizers of this criterion frequently coincide with those of the original problem (2); allowing the loss to depend on the conditional distribution permits greater sophistication. For example, when $\mathcal Y=\{1,\ldots,k\}$ and
\[
\ell(\widehat y,P)=\sum_y P(Y=y)1\{\widehat y\neq y\},
\]
the criterion is the standard zero-one error rate. The notation $1\{\cdot\}$ denotes the indicator of the condition inside braces.

In the instantaneous formulation, the loss is instead a map $\ell:\mathcal Y\times\mathcal Y\to\mathbb R_+$, where $\mathbb R_+$ denotes the nonnegative real numbers. This formulation minimizes, over $f\in\mathcal F$,
\[
R(f):=\mathbb E_P[\ell(\mathsf d\circ f(X),Y)].
\tag{1}
\]
For binary classification, $d=1$, the decoder is $\mathsf d(s)=\mathsf{sgn}(s)$, and
\[
\ell(y,y')=1\{yy'\leq0\},
\qquad
R(f)=\mathbb P(Yf(X)\leq0).
\]
Task losses may be nonsmooth, nonconvex, and uninformative for optimization. For example, the zero-one landscape is flat almost everywhere, precluding even practical first-order optimization. An easier surrogate loss $\varphi:\mathbb R^d\times\mathcal Y\to\mathbb R$ therefore replaces the task loss, giving the surrogate risk
\[
R_\varphi(f):=\mathbb E_P[\varphi(f(X),Y)].
\]
Because the surrogate replaces the task loss, this replacement requires consistency with the task criterion.

### 2.1 Label aggregation

Let $Z\in\mathcal Z$ be a replacement for the ordinary response, let $\mathcal A$ be an aggregation space, and let $A:\mathcal Z\to\mathcal A$ be an aggregation map. For a nonnegative surrogate $\varphi:\mathbb R^d\times\mathcal A\to\mathbb R_+$, define the **aggregated surrogate risk**, denoted P9S, by
\[
R_{\varphi,A}(f):=\mathbb E[\varphi(f(X),A(Z))].
\tag{3}
\]
Minimization of the aggregated surrogate risk is assessed against minimization of the actual task risk. Thus the surrogate uses $A(Z)$ rather than an ordinary response $Y_i$ from a pair $(X_i,Y_i)$.

Repeated labeling takes
\[
Z=(Y_1,\ldots,Y_m)\in\mathcal Y^m,
\qquad
Y_i\mid X=x\stackrel{\rm iid}{\sim}P_{Y\mid X=x},
\]
where $m$ is the repetition count. An aggregator can then perform empirical task minimization:
\[
A(Z)=A(\{Y_1,\ldots,Y_m\})
=\mathop{\rm argmin}_{y\in\mathcal Y}
\sum_{l=1}^m\ell(y,Y_l).
\]
For $\ell(\widehat y,y)=1\{y\neq\widehat y\}$, this aggregator is precisely **majority vote**, denoted T4K. General task losses $\ell$ permit more abstract aggregation procedures. We write $A_m(Z)$ when the repetition count needs to be explicit. Mao et al. [28] use a similar multilabeled structure for multilabel losses, rather than for the aggregation pipeline considered here.

A rough $K$-nearest-neighbor construction provides another aggregator. In this construction, $X$ is supplied with its label $Y$ and an additional independent draw
\[
(X_i,Y_i)_{i=1}^m\stackrel{\rm iid}{\sim}P,
\qquad
Z=(Y,(X_i,Y_i)_{i=1}^m).
\]
Given a metric $\operatorname{dist}:\mathcal X\times\mathcal X\to\mathbb R_+$, order the additional inputs by
\[
\operatorname{dist}(X,X_{(1)})\leq\cdots\leq
\operatorname{dist}(X,X_{(m)}),\qquad X_{(0)}=X.
\]
The notation $Y_{(l)}$ denotes the label paired with $X_{(l)}$, including $Y_{(0)}=Y$. For $K\geq0$, take
\[
A_{m,K}(Z)\coloneqq
\mathop{\rm argmin}_{y\in\mathcal Y}
\sum_{l=0}^K\ell(y,Y_{(l)}).
\]
Although this treatment uses repeated sampling at the population level, the treatment also extends to aggregation from one sample $(X_i,Y_i)_{i=1}^n$; see Appendix C.

## 3 Surrogate consistency

The standard framework [40] takes $\mathcal F$ to contain all Borel measurable functions $f:\mathcal X\to\mathbb R^d$. Define the optimal task and surrogate risks by
\[
R^\star:=\inf_{f\in\mathcal F}R(f),
\qquad
R_\varphi^\star:=\inf_{f\in\mathcal F}R_\varphi(f).
\]
For all distributions $P$, the framework considers two requirements [40, 4, 52]:

(i) Every sequence $f_n\in\mathcal F$ with
$R_\varphi(f_n)\to R_\varphi^\star$ satisfies $R(f_n)\to R^\star$. This is the Fisher-consistency requirement.

(ii) There is a nondecreasing function $\psi:\mathbb R_+\to\mathbb R_+$, positive at every $\epsilon>0$, such that
\[
\psi(R(f)-R^\star)\leq R_\varphi(f)-R_\varphi^\star
\quad\text{for every }f\in\mathcal F.
\]
This quantitative requirement is the **uniform comparison inequality**, denoted R2V. The uniform comparison inequality is stronger than requirement (i), and need not hold. However, requirements (i) and (ii) coincide for convex margin-based binary surrogates [4].

To state the conditional characterization, let $s\in\mathbb R^d$ be a score vector. Define the conditional task risk, conditional surrogate risk, and their excesses by
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
Thus $\delta_\ell(s,x)$ and $\delta_\varphi(s,x)$ measure conditional risk above the best conditional score.

The canonical excess-risk construction is the **calibration function**, denoted H8N. Its pointwise and uniform forms are
\[
\psi(\epsilon,x)\coloneqq
\inf_{s\in\mathbb R^d}
\{\delta_\varphi(s,x)\mid\delta_\ell(s,x)\geq\epsilon\},
\qquad
\overline\psi(\epsilon)\coloneqq
\inf_{x\in\mathcal X}\psi(\epsilon,x).
\]
The pointwise function therefore asks for the smallest surrogate excess among scores whose task excess is at least $\epsilon$, while the uniform form takes the infimum over inputs.

###### Corollary 3.1.

Positivity of $\psi(\epsilon,x)$ for all $x\in\mathcal X$ and $\epsilon>0$ is equivalent to $\varphi$ being Fisher consistent for $\ell$. If $\psi$ is the Fenchel biconjugate of $\overline\psi$, then
\[
\overline\psi(\epsilon)>0\quad\Longleftrightarrow\quad
\psi(\epsilon)>0,
\qquad
\psi(R(f)-R^\star)\leq R_\varphi(f)-R_\varphi^\star
\]
for every measurable $f$.

The equivalence and inequality follow from Zhang [51, Prop. 25] and Steinwart [40, Thm. 2.8 and Lemma 2.9].

A Fisher-consistent surrogate for the general task would make single input–label pairs sufficient for consistent estimation. Consequently, such a surrogate would simplify collection and admit the full empirical-process toolkit. However, such pair-only results are generally impossible.

### 3.1 Q7D failure without aggregation: ranking

Pairwise-comparison ranking [20, 15, 14, 30] has response space
\[
\mathcal Y=\{(i,j):i\neq j,\ 1\leq i,j\leq k\},
\]
where $k$ is the number of items and the response $(i,j)$ says that item $i$ outranks item $j$. For each input $x$, population aggregation transforms the conditional probabilities
\[
p_{ij}=P(Y=(i,j)\mid x)
\]
into an ordering. A simple comparison scheme [20, cf.] uses the matrix $C_x$ with entries
\[
(C_x)_{ii}=0,\qquad
(C_x)_{ij}=\frac{p_{ij}}{\sum_{l\neq j}p_{lj}}
\quad(i\neq j),\qquad
0/0=1/(k-1).
\]
Write $\mathbf1$ for the all-ones vector. Then $C_x\in\mathbb R_+^{k\times k}$ is stochastic, with
\[
C_x^T\mathbf1=\mathbf1,
\]
and $C_x\mathbf1$ measures preference frequency for each item. Higher matrix powers $C_x^p\mathbf1$, Perron vectors [20], and more sophisticated schemes admit similar negative results.

Let $e_i$ be the $i$th coordinate vector. Absorbing decoding into the loss gives
\[
\ell(s,C)\coloneqq
\max_{i<j}1\!\left\{
(s_i-s_j)(e_i-e_j)^TC\mathbf1\leq0,\ 
(e_i-e_j)^TC\mathbf1\neq0
\right\}.
\]
This loss penalizes disagreement between the ordering of the scores and the ordering induced by the matrix. The ranking risk is therefore
\[
R(f)\coloneqq
\mathbb P\!\left(f(X)\text{ and }C_X\mathbf1
\text{ order differently}\right).
\tag{4}
\]
Restriction to scores satisfying $s^T\mathbf1=0$ is minor, familiar in multiclass classification [51, 43], and natural because only ordering is decoded.

###### Proposition 1.

For the ranking task with $k\geq3$ outcomes, every $\varphi:\mathbb R^k\times\mathcal Y\to\mathbb R$ convex in its first argument fails Fisher consistency. The proof is in Appendix A.1.

For the aggregated construction, take multiple independent comparisons $Z=(Y_1,\ldots,Y_m)$ conditional on $X$. Write $[k]=\{1,\ldots,k\}$, and put
\[
m_{ij}=\sum_{y\in\mathcal Y^m}1\{y=(i,j)\},
\qquad m_j=\sum_{i=1}^k m_{ij}.
\]
The aggregator has outputs in $\mathbb R^k$ together with a special output $\star$:
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
Regression acts directly on the non-$\star$ outputs.

###### Proposition 2.

For $s,q\in\mathbb R^k$, set
\[
\varphi(s,q)=\|s-q\|_2^2,\qquad \varphi(s,\star)=0,
\]
where $\|\cdot\|_2$ is the Euclidean norm. With the aggregation just specified and $m\geq k$, $\varphi$ is Fisher consistent for the ranking risk. The proof is in Appendix A.2.

The expansion of the response space from $\mathcal Y$ to $\mathbb R^k$ separates Proposition 1 from Proposition 2. This expansion is unavailable to the convex single-comparison formulation, whereas regression on ordering frequencies yields the positive result.

### 3.2 Aggregation and stronger surrogate consistency

Aggregation both extends ordinary surrogate-risk theory and upgrades nominally inconsistent losses.

#### 3.2.1 Basic extensions

For an arbitrary aggregation map $A:\mathcal Z\to\mathcal A$, define the conditional aggregated surrogate risk and its excess by
\[
\begin{aligned}
R_{\varphi,A}(s\mid x)&\coloneqq
\mathbb E[\varphi(s,A(Z))\mid X=x],\\
\delta_{\varphi,A}(s,x)&:=
R_{\varphi,A}(s\mid x)-
\inf_{s\in\mathbb R^d}R_{\varphi,A}(s\mid x).
\end{aligned}
\]
The corresponding calibration objects are
\[
\overline\psi_A(\epsilon,x)\coloneqq
\inf_{s\in\mathbb R^d}
\{\delta_{\varphi,A}(s,x)\mid\delta_\ell(s,x)\geq\epsilon\},
\qquad
\overline\psi_A(\epsilon)\coloneqq
\inf_{s\in\mathbb R^d}\overline\psi_A(\epsilon,x).
\tag{5}
\]
Also write
\[
R_{\varphi,A}^\star:=\inf_{f\in\mathcal F}R_{\varphi,A}(f),
\qquad
\psi_A=(\overline\psi_A)^{**},
\]
where $^{**}$ denotes the Fenchel biconjugate.

###### Proposition 3.

Suppose an integrable function $b:\mathcal X\to\mathbb R_+$ satisfies
\[
\int b(x)\,dP(x)<\infty,\qquad
\delta_\ell(f(x),x)\leq b(x).
\]
Then $\varphi$ is Fisher consistent for the task risk exactly when
$\overline\psi_A(\epsilon,x)>0$ for every $x\in\mathcal X$ and $\epsilon>0$. With
$\psi_A=(\overline\psi_A)^{**}$,
\[
\psi_A(R(f)-R^\star)
\leq R_{\varphi,A}(f)-R_{\varphi,A}^\star.
\]

Appropriate measurability conditions are tacit and noncentral. Proposition 3 is a near-corollary of Steinwart [40, Thm. 2.8], with a completeness proof in Appendix B.1. Thus Proposition 3 preserves the classical guarantees, but does not by itself supply a particular improvement from aggregation.

#### 3.2.2 W3C and consistency

This subsection uses the **identifying-surrogate property**, denoted W3C. The property requires parameters $a_y$ and decoded score vectors $s_y$ for each class $y$: at parameter $a_y$, an incorrectly decoded score must incur a positive loss gap, while the losses of the selected score vectors at other class parameters remain within a finite gap of their infima. Definition 3.1 gives the precise inequalities and constants.

The regime is a finite label space with cardinality $\operatorname{card}(\mathcal Y)=k<\infty$ and instantaneous task loss
\[
R(f)=\mathbb E[\ell(f(X),Y)].
\]
The ranking construction belongs to this regime. In the ranking construction, however, $k$ denotes ranking length and the response-space cardinality is $k(k-1)$. Henceforth, $k$ denotes label-space cardinality, with this slight abuse of notation.

For parameters $\{a_y\}_{y\in\mathcal Y}$, empirical task minimization followed by parameter substitution defines **generalized majority vote**, denoted J9P. With arbitrary tie-breaking, the rule is
\[
A_m(y_1,\ldots,y_m)\coloneqq a_{\hat y},
\qquad
\hat y=\mathop{\rm argmin}_{y\in\mathcal Y}
\sum_{i=1}^m\ell(y,y_i).
\tag{7}
\]
Define the set of conditionally optimal predictions by
\[
y^\star(x)\coloneqq
\mathop{\rm argmin}_{y\in\mathcal Y}
\mathbb E[\ell(y,Y)\mid X=x].
\]
For i.i.d. $Y_i$ and finite $\mathcal Y$, a random $M<\infty$ necessarily exists such that every $m\geq M$ satisfies
\[
\mathop{\rm argmin}_{y\in\mathcal Y}
\left\{\sum_{i=1}^m\ell(y,Y_i)\right\}
\subset
y^\star(x).
\]
Consequently, the generalized-majority-vote strategy supplies asymptotic consistency under minimal identifiability for any such task loss.

###### Corollary 3.2.

For $m=m(n)\to\infty$ and an identifying surrogate $\varphi$,
\[
R_{\varphi,A_m}(f_n)-R_{\varphi,A_m}^\star\to0
\quad\Longrightarrow\quad R(f_n)-R^\star\to0.
\]

###### Definition 3.1.

A surrogate $\varphi:\mathbb R^d\times\mathcal A\to\mathbb R$ is an **identifying surrogate** with constants
\[
0<\mathsf C_{\varphi,1}\leq\mathsf C_{\varphi,2}<\infty
\]
when there exist parameters $\{a_y\}_{y\in\mathcal Y}\subset\mathcal A$ and vectors $\{s_y\}_{y\in\mathcal Y}\subset\mathbb R^d$ such that $\mathsf d(s_y)=y$ and, for all $y\neq y'$,
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
The first inequality makes a minimizer at parameter $a_y$ identify $y$. A finite lower bound on $\varphi(\cdot,a)$ supplies finite $\mathsf C_{\varphi,2}$ for the second inequality. The definition imposes neither convexity, unaggregated consistency with $\mathcal A=\mathcal Y$ and $Z=Y$, nor a decoder requirement such as pointwise $\operatorname{argmax}$.

For example, binary hinge loss
\[
\varphi(s,a)=\max\{1-sa,0\}
\]
on $\mathcal A=\mathcal Y=\{\pm1\}$ uses $a_y=s_y=y$. The relevant values are
\[
\varphi(s_1,a_1)=\varphi(s_{-1},a_{-1})=0,\qquad
\inf_{sa\leq0}\varphi(s,a)=1,
\]
and
\[
\varphi(s_1,a_{-1})=\varphi(s_{-1},a_1)=2.
\]
Therefore the identifying-surrogate constants for binary hinge loss are $(1,2)$.

#### 3.2.3 W3C and consistency amplification

We first specify the comparison and noise quantities used in the amplification result. A comparison has **level-$(\xi,\zeta)$ consistency**, denoted U6A, when a comparison function grows at least linearly with slope $\zeta$ for excess risks at or above a threshold $\xi$. Definition 3.2 below states this property formally.

Low noise permits a comparison written as
\[
\psi(R(f)-R^\star)\leq R_\varphi(f)-R^\star,
\]
with $\psi$ linear over some positive-$\epsilon$ range. Linear growth is the strongest possible [34, 32]. However, strict comparison requirements may exclude practically convenient inconsistent surrogates [23, 34, 32]. Following Osokin et al. [34], we therefore use a positive-range relaxation.

Define the incorrect-prediction gap, or **minimal excess conditional risk**, denoted F8Q, by
\[
\Delta(x)\coloneqq
\min_{\mathsf d(s)\notin y^\star(x)}
\delta_\ell(s,x).
\tag{8}
\]
In binary classification,
\[
\Delta(x)=|2P(Y=1\mid X=x)-1|.
\]
Estimation is expected to be harder when this gap is nearer zero.

Adapting classical assumptions [25], and also Bartlett et al. [4], the **Mammen-Tsybakov noise conditions**, denoted N6Y, use a positive constant $\mathsf C_{\mathsf{MT}}$ that may change:
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
Here $f^\star$ is an optimal task-risk predictor, $\alpha\in[0,1]$ is the noise exponent, and $\beta\in[0,\infty]$. Both conditions hold trivially at $\alpha=\beta=0$. As in the binary result [4, Thm. 3], the conditions are equivalent under
\[
\beta=\alpha/(1-\alpha);
\]
see Appendix B.2.

Mao et al. [29] use exactly the same noise statistic for multiclass $H$-consistency, where $H$ denotes a hypothesis class. Their analysis imposes structure on the surrogate and hypothesis class to obtain a global convex lower bound. Under minimal structure, we instead require the **noise condition number**, denoted B2Z:
\[
\kappa(x)\coloneqq
\frac{\max_{\mathsf d(s)\neq y^\star(x)}\delta_\ell(s,x)}
{\min_{\mathsf d(s)\neq y^\star(x)}\delta_\ell(s,x)}.
\tag{9}
\]
For $\mathsf d(s)\neq y^\star(x)$,
\[
\Delta(x)\geq\delta_\ell(s,x)/\kappa(x).
\]
Binary classification has $\kappa(x)=1$ provided $\mathbb P(\Delta(X)>0)=1$.

Finally, define the auxiliary **error function**, denoted V5R, by
\[
e_m(t)\coloneqq
t\sqrt{\frac2m
\log\!\left(
\frac{4k(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})}
{\mathsf C_{\varphi,1}}
\right)}.
\tag{10}
\]
With $\kappa(x)=t$, this error function expresses likely correctness of $A_m$ when the error function is much smaller than one.

###### Theorem 1.

Let $\varphi$ be an identifying surrogate with constants
$(\mathsf C_{\varphi,1},\mathsf C_{\varphi,2})$ and parameters $\{a_y\}_{y\in\mathcal Y}$. Let $A_m$ be generalized majority vote. Assume $0\leq\ell\leq1$ and condition (Nα), with $\alpha\in[0,1]$. For any $M>0$ and $f\in\mathcal F$ satisfying
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
Define the optimized threshold by
\[
\xi_m\coloneqq
\inf_M\left\{
2\mathbb P(\kappa(X)>M)+
(4\mathsf C_{\mathsf{MT}}e_m(M))^{\frac1{1-\alpha}}
\right\}.
\]
Then $\xi_m\to0$, because $\mathbb P(\kappa(X)>M)\to0$ as $M\uparrow\infty$, while $e_m(M)\to0$ with $m$ for each fixed $M$. Taking $\alpha=0$ therefore yields asymptotic consistency.

The crude bound $\kappa(x)\leq1/\Delta(x)$, together with the choice $M=1$ for $\operatorname{card}(\mathcal Y)=k=2$ and optimization over $M$ otherwise, lower bounds the promised $\xi_{m,k}$ by setting
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
The algebraic substitutions and manipulations are in Appendix B.4.

###### Corollary 3.3.

Under the conditions of Theorem 1, every $f$ with $R(f)-R^\star\geq\xi_{m,k}$ satisfies
\[
R(f)-R^\star\leq
\frac{16}{\mathsf C_{\varphi,1}}
\bigl(R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star\bigr).
\]

For a fixed $f$, the comparison requires fewer aggregated samples as $\alpha$ approaches $1$. At $\alpha=1$, the condition
\[
m\geq
32\max\{\mathsf C_{\mathsf{MT}}^2,\mathsf C_{\mathsf{MT}}^4\}
\log\!\left(
\frac{4k(\mathsf C_{\varphi,1}+\mathsf C_{\varphi,2})}
{\mathsf C_{\varphi,1}}
\right)
=O(\log k)
\]
gives $\xi_{m,k}=0$ and a linear uniform comparison inequality. Thus lower-noise problems require less aggregation for stronger consistency.

###### Definition 3.2.

The surrogate and aggregator have **level-$(\xi,\zeta)$ consistency** when some $\psi$ satisfies both
\[
\psi(\epsilon)\geq\zeta\epsilon\quad(\epsilon\geq\xi),
\qquad
\psi(R(f)-R^\star)\leq
R_{\varphi,A}(f)-R_{\varphi,A}^\star.
\]
Only risks above the threshold $\xi$ therefore require linear growth. This relaxation retains a “good enough” criterion: minimal assumptions permit levels $(o_m(1),\zeta)$, with a threshold tending to zero as $m$ increases, even for a surrogate that is not Fisher consistent.

Theorem 1 is proved in Appendix B.3. The coarse bounds in Theorem 1 reflect weak identifiability. Tighter relationships between $\ell$ and $\varphi$, making generalized majority vote correct or sufficiently correct for $\varphi$, should strengthen the bounds. Near-consistency without aggregation could also link smaller initial inconsistency to better dependence on $m$. These refinements are not pursued.

### 3.3 Surrogate consistency with T4K

The following instances use majority vote and assume noise exponent $\alpha\in[0,1]$. This assumption is without loss of generality because $\alpha=0$ always satisfies condition (Nα). Proofs for each instance are in Appendix B.5.

#### 3.3.1 Binary classification, nonsmooth surrogate

Consider a binary margin surrogate
\[
\varphi(f(x),y)=\phi(yf(x)),
\]
where $\phi$ is convex,
\[
\partial\phi(0)\subset(-\infty,0),
\qquad
\lim_{t\to\infty}\phi(t)=0.
\]
Here $\partial\phi(0)$ denotes the subdifferential at zero.

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

For convex binary margin surrogates, Fisher consistency and a uniform comparison inequality over all measurable functions hold exactly when $\phi'(0)<0$ [4]. Aggregation can make an otherwise inconsistent $\phi$ consistent. This is a somewhat trivial thought experiment, rather than a surrogate one would use.

#### 3.3.2 Bipartite matching

For structured spaces, which are typically large, an embedding $v:\mathcal Y\to\mathbb R^d$ represents structural information [33]. With decoder
\[
\mathsf d(s)=\operatorname{argmax}_{y\in\mathcal Y}\langle s,v(y)\rangle
\]
and task loss $\ell:\mathcal Y\times\mathcal Y\to\mathbb R_+$ satisfying $\ell(y,y)=0$, define the **maximum-margin surrogate**, denoted A3J and also called generalized hinge loss or structured hinge loss, by
\[
\varphi(s,y)=
\max_{\hat y\in\mathcal Y}
\left(\ell(\hat y,y)+
\langle v(\hat y)-v(y),s\rangle\right).
\tag{11}
\]
This surrogate originates in [42, 44, 19] and is generally inconsistent except for certain low-noise cases [34, 32].

For the matching instance, let $G=(V,E)$ be a graph with vertex set
\[
V=V_1\cup V_2,\qquad
V_1=\{1,\ldots,N\},\qquad
V_2=\{N+1,\ldots,2N\}.
\]
Each of the $N$ edges joins exactly one unique node on each side. The label space $\mathcal Y$ consists of all matchings, so
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
Here $\|\cdot\|_1$ is the sum of absolute coordinate values.

###### Lemma 3.2.

On $2N$ vertices, the matching surrogate is identifying with
\[
\mathsf C_{\varphi,1}=\frac1N,\qquad
\mathsf C_{\varphi,2}=2.
\]
Despite the factorial label cardinality $k=N!$, aggregation yields consistency once the aggregation count exceeds $O(N\log N)$. For $k\geq3$, measurable $f:\mathcal X\to\mathbb R^d$ obeying
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

#### 3.3.3 Structured prediction

For the general structured maximum-margin surrogate, assume that the decoder can pick every class. Precisely, for each $y$, the set
\[
\mathcal S(y)\coloneqq
\{s:\langle v(y),s\rangle>
\langle v(\hat y),s\rangle
\text{ for }\hat y\neq y\}
\]
is nonempty. Define the identifiability gap by
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

###### Lemma 3.3.

The maximum-margin surrogate is identifying with constants
\[
\mathsf C_{\varphi,1}=
\min_{\hat y\neq y}\ell(\hat y,y),
\qquad
\mathsf C_{\varphi,2}=
\max_{y\in\mathcal Y}\tau(y)+1.
\]
In particular, the choices
\[
v(y)\in\{0,1\}^d,
\qquad
\ell(\hat y,y)=\frac1{2d}\|v(\hat y)-v(y)\|_1
\]
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

## 4 Robustness and consistency for models

We now replace the unrestricted class of measurable functions by a restricted, often parametric hypothesis class $\mathcal F$. Well-specification makes consistency straightforward, but well-specification is brittle and unrealistic. Surrogate consistency on subsets of measurable functions is a long-standing question [12, 31]. Recent $H$-consistency bounds connect excess target risk and surrogate regret within a regret hypothesis class $H$ in diverse settings [2, 3, 26, 28, 29]. Those results are primarily surrogate-centric. In contrast, the aggregation contribution is orthogonal to those results and depends less on a particular hypothesis class and surrogate.

### 4.1 Binary consistency failure in finite dimensions

The binary setting uses $\mathcal Y=\{\pm1\}$, task loss
\[
\ell(\mathsf d(s),y)=1\{ys\leq0\},
\]
margin surrogate
\[
\varphi(s,y)=\phi(sy)
\]
for convex $\phi:\mathbb R\to\mathbb R_+$, input space $\mathcal X=\mathbb R^d$, and linear hypothesis class
\[
\mathcal F=
\{f_\theta:f_\theta(x)=\langle\theta,x\rangle\}_{\theta\in\mathbb R^d}.
\]

Even including an optimal classifier in this class does not preserve classical consistency. The inclusion requirement is **optimal predictability**, denoted K7M: some $f\in\mathcal F$ satisfies
\[
\mathsf{sgn}(f(x))=
\mathsf{sgn}(P(Y=1\mid X=x)-1/2).
\tag{12}
\]
For a vector $\theta^\star$ realizing optimal predictability,
\[
\mathsf{sgn}(\langle\theta^\star,x\rangle)
=
\mathsf{sgn}(P(Y=1\mid x)-\tfrac12).
\]
Then $f_{\theta^\star}$ minimizes $R$ over every measurable function, and we say that $P$ has optimal predictability along $\theta^\star$.

###### Proposition 4.

For every $\epsilon>0$ and nonzero $\theta^\star\in\mathbb R^d$, some distribution $P$ on $(X,Y)$ satisfies optimal predictability along $\theta^\star$, yet every
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
Thus the surrogate minimizer can be essentially orthogonal to the optimal direction, with a strictly worse task risk. This essential orthogonality occurs for any nonnegative loss $\phi$. The proof is in Appendix D.1.

To state the contrasting aggregation result, define approximate minimizers by
\[
\operatorname{\epsilon\textup{-argmin}}g
=
\operatorname{\epsilon\textup{-argmin}}_\theta g(\theta)
\coloneqq
\{\theta:g(\theta)\leq\inf_\theta g(\theta)+\epsilon\}.
\]
For independent repeated labels $Z=(Y_1,\ldots,Y_m)$ and majority vote $A_m$, consider
\[
\theta_m\in
\operatorname{\epsilon_m\textup{-argmin}}_\theta
R_{\varphi,A_m}(f_\theta)
=
\operatorname{\epsilon_m\textup{-argmin}}_\theta
\mathbb E[
\phi(A_m(Y_1,\ldots,Y_m)f_\theta(X))].
\]
Here $\epsilon_m$ is the allowed optimization error for aggregation count $m$.

###### Corollary 4.1.

As a corollary to Theorem 2, if $P$ satisfies optimal predictability along $\theta^\star$ and $\epsilon_m\to0$ as $m\to\infty$, the aggregated approximate minimizers satisfy
\[
R(f_{\theta_m})\to R(f_{\theta^\star}),
\qquad
\cos\angle(\theta_m,\theta^\star)\to1.
\]

Thus the linear restriction can make unaggregated surrogate minimization arbitrarily incorrect, as Proposition 4 shows, whereas aggregation retains asymptotic consistency, as Corollary 4.1 shows.

### 4.2 Aggregation and restricted classes

For the multiclass setting, let $\mathcal Y=\{1,\ldots,k\}$, use zero-one task loss
\[
\ell(y,y')=1\{y\neq y'\},
\]
and decode by
\[
\mathsf d(s)=\operatorname{argmax}_{y\in[k]}s_y.
\]
The linear-cone conditions on $\mathcal F$ are
\[
f(x)^T\mathbf1=0,\qquad
f\in\mathcal F,\ t>0\ \Longrightarrow\ tf\in\mathcal F.
\]
Thus the class is closed under positive scaling. The normalization $f(x)^T\mathbf1=0$ does not affect decoding.

Assume that the nonnegative surrogate
\[
\varphi:\mathbb R^k\times[k]\to\mathbb R_+
\]
is Fisher consistent and satisfies
\[
\varphi(s,y)\to0
\quad\text{if}\quad
s_y-s_j\to+\infty\text{ for every }j\neq y.
\tag{13}
\]
The response space is $\mathcal Z=\mathcal Y^m$, the aggregator is majority vote, and the population maximizer
\[
y^\star(x)=\operatorname{argmax}_yP(Y=y\mid x)
\]
is unique.

Multiclass optimal predictability requires an $f\in\mathcal F$, with $f:\mathcal X\to\mathbb R^k$, such that
\[
\operatorname{argmax}_y f_y(x)
\in\operatorname{argmax}_yP(Y=y\mid X=x)
\quad\text{for every }x.
\]

###### Theorem 2.

Let a cone $\mathcal F$ satisfy optimal predictability for $P$, and assume
\[
P(\Delta(X)>0)=1.
\]
For $\epsilon_m\geq0$ with $\epsilon_m\to0$, every sequence
\[
f_m\in
\operatorname{\epsilon_m\textup{-argmin}}_{f\in\mathcal F}
R_{\varphi,A_m}(f)
=
\operatorname{\epsilon_m\textup{-argmin}}_{f\in\mathcal F}
\mathbb E[
\varphi(f(X),A_m(Y_1,\ldots,Y_m))]
\]
satisfies $R(f_m)\to R^\star$. The proof is in Appendix D.2.

This broad cone result requires aggregation. Without aggregation, even binary subclasses containing an optimal predictor can have arbitrarily uninformative surrogates, as Proposition 4 demonstrates.

Examples of Fisher-consistent losses meeting the limiting condition (13) are multiclass logistic loss,
\[
\varphi(s,y)=\log\left(\sum_{j=1}^k e^{s_j-s_y}\right),
\]
and losses of the form
\[
\varphi(s,y)=\sum_{i\neq y}\phi(s_y-s_i)
\]
for convex, nonincreasing $\phi$ with $\phi'(0)<0$ and $\inf_t\phi(t)=0$. Zhang [51, Thm. 8] establishes consistency of these losses over
\[
\mathcal F=\{f:\mathcal X\to\mathbb R^k:\mathbf1^Tf=0\}.
\]

Binary margin predictors correspond to
\[
f(x)=(g(x),-g(x))
\]
for measurable $g$. For $\mathcal X=\mathbb R^d$, linear functions
\[
f(x)=(\langle\theta_1,x\rangle,\ldots,\langle\theta_k,x\rangle)
\]
with $\sum_{i=1}^k\theta_i=0$ form a convex cone. Without aggregation, such examples typically require strong structure such as convexity. With aggregation, the minimal limiting condition permits asymptotic consistency.

### 4.3 Finite-dimensional multiclass classification

Pair-based surrogate risks can be inconsistent even with optimal predictability and a linear model that is correct except on an $\epsilon$-fraction of data. In contrast, majority vote recovers the optimal linear predictor asymptotically. For $k\geq3$, arbitrarily slight mis-specification of the true link defeats risk consistency, whereas aggregation requires no true-link assumption beyond ordering consistency.

We first define the model, aggregation procedure, and fitted loss used in Proposition 5 and Theorem 3. Let
\[
\mathcal X=\mathbb R^d,\qquad
\mathcal Y=[k],\qquad
\Theta=[\theta_1~\cdots~\theta_{k-1}]\in\mathbb R^{d\times(k-1)}.
\]
The conditional label distribution is categorical:
\[
Y\mid X=x\sim\mathsf{Cat}(p_\Theta(x)),
\]
where $\mathsf{Cat}$ denotes the categorical distribution,
$p_\Theta(x)\in\mathbb R_+^k$, $\mathbf1^\top p_\Theta(x)=1$, and
\[
p_\Theta(x)=
\begin{bmatrix}
\sigma(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)\\
1-\mathbf1^\top
\sigma(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)
\end{bmatrix}.
\]
The link $\sigma:\mathbb R^{k-1}\to\mathbb R_+^{k-1}$ satisfies
$\mathbf1^\top\sigma\leq1$. For a score vector $t$, put $t_k=0$ and
\[
\sigma_k(t)=1-\mathbf1^\top\sigma(t).
\]
The ordering-consistency requirement is
\[
t_y=\max_{1\leq i\leq k}t_i
\quad\Longleftrightarrow\quad
\sigma_y(t)=\max_{1\leq i\leq k}\sigma_i(t).
\tag{14}
\]
Multiclass logistic regression uses the link
\[
\sigma_i^{\rm lr}(t)=
\frac{e^{t_i}}{1+e^{t_1}+\cdots+e^{t_{k-1}}},
\]
where the superscript ${\rm lr}$ identifies the logistic-regression link.

The multilabeled dataset is
\[
\{(X_i;(Y_{1i},\cdots,Y_{im}))\}_{i=1}^n,
\]
with
\[
Y_{ij}\mid X_i\stackrel{\rm iid}{\sim}
\mathsf{Cat}(p_{\Theta^\star}(X_i)).
\]
Here $\Theta^\star$ is the true parameter. With arbitrary tie-breaking, majority-vote labels
\[
Y_i^+=A_m(Y_{i1},\cdots,Y_{im})
\]
obey
\[
Y_i^+\mid X_i\sim\mathsf{Cat}(p_{\Theta^\star,m}(X_i)).
\]
If $\rho_m(t)$ is the majority-vote distribution for $m$ items with initial probabilities $\sigma(t)\in\mathbb R_+^{k-1}$, then
\[
p_{\Theta,m}(x)=
\begin{bmatrix}
\rho_m(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)\\
1-\mathbf1^\top
\rho_m(\langle\theta_1,x\rangle,\cdots,\langle\theta_{k-1},x\rangle)
\end{bmatrix}.
\]
The aggregated link $\rho_m$ satisfies ordering consistency.

The fitted logistic loss, with $\theta_k=\mathbf0$, is
\[
\varphi(\Theta^\top x,y)=
-\langle\theta_y,x\rangle+
\log\left(1+\sum_{i=1}^{k-1}
\exp(\langle\theta_i,x\rangle)\right).
\tag{15}
\]
Define the population loss for aggregation count $m$ by
\[
L_m(\Theta)=\mathbb E[\varphi(\Theta^\top X,Y_m^+)].
\]
The loss gradient for class $i$ is
\[
\nabla_{\theta_i}\varphi(\Theta^\top x,y)=
-x(1\{y=i\}-\sigma_i^{\rm lr}(\Theta^\top x)).
\]
Consequently, a minimizer $\Theta_m$ satisfies
\[
\nabla_\Theta L_m(\Theta_m)=
\mathbb E\!\left[
X\left(
\sigma^{\rm lr}(\Theta_m^\top X)
-\rho_m({\Theta^\star}^\top X)
\right)^\top
\right]=0.
\]

Well-specified unaggregated models have standard statistical consistency and efficiency. There is also a binary robustness result: for $m\geq1$, $k=2$, and
\[
X\sim\mathsf N(0,I_d),
\]
where $\mathsf N$ denotes a normal distribution and $I_d$ is the $d$-dimensional identity matrix, Cheng et al. [7, Prop. 3] obtain
\[
\Theta_m=t_m\Theta^\star
\]
for some $t_m>0$, even with a mis-specified link. Thus the direction
$\Theta^\star/\|\Theta^\star\|_2$ suffices for consistent binary classification at every aggregation level.

For the multiclass mis-specification construction, fix
$\mathcal T_\epsilon\subset\mathbb R^{k-1}$ of Lebesgue measure $\epsilon$ and set
\[
\sigma^\epsilon(t)=
\sigma^{\rm lr}(t)1\{t\notin\mathcal T_\epsilon\}
+\frac1k\mathbf1\,1\{t\in\mathcal T_\epsilon\}.
\]
The conditional distribution on $Y\in\{1,\ldots,k\}$ samples
$Y\sim\sigma^{\rm lr}(t)$ if $t\in\mathcal T_\epsilon$ and uniformly otherwise. The link satisfies ordering consistency and optimal predictability.

For $\epsilon>0$, define
\[
L_{m,\epsilon}(\Theta)\coloneqq
\mathbb E[
\mathbb E_{\sigma^\epsilon}[
\varphi(\Theta^\top X,Y_m^+)\mid X]],
\qquad
\Theta_m(\epsilon)=
\operatorname{argmin}_\Theta L_{m,\epsilon}(\Theta).
\]
Thus $L_{m,\epsilon}$ is the population logistic loss for majority vote over $m$ labels when
\[
Y\mid X=x\sim\sigma^\epsilon({\Theta^\star}^\top x).
\]
Although $\Theta_1(0)=\Theta^\star$, arbitrarily small positive corruption destroys unaggregated consistency.

###### Proposition 5.

Let $k\geq3$ and $\Sigma=I$, with $I$ denoting the identity matrix. For
\[
Z\sim\mathsf N(0,I_{k-1}),
\]
assume invertibility of the map
\[
M\mapsto DM\coloneqq
\mathbb E\!\left[
Z\bigl(
\nabla\sigma^{\rm lr}({\Theta^\star}^TZ)MZ
\bigr)^T
\right].
\]
Then some $\epsilon_0>0$ has the property that every
$\epsilon\in(0,\epsilon_0)$ admits $\mathcal T_\epsilon$ of Lebesgue measure at most $\epsilon$ with
\[
\Theta_1(\epsilon)/\|\Theta_1(\epsilon)\|
\neq
\Theta^\star/\|\Theta^\star\|.
\]

The proof of Proposition 5 is in Appendix E.1.

For a general ordering-consistent link $\sigma$, define
\[
L_{m,\sigma}(\Theta)=
\mathbb E[
\mathbb E_\sigma[
\varphi(\Theta^\top X,Y_m^+)\mid X]],
\qquad
\Theta_m=\operatorname{argmin}L_{m,\sigma}(\Theta).
\]
These are the loss and minimizer used in the following aggregation result.

###### Theorem 3.

Suppose
\[
\Theta^\star=U^\star T^\star,
\]
with orthogonal $U^\star\in\mathbb R^{d\times(k-1)}$ and nonsingular
$T^\star\in\mathbb R^{(k-1)\times(k-1)}$. For each $m$ there exists
$T_m\in\mathbb R^{(k-1)\times(k-1)}$ with $\Theta_m=U^\star T_m$, and
\[
\|T_m\|\to\infty,\qquad
T_m/\|T_m\|\to T^\star/\|T^\star\|
\quad(m\to\infty).
\]

The proof of Theorem 3 is in Appendix E.2. The general-link definitions and Theorem 3 give robustness under mis-specification whenever some link describes the relationship between $X$ and $Y$. Without aggregation, multiclass methods are generally inconsistent.

## 5 Numerical illustrations

Two experiments corroborate the theory: nonparametric amplification and finite-dimensional multiclass asymptotic consistency under link mis-specification. The first experiment uses the **truncated quadratic loss**, denoted D4X.

##### Surrogate consistency amplification for D4X.

For the truncated quadratic experiment, the binary margin surrogate is
\[
\varphi(f(x),y)=\phi(yf(x)),
\qquad
\phi(\delta)=\max\{1-\delta,0\}^2.
\]
The exact unaggregated calibration function is
\[
\psi(\epsilon)=\epsilon^2
\]
[4, Example 2]. Taking $\delta=1$ in Lemma 3.1, the binary lemma, and imposing no low-noise condition, so that $\alpha=0$ and $\mathsf C_{\mathsf{MT}}=1$, gives
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
Here $(u)_+=\max\{u,0\}$. The parallel comparisons for unaggregated and aggregated risks are
\[
\psi(R(f)-R^\star)\leq R_\varphi(f)-R_\varphi^\star,
\qquad
\psi_m(R(f)-R^\star)\leq
R_{\varphi,A_m}(f)-R_{\varphi,A_m}^\star.
\]

As Figure 1 illustrates, the aggregated curve changes from quadratic toward linear as $m\to\infty$, upgrading consistency. However, the curves supplied by the general theorem are conservative.

Figure 1: D4X comparison curves for unaggregated $\psi$ (optimal) and T4K $\psi_m$ (suboptimal).

| Figure | Aggregated-label counts |
|---|---|
| 1 | $m\in\{2^{16},2^{20},2^{24},2^{28},2^{32}\}$ |

##### Asymptotic consistency for a mis-specified logistic model.

Synthetic data from a corrupted link exhibit asymptotic consistency as $m\to\infty$. The fitted loss is the finite-dimensional logistic loss in equation (15). The piecewise data-generating link agrees with the logistic-regression softmax link near the origin and shrinks scores outside radius $r$:
\[
\widetilde\sigma(t)=
\sigma^{\rm lr}(t)1\{\|t\|\leq r\}
+\sigma^{\rm lr}(\alpha t)1\{\|t\|>r\}.
\]
Here $\alpha$ is the score-shrinkage parameter for this experiment.

The two experiments hold different collection quantities fixed:

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

Cheng, Asi and Duchi [7] discuss the collection debate: using all unaggregated labels appears statistically more efficient when models can represent all uncertainty, whereas cleaning appears more robust otherwise. Dorner and Hardt [11] favor more noisy labels rather than cleaned labels for validation comparisons of binary classifiers. The present results establish fundamental limitations of uncleaned supervised-label information and robustness gains from cleaning.

##### Finite $m$ and fundamental limits.

Many consistency results require $m\to\infty$, and reliance on this limit remains unsatisfying. In contrast, ranking admits explicit finite-$m$ guarantees. A theory depending jointly on the task loss $\ell$, surrogate $\varphi$, and available aggregation count $m$ remains open. For example, second- or third-order item comparisons, including the powers $C_x^p$ suggested by Keener [20], may require increasing counts. Precisely separating problems that require cleaning and aggregation from problems that do not is a central challenge and a natural, important direction.

##### Fundamental limits of B2Z.

Beyond binary classification, the noise condition number $\kappa(X)$ may remain large even with low Mammen-Tsybakov noise, $\alpha\approx1$. In contrast, binary problems trivially have $\kappa(X)=1$. The minimal surrogate assumptions produce this behavior. Closer task–surrogate connections could better represent difficulty, while explicit failure constructions and sharper fundamental limits could improve understanding of predictive-model fitting.

##### Mis-specified models.

The mis-specified-model results require optimal predictability: a Bayes-optimal classifier must belong to $\mathcal F$. Classical surrogate consistency can fail even under this requirement, while aggregation succeeds. Beyond this restriction, Nguyen et al. [31] for binary classification and Duchi et al. [12] for multiclass classification show that joint inference of predictor $f$ and a representation of $x$ forces a task-loss-dependent surrogate form. However, their decision rules remain infinitely powerful, so new approaches are needed.

##### Restricted-class comparison with aggregation.

Nonparametric comparison inequalities mitigate surrogate-induced regret, whereas the parametric results establish statistical consistency. Recent $H$-consistency work [2, 28, 27, 29] suggests extending aggregation comparisons to restricted classes. Mao et al. [27] establish rate-optimal quadratic growth of $\psi$ for smooth convex loss; aggregation-based upgrades in $\mathcal H$-consistency remain of interest, with $\mathcal H$ denoting the hypothesis class. Such an extension could unify the parametric and nonparametric advantages and characterize how hypothesis-class “size” affects aggregation.
