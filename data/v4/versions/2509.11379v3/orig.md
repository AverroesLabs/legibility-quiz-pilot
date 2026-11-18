# Some Robustness Properties of Label Cleaning

Chen Cheng

  
John Duchi

###### Abstract

We demonstrate that learning procedures that rely on aggregated labels, e.g.,
label information distilled from noisy responses, enjoy robustness properties
impossible without data cleaning.
This robustness appears in several ways.
In the context of risk consistency—when one takes the standard approach
in machine learning of minimizing a surrogate (typically convex) loss in
place of a desired task loss (such as the zero-one mis-classification
error)—procedures using label aggregation obtain stronger consistency
guarantees than those even possible using raw labels.
And while classical statistical scenarios of fitting perfectly-specified
models suggest that incorporating all possible information—modeling
uncertainty in labels—is statistically efficient, consistency fails for
“standard” approaches as soon as a loss to be minimized is even slightly
mis-specified.
Yet procedures leveraging aggregated information still converge to optimal
classifiers, highlighting how incorporating a fuller view of the data
analysis pipeline, from collection to model-fitting to prediction time,
can yield a more robust methodology by refining noisy signals.

## 1 Introduction

Consider the data collection pipeline in a supervised learning problem.
Naively, we say that we collect pairs $(X_{i},Y_{i})_{i=1}^{n}$ of features
$X_{i}$ and labels $Y_{i}$ , fit a model, and away we go [17].
But this belies the complexity of modern datasets [9, 22, 39], which require
substantial data cleaning, filtering, often crowdsourcing multiple labels
and then denoising them.
The crowdsourcing community has intensively studied such data cleaning,
especially in the context of obtaining “gold standard”
labels [8, 48, 47, 46, 37].
We take a complementary view of this process, investigating the
ways in which data aggregation fundamentally and necessarily
improves the consistency of models we fit.

In a sense, this paper argues that label cleaning, or aggregating labels
together, provides robustness that is impossible to achieve
without aggregating labels.
There are two faces to this robustness.
First, we improve consistency of estimation:
when minimizing a surrogate loss (e.g., the multiclass logistic
loss) instead of a task loss (e.g., the zero-one error),
procedures that use aggregated labels can achieve consistent and optimal
prediction in the limit when this is impossible without data aggregation.
Second, even in finite-dimensional statistical problems, this
aggregation can provide consistent classifiers when
standard methods fail.

Important contributions to the theory of surrogate risk consistency trace to
the 2000s [52, 24, 40], with
Bartlett, Jordan and McAuliffe [4] characterizing when fitting a model using a convex
surrogate is consistent for binary classification for the zero-one error.
Since this work, there has been an abundance of work on
surrogate risk consistency, including on multi-label
classification [50, 43, 16, 49, 1], ranking problems [13, 14, 36], structured prediction [34, 6, 32], ordinal regression [35], restricted hypothesis class [2, 3, 28, 29] and general theory [40].
On the one hand,
these analyses, which consider the standard supervised learning scenario of
data pairs $(X,Y)$ , enable us to fully exploit the entire
statistical theory of empirical processes [45, 5, 21, 4].
On the other, they do not address the data aggregation machinery now
common in modern dataset creation.

It is thus natural to ask about the interaction between consistency
and data aggregation—to begin with, do we need to aggregate at all? If we
can achieve surrogate consistency without data aggregation, we should
perhaps just rely on our mature theoretical
understanding of processes with $(X,Y)$ pairs.
Going one step further, if we aggregate, does aggregation help
consistency, and in what sense does it help?
These main questions motivate this paper.

To further underpin the importance of studying consistency and aggregated
labels, we propose concrete examples—in ranking and binary- and
multiclass-classification with linear estimators—where estimators using
only pairs $(X,Y)$ necessarily fail, but label aggregation methods yield
consistency.
We develop new notions and theory for surrogate consistency with data
aggregation.
In fully nonparametric scenarios, we show how the
number of samples aggregated combine with noise conditions to
improve consistency.
Aggregation will also allow us to demonstrate surrogate risk consistency
under only weak conditions
the surrogate loss;
in the language of the field, losses using aggregated labels
admit (approximate) linear comparison inequalities.
Additionally, in contrast to conventional risk consistency theory, which
requires taking a hypothesis class $\mathcal{F}$ consisting of all measurable
functions, we will show results in classification problems where aggregating
labels guarantees consistency even over restricted hypothesis classes, which
may fail without aggregation.

## 2 Preliminaries

We first review classical surrogate risk minimization.
Let $\mathcal{X}$ be the input space and $\mathcal{Y}$ be the output space, with
data $(X_{i},Y_{i})_{i=1}^{n}\in\mathcal{X}\times\mathcal{Y}$ drawn i.i.d. $P$ .
Consider learning a scoring function $f:\mathcal{X}\to\mathbb{R}^{d}$ that maps an input $x\in\mathcal{X}$ to a score $s\in\mathbb{R}^{d}$
for some $d\geq 1$ , where a decoder $\mathsf{d}:\mathbb{R}^{d}\to\mathcal{Y}$
determines the final prediction via $\widehat{y}=\mathsf{d}\circ f(x)$ .
Given a loss $\ell:\mathcal{Y}\times\mathcal{Y}\to\mathbb{R}_{+}$ and
hypothesis class $\mathcal{F}$ , the goal is to minimize
the task risk over $f\in\mathcal{F}$

| | $\displaystyle R(f):=\mathbb{E}_{P}\left[{\ell(\mathsf{d}\circ f(X),Y)}\right].$
| | (1)

For example, in binary classification, $d=1$ ,
$\mathsf{d}(s)=\mathsf{sgn}(s)$ , and $\ell(y,y^{\prime})=1\!\left\{yy^{\prime}\leq 0\right\}$ ,
yielding $R(f)=\mathbb{P}(Yf(X)\leq 0)$ .
The challenge of minimizing $R(f)$ is that the task loss $\ell$ can be
nonsmooth, nonconvex, and—even more—uninformative: the loss landscape of
the $0$ - $1$ loss is flat almost everywhere.
This makes even practical (e.g., first-order) optimization impossible.
We will consider a slightly more sophisticated
version of the problem (1),
where instead of the loss $\ell$ being defined only
in terms of the instantaneous label $Y$ , we will allow it to depend
on $P(Y\in\cdot\mid X)$ , so that
we investigate

| | $R(f)\coloneqq\mathbb{E}_{P}\left[\ell(\mathsf{d}\circ f(X),P(\cdot\mid X))\right],$
| | (2)

whose minimizers frequently coincide with the original
problem (2), but which allows more sophistication.
(For example, in multiclass classification, $Y\in\{1,\ldots,k\}$ ,
and taking $\ell(\widehat{y},P)=\sum_{y}P(Y=y)1\{\widehat{y}\neq y\}$ , the risk coincides with the standard 0-1 error rate.)

Instead of the task loss $\ell$ , we thus consider an easier to optimize
surrogate $\varphi:\mathbb{R}^{d}\times\mathcal{Y}\to\mathbb{R}$ .
Then rather than attacking the risk (2) directly,
we minimize surrogate risk

| | $R_{\varphi}(f):=\mathbb{E}_{P}\left[{\varphi(f(X),Y)}\right].$
|

For this to be sensible, we must exhibit some type of consistency with the
task problem (2).
In this paper, we particularly study in two scenarios, which
we will make more formal:

(i)

The “classical” case of Fisher consistency, where $\mathcal{F}$ contains
all Borel functions;

(ii)

Statistical scenarios in which the hypothesis class $\mathcal{F}$ is
parametric but may be mis-specified.

Our main message is that label aggregation improves
consistency in both scenarios, demonstrating the robustness of label
cleaning.

### 2.1 Label aggregation

Instead of obtaining $(X_{i},Y_{i})$ pairs, consider
the case that we replace the output $Y$ with a more abstract
variable $Z\in\mathcal{Z}$ .
For example, in the motivating scenario in the introduction in which we
collect multiple (say, $m$ ) noisy labels for each example $X$ , we
take $Z=(Y_{1},\ldots,Y_{m})\in\mathcal{Y}^{m}$ .
Mao et al. [28] consider similar multilabeled
data structure for multilabel losses, while we focus on the label aggregation pipeline.
For an abstract “aggregation space” $\mathcal{A}$ , let $A:\mathcal{Z}\to\mathcal{A}$
be an aggregating function (e.g., majority vote),
and
let $\varphi:\mathbb{R}^{d}\times\mathcal{A}\to\mathbb{R}_{+}$ be a surrogate loss
defined on this aggregation space.
We then define the aggregated surrogate risk

| | $\displaystyle R_{\varphi,A}(f):=\mathbb{E}\left[{\varphi(f(X),A(Z))}\right],$
| | (3)

asking when minimizing the surrogate
problem (3) is sufficient to
minimize the actual task risk (2).
Two concrete examples may make this clearer.

Example 1 (Majority vote):

In the repeated sampling regime, data collection takes the
form $Z=(Y_{1},\cdots,Y_{m})$ , $Y_{i}\mid X=x\stackrel{{\scriptstyle\rm iid}}{{\sim}}P_{Y\mid X=x}$ .
Define
$A_{m}(Z)$ to be the
empirical minimizer

| | $\displaystyle A(Z)=A(\{Y_{1},\ldots,Y_{m}\})=\mathop{\rm argmin}_{y\in\mathcal{Y}}\sum_{l=1}^{m}\ell(y,Y_{l}).$
|

When $\ell(\widehat{y},y)=1\{y\neq\widehat{y}\}$ ,
this corresponds exactly to majority vote; the more
general form allows more abstract procedures.
$\Diamond$

We can also (roughly) capture $K$ -nearest neighbor aggregation procedures:

Example 2 ( $K$ -nearest neighbors):

Consider an abstract repeated sampling scenario in which an example $X$
comes with a label $Y$ and an additional draw $(X_{i},Y_{i})_{i=1}^{m}\stackrel{{\scriptstyle\rm iid}}{{\sim}}P$ , where $m$ is the number of additional examples, so $Z=(Y,(X_{i},Y_{i})_{i=1}^{m})$ . Let $\mathop{\rm dist}:\mathcal{X}\times\mathcal{X}\to\mathbb{R}_{+}$ be a
distance metric on $\mathcal{X}$ .
Let $\{X_{(1)},\ldots,X_{(m)}\}$ order the input sample $\{X_{i}\}_{i=1}^{m}$
by distance, $\mathop{\rm dist}(X,X_{(1)})\leq\ldots\leq\mathop{\rm dist}(X,X_{(m)})$ (and let
$X_{(0)}=X$ ).
For $K\geq 0$ , we can aggregate the
$K$ -nearest neighbors of $X$ , for example, by choosing

| | $\displaystyle A_{m,K}(Z)\coloneqq\mathop{\rm argmin}_{y\in\mathcal{Y}}\sum_{l=0}^{K}\ell(y,Y_{(l)}).$
|

In Appendix C, we leverage the results in the
coming sections to move beyond this population-level scenario to address
aggregation from a single sample $(X_{i},Y_{i})_{i=1}^{n}$ .
$\Diamond$

## 3 Surrogate consistency

The standard framework for surrogate consistency [40]
assumes that $\mathcal{F}$ consists of all Borel
measurable functions $f:\mathcal{X}\to\mathbb{R}^{d}$ .
Working in the abstract setting in the preliminaries, define the
conditional task risk $R(s\mid x)$ and the conditional surrogate risk
$R_{\varphi}(s\mid x)$ , $s\in\mathbb{R}^{d}$ by

| | $\displaystyle R(s\mid x)\coloneqq\ell(\mathsf{d}\circ s,P(Y\in\cdot\mid X=x))~~\mbox{and}~~R_{\varphi}(s\mid x):=\mathbb{E}\left[{\varphi(s,Y)\mid X=x}\right].$
|

We then define the pointwise excess risks

| | $\displaystyle\delta_{\ell}(s,x)\coloneqq R(s\mid x)-\inf_{s^{\prime}\in\mathbb{R}^{d}}R(s^{\prime}\mid x),\qquad\delta_{\varphi}(s,x)\coloneqq R_{\varphi}(s\mid x)-\inf_{s^{\prime}\in\mathbb{R}^{d}}R_{\varphi}(s^{\prime}\mid x),$
|

as well as the minimal risks $R^{\star}:=\inf_{f\in\mathcal{F}}R(f)$ and
$R_{\varphi}^{\star}:=\inf_{f\in\mathcal{F}}R_{\varphi}(f)$ .
We follow the standard [40, 4, 52]
that consistency requires at least (i) Fisher consistency and,
if possible, a stronger and quantitative (ii) uniform comparison inequality:
respectively, that for all data distributions $P$ ,

(i)

For any sequence of functions $f_{n}\in\mathcal{F}$ , $R_{\varphi}(f_{n})\to R_{\varphi}^{\star}$ implies $R(f_{n})\to R^{\star}$ .

(ii)

For a non-decreasing $\psi:\mathbb{R}_{+}\to\mathbb{R}_{+}$ , $\psi(R(f)-R^{\star})\leq R_{\varphi}(f)-R_{\varphi}^{\star}$ for all $f\in\mathcal{F}$ , where $\psi$
satisfies $\psi(\epsilon)>0$ for all $\epsilon>0$ ,

In the case of binary classification when $\varphi$ is margin-based and
convex, the two consistency notions coincide [4].
The stronger uniform guarantee (ii) need not
always hold, the calibration function $\overline{\psi}$ provides a
canonical construction through the excess risk:

| | $\displaystyle\psi(\epsilon,x)\coloneqq\inf_{s\in\mathbb{R}^{d}}\left\{\delta_{\varphi}(s,x)\mid\delta_{\ell}(s,x)\geq\epsilon\right\}~~\mbox{and}~~\overline{\psi}(\epsilon)\coloneqq\inf_{x\in\mathcal{X}}\psi(\epsilon,x).$
|

Consistency and comparison inequalities follow
from the calibration functions (see Zhang [51, Prop. 25] and
Steinwart [40, Thm. 2.8 and Lemma 2.9]):

###### Corollary 3.1.

The surrogate $\varphi$ is Fisher consistent (i)
for $\ell$ if and only if $\psi(\epsilon,x)>0$ for all
$x\in\mathcal{X}$ and $\epsilon>0$ .
Let $\psi$ be the Fenchel biconjugate of $\overline{\psi}$ .
Then $\overline{\psi}(\epsilon)>0$ if and only
if $\psi(\epsilon)>0$ , and for all measurable $f$ ,

| | $\displaystyle\psi(R(f)-R^{\star})\leq R_{\varphi}(f)-R_{\varphi}^{\star}.$
|

In the general risk minimization problem (2) we
would like at least a Fisher consistent (i)
surrogate for $\ell$ , so that minimizing $R_{\varphi}(f)=\mathbb{E}[\varphi(f(X),Y)]$ would imply minimizing $R(f)$ .
Given such a result, using only paired observations $(X,Y)$ rather than
tuples $(X,Y_{1},\ldots,Y_{m})$ , we could bring the entire theory of
empirical processes and related statistical tools [45, 5, 21, 4] to bear on the problem.
Moreover, data collection procedures would be simpler, necessitating only
single pairs $(X,Y)$ for consistent estimation.
Unfortunately, such results are generally impossible, as we detail in the
next extended example, necessitating the necessity of a theory of
aggregation that we pursue in Sec. 3.2.

### 3.1 Fisher consistency failure without label aggregation: ranking

Consider the problem of ranking $k$ items using pairwise comparison
data [20, 15, 14, 30],
where the space $\mathcal{Y}$ consists of all pairwise comparisons of these
items, $\mathcal{Y}=\{(i,j):i\neq j,1\leq i,j\leq k\}$ , i.e. $(i,j)$ denotes the $i$ -th item has higher ranking than the $j$ -th item.
The (population) rank aggregation problem is, for each $x$ ,
to transform the
probabilities $p_{ij}=P(Y=(i,j)\mid x)$ into a
ranking of the $k$ items.
While numerous possibilities exist for such aggregation, we consider a
simple comparison-based aggregation scheme [20, cf.]; similar
negative results to the one we show below hold for more sophisticated
schemes.
Define the normalized transition matrix $C_{x}\in\mathbb{R}_{+}^{k\times k}$ with entries $(C_{x})_{ii}=0$ and

| | $\displaystyle(C_{x})_{ij}=\frac{p_{ij}}{\sum_{l\neq j}p_{lj}}~~\mbox{for~}i\neq j,$
|

where we let $0/0=1/(k-1)$ so that $C_{x}$ is stochastic, satisfying $C_{x}^{T}\mathbf{1}=\mathbf{1}$ .
We then rank the items by the vector $C_{x}\mathbf{1}\in\mathbb{R}_{+}^{k}$ ,
which measures how often a given item is preferred to others.
(One may also
take higher powers $C_{x}^{p}\mathbf{1}$ or Perron vectors [20]; similar
results to ours below hold in such cases.)
Tacitly incorporating the decoding $\mathsf{d}$ into the task loss $\ell$ ,
we

| | $\ell(s,C)\coloneqq\max_{i<j}1\!\left\{(s_{i}-s_{j})(e_{i}-e_{j})^{T}C\mathbf{1}\leq 0,~(e_{i}-e_{j})^{T}C\mathbf{1}\neq 0\right\},$
|

which penalizes mis-ordered scores between $s$ and $C$ .
The population task risk (2) is thus

| | $\displaystyle R(f)$ | $\displaystyle\coloneqq\mathbb{P}\left(f(X)\text{ and }C_{X}\mathbf{1}\text{ order differently}\right)$
| | (4)

Now consider a convex surrogate $\varphi:\mathbb{R}^{k}\times\mathcal{Y}\to\mathbb{R}$ .
We restrict to $s\in\mathbb{R}^{k}$ for which $s^{T}\mathbf{1}=0$ , a
minor restriction familiar from multiclass classification
problems [51, 43], which is natural as for decoding a
ranking we require only the ordering of the $s_{i}$ .
Unfortunately, there is no convex Fisher consistent surrogate for the
problem (4) (see
Appendix A.1).

###### Proposition 1.

Consider the ranking problem with task risk (4)
over $k\geq 3$ outcomes.
If $\varphi:\mathbb{R}^{k}\times\mathcal{Y}\to\mathbb{R}$ is convex in its first argument,
it is not Fisher consistent.

Nonetheless, a reasonably straightforward argument yields consistency when
we allow aggregation methods as soon as $m$ , the number of collected
comparisons, satisfies $m\geq k$ .
The idea is simple: we regress predicted scores $f(x)$ on frequencies
of label orderings.
We assume multiple independent pairwise comparisons $Z=(Y_{1},\ldots,Y_{m})$
conditioned on $X$ , and letting $m_{ij}=\sum_{y\in\mathcal{Y}^{m}}1\!\left\{y=(i,j)\right\}$ and $m_{j}=\sum_{i=1}^{k}m_{ij}$ , we define
the aggregation

| | $\displaystyle A(Z)$ | $\displaystyle=\begin{cases}\star,&\mbox{if}~m_{j}=0\text{ for some }j\text{ in }[k],\\
(\frac{m_{i1}}{m_{1}}+\frac{m_{i2}}{m_{2}}+\cdots+\frac{m_{ik}}{m_{k}})_{i\in[k]}&\mbox{otherwise, i.e.~if}~m_{j}>0\text{ for all }j\in[k].\end{cases}$
|

Regressing directly on $A(Z)$ when $A(Z)\neq\star$ yields consistency, as
the next proposition demonstrates (see
Appendix A.2 for a proof):

###### Proposition 2.

Define $\varphi(s,q)=\left\|{s-q}\right\|_{2}^{2}$ for $s,q\in\mathbb{R}^{k}$ and
$\varphi(s,\star)=0$ .
Then if $m\geq k$ , $\varphi$ is Fisher consistent
for the ranking risk (4).

We point out the reason Proposition 1 fails to guarantee consistency, whereas Proposition 2 does ensure consistency, lies in the expansion of the response space from $\mathcal{Y}$ to $\mathbb{R}^{k}$ induced by aggregating multiple labels.

### 3.2 Label aggregation obtains stronger surrogate consistency

The extended ranking example in ranking suggests potential benefits of
aggregating labels, and it is natural to ask how aggregation interacts with
surrogate consistency more generally.
Thus, we present two results here: one that performs an essentially
basic extension of standard surrogate-risk consistency, and the second
that shows how aggregation-based methods can “upgrade” what might
nominally be inconsistent losses into consistent losses,
as Proposition 2 suggests may be possible.

#### 3.2.1 Basic extensions of surrogate consistency

We begin by making the more or less obvious generalization
of calibration functions for standard cases, extending
the classical comparison
inequalities in Corollary 3.1.
For an arbitrary aggregation method $A:\mathcal{Z}\to\mathcal{A}$ , define the
conditional surrogate risk with data aggregation

| | $\displaystyle R_{\varphi,A}(s\mid x)\coloneqq\mathbb{E}\left[{\varphi(s,A(Z))\mid X=x}\right].$
|

As in the non-aggregated case, the pointwise excess risk

| | $\displaystyle\delta_{\varphi,A}(s,x):=R_{\varphi,A}(s\mid x)-\inf_{s\in\mathbb{R}^{d}}R_{\varphi,A}(s\mid x)$
|

then defines the pointwise and uniform calibration functions

| | $\overline{\psi}_{A}(\epsilon,x)\coloneqq\inf_{s\in\mathbb{R}^{d}}\left\{\delta_{\varphi,A}(s,x)\mid\delta_{\ell}(s,x)\geq\epsilon\right\}~~\mbox{and}~~\overline{\psi}_{A}(\epsilon)\coloneqq\inf_{s\in\mathbb{R}^{d}}\overline{\psi}_{A}(\epsilon,x).$
| | (5)

A consistency result then follows, similar to
Corollary 3.1, under appropriate
measurability conditions (we will leave these tacit as they are not central
to our results).
Then more or less as a corollary of Steinwart [40, Thm. 2.8], we have
the following consistency result.
(We include a proof for completeness in
Appendix B.1.)

###### Proposition 3.

Assume there exists $b:\mathcal{X}\to\mathbb{R}_{+}$ with
$\int b(x)dP(x)<\infty$ such that
$\delta_{\ell}(f(x),x)\leq b(x)$ .
The surrogate $\varphi$ is Fisher consistent (i)
for the task risk (2)
if and only if $\overline{\psi}_{A}(\epsilon,x)>0$ for all $x\in\mathcal{X}$
and $\epsilon>0$ .
Additionally, if $\psi_{A}=(\overline{\psi}_{A})^{**}$ is the Fenchel
biconjugate of $\overline{\psi}_{A}$ ,
then

| | $\displaystyle\psi_{A}(R(f)-R^{\star})\leq R_{\varphi,A}(f)-R_{\varphi,A}^{\star}.$
|

The result captures the classical consistency guarantees—nothing
particularly falls apart because of aggregation—but it provides no
specific guarantees of improved consistency.
We turn to this now.

#### 3.2.2 Identifying surrogates and consistency

We now turn under essentially minimal conditions on the surrogate, there is
a generic aggregating strategy that (asymptotically in the number of
observations $y$ ) guarantees consistency for any task loss that seeks to
minimize $\ell(f(x),y)$ , i.e.,
$R(f)=\mathbb{E}[\ell(f(X),Y)]$ .
We assume that $\mathop{\rm card}(\mathcal{Y})=k<\infty$ 11
1

We emphasize that the example discussed in Sec. 3.1 belongs to this regime, where $k$ denotes the ranking length and $\mathop{\rm card}(\mathcal{Y})=k(k-1)$ . With a slight abuse of notation, we will henceforth use $k$ to denote the cardinality of the label space.,
and we impose a minimal identifiability assumption on the surrogate
loss.

###### Definition 3.1 (Identifying surrogate).

A surrogate $\varphi:\mathbb{R}^{d}\times\mathcal{A}\to\mathbb{R}$ is
$(\mathsf{C}_{\varphi,1},\mathsf{C}_{\varphi,2})$ -identifying
for $\mathcal{Y}$ ,
$0<\mathsf{C}_{\varphi,1}\leq\mathsf{C}_{\varphi,2}<\infty$
if there exist
$\{a_{y}\}_{y\in\mathcal{Y}}\subset\mathcal{A}$ and
vectors $\{s_{y}\}_{y\in\mathcal{Y}}\subset\real^{d}$ such that
$\mathsf{d}(s_{y})=y$ and for which for all $y\neq y^{\prime}$ ,

|

| | $\displaystyle\varphi(s_{y},a_{y})+\mathsf{C}_{\varphi,1}$ | $\displaystyle\leq\inf_{\mathsf{d}\circ s\neq y}\varphi(s,a_{y}),$
| | (6a)

| | $\displaystyle\varphi(s_{y},a_{y^{\prime}})-\mathsf{C}_{\varphi,2}$ | $\displaystyle\leq\inf_{s\in\mathbb{R}^{d}}\varphi(s,a_{y^{\prime}}).$
| | (6b)

Inequality (6a) captures
that for each class $y\in\mathcal{Y}$ ,
there exists a parameter $a_{y}\in\mathcal{A}$ such that the minimizer
of $\varphi(\cdot,a_{y})$ identifies $y$ .
A finite
$\mathsf{C}_{\varphi,2}$ exists for (6b)
if $\varphi(\cdot,a)$ has a finite lower bound.
Notably, Definition 3.1
does not require that $\varphi(\cdot,a)$ is convex or that it is
consistent when $\mathcal{A}=\mathcal{Y}$ and $Z=Y$ , i.e., without label
aggregation. Another benefit is that we also do not require the decoder $\mathsf{d}$ , such as requiring it to coincide with the pointwise $\mathop{\rm argmax}$ operator.

Example 3:

Consider the binary hinge loss $\varphi(s,a)=\max\{1-sa,0\}$ for
$\mathcal{A}=\mathcal{Y}=\{\pm 1\}$ .
For $y\in\{-1,1\}$ , take $a_{y}=s_{y}=y$ , so that
$\varphi(s_{1},a_{1})=\varphi(s_{-1},a_{-1})=0$ , while $\inf_{sa\leq 0}\varphi(s,a)=1$ .
Similarly, $\varphi(s_{1},a_{-1})=\varphi(s_{-1},a_{1})=2$ , so the hinge
loss is $(1,2)$ -identifying.
$\Diamond$

We defer further concrete examples to Sec. 3.3 where we study the consistency amplification by label aggregation. Given an identifying surrogate with parameters $\{a_{y}\}_{y\in\mathcal{Y}}$ , we
consider a naive aggregation strategy: the generalized majority vote

| | $A_{m}(y_{1},\ldots,y_{m})\coloneqq a_{\hat{y}}~~\mbox{for}~~\hat{y}=\mathop{\rm argmin}_{y\in\mathcal{Y}}\sum_{i=1}^{m}\ell(y,y_{i})$
| | (7)

(breaking ties arbitrarily).
As $m\to\infty$ , because $\mathcal{Y}$ is finite, whenever
$Y_{i}$ are i.i.d. there necessarily exists a (random) $M<\infty$ such
that $m\geq M$ implies

| | $\mathop{\rm argmin}_{y\in\mathcal{Y}}\bigg\{\sum_{i=1}^{m}\ell(y,Y_{i})\bigg\}\subset y^{\star}(x)\coloneqq\mathop{\rm argmin}_{y\in\mathcal{Y}}\mathbb{E}\left[\ell(y,Y)\mid X=x\right].$
|

From this, we expect that as $m\to\infty$ , the
surrogate $\varphi(\cdot,A_{m})$ ought to be consistent.
In fact, we have the following corollary of our coming results,
guaranteeing (asymptotic) consistency:

###### Corollary 3.2.

Let $m=m(n)\to\infty$
and $\varphi$ be identifying (Def. 3.1).
Then

| | $R_{\varphi,A_{m}}(f_{n})-R^{\star}_{\varphi,A_{m}}\to 0~~\mbox{implies}~~R(f_{n})-R^{\star}\to 0.$
|

#### 3.2.3 Identifying surrogates and consistency amplification

In cases with low noise in the labels, the aggregation
strategy (7) allows an explicitly improved
comparison inequality $\psi(R(f)-R^{\star})\leq R_{\varphi}(f)-R^{\star}$ , in that
$\psi$ is linear over some range of $\epsilon>0$ —and linear growth is
the strongest comparison inequality possible [34, 32].
More generally, strict comparison inequalities, such as those
present in Proposition 3, can be too
narrow, as it can still be practically convenient to adopt
inconsistent surrogates [23, 34, 32].
Thus, we follow Osokin et al. [34] to introduce $(\xi,\zeta)$
consistency, which requires a comparison function $\psi$ to grow linearly
only for $\epsilon\geq\xi$ , so that the surrogate captures a sort of “good
enough” risk.

###### Definition 3.2.

The surrogate loss $\varphi$ and aggregator $A$ yield level- $(\xi,\zeta)$ consistency if there exists $\psi$ satisfying $\psi(\epsilon)\geq\zeta\epsilon$ for $\epsilon\geq\xi$ , and $\psi(R(f)-R^{\star})\leq R_{\varphi,A}(f)-R^{\star}_{\varphi,A}$ .

In the following discussion, we show under minimal assumptions, label
aggregation (7)
can achieve level- $(o_{m}(1),\zeta)$ consistency even if the
surrogate is Fisher inconsistent.

We introduce a quantifiable noise condition, adapting the
now classical Mammen-Tsybakov noise conditions [25]
(see also Bartlett et al. [4]).
Define

| | $\Delta(x)\coloneqq\min_{\mathsf{d}(s)\not\in y^{\star}(x)}\delta_{\ell}(s,x),$
| | (8)

the minimal excess conditional risk when making an incorrect prediction.
In binary classification problems with
$\mathcal{Y}=\{\pm 1\}$ , one obtains
$\Delta(x)=|2P(Y=1\mid X=x)-1|$ , and more generally,
we expect that consistent estimation should be harder when
$\Delta(x)$ is closer to 0.
We can define the Mammen-Tsybakov conditions
(where the constant $\mathsf{C}_{\mathsf{MT}}>0$ may change) as

| | $\mathbb{P}\left(\mathsf{d}\circ f\neq\mathsf{d}\circ f^{\star}\right)\leq\mathsf{C}_{\mathsf{MT}}\left(R(f)-R^{\star}\right)^{\alpha}~~\mbox{for~all~measurable}~f,$
| | (Nα)

where we refer to condition (Nα)
as having noise exponent $\alpha$ ,
and

| | $\mathbb{P}(\Delta(X)\leq\epsilon)\leq(\mathsf{C}_{\mathsf{MT}}\epsilon)^{\beta}~~\mbox{for}~\epsilon>0.$
| | (Mβ)

Here, $\alpha\in[0,1]$ and $\beta\in[0,\infty]$ , so that
conditions (Nα) and (Mβ) always
trivially hold with $\alpha=\beta=0$ , moreover, as in the binary
case [4, Thm. 3], they are equivalent via
the transformation $\beta=\frac{\alpha}{1-\alpha}$ .
(See Appendix B.2.)

Notably, in the
recent work of Mao et al. [29], the authors consider the exact same noise statistic
for $H$ -consistency in multiclassification—while they operate under structural assumptions
of the surrogate and hypothesis class to enable a global convex lower bound, we consider
the generic setup with minimal structural constraint, which requires us to introduce the
noise condition number in the following.

| | $\displaystyle\kappa(x)$ | $\displaystyle\coloneqq\frac{\max_{\mathsf{d}(s)\neq y^{\star}(x)}\delta_{\ell}(s,x)}{\min_{\mathsf{d}(s)\neq y^{\star}(x)}\delta_{\ell}(s,x)},$
| | (9)

which connects the noise statistic $\Delta(x)$ and the pointwise excess risk
via $\Delta(x)\geq\delta_{\ell}(s,x)/\kappa(x)$ for all $s$ such that
$\mathsf{d}(s)\neq y^{\star}(x)$ , allowing more fine-grained analysis.
In binary classification, we have $\kappa(x)=1$ so long as $\mathbb{P}(\Delta(X)>0)=1$ .

The noise statistic $\Delta(x)$ and condition number $\kappa(x)$ will allow
us to show how (generalized) majority vote (7),
when applied in the context of any identifiable surrogate
(Definition 3.1), achieves level- $(\xi,\zeta)$ consistency.
Define the error function

| | $\displaystyle e_{m}(t)\coloneqq t\sqrt{\frac{2}{m}\log\left({\frac{4k(\mathsf{C}_{\varphi,1}+\mathsf{C}_{\varphi,2})}{\mathsf{C}_{\varphi,1}}}\right)},$
| | (10)

which roughly captures that if $\kappa(x)=t$ , then
majority vote $A_{m}$ is likely correct if $m$ is large enough that
$e_{m}(t)\ll 1$ .
We then have the following theorem, which provides a (near) linear
calibration function; we prove it in
Appendix B.3.

###### Theorem 1.

Let the surrogate loss $\varphi$ be $(\mathsf{C}_{\varphi,1},\mathsf{C}_{\varphi,2})$ -identifying with parameters $\{a_{y}\}_{y\in\mathcal{Y}}$ ,
and $A_{m}$ be the majority vote aggregator (7).
Assume the task loss satisfies $0\leq\ell\leq 1$
and $P$ satisfies condition (Nα)
with noise exponent $\alpha\in[0,1]$ .
Then for any $M>0$ and $f\in\mathcal{F}$ such that $R(f)-R^{\star}\geq 2\mathbb{P}(\kappa(X)>M)+(4\mathsf{C}_{\mathsf{MT}}e_{m}(M))^{\frac{1}{1-\alpha}}$ ,

| | $\displaystyle R(f)-R^{\star}\leq\frac{16}{\mathsf{C}_{\varphi,1}}\cdot\left({R_{\varphi,{A_{m}}}(f)-R_{\varphi,{A_{m}}}^{\star}}\right).$
|

Said differently, under the conditions of the theorem, $\varphi$ with
aggregation provides level $(\xi,\zeta)$ consistency
(Def. 3.2) with $\xi=2\mathbb{P}(\kappa(X)>M)+(4\mathsf{C}_{\mathsf{MT}}e_{m}(M))^{\frac{1}{1-\alpha}}$ and $\zeta=\frac{\mathsf{C}_{\varphi,1}}{16}$ .
Theorem 1 also
provides an immediate proof of Corollary 3.2,
that is, an asymptotic guarantee of consistency.
Indeed, define

| | $\displaystyle\xi_{m}\coloneqq\inf_{M}\left\{2\mathbb{P}(\kappa(X)>M)+(4\mathsf{C}_{\mathsf{MT}}e_{m}(M))^{\frac{1}{1-\alpha}}\right\},$
|

which satisfies $\xi_{m}\to 0$ as $m\to\infty$ ,
because $\mathbb{P}(\kappa(X)>M)\to 0$ as $M\uparrow\infty$ and
for any fixed $M$ , $e_{m}(M)\to 0$ as $m$ grows.
Corollary 3.2
then follows trivially by taking $\alpha=0$ .

Theorem 1
is a somewhat gross result, as the identifiability
conditions in Def. 3.1 are so weak.
With a tighter connection between task loss $\ell$ and surrogate $\varphi$ ,
for example, making the naive majority vote (7)
more likely to be correct (or at least correct enough for $\varphi$ ),
we would expect stronger bounds—e.g., when $\varphi$ is nearly consistent
without aggregation, we may expect smaller inconsistency without aggregation
should imply stronger bounds with majority vote in $m$ .
We do not pursue the details here.

To provide a somewhat more concrete bound, we optimize
over $M$ in Theorem 1,
using the crude bound $\kappa(x)\leq 1/\Delta(x)$ on the condition
number.
By taking $M=1$ for $\mathop{\rm card}(\mathcal{Y})=k=2$ and optimizing $M$ for $k\geq 3$ ,
we may lower bound $\xi_{m,k}$ in the level $(\xi_{m,k},\zeta)$ -consistency
(Def. 3.2) that in
Theorem 1 promises, setting

| | $\displaystyle\xi_{m,k}\coloneqq\begin{cases}\left({\frac{32\mathsf{C}_{\mathsf{MT}}^{2}}{m}\log\left({\frac{8(\mathsf{C}_{\varphi,1}+\mathsf{C}_{\varphi,2})}{\mathsf{C}_{\varphi,1}}}\right)}\right)^{\frac{1}{2(1-\alpha)}},&\mbox{if}~k=2\\
4\cdot\left({\frac{32\mathsf{C}_{\mathsf{MT}}^{4}}{m}\log\left({\frac{4k(\mathsf{C}_{\varphi,1}+\mathsf{C}_{\varphi,2})}{\mathsf{C}_{\varphi,1}}}\right)}\right)^{\frac{\alpha}{2(1-\alpha^{2})}},&\mbox{otherwise}.\end{cases}$
|

Making appropriate algebraic substitutions and manipulations
(see Appendix B.4),
we have the following corollary.

###### Corollary 3.3.

Under the conditions of
Theorem 1, for any $f$ such
that $R(f)-R^{\star}\geq\xi_{m,k}$ ,

| | $\displaystyle R(f)-R^{\star}\leq\frac{16}{\mathsf{C}_{\varphi,1}}\cdot\left({R_{\varphi,{A_{m}}}(f)-R_{\varphi,{A_{m}}}^{\star}}\right).$
|

The above corollary and Corollary 3.2 provide
evidence for the robustness of label cleaning: with minimal assumptions on
the surrogate, data aggregation can still yield consistency.
As the noise exponent $\alpha$ approaches $1$ in
Corollary 3.3, the sample size $m$
required for the comparison inequality to hold for a fixed score function
$f$ shrinks.
Notably, if $\alpha=1$ , whenever

| | $\displaystyle m\geq 32\max\{\mathsf{C}_{\mathsf{MT}}^{2},\mathsf{C}_{\mathsf{MT}}^{4}\}\cdot\log\left({\frac{4k(\mathsf{C}_{\varphi,1}+\mathsf{C}_{\varphi,2})}{\mathsf{C}_{\varphi,1}}}\right)=O(\log k),$
|

we have $\xi_{m,k}=0$ ,
yielding the uniform comparison inequality (ii)
with linear comparison.
The noise level of the learning problem itself affects the aggregation level
needed for consistency—an “easier” problem requires less aggregation to
achieve stronger consistency.

### 3.3 Surrogate consistency examples with majority vote

We collect several examples, of varying levels of concreteness, that allow
us to instantiate Theorem 1 and
Corollary 3.3.
Throughout, we shall assume that $P$ has a noise exponent $\alpha\in[0,1]$ , though this is no loss of generality, as
Condition (Nα) always holds with $\alpha=0$ .
We defer proofs for each result in this section
to Appendix B.5.

#### 3.3.1 Binary classification with a nonsmooth surrogate

Consider the binary classification problem with a margin-based
surrogate $\varphi(f(x),y)=\phi(yf(x))$ , where $\phi$ is convex;
Bartlett et al. [4] show that $\varphi$ is consistent if and
only if $\phi^{\prime}(0)<0$ .
Here, we show a (somewhat trivial) example for the robustness data
aggregation offers by demonstrating that even if $\phi$ is inconsistent
without aggregation, it can become so with it.
Note, of course, that one would never use such a surrogate, so
one ought to think of this as a thought experiment.
Assume that the subgradient set $\partial\phi(0)\subset(-\infty,0)$ and
$\phi$ is convex with $\lim_{t\to\infty}\phi(t)=0$ .

###### Lemma 3.1.

For any $\delta>0$ ,
$\varphi$ is $(\mathsf{C}_{\varphi,1},\mathsf{C}_{\varphi,2})$ -feasible with

| | $\displaystyle\mathsf{C}_{\varphi,1}=\phi(0)-\phi(\delta)>0~~\mbox{and}~~\mathsf{C}_{\varphi,2}=\phi(-\delta).$
|

Corollary 3.3 thus applies with
$k=2$ ,
so if $f:\mathcal{X}\to\mathbb{R}$ satisfies

|
| | $\displaystyle R(f)-R^{\star}\geq\left({\frac{32\mathsf{C}_{\mathsf{MT}}^{2}}{m}\log\left({\frac{8(\phi(-\delta)+\phi(0)-\phi(\delta))}{\phi(0)-\phi(\delta)}}\right)}\right)^{\frac{1}{2(1-\alpha)}},$
|

then

| | $\displaystyle R(f)-R^{\star}\leq\frac{16}{\phi(0)-\phi(\delta)}(R_{\varphi,A_{m}}(f)-R_{\varphi,A_{m}}^{\star}).$
|

#### 3.3.2 Bipartite matching

In general structured prediction problems [33], an embedding
map $v:\mathcal{Y}\to\mathbb{R}^{d}$ encodes structural information about
elements $y\in\mathcal{Y}$ , where $\mathcal{Y}$ is some “structured”
space, which is typically large.
Using decoder $\mathsf{d}(s)=\mathop{\rm argmax}_{y\in\mathcal{Y}}\langle s,v(y)\rangle$ , for a loss
$\ell:\mathcal{Y}\times\mathcal{Y}\to\mathbb{R}_{+}$ with $\ell(y,y)=0$ , the
maximum-margin surrogate (generalized hinge loss) [42, 44, 19] takes the form

| | $\displaystyle\varphi(s,y)=\max_{\hat{y}\in\mathcal{Y}}\left({\ell(\hat{y},y)+\langle v(\hat{y})-v(y),s\rangle}\right).$
| | (11)

Notably, the loss (11) is typically
inconsistent, except in certain low noise cases [34, 32].

Before discussing structured prediction broadly, we consider
bipartite matching.
A bipartite matching consists of a graph $G=(V,E)$ where the vertices $V=V_{1}\cup V_{2}$ partition into left and right sets $V_{1}=\{1,\ldots,N\}$
and $V_{2}=\{N+1,\ldots,2N\}$ , while the $N$ edges $E$ each connect
exactly one (unique) node in $V_{1}$ and $V_{2}$ .
Letting $\mathcal{Y}$ be the collection of all bipartite matching between $V_{1}$ and
$V_{2}$ , we evidently have $k=\mathop{\rm card}(\mathcal{Y})=N!$ .
For any graph $G$ , the embedding map

| | $v(G)\coloneqq\left(1\!\left\{(u,v)\in E\right\}\right)_{u\in V_{1},v\in V_{2}}\in\mathbb{R}^{N^{2}}$
|

indexes edges, yielding $d=N^{2}$ .
The task loss counts the number of mistaken edges,

| | $\displaystyle\ell(y_{1},y_{2})\coloneqq\frac{1}{2N}\left\|{v(y_{1})-v(y_{2})}\right\|_{1}=\frac{1}{2N}\left\|{v(y_{1})-v(y_{2})}\right\|_{2}^{2}.$
|

In this case, the max-margin (structured hinge loss)
surrogate (11)
is identifying:

###### Lemma 3.2.

For the bipartite matching problem on $2N$ vertices, the structured
hinge loss (11) surrogate $\varphi$ is
$(\mathsf{C}_{\varphi,1},\mathsf{C}_{\varphi,2})$ -identifying
(Def. 3.1) with

| | $\displaystyle\mathsf{C}_{\varphi,1}=\frac{1}{N}~~\mbox{and}~~\mathsf{C}_{\varphi,2}=2.$
|

The important consequence of Lemma 3.2 is that
even when $k=\mathop{\rm card}(\mathcal{Y})=N!$ , aggregation-based methods can yield
consistency (via the structured hinge loss) once $m$ , the number of
aggregated labels, exceeds $O(N\log N)$ .
As one specialization, substituting these constants into
Corollary 3.3 for $k\geq 3$ , for all
measurable $f:\mathcal{X}\to\mathbb{R}^{d}$ such that

| | $\displaystyle R(f)-R^{\star}\geq 4\cdot\left({\frac{32\mathsf{C}_{\mathsf{MT}}^{4}}{m}\log\left({4k(2N+1)}\right)}\right)^{\frac{\alpha}{2(1-\alpha^{2})}},$
|

one has

| | $\displaystyle R(f)-R^{\star}\leq 16N(R_{\varphi,A_{n}}(f)-R_{\varphi,A_{n}}^{\star}).$
|

#### 3.3.3 Structured prediction

We return to the more general structured prediction setting, as
at the beginning of the preceding subsection.
Suppose the
decoder $\mathsf{d}$ can pick any class $y\in\mathcal{Y}$ , in that for
each $y\in\mathcal{Y}$ , the collection

| | $\displaystyle\mathcal{S}(y)\coloneqq\left\{s:\langle v(y),s\rangle>\langle v(\hat{y}),s\rangle,\mbox{for~}\hat{y}\neq y\right\}$
|

of selecting $s$
is non-empty.
For each $y\in\mathcal{Y}$ , define the identifiability gap

| | $\displaystyle\tau(y)\coloneqq\inf_{s\in\mathcal{S}(y)}\max_{y_{+},y_{-}\neq y}\frac{\ell(y_{+},y)}{\langle v(y)-v(y_{+}),s\rangle}\cdot\frac{\langle v(y)-v(y_{-}),s\rangle}{\ell(y_{-},y)}.$
|

We have the following identifiability guarantee.

###### Lemma 3.3.

For the structured prediction problem,
the max-margin (11)
surrogate $\varphi$ is $(\mathsf{C}_{\varphi,1},\mathsf{C}_{\varphi,2})$ -identifiable with

| | $\displaystyle\mathsf{C}_{\varphi,1}=\min_{\hat{y}\neq y}\ell(\hat{y},y),\qquad\mathsf{C}_{\varphi,2}=\max_{y\in\mathcal{Y}}\tau(y)+1.$
|

In particular, if $v(y)\in\{0,1\}^{d}$ and $\ell(\hat{y},y)=\frac{1}{2d}\left\|{v(\hat{y})-v(y)}\right\|_{1}$ , $\tau(y)=1$ for all $y$
and $\mathsf{C}_{\varphi,2}=2$ .

Completing the example, as in the binary matching case, we see
that nontrivial consistency guarantees hold once $m\geq\log\mathop{\rm card}(\mathcal{Y})$ .
As $0\leq\ell(\cdot,\cdot)\leq 1$ ,
Corollary 3.3 applies, which
yields for all measurable $f:\mathcal{X}\to\mathbb{R}^{d}$ that

| | $\displaystyle R(f)-R^{\star}\geq\left({\frac{32\mathsf{C}_{\mathsf{MT}}^{4}}{m}\log\left({4|\mathcal{Y}|\left(1+\frac{\max_{\hat{y}\in\mathcal{Y}}\tau(y)+1}{\min_{\hat{y}\neq y}\ell(\hat{y},y)}\right)}\right)}\right)^{\frac{\alpha}{2(1-\alpha^{2})}},$
|

implies

| | $\displaystyle R(f)-R^{\star}\leq\frac{8}{\min_{\hat{y}\neq y}\ell(\hat{y},y)}(R_{\varphi,A_{m}}(f)-R_{\varphi,A_{m}}^{\star}).$
|

## 4 Robustness and consistency for models

The previous section builds off of the now classical theory of surrogate
risk consistency, which assumes $\mathcal{F}$ to be the class of all
measurable functions.
The results there show that aggregation can allow us to “upgrade”
consistency so that even if a surrogate $\varphi$ is inconsistent for paired
(non-aggregated) data $(X,Y)$ , we can achieve level- $(\xi,\zeta)$
consistency (Def. 3.2) with
sufficient aggregation.
Here, we take a different view of the problem of consistency,
considering the consequences of optimizing over a restricted (often
parametric) hypothesis class $\mathcal{F}$ .
Of course, in a well-specified model, obtaining consistency with such a
restricted hypothesis class is no issue, but it is unrealistic to
assume such a brittle condition.
This gives rise to the long-standing challenge of quantifying surrogate
consistency when the hypothesis class contains only a subset of the
measurable functions [12, 31].
We point out a line of recent work develops $H$ -consistency bounds that link excess target risk to surrogate regret
within a regret hypothesis class $H$ , covering various setups [2, 3, 26, 28, 29]. These results are primarily surrogate-centric. Our contribution is orthogonal,
less hinged on the specific class and surrogate, and instead study how label aggregation improves consistency itself.
We tackle some of the issues around this, showing that
aggregating labels allows consistent estimates in scenarios where
consistency might otherwise fail.

### 4.1 Consistency failure for binary classification in
finite dimensions

To see how restricting the hypothesis class can change the problem
substantially even in well-understood cases, we consider binary
classification.
In this case, $\mathcal{Y}=\left\{\pm 1\right\}$ , and
we take the zero-one error $\ell(\mathsf{d}(s),y)=1\!\left\{ys\leq 0\right\}$ .
We consider a margin-based surrogate $\varphi(s,y)=\phi(sy)$ , where
$\phi:\mathbb{R}\to\mathbb{R}_{+}$ is convex, and as we have discussed,
$\varphi$ achieves both Fisher (i)
and uniform consistency (ii)
when $\mathcal{F}$ consists of all measurable functions if and only if
$\phi^{\prime}(0)<0$  [4].

Now we proceed to consider a restricted hypothesis class, showing in this
simple setting that classical consistency fails even when
optimal classifiers lie in $\mathcal{F}$ , in particular,
when $P$ is optimally predictable using $\mathcal{F}$ , meaning that

| | $\mathsf{sgn}(f(x))=\mathsf{sgn}(P(Y=1\mid X=x)-1/2).$
| | (12)

Let $\mathcal{X}=\mathbb{R}^{d}$ and take $\mathcal{F}=\{f_{\theta}\mid f_{\theta}(x)=\langle\theta,x\rangle\}_{\theta\in\mathbb{R}^{d}}$ to be the collection of linear
functionals of $x$ .
When $P$ is optimally predictable from using $\mathcal{F}$ , there
exists $\theta^{\star}$ satisfying
$\mathsf{sgn}(\langle\theta^{\star},x\rangle)=\mathsf{sgn}(P(Y=1\mid x)-\frac{1}{2})$ ,
and $f_{\theta^{\star}}$ minimizes $R(f)$ across all measurable functions.
In this case, we say that $P$ is optimally predictable along
$\theta^{\star}$ .
One might expect a margin-based surrogate $\varphi$ achieving Fisher
consistency in the classical setup should still consistent.
This fails.
Even more, for any nonnegative loss $\phi$ , there is a data distribution $P$
on $(X,Y)$ such that $\theta_{\varphi}=\mathop{\rm argmin}_{\theta}\mathbb{E}_{P}[\varphi(f_{\theta}(X),Y)]$ is essentially orthogonal to $\theta^{\star}$ :

###### Proposition 4.

For any $\epsilon>0$ and nonzero vector $\theta^{\star}\in\mathbb{R}^{d}$ , there
exists an $(X,Y)$ distribution $P$ , optimally
predictable
along $\theta^{\star}$ , such that for all

| | $\displaystyle\theta_{\varphi}\in\mathop{\rm argmin}_{\theta}R_{\varphi}(f_{\theta})=\mathop{\rm argmin}_{\theta}\mathbb{E}\left[\phi(f_{\theta}(X),Y)\right],$
|

we have $R(f_{\theta_{\varphi}})>R(f_{\theta^{\star}})$ and $|\cos\angle(\theta_{\varphi},\theta^{\star})|=|\langle\theta_{\varphi},\theta^{\star}\rangle|/(\left\|{\theta_{\varphi}}\right\|_{2}\left\|{\theta^{\star}}\right\|_{2})\leq\epsilon$ .

We postpone the proof to Appendix D.1.

Data aggregation methods provide one way to circumvent the inconsistency
Proposition 4 highlights.
To state the result, define the approximate minimizers

| | $\mathop{\epsilon\textup{-argmin}}g=\mathop{\epsilon\textup{-argmin}}_{\theta}g(\theta)\coloneqq\Big\{\theta\mid g(\theta)\leq\inf_{\theta}g(\theta)+\epsilon\Big\}.$
|

Suppose the data collection consists of independent samples $Z=(Y_{1},\ldots,Y_{m})$ and we take $A_{m}(Z)$ to be majority vote.
For a sequence $\epsilon_{m}$ take

| | $\displaystyle\theta_{m}\in\mathop{\epsilon_{m}\textup{-argmin}}_{\theta}R_{\varphi,A_{m}}(f_{\theta})=\mathop{\epsilon_{m}\textup{-argmin}}_{\theta}\mathbb{E}\left[{\phi(A_{m}(Y_{1},\ldots,Y_{m})f_{\theta}(X))}\right].$
|

Then as a corollary to the coming Theorem 2,
$f_{\theta_{m}}$ are asymptotically consistent when $m\to\infty$ .

###### Corollary 4.1.

Let $P$ be optimally predictable along $\theta^{\star}$ .
Then if $\epsilon_{m}\to 0$ as $m\to\infty$ ,
$R(f_{\theta_{m}})\to R(f_{\theta^{\star}})$
and $\cos\angle(\theta_{m},\theta^{\star})\to 1$ .

So without aggregation, surrogate risk minimization is (by
Proposition 4) essentially arbitrarily
incorrect when restricting to the class of linear predictors, while with
aggregation, we retain consistency.

### 4.2 Aggregation, consistency, and restricted hypothesis classes

As Proposition 4 shows,
surrogate risk consistency reposes quite fundamentally on
$\mathcal{F}$ containing all measurable functions.
We now consider multiclass classification problems, where
$\mathcal{Y}=\{1,\ldots,k\}$ , and in which $\mathcal{F}$ forms a
linear cone satisfying

| | $f(x)^{T}\mathbf{1}=0~~\mbox{and}~~tf\in\mathcal{F}~\mbox{for~}t>0~\mbox{if}~f\in\mathcal{F}.$
|

We consider the zero-one loss $\ell(y,y^{\prime})=1\!\left\{y\neq y^{\prime}\right\}$ and
$\mathsf{d}(s)=\mathop{\rm argmax}_{y\in[k]}s_{y}$ , making the restriction to predictors
normalized to have $f(x)^{T}\mathbf{1}=0$ immaterial.
Assume the surrogate $\varphi:\mathbb{R}^{k}\times[k]\to\mathbb{R}_{+}$ is
Fisher consistent (i) and satisfies the
limiting loss condition

| | $\varphi(s,y)\to 0~~\mbox{if}~~s_{y}-s_{j}\to+\infty~\mbox{for~all}~j\neq y.$
| | (13)

Many familiar surrogate losses are Fisher
consistent and satisfy (13), including the
multiclass logistic loss $\varphi(s,y)=\log(\sum_{j=1}^{k}e^{s_{j}-s_{y}})$
and any loss of the form

| | $\displaystyle\varphi(s,y)=\sum_{i\neq y}\phi(s_{y}-s_{i})$
|

for $\phi$ convex, non-increasing with $\phi^{\prime}(0)<0$ ,
and $\inf_{t}\phi(t)=0$ .
Zhang [51, Thm. 8] shows that any such loss is
consistent over the class $\mathcal{F}=\{f:\mathcal{X}\to\mathbb{R}^{k}\mid\mathbf{1}^{T}f=0\}$ .
Clearly, the margin-based binary setting in Sec. 4.1
falls into this scenario when we take $f(x)=(g(x),-g(x))$ for a
measurable $g$ .
Additionally, in a parametric setting when $\mathcal{X}=\mathbb{R}^{d}$ , if $\mathcal{F}$
consists of linear functions $f(x)=(\langle\theta_{1},x\rangle,\ldots,\langle\theta_{k},x\rangle)$ with $\sum_{i=1}^{k}\theta_{i}=0$ , then $\mathcal{F}$ is a (convex) cone. However, those examples typically require strong structural assumptions such as convexity for consistency without aggregation. As we shall see next, with the minimal limiting loss condition (13), we can achieve asymptotic consistency by aggregation.

Extending the definition (12)
of optimal predictability in the obvious way,
we shall say $\mathcal{F}$ can optimally predict $P$ if there exists
$f\in\mathcal{F}$ , $f:\mathcal{X}\to\mathbb{R}^{k}$ , for which

| | $\mathop{\rm argmax}_{y}f_{y}(x)\in\mathop{\rm argmax}_{y}P(Y=y\mid X=x)~~\mbox{for~all~}x.$
|

The next theorem shows if $\mathcal{Z}=\mathcal{Y}^{m}$ , we aggregate via
majority vote $A_{m}$ , and there is a unique $y^{\star}(x)=\mathop{\rm argmax}_{y}P(Y=y\mid x)$ , then surrogate risk minimization is consistent whenever $\mathcal{F}$ can
optimally predict $P$ .

###### Theorem 2.

Let $\mathcal{F}$ be a cone that optimally predicts $P$ ,
and assume that the
minimal excess risk (8)
satisfies $P(\Delta(X)>0)=1$ .
Let $\epsilon_{m}\geq 0$ satisfy $\epsilon_{m}\to 0$ .
Then for any sequence

| | $\displaystyle f_{m}\in\mathop{\epsilon_{m}\textup{-argmin}}_{f\in\mathcal{F}}R_{\varphi,A_{m}}(f)=\mathop{\epsilon_{m}\textup{-argmin}}_{f\in\mathcal{F}}\mathbb{E}\left[{\varphi(f(X),A_{m}(Y_{1},\ldots,Y_{m}))}\right],$
|

we have $R(f_{m})\to R^{\star}$ .

See Appendix D.2 for a proof.

Theorem 2 shows that for a broad class of surrogate
problems with a hypothesis class $\mathcal{F}$ that forms a linear cone, we
can achieve consistency asymptotically by aggregation as $m\to\infty$ .
In contrast, as Proposition 4 shows, even in the
“simple” case of binary classification, consistency may fail over
subclasses $\mathcal{F}$ , even when they include the optimal predictor, and the
surrogate can be arbitrarily uninformative.

### 4.3 On finite-dimensional multiclass classification

The final technical content of this paper
considers a multiclass scenario in which $\mathcal{X}=\mathbb{R}^{d}$ , $\mathcal{Y}=[k]$ ,
and we use linear predictors,
but where the predictive model may be mis-specified.
These results will provide a more nuanced and specialized view
of the situation than the negative results in
Proposition 4 and the consistency guarantee
in Theorem 2.
We will show that even when the problem is optimally predictable and the
linear model is well-specified on all except an $\epsilon$ -fraction of
data, surrogate risks based only on $(X,Y)$ pairs are inconsistent;
majority vote-based methods, however, will recover the optimal linear
predictor.

To set the stage, let
$\Theta=[\theta_{1}~\cdots~\theta_{k-1}]\in\mathbb{R}^{d\times k-1}$ ,
and let the labels follow a categorical distribution $Y\mid X=x\sim\mathsf{Cat}(p_{\Theta}(x))$ , where $p_{\Theta}(x)\in\mathbb{R}^{k}_{+}$ satisfies
$\mathbf{1}^{\top}p_{\Theta}(x)=1$ and

| | $\displaystyle p_{\Theta}(x)=\begin{bmatrix}\sigma(\langle\theta_{1},x\rangle,\cdots,\langle\theta_{k-1},x\rangle)\\
1-\mathbf{1}^{\top}\sigma(\langle\theta_{1},x\rangle,\cdots,\langle\theta_{k-1},x\rangle)\end{bmatrix}$
|

for a link $\sigma:\mathbb{R}^{k-1}\to\mathbb{R}^{k-1}_{+}$ ,
$\mathbf{1}^{\top}\sigma\leq 1$ .
We assume $\sigma$ satisfies the consistency condition that for $t\in\mathbb{R}^{k-1}$ , setting $t_{k}=0$ and $\sigma_{k}(t)=1-\mathbf{1}^{\top}\sigma(t)$ ,

| | $t_{y}=\max_{1\leq i\leq k}t_{i}~~\mbox{if and only if}~~\sigma_{y}(t)=\max_{1\leq i\leq k}\sigma_{i}(t).$
| | (14)

One standard example is multiclass logistic regression, where

| | $\displaystyle\sigma_{i}^{\textup{lr}}(t)$ | $\displaystyle=\frac{e^{t_{i}}}{1+e^{t_{1}}+\cdots+e^{t_{k-1}}}.$
|

Let the multilabeled dataset be $\{(X_{i};(Y_{1i},\cdots,Y_{im}))\}_{i=1}^{n}$ with repeated sampling $Y_{ij}\mid X_{i}\stackrel{{\scriptstyle\rm iid}}{{\sim}}\mathsf{Cat}(p_{\Theta^{\star}}(X_{i}))$ , and $Y_{i}^{+}=A_{m}(Y_{i1},\cdots,Y_{im})$ be
the majority vote with ties broken arbitrarily.
Then $Y_{i}^{+}\mid X_{i}\sim\mathsf{Cat}(p_{\Theta^{\star},m}(X_{i}))$ , where if
$\rho_{m}(t)$ denotes the distribution of majority vote on $m$ items with
initial probabilities $\sigma(t)\in\mathbb{R}^{k-1}_{+}$ , then

| | $\displaystyle p_{\Theta,m}(x)=\begin{bmatrix}\rho_{m}(\langle\theta_{1},x\rangle,\cdots,\langle\theta_{k-1},x\rangle)\\
1-\mathbf{1}^{\top}\rho_{m}(\langle\theta_{1},x\rangle,\cdots,\langle\theta_{k-1},x\rangle)\end{bmatrix}.$
|

It is evident that $\rho_{m}$ satisfies link
consistency (14).
Consider fitting a logistic regression with loss

| | $\displaystyle\varphi(\Theta^{\top}x,y)=-\langle\theta_{y},x\rangle+\log\left({1+\sum_{i=1}^{k-1}\exp(\langle\theta_{i},x\rangle)}\right),$
| | (15)

with the convention that $\theta_{k}=\mathbf{0}$ ,
and let $L_{m}(\Theta)=\mathbb{E}[\varphi(\Theta^{\top}X,Y_{m}^{+})]$ be the
logistic loss with $m$ -majority vote.
Then

| | $\displaystyle\nabla_{\theta_{i}}\varphi(\Theta^{\top}x,y)=-x(1\!\left\{y=i\right\}-\sigma_{i}^{\textup{lr}}(\Theta^{\top}x)),$
|

so that the $\Theta_{m}$ minimizing $L_{m}$ satisfies

| | $\displaystyle\nabla_{\Theta}L_{m}(\Theta_{m})=\mathbb{E}\left[{X\left({\sigma^{\textup{lr}}(\Theta_{m}^{\top}X)-\rho_{m}({\Theta^{\star}}^{\top}X)}\right)^{\top}}\right]=0.$
|

Standard results in statistics guarantees both consistency and efficiency
when the model is well-specified without aggregation, and when $m\geq 1$ and
$k=2$ , Cheng et al. [7, Prop. 3] show there exists $t_{m}>0$ such that
$\Theta_{m}=t_{m}\Theta^{\star}$
if $X\sim\mathsf{N}(0,I_{d})$ even with a mis-specified link.
This implies that in binary classification, even if the link function is
incorrect, we can still achieve consistent classification regardless of
the aggregation level $m$ , as the direction $\Theta^{\star}/\left\|{\Theta^{\star}}\right\|_{2}$
determines consistency.
However, as soon as $k\geq 3$ and the true link is slightly
mis-specified, risk consistency fails.
Fixing a set $\mathcal{T}_{\epsilon}\subset\mathbb{R}^{k-1}$ with Lebesgue
measure $\epsilon$ , consider

| | $\displaystyle\sigma^{\epsilon}(t)=\sigma^{\textup{lr}}(t){1}\{t\not\in\mathcal{T}_{\epsilon}\}+\frac{1}{k}\mathbf{1}\cdot 1\!\left\{t\in\mathcal{T}_{\epsilon}\right\},$
|

which defines a distribution on $Y\in\{1,\ldots,k\}$
conditional on $t\in\mathbb{R}^{k-1}$ that samples
$Y\sim\sigma^{\textup{lr}}(t)$ if $t\in\mathcal{T}_{\epsilon}$ and
uniformly otherwise.
Clearly $\sigma^{\epsilon}$ satisfies the link consistency
condition (14) and is optimally
predictable (12).
For $\epsilon>0$ , define

| | $L_{m,\epsilon}(\Theta)\coloneqq\mathbb{E}[\mathbb{E}_{\sigma^{\epsilon}}[\varphi(\Theta^{\top}X,Y_{m}^{+})\mid X]]$
|

to be the (population) logistic loss, based on $m$ -majority vote,
when $Y\mid X=x\sim\sigma^{\epsilon}({\Theta^{\star}}^{\top}x)$ .
Let $\Theta_{m}(\epsilon)=\mathop{\rm argmin}_{\Theta}L_{m,\epsilon}(\Theta)$ .
Evidently, $\Theta_{1}(0)=\Theta^{\star}$ ; nonetheless, the next result shows
that for arbitrarily small $\epsilon>0$ , consistency fails without
aggregation.

###### Proposition 5.

Let $k\geq 3$ , $\Sigma=I$ .
Assume that for $Z\sim\mathsf{N}(0,I_{k-1})$ , the linear mapping $M\mapsto DM\coloneqq\mathbb{E}[Z(\nabla\sigma^{\textup{lr}}({\Theta^{\star}}^{\top}Z)MZ)^{\top}]$ is
invertible.
Then there exists $\epsilon_{0}>0$ such that for any $\epsilon\in(0,\epsilon_{0})$ , there is a set $\mathcal{T}_{\epsilon}$ with
Lebesgue measure at most $\epsilon$ and for which

| | $\Theta_{1}(\epsilon)/\left\|{\Theta_{1}(\epsilon)}\right\|\neq\Theta^{\star}/\left\|{\Theta^{\star}}\right\|.$
|

We postpone the proof to
Appendix E.1.

Majority vote, however, can address this inconsistency as $m\to\infty$
without any assumptions on the true link $\sigma$ except that it satisfies
the consistency condition (14).
Indeed, letting $L_{m,\sigma}(\Theta)=\mathbb{E}[\mathbb{E}_{\sigma}[\varphi(\Theta^{\top}X,Y_{m}^{+})\mid X]]$ and $\Theta_{m}=\mathop{\rm argmin}L_{m,\sigma}(\Theta)$ , we have
the following

###### Theorem 3.

Let $\Theta^{\star}$ have decomposition $\Theta^{\star}=U^{\star}T^{\star}$ , where $U^{\star}\in\mathbb{R}^{d\times(k-1)}$ is orthogonal and $T^{\star}\in\mathbb{R}^{(k-1)\times(k-1)}$ is nonsingular.
Then there exists $T_{m}\in\mathbb{R}^{(k-1)\times(k-1)}$ such that $\Theta_{m}=U^{\star}T_{m}$ for every $m$ , and as $m\to\infty$ ,

| | $\displaystyle\|T_{m}\|\to\infty~~\mbox{and}~~T_{m}/\|T_{m}\|\to T^{\star}/\|T^{\star}\|.$
|

See Appendix E.2 for a proof.

Theorem 3 shows more evidence for the robustness
properties of label aggregation, providing asymptotic consistency even in
mis-specified models so long as there is some link function
describing the relationship between $X$ and $Y$ .
The robustness is striking when $k\geq 3$ : as
Proposition 5 highlights,
methods without label aggregation are generally inconsistent.

## 5 Numerical illustrations

In this section, we provide two experiments that corroborate the theory: (i) a non-parametric example where aggregation amplifies surrogate consistency, and (ii) a finite-dimensional multiclass example where aggregation recovers asymptotic consistency under link mis-specification.

##### Surrogate consistency amplification for truncated quadratic loss.

In binary classification, consider the classical margin-based surrogate $\varphi(f(x),y)=\phi(yf(x))$ where $\phi(\delta)=\max\{1-\delta,0\}^{2}$ is the truncated quadratic loss. By Bartlett et al. [4, Example 2], the calibration function $\psi$ in Corollary 3.1 is exactly $\psi(\epsilon)=\epsilon^{2}$ . Leveraging label aggregation through majority vote with $m$ labels, we can apply Lemma 3.1 taking $\delta=1$ . In the regime with no low-noise condition (i.e. $\alpha=0,\mathsf{C}_{\mathsf{MT}}=1$ ), taking

| | $\displaystyle\psi_{m}(\epsilon)$ | $\displaystyle=\frac{\phi(0)-\phi(\delta)}{16}\left(\epsilon-\sqrt{\frac{32}{m}\log\left({\frac{8(\phi(-\delta)+\phi(0)-\phi(\delta))}{\phi(0)-\phi(\delta)}}\right)}\right)_{+}=\frac{1}{16}\left(\epsilon-\sqrt{\frac{32}{m}\log 40}\right)_{+},$
|

we have parallel comparison inequalities for non-aggregated data and majority-vote aggregation as

| | $\displaystyle\psi(R(f)-R^{\star})\leq R_{\varphi}(f)-R_{\varphi}^{\star},\qquad\psi_{m}(R(f)-R^{\star})\leq R_{\varphi,A_{m}}(f)-R_{\varphi,A_{m}}^{\star}.$
|

Numerically (cf. Fig. 1), the aggregated curve transitions from quadratic toward linear behavior as $m\to\infty$ , showing label aggregation upgrades surrogate consistency. Nonetheless, $\psi_{m}$ are generally conservative as they are from the general result in Theorem 1.

Figure 1: Numerical illustrations for truncated quadratic surrogate $\phi(\delta)=\max\{1-\delta,0\}^{2}$ , showing comparison inequalities for non-aggregated data $\psi$ (optimal) vs. majority vote aggregation $\psi_{m}$ (suboptimal) when $m\in\{2^{16},2^{20},2^{24},2^{28},2^{32}\}$ .

##### Asymptotic consistency for mis-specified logistic model.

In the second example, we consider logistic regression in Sec. 4.3 when $k=3$ and $d=2$ . Let the loss function $\varphi$ be the logistic loss in Eq. (15), with a corrupted logistic link defined as

| | $\displaystyle\tilde{\sigma}(t)$ | $\displaystyle=\sigma^{\textup{lr}}(t){1}\{\left\|{t}\right\|\leq r\}+\sigma^{\textup{lr}}(\alpha t){1}\{\left\|{t}\right\|>r\},$
|

a piecewise link that matches softmax near the origin and shrinks scores outside a radius $r$ . The numerical results are in Fig. 2, showing asymptotic consistency as $m\to\infty$ with synthetic dataset from the corrupted link.

|
|

| (i) Experiment with constant dataset size $n$ | (ii) Experiment with constant label size $nm$ .

Figure 2: Numerical illustrations for mis-specified logistic models when $d=2$ and $k=3$ . The true parameters are $\theta_{1}^{\star}=(0,1)^{\top},\theta_{2}^{\star}=(1,1)^{\top}$ . The corrupted link parameters are $r=3$ and $\alpha=0.1$ . The plot shows the error of classification rule $\left\|\|T_{m}\|-T^{\star}/\|T^{\star}\|\right\|$ vs. number of labels $m\in\{1,4,16,64,256,1024\}$ . (i) In the left panel, for each value of $m$ , the synthetic dataset consists of $n=10{,}000$ multilabeled observations $(X_{i},(Y_{i1},\ldots,Y_{im}))$ sampled from $\tilde{\sigma}$ . (ii) In the right panel, for each $m$ , we set $n=640{,}000/m$ so that the total number of labels remains constant across different values of $m$ . We report a $95\%$ error bar on $T=100$ trials for each $m$ .

## 6 Discussion and future work

The question of whether and how to clean data has
animated much of the research discussion around dataset collection.
Cheng, Asi and Duchi [7] provide a discussion of these issues, highlighting that
there appears to be a phenomenon that using non-aggregated data—all
available labels—leads to better statistical efficiency when models have
the power to fully represent all uncertainty, but otherwise, data cleaning
appears to be more robust.
In a similar vein, Dorner and Hardt [11] argue that, in a validation setting of
comparing binary classifiers, it is better to use more noisy labels rather
than cleaned variants.
This paper contributes to this dialogue by providing evidence for both
fundamental limits to using un-cleaned, un-aggregated label information
in supervised learning while highlighting robustness improvements
that come from label cleaning.
Nonetheless, many questions remain.

##### Finite $m$ results and fundamental limits.

Many of the consistency results we present repose on taking a limit
as $m$ , the number of labels aggregated, tends to infinity.
While at some level, the purpose of this paper is to highlight ways
in which label aggregation can improve robustness, it is
perhaps unsatisfying to rely on this asymptotic setting.
In the context of ranking (Sec. 3.1), we
can provide explicit consistency guarantees at a finite $m$ ,
but developing this further provides one of the most
natural and (we believe) important avenues for future work.
Providing a surrogate consistency theory that depends both on
the loss pairs $(\ell,\varphi)$ and the available label
count $m$ would be interesting; for example,
in the context of ranking in Sec. 3.1,
if we wish to look at second or third-order comparisons
of items (e.g., powers $C_{x}^{p}$ , as Keener [20] suggests),
do we require increasing label counts $m$ ?
Precisely delineating those problems that require label cleaning and
aggregation from those that do not represents a central challenge here.

##### Fundamental limits of the noise condition number.

Our work relies on the noise condition number (9),
$\kappa(X)$ , to characterize comparison inequalities for
multiclassification problems, hinting at the difficulty beyond
binary classification, where trivially $\kappa(X)=1$ .
The condition number can still be large even when the Mammen-Tsybakov noise
level (Nα) is low—i.e. $\alpha\approx 1$ —in cases
beyond binary classification.
This is a consequence of the
minimal assumptions on the surrogate in our setting, and it would
be beneficial to identify
connections between the loss $\ell$ and surrogate $\varphi$
that more closely capture problem difficulty.
A more precise delineation of
fundamental limits by constructing explicit failure modes will
also yield more insights into fitting predictive models.

##### Behavior in mis-specified models.

Our results on mis-specified models, especially those in
Section 4.3, require optimal
predictability (12), that is, that a
Bayes-optimal classifier lie in $\mathcal{F}$ .
While classical surrogate consistency results provably fail even in this
case—and methods based on aggregated labels can evidently succeed—moving
beyond such restricted scenarios seems a fruitful and interesting direction.
Nguyen et al. [31], followed by Duchi et al. [12], identify one
direction here, showing that in binary and multiclass classification
(respectively), jointly inferring a predictor $f$ and a data representation
for $x$ requires that surrogates $\varphi$ take a particular form depending
on the task loss $\ell$ .
These still repose on infinitely powerful decision rules $f$ , however,
so we need new approaches.

##### Restricted class comparison inequality with label aggregation.

Our paper investigates how label aggregation mitigates regret induced by surrogate losses
through comparison inequalities in the non-parametric setting, and establishes statistical consistency in the parametric regime. Motivated by recent developments in $H$ -consistency [2, 28, 27, 29], it would be intriguing to extend these techniques to analyze comparison inequalities under restricted hypothesis classes. In particular, Mao et al. [27] shows that for smooth convex loss, quadratic growth of $\psi$ is rate optimal—it will be interesting to apply aggregation
and upgrade consistency in the context of $\mathcal{H}$ -consistency. Such an extension could offer a unified framework linking the advantages of aggregation across parametric and non-parametric settings, while shedding light on how the “size” of the hypothesis class
$\mathcal{F}$ influences the effectiveness of aggregation.
