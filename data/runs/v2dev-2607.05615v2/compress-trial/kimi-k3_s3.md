Let $\mathcal{M}$ be an $L$-layer decoder-only transformer. For input $\mathbf{x}=(x_{1},\ldots,x_{T})$, let $\mathbf{h}_{t}^{(\ell)}\in\mathbb{R}^{d}$ be the residual-stream activation at layer $\ell$, position $t$. We inject a steering signal $\mathbf{s}_{t}\in\mathbb{R}^{d}$ (Sec. 3.2) at a single intermediate layer $\ell$:

| | $\tilde{\mathbf{h}}_{t}^{(\ell)}=\mathbf{h}_{t}^{(\ell)}+\mathbf{s}_{t},$
| | (1)

### 3.1 Steering Direction Discovery via SAE

An SAE trained on layer-$\ell$ residual activations maps $\mathbf{h}$ to a sparse feature vector $f(\mathbf{h})\in\mathbb{R}^{d_{\mathrm{sae}}}$, where $f_i(\mathbf{h})$ is the activation of feature $i$; the decoder $\mathbf{W}_{\mathrm{dec}}\in\mathbb{R}^{d_{\mathrm{sae}}\times d}$ reconstructs $\mathbf{h}$. Given a positive set $\mathcal{P}$ exhibiting the target behavior and a matched neutral set $\mathcal{N}$, compute each feature's mean activation differential:

| | $\Delta_{i}=\frac{1}{|\mathcal{P}|}\sum_{x\in\mathcal{P}}f_{i}(x)-\frac{1}{|\mathcal{N}|}\sum_{x\in\mathcal{N}}f_{i}(x).$
| | (2)

Select the top-$K$ features by $\Delta_i$, form directions $\mathbf{v}_{k}=\mathbf{W}_{\mathrm{dec},i_{k}}$ ($k=1,\ldots,K$), and aggregate into a unit-norm steering direction $\mathbf{v}=\sum_{k}\mathbf{v}_{k}/\lVert\sum_{k}\mathbf{v}_{k}\rVert$.

Figure 1: Steering effectiveness of STS and SBS versus intervention ratio $p$, measured as behavioral gain over no steering. Dashed lines are static baselines (prompting, full- and no steering); shaded bands show standard deviation.

### 3.2 Stochastic Activation Steering

The steering signal is $\mathbf{s}_{t}=m_{t}\cdot\alpha\,\mathbf{v}$, with magnitude $\alpha\in\mathbb{R}$, unit-norm direction $\mathbf{v}$, and binary gate $m_{t}\in\{0,1\}$. Full Steering (FS) sets $m_t=1$ everywhere; we instead make the gate random to control the fraction of steered positions.

#### Stochastic Token Steering (STS).

Sample an independent gate per position:

| | $\tilde{\mathbf{h}}_{t}^{(\ell)}=\mathbf{h}_{t}^{(\ell)}+m_{t}\cdot\alpha\,\mathbf{v},~m_{t}\stackrel{{\scriptstyle\text{i.i.d.}}}{{\sim}}\mathrm{Bernoulli}(p).$
| | (3)

In expectation a fraction $p$ of positions are steered; $p$ interpolates between no intervention ($p{=}0$) and full steering ($p{=}1$).

#### Stochastic Block Steering (SBS).

Treat a leading window of size $W\leq T$ as one unit, with a single gate $m\sim\mathrm{Bernoulli}(p)$ per sequence:

| | $\tilde{\mathbf{h}}_{t}^{(\ell)}=\begin{cases}\mathbf{h}_{t}^{(\ell)}+m\cdot\alpha\,\mathbf{v}&t\leq W,\\[2.0pt]
\mathbf{h}_{t}^{(\ell)}&t>W,\end{cases}$
| | (4)

With probability $p$ the whole window is steered; with probability $1{-}p$ it is untouched.