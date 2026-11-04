Let $\mathcal{M}$ be an $L$-layer decoder-only transformer. For tokenized input $\mathbf{x}=(x_{1},\ldots,x_{T})$, let $\mathbf{h}_{t}^{(\ell)}\in\mathbb{R}^{d}$ be the residual-stream activation at layer $\ell$ and position $t$. Inject a steering signal at one intermediate layer $\ell$:

(1) $\tilde{\mathbf{h}}_{t}^{(\ell)}=\mathbf{h}_{t}^{(\ell)}+\mathbf{s}_{t},$

where $\mathbf{s}_{t}\in\mathbb{R}^{d}$ is defined in Sec. 3.2.

### 3.1 Steering Direction Discovery via SAE

A Sparse Autoencoder (SAE) trained on layer-$\ell$ residual-stream activations identifies target-behavior directions. Its encoder maps $\mathbf{h}$ to sparse $f(\mathbf{h})\in\mathbb{R}^{d_{\mathrm{sae}}}$, with $f_i(\mathbf{h})$ the activation of learned feature $i$; its decoder $\mathbf{W}_{\mathrm{dec}}\in\mathbb{R}^{d_{\mathrm{sae}}\times d}$ reconstructs $\mathbf{h}$ from $f(\mathbf{h})$. For positive set $\mathcal{P}$ exhibiting target behavior and neutral set $\mathcal{N}$ of matched controls,

(2) $\Delta_{i}=\frac{1}{|\mathcal{P}|}\sum_{x\in\mathcal{P}}f_{i}(x)-\frac{1}{|\mathcal{N}|}\sum_{x\in\mathcal{N}}f_{i}(x).$

Select top-$K$ features by $\Delta_i$, set $\mathbf{v}_{k}=\mathbf{W}_{\mathrm{dec},i_{k}}$ for $k=1,\ldots,K$ with $i_k$ the $k$-th selected feature, and aggregate into unit-norm

$\mathbf{v}=\sum_{k=1}^{K}\mathbf{v}_{k}/\bigl\lVert\sum_{k=1}^{K}\mathbf{v}_{k}\bigr\rVert.$

Figure 1: STS/SBS effectiveness versus intervention ratio $p$ as behavioral gain over no steering; dashed lines: static baselines (prompting, full- and no steering); shaded bands: standard deviation.

### 3.2 Stochastic Activation Steering

At position $t$, $\mathbf{s}_{t}=m_{t}\cdot\alpha\,\mathbf{v}$, with $\alpha\in\mathbb{R}$ intervention magnitude, unit-norm $\mathbf{v}$, and binary gate $m_t\in\{0,1\}$ deciding whether $t$ is steered. The standard approach, Full Steering (FS), sets $m_t=1$ at every position. Two stochastic alternatives treat the gate as a random variable, control the fraction of steered positions, and trade off behavioral strength against fluency.

#### Stochastic Token Steering (STS).

At each $t$, sample independent $m_t$:

(3) $\tilde{\mathbf{h}}_{t}^{(\ell)}=\mathbf{h}_{t}^{(\ell)}+m_{t}\cdot\alpha\,\mathbf{v},~m_{t}\stackrel{{\scriptstyle\text{i.i.d.}}}{{\sim}}\mathrm{Bernoulli}(p).$

In expectation, $p$ fraction of positions receive the signal and the rest are unperturbed; $p$ interpolates between no intervention ($p{=}0$) and full steering ($p{=}1$).

#### Stochastic Block Steering (SBS).

A leading prompt window of size $W\leq T$ is an atomic steering unit. Sample one gate $m\sim\mathrm{Bernoulli}(p)$ per sequence:

(4) $\tilde{\mathbf{h}}_{t}^{(\ell)}=\begin{cases}\mathbf{h}_{t}^{(\ell)}+m\cdot\alpha\,\mathbf{v}&t\leq W,\\[2.0pt]
\mathbf{h}_{t}^{(\ell)}&t>W,\end{cases}$

With probability $p$ the window is steered coherently; with probability $1-p$ it is intact.