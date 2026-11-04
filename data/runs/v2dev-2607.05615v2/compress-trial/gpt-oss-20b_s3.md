Let $\mathcal{M}$ be an $L$‑layer decoder‑only transformer. For a tokenized input $\mathbf{x}=(x_{1},\ldots,x_{T})$, let $\mathbf{h}_{t}^{(\ell)}\in\mathbb{R}^{d}$ be the residual‑stream activation at layer $\ell$ and position $t$. We inject a steering signal into the residual stream at a single intermediate layer $\ell$:

\[
\tilde{\mathbf{h}}_{t}^{(\ell)}=\mathbf{h}_{t}^{(\ell)}+\mathbf{s}_{t},\tag{1}
\]

where $\mathbf{s}_{t}\in\mathbb{R}^{d}$ is defined in Sec. 3.2.

---

### 3.1 Steering Direction Discovery via SAE

A Sparse Autoencoder (SAE) is trained on residual‑stream activations at layer $\ell$. The encoder maps an activation $\mathbf{h}$ to a sparse feature vector $f(\mathbf{h})\in\mathbb{R}^{d_{\mathrm{sae}}}$; $f_{i}(\mathbf{h})$ is the activation of learned feature $i$. The decoder $\mathbf{W}_{\mathrm{dec}}\in\mathbb{R}^{d_{\mathrm{sae}}\times d}$ reconstructs $\mathbf{h}$ from these features. To isolate behavior‑discriminative features, we encode a positive set $\mathcal{P}$ exhibiting the target behavior and a neutral set $\mathcal{N}$ of matched controls, and compute the mean activation differential of each feature:

\[
\Delta_{i}=\frac{1}{|\mathcal{P}|}\sum_{x\in\mathcal{P}}f_{i}(x)-\frac{1}{|\mathcal{N}|}\sum_{x\in\mathcal{N}}f_{i}(x).\tag{2}
\]

Select the top‑$K$ features by $\Delta_{i}$ and form feature directions $\mathbf{v}_{k}=\mathbf{W}_{\mathrm{dec},i_{k}}$ for $k=1,\ldots,K$, where $i_{k}$ indexes the $k$‑th selected feature. Aggregate into a single unit‑norm steering direction

\[
\mathbf{v}=\frac{\sum_{k=1}^{K}\mathbf{v}_{k}}{\bigl\lVert\sum_{k=1}^{K}\mathbf{v}_{k}\bigr\rVert}.\tag{3}
\]

---

#### Figure 1
Steering effectiveness of STS and SBS versus intervention ratio $p$, measured as behavioral gain over no steering. Dashed lines are static baselines (prompting, full‑ and no steering); shaded bands show standard deviation ±2.0.

---

### 3.2 Stochastic Activation Steering

The steering signal at position $t$ takes the form

\[
\mathbf{s}_{t}=m_{t}\cdot\alpha\,\mathbf{v},
\]

where $\alpha\in\mathbb{R}$ is the intervention magnitude, $\mathbf{v}$ is the unit‑norm steering direction, and $m_{t}\in\{0,1\}$ is a binary gate deciding whether position $t$ is steered. Full Steering (FS) sets $m_{t}=1$ at every position.

#### Stochastic Token Steering (STS)

At each position $t$, sample an independent gate $m_{t}\sim\mathrm{Bernoulli}(p)$:

\[
\tilde{\mathbf{h}}_{t}^{(\ell)}=\mathbf{h}_{t}^{(\ell)}+m_{t}\cdot\alpha\,\mathbf{v},\qquad m_{t}\stackrel{\text{i.i.d.}}\sim\mathrm{Bernoulli}(p).\tag{4}
\]

In expectation, a fraction $p$ of positions receive the signal; $p$ interpolates smoothly between no intervention ($p=0$) and full steering ($p=1$).

#### Stochastic Block Steering (SBS)

Treat a leading prompt window of size $W\le T$ as an atomic steering unit. Sample a single gate $m\sim\mathrm{Bernoulli}(p)$ per sequence:

\[
\tilde{\mathbf{h}}_{t}^{(\ell)}=
\begin{cases}
\mathbf{h}_{t}^{(\ell)}+m\cdot\alpha\,\mathbf{v}, & t\le W,\\[2pt]
\mathbf{h}_{t}^{(\ell)}, & t>W,
\end{cases}\tag{5}
\]

so with probability $p$ the entire window is steered coherently; with probability $1-p$ it is left intact.