Journal of Machine Learning Research 15 (2014) 1929–1958  
Submitted 11/13; Published 6/14

# Dropout: A Simple Way to Prevent Neural Networks from Overfitting

| Author | Email |
|---|---|
| Nitish Srivastava | nitish@cs.toronto.edu |
| Geoffrey Hinton | hinton@cs.toronto.edu |
| Alex Krizhevsky | kriz@cs.toronto.edu |
| Ilya Sutskever | ilya@cs.toronto.edu |
| Ruslan Salakhutdinov | rsalakhu@cs.toronto.edu |

Department of Computer Science, University of Toronto  
10 Kings College Road, Rm 3302  
Toronto, Ontario, M5S 3G4, Canada

Editor: Yoshua Bengio

© 2014 Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever and Ruslan Salakhutdinov.

Original-page map, including continuations: 1930—Figure 1 and Introduction; 1931—Figure 2 and Introduction; 1932—Introduction ending, §2 and §3 opening; 1933—§3 continuation and §4 opening; 1934—Figure 3, §4 ending and §5.1 opening; 1935—§5.1 continuation, §5.2 and §6 opening; 1936—§6 continuation, Table 1 and §6.1.1 opening with Table 2; 1937—§6.1.1 continuation, Figure 4 and §6.1.2 opening; 1938—Table 3, §6.1.2 continuation and §6.1.3; 1939—Figure 5, Table 4 and §6.1.4 opening; 1940—Figure 6, Tables 5–6, §6.1.4 continuation and §6.2 opening; 1941—§6.2 continuation, Table 7, §6.3 and §6.4 opening; 1942—Table 8, §6.4 continuation, §6.5 and §7 opening; 1943—Table 9, §7 continuation and §7.1 opening with Figure 7; 1944—§7.1 continuation and §7.2 opening with Figure 8; 1945—§7.2 ending and §7.3 with Figure 9; 1946—§§7.4–7.5 openings with Figures 10–11; 1947—§7.5 continuation, §8 and §8.1 opening; 1948—Figure 12, §8.1 continuation, §8.2 and §8.3 opening; 1949—Figure 13, §8.3 continuation, §8.4 and §9 opening; 1950—§9 continuation and §§9.1–9.2; 1951—Table 10, §10 and §11 opening; 1952—§11 continuation, Acknowledgments and Appendix A through A.1; 1953—A.2–A.4, Appendix B and B.1 opening; 1954—B.1 continuation, B.2 and B.3 opening; 1955—B.3 continuation and B.4–B.6.

## Abstract

Large, parameter-rich deep neural nets are powerful machine-learning systems whose overfitting and slow use obstruct test-time combination of independently trained large models; the technique (QX2: dropout) temporarily removes randomly selected units and their connections during training, inhibits excessive co-adaptation, and samples exponentially many parameter-sharing thinned networks. A single unthinned network with reduced weights approximates their averaged predictions. Relative to other regularizers, this procedure substantially reduces overfitting and produces major improvements; supervised vision, speech-recognition, document-classification and computational-biology experiments attain state-of-the-art performance on many benchmarks.

**Keywords:** neural networks, regularization, model combination, deep learning

## 1. Introduction

With unlimited computation, posterior-weighted averaging over every parameter setting is the best regularization of a fixed-sized model; simple or small models sometimes admit good approximations (Xiong et al., 2011; Salakhutdinov and Mnih, 2008), whereas the proposed less expensive construction approximates an equally weighted geometric mean of exponentially many learned, parameter-sharing models’ predictions. An \(n\)-unit network represents \(2^n\) possible thinned networks, retaining only \(O(n^2)\), or fewer, parameters through weight sharing; a newly sampled member is trained for every presentation of every training case, leaving each particular member trained rarely, if ever.

Multilayer non-linear hidden representations express complicated input–output relationships; limited-data sampling noise supplies relationships present in training but absent in genuine test data, including test data from the same distribution. Established responses include stopping when validation performance deteriorates, soft weight sharing (Nowlan and Hinton, 1992), and weight penalties whose familiar instances (O3U: L1 regularization) and (Y5F: L2 regularization) constrain learned parameters rather than averaging every possible parameter setting.

Model combination nearly always improves machine-learning performance, but separately training many large nets makes output averaging prohibitively expensive; useful diversity requires different architectures or different training data, architecture-specific optimal hyperparameters are difficult to find, each large-net training run consumes substantial computation, and the large data requirements can preclude training different networks on different subsets. Even successful training of many such models leaves their joint inference infeasible for rapid-response applications.

The practical inference rule (T6R: weight scaling) uses one unthinned network rather than explicitly evaluating exponentially many members, while a more expensive alternative (X9M: Monte-Carlo model averaging) averages sampled members’ predictions; the former gives significantly lower generalization error than other regularization methods across varied classification problems. The same unit-removal construction extends beyond feed-forward nets to graphical models, including Boltzmann Machines, with a standard model family (U2L: Restricted Boltzmann Machines, RBM) admitting the augmented form (B8T: Dropout RBM) without abandoning shared parameters.

**Figure 1.** Standard neural net, left: two hidden layers. QX2-produced thinned net, right: crossed units removed, including their incoming and outgoing connections.

Paper map: §2 motivation; §3 previous work; §4 formal QX2 model; §5 training algorithm; §6 cross-domain experiments and comparisons with regularization and model combination; §7 network-property effects and hyperparameter interactions; §8 B8T; §9 QX2 marginalization; Appendix A practical training guidance, including detailed hyperparameter-choice considerations.

## 2. Motivation

Sexual reproduction combines half of one parent’s genes with half of another’s and a very small random mutation; asexual reproduction supplies a slightly mutated copy of one parent’s genes. Direct transmission preserves a successful co-adapted gene set, supporting the plausibility of asexual superiority for individual-fitness optimization; sexual recombination breaks such sets, particularly large sets, with an intuitive fitness disadvantage for organisms already possessing complicated co-adaptations. Most advanced organisms nevertheless evolved through sexual reproduction.

The Livnat et al. (2010) account assigns possible long-run selection priority to gene mix-ability rather than individual fitness: a gene set functioning with another random gene set is more robust, unavailable guarantees of a large partner set favor useful individual action or collaboration with few genes, and reduction of complex co-adaptations assists useful new genes’ spread by reducing barriers to their improving individual fitness. QX2-trained hidden units likewise face randomly selected partners, favoring independently useful, robust features rather than reliance on other hidden units to repair mistakes; units within a layer still learn different functions. Replicating each hidden unit many times is a poor robustness solution for the same reason replica codes poorly address noisy channels.

Ten five-person conspiracies have a probable havoc-producing advantage over one conspiracy requiring fifty correct participants; unchanged conditions and ample rehearsal permit the larger conspiracy to work, while non-stationarity favors smaller conspiracies’ persistence. Complex co-adaptations can succeed on training data yet fail on novel test data more readily than multiple simpler co-adaptations achieving the same result.

## 3. Related Work

Noise in hidden-unit states gives a regularization interpretation of QX2, extending the input corruption and clean-input reconstruction used in autoencoders of the family (S2H: Denoising Autoencoders, DAEs) introduced by Vincent et al. (2008, 2010), with effective hidden-layer corruption, a model-averaging interpretation, supervised as well as unsupervised use, and applicability to other neuron-based architectures such as Boltzmann Machines. Test-time T6R permits much higher noise than usually works best for S2H.

| Setting | Noise level reported |
|---|---|
| S2H, typical best | 5% |
| QX2, often optimal input-unit removal | 20% |
| QX2, often optimal hidden-unit removal | 50% |

Stochastic minimization under a noise distribution minimizes an expected loss; the adversarial alternative of Globerson and Roweis (2006) and Dekel et al. (2010) lets an adversary select removals subject to a fixed maximum number rather than a probability distribution. Those models have no hidden units.

Analytically marginalizing QX2 gives deterministic regularizers in simple cases. Exponential-family input-noise regularizers, including QX2 under the name “blankout noise,” were investigated by van der Maaten et al. (2013), limited to models without hidden layers. Chen et al. (2012) examined marginalization for denoising autoencoders; Wang and Manning (2013) proposed marginalization to accelerate QX2.

## 4. Model Description

For every hidden unit, expected output under the training deletion distribution equals actual test-time output under the prescribed reduced weights. Retention choices are random and, in the simplest construction, mutually independent with a common fixed probability. Validation selection of that probability is possible; the usual default is close to optimal across many tasks and networks, while optimal input retention is generally nearer complete retention.

Let a network have \(L\) hidden layers, \(l\in\{1,\ldots,L\}\) index them, \(\mathbf z^{(l)}\) and \(\mathbf y^{(l)}\) denote layer inputs and outputs, \(\mathbf y^{(0)}=\mathbf x\), and \(\mathbf W^{(l)},\mathbf b^{(l)}\) denote layer weights and biases. For \(l\in\{0,\ldots,L-1\}\) and hidden unit \(i\), ordinary feed-forward computation is

\[
z_i^{(l+1)}
 =\mathbf w_i^{(l+1)}\mathbf y^l+b_i^{(l+1)},
\qquad
y_i^{(l+1)}=f\!\left(z_i^{(l+1)}\right).
\]

Any activation \(f\) is admissible, including

\[
f(x)=\frac{1}{1+\exp(-x)}.
\]

QX2 replaces this computation with

\[
r_j^{(l)}\sim\operatorname{Bernoulli}(p),
\qquad
\widetilde{\mathbf y}^{(l)}
 =\mathbf r^{(l)}*\mathbf y^{(l)},
\]

\[
z_i^{(l+1)}
 =\mathbf w_i^{(l+1)}\widetilde{\mathbf y}^{\,l}
   +b_i^{(l+1)},
\qquad
y_i^{(l+1)}=f\!\left(z_i^{(l+1)}\right).
\]

Here \(*\) is element-wise multiplication; each layer’s vector \(\mathbf r^{(l)}\) contains independent Bernoulli variables with probability \(p\) of equaling one. Its product with the layer outputs forms the thinned outputs passed to the next layer. Applying this operation at every layer samples a subnetwork; loss derivatives backpropagate through that subnetwork. Hidden and visible units can both be removed.

At inference, no units are removed and

\[
\mathbf W_{\mathrm{test}}^{(l)}=p\mathbf W^{(l)}.
\]

**Figure 2.** Training, left: a unit is present with probability \(p\), its connections to the next layer having weights \(w\). Testing, right: the unit is always present, its outgoing weights \(pw\).

**Figure 3.** Standard computation, left: \(y_1^{(l)},y_2^{(l)},y_3^{(l)}\), weights \(\mathbf w_i^{(l+1)}\), bias \(b_i^{(l+1)}\) from a \(+1\) unit, preactivation \(z_i^{(l+1)}\), activation \(f\), output \(y_i^{(l+1)}\). QX2 computation, right: corresponding gates \(r_1^{(l)},r_2^{(l)},r_3^{(l)}\) first produce \(\widetilde y_1^{(l)},\widetilde y_2^{(l)},\widetilde y_3^{(l)}\); the remaining operations retain their form.

## 5. Learning QX2 Nets

### 5.1 Backpropagation

QX2 alone gives significant gains; combining it with high momentum, large decaying learning rates, and the incoming-weight constraint (V9C: max-norm regularization) gives a further significant boost, while that constraint usually improves deep-net stochastic-gradient training even without QX2 and was previously used for collaborative filtering (Srebro and Shraibman, 2005). A proposed optimization account permits huge learning rates inside a fixed-radius weight ball without weight explosion, assigns QX2 noise exploration of otherwise difficult-to-reach weight-space regions, and assigns decaying rates progressively shorter, less exploratory steps ending at a minimum.

Stochastic gradient descent differs from standard-net training only in its per-case sampled thinned network: each mini-batch case receives forward and backward propagation through its own sampled network; parameter gradients are averaged across cases, and a case not using a parameter contributes zero to its gradient. Momentum, annealed learning rates and Y5F remain useful.

For an incoming hidden-unit weight vector \(\mathbf w\), optimization imposes

\[
\|\mathbf w\|_2\leq c.
\]

Whenever \(\mathbf w\) leaves the radius-\(c\) ball, projection returns it to the surface. The name V9C reflects the maximum allowed weight-vector norm \(c\), a validation-tuned hyperparameter.

### 5.2 Unsupervised Pretraining

Smaller finetuning learning rates preserve useful pretrained information and improve final generalization relative to finetuning without QX2; rates comparable to the best randomly initialized-net rates erased that information in the initial experiments. Pretraining itself is unchanged. Before QX2 finetuning, pretrained weights are multiplied by \(1/p\), matching each unit’s expected random-removal output to its pretraining output.

Unlabeled-data use through pretraining can produce significant improvements over random initialization in certain cases, with pretraining followed by backpropagation finetuning using stacked U2L (Hinton and Salakhutdinov, 2006), autoencoders (Vincent et al., 2010), or a further model family (E5Q: Deep Boltzmann Machines, DBM) described by Salakhutdinov and Hinton (2009), all admitting this finetuning procedure.

## 6. Experimental Results

Generalization improved over non-QX2 neural nets on every tested data set; object classification, digit recognition, speech, documents and computational-biology analysis cover diverse applications rather than one domain. The collections include standard handwritten-digit toy data (H4Z: MNIST), clean-speech benchmarking data (N8Q: TIMIT), and tiny natural-image collections (J3V: CIFAR-10) and (W6R: CIFAR-100), the latter pair described by Krizhevsky (2009), with Google Street View house-number images (A2F: Street View House Numbers data set, SVHN) supplied by Netzer et al. (2011). Natural-image scale is represented by (K9D: ImageNet), Reuters newswire documents by (M5X: Reuters-RCV1), and RNA-feature-based gene-splicing prediction by (Z7P: Alternative Splicing data set), from Xiong et al. (2011).

| Data set | Domain | Dimensionality | Training set | Test set |
|---|---|---|---|---|
| H4Z | Vision | 784 (28 × 28 grayscale) | 60K | 10K |
| A2F | Vision | 3072 (32 × 32 color) | 600K | 26K |
| J3V/W6R | Vision | 3072 (32 × 32 color) | 60K | 10K |
| K9D (ILSVRC-2012) | Vision | 65536 (256 × 256 color) | 1.2M | 150K |
| N8Q | Speech | 2520 (120-dim, 21 frames) | 1.1M frames | 58K frames |
| M5X | Text | 2000 | 200K | 200K |
| Z7P | Genetics | 1014 | 2932 | 733 |

*Table 1. Data-set overview.*

### 6.1 Image Data

Models attaining state of the art on all five vision collections use QX2; these collections vary in image type and training-set size.

#### 6.1.1 H4Z

Against the best permutation-invariant neural nets lacking QX2 and unsupervised pretraining, QX2 reduces error, substitution of an activation class (F3N: rectified linear units, ReLUs) for logistic units reduces it further (Jarrett et al., 2009), and V9C adds another reduction; greater network size improves results, while substitution of another activation class (L8J: maxout) improves them further (Goodfellow et al., 2013). Standard regularization and early stopping make the largest reported network difficult to generalize, yet QX2 prevents its overfitting without requiring early stopping.

Pretrained nets improve as well, including those initialized using stacked U2L, whose tabulated designation (G6W: DBN) distinguishes them from E5Q-initialized nets, with the latter QX2 result the best reported permutation-invariant performance. Better results remain possible using two-dimensional spatial information and distorted training-image augmentation, tested here on more interesting collections.

| Method | Unit type | Architecture | Error % |
|---|---|---|---|
| Standard Neural Net (Simard et al., 2003) | Logistic | 2 layers, 800 units | 1.60 |
| SVM Gaussian kernel | NA | NA | 1.40 |
| QX2 NN | Logistic | 3 layers, 1024 units | 1.35 |
| QX2 NN | F3N | 3 layers, 1024 units | 1.25 |
| QX2 NN + V9C | F3N | 3 layers, 1024 units | 1.06 |
| QX2 NN + V9C | F3N | 3 layers, 2048 units | 1.04 |
| QX2 NN + V9C | F3N | 2 layers, 4096 units | 1.01 |
| QX2 NN + V9C | F3N | 2 layers, 8192 units | 0.95 |
| QX2 NN + V9C (Goodfellow et al., 2013) | L8J | 2 layers, (5 × 240) units | 0.94 |
| G6W + finetuning (Hinton and Salakhutdinov, 2006) | Logistic | 500-500-2000 | 1.18 |
| E5Q + finetuning (Salakhutdinov and Hinton, 2009) | Logistic | 500-500-2000 | 0.96 |
| G6W + QX2 finetuning | Logistic | 500-500-2000 | 0.92 |
| E5Q + QX2 finetuning | Logistic | 500-500-2000 | 0.79 |

*Table 2. H4Z model comparison.*

Holding every hyperparameter, including retention probability, fixed across architectures yields two sharply separated test-error trajectory clusters for otherwise identical architectures trained with and without QX2. The improvement is large across architectures without architecture-specific hyperparameter tuning.

**Figure 4.** Test error during training for different architectures, with and without QX2; curves labeled “Without QX2” and “With QX2.”

| Axis | Label | Displayed ticks |
|---|---|---|
| Horizontal | Number of weight updates | 0, 200000, 400000, 600000, 800000, 1000000 |
| Vertical | Classification Error % | 1.0, 1.5, 2.0, 2.5 |

#### 6.1.2 A2F

Convolutional nets (LeCun et al., 1989) outperform other methods on this task; adding QX2 to their fully connected layers improves performance, adding it to convolutional layers improves it further, and L8J permits additional gains. Few convolutional parameters might suggest little overfitting and little benefit from QX2 there, yet lower-layer corruption supplies noisy inputs that prevent the higher fully connected layers from overfitting.

| Method | Error % |
|---|---|
| Binary Features (WDCH) (Netzer et al., 2011) | 36.7 |
| HOG (Netzer et al., 2011) | 15.0 |
| Stacked Sparse Autoencoders (Netzer et al., 2011) | 10.3 |
| KMeans (Netzer et al., 2011) | 9.4 |
| Multi-stage Conv Net with average pooling (Sermanet et al., 2012) | 9.06 |
| Multi-stage Conv Net + L2 pooling (Sermanet et al., 2012) | 5.36 |
| Multi-stage Conv Net + L4 pooling + padding (Sermanet et al., 2012) | 4.90 |
| Conv Net + max-pooling | 3.95 |
| Conv Net + max pooling + QX2 in fully connected layers | 3.02 |
| Conv Net + stochastic pooling (Zeiler and Fergus, 2013) | 2.80 |
| Conv Net + max pooling + QX2 in all layers | 2.55 |
| Conv Net + L8J (Goodfellow et al., 2013) | 2.47 |
| Human Performance | 2.0 |

*Table 3. A2F results.*

The used portion contains color images approximately centered on one digit of a house number; identifying that digit is the task.

#### 6.1.3 J3V and W6R

Without data augmentation, Bayesian hyperparameter optimization gives the Snoek et al. (2012) J3V result; QX2 in fully connected layers improves it, extension to every layer improves it further, and L8J substitution further reduces error (Goodfellow et al., 2013). W6R shows a large QX2 improvement. Neither collection uses augmentation apart from input QX2.

| Method | J3V | W6R |
|---|---|---|
| Conv Net + max pooling (hand tuned) | 15.60 | 43.48 |
| Conv Net + stochastic pooling (Zeiler and Fergus, 2013) | 15.13 | 42.51 |
| Conv Net + max pooling (Snoek et al., 2012) | 14.98 | - |
| Conv Net + max pooling + QX2 fully connected layers | 14.32 | 41.26 |
| Conv Net + max pooling + QX2 in all layers | 12.61 | 37.20 |
| Conv Net + L8J (Goodfellow et al., 2013) | 11.68 | 38.57 |

*Table 4. J3V and W6R error rates.*

**Figure 5.** Image samples: (a) A2F, (b) J3V; each row represents a different category.

#### 6.1.4 K9D

The convolutional-net/QX2 model won ILSVRC-2012; its advantage over standard vision-feature methods is staggering, and its predictions remain very reasonable when its best guess is wrong. On ILSVRC-2010, convolutional nets with QX2 outperform the alternatives by a large margin. Architecture and implementation are detailed by Krizhevsky et al. (2012).

| Model | Top-1 | Top-5 |
|---|---|---|
| Sparse Coding (Lin et al., 2010) | 47.1 | 28.2 |
| SIFT + Fisher Vectors (Sanchez and Perronnin, 2011) | 45.7 | 25.7 |
| Conv Net + QX2 (Krizhevsky et al., 2012) | 37.5 | 17.0 |

*Table 5. ILSVRC-2010 test results.*

| Model | Top-1 (val) | Top-5 (val) | Top-5 (test) |
|---|---|---|---|
| SVM on Fisher Vectors of Dense SIFT and Color Statistics | - | - | 27.3 |
| Avg of classifiers over FVs of SIFT, LBP, GIST and CSIFT | - | - | 26.2 |
| Conv Net + QX2 (Krizhevsky et al., 2012) | 40.7 | 18.2 | - |
| Avg of 5 Conv Nets + QX2 (Krizhevsky et al., 2012) | 38.1 | 16.4 | 16.4 |

*Table 6. ILSVRC-2012 validation/test results.*

| Broad competition comparison | Top-5 error |
|---|---|
| Best standard vision-feature methods | about 26% |
| Convolutional nets with QX2 | about 16% |

K9D contains over 15 million labeled high-resolution images in roughly 22,000 categories. The annual ILSVRC competition began in 2010 as part of the Pascal Visual Object Challenge, using a subset with roughly 1000 images in each of 1000 categories. Conventional reporting includes top-1 and top-5 errors; top-5 counts test images whose correct label is absent from the five model-assigned most probable labels. ILSVRC-2010 alone has available test labels and supplies most experiments. Unavailable ILSVRC-2012 test labels restrict reported test results to the final submission, with model variations evaluated on validation data.

**Figure 6.** K9D test cases with the four most probable predicted labels; horizontal bar length is proportional to assigned probability, pink marks ground truth.

### 6.2 N8Q

QX2 improves the unpretrained network and both pretrained comparisons; the two QX2-pretrained depths attain the same phone error. The reported prose comparison for the deeper pretrained net uses a pre-QX2 baseline distinct from the corresponding tabulated baseline.

| Method | Phone Error Rate % |
|---|---|
| NN (6 layers) (Mohamed et al., 2010) | 23.4 |
| QX2 NN (6 layers) | 21.8 |
| G6W-pretrained NN (4 layers) | 22.7 |
| G6W-pretrained NN (6 layers) (Mohamed et al., 2010) | 22.4 |
| G6W-pretrained NN (8 layers) (Mohamed et al., 2010) | 20.7 |
| mcRBM-G6W-pretrained NN (5 layers) (Dahl et al., 2010) | 20.5 |
| G6W-pretrained NN (4 layers) + QX2 | 19.7 |
| G6W-pretrained NN (8 layers) + QX2 | 19.7 |

*Table 7. N8Q core-test-set phone errors.*

| Prose comparison | Pre-QX2 error |
|---|---|
| 8-layer net | 20.5% |

### 6.3 Text Data

Bag-of-words document classification on a subset of M5X improves with QX2, but less than vision and speech classification. Improvement occurred for every tried architecture; the collection’s size and consequently less serious overfitting provide a possible account of the smaller gain.

| Best neural-net document classifier | Error rate |
|---|---|
| Without QX2 | 31.05% |
| With QX2 | 29.62% |

### 6.4 Bayesian Neural Networks

Bayesian neural nets outperform QX2 on Z7P, while QX2 substantially improves standard neural nets and exceeds every other method. Input reduction by PCA followed by SVMs or logistic regression offers one defense against small-training-set overfitting; QX2 prevents it without dimensionality reduction, using vastly more hidden units than the Bayesian network and exhibiting strong regularization.

| Method | C4Y (bits) |
|---|---|
| Neural Network (early stopping) (Xiong et al., 2011) | 440 |
| Regression, PCA (Xiong et al., 2011) | 463 |
| SVM, PCA (Xiong et al., 2011) | 487 |
| Neural Network with QX2 | 567 |
| Bayesian Neural Network (Xiong et al., 2011) | 623 |

*Table 8. Z7P results.*

Posterior weighting in Bayesian neural networks (Neal, 1996) is the proper averaging over network structures and parameters, incorporating both prior and data fit instead of QX2’s equal weights on exponentially many shared-weight models. Bayesian nets are extremely useful with scarce data in medical diagnosis, genetics, drug discovery and other computational-biology applications, but train slowly, scale poorly to very large networks and require expensive predictions from many large nets; QX2 nets train and infer much faster.

Z7P is a small genetics collection on which Bayesian nets are known to perform well and achieve state of the art (Xiong et al., 2011). Alternative splicing significantly contributes to mammalian-tissue cellular diversity; predicting its tissue- and condition-dependent occurrence matters for understanding many human diseases. RNA features predict probabilities of three biologically important splicing-related events, scored by a distributional measure (C4Y: Code Quality) representing negative KL divergence between target and predicted distributions, with higher values preferred.

### 6.5 Standard Regularizers

QX2 combined with V9C gives the lowest generalization error in the H4Z regularizer comparison. Candidates include Y5F, more generally Tikhonov regularization (Tikhonov, 1943), lasso (Tibshirani, 1996), KL-sparsity and V9C. Validation selected decay constants, target sparsity, QX2 rate and the V9C bound for the shared stochastic-gradient-trained F3N architecture.

| Method | Test Classification error % |
|---|---|
| Y5F | 1.62 |
| Y5F + O3U applied towards the end of training | 1.60 |
| Y5F + KL-sparsity | 1.55 |
| V9C | 1.35 |
| QX2 + Y5F | 1.25 |
| QX2 + V9C | 1.05 |

*Table 9. H4Z regularization comparison.*

## 7. Salient Features

### 7.1 Features

First-layer visual features differ markedly even when test reconstruction errors are similar: without QX2, individual hidden units do not appear to detect meaningful features, whereas QX2 units detect edges, strokes and spots in different image regions. Breaking co-adaptations is proposed as the principal source of lower generalization error.

A parameter’s ordinary derivative prescribes loss reduction conditional on every other unit’s behavior, allowing units to repair one another’s mistakes and build complex, non-generalizing co-adaptations. The QX2 hypothesis makes particular partners unreliable, denies dependence on specific corrective units, and requires success across the many contexts supplied by the remaining units.

**Figure 7.** H4Z features from identical single-hidden-layer autoencoders with 256 F3N units: (a) no QX2, (b) hidden-layer QX2 with \(p=0.5\). The former features are co-adapted for reconstruction; the latter have the localized structures described above.

### 7.2 Sparsity

QX2 produces sparse hidden activations without sparsity-inducing regularizers. The autoencoder comparison uses a random test-set mini-batch; good sparsity requires few strongly active units per case and low average activity of each unit across cases. For each model, the left histogram distributes hidden-unit mean activations across that mini-batch, while the right distributes the activations themselves.

| Observation | No QX2 | QX2 |
|---|---|---|
| Most units’ mean activation; overall hidden-unit mean | about / close to 2.0 | about / around 0.7 |
| Activation histogram | Huge mode away from zero; substantial off-zero mass | Sharp zero peak |
| Highly active units | Large fraction | Very few |

**Figure 8.** Sparsity histograms for the F3N autoencoders: (a) without QX2, (b) QX2 with \(p=0.5\).

### 7.3 Rate

Holding architecture fixed gives underfitting at low retention: few active units and high training error accompany the high test error. Error falls as retention rises, flattens over an intermediate interval, then rises near complete retention. Holding expected retained width fixed substantially reduces the low-retention errors; corresponding inference networks have different sizes, with lower-retention networks larger.

| Comparison quantity | Value |
|---|---|
| Flat-error retention interval, fixed architecture | \(0.4\leq p\leq0.8\) |
| Retention used for low-retention comparison | \(p=0.1\) |
| Error, fixed architecture | 2.7% |
| Error, fixed expected retained width | 1.7% |
| Apparently best retention for the chosen expected widths | close to 0.6 |
| Usual default, close to optimal in that comparison | 0.5 |

**Figure 9.** H4Z training and test errors versus retention probability: (a) fixed \(n\), (b) fixed \(pn\).

| Panel | Horizontal label and ticks | Vertical label and ticks | Curves |
|---|---|---|---|
| (a) | Probability of retaining a unit \((p)\): 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 | Classification Error %: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5 | Test Error; Training Error |
| (b) | Probability of retaining a unit \((p)\): 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 | Classification Error %: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 | Test Error; Training Error |

### 7.4 Data-Set Size

The two smallest tested training sets show no QX2 improvement: the network can memorize them despite the removal noise. Increasing data first increases the gain and then decreases it. For fixed architecture and QX2 rate, the proposed “sweet spot” contains enough data to resist noisy memorization but not enough to make overfitting unimportant; standard training of huge nets on small collections massively overfits, and small-data generalization from many parameters is a test of a regularizer.

**Figure 10.** H4Z error versus randomly selected training-set size, curves “With QX2” and “Without QX2.”

| Axis | Label | Displayed ticks |
|---|---|---|
| Horizontal | Dataset size | \(10^2,10^3,10^4,10^5\) |
| Vertical | Classification Error % | 0, 5, 10, 15, 20, 25, 30 |

### 7.5 X9M vs. T6R

X9M matches T6R after sufficiently many samples and then performs slightly better, within one standard deviation; T6R is a fairly good approximation to the true model average. The H4Z comparison evaluates error for varying sample counts and a horizontal T6R reference, measuring the finite-sample average’s approach to true-average error.

| Quantity | Value |
|---|---|
| Sample count at which X9M becomes as good as T6R | around \(k=50\) |

**Figure 11.** X9M versus T6R.

| Axis | Label | Displayed ticks |
|---|---|---|
| Horizontal | Number of samples used for X9M \((k)\) | 0, 20, 40, 60, 80, 100, 120 |
| Vertical | Test Classification error % | 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35 |

Curves: “X9M” and “Approximate averaging by T6R.”

## 8. B8T

### 8.1 Model Description

B8T is a mixture of exponentially many shared-weight U2L, each using a different hidden-unit subset. Conditional on a retention vector, its visible–hidden distribution is an ordinary U2L distribution with the designated hidden units removed.

For visible units \(\mathbf v\in\{0,1\}^D\), hidden units \(\mathbf h\in\{0,1\}^F\), parameters \(\theta=\{\mathbf W,\mathbf a,\mathbf b\}\), and partition function \(Z\), an ordinary U2L defines

\[
P(\mathbf h,\mathbf v;\theta)
=\frac{1}{Z(\theta)}
 \exp\!\left(
 \mathbf v^\top\mathbf W\mathbf h+
 \mathbf a^\top\mathbf h+
 \mathbf b^\top\mathbf v
 \right).
\]

B8T augments this with \(\mathbf r\in\{0,1\}^F\), independently retaining \(h_j\) when \(r_j=1\), with probability \(p\), and dropping it otherwise:

\[
P(\mathbf r,\mathbf h,\mathbf v;p,\theta)
=P(\mathbf r;p)P(\mathbf h,\mathbf v\mid\mathbf r;\theta),
\]

\[
P(\mathbf r;p)=\prod_{j=1}^{F}p^{r_j}(1-p)^{1-r_j},
\]

\[
P(\mathbf h,\mathbf v\mid\mathbf r;\theta)
=\frac{
 \exp\!\left(
 \mathbf v^\top\mathbf W\mathbf h+
 \mathbf a^\top\mathbf h+
 \mathbf b^\top\mathbf v
 \right)
}{Z'(\theta,\mathbf r)}
\prod_{j=1}^{F}g(h_j,r_j),
\]

\[
g(h_j,r_j)
=\mathbf1(r_j=1)+\mathbf1(r_j=0)\mathbf1(h_j=0).
\]

The normalizer is \(Z'(\theta,\mathbf r)\); \(g\) enforces \(h_j=0\) when \(r_j=0\). The conditional hidden distribution is factorial:

\[
P(\mathbf h\mid\mathbf r,\mathbf v)
=\prod_{j=1}^{F}P(h_j\mid r_j,\mathbf v),
\]

\[
P(h_j=1\mid r_j,\mathbf v)
=\mathbf1(r_j=1)\,
 \sigma\!\left(b_j+\sum_iW_{ij}v_i\right).
\]

The visible conditional remains that of U2L:

\[
P(\mathbf v\mid\mathbf h)
=\prod_{i=1}^{D}P(v_i\mid\mathbf h),
\]

\[
P(v_i=1\mid\mathbf h)
=\sigma\!\left(a_i+\sum_jW_{ij}h_j\right).
\]

### 8.2 Learning B8T

Contrastive Divergence and other U2L learning algorithms apply directly (Hinton et al., 2006): first sample \(\mathbf r\), then train only retained hidden units. Each training case in every mini-batch receives a different \(\mathbf r\). Experiments use CD-1.

### 8.3 Features

B8T features appear coarser than the sharply defined stroke-like U2L features, with very few dead units relative to U2L; this tests transfer of feed-forward QX2’s feature-quality effect.

**Figure 12.** H4Z features from 256-hidden-unit models, ordered by \(L_2\) norm: (a) ordinary U2L, (b) B8T with \(p=0.5\).

### 8.4 Sparsity

Without an additional sparsity-inducing regularizer, B8T learns much sparser representations than U2L. Hidden-activation and mean-activation histograms use a test mini-batch.

**Figure 13.** (a) Without QX2: many activations away from zero. (b) QX2 with \(p=0.5\): many activations near zero and very few high activations.

## 9. Marginalizing QX2

Deterministic noise-marginalized models require no random bits and admit gradients of marginalized losses, unlike standard “Monte-Carlo” QX2. Deterministic robustness to test-time feature deletion was investigated by Globerson and Roweis (2006); Srivastava (2013) discussed linear-regression QX2 marginalization, and Wager et al. (2013) described QX2 as an adaptive regularizer. The denoising-autoencoder, acceleration and input-noise regularizer investigations retain the scopes stated earlier.

### 9.1 Linear Regression

QX2 linear regression is, in expectation, ridge regression with a particular penalty matrix: a dimension’s standard deviation scales its weight cost, squeezing more strongly the weights of highly variable dimensions. Absorbing retention into the weights exposes a regularization constant small near complete retention and larger under greater removal.

Let \(\mathbf X\in\mathbb R^{N\times D}\) contain \(N\) data points, \(\mathbf y\in\mathbb R^N\) contain targets, and \(\mathbf w\in\mathbb R^D\). Ordinary linear regression minimizes

\[
\|\mathbf y-\mathbf X\mathbf w\|^2.
\]

Retaining each input dimension with probability \(p\) yields \(\mathbf R*\mathbf X\), where \(\mathbf R\in\{0,1\}^{N\times D}\), \(R_{ij}\sim\operatorname{Bernoulli}(p)\), and \(*\) is element-wise multiplication. The marginalized objective is

\[
\underset{\mathbf w}{\operatorname{minimize}}\quad
\mathbb E_{\mathbf R\sim\operatorname{Bernoulli}(p)}
\left[\|\mathbf y-(\mathbf R*\mathbf X)\mathbf w\|^2\right],
\]

reducing to

\[
\underset{\mathbf w}{\operatorname{minimize}}\quad
\|\mathbf y-p\mathbf X\mathbf w\|^2+
p(1-p)\|\mathbf\Gamma\mathbf w\|^2,
\]

\[
\mathbf\Gamma
=\left(\operatorname{diag}(\mathbf X^\top\mathbf X)\right)^{1/2}.
\]

With \(\widetilde{\mathbf w}=p\mathbf w\), the alternative form is

\[
\underset{\mathbf w}{\operatorname{minimize}}\quad
\|\mathbf y-\mathbf X\widetilde{\mathbf w}\|^2+
\frac{1-p}{p}\|\mathbf\Gamma\widetilde{\mathbf w}\|^2.
\]

### 9.2 Logistic Regression and Deep Networks

Approximate marginalization outperforms Monte-Carlo QX2 in training time and generalization for logistic regression (Wang and Manning, 2013). Closed-form marginalized models are difficult for logistic regression and deep nets; under reasonable assumptions, distributions of logistic-unit inputs and marginalized-model gradients are Gaussian with efficiently computable means and variances. The assumptions become successively weaker with added layers, leaving the results not directly applicable to deep networks.

## 10. Multiplicative Gaussian Noise

Multiplicative Gaussian corruption works as well as, or perhaps better than, binary corruption in preliminary experiments, the latter variant (P7V: Bernoulli dropout) and the former variant (D4K: Gaussian dropout) permitting matched first and second moments despite opposite entropy extremes. Given those moments, the Gaussian multiplier has the highest entropy and the Bernoulli multiplier the lowest; both work well, with preliminary evidence favoring the high-entropy case slightly.

| Data set | Architecture | P7V | D4K |
|---|---|---|---|
| H4Z | 2 layers, 1024 units each | \(1.08\pm0.04\) | \(0.95\pm0.04\) |
| J3V | 3 conv + 2 fully connected layers | \(12.6\pm0.1\) | \(12.5\pm0.1\) |

*Table 10. Classification error %.*

| Comparison setting | Value |
|---|---|
| H4Z P7V retention, hidden units | \(p=0.5\) |
| H4Z P7V retention, input units | \(p=0.8\) |
| J3V P7V retention, input to top | \(p=(0.9,0.75,0.75,0.5,0.5,0.5)\) |
| D4K standard deviation, each corresponding layer | \(\sigma=\sqrt{(1-p)/p}\) |
| Averaging | 10 different random seeds |

Multiplication of hidden activations by Bernoulli variables retaining value one with probability \(p\), zero otherwise, generalizes to other multiplier distributions. Drawing a multiplier from \(\mathcal N(1,1)\) is equivalent to adding zero-mean Gaussian noise whose standard deviation equals the unit activation:

\[
h_i\longmapsto h_i+h_ir,\qquad r\sim\mathcal N(0,1),
\]

or

\[
h_i\longmapsto h_ir',\qquad r'\sim\mathcal N(1,1).
\]

The extension \(r'\sim\mathcal N(1,\sigma^2)\) makes \(\sigma\) a tunable hyperparameter analogous to \(p\). Unchanged expected activations eliminate test-time T6R.

Retaining activations at training and multiplying outgoing weights by \(p\) at testing is equivalent, with appropriately rescaled layerwise learning rates and weight initializations, to multiplying retained training activations by \(1/p\) and leaving test weights unchanged. In the latter representation, the Bernoulli multiplier \(r_b\) equals \(1/p\) with probability \(p\), zero otherwise:

\[
\mathbb E[r_b]=1,\qquad
\operatorname{Var}[r_b]=\frac{1-p}{p}.
\]

Choosing \(\sigma^2=(1-p)/p\) for Gaussian multiplier \(r_g\) gives

\[
\mathbb E[r_g]=1,\qquad
\operatorname{Var}[r_g]=\frac{1-p}{p}.
\]

## 11. Conclusion

The large-model construction repeatedly samples and trains smaller submodels from an easily overfitted model; U2L fits this construction, and B8T exhibits desirable empirical properties. QX2 methods achieve state of the art on A2F, K9D, W6R and H4Z, with considerable improvement to standard neural nets on other collections.

Training takes typically 2–3 times as long as training a standard network of the same architecture. Noisy parameter updates are a major cause: each case trains a different random architecture, and computed gradients are not gradients of the final inference architecture. The stochasticity likely prevents overfitting, leaving an overfitting–training-time trade-off in which more training time permits stronger QX2 and less overfitting.

Noise marginalization obtains some benefits without stochasticity through an expectation-equivalent regularizer. Linear regression gives modified Y5F; equivalent regularizers for more complicated models are not obvious. Accelerating QX2 remains a future-work direction.

## Acknowledgments

This research was supported by OGS, NSERC and an Early Researcher Award.

## Appendix A. A Practical Guide for Training QX2 Networks

QX2 nets retain neural networks’ requirement for extensive hyperparameter tuning; the following are practical heuristics.

### A.1 Network Size

For a task whose optimal standard-net layer has \(n\) hidden units, a good QX2 layer should contain at least \(n/p\) units. Removing units reduces capacity: only \(pn\) units remain in expectation, their identities change at every sampling, and unrestricted co-adaptation is unavailable. This heuristic was useful for both convolutional and fully connected layer widths.

### A.2 Learning Rate and Momentum

High learning rate and/or momentum significantly accelerate learning. QX2 introduces much more gradient noise than ordinary stochastic gradient descent, with many gradients canceling one another; larger learning rates compensate, while higher momentum reduces the noise’s effect.

| Quantity | Practical values |
|---|---|
| QX2 learning rate relative to optimal standard-net rate | typically 10–100 times |
| Common standard-net momentum | 0.9 |
| QX2 momentum working substantially better | around 0.95 to 0.99 |

### A.3 V9C

Large momentum and learning rates sometimes produce very large weights despite accelerating learning. V9C bounds each hidden unit’s incoming-weight-vector norm.

| Hyperparameter | Typical range |
|---|---|
| \(c\) | 3 to 4 |

### A.4 Rate

Retention probability controls QX2 intensity; its choice for hidden layers is coupled to layer width. Smaller retention requires larger width, slows training and leads to underfitting; large retention can leave insufficient removal to prevent overfitting. Input choice depends on input type.

| Condition | Retention |
|---|---|
| No QX2 | \(p=1\) |
| Stronger QX2 | lower \(p\) |
| Hidden-unit typical range | 0.5 to 0.8 |
| Real-valued inputs, including image patches and speech frames | typically 0.8 |
| Broadly near-optimal simple default | 0.5 |
| Input-unit optimum in general | closer to 1 than to 0.5 |

## Appendix B. Detailed Description of Experiments and Data Sets

Reproduction code is available at <http://www.cs.toronto.edu/~nitish/dropout>. GPU implementations use the excellent CUDA libraries cudamat (Mnih, 2009) and cuda-convnet (Krizhevsky et al., 2012).

### B.1 H4Z

Classification assigns each digit image to one of ten digit classes. Validation images were randomly held out; hyperparameters were selected to give their best validation error at the chosen training endpoint, after which validation and training data were combined and a new run reached that same endpoint before test evaluation. Easy selection of hyperparameters requiring no early stopping justified the long combined-data run.

| Setting | Value |
|---|---|
| Held-out training images for validation | 10,000 |
| Weight-update endpoint, both validation tuning and final combined-data training | 1 million |
| Architecture-sweep depths | 2, 3, 4 hidden layers |
| Architecture-sweep uniform layer widths | 1024, 2048 |
| Full combinations in sweep | six architectures |
| Hidden-layer retention, sweep and reported QX2 models | \(p=0.5\) |
| Input retention, sweep and reported QX2 models | \(p=0.8\) |
| Final momentum, all layers | 0.95 |
| V9C bound, all layers | \(c=2\) |
| Additional regularization-limit experiments | 2 and 3 layers; 4096 and 8192 units |
| Largest reported two-layer model | more than 65 million parameters |
| Regularizer-comparison architecture | 784-1024-1024-2048-10 |
| Regularizer-comparison units and optimizer | F3N; stochastic gradient descent |
| Data-size-comparison architecture | 784-1024-1024-2048-10 |
| Randomly selected data-size-comparison training sets | 100, 500, 1K, 5K, 10K, 50K |
| Data-size-comparison hidden/input retention | \(p=0.5\); \(p=0.8\) |
| Fixed-width rate-comparison architecture | 784-2048-2048-2048-10 |
| Fixed-width rate-comparison input corruption | none |
| Fixed-expected-width comparison, first two hidden layers | \(pn=256\) |
| Fixed-expected-width comparison, last hidden layer | \(pn=512\) |

The additional two-layer networks improve performance; three-layer counterparts are slightly worse at the same QX2 level. Increasing QX2 improves those three-layer nets but does not make them outperform the two-layer nets.

In the fixed-expected-width comparison, \(n\) is a layer’s hidden-unit count and \(pn\) its expected surviving count; small \(p\) requires large \(n\), matching expected training-network widths while leaving inference widths different. X9M, the expensive but more correct averaging procedure, independently samples \(k\) thinned networks for each test case and averages predictions; as \(k\to\infty\), the result approaches the true model average.

### B.2 A2F

The training-image count is approximate. Training data comprise a standard labeled portion and an additional labeled portion of easy examples; validation draws from both by the Sermanet et al. (2012) procedure. RGB pixels were normalized to zero mean and unit variance; global or local contrast normalization and ZCA whitening gave no noticeable improvement.

| Component | Specification |
|---|---|
| Validation from standard portion | two-thirds; 400 per class |
| Validation from additional portion | one-third; 200 per class |
| Total validation samples | 6000 |
| Convolutional layers | three |
| Filters, consecutive convolutional layers | 96, 128, 256 |
| Receptive field, each convolutional layer | \(5\times5\) |
| Convolutional stride | 1 pixel |
| Following each convolutional layer | max-pooling |
| Pooling region | \(3\times3\) |
| Pooling stride | 2 pixels |
| Following convolutional layers | two fully connected hidden layers |
| Units per fully connected hidden layer | 2048 |
| All hidden activations | F3N |
| Retention, input through convolutional and fully connected layers | \(p=(0.9,0.75,0.75,0.5,0.5,0.5)\) |
| V9C, convolutional and fully connected weights | \(c=4\), all weights |
| Momentum, all layers | 0.95 |

Validation tuned these hyperparameters. The large training set remained separate from validation during final training; the model having smallest validation error supplied the reported test error.

### B.3 J3V and W6R

Each collection contains 50,000 training and 10,000 test images, with 10 and 100 categories respectively. Validation used 5,000 training images; its selected hyperparameters were followed by combined validation/training optimization, as in the H4Z procedure.

Preprocessing applies global contrast normalization within each image/color channel, then ZCA whitening. The former computes and subtracts that channel’s mean pixel intensity. The latter mean-centers data, rotates onto its principal components, normalizes each component and rotates back. Architecture and retention rates match the A2F configuration; input-layer learning rates require smaller values.

### B.4 N8Q

Recordings contain 680 speakers covering eight major American-English dialects, each reading ten phonetically rich sentences in a controlled, noise-free environment. The open-source Kaldi toolkit (Povey et al., 2011) generated log-filter banks; a monophone system supplied forced alignment and speech-frame labels. Consecutive-frame windows predict the central-frame label, with no speaker-dependent operations and mean-centered, unit-variance inputs.

| Setting | Specification |
|---|---|
| Consecutive frames per window | 21 |
| Input retention | \(p=0.8\) |
| Hidden retention | \(p=0.5\) |
| V9C, all layers | \(c=4\) |
| Momentum | 0.95 |
| High learning rate | 0.1 |
| Learning-rate decay | \(\epsilon_0(1+t/T)^{-1}\) |
| G6W-pretraining algorithm | U2L trained with CD-1 |
| Gaussian-U2L input-unit variance | fixed to 1 |
| Best QX2 G6W-finetuning learning rate | about 0.01 |

Adding V9C during G6W finetuning gave no improvement; the smaller learning rate was important for best results.

### B.5 M5X

The full corpus contains more than 800,000 documents across 103 classes arranged in a tree. The used subset contains 402,738 articles, a 2000-word vocabulary and 50 disjoint categories, each document belonging to exactly one; equal-sized training and test splits result in more than 200,000 training examples. Articles span varied topics.

### B.6 Z7P

The data comprise 3665 cassette exons, 1014 RNA features and four tissue types derived from 27 mouse tissues. Each input has four target softmax units, one per tissue type; each softmax has biologically important states \(\{\mathrm{inc},\mathrm{exc},\mathrm{nc}\}\). Predicted state distributions should match wet-lab-observed distributions as closely as possible.

\[
\mathrm{C4Y}
=
\sum_{i=1}^{|\text{data points}|}
\sum_{t\in\text{tissue types}}
\sum_{s\in\{\mathrm{inc},\mathrm{exc},\mathrm{nc}\}}
p_{i,t}^{s}
\log\!\left(\frac{q_t^{s}(r_i)}{\bar p^{s}}\right).
\]

Here \(p_{i,t}^{s}\) is the target probability of state \(s\), tissue type \(t\), input \(i\); \(q_t^{s}(r_i)\) is its predicted probability for input \(r_i\); and \(\bar p^{s}\) averages \(p_{i,t}^{s}\) over \(i,t\).

| Setting | Specification |
|---|---|
| QX2 hidden-unit scale | 1000s |
| Bayesian-network hidden-unit scale | a few tens |
| Trained QX2 network | two layers, 1024 units each |
| Hidden retention | \(p=0.5\) |
| Input retention | \(p=0.7\) |
| Regularization and rate schedule | V9C; high decaying learning rates |
| Result averaging | same 5 folds as Xiong et al. (2011) |
