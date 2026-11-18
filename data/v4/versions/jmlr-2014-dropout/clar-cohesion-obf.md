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

Large, parameter-rich deep neural networks are powerful machine-learning systems. However, these networks overfit, and their slow use makes test-time combination of independently trained large models difficult. **Dropout**, designated **QX2** in this paper, addresses these problems by temporarily removing randomly selected units and their connections during training. Random removal inhibits excessive co-adaptation and samples exponentially many thinned networks that share parameters.

At test time, a single unthinned network with reduced weights approximates the thinned networks’ averaged predictions. Relative to other regularizers, dropout substantially reduces overfitting and produces major improvements. Experiments in supervised vision, speech recognition, document classification and computational biology attain state-of-the-art performance on many benchmarks.

**Keywords:** neural networks, regularization, model combination, deep learning

## 1. Introduction

With unlimited computation, the best regularization of a fixed-sized model is to average over every parameter setting, weighting each setting by its posterior probability. Simple or small models sometimes admit good approximations to this averaging procedure (Xiong et al., 2011; Salakhutdinov and Mnih, 2008). In contrast, dropout provides a less expensive construction: dropout approximates an equally weighted geometric mean of the predictions of exponentially many learned models that share parameters.

For a network with \(n\) units, where \(n\) denotes the unit count, there are \(2^n\) possible thinned networks. Weight sharing allows these networks to retain only \(O(n^2)\), or fewer, parameters; \(O(n^2)\) denotes an order-of-growth bound quadratic in the unit count. A newly sampled thinned network is trained for every presentation of every training case. Consequently, each particular thinned network is trained rarely, if ever.

Multilayer non-linear hidden representations can express complicated input–output relationships. However, when data are limited, sampling noise supplies relationships that are present in the training data but absent from genuine test data, including test data drawn from the same distribution. Established responses to this overfitting include stopping training when validation performance deteriorates, soft weight sharing (Nowlan and Hinton, 1992), and weight penalties. Familiar weight penalties include **L1 regularization**, designated **O3U**, and **L2 regularization**, designated **Y5F**. These penalties constrain learned parameters rather than averaging over every possible parameter setting.

Model combination nearly always improves machine-learning performance. Nevertheless, separately training many large networks makes output averaging prohibitively expensive for several reasons. Useful diversity requires different architectures or different training data; architecture-specific optimal hyperparameters are difficult to find; and each large-network training run consumes substantial computation. Moreover, large data requirements can preclude training different networks on different subsets. Even when many such models can be trained successfully, their joint inference remains infeasible for rapid-response applications.

Dropout therefore uses a practical inference rule called **weight scaling**, designated **T6R**: a single unthinned network replaces explicit evaluation of exponentially many thinned networks. A more expensive alternative, **Monte-Carlo model averaging**, designated **X9M**, averages predictions from sampled thinned networks. Across varied classification problems, dropout with weight scaling gives significantly lower generalization error than other regularization methods.

The same unit-removal construction also extends beyond feed-forward networks to graphical models, including Boltzmann Machines. In particular, **Restricted Boltzmann Machines**, abbreviated **RBM** and designated **U2L**, admit an augmented form called a **Dropout RBM**, designated **B8T**, without abandoning shared parameters.

**Figure 1.** Standard neural net, left: two hidden layers. Dropout-produced thinned net, right: crossed units are removed, including their incoming and outgoing connections.

Paper map: §2 gives the motivation; §3 discusses previous work; §4 defines the formal dropout model; §5 describes the training algorithm; §6 presents cross-domain experiments and comparisons with regularization and model combination; §7 examines effects on network properties and interactions with hyperparameters; §8 describes Dropout RBMs; §9 discusses dropout marginalization; and Appendix A gives practical training guidance, including detailed considerations for choosing hyperparameters.

## 2. Motivation

Sexual reproduction combines half of one parent’s genes with half of another’s, together with a very small random mutation. In contrast, asexual reproduction supplies a slightly mutated copy of one parent’s genes. Because direct transmission preserves a successful co-adapted gene set, asexual reproduction might plausibly seem superior for optimizing individual fitness. Sexual recombination instead breaks co-adapted sets, particularly large sets, and therefore appears intuitively disadvantageous for organisms that already possess complicated co-adaptations. Nevertheless, most advanced organisms evolved through sexual reproduction.

The account of Livnat et al. (2010) suggests that long-run selection may prioritize **gene mix-ability** rather than individual fitness. A gene set that functions with another randomly selected gene set is more robust. Because a gene cannot be guaranteed a large set of particular partners, useful individual action or collaboration with only a few genes is favored. Reducing complex co-adaptations also helps useful new genes spread, because fewer co-adaptations create fewer barriers to those genes improving individual fitness.

Dropout-trained hidden units face an analogous situation: each hidden unit must work with randomly selected partners. Consequently, dropout favors independently useful, robust features rather than reliance on other hidden units to repair mistakes. Units within a layer nevertheless learn different functions. Simply replicating each hidden unit many times is a poor solution to robustness, for the same reason that replica codes poorly address noisy channels.

A second analogy compares conspiracies. Ten five-person conspiracies have a probable havoc-producing advantage over one conspiracy requiring fifty correct participants. Under unchanged conditions and with ample rehearsal, the larger conspiracy can work. Under non-stationarity, however, smaller conspiracies are more likely to persist. Similarly, complex co-adaptations can succeed on training data yet fail on novel test data more readily than multiple simpler co-adaptations that achieve the same result.

## 3. Related Work

Noise in hidden-unit states provides a regularization interpretation of dropout. In this interpretation, dropout extends the input corruption and clean-input reconstruction used in **Denoising Autoencoders**, abbreviated **DAEs** and designated **S2H**, introduced by Vincent et al. (2008, 2010). The extension includes effective hidden-layer corruption, a model-averaging interpretation, supervised as well as unsupervised use, and applicability to other neuron-based architectures such as Boltzmann Machines.

Test-time weight scaling permits much higher noise levels than usually work best for denoising autoencoders. The following comparison reports typical or often optimal levels; S2H means denoising autoencoders, and QX2 means dropout.

| Setting | Noise level reported |
|---|---|
| S2H, typical best | 5% |
| QX2, often optimal input-unit removal | 20% |
| QX2, often optimal hidden-unit removal | 50% |

Stochastic minimization under a noise distribution minimizes an expected loss. In contrast, the adversarial approaches of Globerson and Roweis (2006) and Dekel et al. (2010) let an adversary choose removals subject to a fixed maximum number, rather than drawing removals from a probability distribution. Those adversarial models have no hidden units.

In simple cases, analytically marginalizing dropout produces deterministic regularizers. Van der Maaten et al. (2013) investigated exponential-family input-noise regularizers, including dropout under the name “blankout noise,” but limited the investigation to models without hidden layers. Chen et al. (2012) examined marginalization for denoising autoencoders. Wang and Manning (2013) proposed marginalization as a way to accelerate dropout.

## 4. Model Description

Dropout makes random retention choices for units. In the simplest construction, these choices are mutually independent and use a common fixed retention probability \(p\). Validation can select the retention probability. The usual default is close to optimal across many tasks and networks, whereas optimal retention for input units is generally nearer complete retention.

Under the prescribed reduced test-time weights, every hidden unit’s expected output under the training deletion distribution equals that hidden unit’s actual test-time output.

To define the computation, let a network have \(L\) hidden layers. Let \(l\in\{1,\ldots,L\}\) index hidden layers; let \(\mathbf z^{(l)}\) and \(\mathbf y^{(l)}\) denote the inputs and outputs of layer \(l\); and set \(\mathbf y^{(0)}=\mathbf x\), where \(\mathbf x\) is the network input. Let \(\mathbf W^{(l)}\) and \(\mathbf b^{(l)}\) denote the weights and biases for layer \(l\), and let \(\mathbf w_i^{(l)}\) denote the weight vector for unit \(i\). For \(l\in\{0,\ldots,L-1\}\), ordinary feed-forward computation for hidden unit \(i\) is

\[
z_i^{(l+1)}
 =\mathbf w_i^{(l+1)}\mathbf y^l+b_i^{(l+1)},
\qquad
y_i^{(l+1)}=f\!\left(z_i^{(l+1)}\right).
\]

Here \(z_i^{(l+1)}\) is the unit’s preactivation, \(y_i^{(l+1)}\) is the unit’s output, and \(f\) is the activation function. Any activation function is admissible, including the logistic function

\[
f(x)=\frac{1}{1+\exp(-x)}.
\]

Dropout replaces the ordinary computation by first applying a random retention mask. For unit \(j\) in layer \(l\), let \(r_j^{(l)}\) be a Bernoulli random variable: \(r_j^{(l)}=1\) with probability \(p\), and \(r_j^{(l)}=0\) otherwise. Let \(\mathbf r^{(l)}\) collect the layer’s independent retention variables, and let \(*\) denote element-wise multiplication. The thinned layer outputs are then

\[
r_j^{(l)}\sim\operatorname{Bernoulli}(p),
\qquad
\widetilde{\mathbf y}^{(l)}
 =\mathbf r^{(l)}*\mathbf y^{(l)}.
\]

These thinned outputs, rather than the complete layer outputs, enter the next layer:

\[
z_i^{(l+1)}
 =\mathbf w_i^{(l+1)}\widetilde{\mathbf y}^{\,l}
   +b_i^{(l+1)},
\qquad
y_i^{(l+1)}=f\!\left(z_i^{(l+1)}\right).
\]

Applying the masking operation at every layer samples a subnetwork. Loss derivatives then backpropagate through that sampled subnetwork. Both hidden and visible units can be removed.

At inference, no units are removed. Instead, the test-time weight matrix \(\mathbf W_{\mathrm{test}}^{(l)}\) is reduced according to

\[
\mathbf W_{\mathrm{test}}^{(l)}=p\mathbf W^{(l)}.
\]

**Figure 2.** Training, left: a unit is present with probability \(p\), and the unit’s connections to the next layer have weights \(w\). Testing, right: the unit is always present, and the unit’s outgoing weights are \(pw\).

**Figure 3.** Standard computation, left: layer outputs \(y_1^{(l)},y_2^{(l)},y_3^{(l)}\) connect through weights \(\mathbf w_i^{(l+1)}\); bias \(b_i^{(l+1)}\) comes from a \(+1\) unit; the computation produces preactivation \(z_i^{(l+1)}\), applies activation \(f\), and gives output \(y_i^{(l+1)}\). Dropout computation, right: corresponding retention gates \(r_1^{(l)},r_2^{(l)},r_3^{(l)}\) first produce thinned outputs \(\widetilde y_1^{(l)},\widetilde y_2^{(l)},\widetilde y_3^{(l)}\); the remaining operations retain their form.

## 5. Learning QX2 Nets

### 5.1 Backpropagation

Stochastic gradient descent for dropout differs from standard-network training only in the sampled thinned network used for each case. Each case in a mini-batch receives forward and backward propagation through its own sampled network. Parameter gradients are then averaged across cases. A case that does not use a parameter contributes zero to that parameter’s gradient. Momentum, annealed learning rates and L2 regularization remain useful.

Dropout alone gives significant gains. Combining dropout with high momentum, large decaying learning rates, and **max-norm regularization**, designated **V9C**, gives a further significant boost. Max-norm regularization constrains each hidden unit’s incoming-weight vector \(\mathbf w\) by

\[
\|\mathbf w\|_2\leq c,
\]

where \(\|\mathbf w\|_2\) is the vector’s Euclidean norm and \(c\) is the maximum allowed norm. Whenever the incoming-weight vector leaves the radius-\(c\) ball, projection returns the vector to the ball’s surface. Validation tunes the hyperparameter \(c\).

The incoming-weight constraint usually improves deep-network stochastic-gradient training even without dropout, and the constraint was previously used for collaborative filtering (Srebro and Shraibman, 2005).

A proposed optimization account connects these training choices as follows. The fixed-radius weight ball permits huge learning rates without weight explosion. Dropout noise then explores weight-space regions that would otherwise be difficult to reach. Finally, decaying learning rates produce progressively shorter and less exploratory steps, ending at a minimum.

### 5.2 Unsupervised Pretraining

Unlabeled-data use through pretraining can produce significant improvements over random initialization in certain cases. Pretraining followed by backpropagation finetuning can use stacked Restricted Boltzmann Machines (Hinton and Salakhutdinov, 2006), autoencoders (Vincent et al., 2010), or **Deep Boltzmann Machines**, abbreviated **DBM** and designated **E5Q** (Salakhutdinov and Hinton, 2009). All of these pretrained models admit dropout finetuning.

Pretraining itself is unchanged. Before dropout finetuning, pretrained weights are multiplied by \(1/p\), where \(p\) is the retention probability. This multiplication matches each unit’s expected output under random removal to that unit’s pretraining output.

Smaller finetuning learning rates preserve useful pretrained information and improve final generalization relative to finetuning without dropout. In contrast, initial experiments using learning rates comparable to the best rates for randomly initialized networks erased the pretrained information.

## 6. Experimental Results

The experiments cover object classification, digit recognition, speech, documents and computational-biology analysis. Thus, the evaluations span diverse applications rather than a single domain. Generalization improved over non-dropout neural networks on every tested data set; the detailed comparisons appear in §§6.1–6.5.

The collections are:

- **MNIST**, designated **H4Z**, standard handwritten-digit toy data.
- **TIMIT**, designated **N8Q**, clean-speech benchmarking data.
- **CIFAR-10**, designated **J3V**, and **CIFAR-100**, designated **W6R**, tiny natural-image collections described by Krizhevsky (2009).
- **Street View House Numbers**, abbreviated **SVHN** and designated **A2F**, Google Street View house-number images supplied by Netzer et al. (2011).
- **ImageNet**, designated **K9D**, representing natural images at larger scale.
- **Reuters-RCV1**, designated **M5X**, Reuters newswire documents.
- The **Alternative Splicing data set**, designated **Z7P**, for predicting gene splicing from RNA features, from Xiong et al. (2011). RNA denotes ribonucleic acid.

Table 1 uses these data-set codes. In the size columns, K denotes thousands and M denotes millions; “dim” denotes dimensions.

| Data set | Domain | Dimensionality | Training set | Test set |
|---|---|---|---|---|
| H4Z | Vision | 784 (28 × 28 grayscale) | 60K | 10K |
| A2F | Vision | 3072 (32 × 32 color) | 600K | 26K |
| J3V/W6R | Vision | 3072 (32 × 32 color) | 60K | 10K |
| K9D (ILSVRC-2012) | Vision | 65536 (256 × 256 color) | 1.2M | 150K |
| N8Q | Speech | 2520 (120-dim, 21 frames) | 1.1M frames | 58K frames |
| M5X | Text | 2000 | 200K | 200K |
| Z7P | Genetics | 1014 | 2932 | 733 |

*Table 1. Data-set overview. ILSVRC is the ImageNet Large Scale Visual Recognition Challenge.*

### 6.1 Image Data

Models using dropout attain state of the art on all five vision collections discussed below: MNIST, Street View House Numbers, CIFAR-10, CIFAR-100 and ImageNet. These collections vary in image type and training-set size.

#### 6.1.1 H4Z

MNIST evaluations here compare permutation-invariant models. The activation classes used in the comparison include **rectified linear units**, abbreviated **ReLUs** and designated **F3N**, and **maxout** units, designated **L8J**. Pretrained networks include stacked-RBM initialization, whose table designation **G6W** means **DBN**, or deep belief network, and Deep Boltzmann Machine initialization, designated **E5Q**. In Table 2, NN means neural network; SVM means support vector machine; NA marks entries not applicable to the stated model; QX2 means dropout; and V9C means max-norm regularization.

Against the best permutation-invariant neural networks lacking both dropout and unsupervised pretraining, dropout reduces error: Table 2 reports 1.60% for the standard logistic neural network and 1.35% for the dropout logistic network. Replacing logistic units with rectified linear units reduces error further to 1.25% (Jarrett et al., 2009), and adding max-norm regularization reduces error to 1.06%.

Greater network size improves results further: the max-norm dropout networks in Table 2 range from 1.06% error for 3 layers of 1024 units to 0.95% for 2 layers of 8192 units. Replacing the activation class with maxout gives 0.94% (Goodfellow et al., 2013). Standard regularization and early stopping make the largest reported network difficult to generalize; in contrast, dropout prevents that network’s overfitting without requiring early stopping.

Pretrained networks improve as well. Table 2 reports a reduction from 1.18% to 0.92% for stacked-RBM initialization and from 0.96% to 0.79% for Deep Boltzmann Machine initialization. The 0.79% result is the best reported permutation-invariant performance. Better results remain possible by using two-dimensional spatial information and distorted training-image augmentation; those approaches are tested here on more interesting collections.

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

Figure 4 compares otherwise identical architectures trained with and without dropout. Every hyperparameter, including retention probability, is held fixed across architectures. The resulting test-error trajectories form two sharply separated clusters. Thus, Figure 4 shows a large improvement across architectures without architecture-specific hyperparameter tuning.

**Figure 4.** Test error during training for different architectures, with and without dropout; curves labeled “Without QX2” and “With QX2,” where QX2 means dropout.

| Axis | Label | Displayed ticks |
|---|---|---|
| Horizontal | Number of weight updates | 0, 200000, 400000, 600000, 800000, 1000000 |
| Vertical | Classification Error % | 1.0, 1.5, 2.0, 2.5 |

#### 6.1.2 A2F

The portion of Street View House Numbers used here contains color images approximately centered on one digit of a house number. The task is to identify that digit.

Convolutional networks (LeCun et al., 1989) outperform other methods on this task. As Table 3 shows, adding dropout to the fully connected layers of a max-pooling convolutional network improves error from 3.95% to 3.02%. Adding dropout to convolutional layers as well improves error further to 2.55%, and maxout permits an additional gain to 2.47%.

Because convolutional layers have few parameters, one might expect little overfitting and therefore little benefit from dropout in those layers. Nevertheless, corruption in lower layers supplies noisy inputs to the higher fully connected layers, preventing the higher layers from overfitting.

In Table 3, Conv Net means convolutional network; QX2 means dropout; and L8J means maxout. WDCH identifies the reported binary-feature method, and HOG denotes histograms of oriented gradients. L2 and L4 in the pooling rows identify pooling methods, rather than the L2 weight regularizer.

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

#### 6.1.3 J3V and W6R

The CIFAR-10 and CIFAR-100 experiments use no data augmentation apart from input dropout. For CIFAR-10, Bayesian hyperparameter optimization gives the 14.98% result of Snoek et al. (2012). As Table 4 shows, dropout in fully connected layers improves error to 14.32%; extending dropout to every layer improves error to 12.61%; and substituting maxout further reduces error to 11.68% (Goodfellow et al., 2013).

CIFAR-100 also shows a large dropout improvement: Table 4 reports 43.48% for the hand-tuned max-pooling convolutional network and 37.20% for max pooling with dropout in all layers.

In Table 4, J3V means CIFAR-10; W6R means CIFAR-100; Conv Net means convolutional network; QX2 means dropout; and L8J means maxout.

| Method | J3V | W6R |
|---|---|---|
| Conv Net + max pooling (hand tuned) | 15.60 | 43.48 |
| Conv Net + stochastic pooling (Zeiler and Fergus, 2013) | 15.13 | 42.51 |
| Conv Net + max pooling (Snoek et al., 2012) | 14.98 | - |
| Conv Net + max pooling + QX2 fully connected layers | 14.32 | 41.26 |
| Conv Net + max pooling + QX2 in all layers | 12.61 | 37.20 |
| Conv Net + L8J (Goodfellow et al., 2013) | 11.68 | 38.57 |

*Table 4. J3V and W6R error rates.*

**Figure 5.** Image samples: (a) Street View House Numbers, designated A2F, and (b) CIFAR-10, designated J3V; each row represents a different category.

#### 6.1.4 K9D

ImageNet contains over 15 million labeled high-resolution images in roughly 22,000 categories. The annual ImageNet Large Scale Visual Recognition Challenge, or ILSVRC, began in 2010 as part of the Pascal Visual Object Challenge. The competition uses a subset with roughly 1000 images in each of 1000 categories.

Conventional reporting includes top-1 and top-5 errors. Top-1 uses the model’s best guess; top-5 counts test images whose correct label is absent from the five labels assigned the highest probabilities by the model. Only ILSVRC-2010 has available test labels, so ILSVRC-2010 supplies most experiments. Because ILSVRC-2012 test labels are unavailable, reported ILSVRC-2012 test results are restricted to the final submission, while model variations are evaluated on validation data.

The convolutional-network/dropout model won ILSVRC-2012. The advantage over standard vision-feature methods is staggering: Table 6 reports top-5 test errors of 27.3 and 26.2 for the standard vision-feature methods, versus 16.4 for the average of 5 convolutional networks with dropout. The broader comparison below summarizes this contrast as about 26% versus about 16%. Moreover, Figure 6 shows that the model’s predictions remain very reasonable when the best guess is wrong.

On ILSVRC-2010, convolutional networks with dropout also outperform the alternatives by a large margin: Table 5 reports top-1/top-5 errors of 37.5/17.0, versus 47.1/28.2 for sparse coding and 45.7/25.7 for SIFT plus Fisher Vectors. Krizhevsky et al. (2012) detail the architecture and implementation.

In Tables 5–6, Conv Net means convolutional network and QX2 means dropout. SIFT denotes scale-invariant feature transform; SVM denotes support vector machine; FVs denotes Fisher Vectors; LBP denotes local binary patterns; GIST names the listed image descriptor; and CSIFT denotes color SIFT. “Avg” means average, and “val” means validation.

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

**Figure 6.** ImageNet test cases with the four most probable predicted labels; horizontal bar length is proportional to assigned probability, and pink marks ground truth.

### 6.2 N8Q

TIMIT comparisons include unpretrained networks and networks pretrained as deep belief networks, designated G6W. Table 7 also includes an mcRBM-G6W model; mcRBM denotes the mean-covariance Restricted Boltzmann Machine used in that listed pretraining method. In the table, NN means neural network and QX2 means dropout.

Dropout improves the unpretrained network: the 6-layer network’s phone error falls from 23.4% to 21.8%. Dropout also improves both pretrained comparisons. For the 4-layer pretrained network, error falls from 22.7% to 19.7%; for the 8-layer pretrained network, the tabulated comparison is 20.7% to 19.7%. Thus, the two dropout-pretrained depths attain the same 19.7% phone error.

The prose comparison for the deeper pretrained network instead uses a pre-dropout baseline of 20.5%, as recorded separately below. That prose baseline is distinct from the 20.7% baseline in the corresponding 8-layer row of Table 7.

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

Bag-of-words document classification on a subset of Reuters-RCV1 improves with dropout: the best neural-network classifier’s error falls from 31.05% to 29.62%, as shown below. Improvement occurred for every architecture tried. However, the gain is smaller than in vision and speech classification. A possible explanation is the collection’s size and, consequently, its less serious overfitting.

In the following table, QX2 means dropout.

| Best neural-net document classifier | Error rate |
|---|---|
| Without QX2 | 31.05% |
| With QX2 | 29.62% |

### 6.4 Bayesian Neural Networks

Bayesian neural networks average over network structures and parameters using posterior weights (Neal, 1996). This procedure is the proper averaging because the weights incorporate both the prior and data fit. In contrast, dropout assigns equal weights to exponentially many models with shared weights.

Bayesian networks are extremely useful when data are scarce, including in medical diagnosis, genetics, drug discovery and other computational-biology applications. However, Bayesian networks train slowly, scale poorly to very large networks and require expensive predictions from many large networks. Dropout networks train and infer much faster.

The Alternative Splicing data set is a small genetics collection on which Bayesian networks are known to perform well and achieve state of the art (Xiong et al., 2011). Alternative splicing contributes significantly to cellular diversity in mammalian tissues. Therefore, predicting alternative splicing’s tissue- and condition-dependent occurrence matters for understanding many human diseases.

In this task, RNA features predict probabilities of three biologically important splicing-related events. Predictions are scored by **Code Quality**, designated **C4Y**, a distributional measure representing negative **Kullback–Leibler divergence**, abbreviated **KL divergence**, between target and predicted distributions. Higher Code Quality values are preferred.

Table 8 shows that Bayesian neural networks outperform dropout on this data set, with Code Quality of 623 bits versus 567 bits. Nevertheless, dropout substantially improves standard neural networks, whose score is 440 bits, and exceeds every other listed method.

One defense against overfitting with a small training set is input reduction by **principal component analysis**, abbreviated **PCA**, followed by support vector machines or logistic regression. Dropout instead prevents overfitting without dimensionality reduction. The dropout network uses vastly more hidden units than the Bayesian network, yet exhibits strong regularization. Appendix B.6 gives the hidden-unit scales.

In Table 8, C4Y means Code Quality, QX2 means dropout, PCA means principal component analysis, and SVM means support vector machine.

| Method | C4Y (bits) |
|---|---|
| Neural Network (early stopping) (Xiong et al., 2011) | 440 |
| Regression, PCA (Xiong et al., 2011) | 463 |
| SVM, PCA (Xiong et al., 2011) | 487 |
| Neural Network with QX2 | 567 |
| Bayesian Neural Network (Xiong et al., 2011) | 623 |

*Table 8. Z7P results.*

### 6.5 Standard Regularizers

The MNIST regularizer comparison uses a shared rectified-linear-unit architecture trained with stochastic gradient descent. Validation selected decay constants, target sparsity, dropout rate and the max-norm bound.

The candidates include L2 regularization, more generally Tikhonov regularization (Tikhonov, 1943), lasso (Tibshirani, 1996), KL-sparsity and max-norm regularization. KL-sparsity is the listed sparsity regularizer based on Kullback–Leibler divergence. In Table 9, Y5F means L2 regularization; O3U means L1 regularization; V9C means max-norm regularization; and QX2 means dropout.

Dropout combined with max-norm regularization gives the lowest generalization error: Table 9 reports 1.05%, compared with 1.25% for dropout plus L2 regularization, 1.35% for max-norm alone, and 1.55%–1.62% for the other listed combinations.

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

To examine feature differences, Figure 7 compares identical single-hidden-layer autoencoders with 256 rectified linear units. One autoencoder has no dropout; the other applies hidden-layer dropout with retention probability \(p=0.5\).

The first-layer visual features differ markedly even though test reconstruction errors are similar. Without dropout, individual hidden units do not appear to detect meaningful features. In contrast, dropout units detect edges, strokes and spots in different image regions, as shown in Figure 7. Breaking co-adaptations is proposed as the principal source of lower generalization error.

The proposed connection is as follows. A parameter’s ordinary derivative prescribes loss reduction conditional on every other unit’s behavior. This conditional dependence allows units to repair one another’s mistakes and build complex co-adaptations that do not generalize. Under the dropout hypothesis, particular partners become unreliable. Consequently, a hidden unit cannot depend on specific corrective units and must succeed across the many contexts supplied by the remaining units.

**Figure 7.** MNIST features from identical single-hidden-layer autoencoders with 256 rectified linear units, designated F3N: (a) no dropout, (b) hidden-layer dropout with \(p=0.5\). The former features are co-adapted for reconstruction; the latter have the localized structures described above.

### 7.2 Sparsity

Dropout produces sparse hidden activations without sparsity-inducing regularizers. The autoencoder comparison uses a random test-set mini-batch. Good sparsity requires both few strongly active units for each case and low average activity of each unit across cases.

For each model in Figure 8, the left histogram distributes hidden-unit mean activations across the mini-batch, while the right histogram distributes the activations themselves. The observations below show the difference: mean activity is about or close to 2.0 without dropout, versus about or around 0.7 with dropout. Moreover, dropout produces a sharp activation peak at zero and very few highly active units, rather than substantial off-zero mass and a large fraction of highly active units.

In the following table, QX2 means dropout.

| Observation | No QX2 | QX2 |
|---|---|---|
| Most units’ mean activation; overall hidden-unit mean | about / close to 2.0 | about / around 0.7 |
| Activation histogram | Huge mode away from zero; substantial off-zero mass | Sharp zero peak |
| Highly active units | Large fraction | Very few |

**Figure 8.** Sparsity histograms for the rectified-linear-unit autoencoders: (a) without dropout, (b) dropout with \(p=0.5\).

### 7.3 Rate

Let \(p\) denote the probability of retaining a unit, and let \(n\) denote a layer’s hidden-unit count. Thus, \(pn\) is the layer’s expected retained width.

When architecture is held fixed, low retention produces underfitting: few units remain active, and both training error and test error are high. As retention rises, error falls, flattens over \(0.4\leq p\leq0.8\), and then rises near complete retention.

Holding expected retained width fixed instead substantially reduces the errors at low retention. In particular, at \(p=0.1\), error is 2.7% for fixed architecture but 1.7% for fixed expected retained width. The corresponding inference networks differ in size, because maintaining the same expected retained width at lower retention requires a larger unthinned network.

For the chosen expected widths, the apparently best retention is close to 0.6, while the usual default of 0.5 is close to optimal.

| Comparison quantity | Value |
|---|---|
| Flat-error retention interval, fixed architecture | \(0.4\leq p\leq0.8\) |
| Retention used for low-retention comparison | \(p=0.1\) |
| Error, fixed architecture | 2.7% |
| Error, fixed expected retained width | 1.7% |
| Apparently best retention for the chosen expected widths | close to 0.6 |
| Usual default, close to optimal in that comparison | 0.5 |

**Figure 9.** MNIST training and test errors versus retention probability: (a) fixed hidden-unit count \(n\), (b) fixed expected retained width \(pn\).

| Panel | Horizontal label and ticks | Vertical label and ticks | Curves |
|---|---|---|---|
| (a) | Probability of retaining a unit \((p)\): 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 | Classification Error %: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5 | Test Error; Training Error |
| (b) | Probability of retaining a unit \((p)\): 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 | Classification Error %: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 | Test Error; Training Error |

### 7.4 Data-Set Size

Figure 10 compares MNIST error for randomly selected training sets of different sizes. The two smallest tested training sets show no dropout improvement, because the network can memorize those sets despite removal noise. As data size increases, the dropout gain first increases and then decreases.

For a fixed architecture and dropout rate, the proposed “sweet spot” therefore contains enough data to resist noisy memorization, but not enough data to make overfitting unimportant. Standard training of huge networks on small collections massively overfits. Consequently, generalizing from many parameters with little data tests a regularizer.

**Figure 10.** MNIST error versus randomly selected training-set size, with curves “With QX2” and “Without QX2,” where QX2 means dropout.

| Axis | Label | Displayed ticks |
|---|---|---|
| Horizontal | Dataset size | \(10^2,10^3,10^4,10^5\) |
| Vertical | Classification Error % | 0, 5, 10, 15, 20, 25, 30 |

### 7.5 X9M vs. T6R

This comparison evaluates **Monte-Carlo model averaging**, designated X9M, against **weight scaling**, designated T6R. For each test case, Monte-Carlo model averaging samples \(k\) thinned networks and averages their predictions, where \(k\) is the sample count. Weight scaling instead uses a single unthinned network with reduced weights.

The MNIST comparison in Figure 11 varies the Monte-Carlo sample count and includes a horizontal weight-scaling reference. Thus, the comparison measures how the finite-sample average’s error approaches the true model average’s error.

Monte-Carlo model averaging becomes as good as weight scaling at around \(k=50\). With sufficiently many samples, Monte-Carlo model averaging then performs slightly better, within one standard deviation. Consequently, weight scaling is a fairly good approximation to the true model average.

| Quantity | Value |
|---|---|
| Sample count at which X9M becomes as good as T6R | around \(k=50\) |

**Figure 11.** Monte-Carlo model averaging, designated X9M, versus weight scaling, designated T6R.

| Axis | Label | Displayed ticks |
|---|---|---|
| Horizontal | Number of samples used for X9M \((k)\) | 0, 20, 40, 60, 80, 100, 120 |
| Vertical | Test Classification error % | 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35 |

Curves: “X9M” and “Approximate averaging by T6R.”

## 8. B8T

### 8.1 Model Description

A **Dropout Restricted Boltzmann Machine**, or Dropout RBM, designated B8T, is a mixture of exponentially many shared-weight Restricted Boltzmann Machines, designated U2L. Each mixture member uses a different hidden-unit subset. Conditional on a retention vector, the visible–hidden distribution is therefore an ordinary Restricted Boltzmann Machine distribution with the designated hidden units removed.

For the ordinary model, let \(\mathbf v\in\{0,1\}^D\) be the vector of \(D\) binary visible units and \(\mathbf h\in\{0,1\}^F\) the vector of \(F\) binary hidden units. Let \(\theta=\{\mathbf W,\mathbf a,\mathbf b\}\) collect the weight matrix and bias vectors, and let \(Z(\theta)\) be the partition function that normalizes the distribution. Superscript \(\top\) denotes transpose. An ordinary Restricted Boltzmann Machine defines

\[
P(\mathbf h,\mathbf v;\theta)
=\frac{1}{Z(\theta)}
 \exp\!\left(
 \mathbf v^\top\mathbf W\mathbf h+
 \mathbf a^\top\mathbf h+
 \mathbf b^\top\mathbf v
 \right).
\]

The Dropout RBM augments this model with a binary retention vector \(\mathbf r\in\{0,1\}^F\). Each hidden unit \(h_j\) is retained independently when \(r_j=1\), with probability \(p\), and dropped otherwise. The joint distribution factors as

\[
P(\mathbf r,\mathbf h,\mathbf v;p,\theta)
=P(\mathbf r;p)P(\mathbf h,\mathbf v\mid\mathbf r;\theta),
\]

where

\[
P(\mathbf r;p)=\prod_{j=1}^{F}p^{r_j}(1-p)^{1-r_j}.
\]

To enforce removal, define \(\mathbf1(\cdot)\) as the indicator function, equal to one when the stated condition holds and zero otherwise. Let \(g(h_j,r_j)\) enforce \(h_j=0\) whenever \(r_j=0\), and let \(Z'(\theta,\mathbf r)\) normalize the conditional distribution. Then

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

The conditional hidden distribution is factorial, meaning that the distribution factors across hidden units given the retention vector and visible units:

\[
P(\mathbf h\mid\mathbf r,\mathbf v)
=\prod_{j=1}^{F}P(h_j\mid r_j,\mathbf v).
\]

With \(\sigma(u)=1/(1+\exp(-u))\) denoting the logistic function, the hidden-unit conditional is

\[
P(h_j=1\mid r_j,\mathbf v)
=\mathbf1(r_j=1)\,
 \sigma\!\left(b_j+\sum_iW_{ij}v_i\right).
\]

The visible conditional remains that of an ordinary Restricted Boltzmann Machine:

\[
P(\mathbf v\mid\mathbf h)
=\prod_{i=1}^{D}P(v_i\mid\mathbf h),
\]

\[
P(v_i=1\mid\mathbf h)
=\sigma\!\left(a_i+\sum_jW_{ij}h_j\right).
\]

### 8.2 Learning B8T

Contrastive Divergence, abbreviated **CD**, and other Restricted Boltzmann Machine learning algorithms apply directly (Hinton et al., 2006). Training first samples the retention vector \(\mathbf r\), then trains only the retained hidden units. Each training case in every mini-batch receives a different retention vector. The experiments use **CD-1**, Contrastive Divergence with one step.

### 8.3 Features

Figure 12 tests whether the feature-quality effect of feed-forward dropout transfers to Dropout RBMs. Compared with the sharply defined stroke-like features of ordinary Restricted Boltzmann Machines, Dropout RBM features appear coarser. Dropout RBMs also have very few dead units relative to ordinary Restricted Boltzmann Machines.

**Figure 12.** MNIST features from 256-hidden-unit models, ordered by \(L_2\), or Euclidean, norm: (a) ordinary Restricted Boltzmann Machine, designated U2L, (b) Dropout RBM, designated B8T, with retention probability \(p=0.5\).

### 8.4 Sparsity

Without an additional sparsity-inducing regularizer, Dropout RBMs learn much sparser representations than ordinary Restricted Boltzmann Machines. Figure 13 compares hidden-activation and mean-activation histograms from a test mini-batch.

**Figure 13.** (a) Without dropout: many activations are away from zero. (b) Dropout with retention probability \(p=0.5\): many activations are near zero, and very few activations are high.

## 9. Marginalizing QX2

Marginalizing dropout noise produces deterministic models that require no random bits and admit gradients of marginalized losses. This differs from standard “Monte-Carlo” dropout, which samples the noise.

Globerson and Roweis (2006) investigated deterministic robustness to test-time feature deletion. Srivastava (2013) discussed dropout marginalization for linear regression, and Wager et al. (2013) described dropout as an adaptive regularizer. The related investigations retain the scopes described in §3: Chen et al. (2012) examined denoising-autoencoder marginalization; Wang and Manning (2013) proposed marginalization to accelerate dropout; and van der Maaten et al. (2013) investigated exponential-family input-noise regularizers for models without hidden layers.

### 9.1 Linear Regression

Let \(\mathbf X\in\mathbb R^{N\times D}\) contain \(N\) data points with \(D\) input dimensions, let \(\mathbf y\in\mathbb R^N\) contain the targets, and let \(\mathbf w\in\mathbb R^D\) be the weight vector. Here \(\mathbb R\) denotes the real numbers. Ordinary linear regression minimizes the squared residual norm,

\[
\|\mathbf y-\mathbf X\mathbf w\|^2.
\]

For dropout linear regression, each input dimension is retained independently with probability \(p\). Let \(\mathbf R\in\{0,1\}^{N\times D}\) be the retention matrix, with \(R_{ij}\sim\operatorname{Bernoulli}(p)\), and let \(*\) denote element-wise multiplication. The corrupted input matrix is therefore \(\mathbf R*\mathbf X\).

The marginalized objective takes the expectation over retention matrices, where \(\mathbb E\) denotes expectation:

\[
\underset{\mathbf w}{\operatorname{minimize}}\quad
\mathbb E_{\mathbf R\sim\operatorname{Bernoulli}(p)}
\left[\|\mathbf y-(\mathbf R*\mathbf X)\mathbf w\|^2\right].
\]

Marginalizing the retention matrix reduces this objective to

\[
\underset{\mathbf w}{\operatorname{minimize}}\quad
\|\mathbf y-p\mathbf X\mathbf w\|^2+
p(1-p)\|\mathbf\Gamma\mathbf w\|^2,
\]

where the penalty matrix is

\[
\mathbf\Gamma
=\left(\operatorname{diag}(\mathbf X^\top\mathbf X)\right)^{1/2}.
\]

Here \(\operatorname{diag}\) forms the diagonal matrix from the diagonal entries of \(\mathbf X^\top\mathbf X\).

Thus, dropout linear regression is, in expectation, ridge regression with a particular penalty matrix. A dimension’s standard deviation scales the cost on that dimension’s weight, squeezing more strongly the weights of highly variable dimensions.

Absorbing retention into the weights, with \(\widetilde{\mathbf w}=p\mathbf w\), gives the alternative form

\[
\underset{\mathbf w}{\operatorname{minimize}}\quad
\|\mathbf y-\mathbf X\widetilde{\mathbf w}\|^2+
\frac{1-p}{p}\|\mathbf\Gamma\widetilde{\mathbf w}\|^2.
\]

This form exposes the regularization constant \((1-p)/p\). The constant is small near complete retention and becomes larger with greater removal.

### 9.2 Logistic Regression and Deep Networks

For logistic regression, approximate marginalization outperforms Monte-Carlo dropout in both training time and generalization (Wang and Manning, 2013).

However, closed-form marginalized models are difficult to obtain for logistic regression and deep networks. Under reasonable assumptions, distributions of logistic-unit inputs and marginalized-model gradients are Gaussian, with efficiently computable means and variances. The assumptions become successively weaker as layers are added. Consequently, these results are not directly applicable to deep networks.

## 10. Multiplicative Gaussian Noise

Binary dropout multiplies a hidden activation by a Bernoulli variable that is one with retention probability \(p\) and zero otherwise. This construction generalizes to other multiplier distributions. Here, **Bernoulli dropout** is designated **P7V**, and **Gaussian dropout** is designated **D4K**.

Let \(h_i\) denote hidden unit \(i\)’s activation, and let \(\mathcal N(\mu,\sigma^2)\) denote a Gaussian distribution with mean \(\mu\) and variance \(\sigma^2\). Drawing a multiplier from \(\mathcal N(1,1)\) is equivalent to adding zero-mean Gaussian noise whose standard deviation equals the unit activation:

\[
h_i\longmapsto h_i+h_ir,\qquad r\sim\mathcal N(0,1),
\]

or, equivalently,

\[
h_i\longmapsto h_ir',\qquad r'\sim\mathcal N(1,1).
\]

Extending the multiplier to \(r'\sim\mathcal N(1,\sigma^2)\) makes the standard deviation \(\sigma\) a tunable hyperparameter analogous to the Bernoulli retention probability \(p\). Because expected activations are unchanged, Gaussian dropout requires no test-time weight scaling.

To compare the two multiplier distributions, first express Bernoulli dropout with mean-one training multipliers. Retaining activations during training and multiplying outgoing weights by \(p\) at testing is equivalent to multiplying retained training activations by \(1/p\) and leaving test weights unchanged, provided that layerwise learning rates and weight initializations are appropriately rescaled.

In this latter representation, the Bernoulli multiplier \(r_b\) is \(1/p\) with probability \(p\) and zero otherwise. Therefore,

\[
\mathbb E[r_b]=1,\qquad
\operatorname{Var}[r_b]=\frac{1-p}{p},
\]

where \(\operatorname{Var}\) denotes variance. Choosing \(\sigma^2=(1-p)/p\) for a Gaussian multiplier \(r_g\) matches these first and second moments:

\[
\mathbb E[r_g]=1,\qquad
\operatorname{Var}[r_g]=\frac{1-p}{p}.
\]

Despite these matched moments, the multiplier distributions occupy opposite entropy extremes. Given those moments, the Gaussian multiplier has the highest entropy and the Bernoulli multiplier the lowest. Both work well, with preliminary evidence slightly favoring the high-entropy case.

Specifically, Table 10 reports MNIST errors of \(1.08\pm0.04\)% for Bernoulli dropout and \(0.95\pm0.04\)% for Gaussian dropout. For CIFAR-10, the corresponding errors are \(12.6\pm0.1\)% and \(12.5\pm0.1\)%. Thus, in these preliminary experiments, multiplicative Gaussian corruption works as well as, or perhaps better than, binary corruption.

In Table 10, H4Z means MNIST; J3V means CIFAR-10; P7V means Bernoulli dropout; D4K means Gaussian dropout; and “conv” means convolutional.

| Data set | Architecture | P7V | D4K |
|---|---|---|---|
| H4Z | 2 layers, 1024 units each | \(1.08\pm0.04\) | \(0.95\pm0.04\) |
| J3V | 3 conv + 2 fully connected layers | \(12.6\pm0.1\) | \(12.5\pm0.1\) |

*Table 10. Classification error %.*

The following settings specify the matched comparison. Retention probabilities \(p\) determine each corresponding Gaussian standard deviation \(\sigma\), and results are averaged over 10 different random seeds.

| Comparison setting | Value |
|---|---|
| H4Z P7V retention, hidden units | \(p=0.5\) |
| H4Z P7V retention, input units | \(p=0.8\) |
| J3V P7V retention, input to top | \(p=(0.9,0.75,0.75,0.5,0.5,0.5)\) |
| D4K standard deviation, each corresponding layer | \(\sigma=\sqrt{(1-p)/p}\) |
| Averaging | 10 different random seeds |

## 11. Conclusion

Dropout repeatedly samples and trains smaller submodels from a large model that would otherwise overfit easily. Restricted Boltzmann Machines fit this construction as well, and Dropout RBMs exhibit desirable empirical properties.

Dropout methods achieve state of the art on Street View House Numbers, ImageNet, CIFAR-100 and MNIST, as reported in Tables 3, 5–6, 4 and 2, respectively. Dropout also considerably improves standard neural networks on other collections, as the speech, text and alternative-splicing comparisons show.

However, training typically takes 2–3 times as long as training a standard network with the same architecture. Noisy parameter updates are a major cause of this additional time: every case trains a different random architecture, and the computed gradients are not gradients of the final inference architecture. The stochasticity likely prevents overfitting. Consequently, dropout presents an overfitting–training-time trade-off in which more training time permits stronger dropout and less overfitting.

Noise marginalization obtains some dropout benefits without stochasticity by producing a regularizer equivalent in expectation. For linear regression, the resulting regularizer is modified L2 regularization. Equivalent regularizers for more complicated models are not obvious. Accelerating dropout therefore remains a direction for future work.

## Acknowledgments

This research was supported by OGS, NSERC and an Early Researcher Award.

## Appendix A. A Practical Guide for Training QX2 Networks

Dropout networks retain neural networks’ requirement for extensive hyperparameter tuning. The following guidance gives practical heuristics for that tuning.

### A.1 Network Size

Let \(n\) be the number of hidden units in an optimal standard-network layer for a task, and let \(p\) be the dropout retention probability. A good corresponding dropout layer should contain at least \(n/p\) units.

The reason is that removal reduces capacity. In a dropout layer containing \(n\) units, only \(pn\) units remain in expectation; moreover, the retained units’ identities change at every sampling, and unrestricted co-adaptation is unavailable. The \(n/p\) width heuristic was useful for both convolutional and fully connected layers.

### A.2 Learning Rate and Momentum

High learning rate and/or momentum significantly accelerate learning. Dropout introduces much more gradient noise than ordinary stochastic gradient descent, causing many gradients to cancel one another. Larger learning rates compensate for this cancellation, while higher momentum reduces the noise’s effect.

The practical values below compare dropout, designated QX2, with standard-network training.

| Quantity | Practical values |
|---|---|
| QX2 learning rate relative to optimal standard-net rate | typically 10–100 times |
| Common standard-net momentum | 0.9 |
| QX2 momentum working substantially better | around 0.95 to 0.99 |

### A.3 V9C

Although large momentum and learning rates accelerate learning, these choices sometimes produce very large weights. Max-norm regularization, designated V9C, addresses this problem by bounding each hidden unit’s incoming-weight-vector norm. The bound is the hyperparameter \(c\).

| Hyperparameter | Typical range |
|---|---|
| \(c\) | 3 to 4 |

### A.4 Rate

Retention probability \(p\) controls dropout intensity. For hidden layers, choosing the retention probability is coupled to choosing layer width. Smaller retention requires greater width, slows training and leads to underfitting. Conversely, large retention can leave too little removal to prevent overfitting. The appropriate input retention also depends on input type.

In the following table, QX2 means dropout.

| Condition | Retention |
|---|---|
| No QX2 | \(p=1\) |
| Stronger QX2 | lower \(p\) |
| Hidden-unit typical range | 0.5 to 0.8 |
| Real-valued inputs, including image patches and speech frames | typically 0.8 |
| Broadly near-optimal simple default | 0.5 |
| Input-unit optimum in general | closer to 1 than to 0.5 |

## Appendix B. Detailed Description of Experiments and Data Sets

Reproduction code is available at <http://www.cs.toronto.edu/~nitish/dropout>. Graphics processing unit, or GPU, implementations use the excellent CUDA libraries cudamat (Mnih, 2009) and cuda-convnet (Krizhevsky et al., 2012).

### B.1 H4Z

The MNIST task assigns each digit image to one of ten digit classes. Validation images were randomly held out from the training set. Hyperparameters were selected to give their best validation error at the chosen training endpoint. After selection, validation and training data were combined, and a new run reached that same endpoint before test evaluation. Easy selection of hyperparameters requiring no early stopping justified the long combined-data run.

The settings below cover the architecture sweep and the additional MNIST experiments. In the table, QX2 means dropout; V9C means max-norm regularization; F3N means rectified linear units; \(p\) is retention probability; \(c\) is the maximum incoming-weight norm; and \(n\) is the hidden-unit count of a layer. Thus, \(pn\) denotes the expected surviving count. K denotes thousands.

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

The additional two-layer networks improve performance; Table 2 reports the corresponding large-network results. Three-layer counterparts are slightly worse at the same dropout level. Increasing dropout improves those three-layer networks, but does not make the three-layer networks outperform the two-layer networks.

For the fixed-expected-width comparison, small retention probability \(p\) requires large hidden-unit count \(n\). Consequently, the comparison matches expected training-network widths while leaving inference-network widths different.

Monte-Carlo model averaging, designated X9M, is the expensive but more correct averaging procedure. For each test case, Monte-Carlo model averaging independently samples \(k\) thinned networks and averages their predictions. As \(k\to\infty\), this result approaches the true model average.

### B.2 A2F

The Street View House Numbers training-image count is approximate. Training data comprise a standard labeled portion and an additional labeled portion of easy examples. Validation draws from both portions using the procedure of Sermanet et al. (2012), with the exact counts shown below.

Red, green and blue, or RGB, pixels were normalized to zero mean and unit variance. Global or local contrast normalization and zero-phase component analysis, or ZCA, whitening gave no noticeable improvement.

The architecture and training settings follow. F3N means rectified linear units; V9C means max-norm regularization; \(p\) denotes retention probability; and \(c\) denotes the maximum incoming-weight norm.

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

Validation tuned these hyperparameters. Because the training set was large, training and validation remained separate during final training. The model with the smallest validation error supplied the reported test error.

### B.3 J3V and W6R

CIFAR-10 and CIFAR-100 each contain 50,000 training images and 10,000 test images, with 10 and 100 categories respectively. Validation used 5,000 training images. After hyperparameter selection, validation and training data were combined for optimization, as in the MNIST procedure in Appendix B.1.

Preprocessing first applies global contrast normalization within each image/color channel, then ZCA whitening. Global contrast normalization computes and subtracts the channel’s mean pixel intensity. ZCA whitening then mean-centers the data, rotates the data onto the principal components, normalizes each component and rotates back.

Architecture and retention rates match the Street View House Numbers configuration in Appendix B.2. However, input-layer learning rates require smaller values.

### B.4 N8Q

TIMIT recordings contain 680 speakers covering eight major American-English dialects. Each speaker reads ten phonetically rich sentences in a controlled, noise-free environment.

The open-source Kaldi toolkit (Povey et al., 2011) generated log-filter banks. A monophone system supplied forced alignment and speech-frame labels. Windows of consecutive frames predict the label of the central frame. Inputs are mean-centered and have unit variance, and no speaker-dependent operations are used.

The settings below use \(p\) for retention probability and \(c\) for the maximum incoming-weight norm. V9C means max-norm regularization; G6W means deep belief network; U2L means Restricted Boltzmann Machine; CD-1 means Contrastive Divergence with one step; and QX2 means dropout. In the learning-rate schedule, \(\epsilon_0\) is the initial learning rate, \(t\) indexes training progress, and \(T\) is the decay scale.

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

Adding max-norm regularization during deep-belief-network finetuning gave no improvement. The smaller finetuning learning rate, about 0.01, was important for the best results.

### B.5 M5X

The full Reuters-RCV1 corpus contains more than 800,000 documents across 103 classes arranged in a tree. The subset used here contains 402,738 articles, a 2000-word vocabulary and 50 disjoint categories. Each document belongs to exactly one category. Equal-sized training and test splits therefore result in more than 200,000 training examples. The articles span varied topics.

### B.6 Z7P

The Alternative Splicing data comprise 3665 cassette exons, 1014 RNA features and four tissue types derived from 27 mouse tissues. Each input has four target softmax units, one per tissue type. Each softmax has the biologically important states \(\{\mathrm{inc},\mathrm{exc},\mathrm{nc}\}\), denoting inclusion, exclusion and no change. The predicted state distributions should match the wet-lab-observed distributions as closely as possible.

Code Quality, designated C4Y, scores these predictions. To define the score, let \(i\) index data points, \(t\) index tissue types, and \(s\) index the three states. Let \(p_{i,t}^{s}\) be the target probability of state \(s\) for tissue type \(t\) and input \(i\). Let \(r_i\) denote input \(i\), and let \(q_t^{s}(r_i)\) be the predicted probability for that input, tissue type and state. Finally, let \(\bar p^{s}\) be the average of \(p_{i,t}^{s}\) over data points \(i\) and tissue types \(t\). Then

\[
\mathrm{C4Y}
=
\sum_{i=1}^{|\text{data points}|}
\sum_{t\in\text{tissue types}}
\sum_{s\in\{\mathrm{inc},\mathrm{exc},\mathrm{nc}\}}
p_{i,t}^{s}
\log\!\left(\frac{q_t^{s}(r_i)}{\bar p^{s}}\right).
\]

Here \(|\text{data points}|\) denotes the number of data points.

The following settings compare dropout with the Bayesian network. QX2 means dropout; V9C means max-norm regularization; and \(p\) in the retention rows denotes retention probability, rather than the target probabilities used in the score above.

| Setting | Specification |
|---|---|
| QX2 hidden-unit scale | 1000s |
| Bayesian-network hidden-unit scale | a few tens |
| Trained QX2 network | two layers, 1024 units each |
| Hidden retention | \(p=0.5\) |
| Input retention | \(p=0.7\) |
| Regularization and rate schedule | V9C; high decaying learning rates |
| Result averaging | same 5 folds as Xiong et al. (2011) |
