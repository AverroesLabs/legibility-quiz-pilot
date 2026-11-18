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

Deep neural networks are models with multiple layers of computational units and many learned parameters. Their size makes them powerful, but also prone to overfitting: learning training-data patterns that do not generalize to new data. Combining independently trained large networks is also slow at test time, when the models make predictions.

QX2, called **dropout**, temporarily removes randomly selected units and their connections during training. This discourages excessive co-adaptation, in which units depend on particular other units to work well. Each removal pattern selects a smaller, or *thinned*, network. The exponentially many possible thinned networks share parameters. A single unthinned network with reduced connection weights approximates their averaged predictions.

Compared with other regularizers—methods that constrain learning to reduce overfitting—QX2 substantially reduces overfitting and gives major improvements. Supervised experiments, which train on examples with target outputs, cover vision, speech recognition, document classification and computational biology. They attain state-of-the-art performance on many benchmarks; results appear in §6, Tables 2–9.

**Keywords:** neural networks, regularization, model combination, deep learning

## 1. Introduction

For a model of fixed size, and with unlimited computation, the best regularization is to average over every parameter setting, weighted by its posterior probability. The posterior is the probability assigned to a parameter setting after observing the training data. Good approximations are sometimes possible for simple or small models (Xiong et al., 2011; Salakhutdinov and Mnih, 2008).

The less expensive construction proposed here approximates an equally weighted geometric mean of predictions from exponentially many learned models. A geometric mean combines values by multiplication and a corresponding root. The models share parameters. A network with \(n\) units has \(2^n\) possible thinned networks, because each unit can be present or absent. Weight sharing keeps the number of stored parameters at \(O(n^2)\), or fewer; \(O(n^2)\) means that the count grows at most on the order of the square of \(n\). A new member is sampled for every presentation of every training case. Each particular member is therefore trained rarely, if ever.

Multiple layers of nonlinear hidden representations can express complicated input–output relationships. A hidden unit is an internal unit, rather than an input or output unit. A nonlinear representation does not restrict its output to a linear function of its input. With limited training data, sampling noise creates apparent relationships that do not occur in genuine test data, even when training and test data come from the same distribution.

Established responses include:

- Stopping training when performance deteriorates on validation data, which are held out to select training choices.
- Soft weight sharing, which encourages learned weights to share values (Nowlan and Hinton, 1992).
- Weight penalties. Familiar examples are O3U, or **L1 regularization**, which penalizes absolute parameter values, and Y5F, or **L2 regularization**, which penalizes squared parameter values.

These methods constrain learned parameters rather than averaging every possible parameter setting.

Model combination nearly always improves machine-learning performance. But training many large networks separately, and then averaging their outputs, is prohibitively expensive. Useful diversity requires different architectures—the arrangements and sizes of layers—or different training data. Optimal hyperparameters, the training choices not learned as ordinary model parameters, are difficult to find for each architecture. Each training run requires substantial computation. Large data requirements can also prevent training different networks on different subsets. Even if many such models can be trained, using all of them for inference may be infeasible in rapid-response applications.

The practical inference rule, T6R or **weight scaling**, uses one unthinned network instead of explicitly evaluating exponentially many members. A more expensive alternative, X9M or **Monte-Carlo model averaging**, samples members and averages their predictions. T6R gives significantly lower generalization error—the error on new data—than other regularizers across varied classification problems; see §6 and Table 9. Its approximation to X9M is tested in §7.5, Figure 11.

The same random unit removal extends beyond feed-forward networks, whose computations pass from inputs toward outputs, to graphical models, which represent relationships among random variables. These include Boltzmann Machines. One such family, U2L or **Restricted Boltzmann Machines (RBM)**, has visible and hidden units. Its augmented form is B8T, or **Dropout RBM**. B8T retains parameter sharing and is defined in §8.

**Figure 1.** Standard neural net, left: two hidden layers. QX2-produced thinned net, right: crossed units removed, including their incoming and outgoing connections.

Paper map: §2 motivation; §3 previous work; §4 formal QX2 model; §5 training algorithm; §6 cross-domain experiments and comparisons with regularization and model combination; §7 network-property effects and hyperparameter interactions; §8 B8T; §9 QX2 marginalization, meaning averaging out the removal noise analytically; Appendix A practical training guidance, including detailed hyperparameter-choice considerations.

## 2. Motivation

Sexual reproduction combines half of one parent’s genes with half of another’s, plus a very small random mutation. Asexual reproduction produces a slightly mutated copy of one parent’s genes. Direct transmission preserves a successful co-adapted gene set. This makes asexual reproduction seem plausibly superior for optimizing individual fitness.

Sexual recombination breaks up co-adapted sets, especially large ones. Intuitively, this disadvantages organisms that already possess complicated co-adaptations. Nevertheless, most advanced organisms evolved through sexual reproduction.

Livnat et al. (2010) suggest that long-run selection may favor gene *mix-ability* rather than individual fitness. A gene set that works with another random gene set is more robust. If a gene cannot rely on a large set of particular partners, it must be useful alone or with only a few other genes. Reducing complex co-adaptations also helps useful new genes spread, because fewer dependencies obstruct their contribution to individual fitness.

QX2-trained hidden units likewise work with randomly selected partners. This favors independently useful, robust features rather than dependence on other hidden units to repair mistakes. Units within a layer still learn different functions. Making many copies of each hidden unit is a poor robustness strategy, just as replica codes—codes that repeat information—poorly address noisy communication channels.

Ten five-person conspiracies probably have an advantage in producing havoc over one conspiracy that needs fifty participants to act correctly. With unchanged conditions and ample rehearsal, the larger conspiracy can work. Under non-stationarity, meaning changing conditions, smaller conspiracies are more likely to persist. Similarly, complex co-adaptations can succeed on training data but fail on novel test data more readily than several simpler co-adaptations that achieve the same result.

## 3. Related Work

QX2 can be interpreted as regularization by noise in hidden-unit states. It extends the input corruption used in S2H, or **Denoising Autoencoders (DAEs)**, introduced by Vincent et al. (2008, 2010). An autoencoder reconstructs its input; a denoising autoencoder reconstructs a clean input from a corrupted version.

QX2 provides effective corruption within hidden layers and a model-averaging interpretation. It applies to supervised and unsupervised learning, the latter using data without target labels. It also applies to other neuron-based architectures, including Boltzmann Machines. Test-time T6R allows much more noise than usually works best for S2H:

| Setting | Noise level reported |
|---|---|
| S2H, typical best | 5% |
| QX2, often optimal input-unit removal | 20% |
| QX2, often optimal hidden-unit removal | 50% |

Stochastic minimization under a noise distribution minimizes expected loss: the average training error measure over that distribution. Globerson and Roweis (2006) and Dekel et al. (2010) instead let an adversary choose removals, subject to a fixed maximum number. Their models have no hidden units.

In simple cases, analytically marginalizing QX2 gives deterministic regularizers. Such regularizers replace sampled noise with an objective that averages over it. Van der Maaten et al. (2013) investigated input-noise regularizers for the exponential family, a class of probability distributions. These included QX2 under the name “blankout noise.” Their work was limited to models without hidden layers. Chen et al. (2012) studied marginalization for denoising autoencoders. Wang and Manning (2013) proposed marginalization to accelerate QX2.

## 4. Model Description

For every hidden unit, its expected output under the training deletion distribution equals its actual test-time output under the prescribed reduced weights. Retention choices are random. In the simplest construction, they are mutually independent and use one fixed probability.

The retention probability can be selected using validation data. The usual default is close to optimal across many tasks and networks. Optimal input retention is generally closer to complete retention; practical values appear in Appendix A.4.

Let a network have \(L\) hidden layers. The index \(l\in\{1,\ldots,L\}\) identifies a hidden layer. Let:

- \(\mathbf z^{(l)}\) be its vector of inputs before applying the activation function.
- \(\mathbf y^{(l)}\) be its vector of outputs.
- \(\mathbf y^{(0)}=\mathbf x\) be the network’s input vector.
- \(\mathbf W^{(l)}\) and \(\mathbf b^{(l)}\) be its weight matrix and bias vector.
- \(\mathbf w_i^{(l)}\) be the weight row for unit \(i\), and \(b_i^{(l)}\) its bias.
- \(f\) be the activation function that converts a unit’s preactivation input to its output.

For \(l\in\{0,\ldots,L-1\}\), ordinary feed-forward computation for hidden unit \(i\) is

\[
z_i^{(l+1)}
 =\mathbf w_i^{(l+1)}\mathbf y^l+b_i^{(l+1)},
\qquad
y_i^{(l+1)}=f\!\left(z_i^{(l+1)}\right).
\]

Here \(\mathbf y^l\) denotes the same layer-output vector as \(\mathbf y^{(l)}\). Multiplying the weight row by that vector sums the weighted inputs. Any activation \(f\) is admissible. One example is the logistic function,

\[
f(x)=\frac{1}{1+\exp(-x)},
\]

where \(x\) is a scalar input and \(\exp\) is the exponential function.

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

The symbol \(\sim\) means “is sampled from.” A Bernoulli variable equals one with probability \(p\), and zero otherwise. The index \(j\) identifies a unit in layer \(l\). Each layer’s vector \(\mathbf r^{(l)}\) contains independent such variables. The symbol \(*\) means element-wise multiplication. Thus \(\widetilde{\mathbf y}^{(l)}\) retains selected outputs and sets the others to zero.

Applying this operation at every layer samples a subnetwork. Backpropagation, which computes loss derivatives backward through the network, then operates through that subnetwork. Both hidden units and visible units—the input units here—can be removed.

At inference, no units are removed. Each outgoing weight is multiplied by its source unit’s retention probability:

\[
\mathbf W_{\mathrm{test}}^{(l)}=p\mathbf W^{(l)}.
\]

Here \(\mathbf W_{\mathrm{test}}^{(l)}\) is the weight matrix used at test time.

**Figure 2.** Training, left: a unit is present with probability \(p\), its connections to the next layer having weights \(w\). Testing, right: the unit is always present, its outgoing weights \(pw\).

**Figure 3.** Standard computation, left: \(y_1^{(l)},y_2^{(l)},y_3^{(l)}\), weights \(\mathbf w_i^{(l+1)}\), bias \(b_i^{(l+1)}\) from a \(+1\) unit, preactivation \(z_i^{(l+1)}\), activation \(f\), output \(y_i^{(l+1)}\). QX2 computation, right: corresponding gates \(r_1^{(l)},r_2^{(l)},r_3^{(l)}\) first produce \(\widetilde y_1^{(l)},\widetilde y_2^{(l)},\widetilde y_3^{(l)}\); the remaining operations retain their form.

## 5. Learning QX2 Nets

### 5.1 Backpropagation

QX2 alone gives significant gains. Combining it with high momentum, large decaying learning rates and V9C, or **max-norm regularization**, gives a further significant boost. Momentum carries part of earlier parameter updates into the current update. The learning rate controls update size; a decaying, or annealed, rate decreases during training.

V9C limits the size of each hidden unit’s incoming-weight vector. It usually improves stochastic-gradient training of deep networks even without QX2. Stochastic gradient descent updates parameters using gradients estimated from sampled training data. The constraint was previously used in collaborative filtering, which predicts preferences from observed user–item relationships (Srebro and Shraibman, 2005). Table 9 compares QX2 and V9C on digit classification.

One proposed optimization account is as follows. A fixed-radius bound allows huge learning rates without exploding weights. QX2 noise helps explore regions of weight space that would otherwise be hard to reach. As the learning rate decays, steps become shorter and less exploratory, ending at a minimum.

Training differs from ordinary stochastic gradient descent only in the sampled thinned network for each case. In a mini-batch—a group of cases used for one update—each case gets its own sampled network for forward and backward propagation. Parameter gradients are averaged across cases. A case that does not use a parameter contributes zero to that parameter’s gradient. Momentum, annealed learning rates and Y5F remain useful.

For an incoming hidden-unit weight vector \(\mathbf w\), V9C imposes

\[
\|\mathbf w\|_2\leq c.
\]

The Euclidean norm \(\|\mathbf w\|_2\) is the square root of the sum of squared weights. The hyperparameter \(c\) is the largest allowed norm and is tuned on validation data. Whenever an update moves \(\mathbf w\) outside the radius-\(c\) ball, projection rescales it back to the ball’s surface.

### 5.2 Unsupervised Pretraining

Unsupervised pretraining learns initial parameters from unlabeled data. Finetuning then adjusts those parameters by backpropagation for the target task.

With QX2 finetuning, smaller learning rates preserve useful pretrained information and improve final generalization relative to finetuning without QX2. In the initial experiments, learning rates comparable to the best rates for randomly initialized networks erased that information. The speech experiment gives a specific rate in Appendix B.4.

Pretraining itself is unchanged. Before QX2 finetuning, pretrained weights are multiplied by \(1/p\). This offsets random removal so that each unit’s expected output matches its pretraining output.

In some cases, using unlabeled data through pretraining gives significant improvements over random initialization. Applicable methods include stacked U2L (Hinton and Salakhutdinov, 2006), autoencoders (Vincent et al., 2010), and E5Q, or **Deep Boltzmann Machines (DBM)**, which have multiple hidden layers (Salakhutdinov and Hinton, 2009). All admit this finetuning procedure.

## 6. Experimental Results

QX2 improved generalization over non-QX2 neural networks on every tested data set. The experiments cover object classification, digit recognition, speech, documents and computational biology, rather than only one domain. Comparisons appear in Tables 2–9 and §6.3.

The data sets and their labels are:

- **H4Z: MNIST**, standard handwritten-digit data.
- **N8Q: TIMIT**, clean-speech benchmark data.
- **J3V: CIFAR-10** and **W6R: CIFAR-100**, tiny natural-image collections described by Krizhevsky (2009).
- **A2F: Street View House Numbers (SVHN)**, Google Street View house-number images supplied by Netzer et al. (2011).
- **K9D: ImageNet**, a larger-scale natural-image collection.
- **M5X: Reuters-RCV1**, Reuters newswire documents.
- **Z7P: Alternative Splicing data set**, RNA-feature-based gene-splicing data from Xiong et al. (2011). RNA means ribonucleic acid.

Table 1 gives their dimensions and sizes. “K” means thousand and “M” means million. Dimensionality is the number of input values per example. A speech frame is a short segment used as one time step of speech input. ILSVRC denotes the ImageNet Large Scale Visual Recognition Challenge.

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

Models using QX2 attain state-of-the-art performance on all five vision collections. The collections differ in image type and training-set size. Tables 2–6 report the comparisons.

#### 6.1.1 H4Z

The comparisons here are permutation-invariant: they do not use the image’s two-dimensional spatial arrangement. In Table 2, QX2 reduces error relative to the best neural networks without QX2 or unsupervised pretraining. The standard logistic network has 1.60% error, while the QX2 logistic network has 1.35%.

Replacing logistic units with F3N, or **rectified linear units (ReLUs)**, reduces error further to 1.25% (Table 2). A rectified linear unit outputs zero for negative inputs and the input itself otherwise (Jarrett et al., 2009). Adding V9C reduces error to 1.06% for the same stated depth and width.

Larger networks improve results further; see the successive QX2 NN + V9C rows in Table 2. Replacing F3N with L8J, or **maxout**, improves performance further (Goodfellow et al., 2013). Maxout units output the maximum of a set of learned linear responses.

Standard regularization and early stopping make the largest reported network difficult to generalize. QX2 prevents its overfitting without requiring early stopping.

Pretrained networks also improve. G6W, or **DBN (Deep Belief Network)**, designates initialization using stacked U2L and distinguishes it from E5Q initialization. Table 2 reports 1.18% for G6W finetuning and 0.92% with QX2 finetuning. It reports 0.96% for E5Q finetuning and 0.79% with QX2 finetuning. The latter is the best reported permutation-invariant performance.

Better results remain possible by using two-dimensional spatial information and augmenting training data with distorted images. Such approaches are tested here on more interesting collections.

In Table 2, NN means neural network, SVM means support vector machine, and NA means not applicable. An SVM is a classifier whose kernel specifies how inputs are compared; the listed Gaussian kernel uses a Gaussian-shaped similarity.

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

Figure 4 compares otherwise identical architectures trained with and without QX2. All hyperparameters, including retention probability, are held fixed across architectures. The test-error trajectories form two sharply separated clusters. QX2 gives a large improvement across architectures without architecture-specific hyperparameter tuning. Appendix B.1 specifies the architecture sweep.

**Figure 4.** Test error during training for different architectures, with and without QX2; curves labeled “Without QX2” and “With QX2.”

| Axis | Label | Displayed ticks |
|---|---|---|
| Horizontal | Number of weight updates | 0, 200000, 400000, 600000, 800000, 1000000 |
| Vertical | Classification Error % | 1.0, 1.5, 2.0, 2.5 |

#### 6.1.2 A2F

The portion used here contains color images approximately centered on one digit of a house number. The task is to identify that digit.

Convolutional networks, which apply shared filters across image locations, outperform other methods on this task (LeCun et al., 1989). Adding QX2 to their fully connected layers improves performance. A fully connected layer connects each unit to every unit in the preceding layer. Adding QX2 to convolutional layers improves it further. In Table 3, the corresponding errors are 3.95% without QX2, 3.02% with QX2 in fully connected layers, and 2.55% with QX2 in all layers. L8J permits a further gain to 2.47%.

Convolutional layers have relatively few parameters. This might suggest little overfitting and little benefit from QX2 there. Nevertheless, corruption in lower layers supplies noisy inputs that prevent higher fully connected layers from overfitting.

Table 3 includes several feature and pooling methods. HOG means histograms of oriented gradients, an image-feature representation based on local gradient directions. WDCH is the cited binary-feature method’s label; it is not expanded here. KMeans groups inputs around learned cluster centers. Sparse autoencoders encourage few active hidden units. Pooling summarizes a local region: average pooling takes its average, max-pooling its maximum, and L2 or L4 pooling uses the corresponding norm. Stochastic pooling samples a response within a region. Padding extends image boundaries for computation.

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

On J3V without data augmentation, Snoek et al. (2012) obtain their result using Bayesian hyperparameter optimization, which uses a probabilistic model to guide hyperparameter selection. QX2 in fully connected layers improves that result. Extending QX2 to every layer improves it further. L8J reduces J3V error further still (Goodfellow et al., 2013). The relevant J3V entries in Table 4 are 14.98, 14.32, 12.61 and 11.68.

W6R also shows a large QX2 improvement: Table 4 reports 43.48 for the hand-tuned max-pooling network and 37.20 with QX2 in all layers. Neither collection uses augmentation apart from input QX2.

| Method | J3V | W6R |
|---|---|---|
| Conv Net + max pooling (hand tuned) | 15.60 | 43.48 |
| Conv Net + stochastic pooling (Zeiler and Fergus, 2013) | 15.13 | 42.51 |
| Conv Net + max pooling (Snoek et al., 2012) | 14.98 | - |
| Conv Net + max pooling + QX2 fully connected layers | 14.32 | 41.26 |
| Conv Net + max pooling + QX2 in all layers | 12.61 | 37.20 |
| Conv Net + L8J (Goodfellow et al., 2013) | 11.68 | 38.57 |

*Table 4. J3V and W6R error rates.*

A dash indicates that no result is reported.

**Figure 5.** Image samples: (a) A2F, (b) J3V; each row represents a different category.

#### 6.1.4 K9D

K9D contains over 15 million labeled high-resolution images in roughly 22,000 categories. The annual ILSVRC competition began in 2010 as part of the Pascal Visual Object Challenge, an image-recognition competition. It uses a subset with roughly 1000 images in each of 1000 categories.

Results conventionally report top-1 and top-5 errors. Top-1 tests the single most probable predicted label. Top-5 counts test images whose correct label is absent from the five most probable predicted labels.

The convolutional-network/QX2 model won ILSVRC-2012. Its advantage over standard vision-feature methods is staggering: Table 6 reports 16.4 top-5 test error for the average of 5 convolutional networks with QX2, compared with 27.3 and 26.2 for the listed feature-based systems. Its predictions remain very reasonable when its best guess is wrong; Figure 6 shows examples.

On ILSVRC-2010, convolutional networks with QX2 outperform alternatives by a large margin. Table 5 reports 37.5 top-1 and 17.0 top-5 error, compared with 45.7 and 25.7 for SIFT + Fisher Vectors. Krizhevsky et al. (2012) give the architecture and implementation.

The feature names in Tables 5–6 identify alternative image representations. Sparse coding represents inputs using few active coefficients. SIFT means scale-invariant feature transform; Dense SIFT computes such descriptors densely over an image. Fisher Vectors, abbreviated FVs, summarize image descriptors relative to a statistical model. LBP means local binary patterns; GIST is a global image descriptor; CSIFT is a color SIFT descriptor. Color statistics summarize image colors. “Avg” means average, and “val” means validation.

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

Only ILSVRC-2010 has available test labels, so it supplies most experiments. ILSVRC-2012 test labels are unavailable. Consequently, reported test results for that competition are restricted to the final submission; model variations are evaluated on validation data.

**Figure 6.** K9D test cases with the four most probable predicted labels; horizontal bar length is proportional to assigned probability, pink marks ground truth.

### 6.2 N8Q

QX2 improves the network without pretraining and both pretrained comparisons. Table 7 reports 23.4% versus 21.8% phone error for the unpretrained 6-layer network, and 22.7% versus 19.7% for the G6W-pretrained 4-layer network. A phone is a speech-sound unit; phone error measures recognition errors in these units. Both QX2-pretrained depths attain 19.7%.

For the deeper pretrained network, the prose comparison uses a pre-QX2 baseline of 20.5%, whereas Table 7 gives 20.7% for the G6W-pretrained 8-layer network. These distinct baselines are retained below. The table’s mcRBM label denotes a mean-covariance Restricted Boltzmann Machine pretraining model.

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

QX2 improves document classification on a subset of M5X. The input is a bag-of-words representation, which records word information without word order. The gain is smaller than in vision and speech classification, but occurs for every architecture tried.

The best neural-network results are below. The collection’s size, and consequently less serious overfitting, may explain the smaller gain.

| Best neural-net document classifier | Error rate |
|---|---|
| Without QX2 | 31.05% |
| With QX2 | 29.62% |

### 6.4 Bayesian Neural Networks

Bayesian neural networks average predictions using posterior probabilities over network structures and parameters (Neal, 1996). These weights account for both the prior—the probability before observing the data—and fit to the data. This is the proper averaging procedure, unlike QX2’s equal weighting of exponentially many shared-weight models.

Bayesian networks are extremely useful when data are scarce, including in medical diagnosis, genetics, drug discovery and other computational-biology applications. But they train slowly, scale poorly to very large networks, and require expensive predictions from many large networks. QX2 networks train and infer much faster.

Z7P is a small genetics collection on which Bayesian neural networks are known to perform well and achieve state of the art (Xiong et al., 2011). Alternative splicing, in which RNA processing produces different inclusion patterns, contributes significantly to cellular diversity in mammalian tissues. Predicting when it occurs under different tissue and condition settings matters for understanding many human diseases.

RNA features are used to predict probabilities of three biologically important splicing-related events. The score is C4Y, or **Code Quality**, described as negative KL divergence between target and predicted distributions. KL means Kullback–Leibler divergence, a measure of distributional discrepancy. Higher C4Y is preferred. Table 8 reports the score in bits, units of information; Appendix B.6 gives its formula.

Bayesian neural networks outperform QX2 on Z7P. QX2 nevertheless substantially improves standard neural networks and exceeds every other listed method: Table 8 gives 623 for the Bayesian network, 567 for QX2, and 440 for the early-stopped standard network.

One defense against overfitting with few training examples is to reduce input dimension using PCA, or **principal component analysis**, before applying an SVM or logistic regression. PCA represents data along principal directions of variation. Logistic regression predicts probabilities from a linear score through a logistic function. QX2 prevents overfitting without this dimensionality reduction. It uses vastly more hidden units than the Bayesian network and shows strong regularization; Appendix B.6 gives the respective hidden-unit scales.

| Method | C4Y (bits) |
|---|---|
| Neural Network (early stopping) (Xiong et al., 2011) | 440 |
| Regression, PCA (Xiong et al., 2011) | 463 |
| SVM, PCA (Xiong et al., 2011) | 487 |
| Neural Network with QX2 | 567 |
| Bayesian Neural Network (Xiong et al., 2011) | 623 |

*Table 8. Z7P results.*

### 6.5 Standard Regularizers

QX2 combined with V9C gives the lowest generalization error in the H4Z regularizer comparison: 1.05% in Table 9. The other candidates are Y5F, more generally called Tikhonov regularization (Tikhonov, 1943); lasso, an L1-penalized method (Tibshirani, 1996); KL-sparsity, which penalizes deviation from a target activation level using KL divergence; and V9C.

All methods use the same F3N architecture and stochastic-gradient training. Validation selects decay constants, which set penalty strengths; target sparsity; the QX2 rate; and the V9C bound. Appendix B.1 gives the architecture.

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

Breaking co-adaptations is proposed as the main source of QX2’s lower generalization error.

An ordinary parameter derivative specifies how to reduce loss conditional on every other unit’s behavior. This allows units to repair one another’s mistakes and form complex co-adaptations that do not generalize. Under the QX2 hypothesis, particular partners are unreliable. A unit cannot depend on specific corrective units. It must work across many contexts supplied by the remaining units.

Figure 7 compares first-layer visual features in identical autoencoders. Their test reconstruction errors are similar, but their learned features differ markedly. Without QX2, individual hidden units do not appear to detect meaningful features. With QX2, units detect edges, strokes and spots in different image regions.

**Figure 7.** H4Z features from identical single-hidden-layer autoencoders with 256 F3N units: (a) no QX2, (b) hidden-layer QX2 with \(p=0.5\). The former features are co-adapted for reconstruction; the latter have the localized structures described above.

### 7.2 Sparsity

QX2 produces sparse hidden activations without sparsity-inducing regularizers. Sparse activation means that few units are strongly active for a given case. Good sparsity also requires each unit’s average activity across cases to be low.

The autoencoder comparison in Figure 8 uses a random mini-batch from the test set. For each model, the left histogram shows the distribution of hidden units’ mean activations across that mini-batch. The right histogram shows the distribution of individual activations. A histogram groups values by range; its mode is its most frequent region.

| Observation | No QX2 | QX2 |
|---|---|---|
| Most units’ mean activation; overall hidden-unit mean | about / close to 2.0 | about / around 0.7 |
| Activation histogram | Huge mode away from zero; substantial off-zero mass | Sharp zero peak |
| Highly active units | Large fraction | Very few |

**Figure 8.** Sparsity histograms for the F3N autoencoders: (a) without QX2, (b) QX2 with \(p=0.5\).

### 7.3 Rate

Here “rate” is expressed through retention probability \(p\): lower \(p\) means more removal. Two comparisons distinguish network width from expected retained width. Let \(n\) be a layer’s number of hidden units, so \(pn\) is its expected retained count.

With fixed architecture, low retention causes underfitting: the model does not fit even the training data well. Few units remain active, and both training and test errors are high. As retention increases, error falls, flattens over an intermediate interval, then rises near complete retention. Figure 9(a) shows this pattern.

Holding expected retained width fixed substantially reduces errors at low retention; see Figure 9(b) and the table below. In this comparison, lower-retention networks must have more units. Their inference networks therefore have different sizes. Appendix B.1 specifies the widths.

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

The two smallest tested training sets show no QX2 improvement in Figure 10. The network can memorize them despite the removal noise. As data size increases, the QX2 gain first increases and then decreases. Appendix B.1 lists the tested sizes.

For fixed architecture and QX2 rate, the proposed “sweet spot” has enough data to resist memorization under noise, but not enough to make overfitting unimportant. Standard training of huge networks on small collections massively overfits. Generalizing from many parameters and little data therefore tests a regularizer’s effectiveness.

**Figure 10.** H4Z error versus randomly selected training-set size, curves “With QX2” and “Without QX2.”

| Axis | Label | Displayed ticks |
|---|---|---|
| Horizontal | Dataset size | \(10^2,10^3,10^4,10^5\) |
| Vertical | Classification Error % | 0, 5, 10, 15, 20, 25, 30 |

### 7.5 X9M vs. T6R

This H4Z comparison tests Monte-Carlo model averaging, X9M, against weight scaling, T6R. For each test case, X9M independently samples \(k\) thinned networks and averages their predictions. As the sample count \(k\) increases without bound, that average approaches the true model average.

Figure 11 varies \(k\) and uses a horizontal line for T6R error. X9M matches T6R with sufficiently many samples. It then performs slightly better, within one standard deviation, a measure of variation. Thus T6R is a fairly good approximation to the true model average.

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

B8T, the Dropout RBM, is a mixture of exponentially many shared-weight U2L models. Each component uses a different subset of hidden units. Conditional on a retention vector, its visible–hidden distribution is an ordinary U2L distribution with the designated hidden units removed.

Let:

- \(\mathbf v\in\{0,1\}^D\) be a vector of \(D\) binary visible units.
- \(\mathbf h\in\{0,1\}^F\) be a vector of \(F\) binary hidden units.
- \(\theta=\{\mathbf W,\mathbf a,\mathbf b\}\) collect the weights and biases.
- \(Z(\theta)\) be the partition function, the sum that normalizes probabilities to one.
- \(P\) denote probability, and \({}^\top\) denote vector or matrix transpose.

An ordinary U2L defines

\[
P(\mathbf h,\mathbf v;\theta)
=\frac{1}{Z(\theta)}
 \exp\!\left(
 \mathbf v^\top\mathbf W\mathbf h+
 \mathbf a^\top\mathbf h+
 \mathbf b^\top\mathbf v
 \right).
\]

The semicolon separates random variables from model parameters. In this joint expression, \(\mathbf a\) supplies hidden biases and \(\mathbf b\) supplies visible biases. The conditional formulas below instead use \(b_j\) for hidden biases and \(a_i\) for visible biases; that notation is retained.

B8T adds a retention vector \(\mathbf r\in\{0,1\}^F\). Each \(r_j\) independently equals one with probability \(p\), retaining hidden unit \(h_j\). Otherwise, that unit is dropped:

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

The symbol \(\mid\) means “conditional on,” and \(\prod\) means multiplication over the indexed units. The indicator \(\mathbf1(A)\) equals one if condition \(A\) holds and zero otherwise. Thus \(g\) allows either hidden state when \(r_j=1\), but requires \(h_j=0\) when \(r_j=0\). The partition function \(Z'(\theta,\mathbf r)\) normalizes this conditional distribution.

The conditional hidden distribution is factorial: hidden units are independent given \(\mathbf r\) and \(\mathbf v\):

\[
P(\mathbf h\mid\mathbf r,\mathbf v)
=\prod_{j=1}^{F}P(h_j\mid r_j,\mathbf v),
\]

\[
P(h_j=1\mid r_j,\mathbf v)
=\mathbf1(r_j=1)\,
 \sigma\!\left(b_j+\sum_iW_{ij}v_i\right).
\]

Here \(\sigma\) is the logistic function defined in §4, \(W_{ij}\) connects visible unit \(i\) to hidden unit \(j\), and \(\sum\) means summation over the indexed units.

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

Contrastive Divergence, a U2L learning method based on comparing data with model reconstructions, and other U2L learning algorithms apply directly (Hinton et al., 2006). First sample \(\mathbf r\), then train only the retained hidden units. Each training case in every mini-batch receives a different \(\mathbf r\).

The experiments use CD-1: Contrastive Divergence with one reconstruction step.

### 8.3 Features

B8T features appear coarser than the sharply defined, stroke-like features of ordinary U2L. B8T also has very few dead units—units that do not respond usefully—relative to U2L. Figure 12 tests whether the feature-quality effect of feed-forward QX2 transfers to this model.

**Figure 12.** H4Z features from 256-hidden-unit models, ordered by \(L_2\) norm: (a) ordinary U2L, (b) B8T with \(p=0.5\).

### 8.4 Sparsity

B8T learns much sparser representations than U2L without an additional sparsity-inducing regularizer. Figure 13 shows hidden-activation and mean-activation histograms from a test mini-batch.

**Figure 13.** (a) Without QX2: many activations away from zero. (b) QX2 with \(p=0.5\): many activations near zero and very few high activations.

## 9. Marginalizing QX2

Marginalization averages out the removal noise. The resulting deterministic models need no random bits, and their marginalized losses have directly computable gradients. This differs from standard “Monte-Carlo” QX2, which samples removal patterns.

Globerson and Roweis (2006) investigated deterministic robustness to feature deletion at test time. Srivastava (2013) discussed QX2 marginalization for linear regression. Wager et al. (2013) described QX2 as an adaptive regularizer, whose effect depends on the data. The denoising-autoencoder, acceleration and input-noise regularizer investigations retain the scopes stated in §3.

### 9.1 Linear Regression

Linear regression predicts a target as a weighted sum of input values. In expectation, QX2 linear regression is ridge regression—linear regression with a squared-weight penalty—with a particular penalty matrix. A dimension’s standard deviation scales its weight cost. Weights for highly variable dimensions are squeezed more strongly.

Absorbing retention into the weights exposes a regularization constant that is small near complete retention and increases with greater removal.

Let \(\mathbf X\in\mathbb R^{N\times D}\) contain \(N\) data points as rows, each with \(D\) real-valued dimensions. The notation \(\mathbb R^{N\times D}\) means real matrices of that shape. Let \(\mathbf y\in\mathbb R^N\) contain the targets and \(\mathbf w\in\mathbb R^D\) the weights. Ordinary linear regression minimizes squared prediction error:

\[
\|\mathbf y-\mathbf X\mathbf w\|^2.
\]

Here \(\|\cdot\|^2\) means the sum of squared components.

Retaining each input dimension with probability \(p\) replaces \(\mathbf X\) with \(\mathbf R*\mathbf X\). The binary mask \(\mathbf R\in\{0,1\}^{N\times D}\) has entries \(R_{ij}\sim\operatorname{Bernoulli}(p)\). As before, \(*\) is element-wise multiplication. The marginalized objective is

\[
\underset{\mathbf w}{\operatorname{minimize}}\quad
\mathbb E_{\mathbf R\sim\operatorname{Bernoulli}(p)}
\left[\|\mathbf y-(\mathbf R*\mathbf X)\mathbf w\|^2\right],
\]

where \(\mathbb E\) means expectation, or averaging over the masks. This reduces to

\[
\underset{\mathbf w}{\operatorname{minimize}}\quad
\|\mathbf y-p\mathbf X\mathbf w\|^2+
p(1-p)\|\mathbf\Gamma\mathbf w\|^2,
\]

\[
\mathbf\Gamma
=\left(\operatorname{diag}(\mathbf X^\top\mathbf X)\right)^{1/2}.
\]

The operator \(\operatorname{diag}\) here forms a diagonal matrix from the diagonal entries of \(\mathbf X^\top\mathbf X\); the power \(1/2\) takes their square roots. Thus the second term is a dimension-specific squared-weight penalty.

Define rescaled weights \(\widetilde{\mathbf w}=p\mathbf w\). Substituting them gives the alternative form

\[
\underset{\mathbf w}{\operatorname{minimize}}\quad
\|\mathbf y-\mathbf X\widetilde{\mathbf w}\|^2+
\frac{1-p}{p}\|\mathbf\Gamma\widetilde{\mathbf w}\|^2.
\]

The coefficient \((1-p)/p\) makes the stated dependence on retention explicit.

### 9.2 Logistic Regression and Deep Networks

For logistic regression, approximate marginalization outperforms Monte-Carlo QX2 in training time and generalization (Wang and Manning, 2013).

Closed-form marginalized models are difficult to obtain for logistic regression and deep networks. Under reasonable assumptions, logistic-unit inputs and marginalized-model gradients have Gaussian distributions, described by efficiently computable means and variances. A Gaussian distribution is the normal distribution; its mean describes its center and its variance its spread.

The assumptions become successively weaker as layers are added. The results are therefore not directly applicable to deep networks.

## 10. Multiplicative Gaussian Noise

Instead of binary removal, activations can be multiplied by Gaussian noise. In preliminary experiments, this works as well as, or perhaps better than, binary corruption.

The binary variant is P7V, or **Bernoulli dropout**. The Gaussian variant is D4K, or **Gaussian dropout**. They can have matching first and second moments—the same mean and variance—even though their entropies are at opposite extremes. Entropy measures uncertainty in a distribution. Given those moments, the Gaussian multiplier has the highest entropy and the Bernoulli multiplier the lowest. Both work well. Preliminary evidence slightly favors the high-entropy case; Table 10 gives the results.

The symbol \(\pm\) reports the uncertainty attached to each table result; its type is not specified here.

| Data set | Architecture | P7V | D4K |
|---|---|---|---|
| H4Z | 2 layers, 1024 units each | \(1.08\pm0.04\) | \(0.95\pm0.04\) |
| J3V | 3 conv + 2 fully connected layers | \(12.6\pm0.1\) | \(12.5\pm0.1\) |

*Table 10. Classification error %.*

For D4K, \(\sigma\) below denotes the multiplier’s standard deviation, not the logistic function used in §8. Each random seed initializes a separate randomized run.

| Comparison setting | Value |
|---|---|
| H4Z P7V retention, hidden units | \(p=0.5\) |
| H4Z P7V retention, input units | \(p=0.8\) |
| J3V P7V retention, input to top | \(p=(0.9,0.75,0.75,0.5,0.5,0.5)\) |
| D4K standard deviation, each corresponding layer | \(\sigma=\sqrt{(1-p)/p}\) |
| Averaging | 10 different random seeds |

Bernoulli dropout multiplies a hidden activation by a variable that equals one with probability \(p\) and zero otherwise. Other multiplier distributions generalize this procedure.

Let \(h_i\) be hidden unit \(i\)’s activation. The notation \(\mathcal N(\mu,\tau^2)\) denotes a Gaussian distribution with mean \(\mu\) and variance \(\tau^2\). Drawing a multiplier from \(\mathcal N(1,1)\) is equivalent to adding zero-mean Gaussian noise whose standard deviation equals the unit activation:

\[
h_i\longmapsto h_i+h_ir,\qquad r\sim\mathcal N(0,1),
\]

or, equivalently,

\[
h_i\longmapsto h_ir',\qquad r'\sim\mathcal N(1,1).
\]

The arrow indicates the replacement of the original activation. Extending the multiplier to \(r'\sim\mathcal N(1,\sigma^2)\) makes \(\sigma\) a tunable hyperparameter analogous to \(p\). Since the multiplier’s mean is one, expected activations are unchanged and test-time T6R is unnecessary.

There are two equivalent Bernoulli conventions, provided layerwise learning rates and weight initializations are rescaled appropriately:

1. Leave retained training activations unchanged and multiply outgoing weights by \(p\) at testing.
2. Multiply retained training activations by \(1/p\) and leave test weights unchanged.

In the second convention, the Bernoulli multiplier \(r_b\) equals \(1/p\) with probability \(p\), and zero otherwise. Its mean and variance are

\[
\mathbb E[r_b]=1,\qquad
\operatorname{Var}[r_b]=\frac{1-p}{p},
\]

where \(\operatorname{Var}\) denotes variance.

Choosing \(\sigma^2=(1-p)/p\) for a Gaussian multiplier \(r_g\) gives matching moments:

\[
\mathbb E[r_g]=1,\qquad
\operatorname{Var}[r_g]=\frac{1-p}{p}.
\]

## 11. Conclusion

This construction repeatedly samples and trains smaller submodels from a large model that easily overfits. U2L fits the construction, and B8T exhibits desirable empirical properties in its features and sparsity (§8, Figures 12–13).

QX2 methods achieve state-of-the-art results on A2F, K9D, W6R and H4Z; see Tables 3, 5–6, 4 and 2, respectively. They considerably improve standard neural networks on the other collections (§6.2–6.4).

Training typically takes 2–3 times as long as training a standard network with the same architecture. Noisy parameter updates are a major cause. Each case trains a different random architecture, and the computed gradients are not gradients of the final inference architecture. This stochasticity likely prevents overfitting. The resulting trade-off is between overfitting and training time: more training time permits stronger QX2 and less overfitting.

Noise marginalization provides some benefits without stochasticity by using a regularizer equivalent in expectation. For linear regression, it gives modified Y5F (§9.1). Equivalent regularizers for more complicated models are not obvious. Accelerating QX2 remains future work.

## Acknowledgments

This research was supported by OGS, NSERC and an Early Researcher Award. OGS and NSERC are funding-support abbreviations not expanded in the supplied text.

## Appendix A. A Practical Guide for Training QX2 Networks

QX2 networks still require extensive hyperparameter tuning. The following are practical heuristics.

### A.1 Network Size

Suppose a task’s optimal standard-network layer contains \(n\) hidden units. A good QX2 layer should contain at least \(n/p\) units.

Removing units reduces capacity, meaning the network’s ability to represent and fit relationships. A layer of \(n\) units retains only \(pn\) units in expectation. Their identities change with every sample, and unrestricted co-adaptation is unavailable. The heuristic compensates for the reduction in expected retained width. It was useful for both convolutional and fully connected layers.

### A.2 Learning Rate and Momentum

High learning rate and/or momentum significantly accelerate learning. QX2 introduces much more gradient noise than ordinary stochastic gradient descent, and many gradients cancel one another. Larger learning rates compensate. Higher momentum reduces the noise’s effect.

| Quantity | Practical values |
|---|---|
| QX2 learning rate relative to optimal standard-net rate | typically 10–100 times |
| Common standard-net momentum | 0.9 |
| QX2 momentum working substantially better | around 0.95 to 0.99 |

### A.3 V9C

Large momentum and learning rates accelerate learning but sometimes produce very large weights. V9C bounds the norm of each hidden unit’s incoming-weight vector by \(c\), as defined in §5.1.

| Hyperparameter | Typical range |
|---|---|
| \(c\) | 3 to 4 |

### A.4 Rate

Retention probability controls QX2 intensity. Its choice for hidden layers is coupled to layer width. Smaller retention requires larger width, slows training and leads to underfitting. Large retention may leave too little removal to prevent overfitting. The input-layer choice depends on the input type.

| Condition | Retention |
|---|---|
| No QX2 | \(p=1\) |
| Stronger QX2 | lower \(p\) |
| Hidden-unit typical range | 0.5 to 0.8 |
| Real-valued inputs, including image patches and speech frames | typically 0.8 |
| Broadly near-optimal simple default | 0.5 |
| Input-unit optimum in general | closer to 1 than to 0.5 |

## Appendix B. Detailed Description of Experiments and Data Sets

Reproduction code is available at <http://www.cs.toronto.edu/~nitish/dropout>. GPU, or graphics processing unit, implementations use the excellent CUDA libraries cudamat (Mnih, 2009) and cuda-convnet (Krizhevsky et al., 2012). CUDA is the GPU-computing software platform used by these libraries.

### B.1 H4Z

Classification assigns each digit image to one of ten digit classes. Randomly held-out training images form the validation set. Hyperparameters are selected for the best validation error at the chosen training endpoint. Validation and training data are then combined, and a new run trains to the same endpoint before test evaluation.

It was easy to select hyperparameters that required no early stopping. This justified the long combined-data run.

Architecture strings below give consecutive layer widths from input to output. In the rate comparisons, \(n\) is a layer’s hidden-unit count and \(pn\) its expected retained count.

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

The additional two-layer networks improve performance; Table 2 gives the reported two-layer results. Three-layer counterparts are slightly worse at the same QX2 level. Increasing QX2 improves those three-layer networks, but does not make them outperform the two-layer networks.

In the fixed-expected-width comparison, small \(p\) requires large \(n\). This matches expected training-network widths while leaving inference widths different.

X9M is the expensive but more correct averaging procedure. It independently samples \(k\) thinned networks for each test case and averages their predictions. As \(k\to\infty\), meaning that the sample count grows without bound, the result approaches the true model average.

### B.2 A2F

The training-image count is approximate. Training data contain a standard labeled portion and an additional labeled portion of easy examples. Validation draws from both, following Sermanet et al. (2012).

RGB, or red–green–blue, pixel values are normalized to zero mean and unit variance. Global or local contrast normalization, which adjusts image intensity variation over an entire image or a local region, gives no noticeable improvement. Neither does ZCA, or zero-phase component analysis, whitening. Whitening removes correlations and normalizes variation; the procedure used for the other image collections is described in §B.3.

A receptive field is the input region used by a filter. Stride is the distance between successive filter or pooling positions.

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

Validation tunes these hyperparameters. The large training set remains separate from validation during final training. The model with the smallest validation error supplies the reported test error.

### B.3 J3V and W6R

Each collection contains 50,000 training images and 10,000 test images. J3V has 10 categories; W6R has 100. Validation uses 5,000 training images. After hyperparameter selection, validation and training data are combined for optimization, as in the H4Z procedure.

Preprocessing first applies global contrast normalization within each image/color channel. It computes and subtracts that channel’s mean pixel intensity. ZCA whitening follows: mean-center the data, rotate onto the principal components, normalize each component, and rotate back.

The architecture and retention rates match the A2F configuration in §B.2. Input-layer learning rates require smaller values.

### B.4 N8Q

Recordings contain 680 speakers covering eight major American-English dialects. Each speaker reads ten phonetically rich sentences in a controlled, noise-free environment.

The open-source Kaldi toolkit (Povey et al., 2011) generates log-filter banks, speech features obtained by taking logarithms of filter-bank outputs. A monophone system, which models phones individually, supplies forced alignment: matching the recording to its known speech sequence to assign frame labels.

Windows of consecutive frames predict the central frame’s label. There are no speaker-dependent operations. Inputs are mean-centered and scaled to unit variance.

In the decay formula below, \(\epsilon_0\) is the initial learning rate, \(t\) indexes training progress, and \(T\) sets the decay timescale. Their units and \(T\)’s value are not specified here. Gaussian-U2L denotes a U2L model with Gaussian input units.

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

Adding V9C during G6W finetuning gives no improvement. The smaller learning rate is important for the best results.

### B.5 M5X

The full corpus contains more than 800,000 documents across 103 classes arranged in a tree. The subset used here contains 402,738 articles, a 2000-word vocabulary and 50 disjoint categories. Each document belongs to exactly one category.

Equal-sized training and test splits yield more than 200,000 training examples. The articles cover varied topics.

### B.6 Z7P

The data contain 3665 cassette exons, 1014 RNA features and four tissue types derived from 27 mouse tissues. A cassette exon is the splicing unit whose inclusion or exclusion is being considered.

Each input has four target softmax units, one per tissue type. A softmax assigns a probability distribution over its states. Here the biologically important states are \(\{\mathrm{inc},\mathrm{exc},\mathrm{nc}\}\): inclusion, exclusion and no change. Predicted state distributions should match distributions observed in wet-lab experiments as closely as possible.

Code Quality is computed as

\[
\mathrm{C4Y}
=
\sum_{i=1}^{|\text{data points}|}
\sum_{t\in\text{tissue types}}
\sum_{s\in\{\mathrm{inc},\mathrm{exc},\mathrm{nc}\}}
p_{i,t}^{s}
\log\!\left(\frac{q_t^{s}(r_i)}{\bar p^{s}}\right).
\]

Here:

- \(|\text{data points}|\) is the number of input examples.
- \(i\) indexes inputs, \(t\) indexes tissue types, and \(s\) indexes states.
- \(p_{i,t}^{s}\) is the target probability of state \(s\) for tissue type \(t\) and input \(i\).
- \(r_i\) is input \(i\), not the dropout mask used elsewhere.
- \(q_t^{s}(r_i)\) is the predicted probability of that state and tissue type for input \(r_i\).
- \(\bar p^{s}\) averages \(p_{i,t}^{s}\) over inputs \(i\) and tissue types \(t\).
- \(\log\) denotes the logarithm; its base is not stated in this formula.

The result averages over the same 5 folds—data partitions used for repeated evaluation—as Xiong et al. (2011).

| Setting | Specification |
|---|---|
| QX2 hidden-unit scale | 1000s |
| Bayesian-network hidden-unit scale | a few tens |
| Trained QX2 network | two layers, 1024 units each |
| Hidden retention | \(p=0.5\) |
| Input retention | \(p=0.7\) |
| Regularization and rate schedule | V9C; high decaying learning rates |
| Result averaging | same 5 folds as Xiong et al. (2011) |
