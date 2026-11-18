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

Deep neural networks are machine-learning models with multiple layers of learned units. Large networks have many adjustable parameters and are powerful. However, they can overfit: they can learn relationships that work on training data but not on new test data. Their slow prediction also makes it difficult to combine several independently trained large networks at test time.

Dropout, designated **QX2** throughout this paper, temporarily removes randomly selected units and their connections during training. It inhibits excessive co-adaptation, in which units depend on particular other units to correct their mistakes. Each removal pattern produces a smaller, or *thinned*, network. These networks share parameters, so training samples exponentially many networks without storing each separately. A single unthinned network with reduced weights approximates their averaged predictions.

Compared with other regularizers—methods that constrain learning to reduce overfitting—dropout substantially reduces overfitting and produces major improvements. The supervised experiments learn from supplied targets and cover vision, speech recognition, document classification and computational biology. They attain state-of-the-art performance on many benchmarks. The domain comparisons appear in Tables 2–8 and §6.3; Table 9 directly compares regularizers.

**Keywords:** neural networks, regularization, model combination, deep learning

## 1. Introduction

With unlimited computation, the best regularization for a fixed-sized model is to average over every parameter setting, weighted by its posterior probability. A posterior probability reflects both prior assumptions and the observed data. Good approximations are sometimes possible for simple or small models (Xiong et al., 2011; Salakhutdinov and Mnih, 2008).

The construction proposed here is less expensive. It approximates an equally weighted geometric mean of the predictions of exponentially many learned models. A geometric mean combines values by taking their product and then the corresponding root. The models share parameters.

Let \(n\) be the number of units in a network. Each unit can be retained or removed, giving \(2^n\) possible thinned networks. Weight sharing requires only \(O(n^2)\), or fewer, parameters. Here \(O(n^2)\) means that the parameter count grows no faster than a constant multiple of \(n^2\). A new thinned network is sampled for every presentation of every training case. Any particular member is therefore trained rarely, if ever.

A multilayer network uses hidden units—units between its input and output—to express complicated input–output relationships. Nonlinear hidden representations are not limited to linear combinations of the inputs. With limited data, sampling noise creates apparent relationships that occur in the training set but not in genuine test data. This can happen even when training and test data come from the same distribution.

Established responses include early stopping, which stops training when performance on held-out validation data deteriorates; soft weight sharing, which encourages weights to share values (Nowlan and Hinton, 1992); and weight penalties. Two familiar penalties are **O3U**, or **L1 regularization**, which penalizes weight magnitudes, and **Y5F**, or **L2 regularization**, which penalizes squared weight magnitudes. These methods constrain learned parameters rather than average over every possible parameter setting.

Model combination means combining predictions from several models. It nearly always improves machine-learning performance. However, separately training many large neural networks and averaging their outputs is prohibitively expensive. Useful diversity requires different architectures, meaning different arrangements of layers and units, or different training data. Finding optimal hyperparameters—training settings chosen rather than learned—for each architecture is difficult. Each large-network training run consumes substantial computation. Large data requirements can also rule out training different networks on different subsets. Even after successful training, evaluating all the models may be infeasible for applications requiring rapid responses.

The practical inference rule used here is **T6R**, or **weight scaling**. Inference means using a trained model to make predictions. T6R evaluates one unthinned network instead of exponentially many members. If a unit is retained during training with probability \(p\), its outgoing weights are multiplied by \(p\) for testing. Section 4 gives the full rule.

A more expensive alternative is **X9M**, or **Monte-Carlo model averaging**. It independently samples \(k\) thinned networks for each test case and averages their predictions; \(k\) is the sample count. As \(k\) tends to infinity, this approaches the true model average. Section 7.5 compares it with weight scaling.

Weight scaling gives significantly lower generalization error—the error on new data—than other regularization methods across varied classification problems. Table 9 gives the direct regularizer comparison, and §6 reports the cross-domain results.

The unit-removal construction also extends beyond feed-forward networks, which pass information from input toward output without feedback. It applies to graphical models, which express probability relationships among variables, including Boltzmann Machines. One such model family is **U2L**, or **Restricted Boltzmann Machines (RBM)**. Its dropout-augmented form is **B8T**, or **Dropout RBM**. Section 8 defines both distributions and shows how the augmented model retains shared parameters.

**Figure 1.** Standard neural net, left: two hidden layers. QX2-produced thinned net, right: crossed units removed, including their incoming and outgoing connections.

Paper map: §2 gives the motivation; §3 reviews previous work; §4 formally defines QX2; §5 gives the training algorithm; §6 reports cross-domain experiments and comparisons with regularization and model combination; §7 examines network properties and hyperparameter interactions; §8 describes B8T; §9 discusses QX2 marginalization, meaning averaging out the removal noise. Appendix A gives practical training guidance, including detailed hyperparameter-choice considerations.

## 2. Motivation

Sexual reproduction combines half of one parent’s genes with half of another’s, together with a very small random mutation. Asexual reproduction produces a slightly mutated copy of one parent’s genes. Direct transmission preserves a successful co-adapted gene set. This makes asexual superiority plausible if the objective is individual fitness.

Sexual recombination breaks such sets, especially large ones. It therefore seems disadvantageous for an organism that already has complicated co-adaptations. Nevertheless, most advanced organisms evolved through sexual reproduction.

Livnat et al. (2010) offer a possible account. Over the long run, selection may favor gene *mix-ability*—the ability to function with other randomly selected genes—rather than individual fitness alone. A gene set that works with another random gene set is more robust. Because a gene cannot count on a large set of particular partners, it benefits from being useful individually or in collaboration with only a few genes. Reducing complex co-adaptations also helps useful new genes spread: fewer dependencies block their contribution to individual fitness.

QX2-trained hidden units face a similar problem. Their partners are randomly selected. They must therefore learn robust, independently useful features rather than rely on particular hidden units to repair their mistakes. Units within a layer still learn different functions. Replicating each hidden unit many times is a poor solution to robustness, just as replica codes—codes based on repeating information—poorly address noisy communication channels.

Consider ten five-person conspiracies and one conspiracy that requires fifty participants to act correctly. The ten smaller conspiracies probably have an advantage in producing havoc. Under unchanged conditions and with ample rehearsal, the larger conspiracy may work. Under non-stationarity, where conditions change, the smaller conspiracies are more likely to persist. Likewise, a complex co-adaptation can succeed on training data but fail on novel test data more readily than several simpler co-adaptations producing the same result.

## 3. Related Work

QX2 can be viewed as regularization by noise in hidden-unit states. It extends **S2H**, or **Denoising Autoencoders (DAEs)**, introduced by Vincent et al. (2008, 2010). An autoencoder learns to reconstruct its input. A denoising autoencoder receives a corrupted input and learns to reconstruct the clean input.

QX2 extends this idea with effective corruption of hidden layers, a model-averaging interpretation, supervised as well as unsupervised use, and applicability to other neuron-based architectures such as Boltzmann Machines. Supervised learning uses supplied targets; unsupervised learning does not. Test-time T6R permits much higher noise than usually works best for S2H, as the following reported removal percentages show. The symbol % denotes percent.

| Setting | Noise level reported |
|---|---|
| S2H, typical best | 5% |
| QX2, often optimal input-unit removal | 20% |
| QX2, often optimal hidden-unit removal | 50% |

Stochastic minimization samples noise and minimizes the expected loss, meaning the loss averaged over the noise distribution. Globerson and Roweis (2006) and Dekel et al. (2010) instead consider adversarial removal. An adversary chooses which units to remove, subject to a fixed maximum count rather than a probability distribution. Those models have no hidden units.

Analytically marginalizing QX2 means calculating the average over removal patterns rather than sampling them. In simple cases, this gives a deterministic regularizer. Van der Maaten et al. (2013) investigated input-noise regularizers for exponential-family models, including QX2 under the name “blankout noise.” Their scope was limited to models without hidden layers. Chen et al. (2012) examined marginalization for denoising autoencoders. Wang and Manning (2013) proposed marginalization to accelerate QX2.

## 4. Model Description

In the simplest construction, retention choices are random, mutually independent and made with a common fixed probability \(p\). Validation can select \(p\). The usual default, \(p=0.5\), is close to optimal across many tasks and networks. Optimal input retention is generally nearer complete retention.

For every hidden unit, its expected output under the training deletion distribution equals its actual test-time output under the prescribed reduced weights. At test time, all units are present.

Let the network have \(L\) hidden layers, indexed by \(l\in\{1,\ldots,L\}\). Bold symbols denote vectors or matrices. Define:

- \(\mathbf x\): the input vector.
- \(\mathbf y^{(0)}=\mathbf x\): the outputs of the input layer.
- \(\mathbf z^{(l)}\): the weighted inputs to layer \(l\), before its activation function.
- \(\mathbf y^{(l)}\): the outputs of layer \(l\).
- \(\mathbf W^{(l)}\) and \(\mathbf b^{(l)}\): the layer’s weights and biases. A bias is an additive input.
- \(\mathbf w_i^{(l+1)}\): the incoming weight vector for unit \(i\) in layer \(l+1\).
- \(b_i^{(l+1)}\): that unit’s bias.
- \(f\): the activation function that converts a weighted input into a unit output.

For \(l\in\{0,\ldots,L-1\}\), ordinary feed-forward computation for hidden unit \(i\) is

\[
z_i^{(l+1)}
 =\mathbf w_i^{(l+1)}\mathbf y^l+b_i^{(l+1)},
\qquad
y_i^{(l+1)}=f\!\left(z_i^{(l+1)}\right).
\]

The vector product in the first expression sums weighted outputs from the preceding layer. Any activation \(f\) is admissible. One example is the logistic sigmoid,

\[
f(x)=\frac{1}{1+\exp(-x)},
\]

where \(\exp\) is the exponential function and \(x\) here is the function’s scalar argument.

For QX2, let \(j\) index units in layer \(l\). Each retention gate \(r_j^{(l)}\) follows a Bernoulli distribution: it equals one with probability \(p\) and zero otherwise. The notation \(\sim\) means “is sampled from.” Let \(\mathbf r^{(l)}\) collect these independent gates. Then

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

Here \(*\) means element-wise multiplication. Thus \(\widetilde{\mathbf y}^{(l)}\) contains the original outputs for retained units and zeros for removed units. These thinned outputs pass to the next layer.

Applying this operation at every layer samples a subnetwork. Backpropagation calculates loss derivatives backward through that subnetwork. Both hidden units and visible units—the input units—can be removed.

At inference, no units are removed. The test-time weights are

\[
\mathbf W_{\mathrm{test}}^{(l)}=p\mathbf W^{(l)}.
\]

**Figure 2.** Training, left: a unit is present with probability \(p\), its connections to the next layer having weights \(w\). Testing, right: the unit is always present, its outgoing weights \(pw\).

Here \(w\) denotes an individual connection weight.

**Figure 3.** Standard computation, left: \(y_1^{(l)},y_2^{(l)},y_3^{(l)}\), weights \(\mathbf w_i^{(l+1)}\), bias \(b_i^{(l+1)}\) from a \(+1\) unit, preactivation \(z_i^{(l+1)}\), activation \(f\), output \(y_i^{(l+1)}\). QX2 computation, right: corresponding gates \(r_1^{(l)},r_2^{(l)},r_3^{(l)}\) first produce \(\widetilde y_1^{(l)},\widetilde y_2^{(l)},\widetilde y_3^{(l)}\); the remaining operations retain their form.

The \(+1\) unit supplies the constant input used to add the bias.

## 5. Learning QX2 Nets

### 5.1 Backpropagation

Training uses stochastic gradient descent: parameters are updated using loss derivatives estimated from training cases. A mini-batch is a group of cases whose gradients are averaged for an update.

Compared with standard-network training, QX2 changes only the network used for each case. Every case in a mini-batch receives its own sampled thinned network. Forward computation and backpropagation use that network. Parameter gradients are then averaged across cases. A case that does not use a parameter contributes zero to its gradient.

Momentum, which carries part of previous updates into the current update, remains useful. So do annealed learning rates—step sizes reduced during training—and Y5F.

QX2 alone gives significant gains. Combining it with high momentum, large decaying learning rates and **V9C**, or **max-norm regularization**, gives a further significant boost. V9C limits each hidden unit’s incoming weight-vector norm. For example, Table 2 reports 1.25% error for the 3-layer, 1024-unit QX2 network with rectified units, and 1.06% after adding V9C. Table 9 gives the broader regularizer comparison.

For an incoming hidden-unit weight vector \(\mathbf w\), the constraint is

\[
\|\mathbf w\|_2\leq c.
\]

The Euclidean norm \(\|\mathbf w\|_2\) is the square root of the sum of squared weight components. The hyperparameter \(c\), selected by validation, is its maximum permitted value. If an update takes \(\mathbf w\) outside the radius-\(c\) ball, projection returns it to the surface.

V9C usually improves stochastic-gradient training of deep networks even without QX2. Table 9 includes a V9C-only result of 1.35%. The constraint was previously used for collaborative filtering, a prediction task based on shared patterns across users and items (Srebro and Shraibman, 2005).

A proposed optimization account is as follows. The constraint permits huge learning rates without weight explosion. QX2 noise explores regions of weight space that would otherwise be difficult to reach. As the learning rate decays, steps become shorter and less exploratory, ending at a minimum.

### 5.2 Unsupervised Pretraining

Pretraining learns an initial representation before supervised finetuning, which then adjusts the parameters by backpropagation for the target task. Using unlabeled data this way can significantly improve on random initialization in certain cases.

Pretraining can use stacked U2L (Hinton and Salakhutdinov, 2006), autoencoders (Vincent et al., 2010), or **E5Q**, the **Deep Boltzmann Machines (DBM)** model family described by Salakhutdinov and Hinton (2009). All admit QX2 finetuning. Tables 2 and 7 report pretrained-network comparisons.

Pretraining itself is unchanged. Before QX2 finetuning, multiply the pretrained weights by \(1/p\). This matches each unit’s expected output under random removal to its pretraining output.

Finetuning requires smaller learning rates. In the initial experiments, rates comparable to those best for randomly initialized networks erased useful pretrained information. Smaller rates preserved it and improved final generalization relative to finetuning without QX2. Appendix B.4 reports about 0.01 as the best QX2 finetuning rate for the pretrained speech model, compared with the listed high learning rate of 0.1.

## 6. Experimental Results

The experiments cover object classification, digit recognition, speech, documents and computational biology. Generalization improved over non-QX2 neural networks on every tested data set. Tables 2–8 and §6.3 report the comparisons.

The data-set codes are:

- **H4Z: MNIST**, standard handwritten-digit data. The task assigns an image to one of ten digit classes.
- **N8Q: TIMIT**, clean-speech benchmarking data.
- **J3V: CIFAR-10** and **W6R: CIFAR-100**, tiny natural-image collections with 10 and 100 categories, respectively (Krizhevsky, 2009).
- **A2F: Street View House Numbers data set (SVHN)**, Google Street View house-number images (Netzer et al., 2011).
- **K9D: ImageNet**, a large natural-image collection.
- **M5X: Reuters-RCV1**, Reuters newswire documents.
- **Z7P: Alternative Splicing data set**, gene-splicing prediction from RNA features (Xiong et al., 2011). RNA features are the biological input measurements used for this task; the paper does not further define them.

Table 1 gives the overview. “Dimensionality” is the number of input components. K means thousand, M means million, and “dim” abbreviates dimensional. Grayscale images have intensity values; color images have color components. ILSVRC-2012 names the 2012 image-recognition competition subset described in §6.1.4; the abbreviation is not expanded in the paper.

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

The five vision collections are H4Z, A2F, J3V, W6R and K9D. They differ in image type and training-set size. Models attaining state of the art on all five use QX2; Tables 2–6 report their results. Classification error percentages measure the proportion of incorrectly classified examples.

#### 6.1.1 H4Z

These comparisons are permutation-invariant: they do not use the images’ two-dimensional spatial arrangement. Validation used 10,000 randomly held-out training images. Hyperparameters were selected for performance at 1 million weight updates. Training and validation data were then combined, and a new run reached the same endpoint before test evaluation. Appendix B.1 gives the full settings.

Table 2 compares several unit types and initialization methods:

- Logistic units use the sigmoid activation in §4.
- **F3N** means **rectified linear units (ReLUs)**, the alternative activation class used here (Jarrett et al., 2009).
- **L8J** means **maxout**, another activation class (Goodfellow et al., 2013).
- **G6W**, labeled **DBN**, designates networks initialized with stacked U2L. The paper does not expand DBN. This designation distinguishes them from E5Q-initialized networks.
- NN abbreviates neural network. SVM names the comparison classifier, used here with a Gaussian kernel; the paper gives no further definition of that classifier or kernel. NA marks entries that do not apply.
- A sequence such as 500-500-2000 gives successive layer widths. The maxout row retains the reported grouping \(5\times240\), which is not further specified.

QX2 reduces error relative to the best permutation-invariant neural networks without QX2 or unsupervised pretraining. Table 2 reports 1.60% for the standard logistic network and 1.35% for the QX2 logistic network. Replacing logistic units with F3N reduces the QX2 result to 1.25%. Adding V9C reduces it to 1.06%.

Greater size improves results further: the listed F3N networks with V9C reach 1.04%, 1.01% and 0.95%. Maxout improves the listed result to 0.94%.

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

The largest reported two-layer network has more than 65 million parameters. Standard regularization and early stopping make it difficult to generalize. QX2 prevents its overfitting without requiring early stopping, using the selected training endpoint described above.

Pretrained networks also improve. In Table 2, G6W finetuning changes from 1.18% without QX2 to 0.92% with it. E5Q finetuning changes from 0.96% to 0.79%. The latter is the best reported permutation-invariant performance.

Better results remain possible by using two-dimensional spatial information and augmenting training with distorted images. These approaches are tested here on more interesting collections.

The architecture sweep used six combinations: 2, 3 or 4 hidden layers, with uniform widths of 1024 or 2048. Every hyperparameter, including retention probability, was held fixed across architectures. Hidden retention was \(p=0.5\), and input retention was \(p=0.8\). Figure 4 shows two sharply separated test-error trajectory clusters for otherwise identical architectures trained with and without QX2. The gain is large without architecture-specific hyperparameter tuning.

**Figure 4.** Test error during training for different architectures, with and without QX2; curves labeled “Without QX2” and “With QX2.”

| Axis | Label | Displayed ticks |
|---|---|---|
| Horizontal | Number of weight updates | 0, 200000, 400000, 600000, 800000, 1000000 |
| Vertical | Classification Error % | 1.0, 1.5, 2.0, 2.5 |

#### 6.1.2 A2F

The used images are color images approximately centered on one digit of a house number. The task is to identify that digit. Training uses a standard labeled portion and an additional labeled portion of easy examples. The training count is approximate. Validation contains 6000 examples drawn from both portions. Appendix B.2 gives the selection procedure and full architecture.

A convolutional network applies learned filters across image regions (LeCun et al., 1989). A fully connected layer connects its units to all outputs of the preceding layer. The model here has three convolutional layers followed by two fully connected hidden layers. All hidden units are F3N.

Pooling combines nearby responses. Table 3 names average, L2, L4, max and stochastic pooling variants; their detailed rules are not specified in the paper. Padding is another listed configuration choice. “Conv Net” abbreviates convolutional network. WDCH and HOG label comparison feature methods without further definition. KMeans is the named comparison method; stacked sparse autoencoders are stacked reconstruction models that encourage low activity.

Convolutional networks outperform other methods on this task. Adding QX2 to the fully connected layers improves the max-pooling result from 3.95% to 3.02%. Adding QX2 to all layers improves it to 2.55%. Maxout permits a further gain to 2.47%.

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

Convolutional layers have few parameters, which might suggest little overfitting and little benefit from QX2 there. Nevertheless, lower-layer corruption supplies noisy inputs to the higher fully connected layers and prevents those layers from overfitting. Table 3’s fully-connected-only and all-layer results are 3.02% and 2.55%, respectively.

#### 6.1.3 J3V and W6R

For these experiments, validation used 5,000 training images. After hyperparameter selection, validation and training data were combined, as for H4Z. Preprocessing and the architecture are detailed in Appendix B.3. The architecture and retention rates match A2F, but input-layer learning rates must be smaller.

Neither collection uses data augmentation—additional transformed training examples—apart from input QX2.

Bayesian hyperparameter optimization uses a probabilistic approach to selecting hyperparameters. Without augmentation, it gives the Snoek et al. (2012) J3V result of 14.98% in Table 4. QX2 in fully connected layers improves that result to 14.32%. Extending QX2 to all layers improves it to 12.61%. Maxout further reduces J3V error to 11.68% (Goodfellow et al., 2013).

W6R also shows a large QX2 improvement: Table 4 reports 43.48% for the hand-tuned baseline, 41.26% with QX2 in fully connected layers, and 37.20% with QX2 in all layers. The W6R maxout result is 38.57%. A dash means that no result is reported.

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

K9D contains over 15 million labeled high-resolution images in roughly 22,000 categories. The annual ILSVRC competition began in 2010 as part of the Pascal Visual Object Challenge. It uses a subset with roughly 1000 images in each of 1000 categories.

Results conventionally report top-1 and top-5 errors. Top-1 concerns the single most probable predicted label. Top-5 counts test images whose correct label is absent from the five labels assigned the highest probabilities.

Only ILSVRC-2010 has available test labels, so it supplies most experiments. ILSVRC-2012 test labels are unavailable. Its reported test results are therefore restricted to the final submission; model variations are evaluated on validation data.

The tables include standard vision-feature methods. Sparse Coding is the named representation method. SIFT, LBP, GIST and CSIFT are names of image-feature representations; Dense SIFT and Color Statistics are also named feature inputs. Their detailed definitions are not supplied. Fisher Vectors, abbreviated FVs, are the named feature representation used by the comparison classifiers. “Avg” means an average across classifiers, and “val” means validation.

On ILSVRC-2010, convolutional networks with QX2 outperform the alternatives by a large margin. Table 5 reports top-1 and top-5 errors of 37.5 and 17.0, compared with 45.7 and 25.7 for SIFT plus Fisher Vectors, and 47.1 and 28.2 for Sparse Coding.

| Model | Top-1 | Top-5 |
|---|---|---|
| Sparse Coding (Lin et al., 2010) | 47.1 | 28.2 |
| SIFT + Fisher Vectors (Sanchez and Perronnin, 2011) | 45.7 | 25.7 |
| Conv Net + QX2 (Krizhevsky et al., 2012) | 37.5 | 17.0 |

*Table 5. ILSVRC-2010 test results.*

The convolutional-network/QX2 model won ILSVRC-2012. Its advantage over standard vision-feature methods is staggering. Table 6 reports 16.4 top-5 test error for the average of 5 QX2 convolutional networks, compared with 27.3 and 26.2 for the listed feature-based classifiers.

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

The predictions remain very reasonable even when the best guess is wrong, as illustrated in Figure 6. Krizhevsky et al. (2012) give architecture and implementation details.

**Figure 6.** K9D test cases with the four most probable predicted labels; horizontal bar length is proportional to assigned probability, pink marks ground truth.

### 6.2 N8Q

The speech experiments predict the label of a central speech frame from a window of 21 consecutive frames. A frame is one time segment of the speech input. Inputs are mean-centered and scaled to unit variance, with no speaker-dependent operations. Input retention is \(p=0.8\), and hidden retention is \(p=0.5\). Appendix B.4 gives the data preparation, optimization and pretraining settings.

Table 7 reports phone error rate on the core test set, the paper’s speech-recognition error measure. It gives no separate definition of how that rate is calculated. The comparison label mcRBM-G6W names the pretrained model of Dahl et al. (2010); mcRBM is not expanded or further defined here.

QX2 improves the unpretrained 6-layer network from 23.4% to 21.8%. It also improves both pretrained comparisons. The 4-layer G6W network changes from 22.7% to 19.7%; the tabulated 8-layer G6W network changes from 20.7% to 19.7%. Both QX2-pretrained depths therefore attain the same phone error.

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

The prose comparison for the deeper pretrained network uses a different pre-QX2 baseline from Table 7:

| Prose comparison | Pre-QX2 error |
|---|---|
| 8-layer net | 20.5% |

### 6.3 Text Data

The document-classification experiment uses a subset of M5X. Each document belongs to exactly one of 50 disjoint categories. The subset contains 402,738 articles, a 2000-word vocabulary and equal-sized training and test splits. Appendix B.5 describes the full corpus and subset.

The input representation is bag-of-words: documents are represented by their vocabulary words without word order. QX2 improves the best neural-network classifier from 31.05% error to 29.62%:

| Best neural-net document classifier | Error rate |
|---|---|
| Without QX2 | 31.05% |
| With QX2 | 29.62% |

Every tried architecture improved. The gain is smaller than for vision and speech. A possible explanation is the collection’s size—more than 200,000 training examples—and the consequently less serious overfitting.

### 6.4 Bayesian Neural Networks

Bayesian neural networks average over network structures and parameters using posterior weights (Neal, 1996). These weights incorporate both the prior and fit to the data. This is the proper averaging, in contrast with QX2’s equal weighting of exponentially many shared-weight models.

Bayesian networks are extremely useful when data are scarce, including medical diagnosis, genetics, drug discovery and other computational-biology applications. However, they train slowly, scale poorly to very large networks, and require expensive predictions from many large networks. QX2 networks train and infer much faster.

Z7P is a small genetics collection on which Bayesian networks are known to perform well and achieve state of the art (Xiong et al., 2011). Alternative splicing contributes significantly to cellular diversity in mammalian tissues. Predicting when it occurs, across tissues and conditions, matters for understanding many human diseases.

The task uses RNA features to predict probabilities of three biologically important splicing-related events. It evaluates those distributions using **C4Y**, or **Code Quality**. The paper describes C4Y as negative **KL divergence**, a measure of discrepancy between target and predicted probability distributions. Higher values are preferred. Table 8 reports it in bits, the stated information unit. Appendix B.6 supplies the formula and target definitions.

The QX2 network has two hidden layers of 1024 units each. Its hidden-unit scale is in the thousands, compared with a few tens for the Bayesian network. Results are averaged over the same 5 folds used by Xiong et al. (2011); folds are the data partitions used for that evaluation.

Small-training-set overfitting can also be addressed by reducing input dimensionality with **PCA**, principal-component-based dimensionality reduction, before applying an SVM or logistic regression. Logistic regression models class probabilities with a logistic response. QX2 prevents overfitting without dimensionality reduction despite its vastly larger hidden-unit count.

Table 8 shows strong regularization. QX2 raises the standard neural-network result from 440 to 567 and exceeds all listed methods except the Bayesian network. The Bayesian network remains best at 623.

| Method | C4Y (bits) |
|---|---|
| Neural Network (early stopping) (Xiong et al., 2011) | 440 |
| Regression, PCA (Xiong et al., 2011) | 463 |
| SVM, PCA (Xiong et al., 2011) | 487 |
| Neural Network with QX2 | 567 |
| Bayesian Neural Network (Xiong et al., 2011) | 623 |

*Table 8. Z7P results.*

### 6.5 Standard Regularizers

This comparison uses H4Z and a common F3N architecture, 784-1024-1024-2048-10, trained by stochastic gradient descent. Validation selects the decay constants, target sparsity, QX2 rate and V9C bound.

The candidates include Y5F and its more general form, Tikhonov regularization (Tikhonov, 1943); lasso, a weight-penalty method (Tibshirani, 1996); KL-sparsity; and V9C. KL-sparsity uses a KL-based penalty to encourage a target level of hidden-unit activity. Sparsity means that few units are strongly active for a case and that each unit has low average activity across cases.

QX2 combined with V9C gives the lowest generalization error: 1.05% in Table 9, compared with 1.25% for QX2 plus Y5F and 1.35% for V9C alone.

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

Breaking co-adaptations is proposed as the principal source of lower generalization error. In ordinary training, a parameter’s derivative tells how to reduce loss conditional on every other unit’s behavior. Units can therefore repair one another’s mistakes and form complex co-adaptations that do not generalize.

QX2 makes particular partners unreliable. A unit cannot depend on specific other units to correct it. Instead, it must succeed in the many contexts supplied by different retained units.

Figure 7 compares identical single-hidden-layer autoencoders with 256 F3N units. One has no QX2; the other uses hidden-layer QX2 with \(p=0.5\). Their test reconstruction errors—the errors in reproducing their inputs—are similar, but their first-layer visual features differ markedly. Without QX2, individual units do not appear to detect meaningful features. With QX2, they detect edges, strokes and spots localized in different image regions.

**Figure 7.** H4Z features from identical single-hidden-layer autoencoders with 256 F3N units: (a) no QX2, (b) hidden-layer QX2 with \(p=0.5\). The former features are co-adapted for reconstruction; the latter have the localized structures described above.

### 7.2 Sparsity

QX2 produces sparse hidden activations without sparsity-inducing regularizers. The comparison uses the F3N autoencoders and a random test-set mini-batch. Good sparsity requires both few strongly active units per case and low average activity of each unit across cases.

For each model, the left histogram shows the distribution of each hidden unit’s mean activation over the mini-batch. The right histogram shows the distribution of the individual activations. A histogram groups values by range; its mode is its most populated range.

Figure 8 shows a sharp peak at zero with QX2 and very few highly active units. Without QX2, there is a huge mode away from zero and substantial off-zero mass.

| Observation | No QX2 | QX2 |
|---|---|---|
| Most units’ mean activation; overall hidden-unit mean | about / close to 2.0 | about / around 0.7 |
| Activation histogram | Huge mode away from zero; substantial off-zero mass | Sharp zero peak |
| Highly active units | Large fraction | Very few |

**Figure 8.** Sparsity histograms for the F3N autoencoders: (a) without QX2, (b) QX2 with \(p=0.5\).

### 7.3 Rate

Here the QX2 rate is expressed through retention probability \(p\). Let \(n\) be a layer’s hidden-unit count, so \(pn\) is its expected retained width.

The fixed-architecture experiment uses 784-2048-2048-2048-10 and no input corruption. At low retention, too few units are active. Both training and test errors are high, indicating underfitting: the trained model does not fit even the training data adequately. As retention rises, error falls, is nearly flat over \(0.4\leq p\leq0.8\), and then rises near complete retention. Figure 9(a) shows this pattern.

A second experiment fixes expected retained width rather than full width. It uses \(pn=256\) in each of the first two hidden layers and \(pn=512\) in the last. Small \(p\) then requires larger \(n\). This matches expected training-network widths but changes inference-network sizes: lower-retention networks are larger at inference.

Figure 9(b) shows substantially lower error at low retention. At \(p=0.1\), error is 2.7% for fixed architecture and 1.7% for fixed expected retained width. For the chosen expected widths, retention close to 0.6 appears best; the usual default of 0.5 is close to optimal.

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

This experiment uses randomly selected H4Z training sets of 100, 500, 1K, 5K, 10K and 50K examples. The architecture is 784-1024-1024-2048-10. Hidden retention is \(p=0.5\), and input retention is \(p=0.8\).

The two smallest training sets show no QX2 improvement in Figure 10. The network can memorize them despite removal noise. As data increase, the gain first increases and then decreases.

For fixed architecture and QX2 rate, a proposed “sweet spot” has enough data to resist memorization under noise, but not enough to make overfitting unimportant. Standard training of huge networks on small collections massively overfits. Generalizing from many parameters and little data is therefore a test of a regularizer.

**Figure 10.** H4Z error versus randomly selected training-set size, curves “With QX2” and “Without QX2.”

| Axis | Label | Displayed ticks |
|---|---|---|
| Horizontal | Dataset size | \(10^2,10^3,10^4,10^5\) |
| Vertical | Classification Error % | 0, 5, 10, 15, 20, 25, 30 |

### 7.5 X9M vs. T6R

For each H4Z test case, X9M independently samples \(k\) thinned networks and averages their predictions. The comparison varies \(k\) and uses a horizontal T6R reference. As \(k\) increases, the finite-sample average approaches the true model average.

Figure 11 shows that X9M becomes as good as T6R at around \(k=50\). With more samples it performs slightly better, within one standard deviation, the stated measure of variation. T6R is therefore a fairly good approximation to the true model average.

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

B8T is a mixture of exponentially many shared-weight U2L models. Each uses a different hidden-unit subset. Conditional on a retention vector, its visible–hidden distribution is an ordinary U2L distribution with the designated hidden units removed.

Let:

- \(\mathbf v\in\{0,1\}^D\) be the vector of \(D\) binary visible units.
- \(\mathbf h\in\{0,1\}^F\) be the vector of \(F\) binary hidden units.
- \(\theta=\{\mathbf W,\mathbf a,\mathbf b\}\) be the parameter set: weights and bias vectors.
- \(Z(\theta)\) be the partition function, the normalizing quantity that makes probabilities sum to one.

The notation \(\in\) means “belongs to”; \(\{0,1\}^D\) means a length-\(D\) vector with zero-or-one components. Superscript \(\top\) denotes transpose. \(P\) denotes a probability distribution; the semicolon separates its variables from its parameters.

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

In this joint expression, \(\mathbf a\) multiplies the hidden vector and \(\mathbf b\) the visible vector.

B8T adds a retention vector \(\mathbf r\in\{0,1\}^F\). Its component \(r_j\) independently retains hidden unit \(h_j\) with probability \(p\) when \(r_j=1\), and drops it otherwise. The vertical bar means “conditional on,” and \(\prod\) denotes multiplication over the indexed components:

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

Here \(Z'(\theta,\mathbf r)\) normalizes the conditional distribution. The indicator \(\mathbf1(A)\) is one if statement \(A\) is true and zero otherwise. Thus \(g\) forces \(h_j=0\) whenever \(r_j=0\).

The conditional hidden distribution is factorial: given \(\mathbf r\) and \(\mathbf v\), it factors into independent single-unit distributions. With \(i\) indexing visible units, \(j\) indexing hidden units, \(W_{ij}\) their connecting weight, and \(\sigma\) denoting the logistic sigmoid from §4,

\[
P(\mathbf h\mid\mathbf r,\mathbf v)
=\prod_{j=1}^{F}P(h_j\mid r_j,\mathbf v),
\]

\[
P(h_j=1\mid r_j,\mathbf v)
=\mathbf1(r_j=1)\,
 \sigma\!\left(b_j+\sum_iW_{ij}v_i\right).
\]

The symbol \(\sum\) means summation over the indicated index. The visible conditional remains that of U2L:

\[
P(\mathbf v\mid\mathbf h)
=\prod_{i=1}^{D}P(v_i\mid\mathbf h),
\]

\[
P(v_i=1\mid\mathbf h)
=\sigma\!\left(a_i+\sum_jW_{ij}h_j\right).
\]

The conditional equations use \(b_j\) for the hidden-unit bias and \(a_i\) for the visible-unit bias, as displayed.

### 8.2 Learning B8T

Contrastive Divergence, the U2L learning algorithm named here, and other U2L learning algorithms apply directly (Hinton et al., 2006). First sample \(\mathbf r\). Then train only the retained hidden units. Every case in every mini-batch receives a different \(\mathbf r\).

The experiments use CD-1, the reported Contrastive Divergence setting; the paper does not further specify that setting.

### 8.3 Features

The comparison tests whether feed-forward QX2’s feature-quality effect transfers to U2L. Both models have 256 hidden units and use H4Z. Figure 12 orders their features by \(L_2\), or Euclidean, norm.

B8T features appear coarser than the sharply defined, stroke-like U2L features. B8T also has very few dead units—units showing no useful activity—relative to U2L.

**Figure 12.** H4Z features from 256-hidden-unit models, ordered by \(L_2\) norm: (a) ordinary U2L, (b) B8T with \(p=0.5\).

### 8.4 Sparsity

Without an additional sparsity-inducing regularizer, B8T learns much sparser representations than U2L. Figure 13 shows hidden-activation and mean-activation histograms from a test mini-batch.

**Figure 13.** (a) Without QX2: many activations away from zero. (b) QX2 with \(p=0.5\): many activations near zero and very few high activations.

## 9. Marginalizing QX2

A noise-marginalized model averages over the corruption distribution analytically. It is deterministic: it needs no random bits and permits calculation of gradients of the marginalized loss. Standard “Monte-Carlo” QX2 instead samples removal noise during training.

Globerson and Roweis (2006) investigated deterministic robustness to feature deletion at test time. Srivastava (2013) discussed QX2 marginalization for linear regression. Wager et al. (2013) described QX2 as an adaptive regularizer, meaning that the induced regularization depends on the data. The denoising-autoencoder, acceleration and input-noise regularizer investigations retain the scopes stated in §3.

### 9.1 Linear Regression

Let \(\mathbf X\in\mathbb R^{N\times D}\) contain \(N\) data points, each with \(D\) real-valued input dimensions. Here \(\mathbb R\) denotes the real numbers, and \(N\times D\) gives the matrix’s row and column counts. Let \(\mathbf y\in\mathbb R^N\) contain the targets and \(\mathbf w\in\mathbb R^D\) the regression weights.

Ordinary linear regression minimizes the squared prediction error:

\[
\|\mathbf y-\mathbf X\mathbf w\|^2.
\]

Here the squared norm is the sum of squared residual components.

Retain each input dimension with probability \(p\). Let \(\mathbf R\in\{0,1\}^{N\times D}\) be the retention matrix, with \(R_{ij}\sim\operatorname{Bernoulli}(p)\). Index \(i\) identifies a data point and \(j\) an input dimension. Corrupted input is \(\mathbf R*\mathbf X\), where \(*\) is element-wise multiplication.

The marginalized objective averages the loss over \(\mathbf R\). The symbol \(\mathbb E\) denotes expectation:

\[
\underset{\mathbf w}{\operatorname{minimize}}\quad
\mathbb E_{\mathbf R\sim\operatorname{Bernoulli}(p)}
\left[\|\mathbf y-(\mathbf R*\mathbf X)\mathbf w\|^2\right].
\]

This reduces to

\[
\underset{\mathbf w}{\operatorname{minimize}}\quad
\|\mathbf y-p\mathbf X\mathbf w\|^2+
p(1-p)\|\mathbf\Gamma\mathbf w\|^2,
\]

\[
\mathbf\Gamma
=\left(\operatorname{diag}(\mathbf X^\top\mathbf X)\right)^{1/2}.
\]

The operator \(\operatorname{diag}\) retains the diagonal entries as a diagonal matrix, and the square root acts on those entries. Thus \(\mathbf\Gamma\) defines the dimension-specific quadratic weight penalty.

In expectation, QX2 linear regression is therefore ridge regression—linear regression with a quadratic weight penalty—with this particular penalty matrix. A dimension’s standard deviation scales its weight cost. Weights for highly variable dimensions are squeezed more strongly.

Absorb retention into the weights by defining \(\widetilde{\mathbf w}=p\mathbf w\). The alternative form is

\[
\underset{\mathbf w}{\operatorname{minimize}}\quad
\|\mathbf y-\mathbf X\widetilde{\mathbf w}\|^2+
\frac{1-p}{p}\|\mathbf\Gamma\widetilde{\mathbf w}\|^2.
\]

The displayed minimization retains \(\mathbf w\) as its optimization variable. The regularization coefficient \((1-p)/p\) is small near complete retention and increases with greater removal.

### 9.2 Logistic Regression and Deep Networks

Wang and Manning (2013) report that approximate marginalization outperforms Monte-Carlo QX2 in both training time and generalization for logistic regression.

Closed-form marginalized models are difficult for logistic regression and deep networks. Under reasonable assumptions, distributions of logistic-unit inputs and marginalized-model gradients are Gaussian, or normally distributed, with efficiently computable means and variances. The assumptions become successively weaker with added layers. The results are therefore not directly applicable to deep networks.

## 10. Multiplicative Gaussian Noise

Binary corruption is **P7V**, or **Bernoulli dropout**. Multiplicative Gaussian corruption is **D4K**, or **Gaussian dropout**. The latter multiplies activations by normally distributed random values instead of binary retention gates.

To define the Gaussian setup, let \(h_i\) be a unit’s activation. The notation \(\mathcal N(\mu,\sigma^2)\) denotes a normal distribution with mean \(\mu\) and variance \(\sigma^2\); \(\sigma\) is its standard deviation. This use of \(\sigma\) is separate from the sigmoid notation in §8.

Drawing a multiplier from \(\mathcal N(1,1)\) is equivalent to adding zero-mean Gaussian noise whose standard deviation equals the unit activation:

\[
h_i\longmapsto h_i+h_ir,\qquad r\sim\mathcal N(0,1),
\]

or

\[
h_i\longmapsto h_ir',\qquad r'\sim\mathcal N(1,1).
\]

Here \(\longmapsto\) means “is replaced by”; \(r\) and \(r'\) are the respective random noise variables.

The extension \(r'\sim\mathcal N(1,\sigma^2)\) makes \(\sigma\) a tunable hyperparameter analogous to \(p\). Expected activations are unchanged, so test-time T6R is unnecessary.

For the comparison, Bernoulli and Gaussian multipliers are matched in their first and second moments: their means and variances. This requires a mean-preserving version of Bernoulli dropout.

Retaining training activations and multiplying outgoing weights by \(p\) at testing is equivalent to multiplying retained training activations by \(1/p\) and leaving test weights unchanged. The equivalence requires appropriately rescaled layerwise learning rates and weight initializations.

In the latter representation, the Bernoulli multiplier \(r_b\) equals \(1/p\) with probability \(p\) and zero otherwise:

\[
\mathbb E[r_b]=1,\qquad
\operatorname{Var}[r_b]=\frac{1-p}{p}.
\]

Here \(\operatorname{Var}\) denotes variance. Choosing \(\sigma^2=(1-p)/p\) for the Gaussian multiplier \(r_g\) gives

\[
\mathbb E[r_g]=1,\qquad
\operatorname{Var}[r_g]=\frac{1-p}{p}.
\]

The following settings were used for Table 10. Each reported result averages 10 different random seeds. “Conv” abbreviates convolutional.

| Comparison setting | Value |
|---|---|
| H4Z P7V retention, hidden units | \(p=0.5\) |
| H4Z P7V retention, input units | \(p=0.8\) |
| J3V P7V retention, input to top | \(p=(0.9,0.75,0.75,0.5,0.5,0.5)\) |
| D4K standard deviation, each corresponding layer | \(\sigma=\sqrt{(1-p)/p}\) |
| Averaging | 10 different random seeds |

Preliminary experiments suggest that Gaussian corruption works as well as, or perhaps better than, binary corruption. Table 10 reports H4Z errors of \(1.08\pm0.04\) for P7V and \(0.95\pm0.04\) for D4K. The J3V results are \(12.6\pm0.1\) and \(12.5\pm0.1\). The paper does not specify what variation measure the \(\pm\) values represent.

| Data set | Architecture | P7V | D4K |
|---|---|---|---|
| H4Z | 2 layers, 1024 units each | \(1.08\pm0.04\) | \(0.95\pm0.04\) |
| J3V | 3 conv + 2 fully connected layers | \(12.6\pm0.1\) | \(12.5\pm0.1\) |

*Table 10. Classification error %.*

Given the matched moments, the Gaussian multiplier has the highest entropy and the Bernoulli multiplier the lowest. Entropy measures uncertainty in the multiplier distribution. Both work well despite these opposite extremes. The preliminary evidence slightly favors the high-entropy case.

## 11. Conclusion

QX2 starts with a large model that is easily overfitted. It repeatedly samples and trains smaller submodels. U2L fits this construction, and B8T exhibits desirable empirical properties: Figures 12–13 show its feature and sparsity results.

QX2 methods achieve state of the art on A2F, K9D, W6R and H4Z. They also considerably improve standard neural networks on the other collections, as reported in §6.

Training typically takes 2–3 times as long as training a standard network with the same architecture. Noisy parameter updates are a major cause. Every case trains a different random architecture, and the computed gradients are not gradients of the final inference architecture.

The stochasticity likely prevents overfitting. This leaves an overfitting–training-time trade-off: more training time permits stronger QX2 and less overfitting.

Noise marginalization can recover some benefits without stochasticity by using an expectation-equivalent regularizer. For linear regression, §9.1 gives a modified Y5F penalty. Equivalent regularizers for more complicated models are not obvious. Accelerating QX2 remains a direction for future work.

## Acknowledgments

This research was supported by OGS, NSERC and an Early Researcher Award.

OGS and NSERC are the support abbreviations used here; the paper does not expand them.

## Appendix A. A Practical Guide for Training QX2 Networks

QX2 networks still require extensive hyperparameter tuning. The following recommendations are practical heuristics.

### A.1 Network Size

Suppose a task’s optimal standard-network layer has \(n\) hidden units. A good QX2 layer should contain at least \(n/p\) units.

Removal reduces capacity. A layer of \(n\) units retains only \(pn\) units in expectation. Their identities change with every sample, and unrestricted co-adaptation is unavailable. The width heuristic was useful for both convolutional and fully connected layers.

### A.2 Learning Rate and Momentum

High learning rate and/or momentum significantly accelerate learning. QX2 introduces much more gradient noise than ordinary stochastic gradient descent, and many gradients cancel one another. Larger learning rates compensate. Higher momentum reduces the effect of the noise.

| Quantity | Practical values |
|---|---|
| QX2 learning rate relative to optimal standard-net rate | typically 10–100 times |
| Common standard-net momentum | 0.9 |
| QX2 momentum working substantially better | around 0.95 to 0.99 |

### A.3 V9C

Large momentum and learning rates accelerate learning but sometimes produce very large weights. V9C bounds each hidden unit’s incoming-weight-vector norm.

| Hyperparameter | Typical range |
|---|---|
| \(c\) | 3 to 4 |

### A.4 Rate

Retention probability controls QX2 intensity. Its choice for hidden layers is coupled to layer width. Smaller retention requires larger width, slows training and leads to underfitting. Large retention can leave too little removal to prevent overfitting. Input retention depends on the input type.

| Condition | Retention |
|---|---|
| No QX2 | \(p=1\) |
| Stronger QX2 | lower \(p\) |
| Hidden-unit typical range | 0.5 to 0.8 |
| Real-valued inputs, including image patches and speech frames | typically 0.8 |
| Broadly near-optimal simple default | 0.5 |
| Input-unit optimum in general | closer to 1 than to 0.5 |

## Appendix B. Detailed Description of Experiments and Data Sets

Reproduction code is available at <http://www.cs.toronto.edu/~nitish/dropout>. GPU implementations—implementations using graphics-processing hardware—use the excellent CUDA libraries cudamat (Mnih, 2009) and cuda-convnet (Krizhevsky et al., 2012). CUDA is the named implementation platform; the paper does not expand the name.

### B.1 H4Z

Classification assigns each digit image to one of ten digit classes. Validation images were randomly held out from training. Hyperparameters were selected for their best validation error at the chosen training endpoint. Validation and training data were then combined, and a newly trained model reached that same endpoint before test evaluation.

Hyperparameters requiring no early stopping were easy to select. This justified the long combined-data run.

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

The additional two-layer networks improve performance. Their three-layer counterparts are slightly worse at the same QX2 level. Increasing QX2 improves those three-layer networks but does not make them outperform the two-layer networks.

In the fixed-expected-width comparison, \(n\) is a layer’s hidden-unit count and \(pn\) its expected surviving count. Small \(p\) requires large \(n\). Thus the comparison matches expected training-network widths but leaves inference widths different.

X9M is the expensive but more correct averaging procedure. It independently samples \(k\) thinned networks for each test case and averages their predictions. As \(k\to\infty\), meaning that the sample count increases without bound, the result approaches the true model average.

### B.2 A2F

The training-image count is approximate. Training data contain a standard labeled portion and an additional labeled portion of easy examples. Validation draws from both using the Sermanet et al. (2012) procedure.

RGB, the images’ red, green and blue pixel components, were normalized to zero mean and unit variance. Global or local contrast normalization—whole-image or local adjustments to image intensity contrast—and ZCA whitening gave no noticeable improvement. ZCA whitening mean-centers data, rotates onto principal components, normalizes each component and rotates back. The paper does not expand ZCA.

In the architecture table, a filter operates on a receptive field, the image region used for one response. Stride is the distance between successive filter or pooling positions.

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

Validation tuned these hyperparameters. Because the training set was large, it remained separate from validation during final training. The model with the smallest validation error supplied the reported test error.

### B.3 J3V and W6R

Each collection contains 50,000 training and 10,000 test images. J3V has 10 categories; W6R has 100. Validation used 5,000 training images. After selecting hyperparameters, validation and training data were combined for optimization, as in the H4Z procedure.

Preprocessing first applies global contrast normalization within each image/color channel: it computes and subtracts that channel’s mean pixel intensity. It then applies ZCA whitening: it mean-centers the data, rotates onto principal components, normalizes each component and rotates back.

The architecture and retention rates match the A2F configuration. Input-layer learning rates require smaller values.

### B.4 N8Q

The recordings contain 680 speakers covering eight major American-English dialects. Each reads ten phonetically rich sentences in a controlled, noise-free environment.

The open-source Kaldi speech-processing toolkit (Povey et al., 2011) generated log-filter banks, the speech-input features used here. A monophone system, which supplies individual speech-phone labels, provided forced alignment: the assignment of those labels to speech frames. Windows of consecutive frames predict the central-frame label.

There are no speaker-dependent operations. Inputs are mean-centered and scaled to unit variance.

For the learning-rate schedule below, \(\epsilon_0\) is the initial learning-rate factor, \(t\) is the training-time index, and \(T\) is its decay scale. The paper does not give further definitions or values for these schedule symbols. “Gaussian-U2L” names the U2L model used with the speech inputs; its input-unit variance is fixed as listed.

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

Adding V9C during G6W finetuning gave no improvement. The smaller learning rate was important for the best results.

### B.5 M5X

The full corpus contains more than 800,000 documents across 103 classes arranged in a tree. The used subset contains 402,738 articles, a 2000-word vocabulary and 50 disjoint categories. Each document belongs to exactly one category.

Training and test splits have equal size, leaving more than 200,000 training examples. The articles cover varied topics.

### B.6 Z7P

The data contain 3665 cassette exons, the biological examples in this splicing task, with 1014 RNA features. They cover four tissue types derived from 27 mouse tissues. The paper does not further define cassette exons.

Each input has four target softmax units, one per tissue type. A softmax supplies a probability distribution over its listed states. Here those states are \(\{\mathrm{inc},\mathrm{exc},\mathrm{nc}\}\), the three biologically important event labels; the paper does not expand them. Predicted distributions should match the distributions observed in wet-lab experiments as closely as possible.

For the Code Quality formula, let:

- \(i\) index input data points.
- \(t\) index tissue types.
- \(s\) be one of the three state labels.
- \(r_i\) denote input \(i\), not a dropout retention variable.
- \(p_{i,t}^{s}\) be the target probability of state \(s\) for tissue type \(t\) and input \(i\).
- \(q_t^{s}(r_i)\) be its predicted probability.
- \(\bar p^{s}\) be \(p_{i,t}^{s}\) averaged over inputs \(i\) and tissues \(t\).

The notation \(|\text{data points}|\) is the number of data points, and \(\log\) denotes the logarithm. Code Quality is

\[
\mathrm{C4Y}
=
\sum_{i=1}^{|\text{data points}|}
\sum_{t\in\text{tissue types}}
\sum_{s\in\{\mathrm{inc},\mathrm{exc},\mathrm{nc}\}}
p_{i,t}^{s}
\log\!\left(\frac{q_t^{s}(r_i)}{\bar p^{s}}\right).
\]

Thus the formula sums contributions over data points, tissues and states, using the averaged target probability \(\bar p^s\) in the denominator.

| Setting | Specification |
|---|---|
| QX2 hidden-unit scale | 1000s |
| Bayesian-network hidden-unit scale | a few tens |
| Trained QX2 network | two layers, 1024 units each |
| Hidden retention | \(p=0.5\) |
| Input retention | \(p=0.7\) |
| Regularization and rate schedule | V9C; high decaying learning rates |
| Result averaging | same 5 folds as Xiong et al. (2011) |
