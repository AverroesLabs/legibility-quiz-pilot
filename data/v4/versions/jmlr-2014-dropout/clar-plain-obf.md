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

Deep neural networks are machine-learning systems with many layers of connected processing units. Large networks with many learned parameters are powerful, but they can overfit: they learn patterns in training data that do not hold in new data. They also run slowly, which makes combining several independently trained large models difficult at test time.

Dropout, labeled QX2 in this paper, temporarily removes randomly selected units and their connections during training. It limits excessive co-adaptation, in which units depend too much on particular other units. It samples exponentially many thinned networks that share parameters. A single full network with reduced weights approximates their average predictions.

Compared with other regularizers—methods that limit overfitting—dropout substantially reduces overfitting and gives major improvements. Experiments using labeled data in vision, speech recognition, document classification, and computational biology achieve state-of-the-art performance on many benchmarks.

**Keywords:** neural networks, regularization, model combination, deep learning

## 1. Introduction

With unlimited computation, the best way to regularize a fixed-sized model is to average over every parameter setting, weighted by its posterior probability. This probability reflects both prior beliefs and the training data. Simple or small models sometimes allow good approximations to this average (Xiong et al., 2011; Salakhutdinov and Mnih, 2008).

We propose a less expensive method. It approximates an equally weighted geometric mean of predictions from exponentially many learned models that share parameters. A geometric mean averages values by multiplying them and taking the appropriate root. A network with \(n\) units represents \(2^n\) possible thinned networks. Weight sharing keeps the parameter count at \(O(n^2)\), or fewer: at most on the order of \(n^2\). Each presentation of each training case trains a newly sampled network. Any particular network therefore receives training rarely, if ever.

Multiple layers of nonlinear hidden representations can express complex input–output relationships. Hidden representations are internal features computed between the input and output. With limited data, sampling noise creates relationships that appear in training data but not in genuine test data. This can happen even when the test data come from the same distribution.

Established responses include stopping training when performance on held-out validation data worsens and using soft weight sharing (Nowlan and Hinton, 1992), which encourages weights to take similar values. Other responses penalize weights. Familiar examples are L1 regularization, labeled O3U, which penalizes absolute weight values, and L2 regularization, labeled Y5F, which penalizes squared weight values. These methods constrain learned parameters rather than average over every possible parameter setting.

Combining models nearly always improves machine-learning performance. However, training many large networks separately makes averaging their outputs prohibitively expensive. Useful diversity requires different architectures or different training data. Finding the best hyperparameters—settings chosen outside parameter learning—for each architecture is difficult. Each large-network training run also requires substantial computation. Large data requirements can prevent training different networks on different subsets. Even if training succeeds, running all the models together may be infeasible for applications that need fast responses.

The practical prediction rule, weight scaling, is labeled T6R. It uses one full network instead of explicitly evaluating exponentially many thinned networks. A more expensive alternative, Monte-Carlo model averaging, is labeled X9M. It averages predictions from randomly sampled networks. Across varied classification problems, weight scaling gives significantly lower generalization error—error on new data—than other regularization methods.

The same unit-removal method extends beyond feed-forward networks, whose signals move from input toward output. It also applies to graphical models, which represent statistical relationships among variables. These include Boltzmann Machines. One standard family, Restricted Boltzmann Machines, is abbreviated RBM and labeled U2L. Its dropout version, Dropout RBM, is labeled B8T and still uses shared parameters.

**Figure 1.** Left: a standard neural network with two hidden layers. Right: a thinned network produced by QX2. Crossed-out units and all their incoming and outgoing connections are removed.

Paper map: Section 2 gives the motivation. Section 3 reviews previous work. Section 4 defines the formal QX2 model. Section 5 describes training. Section 6 reports experiments across domains and compares regularization and model-combination methods. Section 7 examines network properties and interactions with hyperparameters. Section 8 presents B8T. Section 9 examines QX2 marginalization, which averages out the removal noise. Appendix A gives practical training guidance, including detailed advice on choosing hyperparameters.

## 2. Motivation

Sexual reproduction combines half of one parent’s genes with half of another’s, along with a very small random mutation. Asexual reproduction produces a slightly mutated copy of one parent’s genes. Direct transmission preserves a successful set of co-adapted genes. This makes it plausible that asexual reproduction could better optimize an individual’s fitness.

Sexual recombination breaks up these sets, especially large ones. This seems disadvantageous for organisms that already have complex co-adaptations. Nevertheless, most advanced organisms evolved through sexual reproduction.

Livnat et al. (2010) suggest that long-term selection may favor gene mix-ability rather than individual fitness. A gene set that works with another randomly selected gene set is more robust. Because a gene cannot count on a large set of partners, it benefits from acting usefully alone or with only a few other genes. Reducing complex co-adaptations also helps useful new genes spread. It lowers the barriers to those genes improving individual fitness.

Hidden units trained with dropout likewise work with randomly selected partners. This favors robust features that help independently, rather than features that rely on other hidden units to fix their mistakes. Units within a layer still learn different functions. Making many copies of each hidden unit is a poor way to gain robustness, just as repetition codes poorly handle noisy communication channels.

Ten conspiracies of five people each probably have a better chance of causing havoc than one conspiracy that requires fifty people to act correctly. The larger conspiracy may work when conditions remain unchanged and participants have ample rehearsal. When conditions change, smaller conspiracies are more likely to persist. Complex co-adaptations can succeed on training data yet fail on new test data more readily than several simpler co-adaptations that achieve the same result.

## 3. Related Work

Noise in hidden-unit states gives one regularization view of dropout. This extends the approach used by Denoising Autoencoders, abbreviated DAEs and labeled S2H (Vincent et al., 2008, 2010). An autoencoder learns to reconstruct its input. A denoising autoencoder reconstructs clean input from a corrupted version.

Dropout effectively corrupts hidden layers. It also has a model-averaging interpretation, works with supervised and unsupervised learning, and applies to other neuron-based architectures such as Boltzmann Machines. Supervised learning uses target labels; unsupervised learning does not. Test-time weight scaling permits much higher noise levels than usually work best for denoising autoencoders.

| Setting | Noise level reported |
|---|---|
| S2H, typical best | 5% |
| QX2, often optimal input-unit removal | 20% |
| QX2, often optimal hidden-unit removal | 50% |

Stochastic minimization uses random samples to minimize the expected loss under a noise distribution. The loss measures prediction error. Globerson and Roweis (2006) and Dekel et al. (2010) use an adversarial alternative. An adversary chooses which features to remove, subject to a fixed maximum number of removals rather than a probability distribution. Those models have no hidden units.

In simple cases, analytically averaging out dropout gives deterministic regularizers. Van der Maaten et al. (2013) studied input-noise regularizers for exponential-family models, a class of probability models. They included dropout under the name “blankout noise,” but considered only models without hidden layers. Chen et al. (2012) examined marginalization for denoising autoencoders. Wang and Manning (2013) proposed marginalization to speed up dropout.

## 4. Model Description

For every hidden unit, its expected output under the training deletion distribution equals its actual test-time output with the prescribed reduced weights. Retention choices are random. In the simplest version, they are mutually independent and use the same fixed probability.

Validation data can select this probability. The usual default is close to optimal across many tasks and networks. The best input-retention probability is generally closer to keeping all inputs.

Let a network have \(L\) hidden layers, indexed by \(l\in\{1,\ldots,L\}\). Let \(\mathbf z^{(l)}\) and \(\mathbf y^{(l)}\) denote a layer’s inputs and outputs. Set \(\mathbf y^{(0)}=\mathbf x\), the network input. Let \(\mathbf W^{(l)}\) and \(\mathbf b^{(l)}\) denote weights and biases. A bias is an added offset.

For \(l\in\{0,\ldots,L-1\}\) and hidden unit \(i\), ordinary feed-forward computation is

\[
z_i^{(l+1)}
 =\mathbf w_i^{(l+1)}\mathbf y^l+b_i^{(l+1)},
\qquad
y_i^{(l+1)}=f\!\left(z_i^{(l+1)}\right).
\]

The activation function \(f\) converts a unit’s weighted input into its output. Any activation function is allowed, including the logistic function:

\[
f(x)=\frac{1}{1+\exp(-x)}.
\]

Dropout replaces this computation with

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

Here \(*\) means element-wise multiplication: multiply corresponding entries. A Bernoulli variable equals one with probability \(p\) and zero otherwise. Each layer’s vector \(\mathbf r^{(l)}\) contains independent Bernoulli variables. Multiplying this vector by the layer outputs gives the thinned outputs sent to the next layer.

Applying this operation at every layer samples a subnetwork. Backpropagation, which passes loss derivatives backward through the network, operates on that subnetwork. Dropout can remove both hidden and visible, or input, units.

At prediction time, no units are removed, and

\[
\mathbf W_{\mathrm{test}}^{(l)}=p\mathbf W^{(l)}.
\]

**Figure 2.** Left, training: a unit is present with probability \(p\), and its connections to the next layer have weights \(w\). Right, testing: the unit is always present, and its outgoing weights are \(pw\).

**Figure 3.** Left, standard computation: outputs \(y_1^{(l)},y_2^{(l)},y_3^{(l)}\) combine with weights \(\mathbf w_i^{(l+1)}\) and bias \(b_i^{(l+1)}\) from a \(+1\) unit. This gives the preactivation, or input before applying the activation function, \(z_i^{(l+1)}\). Applying \(f\) gives output \(y_i^{(l+1)}\). Right, QX2 computation: gates \(r_1^{(l)},r_2^{(l)},r_3^{(l)}\) first produce \(\widetilde y_1^{(l)},\widetilde y_2^{(l)},\widetilde y_3^{(l)}\). The remaining operations keep the same form.

## 5. Learning QX2 Nets

### 5.1 Backpropagation

Dropout alone gives significant gains. Combining it with high momentum, large learning rates that decay over time, and max-norm regularization gives another significant improvement. Momentum carries part of previous parameter updates into the current update. The learning rate controls the size of an update. Max-norm regularization, labeled V9C, limits the length of each unit’s incoming-weight vector.

This constraint usually improves deep-network training by stochastic gradient descent even without dropout. Stochastic gradient descent updates parameters using gradients estimated from sampled training cases. Srebro and Shraibman (2005) previously used the constraint for collaborative filtering, which predicts preferences from shared patterns.

One proposed explanation concerns optimization. A fixed-radius weight ball permits huge learning rates without exploding weights. Dropout noise explores regions of weight space that would otherwise be hard to reach. As learning rates decay, updates take progressively shorter, less exploratory steps and end at a minimum.

Training differs from standard-network stochastic gradient descent only in the thinned network sampled for each case. Each case in a mini-batch—a small group processed together—gets forward and backward propagation through its own sampled network. Training averages parameter gradients across cases. A case that does not use a parameter contributes zero to that parameter’s gradient. Momentum, gradually reduced learning rates, and L2 regularization remain useful.

For a hidden unit’s incoming-weight vector \(\mathbf w\), optimization requires

\[
\|\mathbf w\|_2\leq c.
\]

The notation \(\|\mathbf w\|_2\) means the vector’s Euclidean length. Whenever \(\mathbf w\) leaves the radius-\(c\) ball, projection moves it back to the surface. The name max-norm reflects the maximum allowed weight-vector length \(c\). Validation data tune this hyperparameter.

### 5.2 Unsupervised Pretraining

Pretraining learns useful weights before supervised finetuning, which adjusts those weights for the target task. Smaller finetuning learning rates preserve useful pretrained information. They improve final generalization compared with finetuning without dropout. In the initial experiments, learning rates comparable to the best rates for randomly initialized networks erased the pretrained information.

Pretraining itself stays unchanged. Before dropout finetuning, multiply pretrained weights by \(1/p\). This matches each unit’s expected output under random removal to its pretraining output.

Using unlabeled data through pretraining can significantly improve on random initialization in certain cases. Pretraining followed by backpropagation finetuning can use stacked Restricted Boltzmann Machines (Hinton and Salakhutdinov, 2006), autoencoders (Vincent et al., 2010), or Deep Boltzmann Machines, abbreviated DBM and labeled E5Q (Salakhutdinov and Hinton, 2009). All allow this finetuning procedure.

## 6. Experimental Results

Dropout improved generalization over neural networks without dropout on every tested data set. The experiments span object classification, digit recognition, speech, documents, and computational-biology analysis, rather than a single domain.

The collections include MNIST, labeled H4Z, a standard toy data set of handwritten digits; TIMIT, labeled N8Q, a clean-speech benchmark; and CIFAR-10 and CIFAR-100, labeled J3V and W6R, collections of tiny natural images described by Krizhevsky (2009). They also include Google Street View house-number images from the Street View House Numbers data set, abbreviated SVHN and labeled A2F (Netzer et al., 2011).

ImageNet, labeled K9D, provides large-scale natural-image data. Reuters-RCV1, labeled M5X, contains Reuters newswire documents. The Alternative Splicing data set, labeled Z7P, uses RNA features to predict gene splicing (Xiong et al., 2011). RNA is ribonucleic acid.

In the tables, K means thousand and M means million.

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

Models using dropout achieve state-of-the-art results on all five vision collections. These collections differ in image type and training-set size.

#### 6.1.1 H4Z

Dropout reduces error compared with the best neural networks that use neither dropout nor unsupervised pretraining and are permutation-invariant. Here, permutation-invariant means that the method does not use the images’ two-dimensional spatial layout.

Replacing logistic units with rectified linear units, abbreviated ReLUs and labeled F3N, reduces error further (Jarrett et al., 2009). A rectified linear unit outputs zero for a negative input and otherwise outputs the input. Max-norm regularization adds another reduction. Larger networks improve results. Replacing the activation with maxout, labeled L8J, improves them further (Goodfellow et al., 2013). Maxout takes the largest value among a group of learned linear responses.

The largest reported network is difficult to make generalize with standard regularization and early stopping. Dropout prevents its overfitting without requiring early stopping.

Pretrained networks also improve. These include networks initialized with stacked Restricted Boltzmann Machines, designated DBN and labeled G6W in the table. DBN means Deep Belief Network. The table distinguishes these from networks initialized with Deep Boltzmann Machines, labeled E5Q. The E5Q result with dropout is the best reported permutation-invariant performance.

Better results remain possible by using two-dimensional spatial information and augmenting training data with distorted images. We test those methods here on more interesting collections. In Table 2, NN means neural network, SVM means support vector machine, and NA means not applicable. A support vector machine is a classifier based on a separating boundary; a Gaussian kernel provides a nonlinear measure of similarity.

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

We held every hyperparameter, including retention probability, fixed across architectures. For otherwise identical architectures, training with and without dropout produced two sharply separated groups of test-error curves. Dropout gave a large improvement across architectures without architecture-specific hyperparameter tuning.

**Figure 4.** Test error during training for different architectures, with and without QX2. The curves are labeled “Without QX2” and “With QX2.”

| Axis | Label | Displayed ticks |
|---|---|---|
| Horizontal | Number of weight updates | 0, 200000, 400000, 600000, 800000, 1000000 |
| Vertical | Classification Error % | 1.0, 1.5, 2.0, 2.5 |

#### 6.1.2 A2F

Convolutional networks outperform other methods on this task (LeCun et al., 1989). These networks apply shared filters across local image regions. Adding dropout to their fully connected layers improves performance. In a fully connected layer, each unit connects to every unit in the preceding layer. Adding dropout to convolutional layers improves performance further. Maxout permits additional gains.

Convolutional layers have few parameters, which might suggest little overfitting and little benefit from dropout there. Yet corrupting lower layers gives noisy inputs to the higher fully connected layers and prevents them from overfitting.

The pooling methods named in Table 3 combine responses within local regions. Max-pooling takes the largest response; average pooling takes the mean. HOG denotes histograms of oriented gradients, an image-feature representation based on local edge directions. Other abbreviated feature-method names remain as reported.

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

The portion used contains color images approximately centered on one digit of a house number. The task is to identify that digit.

#### 6.1.3 J3V and W6R

Without data augmentation, Bayesian hyperparameter optimization gives the CIFAR-10 result reported by Snoek et al. (2012). This optimization uses a probabilistic model to guide the choice of hyperparameters. Dropout in fully connected layers improves the result. Extending dropout to every layer improves it further. Replacing the activation with maxout reduces error again (Goodfellow et al., 2013).

CIFAR-100 also shows a large improvement from dropout. Neither collection uses augmentation apart from input dropout.

| Method | J3V | W6R |
|---|---|---|
| Conv Net + max pooling (hand tuned) | 15.60 | 43.48 |
| Conv Net + stochastic pooling (Zeiler and Fergus, 2013) | 15.13 | 42.51 |
| Conv Net + max pooling (Snoek et al., 2012) | 14.98 | - |
| Conv Net + max pooling + QX2 fully connected layers | 14.32 | 41.26 |
| Conv Net + max pooling + QX2 in all layers | 12.61 | 37.20 |
| Conv Net + L8J (Goodfellow et al., 2013) | 11.68 | 38.57 |

*Table 4. J3V and W6R error rates.*

**Figure 5.** Image samples: (a) A2F, (b) J3V. Each row represents a different category.

#### 6.1.4 K9D

The convolutional-network model with dropout won ILSVRC-2012, the 2012 ImageNet Large Scale Visual Recognition Challenge. Its advantage over standard vision-feature methods is staggering. Even when its best guess is wrong, its predictions remain very reasonable.

On ILSVRC-2010, convolutional networks with dropout outperform the alternatives by a large margin. Krizhevsky et al. (2012) give the architecture and implementation details.

The tables report top-1 and top-5 errors. Top-1 evaluates the highest-ranked label. Top-5 counts test images whose correct label is absent from the five labels assigned the highest probabilities. Validation is abbreviated “val.” SIFT, or Scale-Invariant Feature Transform, and the other named descriptors represent image features. Fisher Vectors, abbreviated FVs, summarize such features.

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

ImageNet contains over 15 million labeled high-resolution images in roughly 22,000 categories. The annual ILSVRC competition began in 2010 as part of the Pascal Visual Object Challenge. It uses a subset with roughly 1000 images in each of 1000 categories.

Only ILSVRC-2010 has available test labels, so it supplies most experiments. ILSVRC-2012 test labels are unavailable. Its reported test results therefore cover only the final submission, while model variations are evaluated on validation data.

**Figure 6.** K9D test cases with the four most probable predicted labels. Horizontal bar length is proportional to assigned probability. Pink marks the ground-truth label.

### 6.2 N8Q

Dropout improves both the network without pretraining and the two pretrained comparisons. The two pretrained depths with dropout achieve the same phone error, which measures errors in recognizing speech sounds. For the deeper pretrained network, the prose comparison uses a pre-dropout baseline different from the corresponding table entry.

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

Dropout improves document classification on a subset of Reuters-RCV1 using bag-of-words inputs, which represent documents by word occurrences without word order. The gain is smaller than for vision and speech classification.

Every architecture tried improved. The collection’s size, and thus its less serious overfitting, may explain the smaller gain.

| Best neural-net document classifier | Error rate |
|---|---|
| Without QX2 | 31.05% |
| With QX2 | 29.62% |

### 6.4 Bayesian Neural Networks

Bayesian neural networks outperform dropout on the Alternative Splicing data set. However, dropout substantially improves standard neural networks and exceeds every other method.

One defense against overfitting on a small training set is to reduce inputs with principal component analysis, abbreviated PCA, before using support vector machines or logistic regression. PCA represents data along its main directions of variation. Logistic regression predicts class probabilities from weighted inputs.

Dropout prevents overfitting without reducing input dimensionality. It uses vastly more hidden units than the Bayesian network and shows strong regularization.

The score in Table 8 is Code Quality, labeled C4Y. It represents negative Kullback–Leibler divergence, abbreviated KL divergence, between target and predicted distributions. KL divergence measures how distributions differ. Higher Code Quality is better.

| Method | C4Y (bits) |
|---|---|
| Neural Network (early stopping) (Xiong et al., 2011) | 440 |
| Regression, PCA (Xiong et al., 2011) | 463 |
| SVM, PCA (Xiong et al., 2011) | 487 |
| Neural Network with QX2 | 567 |
| Bayesian Neural Network (Xiong et al., 2011) | 623 |

*Table 8. Z7P results.*

Posterior weighting in Bayesian neural networks is the proper way to average over network structures and parameters (Neal, 1996). It includes both the prior and fit to the data. In contrast, dropout gives equal weights to exponentially many models with shared weights.

Bayesian networks are extremely useful when data are scarce, including in medical diagnosis, genetics, drug discovery, and other computational-biology applications. However, they train slowly and scale poorly to very large networks. Prediction also requires expensive evaluation of many large networks. Dropout networks train and predict much faster.

The Alternative Splicing data set is a small genetics collection on which Bayesian networks are known to perform well and achieve state of the art (Xiong et al., 2011). Alternative splicing produces different RNA forms and contributes significantly to cellular diversity across mammalian tissues. Predicting when it occurs in particular tissues and conditions matters for understanding many human diseases. RNA features predict the probabilities of three biologically important splicing-related events. Code Quality scores these predicted distributions.

### 6.5 Standard Regularizers

Dropout combined with max-norm regularization gives the lowest generalization error in the MNIST regularizer comparison.

The candidates include L2 regularization, or more generally Tikhonov regularization (Tikhonov, 1943), which penalizes weighted parameter magnitudes. They also include lasso (Tibshirani, 1996), which uses an L1 penalty; KL-sparsity, which penalizes a difference from target sparse activity using KL divergence; and max-norm regularization.

All methods used the same rectified-linear architecture trained by stochastic gradient descent. Validation data selected decay constants, target sparsity, dropout rate, and the max-norm bound.

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

First-layer visual features differ markedly even when test reconstruction errors are similar. Without dropout, individual hidden units do not appear to detect meaningful features. With dropout, units detect edges, strokes, and spots in different image regions. We propose that breaking co-adaptations is the main source of lower generalization error.

An ordinary parameter derivative indicates how to reduce loss given every other unit’s behavior. This lets units repair one another’s mistakes and build complex co-adaptations that do not generalize. Under the dropout hypothesis, particular partners become unreliable. A unit cannot depend on specific units to correct it. It must succeed across the many contexts supplied by the remaining units.

**Figure 7.** H4Z features from identical single-hidden-layer autoencoders with 256 F3N units: (a) no QX2, (b) hidden-layer QX2 with \(p=0.5\). The features without QX2 are co-adapted for reconstruction. The QX2 features have the localized structures described above.

### 7.2 Sparsity

Dropout produces sparse hidden activations without regularizers designed to induce sparsity. Sparse activity means that few units respond strongly.

The autoencoder comparison uses a random test-set mini-batch. Good sparsity requires few strongly active units for each case and low average activity for each unit across cases. For each model, the left histogram shows the distribution of hidden-unit mean activations across the mini-batch. The right histogram shows the distribution of the activations themselves.

| Observation | No QX2 | QX2 |
|---|---|---|
| Most units’ mean activation; overall hidden-unit mean | about / close to 2.0 | about / around 0.7 |
| Activation histogram | Huge mode away from zero; substantial off-zero mass | Sharp zero peak |
| Highly active units | Large fraction | Very few |

**Figure 8.** Sparsity histograms for the F3N autoencoders: (a) without QX2, (b) QX2 with \(p=0.5\).

### 7.3 Rate

With architecture fixed, low retention causes underfitting: the network does not fit even the training data well. Few units remain active, and both training and test error are high. Error falls as retention rises. It stays nearly flat over an intermediate range, then rises near complete retention.

Keeping the expected retained width fixed substantially reduces errors at low retention. The corresponding prediction networks have different sizes. Lower-retention networks are larger.

| Comparison quantity | Value |
|---|---|
| Flat-error retention interval, fixed architecture | \(0.4\leq p\leq0.8\) |
| Retention used for low-retention comparison | \(p=0.1\) |
| Error, fixed architecture | 2.7% |
| Error, fixed expected retained width | 1.7% |
| Apparently best retention for the chosen expected widths | close to 0.6 |
| Usual default, close to optimal in that comparison | 0.5 |

**Figure 9.** H4Z training and test errors versus retention probability: (a) fixed \(n\), the number of hidden units; (b) fixed \(pn\), the expected number retained.

| Panel | Horizontal label and ticks | Vertical label and ticks | Curves |
|---|---|---|---|
| (a) | Probability of retaining a unit \((p)\): 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 | Classification Error %: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5 | Test Error; Training Error |
| (b) | Probability of retaining a unit \((p)\): 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 | Classification Error %: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 | Test Error; Training Error |

### 7.4 Data-Set Size

Dropout gives no improvement on the two smallest training sets tested. The network can memorize them despite the removal noise. As the amount of data increases, the gain first grows and then shrinks.

For a fixed architecture and dropout rate, the proposed “sweet spot” has enough data to resist memorization under noise, but not enough to make overfitting unimportant. Standard training of huge networks on small collections massively overfits. Generalizing from small data with many parameters tests a regularizer’s strength.

**Figure 10.** H4Z error versus randomly selected training-set size. The curves are labeled “With QX2” and “Without QX2.”

| Axis | Label | Displayed ticks |
|---|---|---|
| Horizontal | Dataset size | \(10^2,10^3,10^4,10^5\) |
| Vertical | Classification Error % | 0, 5, 10, 15, 20, 25, 30 |

### 7.5 X9M vs. T6R

After enough samples, Monte-Carlo model averaging matches weight scaling. It then performs slightly better, within one standard deviation, a measure of variation. Weight scaling is therefore a fairly good approximation to the true model average.

The MNIST comparison varies the number of samples and includes a horizontal weight-scaling reference. It measures how the error of a finite-sample average approaches the true-average error.

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

A Dropout RBM is a mixture of exponentially many Restricted Boltzmann Machines that share weights. Each uses a different subset of hidden units. Given a retention vector, its joint visible–hidden distribution is an ordinary Restricted Boltzmann Machine distribution with the designated hidden units removed.

Let the visible units be \(\mathbf v\in\{0,1\}^D\) and the hidden units be \(\mathbf h\in\{0,1\}^F\). Thus, \(D\) and \(F\) count binary visible and hidden units. Let the parameters be \(\theta=\{\mathbf W,\mathbf a,\mathbf b\}\). The partition function \(Z\) normalizes the probabilities so they sum to one. An ordinary Restricted Boltzmann Machine defines

\[
P(\mathbf h,\mathbf v;\theta)
=\frac{1}{Z(\theta)}
 \exp\!\left(
 \mathbf v^\top\mathbf W\mathbf h+
 \mathbf a^\top\mathbf h+
 \mathbf b^\top\mathbf v
 \right).
\]

Here \(P\) denotes probability, and the superscript \(\top\) means transpose.

The dropout version adds \(\mathbf r\in\{0,1\}^F\). It independently retains \(h_j\) when \(r_j=1\), with probability \(p\), and drops it otherwise:

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

The symbol \(\mathbf1\) is an indicator: it equals one when its stated condition holds and zero otherwise. The normalizer is \(Z'(\theta,\mathbf r)\). The function \(g\) requires \(h_j=0\) when \(r_j=0\).

The conditional hidden distribution is factorial, meaning that it factors into independent distributions for individual hidden units:

\[
P(\mathbf h\mid\mathbf r,\mathbf v)
=\prod_{j=1}^{F}P(h_j\mid r_j,\mathbf v),
\]

\[
P(h_j=1\mid r_j,\mathbf v)
=\mathbf1(r_j=1)\,
 \sigma\!\left(b_j+\sum_iW_{ij}v_i\right).
\]

Here \(\sigma\) denotes the logistic function.

The visible conditional remains that of a Restricted Boltzmann Machine:

\[
P(\mathbf v\mid\mathbf h)
=\prod_{i=1}^{D}P(v_i\mid\mathbf h),
\]

\[
P(v_i=1\mid\mathbf h)
=\sigma\!\left(a_i+\sum_jW_{ij}h_j\right).
\]

### 8.2 Learning B8T

Contrastive Divergence and other Restricted Boltzmann Machine learning algorithms apply directly (Hinton et al., 2006). Contrastive Divergence estimates a learning update using short sampling runs. First sample \(\mathbf r\), then train only the retained hidden units. Each training case in every mini-batch receives a different \(\mathbf r\). The experiments use CD-1, Contrastive Divergence with one sampling step.

### 8.3 Features

Dropout RBM features appear coarser than the sharply defined, stroke-like features of ordinary Restricted Boltzmann Machines. Dropout RBMs also have very few dead units—units that do not respond—compared with ordinary models. This comparison tests whether dropout’s effect on feature quality in feed-forward networks carries over.

**Figure 12.** H4Z features from models with 256 hidden units, ordered by \(L_2\) norm, or Euclidean length: (a) ordinary U2L, (b) B8T with \(p=0.5\).

### 8.4 Sparsity

Without an additional sparsity-inducing regularizer, Dropout RBMs learn much sparser representations than ordinary Restricted Boltzmann Machines. Histograms show hidden activations and mean activations for a test mini-batch.

**Figure 13.** (a) Without QX2: many activations lie away from zero. (b) QX2 with \(p=0.5\): many activations lie near zero, and very few are high.

## 9. Marginalizing QX2

Deterministic noise-marginalized models average out the noise rather than sample it. They need no random bits and allow computation of gradients of the marginalized losses. Standard “Monte-Carlo” dropout instead uses random samples.

Globerson and Roweis (2006) studied deterministic robustness to deletion of features at test time. Srivastava (2013) discussed dropout marginalization for linear regression. Wager et al. (2013) described dropout as an adaptive regularizer, whose penalty adapts to the data. The earlier work on denoising autoencoders, acceleration, and input-noise regularizers retains the scope stated above.

### 9.1 Linear Regression

In expectation, dropout linear regression is ridge regression with a particular penalty matrix. Ridge regression adds a squared-weight penalty to the fitting loss. Here, a dimension’s standard deviation scales its weight cost. This shrinks weights more strongly for dimensions with greater variation.

Absorbing retention into the weights reveals the regularization constant. It is small near complete retention and grows as more inputs are removed.

Let \(\mathbf X\in\mathbb R^{N\times D}\) contain \(N\) data points with \(D\) dimensions. Let \(\mathbf y\in\mathbb R^N\) contain targets, and let \(\mathbf w\in\mathbb R^D\) contain weights. The symbol \(\mathbb R\) denotes real numbers. Ordinary linear regression minimizes

\[
\|\mathbf y-\mathbf X\mathbf w\|^2.
\]

Retaining each input dimension with probability \(p\) gives \(\mathbf R*\mathbf X\), where \(\mathbf R\in\{0,1\}^{N\times D}\), \(R_{ij}\sim\operatorname{Bernoulli}(p)\), and \(*\) means element-wise multiplication. The marginalized objective is

\[
\underset{\mathbf w}{\operatorname{minimize}}\quad
\mathbb E_{\mathbf R\sim\operatorname{Bernoulli}(p)}
\left[\|\mathbf y-(\mathbf R*\mathbf X)\mathbf w\|^2\right],
\]

where \(\mathbb E\) means expectation, or average under the stated distribution. This reduces to

\[
\underset{\mathbf w}{\operatorname{minimize}}\quad
\|\mathbf y-p\mathbf X\mathbf w\|^2+
p(1-p)\|\mathbf\Gamma\mathbf w\|^2,
\]

\[
\mathbf\Gamma
=\left(\operatorname{diag}(\mathbf X^\top\mathbf X)\right)^{1/2}.
\]

Here \(\operatorname{diag}\) keeps the diagonal entries as a diagonal matrix.

With \(\widetilde{\mathbf w}=p\mathbf w\), the alternative form is

\[
\underset{\mathbf w}{\operatorname{minimize}}\quad
\|\mathbf y-\mathbf X\widetilde{\mathbf w}\|^2+
\frac{1-p}{p}\|\mathbf\Gamma\widetilde{\mathbf w}\|^2.
\]

### 9.2 Logistic Regression and Deep Networks

For logistic regression, approximate marginalization beats Monte-Carlo dropout in both training time and generalization (Wang and Manning, 2013).

Closed-form marginalized models—models expressed directly without sampling—are difficult to obtain for logistic regression and deep networks. Under reasonable assumptions, the distributions of logistic-unit inputs and marginalized-model gradients are Gaussian, or normally distributed. Their means and variances can be computed efficiently. The assumptions become successively weaker as layers are added, so the results do not apply directly to deep networks.

## 10. Multiplicative Gaussian Noise

Preliminary experiments show that multiplicative Gaussian corruption works as well as, or perhaps better than, binary corruption. Multiplicative corruption multiplies each activation by a random value. The binary version is Bernoulli dropout, labeled P7V. The Gaussian version is Gaussian dropout, labeled D4K.

The two versions can have matching first and second moments: means and variances. Yet they sit at opposite entropy extremes. Entropy measures uncertainty. Given those moments, the Gaussian multiplier has the highest entropy and the Bernoulli multiplier the lowest. Both work well. Preliminary evidence slightly favors the high-entropy case.

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

Multiplying hidden activations by Bernoulli variables generalizes to other multiplier distributions. A Bernoulli variable here equals one with probability \(p\) and zero otherwise.

The notation \(\mathcal N(\mu,\sigma^2)\) denotes a Gaussian distribution with mean \(\mu\) and variance \(\sigma^2\). Drawing a multiplier from \(\mathcal N(1,1)\) is equivalent to adding zero-mean Gaussian noise whose standard deviation equals the unit activation:

\[
h_i\longmapsto h_i+h_ir,\qquad r\sim\mathcal N(0,1),
\]

or

\[
h_i\longmapsto h_ir',\qquad r'\sim\mathcal N(1,1).
\]

The extension \(r'\sim\mathcal N(1,\sigma^2)\) makes \(\sigma\) a tunable hyperparameter, like \(p\). Expected activations remain unchanged, so test-time weight scaling is unnecessary.

One version retains activations during training and multiplies outgoing weights by \(p\) during testing. An equivalent version multiplies retained training activations by \(1/p\) and leaves test weights unchanged. This equivalence requires appropriately rescaled learning rates and weight initializations for each layer.

In the latter version, the Bernoulli multiplier \(r_b\) equals \(1/p\) with probability \(p\) and zero otherwise:

\[
\mathbb E[r_b]=1,\qquad
\operatorname{Var}[r_b]=\frac{1-p}{p}.
\]

Here \(\operatorname{Var}\) denotes variance. Choosing \(\sigma^2=(1-p)/p\) for the Gaussian multiplier \(r_g\) gives

\[
\mathbb E[r_g]=1,\qquad
\operatorname{Var}[r_g]=\frac{1-p}{p}.
\]

## 11. Conclusion

The method repeatedly samples and trains smaller submodels from a large model that easily overfits. Restricted Boltzmann Machines fit this construction, and Dropout RBMs show desirable empirical properties.

Dropout methods achieve state of the art on SVHN, ImageNet, CIFAR-100, and MNIST. They also considerably improve standard neural networks on other collections.

Training typically takes 2–3 times as long as training a standard network with the same architecture. Noisy parameter updates are a major cause. Each case trains a different random architecture. The computed gradients are not gradients of the final prediction architecture.

This randomness likely prevents overfitting. It creates a trade-off between overfitting and training time: more training time permits stronger dropout and less overfitting.

Noise marginalization gains some benefits without randomness by using a regularizer equivalent in expectation. For linear regression, this gives modified L2 regularization. Equivalent regularizers for more complex models are not obvious. Speeding up dropout remains a direction for future work.

## Acknowledgments

This research was supported by OGS, NSERC and an Early Researcher Award.

## Appendix A. A Practical Guide for Training QX2 Networks

Dropout networks still require extensive hyperparameter tuning. The following are practical rules of thumb.

### A.1 Network Size

If a task’s best standard-network layer has \(n\) hidden units, a good dropout layer should have at least \(n/p\) units.

Removing units reduces capacity. Only \(pn\) units remain on average, their identities change with every sample, and unrestricted co-adaptation is unavailable. This rule was useful for choosing both convolutional and fully connected layer widths.

### A.2 Learning Rate and Momentum

High learning rates, high momentum, or both significantly speed up learning. Dropout adds much more gradient noise than ordinary stochastic gradient descent. Many gradients cancel one another. Larger learning rates compensate for this cancellation, while higher momentum reduces the noise’s effect.

| Quantity | Practical values |
|---|---|
| QX2 learning rate relative to optimal standard-net rate | typically 10–100 times |
| Common standard-net momentum | 0.9 |
| QX2 momentum working substantially better | around 0.95 to 0.99 |

### A.3 V9C

Large momentum and learning rates can produce very large weights, even as they speed up learning. Max-norm regularization bounds the length of each hidden unit’s incoming-weight vector.

| Hyperparameter | Typical range |
|---|---|
| \(c\) | 3 to 4 |

### A.4 Rate

Retention probability controls dropout strength. For hidden layers, its choice depends on layer width. Lower retention requires wider layers, slows training, and leads to underfitting. High retention may remove too few units to prevent overfitting. The choice for inputs depends on input type.

| Condition | Retention |
|---|---|
| No QX2 | \(p=1\) |
| Stronger QX2 | lower \(p\) |
| Hidden-unit typical range | 0.5 to 0.8 |
| Real-valued inputs, including image patches and speech frames | typically 0.8 |
| Broadly near-optimal simple default | 0.5 |
| Input-unit optimum in general | closer to 1 than to 0.5 |

## Appendix B. Detailed Description of Experiments and Data Sets

Code for reproducing the experiments is available at <http://www.cs.toronto.edu/~nitish/dropout>. Implementations for graphics processing units, or GPUs, use the excellent CUDA libraries cudamat (Mnih, 2009) and cuda-convnet (Krizhevsky et al., 2012). CUDA supports computation on GPUs.

### B.1 H4Z

Classification assigns each digit image to one of ten digit classes. We randomly held out training images for validation. We selected hyperparameters for the best validation error at the chosen training endpoint. We then combined validation and training data and ran a new training process to the same endpoint before evaluating test performance.

It was easy to choose hyperparameters that required no early stopping. This justified the long run on the combined data.

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

The additional two-layer networks improve performance. Corresponding three-layer networks perform slightly worse at the same dropout level. Increasing dropout improves those three-layer networks, but they still do not outperform the two-layer networks.

In the fixed-expected-width comparison, \(n\) is a layer’s hidden-unit count, and \(pn\) is its expected surviving count. Small \(p\) requires large \(n\). This matches expected training-network widths while leaving prediction-network widths different.

Monte-Carlo model averaging is expensive but gives a more correct average. For each test case, it independently samples \(k\) thinned networks and averages their predictions. As \(k\) tends to infinity, the result approaches the true model average.

### B.2 A2F

The training-image count is approximate. Training data include a standard labeled portion and an additional labeled portion of easy examples. Validation draws from both using the procedure of Sermanet et al. (2012).

We normalized red, green, and blue, or RGB, pixels to zero mean and unit variance. Global or local contrast normalization, which adjusts image intensity variation, gave no noticeable improvement. Neither did zero-phase component analysis whitening, abbreviated ZCA whitening, which transforms data to remove correlations and normalize component variation.

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

A receptive field is the input region used by a unit. Stride is the distance between successive filter or pooling positions.

Validation data tuned these hyperparameters. During final training, the large training set stayed separate from validation. The model with the smallest validation error supplied the reported test error.

### B.3 J3V and W6R

Each collection contains 50,000 training images and 10,000 test images. They have 10 and 100 categories, respectively. Validation used 5,000 training images. After selecting hyperparameters, we combined validation and training data for optimization, following the MNIST procedure.

Preprocessing first applies global contrast normalization within each image and color channel, then ZCA whitening. Contrast normalization computes and subtracts that channel’s mean pixel intensity. ZCA whitening mean-centers the data, rotates them onto their principal components, normalizes each component, and rotates back.

The architecture and retention rates match the SVHN configuration. The input layer requires smaller learning rates.

### B.4 N8Q

The recordings contain 680 speakers covering eight major American-English dialects. Each speaker reads ten phonetically rich sentences in a controlled, noise-free environment.

The open-source Kaldi toolkit generated log-filter banks, speech features based on logged filter responses (Povey et al., 2011). A monophone system, which models individual speech sounds, supplied forced alignment and speech-frame labels. Forced alignment matches the known speech sequence to the recording.

Windows of consecutive frames predict the central frame’s label. No operations depend on the speaker. Inputs have zero mean and unit variance.

In the learning-rate schedule below, \(t\) is the training index, \(\epsilon_0\) is the initial learning rate, and \(T\) controls decay.

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

Adding max-norm regularization during Deep Belief Network finetuning gave no improvement. The smaller learning rate was important for the best results.

### B.5 M5X

The full corpus contains more than 800,000 documents in 103 classes arranged as a tree. The subset used contains 402,738 articles, a 2000-word vocabulary, and 50 disjoint categories. Each document belongs to exactly one category.

Equal-sized training and test splits give more than 200,000 training examples. The articles cover varied topics.

### B.6 Z7P

The data contain 3665 cassette exons, 1014 RNA features, and four tissue types derived from 27 mouse tissues. Cassette exons are gene segments involved in alternative splicing.

Each input has four target softmax units, one for each tissue type. A softmax produces probabilities across alternative states. Each softmax has the biologically important states \(\{\mathrm{inc},\mathrm{exc},\mathrm{nc}\}\), which denote inclusion, exclusion, and no change. Predicted state distributions should match distributions observed in wet-lab experiments as closely as possible.

\[
\mathrm{C4Y}
=
\sum_{i=1}^{|\text{data points}|}
\sum_{t\in\text{tissue types}}
\sum_{s\in\{\mathrm{inc},\mathrm{exc},\mathrm{nc}\}}
p_{i,t}^{s}
\log\!\left(\frac{q_t^{s}(r_i)}{\bar p^{s}}\right).
\]

Here \(p_{i,t}^{s}\) is the target probability for state \(s\), tissue type \(t\), and input \(i\). The value \(q_t^{s}(r_i)\) is the predicted probability for input \(r_i\). The value \(\bar p^{s}\) averages \(p_{i,t}^{s}\) over \(i\) and \(t\).

| Setting | Specification |
|---|---|
| QX2 hidden-unit scale | 1000s |
| Bayesian-network hidden-unit scale | a few tens |
| Trained QX2 network | two layers, 1024 units each |
| Hidden retention | \(p=0.5\) |
| Input retention | \(p=0.7\) |
| Regularization and rate schedule | V9C; high decaying learning rates |
| Result averaging | same 5 folds as Xiong et al. (2011) |
