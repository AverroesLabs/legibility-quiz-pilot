Journal of Machine Learning Research 15 (2014) 1929–1958  
Submitted 11/13; Published 6/14

# Dropout: A Simple Way to Prevent Neural Networks from Overfitting

```text
Nitish Srivastava                                                                nitish@cs.toronto.edu
Geoffrey Hinton                                                                 hinton@cs.toronto.edu
Alex Krizhevsky                                                                    kriz@cs.toronto.edu
Ilya Sutskever                                                                     ilya@cs.toronto.edu
Ruslan Salakhutdinov                                                          rsalakhu@cs.toronto.edu
Department of Computer Science
University of Toronto
10 Kings College Road, Rm 3302
Toronto, Ontario, M5S 3G4, Canada.
```

Editor: Yoshua Bengio

## Abstract

Deep neural networks are machine-learning models with several layers of processing units and many adjustable parameters. They are powerful, but they have a serious problem with **overfitting**: they can learn details of the training data that do not hold for new data. Large networks are also slow to use. This makes it difficult to reduce overfitting by combining the predictions of many large networks at test time, when the trained models are used on new cases.

**Dropout** addresses this problem. During training, it randomly removes units and their connections. This prevents excessive **co-adaptation**, in which units depend on particular other units to work correctly. Training with dropout samples from an exponential number of “thinned” networks: networks containing only the units that remain. At test time, a single unthinned network with smaller connection weights can approximate the average predictions of these thinned networks.

This substantially reduces overfitting and gives major improvements over other **regularization** methods, which restrict learning to improve performance on new data. We show improvements on supervised-learning tasks, where training cases have target outputs, in vision, speech recognition, document classification, and computational biology. Dropout obtains state-of-the-art results on many benchmark data sets.

**Keywords:** neural networks, regularization, model combination, deep learning

## 1. Introduction

Deep neural networks contain multiple **non-linear hidden layers**: layers between the input and output whose units apply non-linear functions. These layers let networks learn complicated input–output relationships. With limited training data, however, many learned relationships reflect **sampling noise**, or accidental patterns in the particular sample. These patterns need not occur in test data, even when training and test data come from the same distribution. This causes overfitting.

Several methods reduce overfitting. **Early stopping** ends training when performance on a separate validation set starts to worsen. A validation set is data used to choose training settings rather than to fit the model directly. Other methods penalize weights, the adjustable strengths of connections. Examples include **L1 regularization**, which penalizes absolute weight values; **L2 regularization**, which penalizes squared weight values; and **soft weight sharing**, which encourages weights to take similar values (Nowlan and Hinton, 1992).

With unlimited computation, the best way to regularize a model of fixed size is to average predictions over every possible parameter setting. Each setting is weighted by its **posterior probability**, its probability after observing the training data. This Bayesian approach can sometimes be approximated well for small or simple models (Xiong et al., 2011; Salakhutdinov and Mnih, 2008). We want to approach this Bayesian gold standard with considerably less computation. We propose approximating an equally weighted **geometric mean** of predictions from exponentially many learned models that share parameters. A geometric mean combines values by multiplying them and taking the appropriate root.

© 2014 Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever and Ruslan Salakhutdinov.

**Figure 1 panels:**  
(a) Standard Neural Net  
(b) After applying dropout.

**Figure 1: Dropout Neural Net Model.** Left: A standard neural net with 2 hidden layers. Right: An example of a thinned net produced by applying dropout to the network on the left. Crossed units have been dropped.

Combining models nearly always improves machine-learning performance. But averaging the outputs of many separately trained large networks is prohibitively expensive. Combination helps most when the individual models differ. Neural networks can be made different by changing their architectures—their arrangements of layers and connections—or by training them on different data.

Both choices are difficult. Each architecture requires suitable **hyperparameters**, settings chosen outside the parameter-fitting process. Finding good settings is daunting, and training each large network requires substantial computation. Large networks also usually need large training sets. There may not be enough data to train different networks on different subsets. Even if many large networks could be trained, using all of them at test time would be infeasible when responses must be fast.

Dropout addresses both issues. It reduces overfitting and efficiently approximates a combination of exponentially many network architectures. Dropping a unit means temporarily removing it and all its incoming and outgoing connections, as in Figure 1. This can be done to hidden units and to visible, or input, units.

The choice is random. In the simplest case, each unit is retained independently with probability \(p\). Here, \(p\) is the probability that the unit remains in the network. It can be chosen on a validation set or simply set to 0.5. This value seems close to optimal for many networks and tasks. For input units, the optimal retention probability is usually closer to 1 than to 0.5.

**Figure 2 panels:**  
(a) At training time: a unit is present with probability \(p\); its outgoing weights are \(w\).  
(b) At test time: the unit is always present; its outgoing weights are \(pw\).

**Figure 2:** Left: A unit at training time that is present with probability \(p\) and is connected to units in the next layer with weights \(w\). Right: At test time, the unit is always present and the weights are multiplied by \(p\). The output at test time is same as the expected output at training time.

Applying dropout samples a thinned network containing the surviving units (Figure 1b). A network with \(n\) units can be viewed as a collection of \(2^n\) possible thinned networks. These networks share weights. The total parameter count therefore remains \(O(n^2)\), or less. The notation \(O(n^2)\) means that the count grows no faster than a constant times the square of the number of units.

A new thinned network is sampled for every presentation of every training case. Training with dropout thus trains a collection of \(2^n\) networks with extensive weight sharing. Each individual thinned network is trained very rarely, if at all.

Explicitly averaging exponentially many predictions at test time is infeasible. Instead, we use one network without dropout. Its weights are scaled-down versions of the trained weights. If a unit was retained with probability \(p\), its outgoing weights are multiplied by \(p\) at test time (Figure 2). This ensures that, for any hidden unit, its expected training-time output under dropout is the same as its actual test-time output. This scaling combines \(2^n\) shared-weight networks into one test network.

Across many classification problems, training with dropout and using this approximate test-time average gives substantially lower **generalization error**, or error on new data, than other regularization methods. Section 6 reports the comparisons.

Dropout is not limited to **feed-forward networks**, where computation proceeds from inputs toward outputs without feedback connections. It can also be applied to **graphical models**, which describe probability distributions through relationships among variables, such as Boltzmann Machines. We introduce a dropout **Restricted Boltzmann Machine (RBM)**, a model with visible and hidden units and connections between those two groups. Our experiments show that dropout RBMs improve on standard RBMs in certain respects.

The paper is organized as follows. Section 2 motivates dropout. Section 3 reviews related work. Section 4 defines the model, and Section 5 describes training. Section 6 reports experiments across domains and compares dropout with other regularizers and model-combination methods. Section 7 examines its effects on network properties and hyperparameters. Section 8 introduces dropout RBMs. Section 9 explores **marginalizing dropout**, meaning averaging over its random noise analytically rather than sampling it. Appendix A gives practical guidance for choosing hyperparameters and training dropout networks.

## 2. Motivation

One motivation comes from a theory about the role of sex in evolution (Livnat et al., 2010). Sexual reproduction takes half the genes of each parent, adds a very small amount of random mutation, and combines them into an offspring. Asexual reproduction instead produces a slightly mutated copy of one parent’s genes.

Asexual reproduction might seem better for optimizing individual fitness. A good set of genes that works well together can pass directly to the offspring. Sexual reproduction can break up these co-adapted sets, especially large ones. Intuitively, this should reduce the fitness of organisms that have evolved complicated co-adaptations. Yet most advanced organisms evolved through sexual reproduction.

One possible explanation is that over the long term, the criterion for natural selection may be the **mix-ability** of genes rather than individual fitness. Mix-ability means that a set of genes works well with another randomly selected set. This makes the genes more robust. A gene cannot rely on a large collection of partners always being present. It must do something useful alone or with a small number of other genes.

Under this theory, sexual reproduction does more than spread useful new genes. It also reduces complex co-adaptations that could prevent a new gene from improving an individual’s fitness.

Likewise, each hidden unit trained with dropout must work with a random sample of other units. This should make it more robust. It should learn useful features—patterns represented by the unit—without relying on other units to correct its mistakes. Hidden units in the same layer will still learn different things. A network might seem able to resist dropout by making many copies of each unit. But this is a poor solution, for the same reason that **replica codes**, which repeat information, are a poor way to handle a noisy communication channel.

A related motivation concerns successful conspiracies. Ten conspiracies of five people each are probably a better way to create havoc than one conspiracy requiring fifty people to perform correctly. A large conspiracy can work if conditions remain fixed and there is enough rehearsal time. Under **non-stationary conditions**, where conditions change, smaller conspiracies are more likely to continue working.

Complex co-adaptations can work well on training data. On new test data, however, they are much more likely to fail than several simpler co-adaptations that achieve the same result.

## 3. Related Work

Dropout can be understood as regularization through noise added to hidden-unit states. Vincent et al. (2008, 2010) previously added noise to unit states in **Denoising Autoencoders (DAEs)**. An autoencoder learns to reconstruct its input. A denoising autoencoder receives a noisy input and is trained to reconstruct the noise-free version.

We extend this idea in several ways. Dropout works in hidden layers as well as at the input. It can be interpreted as model averaging. Noise is useful not only for **unsupervised feature learning**, where features are learned without target labels, but also for supervised learning. The method also applies to other neuron-based models, including Boltzmann Machines.

For DAEs, 5% noise typically works best. Our test-time weight scaling lets us use much more noise. Dropping 20% of input units and 50% of hidden units was often optimal.

Because dropout is **stochastic**, meaning random, it is natural to consider a deterministic counterpart obtained by averaging out the noise. In simple cases, we show that dropout can be marginalized analytically to produce deterministic regularizers.

Van der Maaten et al. (2013) explored deterministic regularizers corresponding to different **exponential-family noise distributions**, a class of probability distributions that includes their dropout noise. They call dropout “blankout noise.” They add noise only to inputs and study models without hidden layers. Wang and Manning (2013) proposed speeding up dropout by marginalizing its noise. Chen et al. (2012) studied marginalization for denoising autoencoders.

Dropout minimizes a **loss function**, a numerical measure of prediction error, stochastically under a noise distribution. This can be viewed as minimizing expected loss, the loss averaged over that distribution. Globerson and Roweis (2006) and Dekel et al. (2010) studied a different setup. An adversary chooses which units to drop, subject to a fixed maximum number of deletions. Their work also does not study models with hidden units.

## 4. Model Description

Consider a neural network with \(L\) hidden layers. The index \(l\in\{1,\ldots,L\}\) identifies a hidden layer. Let:

- \(\mathbf z^{(l)}\) be the vector of inputs to the units in layer \(l\), before applying their activation functions.
- \(\mathbf y^{(l)}\) be the vector of outputs from that layer.
- \(\mathbf y^{(0)}=\mathbf x\) be the network input.
- \(\mathbf W^{(l)}\) and \(\mathbf b^{(l)}\) be the weights and biases at layer \(l\). A bias is an additive adjustable offset.
- \(\mathbf w_i^{(l)}\) be the weight vector for unit \(i\) in layer \(l\).
- \(f\) be the activation function that converts a unit’s input into its output.

For a standard network, the feed-forward computation is, for \(l\in\{0,\ldots,L-1\}\) and any hidden unit \(i\),

\[
z_i^{(l+1)}
=
\mathbf w_i^{(l+1)}\mathbf y^{(l)}
+
b_i^{(l+1)},
\]

\[
y_i^{(l+1)}
=
f\!\left(z_i^{(l+1)}\right).
\]

The weight-vector product sums the weighted outputs from the preceding layer. One possible activation function is the **logistic function**,

\[
f(x)=\frac{1}{1+\exp(-x)},
\]

where \(\exp\) is the exponential function.

With dropout, the computation becomes

\[
r_j^{(l)}\sim\operatorname{Bernoulli}(p),
\]

\[
\widetilde{\mathbf y}^{(l)}
=
\mathbf r^{(l)}*\mathbf y^{(l)},
\]

\[
z_i^{(l+1)}
=
\mathbf w_i^{(l+1)}\widetilde{\mathbf y}^{(l)}
+
b_i^{(l+1)},
\]

\[
y_i^{(l+1)}
=
f\!\left(z_i^{(l+1)}\right).
\]

Here, \(\sim\) means “is sampled from.” A Bernoulli variable equals 1 with probability \(p\) and 0 otherwise. The symbol \(*\) denotes element-by-element multiplication.

**Figure 3 panels:**  
(a) Standard network: the outputs \(y_1^{(l)},y_2^{(l)},y_3^{(l)}\) feed into unit \(i\), with weights \(\mathbf w_i^{(l+1)}\), bias \(b_i^{(l+1)}\), input \(z_i^{(l+1)}\), and output \(y_i^{(l+1)}\) after applying \(f\). The bias input is \(+1\).  
(b) Dropout network: each \(y_j^{(l)}\) is multiplied by \(r_j^{(l)}\), producing \(\widetilde y_j^{(l)}\), before the same computation.

**Figure 3: Comparison of the basic operations of a standard and dropout network.**

For each layer \(l\), the entries of \(\mathbf r^{(l)}\) are independent Bernoulli variables. Multiplying this sampled vector by \(\mathbf y^{(l)}\) gives the thinned outputs \(\widetilde{\mathbf y}^{(l)}\). These become the next layer’s inputs. Applying this process at each layer samples a sub-network from the full network.

Learning uses **backpropagation**, which computes loss derivatives by passing them backward through the network. These derivatives are passed only through the sampled sub-network. At test time, the weights are scaled as

\[
\mathbf W_{\text{test}}^{(l)}=p\mathbf W^{(l)},
\]

as in Figure 2. The resulting network is used without dropout.

## 5. Learning Dropout Nets

This section describes how to train dropout neural networks.

### 5.1 Backpropagation

Dropout networks can be trained with **stochastic gradient descent**. This method adjusts parameters using loss derivatives estimated from training cases. A **mini-batch** is the group of training cases used for one update.

For each case in a mini-batch, sample a thinned network. Perform the forward computation and backpropagation only through that network. Average each parameter’s gradients over all cases in the mini-batch. A case that does not use a parameter contributes a zero gradient for that parameter.

Methods that improve ordinary stochastic gradient descent also help dropout networks. These include **momentum**, which carries part of previous updates into the current update; **annealed learning rates**, which decrease the update size during training; and L2 weight decay.

One especially useful regularizer bounds the norm of each hidden unit’s incoming weight vector. If \(\mathbf w\) is that vector, impose

\[
\|\mathbf w\|_2\leq c.
\]

Here, \(\|\mathbf w\|_2\) is the Euclidean, or L2, norm: the square root of the sum of squared entries. The constant \(c\) is an upper bound. Whenever an update moves \(\mathbf w\) outside the ball of radius \(c\), project it back onto the ball’s surface. This is **max-norm regularization**: the maximum norm allowed for any weight vector is \(c\).

Choose \(c\) using a validation set. Max-norm regularization has previously been used in **collaborative filtering**, which predicts preferences from patterns across users and items (Srebro and Shraibman, 2005). It typically improves stochastic-gradient training of deep networks even without dropout.

Dropout alone gives substantial improvements. Combining it with max-norm regularization, large decaying learning rates, and high momentum gives a further substantial boost. One possible explanation is that a fixed-radius bound allows very large learning rates without weights blowing up. Dropout noise then lets optimization explore regions of weight space that would otherwise be difficult to reach. As the learning rate decreases, the steps become shorter. Exploration declines, and optimization eventually settles into a minimum.

### 5.2 Unsupervised Pretraining

**Pretraining** initializes a network by first training it on another learning objective. Neural networks can be pretrained with stacks of RBMs (Hinton and Salakhutdinov, 2006), autoencoders (Vincent et al., 2010), or **Deep Boltzmann Machines (DBMs)**, Boltzmann Machines with multiple hidden layers (Salakhutdinov and Hinton, 2009).

Pretraining makes effective use of unlabeled data. In certain cases, pretraining followed by **finetuning**, or further supervised training with backpropagation, substantially outperforms finetuning from random initial weights.

Dropout can be used during finetuning without changing pretraining. Scale pretrained weights up by \(1/p\). This makes each unit’s expected output under dropout match its output during pretraining.

We initially worried that dropout’s randomness might erase information in the pretrained weights. This did happen when finetuning used learning rates comparable to the best rates for randomly initialized networks. Smaller learning rates seemed to preserve that information. With these rates, dropout improved final generalization error relative to finetuning without dropout.

## 6. Experimental Results

We trained dropout networks for classification, the task of assigning inputs to categories, across several domains. On every data set, dropout improved generalization relative to networks without dropout. The comparisons appear below; Table 1 summarizes the data sets.

- **MNIST:** a standard toy data set of handwritten digits.
- **TIMIT:** a standard clean-speech recognition benchmark.
- **CIFAR-10 and CIFAR-100:** data sets of tiny natural images (Krizhevsky, 2009).
- **Street View House Numbers (SVHN):** images of house numbers collected by Google Street View (Netzer et al., 2011).
- **ImageNet:** a large collection of natural images.
- **Reuters-RCV1:** a collection of Reuters newswire articles.
- **Alternative Splicing:** RNA features used to predict alternative gene splicing (Xiong et al., 2011). RNA is ribonucleic acid. Alternative splicing produces different outcomes from the processing of a gene’s RNA.

We chose diverse data sets to show that dropout is a general neural-network technique rather than a method for one domain. This section presents key results. Appendix B gives detailed data and experiment descriptions.

In Table 1, **dimensionality** is the number of input components. “K” means thousand, “M” means million, and “dim” means dimensions. A speech **frame** is a short segment represented by acoustic features. ILSVRC is the ImageNet Large-Scale Visual Recognition Challenge, described in Section 6.1.4.

| Data Set | Domain | Dimensionality | Training Set | Test Set |
|---|---|---|---|---|
| MNIST | Vision | 784 (28 × 28 grayscale) | 60K | 10K |
| SVHN | Vision | 3072 (32 × 32 color) | 600K | 26K |
| CIFAR-10/100 | Vision | 3072 (32 × 32 color) | 60K | 10K |
| ImageNet (ILSVRC-2012) | Vision | 65536 (256 × 256 color) | 1.2M | 150K |
| TIMIT | Speech | 2520 (120-dim, 21 frames) | 1.1M frames | 58K frames |
| Reuters-RCV1 | Text | 2000 | 200K | 200K |
| Alternative Splicing | Genetics | 1014 | 2932 | 733 |

**Table 1: Overview of the data sets used in this paper.**

### 6.1 Results on Image Data Sets

We evaluated dropout on MNIST, SVHN, CIFAR-10, CIFAR-100, and ImageNet. These cover different image types and training-set sizes. Models achieving state-of-the-art results on all these data sets use dropout.

#### 6.1.1 MNIST

Table 2 uses the following terms. **NN** means neural network. A **support vector machine (SVM)** is a classifier based on a separating decision boundary; a Gaussian kernel supplies a similarity function for that classifier. **NA** means not applicable. A **rectified linear unit (ReLU)** uses the activation function \(\max(0,x)\). **Maxout** is the unit type used by Goodfellow et al. (2013), based on taking a maximum over learned responses. A **Deep Belief Network (DBN)** is the model used here for pretraining through stacks of RBMs.

| Method | Unit Type | Architecture | Error % |
|---|---|---|---|
| Standard Neural Net (Simard et al., 2003) | Logistic | 2 layers, 800 units | 1.60 |
| SVM Gaussian kernel | NA | NA | 1.40 |
| Dropout NN | Logistic | 3 layers, 1024 units | 1.35 |
| Dropout NN | ReLU | 3 layers, 1024 units | 1.25 |
| Dropout NN + max-norm constraint | ReLU | 3 layers, 1024 units | 1.06 |
| Dropout NN + max-norm constraint | ReLU | 3 layers, 2048 units | 1.04 |
| Dropout NN + max-norm constraint | ReLU | 2 layers, 4096 units | 1.01 |
| Dropout NN + max-norm constraint | ReLU | 2 layers, 8192 units | 0.95 |
| Dropout NN + max-norm constraint (Goodfellow et al., 2013) | Maxout | 2 layers, (5 × 240) units | 0.94 |
| DBN + finetuning (Hinton and Salakhutdinov, 2006) | Logistic | 500-500-2000 | 1.18 |
| DBM + finetuning (Salakhutdinov and Hinton, 2009) | Logistic | 500-500-2000 | 0.96 |
| DBN + dropout finetuning | Logistic | 500-500-2000 | 0.92 |
| DBM + dropout finetuning | Logistic | 500-500-2000 | 0.79 |

**Table 2: Comparison of different models on MNIST.**

MNIST contains 28 × 28 pixel images of handwritten digits. The task is to assign each image to one of 10 digit classes.

We first consider the **permutation invariant setting**, in which the model does not use the two-dimensional spatial arrangement of pixels. The best networks without dropout or unsupervised pretraining achieve about 1.60% error (Table 2, “Standard Neural Net”; Simard et al., 2003). Dropout reduces this to 1.35% (Table 2, logistic “Dropout NN”). Replacing logistic units with ReLUs (Jarrett et al., 2009) reduces it to 1.25%. Adding max-norm regularization reduces it to 1.06% (Table 2, corresponding 3-layer, 1024-unit rows).

Larger networks give better results. A network with 2 hidden layers and 8192 units per layer reaches 0.95% error (Table 2). This network has more than 65 million parameters and is trained on 60,000 cases. Obtaining good generalization from this size of network is very hard with standard regularizers and early stopping. Dropout prevents overfitting even here and does not need early stopping.

Goodfellow et al. (2013) further improve error to 0.94% with maxout units (Table 2, maxout row). All dropout networks use \(p=0.5\) for hidden units and \(p=0.8\) for input units. Appendix B.1 gives further details.

Dropout also improves networks pretrained with RBM stacks or DBMs. DBN finetuning improves from 1.18% to 0.92%, and DBM finetuning improves from 0.96% to 0.79% (Table 2, last four rows). The 0.79% test error is the best reported performance for the permutation invariant setting.

Better results are possible by using 2-D spatial information and augmenting the training set with distorted images. **Data augmentation** means adding modified versions of training cases. We demonstrate dropout in that setting on more interesting data sets.

To test robustness, we trained many architectures while keeping all hyperparameters, including \(p\), fixed. Figure 4 shows test errors during training. The trajectories form two distinct groups for training with and without dropout. Dropout gives a large improvement across architectures without architecture-specific hyperparameter tuning.

**Figure 4 plot labels:** Vertical axis: Classification Error %, with marks 1.0, 1.5, 2.0, 2.5. Horizontal axis: Number of weight updates, with marks 0, 200000, 400000, 600000, 800000, 1000000. Curves are labeled “Without dropout” and “With dropout.”

**Figure 4: Test error for different architectures with and without dropout. The networks have 2 to 4 hidden layers each with 1024 to 2048 units.**

#### 6.1.2 Street View House Numbers

SVHN contains color images of house numbers collected by Google Street View (Netzer et al., 2011). Figure 5a shows examples. We use 32 × 32 color images roughly centered on one digit in a house number. The task is to identify that digit.

We applied dropout to **convolutional neural networks** (LeCun et al., 1989). These use learned filters across image locations. The best architecture we found has three convolutional layers followed by 2 fully connected hidden layers. A fully connected layer connects each unit to all units in the preceding layer. All hidden units were ReLUs.

Each convolutional layer was followed by **max-pooling**, which takes the maximum response within a local region. Appendix B.2 gives the architecture in detail.

Table 3 also includes the following method labels. **WDCH** is the binary-feature method label used by Netzer et al. (2011). **HOG** denotes histograms of oriented gradients, an image-feature representation. **KMeans** is a method that groups inputs around learned cluster centers. Sparse autoencoders encourage most hidden responses to be small or zero. Average pooling averages local responses; L2 and L4 pooling use the corresponding norm-based summaries. Padding adds a border for image processing. Stochastic pooling selects pooled responses randomly.

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
| Conv Net + max pooling + dropout in fully connected layers | 3.02 |
| Conv Net + stochastic pooling (Zeiler and Fergus, 2013) | 2.80 |
| Conv Net + max pooling + dropout in all layers | 2.55 |
| Conv Net + maxout (Goodfellow et al., 2013) | 2.47 |
| Human Performance | 2.0 |

**Table 3: Results on the Street View House Numbers data set.**

Dropout was applied throughout the network. The retention probabilities, ordered from the input through the convolutional and fully connected layers, were

\[
p=(0.9,\ 0.75,\ 0.75,\ 0.5,\ 0.5,\ 0.5).
\]

Max-norm regularization was used for both convolutional and fully connected weights.

Convolutional networks outperform other methods (Table 3). The best-performing convolutional networks without dropout achieve 3.95% error. Adding dropout only in fully connected layers reduces error to 3.02%. Adding it in convolutional layers as well reduces error to 2.55% (Table 3, corresponding max-pooling rows). Maxout units give further gains, reaching 2.47% (Table 3, maxout row).

The gain from adding convolutional-layer dropout, from 3.02% to 2.55%, is worth noting. One might expect little benefit because convolutional layers have relatively few parameters and might not overfit. But dropout in lower layers provides noisy inputs to the higher fully connected layers. This prevents the higher layers from overfitting.

#### 6.1.3 CIFAR-10 and CIFAR-100

CIFAR-10 and CIFAR-100 contain 32 × 32 color images from 10 and 100 categories, respectively. Figure 5b shows examples. Appendix B.3 describes preprocessing, architectures, and training.

Without data augmentation, Bayesian hyperparameter optimization—a method that uses a probabilistic model to choose hyperparameters—gave Snoek et al. (2012) an error of 14.98% on CIFAR-10. Dropout in fully connected layers reduces error to 14.32%. Dropout in every layer reduces it to 12.61% (Table 4, corresponding max-pooling rows). Replacing ReLUs with maxout units reduces error further to 11.68% (Goodfellow et al., 2013; Table 4, maxout row).

On CIFAR-100, dropout reduces error from 43.48% to 37.20%, a large improvement (Table 4, hand-tuned max-pooling and all-layer-dropout rows). Neither data set used data augmentation apart from input dropout.

**Figure 5 panels:**  
(a) Street View House Numbers (SVHN)  
(b) CIFAR-10

**Figure 5: Samples from image data sets. Each row corresponds to a different category.**

| Method | CIFAR-10 | CIFAR-100 |
|---|---|---|
| Conv Net + max pooling (hand tuned) | 15.60 | 43.48 |
| Conv Net + stochastic pooling (Zeiler and Fergus, 2013) | 15.13 | 42.51 |
| Conv Net + max pooling (Snoek et al., 2012) | 14.98 | - |
| Conv Net + max pooling + dropout fully connected layers | 14.32 | 41.26 |
| Conv Net + max pooling + dropout in all layers | 12.61 | 37.20 |
| Conv Net + maxout (Goodfellow et al., 2013) | 11.68 | 38.57 |

**Table 4: Error rates on CIFAR-10 and CIFAR-100.**

#### 6.1.4 ImageNet

ImageNet contains over 15 million labeled high-resolution images in roughly 22,000 categories. Starting in 2010, an annual ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) has been held as part of the Pascal Visual Object Challenge. It uses an ImageNet subset with roughly 1000 images in each of 1000 categories.

Because there are many categories, results conventionally report **top-1 error**, where the highest-probability label is wrong, and **top-5 error**, where the correct label is absent from the five highest-probability labels. Figure 6 shows model predictions on several test images.

ILSVRC-2010 is the only ILSVRC version with available test labels. Most of our experiments therefore use it. Table 5 shows that convolutional networks with dropout outperform other methods by a large margin: top-1/top-5 errors are 37.5/17.0, compared with 47.1/28.2 for Sparse Coding and 45.7/25.7 for SIFT + Fisher Vectors. Krizhevsky et al. (2012) provide architecture and implementation details.

**Figure 6: Some ImageNet test cases with the 4 most probable labels as predicted by our model. The length of the horizontal bars is proportional to the probability assigned to the labels by the model. Pink indicates ground truth.**

In Tables 5 and 6, **sparse coding** represents inputs using a small number of active components. **SIFT**, or scale-invariant feature transform, is an image-feature method. **Fisher Vectors (FVs)** are feature representations used by the cited classifiers. **LBP** means local binary patterns; **GIST** is a global image-descriptor method; and **CSIFT** is a color SIFT descriptor. “Dense” indicates features sampled across image locations. “Avg” means average, and “val” means validation.

| Model | Top-1 | Top-5 |
|---|---|---|
| Sparse Coding (Lin et al., 2010) | 47.1 | 28.2 |
| SIFT + Fisher Vectors (Sanchez and Perronnin, 2011) | 45.7 | 25.7 |
| Conv Net + dropout (Krizhevsky et al., 2012) | 37.5 | 17.0 |

**Table 5: Results on the ILSVRC-2010 test set.**

| Model | Top-1 (val) | Top-5 (val) | Top-5 (test) |
|---|---|---|---|
| SVM on Fisher Vectors of Dense SIFT and Color Statistics | - | - | 27.3 |
| Avg of classifiers over FVs of SIFT, LBP, GIST and CSIFT | - | - | 26.2 |
| Conv Net + dropout (Krizhevsky et al., 2012) | 40.7 | 18.2 | - |
| Avg of 5 Conv Nets + dropout (Krizhevsky et al., 2012) | 38.1 | 16.4 | 16.4 |

**Table 6: Results on the ILSVRC-2012 validation/test set.**

Our convolutional-network model with dropout won ILSVRC-2012. Test labels are unavailable, so Table 6 reports test results for the final submission and validation results for model variations.

The best standard-vision-feature methods obtain about 26% top-5 test error: Table 6 reports 27.3 and 26.2. Convolutional networks with dropout obtain about 16%: the average of 5 networks gives 16.4% test error. This is a staggering difference. Figure 6 shows that the model makes very reasonable predictions even when its best guess is wrong.

### 6.2 Results on TIMIT

We next applied dropout to speech recognition. TIMIT contains recordings of 680 speakers from 8 major American English dialects. Each speaker reads ten phonetically rich sentences in a controlled, noise-free environment.

Networks received windows of 21 **log-filter bank frames**, acoustic measurements represented on a logarithmic scale, and predicted the label of the central frame. No speaker-dependent operations were performed. Appendix B.4 gives preprocessing and training details.

A 6-layer network gives a **phone error rate**, the speech-recognition error measure for phone units, of 23.4%. Dropout reduces this to 21.8% (Table 7, first two rows).

We also initialized dropout networks with pretrained weights. A 4-layer network pretrained with RBMs gives 22.7% phone error, reduced to 19.7% with dropout (Table 7, corresponding 4-layer rows). Similarly, for an 8-layer network, error falls from 20.5% to 19.7%.

In Table 7, **mcRBM** denotes the mean-covariance RBM model used by Dahl et al. (2010).

| Method | Phone Error Rate% |
|---|---|
| NN (6 layers) (Mohamed et al., 2010) | 23.4 |
| Dropout NN (6 layers) | 21.8 |
| DBN-pretrained NN (4 layers) | 22.7 |
| DBN-pretrained NN (6 layers) (Mohamed et al., 2010) | 22.4 |
| DBN-pretrained NN (8 layers) (Mohamed et al., 2010) | 20.7 |
| mcRBM-DBN-pretrained NN (5 layers) (Dahl et al., 2010) | 20.5 |
| DBN-pretrained NN (4 layers) + dropout | 19.7 |
| DBN-pretrained NN (8 layers) + dropout | 19.7 |

**Table 7: Phone error rate on the TIMIT core test set.**

### 6.3 Results on a Text Data Set

We tested dropout on document classification using a subset of Reuters-RCV1. The full data set contains over 800,000 Reuters newswire articles on varied topics.

The task uses a **bag-of-words representation**, which represents a document by its words without their order, and assigns the document to one of 50 disjoint topics. Appendix B.5 gives details.

Our best network without dropout gives 31.05% error. Dropout reduces this to 29.62%. The improvement is much smaller than on the vision and speech data sets.

### 6.4 Comparison with Bayesian Neural Networks

Dropout can be viewed as equally weighted averaging over exponentially many models with shared weights. **Bayesian neural networks** (Neal, 1996) provide the proper way to average over network structures and parameters. They weight models using both a **prior**, the distribution before observing the data, and how well each model fits the data. This is more correct than weighting every model equally.

Bayesian neural networks are especially useful when data is scarce, as in medical diagnosis, genetics, drug discovery, and other computational-biology applications. But they are slow to train and difficult to scale to very large networks. Predicting with many large networks is also expensive. Dropout networks are much faster to train and use.

We compare the methods on a small data set where Bayesian neural networks are known to perform well and obtain state-of-the-art results. The aim is to measure how much dropout loses relative to Bayesian neural networks.

The data set comes from genetics (Xiong et al., 2011). The task predicts alternative splicing from RNA features. Alternative splicing is a significant source of cellular diversity in mammalian tissues. Predicting it across tissues and conditions matters for understanding many human diseases. Given RNA features, the model predicts probabilities of three biologically important splicing events.

The metric is **Code Quality**, measured in bits. It measures negative **Kullback–Leibler (KL) divergence**, a measure of discrepancy between target and predicted probability distributions. Higher Code Quality is better. Appendix B.6 defines the metric and data in detail.

**Principal component analysis (PCA)**, used in Table 8, reduces input dimensionality by representing data along principal directions of variation.

| Method | Code Quality (bits) |
|---|---|
| Neural Network (early stopping) (Xiong et al., 2011) | 440 |
| Regression, PCA (Xiong et al., 2011) | 463 |
| SVM, PCA (Xiong et al., 2011) | 487 |
| Neural Network with dropout | 567 |
| Bayesian Neural Network (Xiong et al., 2011) | 623 |

**Table 8: Results on the Alternative Splicing Data Set.**

As expected, Bayesian neural networks outperform dropout: Code Quality is 623 versus 567 (Table 8). But dropout substantially improves on standard networks, which score 440, and beats all other listed methods, which score 463 and 487.

The small training set makes overfitting the central challenge. One approach is to reduce dimensionality with PCA and then use SVMs or **logistic regression**, a probabilistic classifier based on a logistic response. Dropout prevents overfitting without dimensionality reduction.

The dropout networks have 1000s of hidden units, compared with a few tens in the Bayesian network. This shows dropout’s strong regularizing effect.

### 6.5 Comparison with Standard Regularizers

Common regularizers include L2 weight decay, more generally **Tikhonov regularization** (Tikhonov, 1943); **lasso**, an L1 penalty (Tibshirani, 1996); **KL-sparsity**, a KL-divergence-based penalty encouraging a target level of sparse activation; and max-norm regularization. We compare these with dropout on MNIST.

Every method used the same ReLU architecture, 784-1024-1024-2048-10, trained with stochastic gradient descent. The numbers give successive layer sizes: input, hidden layers, and output.

A validation set selected each regularizer’s hyperparameters: decay constants, target sparsity, dropout rate, or max-norm bound. Dropout with max-norm gives the lowest generalization error, 1.05% (Table 9, “Dropout + Max-norm”). The complete comparison is in Table 9.

## 7. Salient Features

Section 6 provides strong evidence that dropout improves neural networks. Here we examine its effects on learned features and on **activation sparsity**, where few units have large outputs. We also vary retention probability, network size, and training-set size. These observations give insight into why dropout works.

| Method | Test Classification error % |
|---|---|
| L2 | 1.62 |
| L2 + L1 applied towards the end of training | 1.60 |
| L2 + KL-sparsity | 1.55 |
| Max-norm | 1.35 |
| Dropout + L2 | 1.25 |
| Dropout + Max-norm | 1.05 |

**Table 9: Comparison of different regularization methods on MNIST.**

### 7.1 Effect on Features

**Figure 7 panels:**  
(a) Without dropout  
(b) Dropout with \(p=0.5\).

**Figure 7: Features learned on MNIST with one hidden layer autoencoders having 256 rectified linear units.**

In a standard network, the derivative for each parameter says how to reduce the final loss given what all other units are currently doing. A unit can therefore change to correct other units’ mistakes. This may create complex co-adaptations. Those co-adaptations cause overfitting when they fail on unseen data.

We hypothesize that dropout prevents this by making other hidden units’ presence unreliable. A unit cannot depend on particular other units to correct its mistakes. It must work in many contexts supplied by different subsets of units.

To examine this directly, we compare first-level visual features learned with and without dropout. Figure 7a shows a single-hidden-layer MNIST autoencoder with 256 ReLUs and no dropout. Figure 7b shows an identical autoencoder using hidden-layer dropout with \(p=0.5\).

The two models have similar test reconstruction errors. But the features without dropout appear co-adapted for reconstruction: individual units do not seem to detect meaningful features. With dropout, units appear to detect edges, strokes, and spots at different image locations (Figure 7b). This shows that dropout breaks up co-adaptations, probably the main reason for its lower generalization errors.

### 7.2 Effect on Sparsity

**Figure 8 panels:**  
(a) Without dropout  
(b) Dropout with \(p=0.5\).

**Figure 8: Effect of dropout on sparsity.** ReLUs were used for both models. Left: The histogram of mean activations shows that most units have a mean activation of about 2.0. The histogram of activations shows a huge mode away from zero. Clearly, a large fraction of units have high activation. Right: The histogram of mean activations shows that most units have a smaller mean activation of about 0.7. The histogram of activations shows a sharp peak at zero. Very few units have high activation.

Dropout makes hidden activations sparse even without a sparsity-inducing regularizer. It therefore automatically produces sparse representations.

We examine the autoencoders from Section 7.1 on a random test mini-batch. A good sparse model should have only a few highly active units for each case. Each unit should also have a low average activation across cases.

Figure 8 uses two histograms per model. A **histogram** groups values into ranges and displays their frequencies. The left histogram for each model gives the distribution of units’ mean activations across the mini-batch. The right gives the distribution of individual hidden-unit activations.

With dropout, fewer activations are high (Figure 8b). Without dropout, substantial probability mass lies away from zero (Figure 8a). Mean activations are also lower with dropout. Overall mean hidden activation is close to 2.0 without dropout and around 0.7 with dropout (Figure 8).

### 7.3 Effect of Dropout Rate

We vary the retention probability \(p\) in two setups:

1. Keep the number of hidden units fixed.
2. Change the number of hidden units so that the expected number retained remains fixed.

In the first setup, every network has architecture 784-2048-2048-2048-10. There is no input dropout. Figure 9a shows test error against \(p\).

With a fixed architecture, small \(p\) leaves very few units active during training. Both training and test error are high, indicating **underfitting**, where the model does not fit even the training data well. Error falls as \(p\) increases. It is flat for \(0.4\leq p\leq0.8\), then rises as \(p\) approaches 1.

**Figure 9 plot labels:** Both panels show Classification Error % against Probability of retaining a unit (\(p\)). Both contain “Test Error” and “Training Error.” Horizontal marks are 0.0, 0.2, 0.4, 0.6, 0.8, 1.0.  
(a) Keeping \(n\) fixed. Vertical marks are 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5.  
(b) Keeping \(pn\) fixed. Vertical marks are 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0.

**Figure 9: Effect of changing dropout rates on MNIST.**

In the second setup, \(pn\) stays fixed, where \(n\) is the number of hidden units in a layer. Small \(p\) therefore requires larger layers. After dropout, every architecture has the same expected number of surviving units, although the test networks differ in size.

We set \(pn=256\) for the first two hidden layers and \(pn=512\) for the last. Errors at small \(p\) are much lower than in the fixed-size setup. For \(p=0.1\), error falls from 2.7% to 1.7% (Figure 9a versus Figure 9b). Values near 0.6 work best for this choice of \(pn\), but our usual default, 0.5, is close to optimal.

### 7.4 Effect of Data Set Size

A good regularizer should allow a model with many parameters to generalize from a small data set. Large feed-forward networks trained conventionally overfit severely on small data sets. We therefore vary the amount of MNIST training data.

Each experiment uses architecture 784-1024-1024-2048-10. Training subsets contain 100, 500, 1K, 5K, 10K, or 50K cases randomly selected from MNIST. Dropout uses \(p=0.5\) in every hidden layer and \(p=0.8\) at the input.

Figure 10 shows no improvement from dropout on the smallest sets, 100 and 500 cases. The model still has enough parameters to overfit despite dropout noise. As data size grows, dropout’s gain first increases and then declines.

This suggests a “sweet spot” for a given architecture and dropout rate. The data set must be large enough that the model cannot memorize it despite the noise, but not so large that overfitting ceases to be a problem.

**Figure 10 plot labels:** Vertical axis: Classification Error %, with marks 0, 5, 10, 15, 20, 25, 30. Horizontal axis: Dataset size, with marks \(10^2,10^3,10^4,10^5\). Curves are labeled “With dropout” and “Without dropout.”

**Figure 10: Effect of varying data set size.**

### 7.5 Monte-Carlo Model Averaging vs. Weight Scaling

Our efficient test procedure approximates model combination by scaling down weights. A more expensive but more correct method samples \(k\) dropout networks for each test case and averages their predictions. Here, \(k\) is the number of sampled networks.

This is **Monte-Carlo model averaging**, meaning that random samples approximate the full average. As \(k\to\infty\), the sample average approaches the true model average. We ask how many samples are needed to match weight scaling and how quickly error approaches that of the true average.

**Figure 11 plot labels:** Vertical axis: Test Classification error %, with marks 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35. Horizontal axis: Number of samples used for Monte-Carlo averaging (\(k\)), with marks 0, 20, 40, 60, 80, 100, 120. Curves are labeled “Monte-Carlo Model Averaging” and “Approximate averaging by weight scaling.”

**Figure 11: Monte-Carlo model averaging vs. weight scaling.**

On MNIST, we classify by averaging predictions from \(k\) randomly sampled networks. Figure 11 compares test error with the weight-scaling error, shown as a horizontal line.

At around \(k=50\), Monte-Carlo averaging matches the approximate method. Beyond that, it is slightly better, but the difference remains well within one standard deviation, a measure of variability. This suggests that weight scaling approximates the true model average fairly well.

## 8. Dropout Restricted Boltzmann Machines

Dropout also applies to RBMs. This section defines dropout RBMs and illustrates their main properties.

### 8.1 Model Description

Consider an RBM with visible vector \(\mathbf v\in\{0,1\}^D\) and hidden vector \(\mathbf h\in\{0,1\}^F\). Thus, it has \(D\) binary visible units and \(F\) binary hidden units. It defines

\[
P(\mathbf h,\mathbf v;\theta)
=
\frac{1}{Z(\theta)}
\exp\!\left(
\mathbf v^\top\mathbf W\mathbf h
+
\mathbf a^\top\mathbf h
+
\mathbf b^\top\mathbf v
\right).
\]

Here, \(P\) denotes probability; \(\theta=\{\mathbf W,\mathbf a,\mathbf b\}\) is the parameter set; and \({}^\top\) denotes transpose, which changes a column vector into a row vector. The matrix \(\mathbf W\) contains connection weights. In this joint expression, \(\mathbf a\) and \(\mathbf b\) supply the hidden and visible bias terms. The **partition function** \(Z(\theta)\) normalizes the probabilities so that they sum to 1.

A dropout RBM adds a binary vector \(\mathbf r\in\{0,1\}^F\). Each \(r_j\) independently equals 1 with probability \(p\). If \(r_j=1\), hidden unit \(h_j\) is retained. Otherwise it is dropped.

The joint distribution is

\[
P(\mathbf r,\mathbf h,\mathbf v;p,\theta)
=
P(\mathbf r;p)P(\mathbf h,\mathbf v\mid\mathbf r;\theta),
\]

\[
P(\mathbf r;p)
=
\prod_{j=1}^{F}p^{r_j}(1-p)^{1-r_j},
\]

\[
P(\mathbf h,\mathbf v\mid\mathbf r;\theta)
=
\frac{1}{Z'(\theta,\mathbf r)}
\exp\!\left(
\mathbf v^\top\mathbf W\mathbf h+
\mathbf a^\top\mathbf h+
\mathbf b^\top\mathbf v
\right)
\prod_{j=1}^{F}g(h_j,r_j),
\]

\[
g(h_j,r_j)
=
\mathbf 1(r_j=1)
+
\mathbf 1(r_j=0)\mathbf 1(h_j=0).
\]

The product symbol \(\prod\) means multiply over the indexed units. The vertical bar in a probability means “conditioned on.” The indicator \(\mathbf 1(A)\) equals 1 when statement \(A\) is true and 0 otherwise.

The constant \(Z'(\theta,\mathbf r)\) normalizes the conditional distribution. The factor \(g\) requires \(h_j=0\) whenever \(r_j=0\).

Conditioned on \(\mathbf v\) and \(\mathbf r\), hidden units are **factorial**, meaning that their joint distribution factors into independent unit distributions:

\[
P(\mathbf h\mid\mathbf r,\mathbf v)
=
\prod_{j=1}^{F}P(h_j\mid r_j,\mathbf v),
\]

\[
P(h_j=1\mid r_j,\mathbf v)
=
\mathbf 1(r_j=1)
\sigma\!\left(
b_j+\sum_i W_{ij}v_i
\right).
\]

Here, \(\sum\) means sum over the indexed units, and \(\sigma(x)=1/(1+\exp(-x))\) is the logistic function. The entry \(W_{ij}\) connects visible unit \(i\) to hidden unit \(j\).

**Figure 12 panels:**  
(a) Without dropout  
(b) Dropout with \(p=0.5\).

**Figure 12: Features learned on MNIST by 256 hidden unit RBMs. The features are ordered by L2 norm.**

The visible distribution conditioned on \(\mathbf h\) is the same as in an RBM:

\[
P(\mathbf v\mid\mathbf h)
=
\prod_{i=1}^{D}P(v_i\mid\mathbf h),
\]

\[
P(v_i=1\mid\mathbf h)
=
\sigma\!\left(
a_i+\sum_j W_{ij}h_j
\right).
\]

Conditioned on \(\mathbf r\), the distribution over \(\{\mathbf v,\mathbf h\}\) is an RBM distribution with the units having \(r_j=0\) removed. A dropout RBM is therefore a **mixture**, or probabilistic combination, of exponentially many RBMs. They share weights and use different hidden-unit subsets.

### 8.2 Learning Dropout RBMs

RBM learning algorithms, including **Contrastive Divergence** (Hinton et al., 2006), apply directly. Contrastive Divergence is the RBM training method used here to approximate the learning update.

First sample \(\mathbf r\). Then train using only retained hidden units. As in dropout feed-forward networks, sample a different \(\mathbf r\) for every case in every mini-batch.

Our experiments use **CD-1**, Contrastive Divergence with one sampling step.

### 8.3 Effect on Features

In feed-forward networks, dropout improved features by reducing co-adaptations. We test whether this also happens in RBMs.

Figure 12a shows features from a binary RBM with 256 hidden units. Figure 12b shows a dropout RBM with the same number. Dropout features appear qualitatively different. They seem coarser than the sharply defined, stroke-like features of the standard RBM. The dropout RBM also appears to have very few **dead units**, units that do not provide active features, relative to the standard RBM.

### 8.4 Effect on Sparsity

**Figure 13 panels:**  
(a) Without dropout  
(b) Dropout with \(p=0.5\).

**Figure 13: Effect of dropout on sparsity.** Left: The activation histogram shows that a large number of units have activations away from zero. Right: A large number of units have activations close to zero and very few units have high activation.

We next examine hidden-unit sparsity. Figure 13a shows activation histograms and unit-mean histograms on a test mini-batch after ordinary RBM training. Figure 13b shows the corresponding dropout RBM histograms.

The dropout RBMs learn much sparser representations, even without any additional sparsity-inducing regularizer (Figure 13).

## 9. Marginalizing Dropout

Dropout adds noise to hidden-unit states. Here we explore models obtained by marginalizing, or averaging out, this noise. These are deterministic versions of dropout. Unlike standard Monte-Carlo dropout, they do not need random bits. Their marginalized loss functions can be differentiated directly.

Globerson and Roweis (2006) proposed deterministic methods for robustness to feature deletion at test time. Chen et al. (2012) explored marginalization for denoising autoencoders. Srivastava (2013) discussed marginalizing dropout noise for linear regression. Wang and Manning (2013) explored it to speed training. Van der Maaten et al. (2013) studied regularizers obtained by marginalizing different input-noise distributions. Wager et al. (2013) describe dropout as an **adaptive regularizer**, one whose penalty depends on the data.

### 9.1 Linear Regression

Consider **linear regression**, which predicts a numerical target using a weighted sum of inputs. Let \(\mathbf X\in\mathbb R^{N\times D}\) contain \(N\) data points with \(D\) input dimensions. The notation \(\mathbb R\) denotes real-valued entries. Let \(\mathbf y\in\mathbb R^N\) contain the targets. We seek weights \(\mathbf w\in\mathbb R^D\) minimizing

\[
\|\mathbf y-\mathbf X\mathbf w\|^2.
\]

Here, the squared norm sums squared prediction errors.

Retain each input dimension with probability \(p\). The dropped-out input is \(\mathbf R*\mathbf X\), where \(\mathbf R\in\{0,1\}^{N\times D}\), each \(R_{ij}\sim\operatorname{Bernoulli}(p)\), and \(*\) is element-by-element multiplication.

Averaging over the noise gives

\[
\underset{\mathbf w}{\operatorname{minimize}}
\quad
\mathbb E_{\mathbf R\sim\operatorname{Bernoulli}(p)}
\left[
\|\mathbf y-(\mathbf R*\mathbf X)\mathbf w\|^2
\right].
\]

The symbol \(\mathbb E\) denotes expectation, or an average over the stated random distribution.

This reduces to

\[
\underset{\mathbf w}{\operatorname{minimize}}
\quad
\|\mathbf y-p\mathbf X\mathbf w\|^2
+
p(1-p)\|\mathbf\Gamma\mathbf w\|^2,
\]

where

\[
\mathbf\Gamma
=
\left(\operatorname{diag}(\mathbf X^\top\mathbf X)\right)^{1/2}.
\]

Here, \(\operatorname{diag}\) keeps the diagonal entries as a diagonal matrix, and the power \(1/2\) takes their square roots. The objective consists of prediction error plus a weighted quadratic penalty.

Thus, in expectation, linear regression with dropout equals **ridge regression**, linear regression with a quadratic weight penalty, using this particular \(\mathbf\Gamma\). This form essentially scales the cost of weight \(w_i\) by the standard deviation of input dimension \(i\). If a dimension varies greatly, the regularizer squeezes its weight more.

Alternatively, absorb \(p\) into the weights. Define \(\widetilde{\mathbf w}=p\mathbf w\). The objective becomes

\[
\underset{\mathbf w}{\operatorname{minimize}}
\quad
\|\mathbf y-\mathbf X\widetilde{\mathbf w}\|^2
+
\frac{1-p}{p}
\|\mathbf\Gamma\widetilde{\mathbf w}\|^2.
\]

This exposes the dependence of the regularization constant on \(p\). When \(p\) is close to 1, nearly all inputs remain and the constant is small. Decreasing \(p\), and therefore applying more dropout, increases it.

### 9.2 Logistic Regression and Deep Networks

For logistic regression and deep networks, obtaining an exact **closed-form** marginalized model—an explicit analytical expression—is difficult.

Wang and Manning (2013) show that the marginalized logistic-regression model can be trained approximately. Under reasonable assumptions, both the inputs to the logistic unit and the gradients of the marginalized model have **Gaussian**, or normal, distributions. Their means and variances can be computed efficiently. Mean describes the average; variance describes spread around that average.

This approximate marginalization outperforms Monte-Carlo dropout in training time and generalization. But the technique’s assumptions become successively weaker as layers are added. The results therefore do not apply directly to deep networks.

| Data Set | Architecture | Bernoulli dropout | Gaussian dropout |
|---|---|---|---|
| MNIST | 2 layers, 1024 units each | 1.08 ± 0.04 | 0.95 ± 0.04 |
| CIFAR-10 | 3 conv + 2 fully connected layers | 12.6 ± 0.1 | 12.5 ± 0.1 |

**Table 10: Comparison of classification error % with Bernoulli and Gaussian dropout.** For MNIST, the Bernoulli model uses \(p=0.5\) for the hidden units and \(p=0.8\) for the input units. For CIFAR-10, we use \(p=(0.9,0.75,0.75,0.5,0.5,0.5)\), going from the input layer to the top. The value of \(\sigma\), the standard deviation used for Gaussian dropout, was set to \(\sqrt{(1-p)/p}\). Results were averaged over 10 different random seeds, which determine the random choices in each run.

## 10. Multiplicative Gaussian Noise

Dropout multiplies hidden activations by Bernoulli variables: 1 with probability \(p\), and 0 otherwise. More generally, activations can be multiplied by variables from other distributions.

We recently found that multiplying by a variable from \(\mathcal N(1,1)\) works as well as, or perhaps better than, Bernoulli noise. The notation \(\mathcal N(\mu,\sigma^2)\) denotes a Gaussian distribution with mean \(\mu\) and variance \(\sigma^2\).

This new dropout form adds zero-mean Gaussian noise whose standard deviation equals the unit’s activation. If \(h_i\) is hidden activation \(i\), its perturbed value is

\[
h_i+h_i r,
\qquad r\sim\mathcal N(0,1),
\]

or, equivalently,

\[
h_i r',
\qquad r'\sim\mathcal N(1,1).
\]

The variables \(r\) and \(r'\) are the additive and multiplicative noise variables, respectively. We can generalize to

\[
r'\sim\mathcal N(1,\sigma^2),
\]

where \(\sigma\) becomes a tunable hyperparameter, like \(p\) in Bernoulli dropout. Expected activations remain unchanged. Test-time weight scaling is therefore unnecessary.

So far, we have described Bernoulli dropout by retaining units with probability \(p\) during training and multiplying weights by \(p\) at test time. An equivalent approach multiplies retained activations by \(1/p\) during training and leaves test weights unchanged. The two are equivalent with suitable scaling of each layer’s learning rate and weight initialization.

Under this alternative convention, dropout multiplies \(h_i\) by \(r_b\), a variable equal to \(1/p\) with probability \(p\) and 0 otherwise. Its mean and variance are

\[
\mathbb E[r_b]=1,
\qquad
\operatorname{Var}[r_b]=\frac{1-p}{p}.
\]

Here, \(\operatorname{Var}\) denotes variance. If Gaussian dropout uses \(\sigma^2=(1-p)/p\), its multiplier \(r_g\) satisfies

\[
\mathbb E[r_g]=1,
\qquad
\operatorname{Var}[r_g]=\frac{1-p}{p}.
\]

Thus, the multipliers can have the same mean and variance. Given these first- and second-order moments, \(r_g\) has the highest **entropy**, a measure of randomness, and \(r_b\) has the lowest. Both extremes work well.

Preliminary results suggest that the high-entropy case might work slightly better. Table 10 reports MNIST errors of \(1.08\pm0.04\) for Bernoulli dropout and \(0.95\pm0.04\) for Gaussian dropout. CIFAR-10 errors are \(12.6\pm0.1\) and \(12.5\pm0.1\), respectively. For each layer, Gaussian dropout used

\[
\sigma=\sqrt{\frac{1-p}{p}},
\]

with the corresponding Bernoulli layer’s \(p\).

## 11. Conclusion

Dropout improves neural networks by reducing overfitting. Standard backpropagation builds brittle co-adaptations that work on training data but not on new data. Random dropout breaks them up by making each hidden unit’s presence unreliable.

We found improvements across object classification, digit recognition, speech recognition, document classification, and computational-biology analysis (Section 6). This suggests that dropout is general rather than domain-specific. Methods using dropout achieve state-of-the-art results on SVHN, ImageNet, CIFAR-100, and MNIST (Tables 2–6). Dropout also substantially improves standard networks on other data sets (Tables 7–8 and Section 6.3).

The idea extends to RBMs and other graphical models. Its central principle is to start with a large model that overfits easily, then repeatedly sample and train smaller sub-models. RBMs fit this framework naturally. Our dropout RBMs show desirable feature and sparsity properties (Figures 12–13).

A drawback is longer training. A dropout network typically takes 2–3 times as long to train as a standard network with the same architecture. A major cause is noisy parameter updates. Each case effectively trains a different random architecture. The calculated gradients are therefore not gradients of the final test architecture. Longer training is not surprising.

This randomness likely also prevents overfitting. It creates a trade-off: more training time allows higher dropout and less overfitting. One way to obtain some benefits without randomness is to marginalize the noise and derive a regularizer that matches dropout in expectation. For linear regression, we showed that this is a modified L2 regularizer (Section 9.1). For more complicated models, an equivalent regularizer is not obvious. Speeding up dropout is an interesting direction for future work.

## Acknowledgments

This research was supported by OGS, NSERC and an Early Researcher Award.

## Appendix A. A Practical Guide for Training Dropout Networks

Neural networks require extensive hyperparameter tuning. Dropout networks are no exception. This appendix gives heuristics that may help.

### A.1 Network Size

Dropping units is expected to reduce network capacity, its ability to represent and fit relationships. If a layer has \(n\) hidden units and each remains with probability \(p\), the expected number present is \(pn\), not \(n\). The surviving set changes each time, and units cannot freely build co-adaptations.

If \(n\) units are optimal for a standard network on a task, a good dropout layer should therefore have at least \(n/p\) units. We found this useful for choosing both convolutional and fully connected layer sizes.

### A.2 Learning Rate and Momentum

Dropout introduces much more gradient noise than standard stochastic gradient descent. Many gradient contributions tend to cancel. To compensate, dropout networks should typically use learning rates 10–100 times the best rate for a standard network.

High momentum also reduces the effect of noise. Standard networks commonly use momentum 0.9. With dropout, values around 0.95 to 0.99 work much better. High learning rates and/or momentum substantially speed learning.

### A.3 Max-norm Regularization

Large learning rates and momentum sometimes make weights grow very large. Max-norm regularization prevents this by bounding each hidden unit’s incoming weight-vector norm by \(c\). Typical values of \(c\) range from 3 to 4.

### A.4 Dropout Rate

Dropout adds the retention-probability hyperparameter \(p\). At \(p=1\), there is no dropout. Smaller \(p\) means more dropout.

Typical hidden-unit values range from 0.5 to 0.8. Input-layer values depend on the input. For real-valued inputs, such as image patches or speech frames, 0.8 is typical.

For hidden layers, \(p\) must be chosen together with the unit count \(n\). Smaller \(p\) requires large \(n\), which slows training and leads to underfitting. Large \(p\) may provide too little dropout to prevent overfitting.

## Appendix B. Detailed Description of Experiments and Data Sets

.

This appendix describes the architectures and training procedures used for the reported results. Code is available at <http://www.cs.toronto.edu/~nitish/dropout>.

The implementation uses a **graphics processing unit (GPU)**. We used the CUDA GPU-computing libraries cudamat (Mnih, 2009) and cuda-convnet (Krizhevsky et al., 2012).

### B.1 MNIST

MNIST has 60,000 training and 10,000 test cases, each a 28 × 28 digit image. We held out 10,000 random training images for validation.

We tuned hyperparameters so that the best validation error occurred after 1 million weight updates. We then recombined training and validation data and trained for 1 million updates before evaluating the test set.

We chose this procedure because it was easy to find hyperparameters that required no early stopping. Once those settings were fixed, combining the sets and training for a very long time made sense.

Figure 4 includes every combination of 2, 3, or 4 hidden layers and 1024 or 2048 units per layer: six architectures in total. All architectures, including those in Table 2, used \(p=0.5\) in hidden layers and \(p=0.8\) at the input. Every layer used final momentum 0.95 and max-norm bound \(c=2\).

To test dropout’s regularization limits, we also tried 2- and 3-layer networks with 4096 and 8192 units. The 2-layer networks improved results (Table 2). With the same dropout level, the 3-layer networks were slightly worse than 2-layer networks. Increasing dropout helped, but did not make them outperform the 2-layer networks.

### B.2 SVHN

SVHN has approximately 600,000 training images and 26,000 test images. The training set has two parts: a standard labeled set and an additional labeled set of easy examples.

We formed a 6000-case validation set from both parts. Two-thirds came from the standard set, with 400 per class. One-third came from the extra set, with 200 per class. Sermanet et al. (2012) use the same procedure.

Inputs were **RGB pixels**, with red, green, and blue components, normalized to zero mean and unit variance. Other preprocessing methods gave no noticeable improvement. These included global or local contrast normalization, which adjusts image-intensity variation over the whole image or local regions, and **ZCA whitening**, a transformation that normalizes variation along principal components and rotates back to the original coordinates. Appendix B.3 describes the transformations used there.

The best architecture has three convolutional layers, each followed by max-pooling. They have 96, 128, and 256 filters. Each filter uses a 5 × 5 **receptive field**, the input region it examines, and moves with a **stride**, or step size, of 1 pixel.

Each pooling layer pools 3 × 3 regions with stride 2 pixels. Two fully connected hidden layers, each with 2048 units, follow the convolutional layers. All units use rectified linear activation.

Retention probabilities from input through convolutional and fully connected layers are

\[
p=(0.9,\ 0.75,\ 0.75,\ 0.5,\ 0.5,\ 0.5).
\]

All weights use max-norm bound \(c=4\). All layers use momentum 0.95. These settings were tuned on validation data.

Because the training set was large, validation cases were not recombined with training cases for the final fit. We report the test error of the model with the smallest validation error.

### B.3 CIFAR-10 and CIFAR-100

Each data set has 50,000 training and 10,000 test images. CIFAR-10 has 10 categories; CIFAR-100 has 100. Images are 32 × 32 and in color.

We used 5,000 training images for validation. As with MNIST, we selected hyperparameters using validation data, then recombined it with the training set.

Preprocessing used global contrast normalization separately for each color channel, followed by ZCA whitening. For each image and channel, global contrast normalization computes the mean pixel intensity and subtracts it from that channel. ZCA whitening mean-centers the data, rotates it onto its principal components, normalizes each component, then rotates it back.

Architecture and dropout rates are the same as for SVHN. Input-layer learning rates, however, had to be smaller.

### B.4 TIMIT

We used the open-source Kaldi toolkit (Povey et al., 2011) to obtain log-filter bank features. A **monophone system**, which models individual phone units, was trained to perform **forced alignment**, matching the known speech content to recording times, and to label speech frames.

Dropout networks took windows of 21 consecutive frames and predicted the central frame’s label. No speaker-dependent operations were performed. Inputs were mean-centered and normalized to unit variance.

Retention probabilities were \(p=0.8\) at the input and 0.5 in hidden layers. Every layer used max-norm bound \(c=4\). Momentum was 0.95, with a high learning rate of 0.1.

The learning rate decayed as

\[
\epsilon_0(1+t/T)^{-1},
\]

where \(\epsilon_0\) is the initial learning rate, \(t\) indexes training progress, and \(T\) sets the decay scale.

DBN pretraining used RBMs trained with CD-1. The **Gaussian RBM**, which uses Gaussian-valued visible inputs, fixed each input unit’s variance at 1.

Best dropout finetuning required a smaller learning rate, about 0.01. Adding max-norm constraints did not improve results.

### B.5 Reuters

Reuters RCV1 contains more than 800,000 documents in 103 classes arranged in a tree hierarchy.

We constructed a subset with 402,738 articles, a 2000-word vocabulary, and 50 categories. Each document belongs to exactly one class. We split the data into equally sized training and test sets.

We tried many architectures. Dropout improved classification accuracy for all of them, but less than for image and speech data. One possible explanation is the large training set, with more than 200,000 cases. Overfitting is not a very serious problem here.

### B.6 Alternative Splicing

This data set covers 3665 **cassette exons**, the gene segments whose alternative splicing is being predicted; 1014 RNA features; and 4 tissue types derived from 27 mouse tissues.

For each input, the target contains 4 **softmax units**, one per tissue type. A softmax unit produces a probability distribution over its states. Each has 3 biologically important states, labeled inc, exc, and nc.

For each tissue type, the goal is to match the observed three-state distribution from wet-lab experiments, meaning physical biological experiments, as closely as possible.

Code Quality is defined as

\[
\sum_{i=1}^{|\text{data points}|}
\sum_{t\in\text{tissue types}}
\sum_{s\in\{\mathrm{inc},\mathrm{exc},\mathrm{nc}\}}
p_{i,t}^{s}
\log\!\left(
\frac{q_t^{s}(r_i)}{\bar p^{s}}
\right).
\]

Here:

- \(|\text{data points}|\) is the number of evaluated inputs.
- \(i\) indexes inputs, \(t\) indexes tissue types, and \(s\) indexes the three states.
- \(p_{i,t}^{s}\) is the target probability of state \(s\) for tissue \(t\) in input \(i\).
- \(r_i\) denotes input \(i\).
- \(q_t^{s}(r_i)\) is the predicted probability of state \(s\) for tissue \(t\) at input \(r_i\).
- \(\bar p^{s}\) is the average of \(p_{i,t}^{s}\) over inputs and tissues.
- \(\log\) is the logarithm used in the Code Quality expression.

We trained a two-layer dropout network with 1024 units per layer. Hidden layers used \(p=0.5\); the input layer used \(p=0.7\). Training used max-norm regularization and high, decaying learning rates.

Results were averaged over the same 5 **folds**, or data partitions used for repeated training and evaluation, as Xiong et al. (2011).
