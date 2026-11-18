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

Deep neural networks are networks with multiple nonlinear hidden layers between their inputs and outputs. Networks with many parameters are powerful machine learning systems. However, they can overfit: they can learn relationships that occur in the training data but do not hold in new test data.

Combining predictions from many different large networks can reduce overfitting. But large networks are slow to use, so evaluating many of them at test time is difficult.

Dropout addresses this problem. During training, it randomly removes units, along with their incoming and outgoing connections. This prevents excessive co-adaptation, in which units rely on particular other units to correct their mistakes. Training with dropout samples from an exponential number of “thinned” networks, each containing only the units that remain. At test time, a single unthinned network with smaller weights approximates the effect of averaging their predictions.

Dropout significantly reduces overfitting and gives major improvements over other regularization methods—methods that constrain learning to reduce overfitting. Dropout improves neural-network performance on supervised learning tasks in vision, speech recognition, document classification, and computational biology. In supervised learning, training examples have target outputs. Methods using dropout obtain state-of-the-art results on many benchmark data sets. The test-time approximation is examined directly in Section 7.5 and Figure 11.

**Keywords:** neural networks, regularization, model combination, deep learning

## 1. Introduction

Deep neural networks contain multiple nonlinear hidden layers. A hidden layer is an intermediate layer of computational units between the network’s input and output. Multiple nonlinear layers make these models expressive: they can learn complicated relationships between inputs and outputs.

With limited training data, many learned relationships result from sampling noise. They occur in the training set but not in real test data, even when the test data come from the same distribution. This is overfitting.

Existing methods for reducing overfitting include:

- **Early stopping:** stop training when performance begins to worsen on a validation set, a separate set used to select training choices.
- **Weight penalties:** penalize learned connection weights. Examples include L1 and L2 regularization, which penalize weights through their absolute values or squared magnitudes.
- **Soft weight sharing:** encourage weights to share similar values (Nowlan and Hinton, 1992).

With unlimited computation, the best way to regularize a fixed-sized model is to average predictions over all possible parameter settings. Each setting is weighted by its posterior probability: its probability after observing the training data. This Bayesian approach can sometimes be approximated well for simple or small models (Xiong et al., 2011; Salakhutdinov and Mnih, 2008).

We want to approach this Bayesian gold standard with much less computation. We propose approximating an equally weighted **geometric mean** of predictions from exponentially many learned models that share parameters. A geometric mean combines values through their product, rather than through an ordinary arithmetic average.

**Figure 1: Dropout Neural Net Model.** Left, panel (a), “Standard Neural Net”: a standard neural net with 2 hidden layers. Right, panel (b), “After applying dropout”: an example of a thinned net produced by applying dropout to the network on the left. Crossed units have been dropped.

Model combination nearly always improves machine learning performance. However, averaging outputs from many separately trained large neural networks is prohibitively expensive.

Combination is most useful when the individual models differ. Neural networks can be made different by using different architectures—the arrangements and sizes of their layers—or different training data. Both choices are difficult:

- Different architectures require separate searches for suitable hyperparameters. Hyperparameters are training choices, such as learning rates or layer sizes, rather than parameters learned directly from the data.
- Training each large network requires substantial computation.
- Large networks usually need much training data. There may not be enough data to train different networks on different subsets.
- Even if many large networks can be trained, evaluating all of them is infeasible in applications that require fast responses.

Dropout addresses both overfitting and the cost of model combination. To **drop a unit** means to remove it temporarily, together with all its incoming and outgoing connections. Dropout can remove hidden units and visible, or input, units.

The selection is random. In the simplest case, each unit is retained independently with probability \(p\). Thus, \(p\) is the retention probability, not the probability of removal. It can be selected using a validation set. Alternatively, \(p=0.5\) seems close to optimal for many networks and tasks. For input units, the best retention probability is usually closer to 1 than to 0.5. Section 7.3 examines this choice experimentally.

Applying dropout samples a thinned network containing all units that survive. A network with \(n\) units has \(2^n\) possible thinned networks, because every unit can be present or absent. These networks share weights. The total parameter count therefore remains \(O(n^2)\), or less; this notation means that the count grows no faster than quadratically with \(n\).

A new thinned network is sampled for every presentation of every training case. Training with dropout can therefore be viewed as training \(2^n\) networks with extensive weight sharing. Each individual thinned network is trained very rarely, if at all.

Explicitly averaging predictions from exponentially many networks is infeasible at test time. Instead, we use one unthinned network whose weights are scaled-down versions of the trained weights. If a unit was retained with probability \(p\), its outgoing weights are multiplied by \(p\) at test time.

**Figure 2:** Left, panel (a), “At training time”: a unit is present with probability \(p\) and connects to the next layer with weights \(w\). Right, panel (b), “At test time”: the unit is always present, and its weights are \(pw\). The output at test time is the same as the expected output at training time. Here, “expected” means averaged over the random dropout choices.

This scaling ensures that, for any hidden unit, its expected training-time output under dropout is the same as its actual test-time output. It combines the \(2^n\) weight-sharing networks into one network for testing. The resulting model averaging is approximate. In the MNIST experiment in Section 7.5, averaging about 50 sampled networks matches weight scaling; averaging more is slightly better, but within one standard deviation of the scaled network.

Across many classification problems, training with dropout and using this test procedure gives significantly lower generalization error—error on new data—than training with other regularizers. For example, Table 9 reports MNIST test error of 1.05% for dropout plus max-norm regularization, compared with 1.35% for max-norm alone and 1.62% for L2 alone.

Dropout is not limited to feed-forward networks, whose computations proceed from input to output. It can also be applied to graphical models, which describe joint probability distributions through relationships among variables. We introduce dropout **Restricted Boltzmann Machines (RBMs)**, a model with visible and hidden variables defined in Section 8. Dropout RBMs are better than standard RBMs in certain respects. Figures 12–13 show their feature and sparsity properties.

The paper is organized as follows. Section 2 gives the motivation. Section 3 reviews related work. Section 4 defines the dropout model. Section 5 describes training. Section 6 presents experiments across domains and comparisons with regularization and model combination methods. Section 7 examines network properties and interactions with hyperparameters. Section 8 introduces dropout RBMs. Section 9 explores marginalizing dropout—that is, averaging out its random noise analytically. Appendix A gives practical guidance on hyperparameter choices. Appendix B gives experimental details.

© 2014 Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever and Ruslan Salakhutdinov.

## 2. Motivation

One motivation comes from a theory about the role of sex in evolution (Livnat et al., 2010).

Sexual reproduction combines half the genes of one parent with half the genes of the other, with a very small amount of random mutation. Asexual reproduction instead makes a slightly mutated copy of one parent’s genes.

Asexual reproduction might seem better for optimizing individual fitness. A set of genes that already works well together can be passed directly to offspring. Sexual reproduction can break up these co-adapted sets, especially large ones. Intuitively, this should reduce the fitness of organisms with complicated co-adaptations. Yet most advanced organisms evolved through sexual reproduction.

One possible explanation is that long-term natural selection favors **mix-ability** rather than individual fitness. Mix-ability is the ability of genes to work well with another randomly selected set of genes. This makes them more robust.

A gene that cannot rely on many particular partners must be useful on its own or with a small number of other genes. According to this theory, sexual reproduction does more than spread useful new genes. It also reduces complex co-adaptations that could prevent a new gene from improving an individual’s fitness.

Dropout gives a similar requirement to each hidden unit. The unit must work with a randomly selected sample of other units. This should make it robust and encourage it to create a useful feature without relying on other hidden units to correct its mistakes.

Units in a layer will still learn different things. One might expect a network to resist dropout by making many copies of each hidden unit. But this is a poor solution for the same reason that replica codes—codes that repeat information—are a poor way to handle a noisy communication channel.

A related motivation comes from conspiracies. Ten conspiracies involving five people each are probably better at creating havoc than one conspiracy that requires fifty people to perform correctly.

A large conspiracy can work when conditions remain fixed and there is ample rehearsal time. Under non-stationary, or changing, conditions, a smaller conspiracy is more likely to keep working. Likewise, complex co-adaptations can work well on the training set but fail on novel test data. Multiple simpler co-adaptations that achieve the same result are less likely to fail.

## 3. Related Work

Dropout can be viewed as regularization through noise added to hidden-unit states.

Noise has previously been used in **Denoising Autoencoders (DAEs)** (Vincent et al., 2008, 2010). An autoencoder learns to reconstruct its input. A denoising autoencoder receives a noisy input and learns to reconstruct the noise-free version.

Our work extends this idea in several ways. We show that dropout works in hidden layers, can be interpreted as model averaging, and applies to supervised learning as well as unsupervised feature learning, which learns representations without target labels. It also applies to other neuron-based architectures, including Boltzmann Machines.

While 5% noise typically works best for DAEs, our test-time weight scaling permits much higher noise. Removing 20% of input units and 50% of hidden units was often optimal. These amounts correspond to retention probabilities of 0.8 and 0.5, used in many experiments below.

Because dropout is a stochastic, or random, regularizer, it is natural to ask about its deterministic counterpart. This counterpart is obtained by marginalizing out the noise: averaging the loss over the noise distribution rather than sampling noise during training. Section 9 shows that this can be done analytically in simple cases.

van der Maaten et al. (2013) also studied deterministic regularizers arising from exponential-family noise distributions, a family of probability distributions that includes the dropout noise they call “blankout noise.” Their experiments apply noise only to inputs and use models without hidden layers. Wang and Manning (2013) proposed speeding up dropout by marginalizing its noise. Chen et al. (2012) studied marginalization for denoising autoencoders.

Dropout stochastically minimizes a loss function, which measures prediction error, under a noise distribution. This can be viewed as minimizing the expected loss.

Globerson and Roweis (2006) and Dekel et al. (2010) studied a different setting. An adversary chooses which units to remove, and training minimizes the loss under that choice. Instead of specifying a noise distribution, they fix the maximum number of removable units. Their work also does not explore models with hidden units.

## 4. Model Description

Consider a neural network with \(L\) hidden layers. The index \(l\in\{1,\ldots,L\}\) identifies a hidden layer; braces denote the listed set of allowed indices.

Use the following notation:

- \(x\): the network’s input vector.
- \(y^{(0)}=x\): the input treated as the output of layer 0.
- \(z^{(l)}\): the vector of inputs to the activation functions in layer \(l\).
- \(y^{(l)}\): the vector of outputs from layer \(l\).
- \(W^{(l)}\) and \(b^{(l)}\): the layer’s weight matrix and bias vector.
- \(i\): an individual hidden unit.
- \(w_i^{(l+1)}\): the weights feeding unit \(i\) in layer \(l+1\).
- \(b_i^{(l+1)}\): that unit’s bias, an additive parameter.
- \(f\): an activation function, which converts a unit’s weighted input into its output.

For example, the logistic activation is
\[
f(x)=\frac{1}{1+\exp(-x)},
\]
where \(\exp\) denotes the exponential function.

For \(l\in\{0,\ldots,L-1\}\), a standard network computes
\[
z_i^{(l+1)}
=
w_i^{(l+1)}y^l+b_i^{(l+1)},
\]
\[
y_i^{(l+1)}
=
f\!\left(z_i^{(l+1)}\right).
\]
The product \(w_i^{(l+1)}y^l\) is the weighted sum of outputs from the preceding layer.

For dropout, let \(r^{(l)}\) be a random vector with one entry \(r_j^{(l)}\) for each unit \(j\) in layer \(l\). Each entry is independently drawn from a **Bernoulli distribution**: it is 1 with probability \(p\), and 0 otherwise. The notation \(\sim\) means “is drawn from.”

Let \(*\) denote element-wise multiplication, so that corresponding vector entries are multiplied. Let \(\widetilde y^{(l)}\) denote the thinned outputs. The forward computation becomes
\[
r_j^{(l)}\sim\operatorname{Bernoulli}(p),
\]
\[
\widetilde y^{(l)}=r^{(l)}*y^{(l)},
\]
\[
z_i^{(l+1)}
=
w_i^{(l+1)}\widetilde y^l+b_i^{(l+1)},
\]
\[
y_i^{(l+1)}
=
f\!\left(z_i^{(l+1)}\right).
\]

**Figure 3: Comparison of the basic operations of a standard and dropout network.** Panel (a), “Standard network,” applies weights \(w_i^{(l+1)}\), bias \(b_i^{(l+1)}\), and activation \(f\) directly to the preceding outputs. Panel (b), “Dropout network,” first multiplies those outputs by the random entries of \(r^{(l)}\). The \(+1\) input in each panel supplies the bias.

The mask \(r^{(l)}\) sets dropped outputs to zero and leaves retained outputs unchanged. The resulting \(\widetilde y^{(l)}\) becomes the input to the next layer. Applying this process at each layer samples a subnetwork.

During learning, derivatives of the loss are backpropagated through this subnetwork. **Backpropagation** computes how changes in each parameter affect the loss by passing derivatives backward through the network.

At test time, use
\[
W_{\mathrm{test}}^{(l)}=pW^{(l)},
\]
where \(W_{\mathrm{test}}^{(l)}\) denotes the test-time weights. Use the resulting network without dropout, as in Figure 2.

## 5. Learning Dropout Nets

This section describes how to train dropout neural networks.

### 5.1 Backpropagation

Dropout networks can be trained using **stochastic gradient descent**: repeatedly estimate loss derivatives from a small group of training cases and update parameters in the direction that reduces the loss. Such a group is a **mini-batch**.

Training differs from standard networks only in the sampled architecture:

1. For each case in a mini-batch, sample a thinned network.
2. Run forward computation and backpropagation only through that network.
3. Average each parameter’s gradients over all cases in the mini-batch.
4. If a case does not use a parameter, it contributes a zero gradient for that parameter.

Methods that improve ordinary stochastic gradient descent are also useful here. They include **momentum**, which carries information from previous updates into the current update; **annealed learning rates**, which decrease the update step size during training; and L2 weight decay.

One regularizer is especially useful with dropout: bound the length of each hidden unit’s incoming weight vector. If \(w\) is this vector, impose
\[
\|w\|_2\leq c.
\]
Here, \(\|w\|_2\) is the Euclidean norm, or square root of the sum of squared entries, and \(c\) is a fixed upper bound.

Whenever an update moves \(w\) outside the ball of radius \(c\), project it back onto that ball’s surface. This is **max-norm regularization**: the maximum permitted norm is \(c\). Choose \(c\) using a validation set.

Max-norm regularization has previously been used for collaborative filtering, which predicts preferences from shared patterns among users and items (Srebro and Shraibman, 2005). It typically improves stochastic-gradient training of deep networks even without dropout.

Dropout alone gives significant gains. Combining it with max-norm regularization, large decaying learning rates, and high momentum gives a further significant boost. For example, Table 2 reports 1.25% MNIST error for a three-layer, 1024-unit ReLU dropout network and 1.06% after adding max-norm. Table 9 gives the related fixed-architecture comparison.

A possible justification is that bounding weight-vector lengths permits huge learning rates without weights blowing up. Dropout noise then lets optimization explore regions of weight space that might otherwise be difficult to reach. As the learning rate decreases, the steps become shorter, exploration declines, and optimization eventually settles into a minimum.

### 5.2 Unsupervised Pretraining

Networks can first be trained without labels using stacks of RBMs (Hinton and Salakhutdinov, 2006), autoencoders (Vincent et al., 2010), or Deep Boltzmann Machines (Salakhutdinov and Hinton, 2009). Deep Boltzmann Machines are the multilayer Boltzmann models referred to below as DBMs.

This **pretraining** uses unlabeled data to obtain initial weights. **Finetuning** then adjusts those weights for the target task using backpropagation. In certain cases, pretraining followed by finetuning substantially outperforms training from random initial weights.

Dropout can be used during finetuning without changing pretraining. First multiply pretrained weights by \(1/p\). This makes each unit’s expected output under random dropout equal to its output during pretraining.

We initially worried that dropout’s randomness would erase information in the pretrained weights. This did happen when finetuning learning rates were comparable to the best rates for randomly initialized networks. With smaller rates, the pretrained information seemed to be retained. Dropout then improved final generalization error over finetuning without dropout. Tables 2 and 7 give numerical comparisons.

## 6. Experimental Results

We trained dropout networks on classification problems from several domains. Dropout improved generalization on every data set compared with neural networks that did not use it. The detailed results follow in Tables 2–9 and the Reuters comparison in Section 6.3.

The data sets are:

- **MNIST:** a standard toy data set of handwritten digits.
- **TIMIT:** a standard benchmark for clean speech recognition.
- **CIFAR-10 and CIFAR-100:** tiny natural images from 10 and 100 categories, respectively (Krizhevsky, 2009).
- **Street View House Numbers (SVHN):** house-number images collected by Google Street View (Netzer et al., 2011).
- **ImageNet:** a large collection of natural images.
- **Reuters-RCV1:** Reuters newswire articles. RCV1 is the corpus identifier used here.
- **Alternative Splicing:** RNA features for predicting alternative gene splicing (Xiong et al., 2011). RNA means ribonucleic acid. The task concerns differing splicing outcomes in tissues.

Table 1 summarizes the data. In this table, **dimensionality** is the number of input features, **K** means thousand, **M** means million, and **dim** means dimensions. A speech frame is one time-local representation of the speech signal. **ILSVRC-2012** identifies the 2012 ImageNet Large-Scale Visual Recognition Challenge, discussed in Section 6.1.4.

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

We chose diverse data sets to show that dropout is a general technique, not one specific to an application domain. This section presents key results. Appendix B provides detailed data and training descriptions.

### 6.1 Results on Image Data Sets

We evaluated dropout on MNIST, SVHN, CIFAR-10, CIFAR-100, and ImageNet. These cover different image types and training-set sizes. Models achieving state-of-the-art results on all five data sets use dropout; Tables 2–6 report the comparisons.

### 6.1.1 MNIST

MNIST contains 28 × 28 pixel images of handwritten digits. The task is to assign each image to one of 10 digit classes.

The results here use the **permutation invariant setting**: they do not use the images’ 2-D spatial structure. Here, 2-D means two-dimensional. Appendix B.1 gives the validation and training procedure.

Table 2 uses these labels:

- **NN:** neural network.
- **SVM:** support vector machine; the listed comparison uses a Gaussian kernel, a similarity function. The paper gives no further specification of this comparison.
- **NA:** not applicable.
- **Logistic:** units using the logistic activation defined in Section 4.
- **ReLU:** rectified linear unit, which outputs zero for negative input and the input itself otherwise (Jarrett et al., 2009).
- **Maxout:** the unit type used by Goodfellow et al. (2013); its operation is not further defined in this paper.
- **DBN:** the model label used for networks pretrained with stacked RBMs; the abbreviation is not expanded in the supplied paper.
- **DBM:** Deep Boltzmann Machine.

An architecture such as 500-500-2000 lists hidden-layer sizes in order. In entries such as “3 layers, 1024 units,” the stated unit count is per hidden layer.

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

The best permutation-invariant networks without dropout or unsupervised pretraining achieve about 1.60% error (Simard et al., 2003). Dropout reduces this to 1.35%. Replacing logistic units with ReLUs reduces it further to 1.25%. Adding max-norm regularization reduces it to 1.06%. These are the corresponding first, third, fourth, and fifth rows of Table 2.

Larger networks improve the results further. Table 2 reports 1.04% for three layers of 2048 units, 1.01% for two layers of 4096 units, and 0.95% for two layers of 8192 units.

The last of these networks has more than 65 million parameters and is trained on only 60,000 examples. Obtaining good generalization at this size is very hard with standard regularization and early stopping. Dropout prevents overfitting even here and does not need early stopping. Appendix B.1 describes the fixed-duration training procedure.

Goodfellow et al. (2013) further reduced error to 0.94% by replacing ReLUs with maxout units, as shown in Table 2. All dropout networks use \(p=0.5\) for hidden units and \(p=0.8\) for inputs.

Dropout also improves pretrained networks. In Table 2, DBN error falls from 1.18% to 0.92%, and DBM error falls from 0.96% to 0.79%. The DBM-pretrained dropout result of 0.79% is the best performance ever reported for the permutation invariant setting.

Better results are possible by using 2-D spatial information and augmenting the training set with distorted copies of its images. Data augmentation means adding such modified training examples. We examine dropout in that setting on more interesting data sets below.

We also tested robustness across architectures. We held every hyperparameter, including \(p\), fixed and trained different architectures with and without dropout. The networks had 2, 3, or 4 hidden layers, with either 1024 or 2048 units per layer: six architectures in total.

Figure 4 shows two distinct clusters of test-error trajectories. Dropout gives a huge improvement across all architectures without architecture-specific hyperparameter tuning.

**Figure 4: Test error for different architectures with and without dropout.** The networks have 2 to 4 hidden layers, each with 1024 to 2048 units. The horizontal axis is the number of weight updates, with ticks at 0, 200000, 400000, 600000, 800000, and 1000000. The vertical axis is classification error %, with ticks at 1.0, 1.5, 2.0, and 2.5. The two trajectory groups are labeled “Without dropout” and “With dropout.”

### 6.1.2 Street View House Numbers

SVHN contains color images of house numbers collected by Google Street View (Netzer et al., 2011). Figure 5a shows examples. We use the portion containing 32 × 32 color images approximately centered on one digit. The task is to identify that digit.

We applied dropout to **convolutional neural networks** (LeCun et al., 1989), which apply learned filters across image locations. A **convolutional layer** uses these filters locally; a **fully connected layer** connects each unit to all outputs of the preceding layer.

The best architecture we found has three convolutional layers followed by 2 fully connected hidden layers. All hidden units are ReLUs. Each convolutional layer is followed by **max-pooling**, which retains the largest response within a local region. Appendix B.2 gives the filter sizes and other details.

From input through the convolutional and fully connected layers, the retention probabilities are
\[
p=(0.9,0.75,0.75,0.5,0.5,0.5).
\]
Max-norm regularization is applied to both convolutional and fully connected weights.

In Table 3, **Conv Net** means convolutional network. The comparison labels **WDCH** and **HOG** identify feature methods from Netzer et al. (2011); HOG denotes histograms of oriented gradients, while WDCH is not expanded in the paper. **KMeans** denotes clustering into representative groups. **Stacked Sparse Autoencoders** are successive autoencoders trained for sparse representations, in which few units are highly active. Average pooling averages local responses; L2 and L4 pooling combine them through the corresponding norms. Padding extends image boundaries for computation. Stochastic pooling uses a random pooling choice. The paper gives no further definitions of these comparison procedures.

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

Convolutional networks outperform the other methods in Table 3. The best-performing convolutional networks that do not use dropout achieve 3.95% error. Adding dropout only to fully connected layers reduces error to 3.02%. Adding it to convolutional layers as well reduces error to 2.55%. Maxout gives a further gain, to 2.47%.

The gain from convolutional-layer dropout, from 3.02% to 2.55%, is worth noting. Because these layers have relatively few parameters, one might expect little overfitting and little benefit from dropout. However, dropout in lower layers still helps by supplying noisy inputs to the higher fully connected layers, preventing them from overfitting.

### 6.1.3 CIFAR-10 and CIFAR-100

CIFAR-10 and CIFAR-100 contain 32 × 32 color images from 10 and 100 categories, respectively. Figure 5b shows examples. Appendix B.3 describes preprocessing, architectures, and training.

Table 4 reports the results. Without data augmentation, Snoek et al. (2012) used Bayesian hyperparameter optimization—selection of hyperparameters using a probabilistic model of their performance—to obtain 14.98% error on CIFAR-10. Dropout in fully connected layers reduces error to 14.32%. Dropout in every layer reduces it to 12.61%. Goodfellow et al. (2013) further reduce error to 11.68% using maxout units instead of ReLUs.

On CIFAR-100, dropout reduces error from 43.48% to 37.20%, a huge improvement. Table 4 also reports 41.26% for dropout in fully connected layers only. No data augmentation was used for either data set, apart from input dropout.

**Figure 5: Samples from image data sets.** Panel (a): Street View House Numbers (SVHN). Panel (b): CIFAR-10. Each row corresponds to a different category.

| Method | CIFAR-10 | CIFAR-100 |
|---|---|---|
| Conv Net + max pooling (hand tuned) | 15.60 | 43.48 |
| Conv Net + stochastic pooling (Zeiler and Fergus, 2013) | 15.13 | 42.51 |
| Conv Net + max pooling (Snoek et al., 2012) | 14.98 | - |
| Conv Net + max pooling + dropout fully connected layers | 14.32 | 41.26 |
| Conv Net + max pooling + dropout in all layers | 12.61 | 37.20 |
| Conv Net + maxout (Goodfellow et al., 2013) | 11.68 | 38.57 |

**Table 4: Error rates on CIFAR-10 and CIFAR-100.** A dash indicates that no result is reported.

### 6.1.4 ImageNet

ImageNet contains over 15 million labeled high-resolution images in roughly 22,000 categories. Beginning in 2010, an annual ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) has been held as part of the Pascal Visual Object Challenge. It uses an ImageNet subset with roughly 1000 images in each of 1000 categories.

Because there are many categories, results are conventionally reported as **top-1** and **top-5** error. Top-1 error concerns the single most probable label. Top-5 error is the fraction of test images whose correct label is absent from the five labels the model considers most probable. Figure 6 shows predictions on test images.

ILSVRC-2010 is the only version for which test labels are available. We therefore performed most experiments on it. Table 5 shows that convolutional networks with dropout outperform the listed alternatives by a large margin: their top-1 and top-5 errors are 37.5 and 17.0, compared with 45.7 and 25.7 for SIFT plus Fisher Vectors, and 47.1 and 28.2 for Sparse Coding. Krizhevsky et al. (2012) give the architecture and implementation details.

In Tables 5–6, **Sparse Coding** is the named sparse-representation method. **SIFT** means scale-invariant feature transform; **Fisher Vectors**, also abbreviated **FVs**, are the named feature representation. **Dense SIFT** uses that feature description at densely sampled locations. **LBP** means local binary patterns; **GIST** is the named image descriptor; **CSIFT** is the named color-SIFT descriptor. The paper does not further specify these comparison features. **Color Statistics** names the color-feature comparison. **Avg** means average, **val** means validation set, and **test** means test set.

**Figure 6: Some ImageNet test cases with the 4 most probable labels as predicted by our model.** Horizontal bar lengths are proportional to the probabilities assigned to the labels. Pink indicates ground truth, the correct label.

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

Our convolutional-network model with dropout won ILSVRC-2012. Because its test labels are unavailable, Table 6 gives test results for the final submission and validation results for model variations.

The best methods based on standard vision features achieve about 26% top-5 error. Convolutional networks with dropout achieve about 16% test error, a staggering difference. The exact Table 6 entries are 26.2% for the averaged conventional-feature classifiers and 16.4% for the average of 5 dropout convolutional networks. The other conventional-feature method gives 27.3%.

Figure 6 shows that the model makes very reasonable predictions even when its best guess is incorrect.

### 6.2 Results on TIMIT

We next applied dropout to speech recognition. TIMIT contains recordings of 680 speakers covering 8 major American-English dialects. Each reads ten phonetically rich sentences in a controlled, noise-free environment.

The networks receive windows of 21 log-filter bank frames and predict the label of the central frame. Log-filter bank frames represent speech through logarithmic filter-bank measurements. No speaker-dependent operations were performed. Appendix B.4 gives preprocessing and training details.

Table 7 reports **phone error rate**, the speech-recognition error measure for phones, or speech sounds. A 6-layer network gives 23.4%; dropout reduces this to 21.8%.

We also used pretrained weights. A 4-layer network pretrained with stacked RBMs gives 22.7%; dropout reduces this to 19.7%. Similarly, for an 8-layer network, error falls from 20.5% to 19.7%.

The Table 7 label **mcRBM-DBN** names the pretrained model from Dahl et al. (2010); mcRBM is not expanded or further defined in this paper.

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

To test dropout on text, we trained a document classifier using a subset of Reuters-RCV1. The full corpus contains over 800,000 Reuters newswire articles on many topics.

The task uses a **bag-of-words representation**, which represents a document by its words without their order, to classify it into 50 disjoint topics. Appendix B.5 gives the subset and split.

Our best neural network without dropout gives 31.05% error. Adding dropout reduces this to 29.62%. This gain is much smaller than the gains for vision and speech.

### 6.4 Comparison with Bayesian Neural Networks

Dropout can be viewed as equally weighted averaging over exponentially many models with shared weights.

Bayesian neural networks (Neal, 1996), by contrast, provide the proper model average over network structures and parameters. Their model weights account for both the prior—the probability assigned before observing the data—and fit to the data. This is more correct than dropout’s equal weighting.

Bayesian neural networks are particularly useful when data are scarce, as in medical diagnosis, genetics, drug discovery, and other computational biology applications. But they are slow to train and difficult to scale to very large networks. Obtaining predictions from many large networks is also expensive. Dropout networks are much faster to train and use at test time.

We compare the methods on a small genetics data set where Bayesian neural networks are known to perform well and obtain state-of-the-art results. The aim is to measure how much dropout loses relative to them.

The task is to predict alternative splicing from RNA features (Xiong et al., 2011). Alternative splicing is a significant source of cellular diversity in mammalian tissues. Predicting it in particular tissues under different conditions is important for understanding many human diseases.

Given RNA features, the model predicts the probabilities of three splicing-related events. Appendix B.6 specifies the targets and network. The evaluation measure is **Code Quality**, a measure of negative **Kullback–Leibler (KL) divergence**, which compares target and predicted probability distributions. Higher Code Quality is better. Table 8 reports it in bits, a unit of information; Appendix B.6 gives the exact formula.

Table 8 also uses **PCA**, principal component analysis, to name the dimensionality-reduction preprocessing used by comparison methods.

| Method | Code Quality (bits) |
|---|---|
| Neural Network (early stopping) (Xiong et al., 2011) | 440 |
| Regression, PCA (Xiong et al., 2011) | 463 |
| SVM, PCA (Xiong et al., 2011) | 487 |
| Neural Network with dropout | 567 |
| Bayesian Neural Network (Xiong et al., 2011) | 623 |

**Table 8: Results on the Alternative Splicing Data Set.**

As expected, the Bayesian neural network performs better: 623 bits versus dropout’s 567. But dropout substantially improves the standard network’s 440 and outperforms the other methods’ 463 and 487.

The challenge is avoiding overfitting on a small training set. One approach is to reduce input dimensionality with PCA, then apply an SVM or logistic regression, a probability-prediction model using a logistic response. Dropout prevents overfitting without this dimensionality reduction.

The dropout networks have 1000s of hidden units, compared with only a few tens in the Bayesian neural network. Appendix B.6 specifies two dropout layers with 1024 units each. Their performance shows dropout’s strong regularizing effect.

### 6.5 Comparison with Standard Regularizers

Existing neural-network regularizers include L2 weight decay, more generally Tikhonov regularization (Tikhonov, 1943); lasso, an L1-based penalty (Tibshirani, 1996); KL-sparsity, which uses KL divergence to encourage a target activation sparsity; and max-norm regularization.

We compare these with dropout on MNIST. Every method uses the same ReLU architecture:
\[
784\text{-}1024\text{-}1024\text{-}2048\text{-}10.
\]
This notation lists input size, hidden-layer sizes, and output size in order.

All networks are trained by stochastic gradient descent. Each regularizer’s hyperparameters—decay constants, target sparsity, dropout rate, or max-norm bound—are selected using validation data.

| Method | Test Classification error % |
|---|---|
| L2 | 1.62 |
| L2 + L1 applied towards the end of training | 1.60 |
| L2 + KL-sparsity | 1.55 |
| Max-norm | 1.35 |
| Dropout + L2 | 1.25 |
| Dropout + Max-norm | 1.05 |

**Table 9: Comparison of different regularization methods on MNIST.**

Dropout plus max-norm gives the lowest generalization error, 1.05%, among these methods. Table 9 gives the full comparison.

## 7. Salient Features

Section 6 provides strong evidence that dropout improves neural networks. We now examine its effects on learned features and the sparsity of hidden-unit activations, where an activation is a unit’s output value.

We also examine how its benefits vary with retention probability, network size, and training-set size. These observations give insight into why dropout works well.

### 7.1 Effect on Features

In an ordinary network, each parameter’s derivative indicates how to reduce the final loss given the current behavior of all other units. Units can therefore change to correct other units’ mistakes. This can create complex co-adaptations that fail on unseen data and cause overfitting.

We hypothesize that dropout prevents these co-adaptations by making the presence of other hidden units unreliable. A hidden unit cannot depend on particular partners to correct its mistakes. It must perform well across many contexts supplied by different subsets of other units.

To examine this directly, we compare first-level features learned on a visual task. We train two MNIST autoencoders, each with one hidden layer of 256 ReLUs. One has no dropout; the other uses hidden-layer dropout with \(p=0.5\).

**Figure 7: Features learned on MNIST with one hidden layer autoencoders having 256 rectified linear units.** Panel (a): “Without dropout.” Panel (b): “Dropout with \(p=0.5\).”

Both autoencoders have similar test reconstruction errors. Their features differ, however. In Figure 7a, the features appear co-adapted to produce good reconstructions. An individual hidden unit does not seem to detect a meaningful feature.

In Figure 7b, units appear to detect edges, strokes, and spots in different image locations. This shows that dropout breaks up co-adaptations, probably the main reason it lowers generalization error.

### 7.2 Effect on Sparsity

A sparse representation has few highly active units for each data case. Each unit should also have low average activation across cases.

We found that dropout makes hidden activations sparse even without a sparsity-inducing regularizer. To examine this, we use the autoencoders from Section 7.1 and a random mini-batch from the test set.

For each model, we plot two histograms. A histogram displays how frequently values fall in different ranges. The left histogram shows each hidden unit’s mean activation over the mini-batch. The right shows the distribution of all hidden-unit activations.

**Figure 8: Effect of dropout on sparsity. ReLUs were used for both models.** Panel (a), “Without dropout”: most units have mean activation about 2.0, and the activation histogram has a huge mode away from zero. A large fraction of units have high activation. Panel (b), “Dropout with \(p=0.5\)”: most units have a smaller mean activation, about 0.7. The activation histogram has a sharp peak at zero, and very few units have high activation.

The activation histogram without dropout has substantial mass away from zero. With dropout, fewer units have high activations. Mean activations are also lower. The overall mean is close to 2.0 without dropout and around 0.7 with dropout.

### 7.3 Effect of Dropout Rate

We vary \(p\), the probability of retaining a unit, in two settings:

1. Keep the number of hidden units fixed.
2. Change the number of hidden units so that the expected number retained after dropout stays fixed.

For the first setting, we use the architecture
\[
784\text{-}2048\text{-}2048\text{-}2048\text{-}10
\]
without input dropout. Figure 9a plots test error against \(p\).

With a fixed architecture, small \(p\) activates very few units. Both training and test errors are high, indicating **underfitting**: the trained model does not fit even its training data well. As \(p\) increases, error decreases. It is flat for
\[
0.4\leq p\leq0.8
\]
and rises again as \(p\) approaches 1.

**Figure 9: Effect of changing dropout rates on MNIST.** Panel (a): “Keeping \(n\) fixed.” Panel (b): “Keeping \(pn\) fixed.” Here, \(n\) is the number of hidden units in a layer, and \(pn\) is the expected retained count. Both panels plot training and test classification error % against retention probability \(p\), with horizontal ticks 0.0, 0.2, 0.4, 0.6, 0.8, and 1.0. Panel (a) has vertical ticks from 0.0 to 3.5 in steps of 0.5; panel (b) has vertical ticks from 0.0 to 3.0 in steps of 0.5.

For the second setting, hold \(pn\) fixed. A smaller \(p\) then requires a larger full layer. After dropout, the expected retained count is the same across architectures, although their test-time sizes differ.

We set \(pn=256\) for the first two hidden layers and \(pn=512\) for the last. Figure 9b shows that errors at small \(p\) are much lower than in Figure 9a. At \(p=0.1\), error falls from 2.7% to 1.7%.

For this choice of \(pn\), values near 0.6 perform best. Our usual default of 0.5 is close to optimal.

### 7.4 Effect of Data Set Size

A useful regularizer should permit good generalization from a highly parameterized model trained on a small data set. Large networks trained conventionally overfit massively on small data sets.

We test dropout by varying the amount of MNIST training data. We randomly select data sets of size 100, 500, 1K, 5K, 10K, and 50K. Every experiment uses the architecture
\[
784\text{-}1024\text{-}1024\text{-}2048\text{-}10.
\]
Dropout uses \(p=0.5\) in hidden layers and \(p=0.8\) at the input.

**Figure 10: Effect of varying data set size.** The curves are labeled “With dropout” and “Without dropout.” The horizontal data-set-size axis has ticks \(10^2,10^3,10^4,10^5\). The vertical classification-error-% axis has ticks 0, 5, 10, 15, 20, 25, and 30.

For the extremely small sets of 100 and 500 examples, dropout gives no improvement. The model has enough parameters to overfit despite the dropout noise.

As data size increases, the gain from dropout first increases and then declines, as Figure 10 shows. This suggests a data-size “sweet spot” for a given architecture and dropout rate. There must be enough data to prevent memorization despite the noise, but not so much that overfitting is no longer a problem.

### 7.5 Monte-Carlo Model Averaging vs. Weight Scaling

Our efficient test procedure approximates model combination by scaling down trained weights. A more expensive but more correct procedure samples \(k\) dropout networks for each test case and averages their predictions. Here, \(k\) is the number of sampled networks.

As \(k\to\infty\), meaning that the sample count grows without bound, this **Monte-Carlo model average** approaches the true model average. Monte-Carlo refers to estimation by random sampling.

We test how many samples are needed to match weight scaling. On MNIST, we average predictions from \(k\) randomly sampled networks and compare their test error with that of the scaled network. Varying \(k\) also shows how quickly a finite-sample average approaches the true average’s error.

**Figure 11: Monte-Carlo model averaging vs. weight scaling.** The horizontal axis is “Number of samples used for Monte-Carlo averaging (\(k\)),” with ticks 0, 20, 40, 60, 80, 100, and 120. The vertical axis is test classification error %, with ticks 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, and 1.35. The horizontal comparison line is labeled “Approximate averaging by weight scaling.”

At around \(k=50\), Monte-Carlo averaging becomes as good as weight scaling. With more samples it is slightly better, but the difference is well within one standard deviation, a measure of variation. This suggests that weight scaling is a fairly good approximation to the true model average.

## 8. Dropout Restricted Boltzmann Machines

Dropout also applies to RBMs. We define the model and examine some of its properties.

### 8.1 Model Description

Consider an RBM with:

- \(v\in\{0,1\}^D\): a vector of \(D\) binary visible units.
- \(h\in\{0,1\}^F\): a vector of \(F\) binary hidden units.
- \(W\): the visible–hidden weight matrix.
- \(a,b\): bias parameter vectors.
- \(\theta=\{W,a,b\}\): the complete parameter set.

The notation \(\{0,1\}^D\) means a vector of length \(D\) whose entries are 0 or 1. A superscript \({}^\top\) denotes transpose. \(P\) denotes probability; a vertical bar denotes conditioning, and a semicolon separates variables from parameters.

The RBM defines
\[
P(h,v;\theta)
=
\frac{1}{Z(\theta)}
\exp\!\left(v^\top Wh+a^\top h+b^\top v\right).
\]
The **partition function** \(Z(\theta)\) is the normalization constant that makes the probabilities sum to one.

A dropout RBM adds a binary vector
\[
r\in\{0,1\}^F.
\]
Each \(r_j\) independently equals 1 with probability \(p\). If \(r_j=1\), hidden unit \(h_j\) is retained. Otherwise, it is dropped.

The joint distribution is
\[
P(r,h,v;p,\theta)
=
P(r;p)P(h,v\mid r;\theta),
\]
with
\[
P(r;p)
=
\prod_{j=1}^{F}p^{r_j}(1-p)^{1-r_j}.
\]
Here, \(\prod\) means multiply the indexed terms.

Conditioned on the mask,
\[
P(h,v\mid r;\theta)
=
\frac{1}{Z'(\theta,r)}
\exp\!\left(v^\top Wh+a^\top h+b^\top v\right)
\prod_{j=1}^{F}g(h_j,r_j),
\]
where
\[
g(h_j,r_j)
=
\mathbf1(r_j=1)
+
\mathbf1(r_j=0)\mathbf1(h_j=0).
\]
The indicator \(\mathbf1(\text{condition})\) is 1 when its condition holds and 0 otherwise. The factor \(g\) requires \(h_j=0\) when \(r_j=0\). The quantity \(Z'(\theta,r)\) normalizes the masked model.

Conditioned on \(v\) and \(r\), hidden units have a **factorial distribution**: their conditional probabilities multiply independently,
\[
P(h\mid r,v)
=
\prod_{j=1}^{F}P(h_j\mid r_j,v),
\]
\[
P(h_j=1\mid r_j,v)
=
\mathbf1(r_j=1)
\sigma\!\left(b_j+\sum_iW_{ij}v_i\right).
\]
Here, \(\sum\) means add the indexed terms, \(W_{ij}\) is a matrix entry, and \(\sigma\) is the logistic function defined in Section 4.

The visible conditional is the same as in an RBM:
\[
P(v\mid h)
=
\prod_{i=1}^{D}P(v_i\mid h),
\]
\[
P(v_i=1\mid h)
=
\sigma\!\left(a_i+\sum_jW_{ij}h_j\right).
\]

Conditioned on \(r\), this is an RBM with the hidden units having \(r_j=0\) removed. The dropout RBM is therefore a mixture—a probability-weighted combination—of exponentially many weight-sharing RBMs, each using a different hidden subset.

### 8.2 Learning Dropout RBMs

Existing RBM learning algorithms, such as **Contrastive Divergence** (Hinton et al., 2006), apply directly. Contrastive Divergence is the RBM training procedure used here.

The only change is to sample \(r\) first and train only retained hidden units. As in dropout networks, sample a different mask for every training case in every mini-batch.

Our experiments use **CD-1**, the one-step Contrastive Divergence setting.

### 8.3 Effect on Features

Dropout improved feed-forward features by reducing co-adaptations. We examine whether the effect transfers to RBMs.

We compare a standard binary RBM, whose units take binary values, with a dropout RBM. Both have 256 hidden units and are trained on MNIST.

**Figure 12: Features learned on MNIST by 256 hidden unit RBMs. The features are ordered by L2 norm.** Panel (a): “Without dropout.” Panel (b): “Dropout with \(p=0.5\).”

The dropout RBM’s features look qualitatively different. They appear coarser than the standard RBM’s sharply defined, stroke-like features. The dropout model also appears to have very few dead units—units that do not contribute active features—relative to the standard model.

### 8.4 Effect on Sparsity

We next examine hidden-activation sparsity. Figure 13a shows activation and mean-activation histograms on a test mini-batch after standard RBM training. Figure 13b shows the same for a dropout RBM.

**Figure 13: Effect of dropout on sparsity.** Panel (a), “Without dropout”: the activation histogram shows many units with activations away from zero. Panel (b), “Dropout with \(p=0.5\)”: many units have activations close to zero, and very few have high activation.

These histograms clearly indicate that dropout RBMs learn much sparser representations than standard RBMs, even without an additional sparsity-inducing regularizer.

## 9. Marginalizing Dropout

Dropout adds noise to hidden-unit states. We now consider models obtained by marginalizing, or averaging out, that noise.

These are deterministic versions of dropout. Unlike standard Monte-Carlo dropout, they do not need random bits, and gradients can be obtained for their marginalized loss functions.

Related work includes deterministic models robust to feature deletion at test time (Globerson and Roweis, 2006), marginalized denoising autoencoders (Chen et al., 2012), and dropout marginalization for linear regression (Srivastava, 2013).

Wang and Manning (2013) further explored marginalization to speed training. van der Maaten et al. (2013) studied regularizers produced by marginalizing different input-noise distributions. Wager et al. (2013) described dropout as an adaptive regularizer, one whose effect depends on the data.

### 9.1 Linear Regression

First consider linear regression, which predicts targets through a linear weighted combination of inputs.

Let:

- \(X\in\mathbb R^{N\times D}\) be the data matrix, with \(N\) data points and \(D\) input dimensions.
- \(\mathbb R\) denote the real numbers.
- \(y\in\mathbb R^N\) be the target vector.
- \(w\in\mathbb R^D\) be the weight vector.

Ordinary linear regression chooses \(w\) to minimize
\[
\|y-Xw\|^2,
\]
the squared Euclidean length of the prediction-error vector.

Retain every input dimension with probability \(p\). The corrupted input is \(R*X\), where
\[
R\in\{0,1\}^{N\times D},
\qquad
R_{ij}\sim\operatorname{Bernoulli}(p).
\]
Thus, \(R\) is a random binary matrix and \(*\) is element-wise multiplication.

Writing \(\mathbb E\) for expectation, or averaging over the random matrix, the marginalized objective is
\[
\underset{w}{\operatorname{minimize}}\quad
\mathbb E_{R\sim\operatorname{Bernoulli}(p)}
\left[\|y-(R*X)w\|^2\right].
\]

This reduces to
\[
\underset{w}{\operatorname{minimize}}\quad
\|y-pXw\|^2+p(1-p)\|\Gamma w\|^2,
\]
where
\[
\Gamma=\left(\operatorname{diag}(X^\top X)\right)^{1/2}.
\]
Here, \(\operatorname{diag}\) keeps the diagonal entries as a diagonal matrix, and the power \(1/2\) takes their square roots.

The first term is the prediction error with inputs scaled by \(p\). The second is a quadratic weight penalty. Dropout in linear regression is therefore equivalent, in expectation, to **ridge regression**, linear regression with a quadratic penalty, using this particular \(\Gamma\).

This \(\Gamma\) essentially scales the cost of weight \(w_i\) by the standard deviation of the \(i\)th data dimension. If a dimension varies greatly, the regularizer squeezes its weight more.

Absorbing \(p\) into the weights makes the regularization strength explicit. Define
\[
\widetilde w=pw.
\]
The objective becomes
\[
\underset{w}{\operatorname{minimize}}\quad
\|y-X\widetilde w\|^2+
\frac{1-p}{p}\|\Gamma\widetilde w\|^2.
\]

For \(p\) close to 1, all inputs are retained and the regularization constant is small. Decreasing \(p\) adds more dropout and increases the constant \((1-p)/p\).

### 9.2 Logistic Regression and Deep Networks

A closed-form marginalized model is difficult to obtain for logistic regression and deep networks.

Wang and Manning (2013) showed that marginalized dropout logistic regression can be trained approximately. Under reasonable assumptions, both the logistic unit’s inputs and the marginalized model’s gradients have Gaussian, or normal, distributions. Their means and variances can be computed efficiently.

This approximation outperforms Monte-Carlo dropout in training time and generalization performance. However, its assumptions become successively weaker as more layers are added. The results therefore do not directly apply to deep networks.

## 10. Multiplicative Gaussian Noise

Dropout multiplies activations by Bernoulli variables that are 1 with probability \(p\) and 0 otherwise. More generally, activations can be multiplied by random variables from other distributions.

We recently found that multiplying by a variable from \(\mathcal N(1,1)\) works just as well, or perhaps better, than Bernoulli noise. The notation \(\mathcal N(\mu,\sigma^2)\) denotes a Gaussian distribution with mean \(\mu\) and variance \(\sigma^2\); its standard deviation is \(\sigma\).

For a hidden activation \(h_i\), the perturbation can be written
\[
h_i+h_ir,
\qquad r\sim\mathcal N(0,1),
\]
or equivalently
\[
h_ir',
\qquad r'\sim\mathcal N(1,1).
\]
This adds zero-mean Gaussian noise with standard deviation equal to the unit’s activation.

We can generalize to
\[
r'\sim\mathcal N(1,\sigma^2),
\]
making \(\sigma\) a tunable hyperparameter, like \(p\) in Bernoulli dropout. The expected activation is unchanged, so no test-time weight scaling is needed.

Our Bernoulli description retains units with probability \(p\) during training and multiplies weights by \(p\) at test time. An equivalent implementation multiplies retained activations by \(1/p\) during training and leaves test-time weights unchanged. Equivalence requires appropriate scaling of each layer’s learning rate and weight initialization.

Under this alternative implementation, the Bernoulli multiplier \(r_b\) is \(1/p\) with probability \(p\) and 0 otherwise. Its expectation and variance are
\[
\mathbb E[r_b]=1,
\qquad
\operatorname{Var}[r_b]=\frac{1-p}{p}.
\]
Here, \(\operatorname{Var}\) measures variation around the mean.

Set the Gaussian noise variance to
\[
\sigma^2=\frac{1-p}{p}.
\]
Then its multiplier \(r_g\) has the same mean and variance:
\[
\mathbb E[r_g]=1,
\qquad
\operatorname{Var}[r_g]=\frac{1-p}{p}.
\]

With these first- and second-order moments—mean and variance—fixed, \(r_g\) has the highest entropy and \(r_b\) the lowest. Entropy measures a distribution’s uncertainty. Both extremes work well.

The preliminary results in Table 10 suggest that the high-entropy case might work slightly better. For each Gaussian layer, we set
\[
\sigma=\sqrt{\frac{1-p}{p}}
\]
using the corresponding Bernoulli layer’s \(p\).

| Data Set | Architecture | Bernoulli dropout | Gaussian dropout |
|---|---|---|---|
| MNIST | 2 layers, 1024 units each | 1.08 ± 0.04 | 0.95 ± 0.04 |
| CIFAR-10 | 3 conv + 2 fully connected layers | 12.6 ± 0.1 | 12.5 ± 0.1 |

**Table 10: Comparison of classification error % with Bernoulli and Gaussian dropout.** “Conv” means convolutional. For MNIST, the Bernoulli model uses \(p=0.5\) for hidden units and \(p=0.8\) for inputs. For CIFAR-10, it uses \(p=(0.9,0.75,0.75,0.5,0.5,0.5)\) from input to top. Gaussian models use \(\sigma=\sqrt{(1-p)/p}\). Results are averaged over 10 random seeds, which determine the random choices in separate runs. The paper does not specify what uncertainty measure the ± values represent.

## 11. Conclusion

Dropout improves neural networks by reducing overfitting. Standard backpropagation builds brittle co-adaptations that work on training data but fail on unseen data. Random dropout breaks them up by making any particular hidden unit’s presence unreliable.

Dropout improved performance in object classification, digit recognition, speech recognition, document classification, and computational biology. Tables 2–9 and the Reuters results provide the comparisons. This breadth suggests a general technique rather than a domain-specific one.

Methods using dropout achieve state-of-the-art results on SVHN, ImageNet, CIFAR-100, and MNIST. Dropout also considerably improves standard networks on other data sets.

The idea extends to RBMs and other graphical models. Its central procedure is to take a large model that overfits easily, then repeatedly sample and train smaller submodels. RBMs fit this framework. Our dropout RBMs empirically have desirable properties: Figures 12–13 show their different features, few apparent dead units, and sparse representations.

A drawback is longer training. A dropout network typically takes 2–3 times as long to train as an ordinary network with the same architecture.

A major reason is the noise in parameter updates. Each case effectively trains a different random architecture. Its gradients are therefore not gradients of the final test-time architecture. Long training is not surprising.

However, this stochasticity likely prevents overfitting. There is a trade-off between training time and overfitting: more training time permits higher dropout and less overfitting.

Marginalizing noise offers one way to obtain some benefits without stochasticity. For linear regression, Section 9.1 shows that the corresponding regularizer is a modified L2 penalty. An equivalent regularizer is not obvious for more complicated models. Speeding up dropout remains an interesting direction for future work.

## Acknowledgments

This research was supported by OGS, NSERC and an Early Researcher Award.

OGS and NSERC are the support-program labels used in the paper; they are not expanded here.

## Appendix A. A Practical Guide for Training Dropout Networks

Neural networks are infamous for extensive hyperparameter tuning. Dropout networks are no exception. The following are heuristics that might help when applying dropout.

### A.1 Network Size

Dropping units is expected to reduce capacity. If a layer has \(n\) hidden units and retention probability \(p\), it contains only \(pn\) units after dropout, in expectation. This subset changes each time, and its units cannot freely build co-adaptations.

Thus, if an \(n\)-unit layer is optimal without dropout, a good dropout layer should have at least \(n/p\) units. We found this heuristic useful for both convolutional and fully connected networks.

### A.2 Learning Rate and Momentum

Dropout adds substantial gradient noise beyond ordinary stochastic gradient descent. Many gradients therefore tend to cancel.

To compensate, a dropout network should typically use a learning rate 10–100 times the optimum for a standard network.

High momentum also reduces the effect of noise. Values of 0.9 are common in standard networks. With dropout, values around 0.95 to 0.99 work much better. High learning rate and/or momentum significantly speeds learning.

### A.3 Max-norm Regularization

Large learning rates and momentum speed learning but can make weights grow very large. Max-norm regularization prevents this by bounding each hidden unit’s incoming weight-vector norm by \(c\).

Typical values of \(c\) range from 3 to 4.

### A.4 Dropout Rate

Retention probability \(p\) controls dropout intensity. At \(p=1\), there is no dropout. Smaller \(p\) means more dropout.

Typical hidden-unit values range from 0.5 to 0.8. For inputs, the choice depends on their type. A typical value for real-valued inputs, such as image patches or speech frames, is 0.8.

For hidden layers, choose \(p\) together with the number of units \(n\). Smaller \(p\) requires larger \(n\), slows training, and leads to underfitting. Larger \(p\) may provide too little dropout to prevent overfitting.

## Appendix B. Detailed Description of Experiments and Data Sets

This appendix gives architectures and training details for the reported results. Reproduction code is available at:

<http://www.cs.toronto.edu/~nitish/dropout>

The implementation runs on a **GPU**, a graphics processing unit. It uses the CUDA libraries **cudamat** (Mnih, 2009) and **cuda-convnet** (Krizhevsky et al., 2012). These are software-library names; CUDA names their computing platform.

### B.1 MNIST

MNIST contains 60,000 training and 10,000 test examples, each a 28 × 28 digit image.

We randomly held out 10,000 training images for validation. Hyperparameters were selected so that the best validation error occurred after 1 million weight updates. We then recombined the training and validation sets, trained for 1 million updates, and evaluated that network on the test set.

We used this procedure because hyperparameters could easily be chosen so that early stopping was unnecessary. Once they were fixed, it made sense to combine the data and train for a long time.

Figure 4 includes every combination of 2, 3, and 4 hidden layers with either 1024 or 2048 units per layer: six architectures.

For all these architectures, including those in Table 2, we used:

- Hidden-layer retention \(p=0.5\).
- Input retention \(p=0.8\).
- Final momentum 0.95.
- Weight constraint \(c=2\) in every layer.

To test dropout’s regularization limits, we also trained 2- and 3-layer networks with 4096 and 8192 units. The 2-layer networks improved results, as Table 2 shows.

At the same dropout level, the 3-layer networks were slightly worse than the corresponding 2-layer networks. Increasing dropout improved them, but not enough to beat the 2-layer networks.

### B.2 SVHN

SVHN has approximately 600,000 training images and 26,000 test images. The training data have two parts: a standard labeled set and an extra set of easy labeled examples.

We constructed a validation set from both parts, using the procedure of Sermanet et al. (2012):

- Two-thirds came from the standard set: 400 examples per class.
- One-third came from the extra set: 200 examples per class.
- The total was 6000 examples.

Inputs were **RGB**, or red–green–blue, pixel values normalized to zero mean and unit variance. Global or local contrast normalization and ZCA whitening gave no noticeable improvement. Global contrast normalization adjusts image-channel intensities, local contrast normalization makes such adjustments locally, and ZCA whitening is the transformation defined in Appendix B.3.

The best architecture we found has three convolutional layers, each followed by max-pooling:

- The convolutional layers have 96, 128, and 256 filters.
- Each filter uses a 5 × 5 **receptive field**, the input region contributing to one response.
- Convolution uses a **stride**, or movement between successive filter locations, of 1 pixel.
- Each pooling layer combines 3 × 3 regions at strides of 2 pixels.
- Two fully connected hidden layers follow, each with 2048 units.
- All units use rectified linear activations.

Retention probabilities from input through the convolutional and fully connected layers are
\[
p=(0.9,0.75,0.75,0.5,0.5,0.5).
\]
We use max-norm \(c=4\) for all weights and momentum 0.95 in all layers.

Hyperparameters were selected on the validation set. Because the training set was large, we did not recombine training and validation data for final training. We report test error for the model with the smallest validation error.

### B.3 CIFAR-10 and CIFAR-100

Each data set contains 50,000 training images and 10,000 test images. CIFAR-10 has 10 categories; CIFAR-100 has 100. All images are 32 × 32 color images.

We held out 5,000 training images for validation. As for MNIST, we selected hyperparameters using validation data and then recombined training and validation data.

Preprocessing used global contrast normalization separately in each color channel, followed by ZCA whitening:

- **Global contrast normalization:** for each image and each color channel, compute the mean pixel intensity and subtract it from that channel.
- **ZCA whitening:** mean-center the data, rotate it onto its principal components, normalize each component, and rotate it back. ZCA is the whitening label used in the paper and is not expanded there.

The architecture and dropout probabilities are the same as for SVHN. Input-layer learning rates had to be smaller.

### B.4 TIMIT

We used the open-source **Kaldi** speech-processing toolkit (Povey et al., 2011) to convert speech into log-filter banks.

A **monophone system**, using individual phone units, was trained for **forced alignment**, which aligns the supplied speech labels with the recording and provides frame labels. Dropout networks receive windows of 21 consecutive frames and predict the central frame’s label.

No speaker-dependent operations were performed. Inputs were mean-centered and normalized to unit variance.

We used:

- Input retention \(p=0.8\).
- Hidden-layer retention \(p=0.5\).
- Max-norm \(c=4\) in every layer.
- Momentum 0.95.
- A high learning rate of 0.1.

The learning rate decayed according to
\[
\epsilon_0(1+t/T)^{-1}.
\]
The initial learning rate is \(\epsilon_0=0.1\). The paper does not separately define \(t\) and \(T\) or give their values.

For DBN pretraining, RBMs were trained with CD-1. The Gaussian RBM, the RBM variant used for real-valued inputs, had each input unit’s variance fixed at 1.

For the best dropout finetuning results, a smaller learning rate, about 0.01, was important. Adding max-norm constraints gave no improvement.

### B.5 Reuters

Reuters RCV1 contains more than 800,000 documents in 103 classes arranged in a tree hierarchy.

We created a subset with:

- 402,738 articles.
- A vocabulary of 2000 words.
- 50 categories.
- Exactly one class per document.

We split it into equally sized training and test sets.

We tried many architectures. Dropout improved classification accuracy for all of them. However, the improvement was smaller than on image and speech data.

This might be because the data set is large—more than 200,000 training examples—and overfitting is not a very serious problem.

### B.6 Alternative Splicing

The data describe 3665 cassette exons, 1014 RNA features, and 4 tissue types derived from 27 mouse tissues. A cassette exon is the exon unit whose splicing outcome is represented by the task.

Each input has 4 target **softmax units**, one per tissue type. A softmax unit represents a probability distribution over mutually exclusive states. Each has 3 biologically important states labeled **inc**, **exc**, and **nc**. The paper does not expand these codes.

For each unit, the aim is to match the distribution over these states observed in wet-lab experiments, meaning physical biological experiments.

To define Code Quality, let:

- \(i\) index data points, with \(|\text{data points}|\) denoting their count.
- \(t\) index tissue types.
- \(s\in\{\mathrm{inc},\mathrm{exc},\mathrm{nc}\}\) index states.
- \(r_i\) be input \(i\).
- \(p_{i,t}^s\) be the target probability of state \(s\) in tissue \(t\) for input \(i\).
- \(q_t^s(r_i)\) be the model’s corresponding predicted probability.
- \(\bar p^s\) be the average of \(p_{i,t}^s\) over inputs \(i\) and tissues \(t\).
- \(\log\) denote the logarithm.

Code Quality is
\[
\sum_{i=1}^{|\text{data points}|}
\sum_{t\in\text{tissue types}}
\sum_{s\in\{\mathrm{inc},\mathrm{exc},\mathrm{nc}\}}
p_{i,t}^s
\log\!\left(\frac{q_t^s(r_i)}{\bar p^s}\right).
\]

We trained a two-layer dropout network with 1024 units in each layer. Hidden layers used \(p=0.5\); the input used \(p=0.7\). Training used max-norm regularization and high, decaying learning rates.

Results were averaged over the same 5 folds used by Xiong et al. (2011), where folds are the data partitions used for repeated training and evaluation.
