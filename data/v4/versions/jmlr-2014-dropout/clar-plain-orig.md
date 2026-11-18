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

Deep neural networks with many parameters are powerful machine learning systems. These networks learn connections among layers of processing units. However, they face a serious problem: **overfitting**, in which a model learns patterns in its training data that do not hold in new data. Large networks also make predictions slowly. This makes it hard to address overfitting by combining predictions from many large networks at test time.

**Dropout** addresses this problem by randomly removing units and their connections during training. This prevents units from **co-adapting** too much—that is, becoming too dependent on one another. During training, dropout samples from an exponential number of different “thinned” networks. At test time, a single unthinned network with smaller connection weights can approximate the average predictions of all these networks. This greatly reduces overfitting and gives major improvements over other **regularization methods**, which aim to limit overfitting.

We show that dropout improves neural networks on **supervised learning** tasks, where training examples include target outputs. These tasks cover vision, speech recognition, document classification, and computational biology. Dropout achieves state-of-the-art results on many benchmark data sets.

**Keywords:** neural networks, regularization, model combination, deep learning

## 1. Introduction

Deep neural networks contain several **non-linear hidden layers**: intermediate layers whose outputs are non-linear functions of their inputs. These layers let networks learn complex relationships between inputs and outputs. With limited training data, however, many of these relationships arise from **sampling noise**, or chance patterns in the sampled data. Such patterns exist in the training set but not in real test data, even when both sets come from the same distribution. This causes overfitting.

Researchers have developed many ways to reduce overfitting. One method stops training when performance starts to worsen on a **validation set**, a separate set used to guide training choices. Other methods penalize connection weights. These include **L1 regularization**, which penalizes absolute weight values; **L2 regularization**, which penalizes squared weight values; and **soft weight sharing**, which encourages weights to form groups with similar values (Nowlan and Hinton, 1992).

With unlimited computation, the best way to regularize a model of fixed size is to average the predictions from all possible parameter settings. Each setting receives a weight equal to its **posterior probability**, its probability after accounting for the training data. This approach can sometimes be approximated well for simple or small models (Xiong et al., 2011; Salakhutdinov and Mnih, 2008). We want to approach this Bayesian gold standard with much less computation. We propose to approximate an equally weighted **geometric mean** of predictions from an exponential number of learned models that share parameters. A geometric mean combines values by multiplying them and taking the appropriate root.

© 2014 Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever and Ruslan Salakhutdinov.

**Figure 1: Dropout Neural Net Model.**  
(a) Standard Neural Net. (b) After applying dropout.  
Left: A standard neural network with 2 hidden layers. Right: An example of a thinned network produced by applying dropout to the network on the left. Crossed units have been dropped.

**Model combination**, which combines predictions from several models, nearly always improves machine learning performance. For large neural networks, however, averaging outputs from many separately trained networks costs too much. Combining models helps most when the models differ. To make neural networks different, we should give them different architectures—their arrangements of layers and connections—or train them on different data.

Training many architectures is hard. Each architecture requires a difficult search for good **hyperparameters**, settings chosen outside the parameter-learning process. Training each large network also requires substantial computation. Large networks usually need much training data, and there may not be enough data to train different networks on different subsets. Even if we could train many large networks, using all of them at test time would be impractical when applications must respond quickly.

Dropout addresses both problems. It prevents overfitting and efficiently approximates the combination of exponentially many neural network architectures. “Dropout” means temporarily removing hidden or **visible units**, the units that represent observed inputs, together with all their incoming and outgoing connections. Figure 1 shows this process.

We choose the dropped units randomly. In the simplest case, we retain each unit independently with a fixed probability, \(p\). We can choose \(p\) using a validation set or simply set it to 0.5. This value seems close to optimal across many networks and tasks. For input units, however, the best retention probability is usually closer to 1 than to 0.5.

**Figure 2:**  
(a) At training time: the unit is present with probability \(p\), and its outgoing connections have weights \(w\).  
(b) At test time: the unit is always present, and its outgoing connections have weights \(pw\).  
Left: A training-time unit is present with probability \(p\) and connects to units in the next layer with weights \(w\). Right: At test time, the unit is always present, and we multiply its weights by \(p\). Its test-time output equals its expected training-time output.

Applying dropout samples a thinned network from the full network. The thinned network contains all units that survive dropout, as Figure 1b shows. A network with \(n\) units represents a collection of \(2^n\) possible thinned networks. They all share weights, so the total parameter count remains \(O(n^2)\), or less. Here, \(O(n^2)\) means that the parameter count grows no faster than quadratically with the number of units.

Each time we present a training case, we sample and train a new thinned network. We can therefore view dropout training as training \(2^n\) thinned networks with extensive weight sharing. Each individual thinned network receives training very rarely, if at all.

At test time, we cannot explicitly average predictions from exponentially many thinned models. A simple approximation nevertheless works well in practice. We use one network without dropout, with scaled-down versions of the trained weights. If training retains a unit with probability \(p\), we multiply that unit’s outgoing weights by \(p\) at test time, as Figure 2 shows. This makes each hidden unit’s actual test-time output equal to its expected output under the training-time dropout distribution.

This scaling combines \(2^n\) networks with shared weights into one test-time network. Across many classification problems, dropout training with this approximate test-time averaging gives much lower **generalization error**, or error on new data, than training with other regularization methods.

Dropout also applies beyond **feed-forward neural networks**, in which information moves from inputs toward outputs without feedback connections. It can apply more broadly to **graphical models**, which represent relationships among variables through a graph, such as **Boltzmann Machines**. These models use connected units to describe probability distributions. We introduce the dropout **Restricted Boltzmann Machine**, or **RBM**, a Boltzmann Machine with connections restricted to run between visible and hidden units. We compare it with standard RBMs. Our experiments show that dropout RBMs improve on standard RBMs in certain respects.

The paper proceeds as follows. Section 2 explains the motivation. Section 3 reviews related work. Section 4 formally defines the dropout model. Section 5 gives a training algorithm. Section 6 presents experiments across several domains and compares dropout with other forms of regularization and model combination. Section 7 examines how dropout changes network properties and interacts with hyperparameters. Section 8 presents the Dropout RBM. Section 9 explores **marginalizing dropout**, or averaging over dropout noise rather than sampling it. Appendix A gives a practical training guide, including a detailed discussion of hyperparameter choices.

## 2. Motivation

One motivation for dropout comes from a theory about the role of sex in evolution (Livnat et al., 2010). Sexual reproduction takes half the genes from each parent, adds a very small amount of random mutation, and combines the genes into an offspring. Asexual reproduction instead produces an offspring with a slightly mutated copy of one parent’s genes.

Asexual reproduction might seem better for optimizing individual fitness. It can pass directly to an offspring a good set of genes that work well together. Sexual reproduction, by contrast, tends to break up these co-adapted sets, especially large ones. Intuitively, this should reduce the fitness of organisms that have already evolved complex co-adaptations. Yet most advanced organisms evolved through sexual reproduction.

One possible explanation is that natural selection may favor the **mix-ability** of genes over individual fitness in the long run. Genes become more robust when they can work well with another random set of genes. A gene cannot count on having a large set of partners present at all times. It must therefore do something useful alone or with a small number of other genes.

According to this theory, sexual reproduction does more than spread useful new genes through a population. It also helps this spread by reducing complex co-adaptations that would lower the chance that a new gene improves an individual’s fitness.

Similarly, dropout forces each hidden unit to work with a randomly chosen sample of other units. This should make each unit more robust. It should also encourage the unit to create useful features on its own rather than rely on other hidden units to correct its mistakes. A **feature** is a pattern or property that a unit detects or represents.

Hidden units within a layer will still learn different functions. One might expect a network to resist dropout by making many copies of each hidden unit. But this is a poor solution for the same reason that **replica codes**, which repeat information, are a poor way to handle a noisy communication channel.

A related but slightly different motivation comes from successful conspiracies. Ten conspiracies of five people each probably create havoc more reliably than one conspiracy that requires fifty people to play their parts correctly. A large conspiracy can work well when conditions remain stable and people have plenty of time to rehearse. Under changing conditions, however, a smaller conspiracy has a better chance of continuing to work.

Likewise, training can make complex co-adaptations work well on the training set. On new test data, these complex arrangements are much more likely to fail than several simpler co-adaptations that achieve the same result.

## 3. Related Work

We can interpret dropout as regularizing a neural network by adding noise to its hidden units. Vincent et al. (2008, 2010) previously added noise to unit states in **Denoising Autoencoders**, or **DAEs**. An **autoencoder** learns to reconstruct its input. A denoising autoencoder receives a noisy input and learns to reconstruct the noise-free input.

Our work extends this idea. We show that dropout works in hidden layers and that we can interpret it as model averaging. We also show that adding noise helps not only **unsupervised feature learning**, which learns features without target labels, but supervised learning as well. The method also applies to other neuron-based architectures, such as Boltzmann Machines.

Noise of 5% typically works best for DAEs. Our test-time weight scaling lets us use much higher noise levels. We often found that dropping 20% of input units and 50% of hidden units was optimal.

Dropout is **stochastic**, meaning that it uses random choices. It is therefore natural to consider a **deterministic** counterpart, which produces its result without such random sampling. We can obtain this counterpart by marginalizing, or averaging out, the noise. We show that simple cases allow us to marginalize dropout analytically and obtain deterministic regularization methods.

Recently, van der Maaten et al. (2013) explored deterministic regularizers for several **exponential-family noise distributions**, a class of probability distributions. These include dropout, which they call “blankout noise.” Their work adds noise only to inputs and considers only models without hidden layers. Wang and Manning (2013) proposed speeding up dropout by marginalizing its noise. Chen et al. (2012) studied marginalization for denoising autoencoders.

Dropout stochastically minimizes a **loss function**, which measures how poorly a model predicts, under a noise distribution. We can view this as minimizing an expected loss. Globerson and Roweis (2006) and Dekel et al. (2010) studied a different setting: minimizing loss when an adversary chooses which units to drop. That setting fixes the maximum number of dropped units rather than using a noise distribution. This work also does not consider models with hidden units.

## 4. Model Description

This section defines the dropout neural network model. Consider a network with \(L\) hidden layers. The index \(l \in \{1,\ldots,L\}\) identifies a hidden layer. Let \(z^{(l)}\) be the vector of inputs to layer \(l\), and let \(y^{(l)}\) be its output vector. The network input is \(y^{(0)}=x\).

The symbols \(W^{(l)}\) and \(b^{(l)}\) denote the weights and **biases** at layer \(l\). A bias is an added offset. For \(l \in \{0,\ldots,L-1\}\) and any hidden unit \(i\), a standard network computes its forward pass as follows:

\[
z_i^{(l+1)}
=
w_i^{(l+1)}y^{(l)}+b_i^{(l+1)},
\]

\[
y_i^{(l+1)}
=
f\!\left(z_i^{(l+1)}\right).
\]

Here, \(f\) is an **activation function**, which converts a unit’s input into its output. One example is the logistic function:

\[
f(x)=\frac{1}{1+\exp(-x)}.
\]

With dropout, the forward pass becomes:

\[
r_j^{(l)}\sim\operatorname{Bernoulli}(p),
\]

\[
\widetilde y^{(l)}
=
r^{(l)}*y^{(l)},
\]

\[
z_i^{(l+1)}
=
w_i^{(l+1)}\widetilde y^{(l)}+b_i^{(l+1)},
\]

\[
y_i^{(l+1)}
=
f\!\left(z_i^{(l+1)}\right).
\]

**Figure 3: Comparison of the basic operations of a standard and dropout network.**  
(a) Standard network. (b) Dropout network.  
The standard network passes outputs \(y_1^{(l)},y_2^{(l)},y_3^{(l)}\) through weights \(w_i^{(l+1)}\), adds bias \(b_i^{(l+1)}\) from a \(+1\) input, and applies \(f\) to \(z_i^{(l+1)}\) to produce \(y_i^{(l+1)}\). The dropout network first multiplies each output by its corresponding \(r_j^{(l)}\), producing \(\widetilde y_j^{(l)}\), and then performs the same operations.

The symbol \(*\) means **element-wise multiplication**, which multiplies matching vector entries. A **Bernoulli random variable** equals 1 with probability \(p\) and 0 otherwise. For each layer \(l\), the vector \(r^{(l)}\) contains independent Bernoulli random variables.

We sample this vector and multiply it element by element with the layer’s output, \(y^{(l)}\), to obtain the thinned output, \(\widetilde y^{(l)}\). This output becomes the next layer’s input. Applying the process at every layer samples a sub-network from the full network.

During learning, we **backpropagate** the loss derivatives through this sub-network. Backpropagation computes how changes in earlier parameters affect the loss by working backward through the network. At test time, we scale the weights as

\[
W_{\mathrm{test}}^{(l)}=pW^{(l)},
\]

as Figure 2 shows. We then use the resulting network without dropout.

## 5. Learning Dropout Nets

This section describes how to train dropout neural networks.

### 5.1 Backpropagation

We can train dropout networks with **stochastic gradient descent**, much as we train standard networks. This method updates parameters using gradients—directions of change in the loss—estimated from training cases. A **mini-batch** is a small group of cases used for an update.

The only difference is that we sample a thinned network for each training case in a mini-batch. We perform the forward pass and backpropagation only through that network. We average each parameter’s gradients across the cases in the mini-batch. A case that does not use a parameter contributes a zero gradient for that parameter.

Several methods improve stochastic gradient descent. These include **momentum**, which carries part of previous updates into the next update; **annealed learning rates**, which decrease the update step size during training; and **L2 weight decay**, which shrinks weights through an L2 penalty. These methods also helped dropout networks.

One form of regularization proved especially useful: limiting the norm of each hidden unit’s incoming weight vector to a fixed constant \(c\). The **L2 norm** measures a vector’s length as the square root of the sum of its squared entries. If \(w\) is a hidden unit’s incoming weight vector, we optimize under the constraint

\[
\|w\|_2\leq c.
\]

Whenever an update moves \(w\) outside the ball of radius \(c\), we project it back onto the ball’s surface. This method is called **max-norm regularization**, because the maximum allowed norm of any weight is \(c\).

We choose the hyperparameter \(c\) using a validation set. Max-norm regularization has also been used in **collaborative filtering**, which predicts preferences from patterns across users and items (Srebro and Shraibman, 2005). It usually improves stochastic-gradient training of deep neural networks even without dropout.

Dropout alone gives substantial improvements. Combining it with max-norm regularization, large decaying learning rates, and high momentum gives a further substantial gain.

One possible explanation is that bounding weight vectors within a fixed-radius ball allows a huge learning rate without weights growing without bound. Dropout noise then lets optimization explore regions of weight space that would otherwise be hard to reach. As the learning rate falls, the updates take shorter steps. Exploration decreases, and optimization eventually settles into a minimum.

### 5.2 Unsupervised Pretraining

Neural networks can undergo **pretraining**, an initial learning stage, using stacks of RBMs (Hinton and Salakhutdinov, 2006), autoencoders (Vincent et al., 2010), or **Deep Boltzmann Machines**, which extend Boltzmann Machines to several hidden layers (Salakhutdinov and Hinton, 2009). Pretraining makes effective use of unlabeled data. In some cases, pretraining followed by **finetuning**, or further training, with backpropagation gives much better performance than finetuning from random initial weights.

We can use dropout to finetune networks pretrained by these methods. The pretraining process stays unchanged. We should multiply the pretrained weights by \(1/p\). This makes each unit’s expected output under random dropout equal to its output during pretraining.

We initially worried that dropout’s randomness might erase information stored in the pretrained weights. This happened when finetuning learning rates were comparable to the best rates for randomly initialized networks. With smaller learning rates, however, the pretrained information seemed to remain. Dropout then improved the final generalization error over finetuning without dropout.

## 6. Experimental Results

We trained dropout networks on classification problems across several domains. On every data set, dropout improved generalization over neural networks without dropout. Table 1 summarizes the data sets:

- **MNIST:** A standard toy data set of handwritten digits.
- **TIMIT:** A standard benchmark for clean speech recognition.
- **CIFAR-10 and CIFAR-100:** Tiny natural images (Krizhevsky, 2009).
- **Street View House Numbers (SVHN):** House-number images collected by Google Street View (Netzer et al., 2011).
- **ImageNet:** A large collection of natural images.
- **Reuters-RCV1:** A collection of Reuters newswire articles.
- **Alternative Splicing data set:** RNA features used to predict alternative gene splicing (Xiong et al., 2011). RNA means ribonucleic acid; alternative splicing produces different combinations of parts from an RNA transcript.

We chose diverse data sets to show that dropout is a general method for improving neural networks, rather than a method limited to one application domain. This section presents key results. Appendix B describes all experiments and data sets in more detail.

In the tables, K means thousand and M means million. “Dimensionality” is the number of input components.

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

We evaluated dropout on five image data sets: MNIST, SVHN, CIFAR-10, CIFAR-100, and ImageNet. These sets vary in image type and training-set size. Models that achieve state-of-the-art results on all these data sets use dropout.

#### 6.1.1 MNIST

Table 2 uses several model names. **NN** means neural network. A **support vector machine (SVM)** is a classifier that separates classes using a decision boundary; a **Gaussian kernel** lets it model non-linear boundaries through Gaussian-based similarities. **NA** means not applicable.

A **rectified linear unit (ReLU)** outputs zero for a negative input and otherwise outputs the input. A **maxout unit** outputs the largest of several learned linear responses. A **Deep Belief Network (DBN)** is a layered model that can be pretrained using stacked RBMs. **DBM** means Deep Boltzmann Machine.

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

MNIST contains 28 × 28 pixel images of handwritten digits. The task assigns each image to one of 10 digit classes. Table 2 compares dropout with other methods.

In the **permutation invariant setting**, the model does not use the pixels’ two-dimensional spatial arrangement. Without dropout or unsupervised pretraining, the best neural networks in this setting achieve about 1.60% error (Simard et al., 2003). Dropout reduces this to 1.35%. Replacing logistic units with ReLUs (Jarrett et al., 2009) reduces it further to 1.25%. Adding max-norm regularization lowers it again to 1.06%.

Larger networks perform better. A network with 2 layers and 8192 units per layer reaches 0.95% error. This network has more than 65 million parameters and trains on only 60,000 examples. Standard regularization and early stopping make it very hard to train a network this large to generalize well. Dropout prevents overfitting even here. It does not even need early stopping.

Goodfellow et al. (2013) improved the error to 0.94% by replacing ReLUs with maxout units. All dropout networks use \(p=0.5\) for hidden units and \(p=0.8\) for input units. Appendix B.1 gives more experimental details.

Dropout also improves networks pretrained with stacks of RBMs or Deep Boltzmann Machines, as Table 2 shows. DBM-pretrained dropout networks reach 0.79% test error, the best reported performance for the permutation invariant setting.

Better results are possible when models use two-dimensional spatial information and augment the training set with distorted versions of its images. **Data augmentation** adds modified versions of training examples. We demonstrate dropout’s effectiveness in this setting on more interesting data sets.

To test dropout’s robustness, we ran classification experiments with many architectures while holding every hyperparameter, including \(p\), fixed. Figure 4 shows test error as training proceeds. The same architectures have sharply different errors with and without dropout, producing two separate groups of curves. Dropout gives a large improvement across all architectures without separately tuned hyperparameters.

**Figure 4: Test error for different architectures with and without dropout.** The networks have 2 to 4 hidden layers, each with 1024 to 2048 units. The horizontal axis shows the number of weight updates: 0, 200000, 400000, 600000, 800000, and 1000000. The vertical axis shows classification error percentages, with labels 1.0, 1.5, 2.0, and 2.5. Curves distinguish “Without dropout” from “With dropout.”

#### 6.1.2 Street View House Numbers

The Street View House Numbers data set (Netzer et al., 2011) contains color images of house numbers from Google Street View. Figure 5a gives examples. We use 32 × 32 color images roughly centered on one digit in a house number. The task identifies that digit.

We applied dropout to **convolutional neural networks** (LeCun et al., 1989). These networks apply learned filters across local regions of an input. The best architecture we found has three convolutional layers followed by 2 **fully connected hidden layers**, where every unit connects to every unit in the preceding layer. All hidden units were ReLUs.

Each convolutional layer was followed by a **max-pooling layer**, which keeps the largest response in each local region. Appendix B.2 describes the architecture in more detail.

Table 3 also includes **pooling** methods that combine responses within regions: average pooling takes an average, L2 and L4 pooling use the corresponding norms, and **stochastic pooling** randomly selects responses. **Padding** adds a border around an input. **HOG**, or histograms of oriented gradients, represents patterns of image-edge directions. **KMeans** groups examples around learned cluster centers. Stacked sparse autoencoders are layered autoencoders trained to keep most unit activations low. WDCH labels the binary-feature method reported by Netzer et al. (2011).

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

**Table 3: Results on the Street View House Numbers data set.** “Conv Net” means convolutional neural network.

We applied dropout to every layer. Retention probabilities were

\[
p=(0.9,0.75,0.75,0.5,0.5,0.5),
\]

ordered from the input through the convolutional layers to the fully connected layers. We used max-norm regularization for weights in both types of layer.

Table 3 compares the methods. Convolutional networks outperform other methods. The best convolutional networks without dropout reach 3.95% error. Adding dropout only to the fully connected layers reduces error to 3.02%. Adding dropout to the convolutional layers reduces it further to 2.55%. Maxout units provide further gains.

The improvement from 3.02% to 2.55% after adding dropout to convolutional layers is worth noting. One might expect little benefit because these layers have relatively few parameters and thus might not overfit. Dropout in lower layers still helps, however, because it supplies noisy inputs to the higher fully connected layers and prevents them from overfitting.

#### 6.1.3 CIFAR-10 and CIFAR-100

CIFAR-10 and CIFAR-100 contain 32 × 32 color images from 10 and 100 categories, respectively. Figure 5b gives examples. Appendix B.3 details the data sets, input preprocessing, architectures, and experiments.

Table 4 compares error rates. Without data augmentation, Snoek et al. (2012) used **Bayesian hyperparameter optimization**, a probabilistic method for choosing hyperparameters, to reach 14.98% error on CIFAR-10. Dropout in the fully connected layers reduces error to 14.32%. Dropout in every layer reduces it further to 12.61%.

Goodfellow et al. (2013) reduced error again to 11.68% by replacing ReLUs with maxout units. On CIFAR-100, dropout gives a large improvement, reducing error from 43.48% to 37.20%. Neither data set used data augmentation apart from input dropout.

**Figure 5: Samples from image data sets. Each row corresponds to a different category.**  
(a) Street View House Numbers (SVHN). (b) CIFAR-10.

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

ImageNet contains over 15 million labeled high-resolution images in roughly 22,000 categories. Since 2010, an annual competition called the **ImageNet Large-Scale Visual Recognition Challenge (ILSVRC)** has taken place as part of the Pascal Visual Object Challenge. It uses an ImageNet subset with roughly 1000 images in each of 1000 categories.

Because there are so many categories, results conventionally report two errors. **Top-1 error** measures whether the most probable label is wrong. **Top-5 error** is the fraction of test images whose correct label is absent from the model’s five most probable labels. Figure 6 shows predictions on several test images.

ILSVRC-2010 is the only ILSVRC version with available test labels, so we ran most experiments on it. Table 5 compares methods. Convolutional networks with dropout outperform other methods by a large margin. Krizhevsky et al. (2012) describe the architecture and implementation in detail.

**Figure 6: Some ImageNet test cases with the 4 most probable labels predicted by our model.** Horizontal-bar lengths are proportional to the probabilities the model assigns to the labels. Pink marks the ground truth, or correct label.

In Tables 5 and 6, **sparse coding** represents inputs using a small number of active components. **SIFT**, the Scale-Invariant Feature Transform, describes local image patterns. **Fisher Vectors (FVs)** summarize image features through a statistical model. **Dense SIFT** extracts SIFT features across an image. **LBP**, or local binary patterns, describes local texture; **GIST** summarizes overall image structure; and **CSIFT** is a color-based SIFT descriptor. Color statistics summarize image colors.

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

**Table 6: Results on the ILSVRC-2012 validation/test set.** “Val” means validation, and “Avg” means average.

Our convolutional-network model with dropout won ILSVRC-2012. Because test labels are unavailable, we report test results for the final submission and validation results for model variations. Table 6 gives the competition results.

The best methods based on standard vision features reach about 26% top-5 error. Convolutional networks with dropout reach about 16% test error, a striking difference. Figure 6 gives example predictions. The model makes very reasonable predictions even when its best guess is wrong.

### 6.2 Results on TIMIT

We next applied dropout to speech recognition. TIMIT contains recordings of 680 speakers covering 8 major dialects of American English. Each speaker reads ten phonetically rich sentences in a controlled, noise-free environment.

We trained dropout networks on windows of 21 **log-filter bank frames**, short speech segments represented by logarithmic filter-bank measurements. The task predicts the central frame’s label. We performed no speaker-dependent operations. Appendix B.4 describes preprocessing and training.

Table 7 compares the models using **phone error rate**, which measures errors in recognizing speech sounds. A 6-layer network reaches 23.4%. Dropout improves this to 21.8%.

We also trained dropout networks from pretrained weights. A 4-layer network pretrained with stacked RBMs reaches 22.7% phone error. Dropout reduces this to 19.7%. For an 8-layer network, dropout similarly reduces error from 20.5% to 19.7%.

In Table 7, **mcRBM** means mean-covariance Restricted Boltzmann Machine, an RBM model that represents means and covariances.

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

To test dropout on text, we trained a document classifier on a subset of Reuters-RCV1. The full data set contains over 800,000 Reuters newswire articles on many topics.

The task assigns a document to one of 50 non-overlapping topics from its **bag of words representation**, which records words without their order. Appendix B.5 describes the setup.

Our best network without dropout reached 31.05% error. Dropout reduced this to 29.62%. This improvement was much smaller than the gains on vision and speech data.

### 6.4 Comparison with Bayesian Neural Networks

We can view dropout as equally weighted averaging over exponentially many models with shared weights. **Bayesian neural networks** instead perform proper model averaging over network structures and parameters (Neal, 1996). Dropout gives every model equal weight. A Bayesian network weights each model using its **prior**, or probability before observing the data, and how well it fits the data. The Bayesian weighting is more correct.

Bayesian neural networks are especially useful where data are scarce, including medical diagnosis, genetics, drug discovery, and other computational biology applications. However, they train slowly and do not scale easily to very large networks. Obtaining predictions from many large networks at test time is also expensive. Dropout networks train and predict much faster.

Here, we compare the two approaches on a small data set where Bayesian neural networks perform well and achieve state-of-the-art results. We aim to measure how much performance dropout loses relative to Bayesian networks.

The genetics data set comes from Xiong et al. (2011). The task predicts alternative splicing from RNA features. Alternative splicing is a major cause of cellular diversity in mammalian tissues. Predicting it in particular tissues under different conditions matters for understanding many human diseases.

Given RNA features, the model predicts probabilities for three splicing-related events of biological interest. We evaluate predictions using **Code Quality**, a measure of the negative **Kullback–Leibler (KL) divergence** between target and predicted probability distributions. KL divergence measures how one probability distribution differs from another. Higher Code Quality is better. Appendix B.6 describes the data and metric.

Table 8 also uses **principal component analysis (PCA)**, which reduces input dimensionality by representing data along directions of variation, and **regression**, which predicts outputs from input variables.

| Method | Code Quality (bits) |
|---|---|
| Neural Network (early stopping) (Xiong et al., 2011) | 440 |
| Regression, PCA (Xiong et al., 2011) | 463 |
| SVM, PCA (Xiong et al., 2011) | 487 |
| Neural Network with dropout | 567 |
| Bayesian Neural Network (Xiong et al., 2011) | 623 |

**Table 8: Results on the Alternative Splicing Data Set.**

Table 8 summarizes the results. Xiong et al. (2011) used Bayesian neural networks for this task. As expected, Bayesian networks outperform dropout. Dropout nevertheless greatly improves on standard neural networks and beats all other methods.

The main challenge is avoiding overfitting on the small training set. One approach reduces input dimensionality with PCA, then applies standard methods such as SVMs or **logistic regression**, which models class probabilities through a logistic function. Dropout prevented overfitting without dimensionality reduction.

The dropout networks are very large, with 1000s of hidden units, compared with a few tens of units in the Bayesian network. This shows dropout’s strong regularizing effect.

### 6.5 Comparison with Standard Regularizers

Many regularization methods aim to prevent neural-network overfitting. They include L2 weight decay, which is a form of **Tikhonov regularization** (Tikhonov, 1943); **lasso**, which uses an L1 penalty (Tibshirani, 1996); **KL-sparsity**, which uses KL divergence to encourage low unit activity; and max-norm regularization. We compare dropout with some of these methods on MNIST.

We trained the same ReLU network architecture, 784-1024-1024-2048-10, using stochastic gradient descent with different regularizers. This notation lists successive layer sizes. Table 9 gives the results.

We selected each method’s hyperparameters with a validation set. These included decay constants, the target level of **sparsity** or low activity, dropout rate, and max-norm bound. Dropout combined with max-norm regularization gave the lowest generalization error.

| Method | Test Classification error % |
|---|---|
| L2 | 1.62 |
| L2 + L1 applied towards the end of training | 1.60 |
| L2 + KL-sparsity | 1.55 |
| Max-norm | 1.35 |
| Dropout + L2 | 1.25 |
| Dropout + Max-norm | 1.05 |

**Table 9: Comparison of different regularization methods on MNIST.**

## 7. Salient Features

The experiments above provide strong evidence that dropout improves neural networks. Here, we examine its effects more closely. We study the quality of learned features and the sparsity of hidden-unit **activations**, or outputs. We also examine how dropout’s benefits vary with retention probability, network size, and training-set size. These observations offer insight into why dropout works so well.

### 7.1 Effect on Features

**Figure 7: Features learned on MNIST by one-hidden-layer autoencoders with 256 rectified linear units.**  
(a) Without dropout. (b) Dropout with \(p=0.5\).

In a standard network, each parameter’s derivative tells it how to change to reduce the final loss, given what all other units are doing. Units may therefore change to correct one another’s mistakes. This can create complex co-adaptations. These co-adaptations cause overfitting because they do not generalize to unseen data.

We hypothesize that dropout prevents each hidden unit from co-adapting by making other hidden units’ presence unreliable. A unit cannot rely on particular partners to correct its mistakes. It must perform well in many contexts created by different sets of other units.

To observe this effect directly, we examine first-level features learned on visual tasks with and without dropout. Figure 7a shows features learned on MNIST by an autoencoder with one hidden layer of 256 ReLUs and no dropout. Figure 7b shows an identical autoencoder trained with hidden-layer dropout at \(p=0.5\).

Both autoencoders had similar test reconstruction errors, which measure how well they reproduce their inputs. Yet the Figure 7a features clearly co-adapted to produce good reconstructions. An individual hidden unit does not seem to detect a meaningful feature.

In Figure 7b, by contrast, hidden units seem to detect edges, strokes, and spots in different image regions. This shows that dropout breaks up co-adaptations, probably the main reason it reduces generalization error.

### 7.2 Effect on Sparsity

**Figure 8: Effect of dropout on sparsity. ReLUs were used for both models.**  
(a) Without dropout. (b) Dropout with \(p=0.5\).  
Left: The histogram of mean activations shows that most units have a mean activation of about 2.0. The activation histogram has a large mode, or peak, away from zero. Clearly, a large fraction of units have high activation. Right: Most units have a lower mean activation of about 0.7. The activation histogram has a sharp peak at zero. Very few units have high activation.

Dropout makes hidden-unit activations sparse even without regularizers designed to induce sparsity. It therefore automatically produces sparse representations.

We examine this effect in the autoencoders from the previous section. We measure hidden-unit activations on a randomly chosen test mini-batch. Figures 8a and 8b compare the two models.

A good sparse model should activate only a few units strongly for each case. Each unit should also have low average activity across cases. We assess both properties with two histograms per model. A **histogram** shows how often values fall into different ranges. The left histogram shows the distribution of hidden units’ mean activations across the mini-batch. The right shows the distribution of individual hidden-unit activations.

The activation histograms show fewer highly active units in Figure 8b than in Figure 8a. The network without dropout has substantial histogram mass away from zero. Mean activations are also lower with dropout. The overall mean hidden-unit activation is close to 2.0 without dropout and around 0.7 with dropout.

### 7.3 Effect of Dropout Rate

Dropout introduces the tunable retention probability \(p\). We study its effect in two settings:

1. We hold the number of hidden units fixed.
2. We change the number of hidden units so that the expected number retained after dropout stays fixed.

In the first setting, we train the same 784-2048-2048-2048-10 architecture with different dropout rates. We do not drop input units. Figure 9a plots test error against \(p\).

With a fixed architecture, a small \(p\) leaves very few active units during training. This causes **underfitting**, where the model fails to fit even its training data well, as the high training error shows. Error falls as \(p\) increases. It stays flat over \(0.4\leq p\leq0.8\), then rises as \(p\) approaches 1.

**Figure 9: Effect of changing dropout rates on MNIST.**  
(a) Keeping \(n\) fixed. (b) Keeping \(pn\) fixed.  
Both panels show test error and training error. The horizontal axes show the probability of retaining a unit, \(p\), at 0.0, 0.2, 0.4, 0.6, 0.8, and 1.0. The vertical axes show classification error percentages. Panel (a) labels run from 0.0 to 3.5 in steps of 0.5. Panel (b) labels run from 0.0 to 3.0 in steps of 0.5.

The second setting holds \(pn\) fixed, where \(n\) is the number of hidden units in a layer. Networks with small \(p\) therefore have more hidden units. After dropout, all architectures retain the same expected number of units, although their test-time networks differ in size.

We set \(pn=256\) for the first two hidden layers and \(pn=512\) for the last hidden layer. Figure 9b plots test error against \(p\). Errors at small \(p\) fall greatly relative to Figure 9a. At \(p=0.1\), error falls from 2.7% to 1.7%. Values near 0.6 work best for this choice of \(pn\), but our usual default of 0.5 remains close to optimal.

### 7.4 Effect of Data Set Size

A good regularizer should let a model with many parameters generalize well after training on a small data set. We test this by changing the amount of MNIST training data supplied to a feed-forward network.

Huge networks trained in the standard way overfit severely on small data sets. To test whether dropout helps, we randomly select MNIST training subsets of size 100, 500, 1K, 5K, 10K, and 50K. Every experiment uses the same 784-1024-1024-2048-10 architecture. Retention probabilities are \(p=0.5\) in all hidden layers and \(p=0.8\) in the input layer.

Figure 10 gives the results. Dropout gives no improvement on the extremely small sets of 100 and 500 cases. The model still has enough parameters to overfit despite the dropout noise.

As data size grows, dropout’s benefit first increases and then declines. This suggests a “sweet spot” for any fixed architecture and dropout rate. At that size, there are too many data to memorize despite the noise, but not enough to make overfitting cease to be a problem.

**Figure 10: Effect of varying data set size.** The curves compare “With dropout” and “Without dropout.” The horizontal axis, “Dataset size,” marks \(10^2\), \(10^3\), \(10^4\), and \(10^5\). The vertical axis shows classification error percentages at 0, 5, 10, 15, 20, 25, and 30.

### 7.5 Monte-Carlo Model Averaging vs. Weight Scaling

Our efficient test-time method approximates model combination by scaling down a trained network’s weights. A more expensive but more correct method samples \(k\) dropout networks for each test case and averages their predictions.

This is **Monte-Carlo model averaging**, which estimates an average through random samples. As \(k\) tends to infinity, the estimate approaches the true model average. We ask how many samples are needed to match weight scaling. Measuring errors at different \(k\) also shows how quickly a finite-sample average approaches the true model average’s error.

We again use MNIST and classify by averaging predictions from \(k\) randomly sampled networks. Figure 11 compares the resulting test errors with the weight-scaling error, shown by a horizontal line.

At around \(k=50\), Monte-Carlo averaging matches weight scaling. Beyond that, it performs slightly better, but the difference remains well within one **standard deviation**, a measure of variation. This suggests that weight scaling approximates the true model average fairly well.

**Figure 11: Monte-Carlo model averaging vs. weight scaling.** The curves show “Monte-Carlo Model Averaging” and “Approximate averaging by weight scaling.” The horizontal axis shows the number of Monte-Carlo samples, \(k\), at 0, 20, 40, 60, 80, 100, and 120. The vertical axis shows test classification error percentages at 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, and 1.35.

## 8. Dropout Restricted Boltzmann Machines

Dropout also applies to Restricted Boltzmann Machines. This section formally defines the model and presents results that illustrate its main properties.

### 8.1 Model Description

Consider an RBM with visible units \(v\in\{0,1\}^{D}\) and hidden units \(h\in\{0,1\}^{F}\). Here, \(D\) and \(F\) are the numbers of visible and hidden units. The RBM defines the probability distribution

\[
P(h,v;\theta)
=
\frac{1}{Z(\theta)}
\exp\!\left(v^{\mathsf T}Wh+a^{\mathsf T}h+b^{\mathsf T}v\right).
\]

The parameter set is \(\theta=\{W,a,b\}\). The superscript \(\mathsf T\) means transpose, which exchanges a vector’s or matrix’s rows and columns. The **partition function** \(Z\) normalizes the probabilities so that they sum to one.

A Dropout RBM adds a vector of binary random variables,

\[
r\in\{0,1\}^{F}.
\]

Each \(r_j\) independently equals 1 with probability \(p\). When \(r_j=1\), the model retains hidden unit \(h_j\). Otherwise, it drops that unit.

The joint distribution is

\[
P(r,h,v;p,\theta)
=
P(r;p)P(h,v\mid r;\theta),
\]

\[
P(r;p)
=
\prod_{j=1}^{F}p^{r_j}(1-p)^{1-r_j},
\]

\[
P(h,v\mid r;\theta)
=
\frac{1}{Z'(\theta,r)}
\exp\!\left(v^{\mathsf T}Wh+a^{\mathsf T}h+b^{\mathsf T}v\right)
\prod_{j=1}^{F}g(h_j,r_j),
\]

\[
g(h_j,r_j)
=
\mathbf{1}(r_j=1)
+
\mathbf{1}(r_j=0)\mathbf{1}(h_j=0).
\]

Here, \(\prod\) means multiplication over the stated index. The **indicator** \(\mathbf{1}(\cdot)\) equals 1 when its condition holds and 0 otherwise. The quantity \(Z'(\theta,r)\) is a normalization constant. The function \(g\) requires \(h_j=0\) whenever \(r_j=0\).

Conditioned on \(v\) and \(r\), the distribution over \(h\) is **factorial**, meaning that its components are independent:

\[
P(h\mid r,v)
=
\prod_{j=1}^{F}P(h_j\mid r_j,v),
\]

\[
P(h_j=1\mid r_j,v)
=
\mathbf{1}(r_j=1)
\sigma\!\left(b_j+\sum_i W_{ij}v_i\right).
\]

The symbol \(\sigma\) denotes the logistic function.

**Figure 12: Features learned on MNIST by RBMs with 256 hidden units. The features are ordered by L2 norm.**  
(a) Without dropout. (b) Dropout with \(p=0.5\).

The distribution over \(v\), conditioned on \(h\), matches that of an RBM:

\[
P(v\mid h)
=
\prod_{i=1}^{D}P(v_i\mid h),
\]

\[
P(v_i=1\mid h)
=
\sigma\!\left(a_i+\sum_j W_{ij}h_j\right).
\]

Conditioned on \(r\), the distribution over \(\{v,h\}\) matches an RBM’s distribution except that hidden units with \(r_j=0\) are absent. A Dropout RBM is therefore a **mixture**, or weighted combination, of exponentially many RBMs with shared weights. Each uses a different subset of hidden units.

### 8.2 Learning Dropout RBMs

RBM learning algorithms, such as **Contrastive Divergence** (Hinton et al., 2006), apply directly to Dropout RBMs. Contrastive Divergence estimates learning updates using a short sampling process.

The only difference is that we first sample \(r\), then train using only the retained hidden units. As in dropout neural networks, we sample a different \(r\) for every training case in every mini-batch. Our experiments use **CD-1**, Contrastive Divergence with one sampling step.

### 8.3 Effect on Features

In feed-forward networks, dropout improved features by reducing co-adaptations. We now examine whether this effect carries over to Dropout RBMs.

Figure 12a shows features from a binary RBM with 256 hidden units. Figure 12b shows features from a dropout RBM with the same number of hidden units.

The dropout RBM’s features look qualitatively different. They seem coarser than the sharply defined, stroke-like features of the standard RBM. The dropout RBM also seems to have very few **dead units**, or units that do not meaningfully activate, relative to the standard RBM.

**Figure 13: Effect of dropout on sparsity.**  
(a) Without dropout. (b) Dropout with \(p=0.5\).  
Left: The activation histogram shows many units with activations away from zero. Right: Many units have activations close to zero, and very few have high activation.

### 8.4 Effect on Sparsity

We next examine how dropout RBM training affects hidden-unit sparsity. Figure 13a shows histograms of hidden-unit activations and their means on a test mini-batch after training a standard RBM. Figure 13b shows the corresponding histograms for dropout RBMs.

The histograms clearly show that dropout RBMs learn much sparser representations than standard RBMs, even without an added sparsity-inducing regularizer.

## 9. Marginalizing Dropout

We can view dropout as adding noise to hidden-unit states. Here, we briefly explore models obtained by marginalizing that noise. These are deterministic versions of dropout. Unlike standard Monte-Carlo dropout, they need no random bits, and we can obtain gradients of their marginalized loss functions.

Globerson and Roweis (2006) proposed deterministic algorithms that learn models robust to feature deletion at test time. Chen et al. (2012) studied marginalization for denoising autoencoders. Srivastava (2013) discussed marginalizing dropout noise in linear regression.

Wang and Manning (2013) further explored marginalization as a way to speed training. van der Maaten et al. (2013) examined different input-noise distributions and the regularizers produced by marginalizing them. Wager et al. (2013) described dropout as an **adaptive regularizer**, one whose penalty adapts to the data or model.

### 9.1 Linear Regression

We first consider dropout in **linear regression**, which predicts targets as a weighted sum of inputs. Let

\[
X\in\mathbb{R}^{N\times D}
\]

be a matrix containing \(N\) data points, each with \(D\) dimensions. Let \(y\in\mathbb{R}^{N}\) be the target vector. The notation \(\mathbb{R}\) denotes real-valued entries. Linear regression seeks \(w\in\mathbb{R}^{D}\) that minimizes

\[
\|y-Xw\|^2.
\]

Suppose dropout independently retains each input dimension with probability \(p\). We can write the dropped-out input as \(R*X\), where

\[
R\in\{0,1\}^{N\times D},
\qquad
R_{ij}\sim\operatorname{Bernoulli}(p),
\]

and \(*\) again means element-wise multiplication.

Marginalizing the noise gives the objective

\[
\underset{w}{\operatorname{minimize}}
\quad
\mathbb{E}_{R\sim\operatorname{Bernoulli}(p)}
\left[\|y-(R*X)w\|^2\right].
\]

The symbol \(\mathbb{E}\) denotes an expectation, or probability-weighted average. This objective reduces to

\[
\underset{w}{\operatorname{minimize}}
\quad
\|y-pXw\|^2+p(1-p)\|\Gamma w\|^2,
\]

where

\[
\Gamma=\left(\operatorname{diag}(X^{\mathsf T}X)\right)^{1/2}.
\]

Here, \(\operatorname{diag}\) keeps the diagonal entries.

Thus, dropout in linear regression is equivalent in expectation to **ridge regression**, linear regression with a squared-weight penalty, with a particular \(\Gamma\). This \(\Gamma\) effectively scales the cost on \(w_i\) by the standard deviation of the \(i\)th data dimension. If a dimension varies greatly, the regularizer pushes its weight down more strongly.

We can also absorb \(p\) into the weights:

\[
\underset{w}{\operatorname{minimize}}
\quad
\|y-X\widetilde w\|^2
+
\frac{1-p}{p}\|\Gamma\widetilde w\|^2,
\]

where \(\widetilde w=pw\).

This form makes the regularization constant’s dependence on \(p\) explicit. When \(p\) is close to 1, nearly all inputs remain, and the constant is small. Decreasing \(p\) increases dropout and enlarges the regularization constant.

### 9.2 Logistic Regression and Deep Networks

For logistic regression and deep neural networks, obtaining a **closed form** marginalized model—an explicit mathematical expression—is hard. Wang and Manning (2013) nevertheless showed how to train the marginalized model approximately for logistic regression with dropout.

Under reasonable assumptions, the distributions over inputs to the logistic unit and over gradients of the marginalized model are **Gaussian**, or normally distributed. Their means and variances can be computed efficiently. This approximate marginalization outperforms Monte-Carlo dropout in training time and generalization.

However, the assumptions behind the method become successively weaker as more layers are added. The results therefore do not apply directly to deep networks.

| Data Set | Architecture | Bernoulli dropout | Gaussian dropout |
|---|---|---|---|
| MNIST | 2 layers, 1024 units each | 1.08 ± 0.04 | 0.95 ± 0.04 |
| CIFAR-10 | 3 conv + 2 fully connected layers | 12.6 ± 0.1 | 12.5 ± 0.1 |

**Table 10: Comparison of classification error % with Bernoulli and Gaussian dropout.** For MNIST, the Bernoulli model uses \(p=0.5\) for hidden units and \(p=0.8\) for input units. For CIFAR-10, we use \(p=(0.9,0.75,0.75,0.5,0.5,0.5)\), ordered from the input layer to the top. For Gaussian dropout, we set \(\sigma=\sqrt{(1-p)/p}\). Results average over 10 different random seeds, which initialize the random-number process.

## 10. Multiplicative Gaussian Noise

Dropout multiplies hidden activations by Bernoulli random variables that equal 1 with probability \(p\) and 0 otherwise. We can generalize this idea by multiplying activations by random variables from other distributions.

We recently found that multiplying by a variable drawn from \(\mathcal{N}(1,1)\) works as well as, or perhaps better than, Bernoulli noise. The notation \(\mathcal{N}(\mu,\sigma^2)\) denotes a Gaussian distribution with mean \(\mu\) and variance \(\sigma^2\).

This form of dropout adds a zero-mean Gaussian random variable whose standard deviation equals the unit’s activation. Specifically, it perturbs hidden activation \(h_i\) to

\[
h_i+h_i r,
\qquad r\sim\mathcal{N}(0,1),
\]

or, equivalently, to

\[
h_i r',
\qquad r'\sim\mathcal{N}(1,1).
\]

We can generalize this to

\[
r'\sim\mathcal{N}(1,\sigma^2).
\]

The value \(\sigma\) then becomes a hyperparameter, like \(p\) in Bernoulli dropout. Expected activations stay unchanged, so test time requires no weight scaling.

We have described dropout as retaining units with probability \(p\) during training and multiplying their weights by \(p\) at test time. An alternative multiplies retained activations by \(1/p\) during training and leaves test-time weights unchanged. The two methods are equivalent with suitable scaling of each layer’s learning rate and initial weights.

Under this alternative, dropout multiplies \(h_i\) by a Bernoulli random variable \(r_b\) that equals \(1/p\) with probability \(p\) and 0 otherwise. Thus,

\[
\mathbb{E}[r_b]=1,
\qquad
\operatorname{Var}[r_b]=\frac{1-p}{p}.
\]

The notation \(\operatorname{Var}\) denotes variance, a measure of spread. For Gaussian multiplicative noise, setting

\[
\sigma^2=\frac{1-p}{p}
\]

gives a multiplier \(r_g\) with

\[
\mathbb{E}[r_g]=1,
\qquad
\operatorname{Var}[r_g]=\frac{1-p}{p}.
\]

Both forms can therefore use multipliers with the same mean and variance. Given these first- and second-order **moments**, or distribution summaries, \(r_g\) has the highest **entropy**, a measure of uncertainty, and \(r_b\) has the lowest.

Both extremes work well. The preliminary results in Table 10 suggest that the high-entropy case may work slightly better. For each Gaussian-model layer, we set

\[
\sigma=\sqrt{\frac{1-p}{p}},
\]

using \(p\) from the corresponding Bernoulli-model layer.

## 11. Conclusion

Dropout reduces overfitting to improve neural networks. Standard backpropagation builds fragile co-adaptations that work on training data but do not generalize. Random dropout breaks them up by making any particular hidden unit’s presence unreliable.

Dropout improved neural-network performance across many domains: object classification, digit recognition, speech recognition, document classification, and computational biology data analysis. This suggests that dropout is a general technique rather than a domain-specific one. Methods using dropout achieve state-of-the-art results on SVHN, ImageNet, CIFAR-100, and MNIST. Dropout also greatly improved standard neural networks on other data sets.

The idea extends to RBMs and other graphical models. Its central principle is to start with a large model that overfits easily, then repeatedly sample and train smaller sub-models. RBMs fit this framework readily. We developed Dropout RBMs and showed experimentally that they have certain desirable properties.

A drawback is increased training time. A dropout network typically takes 2–3 times as long to train as a standard network with the same architecture. Highly noisy parameter updates cause much of this increase. Each case effectively trains a different random architecture. The computed gradients therefore do not belong to the final test-time architecture. Long training times are consequently not surprising.

This randomness likely also prevents overfitting, creating a trade-off between overfitting and training time. More training time allows higher dropout and less overfitting.

One way to gain some benefits without randomness is to marginalize the noise. This produces a regularizer that matches the dropout procedure in expectation. For linear regression, we showed that this regularizer is a modified L2 penalty. For more complex models, an equivalent regularizer is not obvious. Speeding up dropout remains an interesting direction for future work.

## Acknowledgments

This research was supported by OGS, NSERC and an Early Researcher Award.

## Appendix A. A Practical Guide for Training Dropout Networks

Neural networks are known for requiring extensive hyperparameter tuning. Dropout networks are no exception. This section offers **heuristics**, or practical rules of thumb, that may help when applying dropout.

### A.1 Network Size

Dropping units should reduce a network’s **capacity**, its ability to fit different patterns. If a layer has \(n\) hidden units and retains each with probability \(p\), it contains \(pn\) units after dropout in expectation. The retained set changes each time, and units cannot freely build co-adaptations.

Thus, if a layer of \(n\) units is optimal for a standard network on a task, a good dropout network should have at least \(n/p\) units. We found this rule useful for choosing hidden-unit counts in both convolutional and fully connected networks.

### A.2 Learning Rate and Momentum

Dropout adds substantial gradient noise compared with standard stochastic gradient descent. Many gradients therefore tend to cancel. To compensate, a dropout network should typically use a learning rate 10–100 times the rate that was optimal for a standard network.

High momentum offers another way to reduce the noise’s effect. Standard networks commonly use momentum of 0.9. With dropout, values around 0.95 to 0.99 worked much better. A high learning rate, high momentum, or both greatly speed learning.

### A.3 Max-norm Regularization

Large momentum and learning rates speed learning but sometimes make weights grow very large. Max-norm regularization prevents this by limiting each hidden unit’s incoming weight-vector norm to a constant \(c\). Typical values of \(c\) range from 3 to 4.

### A.4 Dropout Rate

The extra hyperparameter \(p\), the probability of retaining a unit, controls dropout intensity. A value of \(p=1\) means no dropout. Lower values mean more dropout.

Typical hidden-unit values range from 0.5 to 0.8. For input layers, the choice depends on the input type. A typical value for real-valued inputs, such as image patches or speech frames, is 0.8.

For hidden layers, \(p\) and the number of units \(n\) must be chosen together. Smaller \(p\) requires larger \(n\), which slows training and leads to underfitting. Large \(p\) may provide too little dropout to prevent overfitting.

## Appendix B. Detailed Description of Experiments and Data Sets

.

This section details the architectures and training used for the reported experiments. Code to reproduce the results is available at <http://www.cs.toronto.edu/~nitish/dropout>.

The implementation uses a **graphics processing unit (GPU)**, a processor suited to parallel computation. We implemented the networks with the excellent CUDA libraries cudamat (Mnih, 2009) and cuda-convnet (Krizhevsky et al., 2012). CUDA is a platform for GPU computation.

### B.1 MNIST

MNIST contains 60,000 training and 10,000 test examples, each a 28 × 28 digit image. We randomly held out 10,000 training images for validation.

We tuned hyperparameters so that the best validation error occurred after 1 million weight updates. We then combined the validation and training sets and trained for 1 million updates. We evaluated this network on the test set.

We used validation this way because we could easily choose hyperparameters that removed the need for early stopping. Once those settings were fixed, combining the sets and training for a very long time made sense.

Figure 4 includes all combinations of 2, 3, and 4 layers with 1024 or 2048 units per layer, giving six architectures. For all architectures, including those in Table 2, we used \(p=0.5\) in every hidden layer and \(p=0.8\) in the input layer. Every layer used final momentum of 0.95 and a weight constraint of \(c=2\).

To test the limits of dropout regularization, we also tried 2- and 3-layer networks with 4096 and 8192 units. The 2-layer networks improved performance, as Table 2 shows. At the same dropout level, however, the 3-layer networks performed slightly worse than their 2-layer counterparts. More dropout improved them, but not enough to beat the 2-layer networks.

### B.2 SVHN

SVHN contains approximately 600,000 training images and 26,000 test images. Its training data have two parts: a standard labeled training set and an extra set of easy labeled examples.

We formed a validation set from both parts. Two-thirds came from the standard set, with 400 examples per class. One-third came from the extra set, with 200 per class. This gave 6000 samples, following Sermanet et al. (2012).

Inputs were **RGB pixels**, represented by red, green, and blue values, normalized to zero mean and unit variance. Other preprocessing methods gave no noticeable improvements. These included global or local contrast normalization, which adjusts image intensity levels, and ZCA whitening, defined in the next section.

Our best architecture has three convolutional layers, each followed by max pooling. The layers have 96, 128, and 256 filters, respectively. Each filter has a 5 × 5 **receptive field**, the input region it processes, and moves with a **stride**, or step, of 1 pixel. Each max-pooling layer pools 3 × 3 regions with a stride of 2 pixels.

Two fully connected hidden layers follow the convolutional layers. Each has 2048 units. All units use rectified linear activation.

We applied dropout to every layer, with retention probabilities

\[
p=(0.9,0.75,0.75,0.5,0.5,0.5),
\]

ordered from the input through convolutional layers to fully connected layers. We also used max-norm constraints with \(c=4\) for all weights and momentum of 0.95 in every layer.

We tuned these hyperparameters with a validation set. Because the training set was quite large, we did not combine the validation set with it for final training. We reported the test error of the model with the lowest validation error.

### B.3 CIFAR-10 and CIFAR-100

CIFAR-10 and CIFAR-100 each contain 50,000 training and 10,000 test images. They have 10 and 100 categories, respectively. All images are 32 × 32 color images.

We used 5,000 training images for validation. As with MNIST, we selected hyperparameters using validation, then combined the validation and training sets.

We preprocessed images with global contrast normalization in each color channel, followed by **ZCA whitening**. For each image and color channel, global contrast normalization computes the mean pixel intensity and subtracts it from the channel. ZCA whitening mean-centers the data, rotates them onto their principal components, normalizes each component, and rotates them back. Principal components are the directions used by PCA.

The architecture and dropout rates match those for SVHN, except that input-layer learning rates had to be smaller.

### B.4 TIMIT

We used the open-source Kaldi toolkit (Povey et al., 2011) to convert speech data into log-filter banks. We trained a **monophone system**, which models individual speech sounds, to perform **forced alignment**, matching a known speech sequence to recording times. This supplied labels for speech frames.

We trained dropout networks on windows of 21 consecutive frames to predict the central frame’s label. We performed no speaker-dependent operations. We mean-centered inputs and normalized them to unit variance.

Retention probabilities were \(p=0.8\) for input layers and 0.5 for hidden layers. All layers used max-norm constraints with \(c=4\). We used momentum of 0.95 and a high learning rate of 0.1. The learning rate decayed as

\[
\epsilon_0(1+t/T)^{-1}.
\]

Here, \(\epsilon_0\) is the initial rate, \(t\) tracks training progress, and \(T\) sets the decay timescale.

For DBN pretraining, we trained RBMs with CD-1. We fixed each input unit’s variance in the **Gaussian RBM**, an RBM with Gaussian-valued input units, to 1. Finetuning the DBN with dropout worked best with a smaller learning rate, about 0.01. Max-norm constraints gave no improvements.

### B.5 Reuters

The Reuters RCV1 corpus contains more than 800,000 documents in 103 classes arranged as a tree hierarchy. We created a subset of 402,738 articles with a vocabulary of 2000 words and 50 categories. Each document belongs to exactly one class.

We split the subset into equal-sized training and test sets. We tried many network architectures. Dropout improved classification accuracy in all of them, but the gains were smaller than on image and speech data.

The large data set—more than 200,000 training examples—may explain the smaller improvement, since overfitting is not a very serious problem.

### B.6 Alternative Splicing

The alternative splicing data set contains 3665 **cassette exons**, gene segments that can be included or excluded during splicing; 1014 RNA features; and 4 tissue types derived from 27 mouse tissues.

Each input’s target contains 4 **softmax units**, one per tissue type. A softmax unit produces a probability distribution over possible states. Each unit has 3 biologically important states: inc, exc, and nc, representing inclusion, exclusion, and no change.

For each softmax unit, we seek a predicted distribution over these 3 states that matches the distribution observed in wet-lab experiments as closely as possible. We use Code Quality, defined as

\[
\sum_{i=1}^{|\text{data points}|}
\sum_{t\in\text{tissue types}}
\sum_{s\in\{\mathrm{inc},\mathrm{exc},\mathrm{nc}\}}
p_{i,t}^{s}
\log\!\left(\frac{q_t^{s}(r_i)}{\bar p^{s}}\right).
\]

Here:

- \(p_{i,t}^{s}\) is the target probability of state \(s\) for tissue type \(t\) in input \(i\).
- \(q_t^{s}(r_i)\) is the predicted probability of state \(s\) for tissue type \(t\) and input \(r_i\).
- \(\bar p^{s}\) averages \(p_{i,t}^{s}\) over \(i\) and \(t\).

We trained a two-layer dropout network with 1024 units per layer. Retention probabilities were \(p=0.5\) for the hidden layer and \(p=0.7\) for the input layer. We used max-norm regularization and high, decaying learning rates.

We averaged results across the same 5 **folds**, or data partitions used for evaluation, as Xiong et al. (2011).
