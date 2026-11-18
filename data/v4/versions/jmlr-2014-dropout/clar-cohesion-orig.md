Journal of Machine Learning Research 15 (2014) 1929–1958  
Submitted 11/13; Published 6/14

# Dropout: A Simple Way to Prevent Neural Networks from Overfitting

Nitish Srivastava                                                                nitish@cs.toronto.edu  
Geoffrey Hinton                                                                 hinton@cs.toronto.edu  
Alex Krizhevsky                                                                    kriz@cs.toronto.edu  
Ilya Sutskever                                                                     ilya@cs.toronto.edu  
Ruslan Salakhutdinov                                                          rsalakhu@cs.toronto.edu  
Department of Computer Science  
University of Toronto  
10 Kings College Road, Rm 3302  
Toronto, Ontario, M5S 3G4, Canada.

Editor: Yoshua Bengio

## Abstract

Deep neural nets with a large number of parameters are very powerful machine learning systems. However, overfitting is a serious problem in such networks. Large networks are also slow to use. Therefore, addressing overfitting by combining the predictions of many different large neural nets at test time is difficult.

Dropout addresses this problem by randomly dropping units, along with their connections, from the neural network during training. Removing units in this way prevents units from co-adapting too much. During training, dropout samples from an exponential number of different “thinned” networks: networks containing only the units that survive dropout. At test time, a single unthinned network with smaller weights provides an easy approximation to averaging the predictions of all these thinned networks.

This procedure significantly reduces overfitting and gives major improvements over other regularization methods. We show that dropout improves neural-network performance on supervised learning tasks in vision, speech recognition, document classification and computational biology, obtaining state-of-the-art results on many benchmark data sets.

**Keywords:** neural networks, regularization, model combination, deep learning

## 1. Introduction

Deep neural networks contain multiple non-linear hidden layers. These layers make deep neural networks very expressive models that can learn very complicated relationships between inputs and outputs. With limited training data, however, many complicated relationships will result from sampling noise. Such relationships exist in the training set but not in real test data, even when the test data are drawn from the same distribution. Learning these relationships therefore leads to overfitting.

Many methods have been developed to reduce overfitting. These methods include stopping training as soon as performance on a validation set starts to worsen, introducing weight penalties such as L1 and L2 regularization, and using soft weight sharing (Nowlan and Hinton, 1992). L1 and L2 denote penalties based on the corresponding norms of the weights.

With unlimited computation, the best way to “regularize” a fixed-sized model is to average the predictions of all possible parameter settings, weighting each setting by its posterior probability given the training data. This averaging can sometimes be approximated quite well for simple or small models (Xiong et al., 2011; Salakhutdinov and Mnih, 2008). However, we would like to approach the performance of this Bayesian gold standard with considerably less computation. We propose to do so by approximating an equally weighted geometric mean of the predictions of an exponential number of learned models that share parameters.

**Figure 1: Dropout Neural Net Model.** Left, **(a) Standard Neural Net**: A standard neural net with 2 hidden layers. Right, **(b) After applying dropout**: An example of a thinned net produced by applying dropout to the network on the left. Crossed units have been dropped.

Model combination nearly always improves the performance of machine learning methods. With large neural networks, however, the obvious approach—averaging the outputs of many separately trained nets—is prohibitively expensive. Combining models is most helpful when the individual models differ from one another. To make neural net models different, the models should therefore either have different architectures or be trained on different data.

Both choices create difficulties. Training many different architectures is hard because finding optimal hyperparameters for each architecture is a daunting task, and training each large network requires considerable computation. Moreover, large networks normally require large amounts of training data, so there may not be enough data to train different networks on different subsets. Even if many different large networks could be trained, using all those networks at test time would be infeasible in applications that require a quick response.

Dropout addresses both overfitting and the expense of model combination. Dropout prevents overfitting and provides an efficient way to approximately combine exponentially many different neural network architectures.

The term “dropout” refers to dropping out hidden and visible units in a neural network. Dropping a unit means temporarily removing the unit from the network, along with all incoming and outgoing connections, as illustrated in Figure 1. The choice of units to drop is random. In the simplest case, each unit is retained with a fixed probability \(p\), independently of other units. The retention probability \(p\) can be chosen using a validation set, or \(p\) can simply be set to 0.5, which seems close to optimal for a wide range of networks and tasks. For input units, however, the optimal retention probability is usually closer to 1 than to 0.5.

**Figure 2:** Left, **(a) At training time**: A unit is present with probability \(p\) and is connected to units in the next layer with weights \(w\). Right, **(b) At test time**: The unit is always present, and the weights are multiplied by \(p\), giving weights \(pw\). The output at test time is the same as the expected output at training time.

Applying dropout to a neural network amounts to sampling a “thinned” network from the original network. The thinned network consists of all units that survive dropout, as shown in Figure 1b. A neural net with \(n\) units can therefore be viewed as a collection of \(2^n\) possible thinned neural networks. All these networks share weights, so the total number of parameters remains \(O(n^2)\), or less; \(O(n^2)\) denotes a parameter count that grows at most on the order of \(n^2\).

For each presentation of each training case, a new thinned network is sampled and trained. Thus, training a neural network with dropout can be viewed as training a collection of \(2^n\) thinned networks with extensive weight sharing. Each individual thinned network is trained very rarely, if at all.

Explicitly averaging the predictions of exponentially many thinned models is not feasible at test time. However, a simple approximate averaging method works well in practice: use a single neural net without dropout and scale down the trained weights. Specifically, if a unit is retained with probability \(p\) during training, multiply that unit’s outgoing weights by \(p\) at test time, as shown in Figure 2. This scaling ensures that, for any hidden unit, the expected output under the distribution used to drop units during training is the same as the actual output at test time. In this way, \(2^n\) networks with shared weights can be combined into a single neural network for test-time use.

We found that training with dropout and using this approximate averaging method at test time leads to significantly lower generalization error on a wide variety of classification problems than training with other regularization methods.

Dropout is not limited to feed-forward neural nets. Dropout can also be applied more generally to graphical models such as Boltzmann Machines. In this paper, we introduce the dropout Restricted Boltzmann Machine (RBM) model and compare dropout RBMs with standard RBMs. Our experiments show that dropout RBMs are better than standard RBMs in certain respects.

The paper is structured as follows. Section 2 describes the motivation for dropout. Section 3 describes relevant previous work. Section 4 formally describes the dropout model, and Section 5 gives a training algorithm. Section 6 presents experiments applying dropout to problems in different domains and comparing dropout with other forms of regularization and model combination. Section 7 analyzes dropout’s effects on neural-network properties and its interactions with hyperparameters. Section 8 describes the Dropout RBM model. Section 9 explores marginalizing dropout. Appendix A presents a practical guide to training dropout nets, including a detailed analysis of practical considerations in choosing hyperparameters.

© 2014 Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever and Ruslan Salakhutdinov.

## 2. Motivation

One motivation for dropout comes from a theory of the role of sex in evolution (Livnat et al., 2010). Sexual reproduction takes half the genes of one parent and half the genes of another, adds a very small amount of random mutation, and combines the genes to produce an offspring. The asexual alternative produces an offspring with a slightly mutated copy of the parent’s genes.

Asexual reproduction might seem a better way to optimize individual fitness because a good set of genes that work well together can be passed directly to the offspring. In contrast, sexual reproduction is likely to break up co-adapted sets of genes, especially large sets. Intuitively, breaking up these sets should decrease the fitness of organisms that have already evolved complicated co-adaptations. Nevertheless, sexual reproduction is the way most advanced organisms evolved.

One possible explanation for the superiority of sexual reproduction is that, over the long term, the criterion for natural selection may be the mix-ability of genes rather than individual fitness. A set of genes becomes more robust when the genes can work well with another random set of genes. Because a gene cannot rely on a large set of partners always being present, the gene must learn to do something useful either on its own or in collaboration with a small number of other genes.

According to this theory, sexual reproduction does more than allow useful new genes to spread throughout the population. Sexual reproduction also facilitates the spread of useful new genes by reducing complex co-adaptations that would otherwise reduce the chance of a new gene improving an individual’s fitness.

Similarly, each hidden unit in a neural network trained with dropout must learn to work with a randomly chosen sample of other units. This requirement should make each hidden unit more robust and drive the hidden unit towards creating useful features on its own, without relying on other hidden units to correct its mistakes. However, hidden units within a layer will still learn to do different things from one another. One might imagine that a network could become robust against dropout by making many copies of each hidden unit. However, making copies is a poor solution for exactly the same reason that replica codes are a poor way to deal with a noisy channel.

A closely related, but slightly different, motivation comes from considering successful conspiracies. Ten conspiracies involving five people each are probably a better way to create havoc than one large conspiracy requiring fifty people to play their parts correctly. A large conspiracy can work well if conditions do not change and there is plenty of rehearsal time. With non-stationary conditions, however, a smaller conspiracy has a greater chance of continuing to work.

The analogous concern in neural networks is that complex co-adaptations can be trained to work well on a training set. On novel test data, however, those complex co-adaptations are far more likely to fail than multiple simpler co-adaptations that achieve the same thing.

## 3. Related Work

Dropout can be interpreted as regularizing a neural network by adding noise to hidden-unit states. Adding noise to unit states has previously been used in Denoising Autoencoders (DAEs) by Vincent et al. (2008, 2010). In a DAE, noise is added to an autoencoder’s input units, and the network is trained to reconstruct the noise-free input.

Our work extends this idea in two ways: dropout can be applied effectively in hidden layers, and dropout can be interpreted as model averaging. We also show that adding noise is useful not only for unsupervised feature learning but also for supervised learning problems. Moreover, dropout can be applied to other neuron-based architectures, including Boltzmann Machines.

While 5% noise typically works best for DAEs, our test-time weight-scaling procedure enables much higher noise levels. Dropping out 20% of input units and 50% of hidden units was often optimal.

Because dropout is a stochastic regularization technique, a natural question is whether a deterministic counterpart can be obtained by marginalizing out the noise—that is, averaging over the noise distribution. In this paper, we show that dropout can be analytically marginalized in simple cases to obtain deterministic regularization methods.

Recently, van der Maaten et al. (2013) also explored deterministic regularizers corresponding to different exponential-family noise distributions, including dropout, which they call “blankout noise.” However, van der Maaten et al. apply noise to inputs and explore only models without hidden layers. Wang and Manning (2013) proposed speeding up dropout by marginalizing dropout noise. Chen et al. (2012) explored marginalization for denoising autoencoders.

In dropout, the loss function is minimized stochastically under a noise distribution. This procedure can be viewed as minimizing an expected loss function. In contrast, Globerson and Roweis (2006) and Dekel et al. (2010) explored a setting in which the loss is minimized when an adversary chooses which units to drop. Their setting fixes the maximum number of units that can be dropped rather than specifying a noise distribution. However, this work also does not explore models with hidden units.

## 4. Model Description

Consider a neural network with \(L\) hidden layers. The index \(l\in\{1,\ldots,L\}\) identifies a hidden layer. Let \(z^{(l)}\) be the vector of inputs into layer \(l\), and let \(y^{(l)}\) be the vector of outputs from layer \(l\). The input layer is \(y^{(0)}=x\), where \(x\) is the network input. Let \(W^{(l)}\) and \(b^{(l)}\) be the weights and biases at layer \(l\), and let \(w_i^{(l)}\) be the weight vector for unit \(i\) at that layer.

The activation function is denoted by \(f\); one example is the logistic function,
\[
f(x)=\frac{1}{1+\exp(-x)}.
\]

With this notation, the feed-forward operation of a standard neural network, shown in Figure 3a, is, for \(l\in\{0,\ldots,L-1\}\) and any hidden unit \(i\),
\[
\begin{aligned}
z_i^{(l+1)} &= w_i^{(l+1)}y^{(l)}+b_i^{(l+1)},\\
y_i^{(l+1)} &= f\!\left(z_i^{(l+1)}\right).
\end{aligned}
\]

For dropout, define \(r^{(l)}\) as a vector of independent Bernoulli random variables. A Bernoulli random variable with parameter \(p\) is 1 with probability \(p\) and 0 otherwise. Let \(*\) denote an element-wise product, and let \(\widetilde y^{(l)}\) denote the thinned outputs of layer \(l\). The dropout feed-forward operation, shown in Figure 3b, is
\[
\begin{aligned}
r_j^{(l)} &\sim \operatorname{Bernoulli}(p),\\
\widetilde y^{(l)} &= r^{(l)}*y^{(l)},\\
z_i^{(l+1)} &= w_i^{(l+1)}\widetilde y^{(l)}+b_i^{(l+1)},\\
y_i^{(l+1)} &= f\!\left(z_i^{(l+1)}\right).
\end{aligned}
\]

Thus, at each layer, the sampled vector \(r^{(l)}\) is multiplied element-wise with that layer’s outputs \(y^{(l)}\). The resulting thinned outputs \(\widetilde y^{(l)}\) become the inputs to the next layer. Applying this process at each layer amounts to sampling a sub-network from the larger network.

**Figure 3: Comparison of the basic operations of a standard and dropout network.** Panel **(a), Standard network**, connects the outputs \(y_1^{(l)},y_2^{(l)},y_3^{(l)}\), together with a \(+1\) bias input, to unit \(i\) in layer \(l+1\). Panel **(b), Dropout network**, first multiplies the layer outputs by \(r_1^{(l)},r_2^{(l)},r_3^{(l)}\) to obtain \(\widetilde y_1^{(l)},\widetilde y_2^{(l)},\widetilde y_3^{(l)}\). Both panels then form \(z_i^{(l+1)}\) using weights \(w_i^{(l+1)}\) and bias \(b_i^{(l+1)}\), and apply \(f\) to obtain \(y_i^{(l+1)}\).

During learning, derivatives of the loss function are backpropagated through the sampled sub-network. At test time, the weights are scaled as
\[
W_{\mathrm{test}}^{(l)}=pW^{(l)},
\]
as shown in Figure 2. The resulting neural network is then used without dropout.

## 5. Learning Dropout Nets

This section describes a procedure for training dropout neural nets.

### 5.1 Backpropagation

Dropout neural networks can be trained using stochastic gradient descent in much the same way as standard neural nets. The difference is that, for each training case in a mini-batch, units are dropped to sample a thinned network. Forward propagation and backpropagation for that training case are performed only on the sampled thinned network.

The gradient for each parameter is averaged over the training cases in the mini-batch. A training case that does not use a parameter contributes a gradient of zero for that parameter. Methods that improve stochastic gradient descent, including momentum, annealed learning rates and L2 weight decay, were also useful for dropout neural networks.

One form of regularization was especially useful with dropout: upper-bounding the norm of the incoming weight vector at each hidden unit by a fixed constant \(c\). If \(w\) is the vector of weights incident on a hidden unit and \(\|w\|_2\) is its Euclidean norm, the optimization constraint is
\[
\|w\|_2\leq c.
\]
During optimization, whenever \(w\) leaves the ball of radius \(c\), \(w\) is projected onto the ball’s surface. This procedure is called **max-norm regularization**, because the maximum norm allowed is \(c\).

The constant \(c\) is a tunable hyperparameter determined using a validation set. Max-norm regularization has previously been used in collaborative filtering (Srebro and Shraibman, 2005). Max-norm regularization typically improves stochastic-gradient-descent training of deep neural nets even without dropout.

Although dropout alone gives significant improvements, combining dropout with max-norm regularization, large decaying learning rates and high momentum gives a significant further boost. A possible justification is that constraining weight vectors to a ball of fixed radius allows a huge learning rate without the weights blowing up. Dropout’s noise then allows optimization to explore regions of weight space that would otherwise be difficult to reach. As the learning rate decays, optimization takes shorter steps, explores less, and eventually settles into a minimum.

### 5.2 Unsupervised Pretraining

Neural networks can be pretrained using stacks of Restricted Boltzmann Machines (RBMs) (Hinton and Salakhutdinov, 2006), autoencoders (Vincent et al., 2010), or Deep Boltzmann Machines (Salakhutdinov and Hinton, 2009). Pretraining is an effective way to use unlabeled data. In certain cases, pretraining followed by finetuning with backpropagation gives significant performance improvements over finetuning from random initializations.

Dropout can be used when finetuning networks pretrained with these techniques. The pretraining procedure remains unchanged. Before dropout finetuning, the pretrained weights should be scaled up by \(1/p\), where \(p\) is the probability of retaining a unit. This scaling ensures that each unit’s expected output under random dropout is the same as the unit’s output during pretraining.

We initially worried that dropout’s stochastic nature might wipe out the information in the pretrained weights. This did happen when finetuning learning rates were comparable to the best learning rates for randomly initialized nets. However, smaller finetuning learning rates seemed to retain the information in the pretrained weights. With those smaller learning rates, dropout finetuning improved final generalization error relative to finetuning without dropout.

## 6. Experimental Results

We trained dropout neural networks for classification problems on data sets from different domains. On every data set, dropout improved generalization performance compared with neural networks that did not use dropout. Table 1 describes the data sets:

- **MNIST:** A standard toy data set of handwritten digits.
- **TIMIT:** A standard speech benchmark for clean speech recognition.
- **CIFAR-10 and CIFAR-100:** Tiny natural images (Krizhevsky, 2009).
- **Street View House Numbers (SVHN):** Images of house numbers collected by Google Street View (Netzer et al., 2011).
- **ImageNet:** A large collection of natural images.
- **Reuters-RCV1:** A collection of Reuters newswire articles.
- **Alternative Splicing:** RNA features for predicting alternative gene splicing (Xiong et al., 2011), where RNA denotes ribonucleic acid.

We chose this diverse set to demonstrate that dropout is a general technique for improving neural nets, rather than a technique specific to one application domain. This section presents key results showing dropout’s effectiveness. Appendix B gives more detailed descriptions of the experiments and data sets.

In Table 1, K denotes thousands and M denotes millions; “dim” denotes dimensions.

| Data Set | Domain | Dimensionality | Training Set | Test Set |
|---|---|---|---|---|
| MNIST | Vision | 784 (28 × 28 grayscale) | 60K | 10K |
| SVHN | Vision | 3072 (32 × 32 color) | 600K | 26K |
| CIFAR-10/100 | Vision | 3072 (32 × 32 color) | 60K | 10K |
| ImageNet (ILSVRC-2012) | Vision | 65536 (256 × 256 color) | 1.2M | 150K |
| TIMIT | Speech | 2520 (120-dim, 21 frames) | 1.1M frames | 58K frames |
| Reuters-RCV1 | Text | 2000 | 200K | 200K |
| Alternative Splicing | Genetics | 1014 | 2932 | 733 |

**Table 1: Overview of the data sets used in this paper.** ILSVRC denotes the ImageNet Large-Scale Visual Recognition Challenge, discussed in Section 6.1.4.

### 6.1 Results on Image Data Sets

We evaluated dropout on five image data sets: MNIST, SVHN, CIFAR-10, CIFAR-100 and ImageNet. These data sets differ in image types and training-set sizes. Models achieving state-of-the-art results on all these data sets use dropout.

#### 6.1.1 MNIST

The MNIST data set consists of 28 × 28 pixel handwritten digit images. The task is to classify each image into one of 10 digit classes. In the permutation invariant setting discussed here, the comparison does not use 2-D spatial information.

Table 2 compares dropout with other techniques. In the table, NN means neural network, SVM means support vector machine, DBN means Deep Belief Network, and DBM means Deep Boltzmann Machine. Logistic units use the logistic activation function given in Section 4. ReLU means rectified linear unit (Jarrett et al., 2009). Maxout identifies the unit type used by Goodfellow et al. (2013). NA means not applicable.

All dropout nets in this comparison use retention probability \(p=0.5\) for hidden units and \(p=0.8\) for input units. Appendix B.1 gives further experimental details.

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

For the permutation invariant setting, the best-performing neural networks without dropout or unsupervised pretraining achieve an error of about 1.60% (Simard et al., 2003). With dropout, the error falls to 1.35%, as shown by the logistic-unit dropout row in Table 2. Replacing logistic units with rectified linear units further reduces error to 1.25%, and adding max-norm regularization reduces error again to 1.06%.

Increasing network size gives further improvements. Table 2 reports 1.04% error for 3 layers of 2048 units, 1.01% for 2 layers of 4096 units, and 0.95% for 2 layers of 8192 units. The last network has more than 65 million parameters and is trained on a data set of size 60,000. Obtaining good generalization from a network of this size is very hard with standard regularization methods and early stopping. In contrast, dropout prevents overfitting even in this case, without requiring early stopping. Goodfellow et al. (2013) further improved the result to 0.94% by replacing ReLU units with maxout units.

Dropout also improves networks pretrained with stacks of RBMs and Deep Boltzmann Machines. In Table 2, dropout finetuning lowers the Deep Belief Network’s error from 1.18% to 0.92% and the Deep Boltzmann Machine’s error from 0.96% to 0.79%. The DBM-pretrained dropout network’s 0.79% test error is the best performance ever reported for the permutation invariant setting.

Better results are possible by using 2-D spatial information and augmenting the training set with distorted versions of images from the standard training set. We demonstrate dropout’s effectiveness in that setting on more interesting data sets.

To test dropout’s robustness, we also ran classification experiments with many different architectures while holding all hyperparameters, including \(p\), fixed. Figure 4 shows test error as training progresses. The same architectures trained with and without dropout have drastically different test errors, forming two separate clusters of trajectories. Thus, dropout gives a huge improvement across architectures without architecture-specific hyperparameter tuning.

**Figure 4: Test error for different architectures with and without dropout. The networks have 2 to 4 hidden layers each with 1024 to 2048 units.** The vertical axis is Classification Error %, with ticks at 1.0, 1.5, 2.0 and 2.5. The horizontal axis is Number of weight updates, with ticks at 0, 200000, 400000, 600000, 800000 and 1000000. The two trajectory groups are labeled “Without dropout” and “With dropout.”

#### 6.1.2 Street View House Numbers

The Street View House Numbers (SVHN) data set (Netzer et al., 2011) consists of color images of house numbers collected by Google Street View. Figure 5a shows examples. The part used in our experiments consists of 32 × 32 color images roughly centered on one digit in a house number. The task is to identify that digit.

We applied dropout to convolutional neural networks (LeCun et al., 1989). The best architecture we found has three convolutional layers followed by 2 fully connected hidden layers. Every convolutional layer is followed by a max-pooling layer, and all hidden units are rectified linear units. Appendix B.2 describes the architecture in more detail.

Dropout was applied to every layer, with retention probabilities
\[
p=(0.9,0.75,0.75,0.5,0.5,0.5)
\]
in order from the input layer through the convolutional layers to the fully connected layers. Max-norm regularization was used for weights in both convolutional and fully connected layers.

Table 3 compares methods. “Conv Net” means convolutional neural network. WDCH is the label used for the binary-feature method of Netzer et al. (2011); HOG denotes histograms of oriented gradients. The pooling labels L2 and L4 distinguish the corresponding pooling methods reported by Sermanet et al. (2012).

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

Convolutional nets outperform the other methods in Table 3. The best-performing convolutional nets that do not use dropout achieve an error rate of 3.95%. Adding dropout only to the fully connected layers reduces error to 3.02%. Adding dropout to convolutional layers as well reduces error further to 2.55%. Using maxout units gives further gains, with the maxout row reporting 2.47%.

The additional improvement from applying dropout to convolutional layers—from 3.02% to 2.55%—is worth noting. Because convolutional layers do not have many parameters, one might presume that overfitting is not a problem in those layers and that dropout would therefore have little effect. However, dropout in lower layers still helps because the lower layers provide noisy inputs to the higher fully connected layers, preventing those higher layers from overfitting.

#### 6.1.3 CIFAR-10 and CIFAR-100

The CIFAR-10 and CIFAR-100 data sets consist of 32 × 32 color images drawn from 10 and 100 categories, respectively. Figure 5b shows examples. Appendix B.3 describes the data sets, input preprocessing, network architectures and other experimental details.

Table 4 compares error rates. Without data augmentation, Snoek et al. (2012) used Bayesian hyperparameter optimization to obtain 14.98% error on CIFAR-10. Applying dropout to the fully connected layers reduces error to 14.32%, and applying dropout to every layer reduces error further to 12.61%. Goodfellow et al. (2013) showed that replacing rectified linear units with maxout units further reduces error to 11.68%.

On CIFAR-100, dropout reduces error from 43.48% for the hand-tuned max-pooling convolutional net to 37.20% for dropout in all layers, a huge improvement. No data augmentation was used for either data set, apart from input dropout.

**Figure 5: Samples from image data sets. Each row corresponds to a different category.** Panel **(a)** shows Street View House Numbers (SVHN); panel **(b)** shows CIFAR-10.

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

ImageNet contains over 15 million labeled high-resolution images belonging to roughly 22,000 categories. Starting in 2010, an annual ImageNet Large-Scale Visual Recognition Challenge (ILSVRC) has been held as part of the Pascal Visual Object Challenge. The challenge uses a subset of ImageNet with roughly 1000 images in each of 1000 categories.

Because the number of categories is large, the conventional evaluation reports two error rates: top-1 and top-5. Top-1 concerns the model’s most probable label. Top-5 error is the fraction of test images for which the correct label is not among the five labels the model considers most probable. Figure 6 shows predictions on a few test images.

ILSVRC-2010 is the only ILSVRC version whose test-set labels are available. Therefore, most of our experiments used ILSVRC-2010. Table 5 compares methods and shows that convolutional nets with dropout outperform the other methods by a large margin: top-1 and top-5 errors are 37.5 and 17.0, compared with 45.7 and 25.7 for SIFT + Fisher Vectors and 47.1 and 28.2 for Sparse Coding. SIFT denotes the scale-invariant feature transform. Krizhevsky et al. (2012) describe the architecture and implementation in detail.

**Figure 6: Some ImageNet test cases with the 4 most probable labels as predicted by our model. The length of the horizontal bars is proportional to the probability assigned to the labels by the model. Pink indicates ground truth.**

| Model | Top-1 | Top-5 |
|---|---|---|
| Sparse Coding (Lin et al., 2010) | 47.1 | 28.2 |
| SIFT + Fisher Vectors (Sanchez and Perronnin, 2011) | 45.7 | 25.7 |
| Conv Net + dropout (Krizhevsky et al., 2012) | 37.5 | 17.0 |

**Table 5: Results on the ILSVRC-2010 test set.**

Our model based on convolutional nets and dropout won ILSVRC-2012. Because test-set labels are unavailable, Table 6 reports test results for the final submission and validation results for different model variations. In Table 6, “val” means validation, “Avg” means average, and FVs means Fisher Vectors. LBP denotes local binary patterns, GIST is the feature name used in the listed method, and CSIFT denotes color SIFT.

| Model | Top-1 (val) | Top-5 (val) | Top-5 (test) |
|---|---|---|---|
| SVM on Fisher Vectors of Dense SIFT and Color Statistics | - | - | 27.3 |
| Avg of classifiers over FVs of SIFT, LBP, GIST and CSIFT | - | - | 26.2 |
| Conv Net + dropout (Krizhevsky et al., 2012) | 40.7 | 18.2 | - |
| Avg of 5 Conv Nets + dropout (Krizhevsky et al., 2012) | 38.1 | 16.4 | 16.4 |

**Table 6: Results on the ILSVRC-2012 validation/test set.**

The best methods based on standard vision features achieve about 26% top-5 error, whereas convolutional nets with dropout achieve about 16% test error—a staggering difference. Table 6 gives the corresponding test results as 26.2 for the average of feature-based classifiers and 16.4 for the average of 5 convolutional nets with dropout. The examples in Figure 6 also show that the model makes very reasonable predictions even when the model’s best guess is incorrect.

### 6.2 Results on TIMIT

We next applied dropout to speech recognition using TIMIT. TIMIT contains recordings from 680 speakers covering 8 major dialects of American English. Each speaker reads ten phonetically rich sentences in a controlled, noise-free environment.

Dropout neural networks were trained on windows of 21 log-filter bank frames to predict the label of the central frame. No speaker-dependent operations were performed. Appendix B.4 describes preprocessing and training details.

Table 7 compares models. A 6-layer neural net gives a phone error rate of 23.4%, and dropout improves the rate to 21.8%. Dropout also improves nets initialized with pretrained weights. A 4-layer net pretrained with a stack of Restricted Boltzmann Machines gives 22.7% phone error; dropout reduces that error to 19.7%. Similarly, for an 8-layer net the error reduces from 20.5% to 19.7%.

In the table, mcRBM denotes the mean-covariance Restricted Boltzmann Machine used by Dahl et al. (2010).

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

To test dropout in the text domain, we trained a document classifier using dropout networks. We used a subset of Reuters-RCV1, a collection of over 800,000 Reuters newswire articles covering a variety of topics. The task is to take a bag-of-words representation of a document and classify the document into one of 50 disjoint topics. Appendix B.5 describes the setup in more detail.

Our best neural net without dropout obtained 31.05% error. Adding dropout reduced error to 29.62%. Thus, dropout improved text classification, but the improvement was much smaller than the improvements on the vision and speech data sets.

### 6.4 Comparison with Bayesian Neural Networks

Dropout can be viewed as equally weighted averaging of exponentially many models with shared weights. In contrast, Bayesian neural networks (Neal, 1996) provide the proper way to average models over neural-network structures and parameters. Dropout weights each model equally, whereas a Bayesian neural network weights each model according to the prior and how well the model fits the data. The Bayesian weighting is the more correct approach.

Bayesian neural nets are extremely useful when data are scarce, as in medical diagnosis, genetics, drug discovery and other computational biology applications. However, Bayesian neural nets are slow to train, difficult to scale to very large networks, and expensive to use when predictions must be obtained from many large nets at test time. Dropout neural nets, in contrast, are much faster to train and use at test time.

We therefore compare Bayesian and dropout neural nets on a small data set where Bayesian neural networks are known to perform well and obtain state-of-the-art results. The aim is to analyze how much dropout loses relative to Bayesian neural nets.

The data set comes from genetics (Xiong et al., 2011). The task is to predict alternative splicing from RNA features. Alternative splicing is a significant cause of cellular diversity in mammalian tissues. Predicting alternative splicing in particular tissues under different conditions is important for understanding many human diseases.

Given RNA features, the task is to predict the probabilities of three splicing-related events of interest to biologists. The evaluation metric, **Code Quality**, measures negative Kullback–Leibler (KL) divergence between the target and predicted probability distributions; higher Code Quality is better. Appendix B.6 describes the data set and metric in detail.

In Table 8, PCA means principal component analysis.

| Method | Code Quality (bits) |
|---|---|
| Neural Network (early stopping) (Xiong et al., 2011) | 440 |
| Regression, PCA (Xiong et al., 2011) | 463 |
| SVM, PCA (Xiong et al., 2011) | 487 |
| Neural Network with dropout | 567 |
| Bayesian Neural Network (Xiong et al., 2011) | 623 |

**Table 8: Results on the Alternative Splicing Data Set.**

Xiong et al. (2011) used Bayesian neural nets for this task. As expected, Bayesian neural nets perform better than dropout: Table 8 reports 623 bits for the Bayesian neural network and 567 bits for dropout. However, dropout significantly improves on the standard neural network’s 440 bits and outperforms the other methods, which obtain 463 and 487 bits.

The main challenge is preventing overfitting because the training set is small. One way to prevent overfitting is to reduce input dimensionality with principal component analysis and then apply standard methods such as support vector machines or logistic regression. With dropout, however, we could prevent overfitting without dimensionality reduction. The dropout nets contain 1000s of hidden units, compared with a few tens of units in the Bayesian network. This ability to use a much larger network shows dropout’s strong regularizing effect.

### 6.5 Comparison with Standard Regularizers

Several methods have been proposed to prevent overfitting in neural networks, including L2 weight decay—more generally, Tikhonov regularization (Tikhonov, 1943)—lasso (Tibshirani, 1996), KL-sparsity, and max-norm regularization. Lasso is the L1 regularization method, and KL-sparsity uses a Kullback–Leibler-divergence sparsity penalty. Dropout is another form of neural-network regularization. Here we compare dropout with these methods on MNIST.

For every method, we trained the same rectified-linear-unit architecture,
\[
784\text{-}1024\text{-}1024\text{-}2048\text{-}10,
\]
using stochastic gradient descent. This notation lists the input, hidden-layer and output sizes in order. Hyperparameters associated with each method—decay constants, target sparsity, dropout rate and max-norm upper bound—were selected using a validation set.

| Method | Test Classification error % |
|---|---|
| L2 | 1.62 |
| L2 + L1 applied towards the end of training | 1.60 |
| L2 + KL-sparsity | 1.55 |
| Max-norm | 1.35 |
| Dropout + L2 | 1.25 |
| Dropout + Max-norm | 1.05 |

**Table 9: Comparison of different regularization methods on MNIST.**

Dropout combined with max-norm regularization gives the lowest generalization error: 1.05%, compared with 1.25% for dropout + L2 and 1.35% or higher for the methods without dropout in Table 9.

## 7. Salient Features

The experiments in Section 6 provide strong evidence that dropout improves neural networks. This section examines how dropout affects feature quality and the sparsity of hidden-unit activations. This section also examines how dropout’s advantages vary with retention probability, network size and training-set size. These observations provide insight into why dropout works so well.

### 7.1 Effect on Features

In a standard neural network, the derivative received by each parameter indicates how that parameter should change to reduce the final loss, given what all other units are doing. Units may therefore change to fix other units’ mistakes. This behavior can produce complex co-adaptations, which lead to overfitting because the co-adaptations do not generalize to unseen data.

We hypothesize that dropout prevents hidden-unit co-adaptation by making the presence of other hidden units unreliable. A hidden unit therefore cannot rely on specific other units to correct its mistakes. Instead, the hidden unit must perform well in a wide variety of contexts created by the other hidden units.

To observe this effect directly, we examine first-level features learned on visual tasks with and without dropout.

**Figure 7: Features learned on MNIST with one hidden layer autoencoders having 256 rectified linear units.** Panel **(a)**: Without dropout. Panel **(b)**: Dropout with \(p=0.5\).

Figure 7a shows the features of an MNIST autoencoder with a single hidden layer of 256 rectified linear units, trained without dropout. Figure 7b shows features from an identical autoencoder trained with hidden-layer dropout and retention probability \(p=0.5\). Both autoencoders had similar test reconstruction errors.

Despite the similar reconstruction errors, the features differ. In Figure 7a, it is apparent that the features have co-adapted to produce good reconstructions; each hidden unit on its own does not seem to detect a meaningful feature. In contrast, the hidden units in Figure 7b seem to detect edges, strokes and spots in different image regions. This comparison shows that dropout breaks up co-adaptations, which is probably the main reason dropout lowers generalization error.

### 7.2 Effect on Sparsity

As a side-effect of dropout, hidden-unit activations become sparse even without sparsity-inducing regularizers. Thus, dropout automatically leads to sparse representations.

To observe this effect, we use the autoencoders from Section 7.1 and examine hidden-unit activations on a random mini-batch from the test set. A good sparse model should have only a few highly activated units for any data case. In addition, each unit’s average activation across data cases should be low.

To assess these two properties, Figure 8 plots two histograms for each model. The left histogram for each model shows the distribution of hidden-unit mean activations across the mini-batch. The right histogram shows the distribution of the hidden-unit activations themselves.

**Figure 8: Effect of dropout on sparsity. ReLUs were used for both models.** Panel **(a), Without dropout**: The histogram of mean activations shows that most units have a mean activation of about 2.0. The histogram of activations shows a huge mode away from zero. Clearly, a large fraction of units have high activation. Panel **(b), Dropout with \(p=0.5\)**: The histogram of mean activations shows that most units have a smaller mean mean activation of about 0.7. The histogram of activations shows a sharp peak at zero. Very few units have high activation.

The activation histograms show fewer highly activated hidden units with dropout in Figure 8b than without dropout in Figure 8a. In particular, the network without dropout has significant activation mass away from zero. Mean activations are also smaller with dropout: the overall mean hidden-unit activation is close to 2.0 without dropout and around 0.7 with dropout.

### 7.3 Effect of Dropout Rate

The dropout hyperparameter \(p\) is the probability of retaining a unit. We examine changes in \(p\) under two conditions:

1. The number of hidden units is held constant.
2. The number of hidden units changes so that the expected number retained after dropout is held constant.

For the first condition, we train the same architecture,
\[
784\text{-}2048\text{-}2048\text{-}2048\text{-}10,
\]
with different amounts of dropout. No input dropout is used. Figure 9a shows test error as a function of \(p\).

When architecture is fixed, small \(p\) means that very few units turn on during training. The resulting high training error indicates underfitting. As \(p\) increases, error decreases, becomes flat for \(0.4\leq p\leq0.8\), and then increases as \(p\) approaches 1.

For the second condition, let \(n\) be the number of hidden units in a particular layer. We hold \(pn\), the expected number of retained units, constant. Networks with smaller \(p\) therefore have more hidden units before dropout. After dropout, the expected number present is the same across architectures, although the test networks have different sizes.

In these experiments, \(pn=256\) for the first two hidden layers and \(pn=512\) for the last hidden layer. Figure 9b shows test error as a function of \(p\). Errors at small \(p\) are much lower than in Figure 9a: at \(p=0.1\), error falls from 2.7% to 1.7%. For this choice of \(pn\), values near \(p=0.6\) seem best, but our usual default of \(p=0.5\) is close to optimal.

**Figure 9: Effect of changing dropout rates on MNIST.** Panel **(a), Keeping \(n\) fixed**, and panel **(b), Keeping \(pn\) fixed**, both plot Test Error and Training Error against Probability of retaining a unit (\(p\)). Both horizontal axes have ticks at 0.0, 0.2, 0.4, 0.6, 0.8 and 1.0. The vertical axes are Classification Error %; panel (a) has ticks at 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 and 3.5, and panel (b) has ticks at 0.0, 0.5, 1.0, 1.5, 2.0, 2.5 and 3.0.

### 7.4 Effect of Data Set Size

One test of a good regularizer is whether a model with many parameters can generalize well when trained on a small data set. Huge neural networks trained in the standard way overfit massively on small data sets. We therefore examine whether dropout helps by varying the amount of MNIST training data given to a feed-forward network.

The training subsets contain 100, 500, 1K, 5K, 10K and 50K examples chosen randomly from the MNIST training set. Every experiment uses the architecture
\[
784\text{-}1024\text{-}1024\text{-}2048\text{-}10,
\]
with retention probability \(p=0.5\) at every hidden layer and \(p=0.8\) at the input layer.

Figure 10 shows no improvement from dropout for the extremely small data sets of 100 and 500 examples. The model has enough parameters to overfit those training sets even with dropout’s noise. As the data set grows, the gain from dropout first increases and then declines.

This pattern suggests a “sweet spot” for any given architecture and dropout rate. At that amount of data, the data set is large enough not to be memorized despite the noise, but not so large that overfitting is no longer a problem.

**Figure 10: Effect of varying data set size.** The curves are labeled “With dropout” and “Without dropout.” The vertical axis is Classification Error %, with ticks at 0, 5, 10, 15, 20, 25 and 30. The horizontal axis is Dataset size, with ticks at \(10^2\), \(10^3\), \(10^4\) and \(10^5\).

### 7.5 Monte-Carlo Model Averaging vs. Weight Scaling

Our efficient test-time procedure approximates model combination by scaling down the trained network’s weights. A more expensive but more correct averaging procedure samples \(k\) neural nets using dropout for each test case and averages those nets’ predictions. Here \(k\) is the number of sampled networks. As \(k\to\infty\), this Monte-Carlo model average approaches the true model average.

We compare the procedures to determine how many samples are needed to match weight scaling and how quickly finite-sample error approaches true-model-average error.

Using MNIST again, we classify by averaging predictions from \(k\) randomly sampled networks. Figure 11 compares test error for different values of \(k\) with the weight-scaling error, shown as a horizontal line. At around \(k=50\), Monte-Carlo averaging becomes as good as weight scaling. Beyond that point, Monte-Carlo averaging is slightly better, but the difference remains well within one standard deviation of the approximate method. This result suggests that weight scaling is a fairly good approximation to the true model average.

**Figure 11: Monte-Carlo model averaging vs. weight scaling.** The plotted methods are “Monte-Carlo Model Averaging” and “Approximate averaging by weight scaling.” The vertical axis is Test Classification error %, with ticks at 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30 and 1.35. The horizontal axis is Number of samples used for Monte-Carlo averaging (\(k\)), with ticks at 0, 20, 40, 60, 80, 100 and 120.

## 8. Dropout Restricted Boltzmann Machines

Dropout can be applied to Restricted Boltzmann Machines (RBMs) as well as feed-forward neural networks. This section formally describes Dropout RBMs and presents results illustrating their key properties.

### 8.1 Model Description

Consider an RBM with \(D\) visible units and \(F\) hidden units. The binary visible-state vector is \(v\in\{0,1\}^D\), and the binary hidden-state vector is \(h\in\{0,1\}^F\). The RBM defines
\[
P(h,v;\theta)
=
\frac{1}{Z(\theta)}
\exp\!\left(v^\top Wh+a^\top h+b^\top v\right).
\]
Here \(\theta=\{W,a,b\}\) denotes the model parameters: weight matrix \(W\) and bias vectors \(a,b\). The superscript \(\top\) denotes transpose, and \(Z(\theta)\) is the partition function that normalizes the distribution.

A Dropout RBM augments the RBM with a binary random vector \(r\in\{0,1\}^F\). Independently for each hidden unit \(j\), \(r_j=1\) with probability \(p\). If \(r_j=1\), hidden unit \(h_j\) is retained; otherwise, hidden unit \(h_j\) is dropped.

Let \(\mathbf 1(\cdot)\) be the indicator function, equal to 1 when the stated condition holds and 0 otherwise. The joint distribution of a Dropout RBM is
\[
\begin{aligned}
P(r,h,v;p,\theta)
&=P(r;p)P(h,v\mid r;\theta),\\
P(r;p)
&=\prod_{j=1}^{F}p^{r_j}(1-p)^{1-r_j},\\
P(h,v\mid r;\theta)
&=
\frac{1}{Z'(\theta,r)}
\exp\!\left(v^\top Wh+a^\top h+b^\top v\right)
\prod_{j=1}^{F}g(h_j,r_j),\\
g(h_j,r_j)
&=
\mathbf 1(r_j=1)+\mathbf 1(r_j=0)\mathbf 1(h_j=0).
\end{aligned}
\]
The normalization constant is \(Z'(\theta,r)\). The factor \(g(h_j,r_j)\) enforces \(h_j=0\) whenever \(r_j=0\).

Let \(\sigma\) denote the logistic function. The conditional distribution over hidden units is factorial—that is, it factors into independent terms conditioned on \(v\) and \(r\):
\[
P(h\mid r,v)=\prod_{j=1}^{F}P(h_j\mid r_j,v),
\]
with
\[
P(h_j=1\mid r_j,v)
=
\mathbf 1(r_j=1)
\sigma\!\left(b_j+\sum_i W_{ij}v_i\right).
\]

The distribution over visible units conditioned on \(h\) is the same as in an RBM:
\[
P(v\mid h)=\prod_{i=1}^{D}P(v_i\mid h),
\]
with
\[
P(v_i=1\mid h)
=
\sigma\!\left(a_i+\sum_j W_{ij}h_j\right).
\]

Conditioned on \(r\), the distribution over \(\{v,h\}\) is the distribution an RBM would impose, except that hidden units with \(r_j=0\) are dropped. Therefore, a Dropout RBM can be viewed as a mixture of exponentially many RBMs with shared weights, each using a different subset of hidden units.

### 8.2 Learning Dropout RBMs

Learning algorithms developed for RBMs, including Contrastive Divergence (Hinton et al., 2006), can be applied directly to Dropout RBMs. The only difference is that \(r\) is sampled first, and only retained hidden units are used for training. As with dropout neural networks, a different \(r\) is sampled for each training case in every mini-batch.

Our experiments use CD-1: Contrastive Divergence with one step.

### 8.3 Effect on Features

Dropout improved feature quality in feed-forward networks by reducing co-adaptations. We now examine whether this effect also occurs in Dropout RBMs.

**Figure 12: Features learned on MNIST by 256 hidden unit RBMs. The features are ordered by L2 norm.** Panel **(a)**: Without dropout. Panel **(b)**: Dropout with \(p=0.5\).

Figure 12a shows features learned by a binary RBM with 256 hidden units. Figure 12b shows features learned by a Dropout RBM with the same number of hidden units. The Dropout RBM’s features appear qualitatively different: the features seem coarser than the sharply defined, stroke-like features of the standard RBM. In addition, the Dropout RBM seems to have very few dead units relative to the standard RBM.

### 8.4 Effect on Sparsity

We next examine how Dropout RBM training affects hidden-unit activation sparsity. Figure 13a shows histograms of hidden-unit activations and their means on a test mini-batch after standard RBM training. Figure 13b shows the corresponding histograms for Dropout RBMs.

**Figure 13: Effect of dropout on sparsity.** Panel **(a), Without dropout**: The activation histogram shows that a large number of units have activations away from zero. Panel **(b), Dropout with \(p=0.5\)**: A large number of units have activations close to zero and very few units have high activation.

These histograms clearly indicate that Dropout RBMs learn much sparser representations than standard RBMs, even without an additional sparsity-inducing regularizer.

## 9. Marginalizing Dropout

Dropout can be viewed as adding noise to hidden-unit states. This section explores models obtained by marginalizing that noise. Such models are deterministic versions of dropout: unlike standard “Monte-Carlo” dropout, these models do not require random bits, and gradients can be obtained for the marginalized loss functions.

Previous deterministic algorithms have sought models robust to feature deletion at test time (Globerson and Roweis, 2006). Chen et al. (2012) explored marginalization for denoising autoencoders. Srivastava (2013) discussed marginalizing dropout noise for linear regression. Wang and Manning (2013) further explored marginalizing dropout to speed up training. van der Maaten et al. (2013) investigated different input-noise distributions and the regularizers obtained by marginalizing those distributions. Wager et al. (2013) describe dropout as an adaptive regularizer.

### 9.1 Linear Regression

First consider dropout for linear regression. Let \(X\in\mathbb R^{N\times D}\) be a data matrix containing \(N\) data points with \(D\) dimensions each. Let \(y\in\mathbb R^N\) be the target vector. Linear regression seeks weights \(w\in\mathbb R^D\) minimizing the squared norm
\[
\|y-Xw\|^2.
\]

Suppose each input dimension is retained with probability \(p\). The dropped-out input is \(R*X\), where \(*\) is element-wise multiplication and \(R\in\{0,1\}^{N\times D}\) is a random matrix with entries
\[
R_{ij}\sim\operatorname{Bernoulli}(p).
\]
Writing \(\mathbb E\) for expectation, marginalizing the noise gives the objective
\[
\underset{w}{\operatorname{minimize}}\quad
\mathbb E_{R\sim\operatorname{Bernoulli}(p)}
\left[\|y-(R*X)w\|^2\right].
\]

This objective reduces to
\[
\underset{w}{\operatorname{minimize}}\quad
\|y-pXw\|^2+p(1-p)\|\Gamma w\|^2,
\]
where
\[
\Gamma=\left(\operatorname{diag}(X^\top X)\right)^{1/2}.
\]
Here \(\operatorname{diag}(X^\top X)\) denotes the diagonal part of \(X^\top X\).

Thus, dropout with linear regression is equivalent, in expectation, to ridge regression with this particular \(\Gamma\). This form of \(\Gamma\) essentially scales the weight cost for \(w_i\) by the standard deviation of the \(i\)th data dimension. Consequently, if a data dimension varies greatly, the regularizer tries to squeeze that dimension’s weight more.

Another way to express the objective is to absorb \(p\) into the weights. Define \(\widetilde w=pw\). Then the objective takes the form
\[
\underset{w}{\operatorname{minimize}}\quad
\|y-X\widetilde w\|^2
+
\frac{1-p}{p}\|\Gamma\widetilde w\|^2.
\]
This expression makes the regularization constant’s dependence on \(p\) explicit. When \(p\) is close to 1, all inputs are retained and the regularization constant is small. As \(p\) decreases, more dropout is applied and the regularization constant grows.

### 9.2 Logistic Regression and Deep Networks

For logistic regression and deep neural nets, obtaining a closed-form marginalized model is difficult. However, Wang and Manning (2013) showed that the marginalized model corresponding to dropout in logistic regression can be trained approximately.

Under reasonable assumptions, the distributions over inputs to the logistic unit and over gradients of the marginalized model are Gaussian. The means and variances of those distributions can be computed efficiently. This approximate marginalization outperforms Monte-Carlo dropout in both training time and generalization performance.

However, the assumptions involved become successively weaker as more layers are added. Therefore, these results do not apply directly to deep networks.

## 10. Multiplicative Gaussian Noise

Standard dropout multiplies hidden activations by Bernoulli-distributed random variables that are 1 with probability \(p\) and 0 otherwise. The same idea can be generalized by multiplying activations by random variables from other distributions.

We recently discovered that multiplying by a variable drawn from \(\mathcal N(1,1)\) works just as well as, or perhaps better than, Bernoulli noise. Here \(\mathcal N(\mu,\sigma^2)\) denotes a Gaussian distribution with mean \(\mu\) and variance \(\sigma^2\). This new form of dropout amounts to adding a Gaussian-distributed random variable with zero mean and standard deviation equal to the unit’s activation. Specifically, for hidden activation \(h_i\), the perturbed activation is
\[
h_i+h_i r,\qquad r\sim\mathcal N(0,1),
\]
or, equivalently,
\[
h_i r',\qquad r'\sim\mathcal N(1,1).
\]

More generally, \(r'\sim\mathcal N(1,\sigma^2)\), so \(\sigma\) becomes an additional tunable hyperparameter, just as \(p\) is for Bernoulli dropout. Because the expected activation remains unchanged, Gaussian dropout requires no test-time weight scaling.

So far, we have described Bernoulli dropout as retaining units with probability \(p\) during training and multiplying weights by \(p\) at test time. An equivalent procedure scales retained activations up by \(1/p\) during training and leaves the weights unchanged at test time. The procedures are equivalent with appropriate scaling of the learning rate and weight initializations at each layer.

Using the latter procedure, Bernoulli dropout multiplies \(h_i\) by a random variable \(r_b\) that is \(1/p\) with probability \(p\) and 0 otherwise. Its expectation and variance are
\[
\mathbb E[r_b]=1,\qquad
\operatorname{Var}[r_b]=\frac{1-p}{p}.
\]

For Gaussian multiplicative noise, choose
\[
\sigma^2=\frac{1-p}{p}.
\]
Then the Gaussian multiplier \(r_g\) has
\[
\mathbb E[r_g]=1,\qquad
\operatorname{Var}[r_g]=\frac{1-p}{p}.
\]
Thus, the Bernoulli and Gaussian multipliers can have the same mean and variance. However, given these first- and second-order moments, \(r_g\) has the highest entropy and \(r_b\) has the lowest.

Both extremes work well. Preliminary results in Table 10 suggest that the high-entropy case might work slightly better. For every layer, the Gaussian model uses
\[
\sigma=\sqrt{\frac{1-p}{p}},
\]
with \(p\) taken from the corresponding layer of the Bernoulli model.

| Data Set | Architecture | Bernoulli dropout | Gaussian dropout |
|---|---|---|---|
| MNIST | 2 layers, 1024 units each | 1.08 ± 0.04 | 0.95 ± 0.04 |
| CIFAR-10 | 3 conv + 2 fully connected layers | 12.6 ± 0.1 | 12.5 ± 0.1 |

**Table 10: Comparison of classification error % with Bernoulli and Gaussian dropout.** For MNIST, the Bernoulli model uses \(p=0.5\) for hidden units and \(p=0.8\) for input units. For CIFAR-10, the Bernoulli model uses \(p=(0.9,0.75,0.75,0.5,0.5,0.5)\), from the input layer to the top. The Gaussian models use \(\sigma=\sqrt{(1-p)/p}\). Results were averaged over 10 different random seeds. “conv” denotes convolutional layers.

## 11. Conclusion

Dropout improves neural networks by reducing overfitting. Standard backpropagation learning builds brittle co-adaptations that work on training data but do not generalize to unseen data. Random dropout breaks up these co-adaptations by making the presence of any particular hidden unit unreliable.

Dropout improved neural-net performance across object classification, digit recognition, speech recognition, document classification and computational biology data analysis. This range of successful applications suggests that dropout is a general technique rather than a domain-specific one. Methods using dropout achieve state-of-the-art results on SVHN, ImageNet, CIFAR-100 and MNIST. Dropout also considerably improved standard neural nets on the other data sets.

The dropout idea extends to Restricted Boltzmann Machines and other graphical models. The central idea is to take a large model that overfits easily, then repeatedly sample and train smaller sub-models. RBMs fit naturally into this framework. We developed Dropout RBMs and empirically showed that Dropout RBMs have certain desirable properties.

One drawback is increased training time. A dropout network typically takes 2–3 times longer to train than a standard neural network with the same architecture. A major cause is that parameter updates are very noisy: each training case effectively tries to train a different random architecture. Therefore, the computed gradients are not gradients of the final architecture used at test time, so the long training time is not surprising.

However, this stochasticity likely prevents overfitting, creating a trade-off between overfitting and training time. With more training time, one can use high dropout and suffer less overfitting.

One way to obtain some dropout benefits without stochasticity is to marginalize the noise, producing a regularizer that does the same thing as dropout in expectation. For linear regression, we showed that this regularizer is a modified form of L2 regularization. For more complicated models, obtaining an equivalent regularizer is not obvious. Speeding up dropout is therefore an interesting direction for future work.

## Acknowledgments

This research was supported by OGS, NSERC and an Early Researcher Award.

## Appendix A. A Practical Guide for Training Dropout Networks

Neural networks are infamous for requiring extensive hyperparameter tuning, and dropout networks are no exception. This appendix describes heuristics that may help when applying dropout.

### A.1 Network Size

Dropping units is expected to reduce a neural network’s capacity. If a layer has \(n\) hidden units and each unit is retained with probability \(p\), the expected number present after dropout is only \(pn\), rather than \(n\). Moreover, the set of approximately \(pn\) retained units changes each time, and units are not allowed to build co-adaptations freely.

Therefore, if a layer of \(n\) units is optimal for a standard neural net on a task, a good dropout net should have at least \(n/p\) units in that layer. We found this a useful heuristic for selecting hidden-unit counts in both convolutional and fully connected networks.

### A.2 Learning Rate and Momentum

Dropout adds significant gradient noise compared with standard stochastic gradient descent. Consequently, many gradients tend to cancel one another. To compensate, a dropout net should typically use a learning rate 10–100 times the rate that was optimal for a standard neural net.

High momentum provides another way to reduce the noise’s effect. Although momentum values of 0.9 are common for standard nets, values around 0.95 to 0.99 worked considerably better with dropout. High learning rate and/or momentum significantly speed up learning.

### A.3 Max-norm Regularization

Although large momentum and learning rates speed up learning, they sometimes cause network weights to grow very large. Max-norm regularization can prevent this growth by bounding each hidden unit’s incoming-weight-vector norm by a constant \(c\). Typical values of \(c\) range from 3 to 4.

### A.4 Dropout Rate

Dropout adds the retention-probability hyperparameter \(p\), which controls dropout intensity. Setting \(p=1\) means no dropout; lower \(p\) means more dropout.

Typical hidden-unit retention probabilities range from 0.5 to 0.8. For input layers, the choice depends on the input type. For real-valued inputs such as image patches or speech frames, a typical value is 0.8.

For hidden layers, \(p\) must be chosen together with the hidden-unit count \(n\). Smaller \(p\) requires larger \(n\), which slows training and leads to underfitting. Larger \(p\), however, may provide too little dropout to prevent overfitting.

## Appendix B. Detailed Description of Experiments and Data Sets

.

This appendix describes network architectures and training details for the reported experiments. Code to reproduce the results is available at http://www.cs.toronto.edu/~nitish/dropout.

The implementation is graphics-processing-unit (GPU) based. We used the excellent CUDA libraries cudamat (Mnih, 2009) and cuda-convnet (Krizhevsky et al., 2012) to implement the networks.

### B.1 MNIST

MNIST contains 60,000 training examples and 10,000 test examples, each representing a 28 × 28 digit image. We held out 10,000 randomly selected training images for validation.

Hyperparameters were tuned so that the best validation error occurred after 1 million weight updates. We then combined the validation and training sets and trained for 1 million weight updates. The resulting network was evaluated on the test set.

We chose this validation procedure because hyperparameters were easy to set so that early stopping was unnecessary. Therefore, once hyperparameters were fixed, combining the validation and training sets and training for a very long time made sense.

The Figure 4 architectures comprise every combination of 2, 3 and 4 layers with either 1024 or 2048 units in each layer, giving six architectures in total. For all architectures, including those in Table 2, retention probability was \(p=0.5\) in every hidden layer and \(p=0.8\) in the input layer. All layers used final momentum 0.95 and max-norm weight constraints with \(c=2\).

To test the limits of dropout’s regularization power, we also tried 2- and 3-layer nets with 4096 and 8192 units. The 2-layer nets improved results, as reported in Table 2. However, with the same dropout level, the 3-layer nets performed slightly worse than the 2-layer nets. Increasing dropout improved the 3-layer nets, but not enough to outperform the 2-layer nets.

### B.2 SVHN

SVHN contains approximately 600,000 training images and 26,000 test images. The training set has two parts: a standard labeled training set and an additional set of easy labeled examples.

We constructed a validation set from both parts. Two-thirds came from the standard set, with 400 examples per class, and one-third came from the extra set, with 200 examples per class. The resulting validation set contained 6000 samples. Sermanet et al. (2012) use the same process.

Inputs were red, green and blue (RGB) pixels normalized to zero mean and unit variance. Other preprocessing techniques—global or local contrast normalization and ZCA whitening—gave no noticeable improvements. ZCA denotes zero-phase component analysis; the whitening operation is described in Appendix B.3.

The best architecture we found has three convolutional layers, each followed by max pooling. The convolutional layers have 96, 128 and 256 filters, respectively. Each convolutional layer uses a 5 × 5 receptive field at a stride of 1 pixel. Each max-pooling layer pools 3 × 3 regions at strides of 2 pixels.

Two fully connected hidden layers, each with 2048 units, follow the convolutional layers. All units use rectified linear activation. Dropout is applied to every layer, using
\[
p=(0.9,0.75,0.75,0.5,0.5,0.5)
\]
from the input layer through the convolutional layers to the fully connected layers.

All weights use max-norm constraints with \(c=4\), and all layers use momentum 0.95. These hyperparameters were tuned on the validation set.

Because the training set was quite large, we did not combine validation and training data for final training. We reported the test error of the model with the smallest validation error.

### B.3 CIFAR-10 and CIFAR-100

CIFAR-10 and CIFAR-100 each contain 50,000 training images and 10,000 test images. The data sets contain 10 and 100 image categories, respectively, and all images are 32 × 32 color images.

We used 5,000 training images for validation. As with MNIST, we selected hyperparameters using validation data and then combined validation data with the training set.

Images were preprocessed using global contrast normalization in each color channel, followed by ZCA whitening. For each image and each color channel, global contrast normalization computes the mean pixel intensity and subtracts that mean from the channel. ZCA whitening mean-centers the data, rotates the data onto its principal components, normalizes each component, and rotates the data back.

The network architecture and dropout rates are the same as for SVHN, except that input-layer learning rates had to be smaller.

### B.4 TIMIT

We used the open-source Kaldi toolkit (Povey et al., 2011) to preprocess TIMIT into log-filter banks. A monophone system was trained for forced alignment and to obtain speech-frame labels.

Dropout neural networks were trained on windows of 21 consecutive frames to predict the central frame’s label. No speaker-dependent operations were performed. Inputs were mean-centered and normalized to unit variance.

Retention probability was \(p=0.8\) in input layers and \(p=0.5\) in hidden layers. Every layer used max-norm constraints with \(c=4\). Training used momentum 0.95 and a high learning rate of 0.1. The learning rate decayed as
\[
\epsilon_0(1+t/T)^{-1},
\]
where \(\epsilon_0\) is the initial learning rate, \(t\) indexes training progress, and \(T\) is the decay timescale.

For Deep Belief Network pretraining, we trained Restricted Boltzmann Machines using CD-1. Each input unit’s variance in the Gaussian RBM was fixed to 1.

When finetuning the pretrained Deep Belief Network with dropout, obtaining the best results required a smaller learning rate, about 0.01. Adding max-norm constraints did not improve results.

### B.5 Reuters

The Reuters RCV1 corpus contains more than 800,000 documents categorized into 103 classes arranged in a tree hierarchy. We created a subset of 402,738 articles, with a vocabulary of 2000 words and 50 categories, in which each document belongs to exactly one class. The subset was split into equally sized training and test sets.

We tried many network architectures, and dropout improved classification accuracy for every architecture. However, the improvement was smaller than on image and speech data sets. A possible explanation is that this data set is quite large—more than 200,000 training examples—so overfitting is not a very serious problem.

### B.6 Alternative Splicing

The alternative splicing data set contains data for 3665 cassette exons, 1014 RNA features, and 4 tissue types derived from 27 mouse tissues.

For each input, the target consists of 4 softmax units, one per tissue type. Each softmax unit has 3 biologically important states, labeled \(\mathrm{inc}\), \(\mathrm{exc}\) and \(\mathrm{nc}\). The task for each softmax unit is to predict a distribution over these 3 states that matches the observed distribution from wet-lab experiments as closely as possible.

The evaluation metric is Code Quality. Let \(i\) index data points, \(t\) index tissue types, and \(s\) index the three states. Let \(p_{i,t}^{s}\) be the target probability of state \(s\) for tissue type \(t\) in input \(i\). Let \(q_t^s(r_i)\) be the predicted probability of state \(s\) for tissue type \(t\) given input \(r_i\), and let \(\bar p^s\) be the average of \(p_{i,t}^s\) over \(i\) and \(t\). Code Quality is
\[
\sum_{i=1}^{|\text{data points}|}
\sum_{t\in\text{tissue types}}
\sum_{s\in\{\mathrm{inc},\mathrm{exc},\mathrm{nc}\}}
p_{i,t}^{s}
\log\!\left(\frac{q_t^{s}(r_i)}{\bar p^{s}}\right).
\]

We trained a two-layer dropout network with 1024 units per layer. Retention probability was \(p=0.5\) for the hidden layer and \(p=0.7\) for the input layer. Training used max-norm regularization with high, decaying learning rates. Results were averaged across the same 5 folds used by Xiong et al. (2011).
