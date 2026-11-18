## Summary

The paper asks how to reduce overfitting in large neural networks without training and using many separate networks. Dropout randomly removes units and their connections during training. This samples many smaller networks that share weights. At test time, one full network with scaled weights approximates their combined predictions. The paper reports improvements on image, speech, text and alternative-splicing tasks. It compares dropout with other regularizers and Bayesian neural networks. It also examines learned features, sparsity, dropout rates, network size, training-set size and test-time averaging. Dropout Restricted Boltzmann Machines learn qualitatively different features and sparser representations. For linear regression, marginalizing dropout gives a modified ridge-regression objective. Preliminary experiments compare Bernoulli dropout with multiplicative Gaussian noise. A limitation is training time: dropout networks typically take 2-3 times longer to train than standard networks with the same architecture.

## Glossary

### Models, methods and technical terms

- **Activation:** A unit’s output; the paper examines its value and distribution.
- **Activation function:** The function \(f\) that converts a unit’s input into its output.
- **Alternative splicing:** A significant cause of cellular diversity in mammalian tissues; the paper predicts splicing-related events from RNA features.
- **Annealed or decaying learning rate:** A learning rate that decreases during training.
- **Architecture:** The arrangement and sizes of a network’s layers.
- **Autoencoder:** A network trained to reconstruct its input.
- **Backpropagation:** The procedure that propagates loss derivatives through the network for learning.
- **Bag of words:** The document representation used for the Reuters classification task.
- **Bayesian neural network:** A neural-network model average that weights models using the prior and their fit to the data.
- **Bernoulli dropout:** Dropout using random variables that retain or remove units.
- **Bernoulli distribution:** The distribution used for variables that equal 1 with probability \(p\) and 0 otherwise.
- **Bias:** A model parameter denoted by \(b\), or by \(a\) and \(b\) in the RBM equations.
- **Boltzmann Machine:** A graphical-model architecture to which dropout can be applied.
- **Capacity:** The network property reduced by dropping units, as discussed in Appendix A.1.
- **CD-1:** The Contrastive Divergence training method used for the paper’s RBMs.
- **Co-adaptation:** A dependence in which units work together or correct one another’s mistakes.
- **Code Quality:** The alternative-splicing evaluation metric, described as a measure of negative KL divergence; higher is better.
- **Contrastive Divergence:** An RBM learning algorithm that can also train Dropout RBMs.
- **Conv Net / convolutional neural network:** The network type used for the image experiments, with convolutional layers.
- **Convolutional layer:** An image-network layer described by filters, receptive fields and strides.
- **DAE / Denoising Autoencoder:** An autoencoder trained to reconstruct noise-free input from noisy input.
- **DBM / Deep Boltzmann Machine:** A model used to pretrain neural networks.
- **DBN:** The label used for networks pretrained with stacks of RBMs.
- **Dead unit:** A unit described as dead in the qualitative comparison of RBM features.
- **Deep neural network / deep neural net:** A neural network containing multiple nonlinear hidden layers.
- **Deterministic dropout model:** A model obtained by marginalizing dropout noise.
- **Dimensionality reduction:** Reducing the number of input dimensions, as done with PCA in the alternative-splicing comparison.
- **Dropout:** Temporarily and randomly removing network units and their incoming and outgoing connections during training.
- **Dropout rate / retention probability:** The hyperparameter \(p\), which gives the probability of retaining a unit.
- **Dropout RBM:** An RBM augmented with binary variables that determine which hidden units are retained.
- **Early stopping:** Stopping training when performance on a validation set begins to worsen.
- **Entropy:** The quantity used to compare the Gaussian and Bernoulli noise variables with matched means and variances.
- **Feature:** A representation learned by a hidden unit or an input characteristic used by a model.
- **Feed-forward operation:** Computing successive layers’ inputs and outputs.
- **Finetuning:** Training a pretrained network with backpropagation.
- **Fully connected layer:** A layer type used after convolutional layers in the image networks.
- **Gaussian dropout / multiplicative Gaussian noise:** Multiplying activations by Gaussian random variables.
- **Generalization error:** Error on data not used to train the model, evaluated through test performance.
- **Global contrast normalization:** Subtracting each image color channel’s mean pixel intensity from that channel.
- **Gradient:** A loss derivative used to update a model parameter.
- **Graphical model:** A model class, including Boltzmann Machines, to which dropout can be applied.
- **Hidden layer / hidden unit:** An internal network layer or unit, distinct from the input.
- **Hyperparameter:** A tunable training or model choice, selected using validation data.
- **Input unit / visible unit:** A unit representing the network’s input.
- **KL divergence:** The divergence used to describe the Code Quality metric.
- **KL-sparsity:** A sparsity regularizer compared with dropout.
- **L1 regularization / lasso:** A regularization method included among weight penalties and comparison methods.
- **L2 regularization / L2 weight decay:** A weight penalty used in training and in the regularizer comparisons.
- **L2 and L4 pooling:** Pooling methods named in the SVHN comparison.
- **Learning rate:** The parameter controlling the size of training updates.
- **Linear regression:** Finding weights that minimize the squared discrepancy between targets and linear predictions.
- **Logistic regression:** A comparison method and a model considered for approximate dropout marginalization.
- **Logistic unit:** A unit using the logistic activation function shown in Section 4.
- **Loss function:** The objective minimized during learning.
- **Marginalizing dropout:** Averaging out dropout noise to obtain a deterministic objective or model.
- **Max-norm regularization / constraint:** Bounding each hidden unit’s incoming weight-vector norm by a constant \(c\).
- **Maxout:** A unit type used in cited experiments that replace ReLUs.
- **Max-pooling:** The pooling method used after convolutional layers in the paper’s image networks.
- **mcRBM-DBN:** The pretrained model named in the TIMIT comparison.
- **Mini-batch / minibatch:** A group of training cases over which parameter gradients are averaged.
- **Mix-ability:** The ability of genes to work well with another random set of genes.
- **Model averaging / model combination:** Combining predictions from multiple models.
- **Momentum:** A method used to improve stochastic gradient descent.
- **Monte-Carlo dropout:** The stochastic dropout procedure, contrasted with marginalized models.
- **Monte-Carlo model averaging:** Sampling dropout networks at test time and averaging their predictions.
- **Overfitting:** Learning relationships that occur in training data because of sampling noise but do not hold in test data.
- **Partition function:** The RBM normalization quantity \(Z(\theta)\).
- **PCA:** The dimensionality-reduction method used before regression or SVMs in the alternative-splicing comparison.
- **Permutation invariant setting:** The MNIST setting distinguished from methods that use 2-D spatial information.
- **Posterior probability:** The probability used to weight parameter settings given training data in the Bayesian averaging description.
- **Pretraining:** Training that supplies initial weights before finetuning and can use unlabeled data.
- **Prior:** A component of the model weighting in Bayesian neural networks.
- **Receptive field:** The image region to which a convolutional filter is applied.
- **Regularization / regularizer:** A method or penalty used to reduce overfitting.
- **ReLU / rectified linear unit:** The activation-unit type compared with logistic and maxout units.
- **Restricted Boltzmann Machine / RBM:** A model with visible and hidden units and the joint distribution given in Section 8.1.
- **Ridge regression:** The regression method to which marginalized linear-regression dropout is equivalent in expectation.
- **Sampling noise:** Training-set relationships that need not exist in real test data.
- **Soft weight sharing:** A regularization method cited in the Introduction.
- **Softmax unit:** A target unit with three states in the alternative-splicing experiment.
- **Sparse representation / sparsity:** A representation with few highly activated units per case and low average unit activation.
- **Stochastic gradient descent:** The training procedure used for dropout and standard neural networks.
- **Stochastic pooling:** A pooling method included in the image-model comparisons.
- **Stride:** The pixel step used when applying convolution or pooling.
- **Sub-network / sub-model / thinned network:** The network consisting of units retained after dropout.
- **Supervised learning:** The learning setting used for the paper’s classification and prediction tasks.
- **SVM:** A comparison method used with a Gaussian kernel, PCA or Fisher vectors.
- **Tikhonov regularization:** The broader regularization class under which the paper places L2 weight decay.
- **Top-1 error:** One of the two ImageNet error rates reported.
- **Top-5 error:** The fraction of test images whose correct label is absent from the model’s five most probable labels.
- **Unsupervised feature learning:** Learning features without labels, discussed in connection with denoising autoencoders.
- **Validation set:** Data used to select hyperparameters or the model with the smallest validation error.
- **Weight:** A learned connection parameter shared by the thinned networks.
- **Weight scaling:** Multiplying a unit’s outgoing weights by its retention probability for testing.
- **ZCA whitening:** Mean centering, rotating onto principal components, normalizing each component and rotating back.

### Data sets, feature labels, software and table codes

- **MNIST:** A handwritten-digit image data set.
- **TIMIT:** A clean-speech recognition benchmark.
- **CIFAR-10 / CIFAR-100:** Data sets of tiny color images from 10 and 100 categories.
- **SVHN / Street View House Numbers:** Images of house numbers collected by Google Street View.
- **ImageNet:** A large collection of labeled natural images.
- **ILSVRC / ImageNet Large-Scale Visual Recognition Challenge:** The annual ImageNet competition described in Section 6.1.4.
- **ILSVRC-2010 / ILSVRC-2012:** The 2010 and 2012 versions of that challenge.
- **Reuters-RCV1 / Reuters RCV1:** A corpus of Reuters newswire documents.
- **Alternative Splicing data set:** RNA features and splicing targets used for the computational-biology experiment.
- **RNA features:** The input features used to predict alternative splicing.
- **Binary Features (WDCH):** A feature method listed in the SVHN comparison.
- **HOG:** A method listed in the SVHN comparison.
- **KMeans:** A method listed in the SVHN comparison.
- **SIFT, LBP, GIST, CSIFT:** Feature labels used in the ImageNet comparison.
- **FVs / Fisher Vectors:** Feature representations used by ImageNet comparison methods.
- **Color Statistics:** Features combined with dense SIFT in an ImageNet comparison method.
- **RGB:** The pixel representation normalized for SVHN.
- **Kaldi:** The open-source toolkit used to preprocess TIMIT into log-filter banks.
- **Monophone system:** The system used for forced alignment and speech-frame labels.
- **GPU-based:** The description of the experimental implementation.
- **CUDA libraries:** The libraries used to implement the networks.
- **cudamat / cuda-convnet:** The CUDA libraries named in Appendix B.
- **`http://www.cs.toronto.edu/~nitish/dropout`:** The address given for code reproducing the results.
- **OGS / NSERC / Early Researcher Award:** The research-support sources named in the acknowledgments.
- **NN / Neural Net:** Neural network.
- **Avg:** Average, in model-combination table labels.
- **val / test:** Validation-set and test-set results.
- **NA:** The entry used for the SVM’s unit type and architecture in Table 2.
- **–:** An unreported result in a table.
- **K / M:** Thousands and millions in data-set sizes.
- **dim:** Dimension, as used in Table 1.
- **inc / exc / nc:** The three biologically important states used by each alternative-splicing softmax target.
- **bits:** The unit used for Code Quality in Table 8.

### Symbols and mathematical notation

- **\(L\), \(l\):** The number of hidden layers and a layer index.
- **\(i\), \(j\):** Unit, dimension or data-point indices, according to the equation.
- **\(x\), \(y^{(0)}\):** The input to a neural network.
- **\(z^{(l)}\):** The vector of inputs into layer \(l\).
- **\(y^{(l)}\):** The vector of outputs from layer \(l\).
- **\(\tilde y^{(l)}\):** Layer outputs after element-wise multiplication by a dropout mask.
- **\(W^{(l)}\), \(w_i^{(l)}\), \(b^{(l)}\):** Layer weights, a unit’s weight vector and layer biases.
- **\(W_{\mathrm{test}}^{(l)}\):** Test-time weights, set to \(pW^{(l)}\).
- **\(f\):** An activation function; the paper gives \(1/(1+\exp(-x))\) as an example.
- **\(\exp\):** The exponential function appearing in the activation and probability equations.
- **\(p\):** The probability of retaining a unit or input dimension.
- **\(r^{(l)}\):** A vector of independent Bernoulli dropout variables for layer \(l\).
- **\(*\):** Element-wise multiplication.
- **\(n\):** The number of network units, or hidden units in a layer, according to the discussion.
- **\(2^n\):** The number of possible thinned networks for \(n\) units.
- **\(O(n^2)\):** The stated order of the shared parameter count, or an upper description when the count is smaller.
- **\(pn\):** The expected number of retained units in a layer.
- **\(w\):** A weight vector, including incoming hidden-unit weights or regression weights.
- **\(\|w\|_2\), \(c\):** The incoming weight-vector norm and its max-norm upper bound.
- **\(\|\cdot\|^2\):** The squared norm used in the regression objectives.
- **\(k\):** The number of sampled networks used in Monte-Carlo prediction averaging.
- **\(v\), \(D\):** An RBM’s visible-unit vector and number of visible units.
- **\(h\), \(F\):** An RBM’s hidden-unit vector and number of hidden units.
- **\(\{0,1\}^D\), \(\{0,1\}^F\):** The binary vector spaces used for RBM units and masks.
- **\(\theta=\{W,a,b\}\):** The RBM’s model parameters.
- **\(Z(\theta)\):** The standard RBM partition function.
- **\(r\):** The binary retention vector in the Dropout RBM; also a Gaussian noise variable in Section 10.
- **\(Z'(\theta,r)\):** The Dropout RBM’s normalization constant conditional on \(r\).
- **\(g(h_j,r_j)\):** The factor requiring \(h_j=0\) when \(r_j=0\).
- **\(\mathbf 1(\cdot)\):** The indicator appearing in the Dropout RBM constraints and conditionals.
- **\(P(\cdot)\):** A probability distribution.
- **\(\sigma\):** The activation function in the RBM conditionals; in Section 10, the Gaussian noise standard deviation.
- **\({}^\top\):** Transposition in the matrix and vector expressions.
- **\(\prod\), \(\sum\):** Products and sums over the indicated indices.
- **\(X\in\mathbb R^{N\times D}\):** A regression data matrix with \(N\) data points and \(D\) dimensions.
- **\(y\in\mathbb R^N\):** The regression target vector.
- **\(w\in\mathbb R^D\):** The regression weight vector.
- **\(R\in\{0,1\}^{N\times D}\):** The random regression-input mask, with \(R_{ij}\sim\operatorname{Bernoulli}(p)\).
- **\(\mathbb E\), \(\operatorname{Var}\):** Expectation and variance.
- **\(\Gamma=(\operatorname{diag}(X^\top X))^{1/2}\):** The matrix specifying the marginalized regression weight cost.
- **\(\operatorname{diag}(X^\top X)\):** The diagonal quantity used to construct \(\Gamma\).
- **\(\tilde w=pw\):** The rescaled weight vector in the alternative regression objective.
- **\(h_i\):** A hidden activation in the multiplicative-noise discussion.
- **\(\mathcal N(1,1)\), \(\mathcal N(0,1)\):** The Gaussian distributions used for multiplicative and additive noise.
- **\(r'\sim\mathcal N(1,\sigma^2)\):** The generalized multiplicative Gaussian noise variable.
- **\(\sigma^2\):** Gaussian noise variance, matched to \((1-p)/p\) in the comparison.
- **\(r_b\):** A rescaled Bernoulli variable equal to \(1/p\) with probability \(p\), and 0 otherwise.
- **\(r_g\):** The Gaussian multiplier with mean 1 and variance \((1-p)/p\).
- **\(\epsilon_0,t,T\):** The quantities in the TIMIT learning-rate schedule \(\epsilon_0(1+t/T)^{-1}\).
- **\(p_{i,t}^s\):** The target probability for state \(s\), tissue type \(t\) and input \(i\).
- **\(q_t^s(r_i)\):** The predicted probability for state \(s\), tissue type \(t\) and input \(r_i\).
- **\(\bar p^s\):** The average of \(p_{i,t}^s\) over \(i\) and \(t\).
- **\(|\text{data points}|\):** The number of data points summed over in Code Quality.
- **\(\log\):** The logarithm used in the Code Quality expression.
- **\(\pm\):** The variation reported beside mean classification errors in Table 10.

---

Journal of Machine Learning Research 15 (2014) 1929-1958  
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

*This section introduces dropout and summarizes its reported benefits.*

Deep neural nets—networks with multiple nonlinear hidden layers—can be powerful machine-learning systems when they have many parameters. However, overfitting is a serious problem. Overfitting means learning relationships that hold in training data but not in real test data. Large networks are also slow to use. This makes it difficult to reduce overfitting by combining predictions from many large networks at test time.

Dropout addresses this problem by randomly removing units and their connections during training. This prevents excessive co-adaptation: units depending too much on one another. Training samples an exponential number of “thinned” networks, each containing only retained units. At test time, one unthinned network with smaller weights approximates the average predictions of these networks.

This significantly reduces overfitting and gives major improvements over other regularization methods—methods for reducing overfitting. Dropout improves supervised-learning performance on vision, speech recognition, document classification and computational-biology tasks. It obtains state-of-the-art results on many benchmark data sets.

**Keywords:** neural networks, regularization, model combination, deep learning

**Claims and evidence**

- Dropout improves results across the reported domains: image results in Tables 2–6; TIMIT rows in Table 7; Reuters results of 31.05% without dropout and 29.62% with dropout in Section 6.3; alternative-splicing rows of 440 and 567 bits in Table 8.
- Dropout improves over other regularizers: Table 9, all rows; Dropout + Max-norm gives 1.05%, compared with 1.35% for Max-norm and 1.62% for L2.

## 1. Introduction

*This section explains overfitting, the cost of model combination and dropout’s training and testing procedures.*

Multiple nonlinear hidden layers make deep neural networks expressive. They can learn complicated input-output relationships. With limited training data, many learned relationships reflect sampling noise: they occur in the training set but not in real test data, even when both come from the same distribution.

Methods for reducing this overfitting include early stopping—ending training when validation-set performance worsens—and weight penalties. Examples are L1 and L2 regularization and soft weight sharing (Nowlan and Hinton, 1992).

With unlimited computation, the best way to “regularize” a fixed-sized model is to average predictions over all parameter settings. Each setting is weighted by its posterior probability given the training data. This Bayesian average can sometimes be approximated well for simple or small models (Xiong et al., 2011; Salakhutdinov and Mnih, 2008). We want to approach this Bayesian gold standard with much less computation. We propose an approximation to an equally weighted geometric mean of predictions from exponentially many learned models that share parameters.

**Figure 1 panels:** (a) Standard Neural Net. (b) After applying dropout.

**Figure 1: Dropout Neural Net Model. Left: A standard neural net with 2 hidden layers. Right: An example of a thinned net produced by applying dropout to the network on the left. Crossed units have been dropped.**

Model combination—combining predictions from several models—nearly always improves machine-learning performance. But training many separate large neural networks and averaging their outputs is prohibitively expensive.

Combination helps most when models differ. Neural networks can differ in architecture, meaning their layer arrangement and sizes, or in training data. Finding optimal hyperparameters—tunable model and training choices—for each architecture is daunting. Training each large network also takes substantial computation. Large networks usually need much data, so there may not be enough to train different networks on separate subsets. Even if many large networks could be trained, using all of them at test time is infeasible when fast responses matter.

Dropout addresses both overfitting and efficient model combination. “Dropping out” a hidden or visible unit means temporarily removing it and all incoming and outgoing connections, as in Figure 1. The choice is random.

In the simplest case, each unit is retained independently with probability \(p\). A validation set can select \(p\), or it can be set to 0.5. This seems close to optimal for many networks and tasks. For input units, the best retention probability is usually closer to 1 than to 0.5.

**Figure 2 panels:** (a) At training time: present with probability \(p\), outgoing weights \(w\). (b) At test time: always present, outgoing weights \(pw\).

**Figure 2: Left: A unit at training time that is present with probability p and is connected to units in the next layer with weights w. Right: At test time, the unit is always present and the weights are multiplied by p. The output at test time is same as the expected output at training time.**

Applying dropout samples a thinned network containing the surviving units. A network with \(n\) units represents \(2^n\) possible thinned networks. These networks share weights, so the total parameter count remains \(O(n^2)\), or less. For every presentation of every training case, a new thinned network is sampled and trained. Thus dropout trains a collection of \(2^n\) networks with extensive weight sharing. Each individual thinned network is trained very rarely, if at all.

Explicitly averaging exponentially many predictions is infeasible at test time. Instead, we use one network without dropout and with scaled-down weights. If a unit was retained with probability \(p\), its outgoing weights are multiplied by \(p\), as in Figure 2. This ensures that any hidden unit’s expected training-time output equals its actual test-time output. Weight scaling combines the \(2^n\) weight-sharing networks into one test network.

Across many classification problems, dropout training followed by this approximate averaging gives significantly lower generalization error—error on unseen data—than other regularization methods.

Dropout is not restricted to feed-forward networks, which compute outputs through successive layers. It can also be applied to graphical models such as Boltzmann Machines. We introduce a dropout Restricted Boltzmann Machine, or RBM, and compare it with standard RBMs. The dropout versions are better in certain respects.

The paper proceeds as follows. Section 2 gives the motivation. Section 3 discusses related work. Section 4 defines the model. Section 5 describes training. Section 6 reports experiments across domains and compares regularization and model-combination methods. Section 7 examines network properties and hyperparameters. Section 8 describes Dropout RBMs. Section 9 explores marginalizing dropout, meaning averaging out its noise. Appendix A gives practical guidance on hyperparameters.

c 2014 Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever and Ruslan Salakhutdinov.

**Claims and evidence**

- Dropout with approximate test-time averaging lowers classification error across domains: Tables 2–7 and 9; Reuters results in Section 6.3, 31.05% versus 29.62%.
- Dropout RBMs have advantages in certain respects: Figures 12 and 13 support the qualitative feature and sparsity comparisons; no numerical performance comparison is reported there.

## 2. Motivation

*This section motivates dropout through analogies with sexual reproduction and small conspiracies.*

One motivation comes from a theory of sex in evolution (Livnat et al., 2010). Sexual reproduction takes half of each parent’s genes, adds a small amount of random mutation and combines them into an offspring. Asexual reproduction instead makes a slightly mutated copy of one parent’s genes.

Asexual reproduction might seem better for optimizing individual fitness. A good set of genes that works together can pass directly to offspring. Sexual reproduction can break up these co-adapted sets, especially large ones. Intuitively, that should reduce the fitness of organisms with complicated co-adaptations. Yet most advanced organisms evolved through sexual reproduction.

One possible explanation is that natural selection favors genes’ long-term *mix-ability*: their ability to work well with another random set of genes. This makes genes robust. A gene cannot depend on many specific partners always being present. It must do something useful alone or with a few other genes.

On this theory, sexual reproduction both spreads useful genes and reduces complex co-adaptations that could prevent new genes from improving fitness.

Similarly, a hidden unit trained with dropout must work with randomly sampled other units. This should make it robust and encourage useful features without relying on other hidden units to correct its mistakes. Units within a layer will still learn different things. Making many copies of every hidden unit might seem to provide robustness, but that is a poor solution for the same reason replica codes are a poor solution for a noisy channel.

A related motivation concerns conspiracies. Ten conspiracies of five people each are probably better at creating havoc than one conspiracy requiring fifty people to perform correctly. A large conspiracy can work under stable conditions with ample rehearsal. Under changing conditions, smaller conspiracies have a greater chance of continuing to work.

Complex co-adaptations can work well on training data. On new test data, they are more likely to fail than several simpler co-adaptations that accomplish the same thing.

## 3. Related Work

*This section relates dropout to noisy autoencoders, noise marginalization and robustness to adversarial feature removal.*

Dropout can be interpreted as regularization through noise added to hidden-unit states. Denoising Autoencoders, or DAEs, already used noise in unit states (Vincent et al., 2008, 2010). A DAE is an autoencoder—a network trained to reconstruct its input—that receives noisy input and reconstructs the noise-free input.

Our work extends this idea to hidden layers and interprets it as model averaging. It also extends noise beyond unsupervised feature learning to supervised learning. The method applies to other neuron-based architectures, including Boltzmann Machines.

Although 5% noise typically works best for DAEs, test-time weight scaling lets us use much higher noise. Dropping 20% of input units and 50% of hidden units was often optimal.

Because dropout is stochastic regularization, it is natural to consider the deterministic counterpart obtained by marginalizing its noise. In simple cases, this can be done analytically.

van der Maaten et al. (2013) explored deterministic regularizers for exponential-family noise distributions, including dropout, which they call “blankout noise.” They applied noise to inputs and considered models without hidden layers. Wang and Manning (2013) proposed speeding up dropout by marginalizing its noise. Chen et al. (2012) studied marginalization for denoising autoencoders.

Dropout stochastically minimizes loss under a noise distribution. This is equivalent to minimizing an expected loss. Globerson and Roweis (2006) and Dekel et al. (2010) studied a different setting: an adversary chooses which units to remove, subject to a fixed maximum number. That work also did not examine hidden-unit models.

**Claims and evidence**

- Higher noise was often effective with weight scaling: 20% input dropout and 50% hidden dropout, compared with the stated typical 5% DAE noise; Section 3 text, with no table.
- Dropout can be analytically marginalized in simple cases: Section 9.1 gives the linear-regression objectives; no table reports this derivation.

## 4. Model Description

*This section defines standard and dropout feed-forward computations and the test-time weight scaling.*

Consider a neural network with \(L\) hidden layers, indexed by \(l\in\{1,\ldots,L\}\). Let:

- \(z^{(l)}\) be the vector of inputs into layer \(l\).
- \(y^{(l)}\) be its output vector.
- \(y^{(0)}=x\) be the network input.
- \(W^{(l)}\) and \(b^{(l)}\) be the weights and biases at layer \(l\).

For \(l\in\{0,\ldots,L-1\}\) and hidden unit \(i\), a standard network computes:

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

Here \(w_i^{(l+1)}\) is the unit’s weight vector. The activation function \(f\) converts its input to its output. One example is the logistic function:

\[
f(x)=\frac{1}{1+\exp(-x)}.
\]

With dropout, the computation becomes:

\[
r_j^{(l)}\sim\operatorname{Bernoulli}(p),
\]

\[
\tilde y^{(l)}=r^{(l)}*y^{(l)},
\]

\[
z_i^{(l+1)}
=
w_i^{(l+1)}\tilde y^l+b_i^{(l+1)},
\]

\[
y_i^{(l+1)}
=
f\!\left(z_i^{(l+1)}\right).
\]

The symbol \(*\) means element-wise multiplication. Each component of the mask \(r^{(l)}\) is an independent Bernoulli random variable: it is 1 with probability \(p\), and 0 otherwise.

The sampled mask multiplies the layer outputs to produce thinned outputs \(\tilde y^{(l)}\). These become the next layer’s input. Repeating this at each layer samples a sub-network.

**Figure 3 panels:** (a) Standard network. (b) Dropout network. Both show a bias input \(+1\), three inputs \(y_1^{(l)},y_2^{(l)},y_3^{(l)}\), weights \(w_i^{(l+1)}\), bias \(b_i^{(l+1)}\), input \(z_i^{(l+1)}\), activation function \(f\) and output \(y_i^{(l+1)}\). The dropout panel adds masks \(r_1^{(l)},r_2^{(l)},r_3^{(l)}\) and thinned inputs \(\tilde y_1^{(l)},\tilde y_2^{(l)},\tilde y_3^{(l)}\).

**Figure 3: Comparison of the basic operations of a standard and dropout network.**

For learning, loss derivatives are backpropagated through the sampled sub-network. Backpropagation propagates these derivatives through the network to train its parameters.

At test time:

\[
W_{\mathrm{test}}^{(l)}=pW^{(l)}.
\]

This scaled network is used without dropout, as in Figure 2.

## 5. Learning Dropout Nets

*This section describes dropout training and its use with pretrained networks.*

Dropout networks can be trained with the procedures below.

### 5.1 Backpropagation

*This subsection explains mini-batch training, max-norm constraints and useful optimization settings.*

Dropout networks use stochastic gradient descent, as standard networks do. For each case in a mini-batch—a group of training cases—we sample a thinned network. Forward computation and backpropagation use only that network.

A parameter’s gradients, or loss derivatives, are averaged across the mini-batch. A case that does not use that parameter contributes zero gradient.

Methods that improve stochastic gradient descent also help dropout networks. These include momentum, decreasing or annealed learning rates, and L2 weight decay.

Max-norm regularization is especially useful with dropout. It bounds each hidden unit’s incoming weight-vector norm by a constant \(c\). For incoming weights \(w\), training obeys:

\[
\|w\|_2\leq c.
\]

Whenever \(w\) leaves the ball of radius \(c\), it is projected onto the ball’s surface. The name “max-norm” refers to the upper bound \(c\). A validation set selects this hyperparameter.

Max-norm regularization was previously used in collaborative filtering (Srebro and Shraibman, 2005). It typically improves deep-network stochastic-gradient training even without dropout.

Dropout alone gives significant improvements. Combining it with max-norm regularization, large decaying learning rates and high momentum gives a further significant boost.

A possible explanation is that bounded weight vectors permit very large learning rates without weights blowing up. Dropout noise then helps optimization explore otherwise hard-to-reach regions of weight space. As the learning rate decreases, steps become shorter, exploration decreases and optimization settles into a minimum.

**Claims and evidence**

- Combining dropout and max-norm improves over dropout with L2 in the MNIST comparison: Table 9, “Dropout + L2,” 1.25%, and “Dropout + Max-norm,” 1.05%.
- Max-norm also helps without dropout: Table 9, “Max-norm,” 1.35%, compared with “L2,” 1.62%. The broader optimization explanation is presented as possible, not as a measured result.

### 5.2 Unsupervised Pretraining

*This subsection explains how dropout finetuning can preserve and improve pretrained networks.*

Pretraining supplies initial network weights before finetuning. Networks can be pretrained with stacks of RBMs (Hinton and Salakhutdinov, 2006), autoencoders (Vincent et al., 2010), or Deep Boltzmann Machines, abbreviated DBMs (Salakhutdinov and Hinton, 2009).

Pretraining uses unlabeled data. In some cases, pretraining followed by backpropagation finetuning gives significant gains over training from random initialization.

Dropout can be used during finetuning without changing pretraining. Pretrained weights should first be multiplied by \(1/p\). This makes each unit’s expected output under dropout equal its output during pretraining.

We initially worried that dropout’s stochasticity would erase pretrained information. This happened when finetuning learning rates were comparable to the best rates for randomly initialized networks. With smaller learning rates, pretrained information seemed to remain. Final generalization error then improved over finetuning without dropout.

**Claims and evidence**

- Dropout finetuning improves pretrained MNIST networks: Table 2, DBN rows, 1.18% versus 0.92%; DBM rows, 0.96% versus 0.79%.
- Pretrained information was lost with larger finetuning rates and seemed retained with smaller rates: Section 5.2 text; no numerical comparison or table is supplied for this observation.

## 6. Experimental Results

*This section introduces the data sets and reports dropout comparisons across several domains.*

We trained dropout networks on classification tasks in several domains. Dropout improved generalization on every data set compared with neural networks without dropout.

The data sets are:

- **MNIST:** A standard toy handwritten-digit data set.
- **TIMIT:** A clean-speech recognition benchmark.
- **CIFAR-10 and CIFAR-100:** Tiny natural-image data sets (Krizhevsky, 2009).
- **Street View House Numbers, or SVHN:** House-number images collected by Google Street View (Netzer et al., 2011).
- **ImageNet:** A large collection of natural images.
- **Reuters-RCV1:** Reuters newswire articles.
- **Alternative Splicing:** RNA features for predicting alternative gene splicing (Xiong et al., 2011).

This diversity was chosen to show that dropout is general, not domain-specific. This section gives key results. Appendix B describes all experiments and data sets in more detail.

In Table 1, K denotes thousands, M denotes millions and dim denotes dimension.

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

**Claims and evidence**

- Dropout improves the neural-network comparisons across the listed domains: Tables 2–7; Section 6.3, 31.05% versus 29.62%; Table 8, “Neural Network (early stopping),” 440 bits, versus “Neural Network with dropout,” 567 bits.

### 6.1 Results on Image Data Sets

*This subsection introduces the five image data sets used to evaluate dropout.*

We evaluated dropout on MNIST, SVHN, CIFAR-10, CIFAR-100 and ImageNet. These differ in image type and training-set size. Models achieving state-of-the-art results on all these data sets use dropout.

**Claims and evidence**

- The reported image comparisons include high-performing dropout models: Table 2, DBM + dropout finetuning, 0.79%; Table 3, Conv Net + maxout, 2.47%; Table 4, Conv Net + maxout, 11.68% on CIFAR-10, and dropout in all layers, 37.20% on CIFAR-100; Tables 5–6, convolutional networks with dropout.

### 6.1.1 MNIST

*This subsection compares MNIST models and tests dropout across network sizes and architectures.*

MNIST contains \(28\times28\)-pixel handwritten digits. The task is classification into 10 digit classes.

In Table 2, NN means neural network. DBN labels networks pretrained with stacks of RBMs. ReLUs are rectified linear units, the unit type compared with logistic units. Maxout is the replacement unit type used by Goodfellow et al. (2013). SVM Gaussian kernel is a comparison method; NA is its listed unit-type and architecture entry.

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

In the permutation invariant setting, the best neural networks without dropout or unsupervised pretraining achieve about 1.60% error (Simard et al., 2003). Dropout reduces this to 1.35%. Replacing logistic units with ReLUs (Jarrett et al., 2009) reduces it to 1.25%. Adding max-norm regularization reduces it to 1.06%.

Larger networks perform better. A network with 2 hidden layers of 8192 units each reaches 0.95%. It has more than 65 million parameters and is trained on 60,000 cases. Standard regularization and early stopping have difficulty giving good generalization with a network this large. Dropout prevents overfitting here without even needing early stopping.

Goodfellow et al. (2013) further reduced error to 0.94% by replacing ReLUs with maxout units. All dropout networks use \(p=0.5\) for hidden units and \(p=0.8\) for input units. Appendix B.1 gives details.

Pretraining with RBM stacks and DBMs also improves dropout networks. DBM-pretrained dropout networks reach 0.79% test error, the best reported performance for the permutation invariant setting.

Better results are possible using 2-D spatial information and augmenting training data with distorted images. We demonstrate dropout in that setting on more interesting data sets.

We also tested robustness across architectures while fixing every hyperparameter, including \(p\). Figure 4 shows test error during training. Identical architectures trained with and without dropout form two markedly different trajectory clusters. Dropout gives a large improvement across architectures without architecture-specific hyperparameter tuning.

**Figure 4 plot labels:** Classification Error %; Number of weight updates; Without dropout; With dropout. Error-axis ticks: 1.0, 1.5, 2.0, 2.5. Update-axis ticks: 0, 200000, 400000, 600000, 800000, 1000000.

**Figure 4: Test error for different architectures with and without dropout. The networks have 2 to 4 hidden layers each with 1024 to 2048 units.**

**Claims and evidence**

- Dropout, ReLUs and max-norm progressively reduce error: Table 2, Standard Neural Net, 1.60%; logistic Dropout NN, 1.35%; ReLU Dropout NN, 1.25%; 3-layer, 1024-unit Dropout NN + max-norm, 1.06%.
- Larger dropout networks improve further: Table 2, max-norm rows with 2048, 4096 and 8192 units, 1.04%, 1.01% and 0.95%; the 8192-unit network has more than 65 million parameters and 60,000 training cases in the text.
- Maxout reaches 0.94%: Table 2, Goodfellow et al. row.
- Dropout improves pretrained networks: Table 2, DBN 1.18% versus 0.92%, and DBM 0.96% versus 0.79%.
- Dropout helps across fixed-hyperparameter architectures: Figure 4, networks with 2 to 4 hidden layers and 1024 to 2048 units; individual numerical endpoints are not supplied.

### 6.1.2 Street View House Numbers

*This subsection compares convolutional networks and shows gains from dropout in fully connected and convolutional layers.*

SVHN contains color house-number images collected by Google Street View (Netzer et al., 2011). Figure 5a gives examples. Our experiments use \(32\times32\) color images roughly centered on one digit. The task is to identify that digit.

We used convolutional neural networks, abbreviated Conv Nets in tables (LeCun et al., 1989). The best architecture we found has three convolutional layers followed by 2 fully connected hidden layers. All hidden units are ReLUs. Each convolutional layer is followed by max-pooling, the pooling method used in these networks. Appendix B.2 gives details.

Table 3 also names Binary Features (WDCH), HOG and KMeans as comparison methods, and L2, L4 and stochastic pooling as alternative pooling methods.

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

Dropout was applied throughout the network. Retention probabilities, from input through convolutional and fully connected layers, were:

\[
p=(0.9,0.75,0.75,0.5,0.5,0.5).
\]

Max-norm regularization was used for convolutional and fully connected weights.

Convolutional networks outperform the other methods. The best convolutional networks without dropout achieve 3.95% error. Dropout only in fully connected layers reduces error to 3.02%. Applying it also in convolutional layers reduces error to 2.55%. Maxout units give further gains.

The improvement from 3.02% to 2.55% is noteworthy. Convolutional layers have relatively few parameters, so one might expect little overfitting and little benefit from dropout. But lower-layer dropout still helps by providing noisy input to the higher fully connected layers, preventing their overfitting.

**Claims and evidence**

- Dropout improves the max-pooling network: Table 3, “Conv Net + max-pooling,” 3.95%; dropout in fully connected layers, 3.02%; dropout in all layers, 2.55%.
- Adding convolutional-layer dropout gives a further gain: Table 3, fully connected dropout versus all-layer dropout, 3.02% versus 2.55%.
- Maxout gives a further gain: Table 3, “Conv Net + maxout,” 2.47%.
- Convolutional networks outperform the other listed machine methods: Table 3, convolutional rows versus Binary Features, HOG, Stacked Sparse Autoencoders and KMeans, 36.7%, 15.0%, 10.3% and 9.4%.

### 6.1.3 CIFAR-10 and CIFAR-100

*This subsection reports all-layer dropout improvements on two small-color-image classification tasks.*

CIFAR-10 and CIFAR-100 contain \(32\times32\) color images from 10 and 100 categories, respectively. Figure 5b gives examples. Appendix B.3 describes preprocessing, architectures and other details.

Without data augmentation, Snoek et al. (2012) used Bayesian hyperparameter optimization to obtain 14.98% error on CIFAR-10. Dropout in fully connected layers reduces error to 14.32%. Dropout in every layer reduces it to 12.61%. Goodfellow et al. (2013) reduced it further to 11.68% using maxout instead of ReLU units.

On CIFAR-100, dropout reduces error from 43.48% to 37.20%, a large improvement. Neither experiment uses data augmentation apart from input dropout.

**Figure 5 panels:** (a) Street View House Numbers (SVHN). (b) CIFAR-10.

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

**Claims and evidence**

- Fully connected and all-layer dropout improve CIFAR-10 error: Table 4, Snoek et al. row, 14.98%; fully connected dropout, 14.32%; all-layer dropout, 12.61%.
- Maxout further improves CIFAR-10: Table 4, maxout row, 11.68%.
- Dropout improves CIFAR-100: Table 4, hand-tuned max-pooling row, 43.48%, versus all-layer dropout, 37.20%.
- No data augmentation other than input dropout was used: Section 6.1.3 experimental description; no separate table entry.

### 6.1.4 ImageNet

*This subsection reports ImageNet challenge results for convolutional networks with dropout.*

ImageNet contains over 15 million labeled high-resolution images in roughly 22,000 categories. Beginning in 2010, an annual ImageNet Large-Scale Visual Recognition Challenge, or ILSVRC, has been held as part of the Pascal Visual Object Challenge. It uses roughly 1000 images in each of 1000 categories.

Because there are many categories, results conventionally report top-1 and top-5 errors. Top-5 error is the fraction of test images whose correct label is not among the model’s five most probable labels. Figure 6 shows predictions on several test images.

ILSVRC-2010 is the only ILSVRC version with available test labels, so most experiments used it. Convolutional networks with dropout outperform the other methods by a large margin. Krizhevsky et al. (2012) give architecture and implementation details.

**Figure 6: Some ImageNet test cases with the 4 most probable labels as predicted by our model. The length of the horizontal bars is proportional to the probability assigned to the labels by the model. Pink indicates ground truth.**

Table 5 names SIFT and Fisher Vectors as comparison features.

| Model | Top-1 | Top-5 |
|---|---|---|
| Sparse Coding (Lin et al., 2010) | 47.1 | 28.2 |
| SIFT + Fisher Vectors (Sanchez and Perronnin, 2011) | 45.7 | 25.7 |
| Conv Net + dropout (Krizhevsky et al., 2012) | 37.5 | 17.0 |

**Table 5: Results on the ILSVRC-2010 test set.**

In Table 6, val means validation, Avg means average, and FVs means Fisher Vectors. SIFT, LBP, GIST, CSIFT and Color Statistics are the listed comparison features.

| Model | Top-1 (val) | Top-5 (val) | Top-5 (test) |
|---|---|---|---|
| SVM on Fisher Vectors of Dense SIFT and Color Statistics | - | - | 27.3 |
| Avg of classifiers over FVs of SIFT, LBP, GIST and CSIFT | - | - | 26.2 |
| Conv Net + dropout (Krizhevsky et al., 2012) | 40.7 | 18.2 | - |
| Avg of 5 Conv Nets + dropout (Krizhevsky et al., 2012) | 38.1 | 16.4 | 16.4 |

**Table 6: Results on the ILSVRC-2012 validation/test set.**

Our convolutional-network model with dropout won ILSVRC-2012. Test labels are unavailable, so Table 6 reports the final submission’s test result and validation results for model variants.

The best standard-vision-feature methods reach about 26% top-5 error. Convolutional networks with dropout reach about 16% test error, a staggering difference. Figure 6 shows reasonable predictions even when the model’s best guess is incorrect.

**Claims and evidence**

- Dropout convolutional networks outperform the listed ILSVRC-2010 methods: Table 5, Conv Net + dropout, 37.5 top-1 and 17.0 top-5; Sparse Coding, 47.1 and 28.2; SIFT + Fisher Vectors, 45.7 and 25.7.
- The ILSVRC-2012 test comparison is about 26% versus about 16%: Table 6, standard-feature rows, 27.3 and 26.2 top-5; average of 5 dropout Conv Nets, 16.4 top-5.
- The model won ILSVRC-2012: Section 6.1.4 text; Table 6 supplies its reported competition results.
- Predictions can be reasonable when the best guess is wrong: qualitative examples in Figure 6; no numerical measure is given.

### 6.2 Results on TIMIT

*This subsection reports dropout improvements for speech recognition, with and without pretraining.*

TIMIT contains recordings of 680 speakers from 8 major American-English dialects. Each reads ten phonetically rich sentences in a controlled, noise-free environment.

Dropout networks received windows of 21 log-filter-bank frames and predicted the central frame’s label. No speaker-dependent operations were used. Appendix B.4 gives preprocessing and training details.

A 6-layer network gives 23.4% phone error. Dropout reduces this to 21.8%.

We also used pretrained weights. A 4-layer network pretrained with an RBM stack gives 22.7% phone error; dropout reduces it to 19.7%. Similarly, for an 8-layer network, error falls from 20.5% to 19.7%.

Table 7 also includes the mcRBM-DBN pretrained comparison model.

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

**Claims and evidence**

- Dropout improves the 6-layer network: Table 7, NN (6 layers), 23.4%, versus Dropout NN (6 layers), 21.8%.
- Dropout improves the pretrained 4-layer network: Table 7, 22.7% versus 19.7%.
- Dropout improves the pretrained 8-layer network: the prose reports 20.5% versus 19.7%; Table 7’s DBN-pretrained 8-layer rows report 20.7% versus 19.7%. Table 7 assigns 20.5% to the 5-layer mcRBM-DBN model.

### 6.3 Results on a Text Data Set

*This subsection reports a smaller dropout improvement for Reuters document classification.*

We trained a document classifier on a subset of Reuters-RCV1, a collection of over 800,000 Reuters newswire articles on varied topics.

The task uses a bag-of-words document representation and assigns each document to one of 50 disjoint topics. Appendix B.5 gives the setup.

Our best neural network without dropout gives 31.05% error. Dropout reduces it to 29.62%. This improvement is much smaller than those on the vision and speech data sets.

**Claims and evidence**

- Dropout improves Reuters classification: Section 6.3 text, 31.05% without dropout versus 29.62% with dropout; no table is supplied.
- The improvement is described as smaller than for vision and speech: Section 6.3 text; comparison results appear in Tables 2–7.

### 6.4 Comparison with Bayesian Neural Networks

*This subsection compares dropout and Bayesian neural networks on a small alternative-splicing data set.*

Dropout approximates an equally weighted average of exponentially many models with shared weights. Bayesian neural networks instead average over network structures and parameters (Neal, 1996). Each model is weighted using the prior and its fit to the data. This is the more correct approach.

Bayesian neural networks are especially useful when data is scarce, including medical diagnosis, genetics, drug discovery and other computational-biology applications. But they are slow to train, difficult to scale to very large networks, and expensive to combine at test time. Dropout networks are much faster to train and use.

We compare them on a small genetics data set where Bayesian networks achieve state-of-the-art results. The aim is to measure what dropout loses relative to Bayesian networks.

The task predicts alternative splicing from RNA features (Xiong et al., 2011). Alternative splicing is a significant source of cellular diversity in mammalian tissues. Predicting it in tissues under different conditions matters for understanding many human diseases. Given RNA features, the model predicts probabilities of three splicing-related events of biological interest.

The evaluation metric, Code Quality, measures negative KL divergence between target and predicted distributions. Higher is better. Appendix B.6 describes the data and metric.

PCA is the dimensionality-reduction method used in some comparison rows.

| Method | Code Quality (bits) |
|---|---|
| Neural Network (early stopping) (Xiong et al., 2011) | 440 |
| Regression, PCA (Xiong et al., 2011) | 463 |
| SVM, PCA (Xiong et al., 2011) | 487 |
| Neural Network with dropout | 567 |
| Bayesian Neural Network (Xiong et al., 2011) | 623 |

**Table 8: Results on the Alternative Splicing Data Set.**

As expected, Bayesian networks perform better than dropout. But dropout substantially improves standard neural networks and outperforms every other method listed.

Preventing overfitting is difficult because the training set is small. One approach is PCA followed by SVMs or logistic regression. Dropout prevents overfitting without dimensionality reduction.

The dropout networks have 1000s of hidden units; the Bayesian network has a few tens. This shows a strong regularizing effect.

**Claims and evidence**

- Bayesian networks outperform dropout: Table 8, Bayesian Neural Network, 623 bits, versus Neural Network with dropout, 567 bits.
- Dropout outperforms the other listed methods: Table 8, 567 bits versus early-stopped neural network, 440; Regression, PCA, 463; SVM, PCA, 487.
- Dropout regularizes large networks on this small data set without dimensionality reduction: Section 6.4, 1000s of hidden units versus a few tens in the Bayesian network; Table 1, 2932 training cases and 1014 input dimensions; Table 8, dropout row, 567 bits.

### 6.5 Comparison with Standard Regularizers

*This subsection compares regularizers using one fixed MNIST architecture.*

Existing regularizers include L2 weight decay, or more generally Tikhonov regularization (Tikhonov, 1943); lasso (Tibshirani, 1996); KL-sparsity; and max-norm regularization. KL-sparsity is the sparsity regularizer included in this comparison. Dropout is another regularizer.

We trained the same ReLU architecture, 784-1024-1024-2048-10, by stochastic gradient descent with different regularizers. Validation data selected decay constants, target sparsity, dropout rate and max-norm upper bound. Dropout combined with max-norm gave the lowest generalization error.

| Method | Test Classification error % |
|---|---|
| L2 | 1.62 |
| L2 + L1 applied towards the end of training | 1.60 |
| L2 + KL-sparsity | 1.55 |
| Max-norm | 1.35 |
| Dropout + L2 | 1.25 |
| Dropout + Max-norm | 1.05 |

**Table 9: Comparison of different regularization methods on MNIST.**

**Claims and evidence**

- Dropout plus max-norm gives the lowest error in this comparison: Table 9, 1.05%, versus 1.25%, 1.35%, 1.55%, 1.60% and 1.62% in the other rows.

## 7. Salient Features

*This section examines learned features, sparsity and how dropout depends on model and data choices.*

The preceding experiments strongly support dropout as a useful way to improve neural networks. We now examine how it changes learned features and hidden-unit sparsity. We also study retention probability, network size and training-set size. These observations offer insight into why dropout works well.

### 7.1 Effect on Features

*This subsection compares autoencoder features and relates their differences to co-adaptation.*

**Figure 7 panels:** (a) Without dropout. (b) Dropout with \(p=0.5\).

**Figure 7: Features learned on MNIST with one hidden layer autoencoders having 256 rectified linear units.**

In a standard network, each parameter’s derivative says how to reduce loss given what the other units are doing. Units may therefore learn to correct other units’ mistakes. This can create complex co-adaptations that overfit because they do not generalize.

We hypothesize that dropout prevents co-adaptation by making other hidden units’ presence unreliable. A unit cannot depend on specific units to correct its errors. It must work in many contexts created by different retained units.

To examine this, we compare first-level visual features. Figure 7a shows a single-hidden-layer MNIST autoencoder with 256 ReLUs and no dropout. Figure 7b shows an identical autoencoder with hidden-layer dropout at \(p=0.5\).

Both have similar test reconstruction error. But the features without dropout appear co-adapted: individual units do not seem to detect meaningful features. With dropout, units seem to detect edges, strokes and spots in different image regions.

This shows that dropout breaks up co-adaptations, probably the main reason for its lower generalization error.

**Claims and evidence**

- The feature comparison uses otherwise identical 256-ReLU autoencoders, with dropout \(p=0.5\): Figure 7 and Section 7.1.
- Test reconstruction errors are similar: Section 7.1 text; exact errors are not supplied.
- Dropout features appear more individually meaningful and less co-adapted: Figure 7a versus Figure 7b; qualitative evidence, with no numerical feature-quality measure.

### 7.2 Effect on Sparsity

*This subsection shows that dropout produces sparser autoencoder activations without an explicit sparsity regularizer.*

**Figure 8 panels:** (a) Without dropout. (b) Dropout with \(p=0.5\).

**Figure 8: Effect of dropout on sparsity. ReLUs were used for both models. Left: The histogram of mean activations shows that most units have a mean activation of about 2.0. The histogram of activations shows a huge mode away from zero. Clearly, a large fraction of units have high activation. Right: The histogram of mean activations shows that most units have a smaller mean mean activation of about 0.7. The histogram of activations shows a sharp peak at zero. Very few units have high activation.**

Dropout makes hidden activations sparse even without sparsity-inducing regularizers. A sparse representation has few highly active units for each case and low average activation for each unit.

We evaluated the preceding autoencoders on a random test mini-batch. For each model, one histogram shows the distribution of units’ mean activations across the mini-batch. The other shows the distribution of individual activations.

Figure 8b has fewer highly active units than Figure 8a. The no-dropout network has substantial activation mass away from zero. Mean activations are also lower with dropout. The overall mean is close to 2.0 without dropout and around 0.7 with dropout.

**Claims and evidence**

- Dropout produces sparser activations without an added sparsity regularizer: Figure 8a versus Figure 8b, from the 256-unit autoencoders of Section 7.1.
- Mean activation falls from close to 2.0 to around 0.7: Section 7.2 and Figure 8 caption; no table is supplied.

### 7.3 Effect of Dropout Rate

*This subsection varies retention probability with either network size or expected retained size held fixed.*

The retention probability \(p\) is tunable. We study two settings:

1. Keep the number of hidden units fixed.
2. Change that number so the expected retained number stays fixed.

For the first setting, we use architecture 784-2048-2048-2048-10 without input dropout. Figure 9a plots error against \(p\).

With small \(p\), few units are active during training. The high training error indicates underfitting. Error decreases as \(p\) rises, stays flat for \(0.4\leq p\leq0.8\), then increases as \(p\) approaches 1.

**Figure 9 panels:** (a) Keeping \(n\) fixed. (b) Keeping \(pn\) fixed. Both plots label Test Error and Training Error, with horizontal axis “Probability of retaining a unit (\(p\))” and vertical axis “Classification Error %.” Horizontal ticks: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0. Panel (a) vertical ticks: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5. Panel (b) vertical ticks: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0.

**Figure 9: Effect of changing dropout rates on MNIST.**

For the second setting, keep \(pn\) fixed, where \(n\) is the hidden-unit count in a layer. Smaller \(p\) then requires larger networks. The expected retained unit count is the same, but test networks differ in size.

We use \(pn=256\) for the first two hidden layers and \(pn=512\) for the last. Figure 9b shows much smaller errors at low \(p\) than Figure 9a. At \(p=0.1\), error falls from 2.7% to 1.7%.

For these retained sizes, \(p\) near 0.6 performs best. Our usual default, 0.5, is close to optimal.

**Claims and evidence**

- With fixed architecture, error is flat over \(0.4\leq p\leq0.8\) and rises near 1; low \(p\) underfits: Figure 9a and the stated high training error.
- Keeping expected size fixed reduces the low-\(p\) error: Figure 9b versus Figure 9a, \(p=0.1\), 2.7% versus 1.7%; fixed sizes are \(pn=256,256,512\).
- \(p\) near 0.6 is best here, with 0.5 close to optimal: Figure 9b and Section 7.3 text; no table is supplied.

### 7.4 Effect of Data Set Size

*This subsection varies MNIST training-set size and identifies where dropout gives the greatest benefit.*

A good regularizer should permit good generalization from a large model trained on a small data set. Large standard networks overfit severely on small data sets. We tested whether dropout helps by varying the amount of MNIST training data.

We randomly selected training sets of size 100, 500, 1K, 5K, 10K and 50K. Every experiment used architecture 784-1024-1024-2048-10. Retention was \(p=0.5\) in all hidden layers and \(p=0.8\) at the input.

**Figure 10 plot labels:** With dropout; Without dropout; Classification Error %; Dataset size. Error-axis ticks: 0, 5, 10, 15, 20, 25, 30. Data-size ticks: \(10^2,10^3,10^4,10^5\).

**Figure 10: Effect of varying data set size.**

For very small sets, 100 and 500 cases, dropout gives no improvement. The model has enough parameters to overfit despite dropout noise.

As data increases, dropout’s gain first increases and then declines. This suggests a “sweet spot” for each architecture and dropout rate: enough data to resist memorization despite noise, but not so much that overfitting ceases to be a problem anyway.

**Claims and evidence**

- Dropout gives no improvement for 100 and 500 training cases: Figure 10 and Section 7.4.
- Its gain first grows and then declines with data size: Figure 10, using 100, 500, 1K, 5K, 10K and 50K cases; exact plotted errors are not supplied.
- The proposed “sweet spot” is a suggestion from this pattern: Figure 10, with architecture 784-1024-1024-2048-10 and retention probabilities 0.5 hidden and 0.8 input.

### 7.5 Monte-Carlo Model Averaging vs. Weight Scaling

*This subsection compares test-time sampling with the cheaper weight-scaling approximation.*

Our efficient test procedure combines models approximately by scaling down trained weights. A more expensive but more correct procedure samples \(k\) dropout networks for each test case and averages their predictions. This is Monte-Carlo model averaging.

As \(k\to\infty\), the sampled average approaches the true model average. We ask how many samples match weight scaling and how quickly finite-sample error approaches true-average error.

**Figure 11 plot labels:** Monte-Carlo Model Averaging; Approximate averaging by weight scaling; Test Classification error %; Number of samples used for Monte-Carlo averaging (\(k\)). Sample-count ticks: 0, 20, 40, 60, 80, 100, 120. Error ticks: 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35.

**Figure 11: Monte-Carlo model averaging vs. weight scaling.**

On MNIST, we average predictions from \(k\) randomly sampled networks. Figure 11 compares the resulting error with the weight-scaling error, shown as a horizontal line.

At about \(k=50\), Monte-Carlo averaging matches weight scaling. Beyond that, it is slightly better, but well within one standard deviation. This suggests that weight scaling approximates the true model average fairly well.

**Claims and evidence**

- About 50 sampled networks match weight scaling: Figure 11 and Section 7.5, \(k\approx50\).
- Further sampling is slightly better but within one standard deviation: Section 7.5 and Figure 11; exact errors and standard-deviation values are not supplied.
- Weight scaling is therefore described as a fairly good approximation: the same Figure 11 comparison.

## 8. Dropout Restricted Boltzmann Machines

*This section defines Dropout RBMs and examines their learned features and sparsity.*

Dropout can also be applied to Restricted Boltzmann Machines. We define the model and show results illustrating its properties.

### 8.1 Model Description

*This subsection defines a Dropout RBM as a mixture of weight-sharing RBMs using different hidden-unit subsets.*

Consider an RBM with visible vector \(v\in\{0,1\}^D\) and hidden vector \(h\in\{0,1\}^F\). Here \(D\) and \(F\) are the visible- and hidden-unit counts.

Its probability distribution is:

\[
P(h,v;\theta)
=
\frac{1}{Z(\theta)}
\exp\!\left(v^\top Wh+a^\top h+b^\top v\right).
\]

The parameters are \(\theta=\{W,a,b\}\). The partition function \(Z(\theta)\) normalizes the distribution. Superscript \(\top\) denotes transposition.

A Dropout RBM adds a binary vector \(r\in\{0,1\}^F\). Each \(r_j\) independently equals 1 with probability \(p\). Hidden unit \(h_j\) is retained when \(r_j=1\), and dropped otherwise.

The joint distribution is:

\[
P(r,h,v;p,\theta)
=
P(r;p)P(h,v\mid r;\theta),
\]

\[
P(r;p)
=
\prod_{j=1}^{F}
p^{r_j}(1-p)^{1-r_j},
\]

\[
P(h,v\mid r;\theta)
=
\frac{1}{Z'(\theta,r)}
\exp\!\left(v^\top Wh+a^\top h+b^\top v\right)
\prod_{j=1}^{F}g(h_j,r_j),
\]

\[
g(h_j,r_j)
=
\mathbf1(r_j=1)+
\mathbf1(r_j=0)\mathbf1(h_j=0).
\]

Here \(Z'(\theta,r)\) is the normalization constant. The indicator \(\mathbf1(\cdot)\) appears in the constraint factor \(g\), which requires \(h_j=0\) if \(r_j=0\).

The hidden-unit distribution given \(v\) and \(r\) factorizes:

\[
P(h\mid r,v)
=
\prod_{j=1}^{F}P(h_j\mid r_j,v),
\]

\[
P(h_j=1\mid r_j,v)
=
\mathbf1(r_j=1)
\sigma\!\left(b_j+\sum_i W_{ij}v_i\right).
\]

The function \(\sigma\) is the activation function used in these conditional probabilities.

The visible conditional is the same as in an RBM:

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

Conditioned on \(r\), the distribution over \(\{v,h\}\) is that of an RBM with the \(r_j=0\) hidden units removed. Thus a Dropout RBM is a mixture of exponentially many RBMs with shared weights, each using a different hidden-unit subset.

### 8.2 Learning Dropout RBMs

*This subsection explains how standard RBM learning is applied after sampling a dropout mask.*

RBM learning algorithms, including Contrastive Divergence (Hinton et al., 2006), apply directly. First sample \(r\), then train using only retained hidden units.

As in dropout neural networks, each training case receives a new mask in every mini-batch. Our experiments use CD-1, the Contrastive Divergence procedure specified for these RBMs.

### 8.3 Effect on Features

*This subsection compares features learned by standard and dropout RBMs.*

Dropout improves feed-forward features by reducing co-adaptation. We ask whether the effect also occurs in RBMs.

**Figure 12 panels:** (a) Without dropout. (b) Dropout with \(p=0.5\).

**Figure 12: Features learned on MNIST by 256 hidden unit RBMs. The features are ordered by L2 norm.**

Figure 12a shows a binary RBM with 256 hidden units. Figure 12b shows a dropout RBM with the same count.

Dropout-RBM features look qualitatively different. They seem coarser than the sharply defined, stroke-like standard-RBM features. The dropout RBM also seems to have very few dead units relative to the standard RBM.

**Claims and evidence**

- Dropout-RBM features appear coarser and include fewer dead units: Figure 12a versus Figure 12b, both with 256 hidden units; dropout \(p=0.5\). These are qualitative observations, without numerical counts or feature-quality scores.

### 8.4 Effect on Sparsity

*This subsection shows sparser hidden-unit activations in dropout RBMs.*

**Figure 13 panels:** (a) Without dropout. (b) Dropout with \(p=0.5\).

**Figure 13: Effect of dropout on sparsity. Left: The activation histogram shows that a large number of units have activations away from zero. Right: A large number of units have activations close to zero and very few units have high activation.**

Figure 13a shows hidden activations and their means on a test mini-batch after standard RBM training. Figure 13b shows the corresponding dropout-RBM histograms.

These histograms clearly indicate much sparser representations with dropout, even without an additional sparsity regularizer.

**Claims and evidence**

- Dropout RBMs learn sparser representations without added sparsity regularization: Figure 13a versus Figure 13b, dropout \(p=0.5\); no numerical sparsity measure is reported.

## 9. Marginalizing Dropout

*This section considers deterministic models obtained by averaging out dropout noise.*

Dropout adds noise to hidden-unit states. Marginalizing that noise yields deterministic versions of dropout. Unlike standard “Monte-Carlo” dropout, these models require no random bits, and gradients of their marginalized loss functions can be obtained.

Globerson and Roweis (2006) proposed deterministic algorithms robust to test-time feature deletion. Chen et al. (2012) studied marginalization for denoising autoencoders. Srivastava (2013) discussed dropout marginalization for linear regression.

Wang and Manning (2013) explored marginalization to accelerate dropout. van der Maaten et al. (2013) examined input-noise distributions and their resulting regularizers. Wager et al. (2013) describes dropout as an adaptive regularizer.

### 9.1 Linear Regression

*This subsection derives the marginalized dropout objective for linear regression.*

Let \(X\in\mathbb R^{N\times D}\) contain \(N\) data points with \(D\) dimensions. Let \(y\in\mathbb R^N\) be the target vector. Linear regression finds \(w\in\mathbb R^D\) minimizing:

\[
\|y-Xw\|^2.
\]

Retain each input dimension with probability \(p\). The masked input is \(R*X\), where \(R\in\{0,1\}^{N\times D}\) and \(R_{ij}\sim\operatorname{Bernoulli}(p)\).

Marginalizing the noise gives:

\[
\operatorname*{minimize}_w
\mathbb E_{R\sim\operatorname{Bernoulli}(p)}
\left[\|y-(R*X)w\|^2\right].
\]

Here \(\mathbb E\) denotes expectation. This reduces to:

\[
\operatorname*{minimize}_w
\|y-pXw\|^2
+
p(1-p)\|\Gamma w\|^2,
\]

where:

\[
\Gamma=\left(\operatorname{diag}(X^\top X)\right)^{1/2}.
\]

Thus linear-regression dropout is equivalent in expectation to ridge regression with this particular \(\Gamma\), the matrix setting the weight cost.

This form essentially scales the cost on weight \(w_i\) by the standard deviation of data dimension \(i\). Dimensions that vary more have their weights squeezed more.

Absorbing \(p\) into the weights gives:

\[
\operatorname*{minimize}_w
\|y-X\tilde w\|^2
+
\frac{1-p}{p}\|\Gamma\tilde w\|^2,
\]

where \(\tilde w=pw\).

This explicitly shows how regularization depends on \(p\). Near \(p=1\), nearly all inputs are retained and the regularization constant is small. Decreasing \(p\) increases dropout and the regularization constant.

**Claims and evidence**

- Marginalized linear-regression dropout equals a ridge-regression objective in expectation: the displayed objective with penalty \(p(1-p)\|\Gamma w\|^2\); no table is involved.
- After rescaling \(\tilde w=pw\), the regularization constant is \((1-p)/p\): the final displayed objective; it increases as \(p\) decreases.

### 9.2 Logistic Regression and Deep Networks

*This subsection describes approximate logistic-regression marginalization and its limitation for deep networks.*

Closed-form marginalized models are difficult to obtain for logistic regression and deep networks. Wang and Manning (2013) showed that a marginalized logistic-regression dropout model can be trained approximately.

Under reasonable assumptions, both the logistic unit’s inputs and the marginalized model’s gradients have Gaussian distributions. Their means and variances can be computed efficiently. This approximation outperforms Monte-Carlo dropout in training time and generalization.

However, the assumptions become successively weaker as layers are added. The results therefore do not apply directly to deep networks.

**Claims and evidence**

- Approximate marginalization improves training time and generalization for logistic regression: attributed to Wang and Manning (2013); this paper supplies no numerical comparison or table for that result.
- The result does not directly extend to deep networks: Section 9.2’s stated limitation concerning assumptions as layers are added.

## 10. Multiplicative Gaussian Noise

*This section generalizes dropout to Gaussian multipliers and compares the two noise distributions.*

Dropout multiplies hidden activations by Bernoulli variables: 1 with probability \(p\), otherwise 0. Other multiplier distributions are possible.

We recently found that a multiplier from \(\mathcal N(1,1)\), a Gaussian distribution, works as well as or perhaps better than Bernoulli noise. This is equivalent to adding zero-mean Gaussian noise whose standard deviation equals the unit’s activation.

For hidden activation \(h_i\):

\[
h_i+h_i r,\qquad r\sim\mathcal N(0,1),
\]

or equivalently:

\[
h_i r',\qquad r'\sim\mathcal N(1,1).
\]

More generally, use \(r'\sim\mathcal N(1,\sigma^2)\), where \(\sigma\), the noise standard deviation, is a tunable hyperparameter like \(p\). Expected activation is unchanged, so test-time weight scaling is unnecessary.

Our earlier description retains units with probability \(p\) during training and multiplies weights by \(p\) at testing. An equivalent procedure multiplies retained training activations by \(1/p\) and leaves test weights unchanged. Equivalence requires appropriate learning-rate and weight-initialization scaling at each layer.

Under this second description, the Bernoulli multiplier \(r_b\) equals \(1/p\) with probability \(p\), otherwise 0. Its expectation and variance are:

\[
\mathbb E[r_b]=1,
\qquad
\operatorname{Var}[r_b]=\frac{1-p}{p}.
\]

Set Gaussian variance to:

\[
\sigma^2=\frac{1-p}{p}.
\]

Then the Gaussian multiplier \(r_g\) has:

\[
\mathbb E[r_g]=1,
\qquad
\operatorname{Var}[r_g]=\frac{1-p}{p}.
\]

Thus the multipliers can have matching first and second moments. Given these moments, \(r_g\) has the highest entropy and \(r_b\) the lowest. Both extremes work well. Preliminary results in Table 10 suggest the high-entropy case may be slightly better.

For every layer, Gaussian \(\sigma\) was set to:

\[
\sigma=\sqrt{\frac{1-p}{p}},
\]

using that layer’s Bernoulli \(p\).

| Data Set | Architecture | Bernoulli dropout | Gaussian dropout |
|---|---|---|---|
| MNIST | 2 layers, 1024 units each | 1.08 ± 0.04 | 0.95 ± 0.04 |
| CIFAR-10 | 3 conv + 2 fully connected layers | 12.6 ± 0.1 | 12.5 ± 0.1 |

**Table 10: Comparison of classification error % with Bernoulli and Gaussian dropout. For MNIST, the Bernoulli model uses p = 0.5 for the hidden units and p = 0.8 for the input units. For CIFAR-10, we use p = (0.9, 0.75, 0.75, 0.5, 0.5, 0.5) going from the input layer to the top. The value of σ for the Gaussian dropout models was set to be \(\sqrt{\frac{1-p}{p}}\). Results were averaged over 10 different random seeds.**

**Claims and evidence**

- Preliminary Gaussian-dropout results are slightly better: Table 10, MNIST, 1.08 ± 0.04% Bernoulli versus 0.95 ± 0.04% Gaussian; CIFAR-10, 12.6 ± 0.1% versus 12.5 ± 0.1%.
- The comparison matches multiplier means and variances: the displayed equations give mean 1 and variance \((1-p)/p\) for both.
- Results use 10 random seeds: Table 10 caption. The claim of superiority remains preliminary.

## 11. Conclusion

*This section summarizes dropout’s benefits, its RBM extension and the training-time limitation.*

Dropout reduces overfitting in neural networks. Standard backpropagation builds brittle co-adaptations that work on training data but not unseen data. Random dropout breaks these dependencies by making any particular hidden unit’s presence unreliable.

Dropout improves object classification, digit recognition, speech recognition, document classification and computational-biology analysis. This suggests a general technique rather than a domain-specific one. Methods using dropout achieve state-of-the-art results on SVHN, ImageNet, CIFAR-100 and MNIST. Dropout also considerably improves standard networks on other data sets.

The idea extends to RBMs and other graphical models. Its central principle is to start with a large model that overfits easily, then repeatedly sample and train smaller sub-models. RBMs fit this framework. Our Dropout RBMs empirically show certain desirable properties.

A drawback is longer training. A dropout network typically takes 2-3 times as long to train as a standard network with the same architecture. A major reason is noisy parameter updates. Each case effectively trains a different random architecture, so its gradients are not gradients of the final test architecture.

This likely helps prevent overfitting but creates a training-time trade-off. More training time permits more dropout and less overfitting.

Marginalizing the noise can provide some benefits without stochasticity. For linear regression, the equivalent expected regularizer is a modified L2 penalty. For more complicated models, obtaining an equivalent regularizer is not obvious. Accelerating dropout is an interesting direction for future work.

**Claims and evidence**

- Dropout improves the reported application domains: Tables 2–8 and Section 6.3, Reuters error 31.05% versus 29.62%.
- Dropout methods achieve the stated image-data results: Table 2, DBM + dropout finetuning, 0.79%; Table 3, maxout, 2.47%; Table 4, all-layer dropout on CIFAR-100, 37.20%; Tables 5–6, ImageNet dropout rows.
- Dropout RBMs show different features and greater sparsity: Figures 12–13, qualitative evidence.
- Training typically takes 2-3 times longer: Section 11 text; no timing table is supplied.
- Linear-regression marginalization yields modified L2 regularization: Section 9.1’s derived objectives.

## Acknowledgments

*This section names the research support.*

This research was supported by OGS, NSERC and an Early Researcher Award.

## Appendix A. A Practical Guide for Training Dropout Networks

*This appendix gives practical heuristics for dropout hyperparameters.*

Neural networks are known for extensive hyperparameter tuning. Dropout networks are no exception. The following heuristics may help.

### A.1 Network Size

*This subsection recommends increasing layer size to compensate for dropped units.*

Dropping units reduces network capacity. A layer with \(n\) hidden units and retention probability \(p\) has \(pn\) units present in expectation. The retained set changes each time, and units cannot freely form co-adaptations.

If \(n\) units are optimal for a standard network, a good dropout layer should therefore have at least \(n/p\) units. We found this useful for both convolutional and fully connected networks.

**Claims and evidence**

- The \(n/p\) sizing heuristic was useful for both network types: Appendix A.1 text, based on an expected retained size of \(pn\); no separate numerical evaluation or table is given.

### A.2 Learning Rate and Momentum

*This subsection recommends larger learning rates and higher momentum to counter noisy gradients.*

Dropout adds substantial gradient noise compared with standard stochastic gradient descent. Many gradients cancel. To compensate, a dropout network should typically use 10-100 times the learning rate optimal for a standard network.

High momentum also reduces noise effects. Standard networks commonly use 0.9; with dropout, values around 0.95 to 0.99 work much better. High learning rate and/or momentum significantly accelerate learning.

**Claims and evidence**

- Recommended learning rates are 10-100 times the standard-network optimum: Appendix A.2 text; no timing table is supplied.
- Momentum around 0.95 to 0.99 worked better than the common 0.9: Appendix A.2 text; no separate numerical comparison is supplied.

### A.3 Max-norm Regularization

*This subsection recommends a weight-norm bound to control growth under large updates.*

Large learning rates and momentum speed learning but can make weights very large. Max-norm regularization prevents this by bounding each hidden unit’s incoming weight-vector norm by \(c\).

Typical \(c\) values range from 3 to 4.

### A.4 Dropout Rate

*This subsection gives typical retention probabilities and explains their interaction with network size.*

The added hyperparameter \(p\) controls dropout intensity. At \(p=1\), there is no dropout. Smaller values mean more dropout.

Typical hidden-unit values are 0.5 to 0.8. Input retention depends on input type. For real-valued image patches or speech frames, 0.8 is typical.

Hidden-layer \(p\) must be chosen together with hidden-unit count \(n\). Smaller \(p\) requires larger \(n\), slowing training and causing underfitting. Larger \(p\) may provide too little dropout to prevent overfitting.

## Appendix B. Detailed Description of Experiments and Data Sets

*This appendix specifies the implementation, data preparation, architectures and training settings.*

This appendix gives the details behind the reported experiments. Reproduction code is available at:

`http://www.cs.toronto.edu/~nitish/dropout`

The implementation is GPU-based. We used the CUDA libraries cudamat (Mnih, 2009) and cuda-convnet (Krizhevsky et al., 2012).

### B.1 MNIST

*This subsection details MNIST validation, training schedules, architectures and larger-network experiments.*

MNIST has 60,000 training and 10,000 test cases, each a \(28\times28\) digit image. We randomly held out 10,000 training images for validation.

Hyperparameters were selected so the best validation error occurred after 1 million weight updates. We then reunited training and validation data and trained for 1 million updates before testing.

We chose this procedure because hyperparameters could readily be set so early stopping was unnecessary. Once fixed, they allowed combined-data training for a very long time.

Figure 4 includes all combinations of 2, 3 and 4 hidden layers with either 1024 or 2048 units per layer: six architectures. For these and the Table 2 architectures, every hidden layer used \(p=0.5\), and the input layer used \(p=0.8\). All layers used final momentum 0.95 and weight constraint \(c=2\).

To test regularization limits, we also tried 2- and 3-layer networks with 4096 and 8192 units. The 2-layer networks improved results, as Table 2 shows. Three-layer networks were slightly worse at the same dropout level. Increasing dropout helped, but not enough to beat the 2-layer networks.

**Claims and evidence**

- Larger 2-layer networks improve performance: Table 2, max-norm rows, 4096 units, 1.01%, and 8192 units, 0.95%.
- Three-layer versions were slightly worse, and stronger dropout did not reverse this: Appendix B.1 text; numerical errors are not supplied.
- Figure 4 uses six architectures with fixed settings: Appendix B.1, 2, 3 or 4 layers; 1024 or 2048 units; hidden \(p=0.5\), input \(p=0.8\), final momentum 0.95 and \(c=2\).

### B.2 SVHN

*This subsection specifies the SVHN split, preprocessing, network and model-selection procedure.*

SVHN contains approximately 600,000 training and 26,000 test images. Training data has two parts: a standard labeled set and an additional set of easy labeled examples.

We formed a 6000-case validation set using 400 examples per class from the standard set and 200 per class from the extra set. Thus two-thirds came from the standard set and one-third from the extra set. Sermanet et al. (2012) use the same process.

Inputs were RGB pixels normalized to zero mean and unit variance. Global or local contrast normalization and ZCA whitening gave no noticeable improvement.

The best architecture found has three convolutional layers, each followed by max-pooling. The layers have 96, 128 and 256 filters. Each filter uses a \(5\times5\) receptive field, the input region to which it is applied, with a stride or step of 1 pixel.

Each pooling layer pools \(3\times3\) regions with a 2-pixel stride. Two fully connected hidden layers follow, each with 2048 units. All units use rectified linear activation.

From input through convolutional and fully connected layers, retention probabilities were:

\[
p=(0.9,0.75,0.75,0.5,0.5,0.5).
\]

All weights used max-norm \(c=4\), and all layers used momentum 0.95. Validation data selected these settings.

Because the training set was large, validation data was not added back for final training. We report the test error of the model with the smallest validation error.

**Claims and evidence**

- Additional preprocessing gave no noticeable improvement: Appendix B.2 text, comparing global/local contrast normalization and ZCA whitening; numerical comparisons are not supplied.
- The selected all-layer-dropout network reaches 2.55% error: Table 3, “Conv Net + max pooling + dropout in all layers”; Appendix B.2 gives its 96, 128 and 256 filters, two 2048-unit layers, \(c=4\), and momentum 0.95.

### B.3 CIFAR-10 and CIFAR-100

*This subsection specifies the CIFAR data splits, normalization and shared SVHN architecture.*

CIFAR-10 and CIFAR-100 each contain 50,000 training and 10,000 test images, with 10 and 100 categories respectively. Images are \(32\times32\) and color.

We used 5,000 training images for validation. As with MNIST, we selected hyperparameters on validation data, then combined it with training data.

Preprocessing first applied global contrast normalization separately in each color channel. For each image and channel, we computed the mean pixel intensity and subtracted it from that channel.

ZCA whitening followed. This means mean centering the data, rotating onto its principle components, normalizing each component and rotating back.

Architecture and dropout rates matched SVHN. The exception was the input-layer learning rates, which had to be smaller.

### B.4 TIMIT

*This subsection specifies speech preprocessing, dropout settings, learning-rate decay and pretraining.*

We used the open-source Kaldi toolkit (Povey et al., 2011) to produce log-filter banks. A monophone system performed forced alignment and provided speech-frame labels.

Networks received 21 consecutive frames and predicted the central frame’s label. No speaker-dependent operation was performed. Inputs were mean centered and normalized to unit variance.

Retention was \(p=0.8\) at the input and 0.5 in hidden layers. All layers used max-norm \(c=4\). Momentum was 0.95, with a high learning rate of 0.1.

The learning-rate schedule was:

\[
\epsilon_0(1+t/T)^{-1},
\]

with \(\epsilon_0,t,T\) the quantities used in this decay schedule.

For DBN pretraining, RBMs were trained with CD-1. Each Gaussian RBM input unit’s variance was fixed at 1.

For dropout finetuning of the DBN, the best results required a smaller learning rate, about 0.01. Adding max-norm constraints gave no improvement.

**Claims and evidence**

- Dropout DBN finetuning required a smaller learning rate for the best results: Appendix B.4, about 0.01, compared with the stated 0.1 training rate; no rate-comparison table is supplied.
- Max-norm added no improvement to this finetuning setup: Appendix B.4 text; no numerical comparison is supplied.
- The reported speech performance is in Table 7: dropout rows, 21.8%, 19.7% and 19.7%.

### B.5 Reuters

*This subsection defines the Reuters subset and reports the consistency and limited size of dropout’s gains.*

Reuters RCV1 contains more than 800,000 documents in 103 classes arranged in a tree hierarchy.

We created a subset of 402,738 articles using a vocabulary of 2000 words and 50 categories. Every document belongs to exactly one class. The subset was split into equally sized training and test sets.

Dropout improved classification accuracy across all the many architectures tried. But the gain was smaller than for image and speech data.

A possible explanation is that this data set is large—more than 200,000 training cases—so overfitting is not a very serious problem.

**Claims and evidence**

- Dropout improved every tried architecture, but less than on image and speech tasks: Appendix B.5 text; individual architecture results are not supplied.
- The best-network comparison is 31.05% without dropout and 29.62% with dropout: Section 6.3; no table.
- The large-data explanation is tentative: Appendix B.5, more than 200,000 training cases from an equally split 402,738-article subset.

### B.6 Alternative Splicing

*This subsection defines the alternative-splicing targets, Code Quality metric and dropout training setup.*

The data contains 3665 cassette exons, 1014 RNA features and 4 tissue types derived from 27 mouse tissues.

For each input, the target contains 4 softmax units, one per tissue type. Each softmax has 3 biologically important states: inc, exc and nc. The goal is to match the state distribution observed in wet-lab experiments as closely as possible.

Code Quality is:

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

- \(p_{i,t}^{s}\) is the target probability for state \(s\), tissue \(t\) and input \(i\).
- \(q_t^{s}(r_i)\) is the predicted probability for state \(s\), tissue \(t\) and input \(r_i\).
- \(\bar p^{s}\) is the average of \(p_{i,t}^{s}\) over \(i\) and \(t\).
- \(|\text{data points}|\) is the number of data points in the sum.

We trained a two-layer dropout network with 1024 units per layer. Retention was \(p=0.5\) in hidden layers and \(p=0.7\) at the input. Training used max-norm regularization and high, decaying learning rates.

Results were averaged over the same 5 folds used by Xiong et al. (2011).
