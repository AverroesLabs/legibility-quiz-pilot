## Summary

The paper studies how to prevent large neural networks from overfitting. Dropout temporarily removes randomly selected units and their connections during training. This trains many smaller networks that share parameters. At test time, one complete network with reduced weights approximates their averaged predictions. Dropout improves generalization over networks without dropout on every tested data set. Experiments cover images, speech, documents and gene-splicing prediction. The paper reports state-of-the-art results on several benchmarks. It also examines learned features, sparsity, retention rates, training-set size, Restricted Boltzmann Machines, noise marginalization and Gaussian corruption. Training typically takes 2–3 times as long as training a standard network of the same architecture.

## Glossary

Definitions below use only the paper’s descriptions. Where the paper does not expand a name or abbreviation, the entry identifies its stated use.

### Methods, models and codes

- **QX2; dropout:** Random removal of units and their connections during training, followed by approximate prediction averaging at test time.
- **Neural network; neural net; NN:** A model whose multilayer, non-linear hidden representations express input–output relationships.
- **Deep learning; deep neural net:** Learning with neural networks containing multiple hidden layers.
- **Unit; neuron:** A network component with an input, an activation function and an output.
- **Hidden unit; hidden layer:** A unit or layer forming an intermediate representation in a network.
- **Visible unit; input unit:** A unit representing a model’s visible data or inputs.
- **Thinned network; subnetwork; submodel:** A network formed by removing selected units and their connections.
- **Unthinned network:** The complete network used for approximate averaging at inference.
- **Weight; parameter:** A learned model quantity, including connection weights and biases.
- **Bias:** The additive term in a unit’s preactivation.
- **Activation; activation function:** The function applied to a unit’s preactivation to produce its output.
- **Logistic unit:** A unit using \(f(x)=1/(1+\exp(-x))\).
- **F3N; rectified linear units; ReLUs:** The activation class compared with logistic units in the experiments.
- **L8J; maxout:** The activation class used for further improvements in several comparisons.
- **Co-adaptation:** Dependence among units that allows them to repair one another’s mistakes.
- **Overfitting:** Learning relationships present in training data but absent in genuine test data.
- **Generalization:** Performance on data outside the training set.
- **Regularization; regularizer:** A method used to limit overfitting.
- **O3U; L1 regularization; lasso:** A weight-penalty regularizer included in the comparisons.
- **Y5F; L2 regularization:** A weight-penalty regularizer used in training and comparisons.
- **Tikhonov regularization:** The more general regularization family mentioned with Y5F.
- **V9C; max-norm regularization:** A bound \(c\) on each hidden unit’s incoming-weight-vector norm.
- **KL-sparsity:** The sparsity regularizer included in the H4Z comparison.
- **Soft weight sharing:** An established response to overfitting cited in the Introduction.
- **Early stopping:** Stopping training when validation performance deteriorates.
- **Model combination; model averaging:** Combining predictions from multiple models.
- **Posterior-weighted averaging:** Averaging parameter settings using weights that incorporate prior information and data fit.
- **Geometric mean:** The equally weighted prediction combination that dropout approximates.
- **T6R; weight scaling:** Test-time use of one unthinned network with outgoing weights multiplied by retention probability.
- **X9M; Monte-Carlo model averaging:** Averaging predictions from independently sampled thinned networks.
- **Monte-Carlo dropout:** The stochastic dropout procedure contrasted with deterministic marginalization.
- **Marginalization:** Taking the expectation over dropout noise to obtain a deterministic objective or regularizer.
- **Blankout noise:** The name used for dropout in the cited input-noise regularization work.
- **P7V; Bernoulli dropout:** Dropout using binary multiplicative corruption.
- **D4K; Gaussian dropout:** Dropout using Gaussian multiplicative corruption.
- **Bernoulli variable:** A variable equal to one with probability \(p\) and zero otherwise.
- **Gaussian noise:** The multiplicative or additive Gaussian corruption studied in §10.
- **Entropy:** The quantity used to compare Gaussian and Bernoulli multipliers with matched moments.
- **First and second moments:** The multiplier properties matched through mean and variance in §10.
- **S2H; Denoising Autoencoders; DAEs:** Autoencoders trained using corrupted inputs and clean-input reconstruction.
- **Autoencoder:** A model used here to reconstruct inputs and inspect learned features.
- **Boltzmann Machines:** A graphical-model family to which dropout can extend.
- **U2L; Restricted Boltzmann Machines; RBM:** The model family used for pretraining and extended to dropout.
- **B8T; Dropout RBM:** A mixture of shared-weight RBMs using different hidden-unit subsets.
- **E5Q; Deep Boltzmann Machines; DBM:** The model family used for one form of unsupervised initialization.
- **G6W; DBN:** The table designation for networks initialized using stacked U2L.
- **mcRBM-G6W:** The pretrained-network designation used for one N8Q comparison.
- **Pretraining:** Initial learning, potentially from unlabeled data, before supervised finetuning.
- **Finetuning:** Backpropagation training after pretraining.
- **Backpropagation:** Propagation of loss derivatives through a network.
- **Stochastic gradient descent:** Training using parameter gradients averaged over mini-batch cases.
- **Mini-batch:** A group of cases whose gradients are averaged for an update.
- **Momentum:** A training setting used to reduce the effect of gradient noise.
- **Learning rate:** The training-step setting increased and later decayed in the reported procedures.
- **Annealing; learning-rate decay:** Progressive reduction of the learning rate.
- **Hyperparameter:** A setting selected through validation, such as retention probability or a norm bound.
- **Validation set:** Held-out data used to select training settings or models.
- **Inference:** Applying the trained model to obtain predictions.
- **Contrastive Divergence; CD-1:** The U2L learning procedure used in the B8T and pretraining experiments.
- **Convolutional net; Conv Net:** The image-model architecture using convolutional layers.
- **Fully connected layer:** The hidden-layer type following the convolutional layers in the reported architecture.
- **Filter; receptive field; stride:** Convolutional-layer specifications reported in Appendix B.
- **Max-pooling; average pooling; L2 pooling; L4 pooling; stochastic pooling:** Pooling alternatives named in the image comparisons.
- **Padding:** The additional component named in one multi-stage convolutional-net comparison.
- **Data augmentation:** Additional image variation, including distorted training images.
- **Permutation-invariant:** The H4Z comparison setting contrasted with use of two-dimensional spatial information.
- **SVM:** The comparison method used with a Gaussian kernel, PCA or image features.
- **Gaussian kernel:** The kernel named for the H4Z SVM baseline.
- **PCA:** The input-reduction method used before regression or SVMs on Z7P.
- **Linear regression:** The model minimizing \(\|\mathbf y-\mathbf X\mathbf w\|^2\).
- **Ridge regression:** The regression form obtained in expectation from dropout, with a particular penalty matrix.
- **Logistic regression:** The model considered for approximate marginalization and reduced-input Z7P comparisons.
- **Bayesian neural network:** A neural network that averages structures and parameters using posterior weights.
- **Bag-of-words:** The document representation used for M5X classification.
- **Sparse representation; sparsity:** Few strongly active units per case and low average activity per unit.
- **Dead unit:** A unit described as dead in the U2L/B8T feature comparison.
- **Mix-ability:** The ability of a gene set to function with another random gene set.
- **Replica code:** The noisy-channel comparison used to reject simple hidden-unit replication.
- **Non-stationarity:** The changing conditions invoked in the conspiracy analogy.

### Data, evaluation and implementation

- **H4Z; MNIST:** The handwritten-digit data set used for recognition and network studies.
- **A2F; Street View House Numbers; SVHN:** Google Street View house-number images used to identify a centered digit.
- **J3V; CIFAR-10:** The tiny natural-image collection with 10 categories.
- **W6R; CIFAR-100:** The tiny natural-image collection with 100 categories.
- **K9D; ImageNet:** The large collection of labeled, high-resolution images.
- **ILSVRC:** The annual image competition that began in 2010 and uses an ImageNet subset.
- **Pascal Visual Object Challenge:** The challenge of which ILSVRC began as a part.
- **N8Q; TIMIT:** The clean-speech benchmark used for phone recognition.
- **M5X; Reuters-RCV1:** The Reuters newswire document collection.
- **Z7P; Alternative Splicing data set:** The RNA-feature data used for gene-splicing prediction.
- **Alternative splicing:** A process contributing to mammalian-tissue cellular diversity.
- **RNA features:** The Z7P inputs used to predict splicing-related event probabilities.
- **Cassette exon:** The Z7P data item whose count is reported in Appendix B.
- **C4Y; Code Quality:** The distributional score described as negative KL divergence, with higher values preferred.
- **KL divergence:** The distributional quantity used to describe C4Y.
- **Softmax unit:** A target unit giving a distribution over the three Z7P states.
- **inc; exc; nc:** The three biologically important state codes used by each Z7P softmax.
- **Wet-lab-observed distribution:** The observed state distribution that Z7P predictions should match.
- **Phone error rate:** The speech-recognition error reported for N8Q.
- **Top-1 error:** The first-choice image error reported in the K9D tables.
- **Top-5 error:** The fraction of test images whose correct label is absent from the five most probable predicted labels.
- **val; test:** Validation and test result labels in Table 6.
- **WDCH:** The designation attached to the Binary Features A2F baseline.
- **HOG:** The named A2F baseline in Table 3.
- **KMeans:** The named A2F baseline in Table 3.
- **SIFT; Dense SIFT; CSIFT; LBP; GIST:** Feature names used in the K9D comparison tables.
- **Fisher Vectors; FVs:** The image-feature representation named in the K9D baselines.
- **Sparse Coding:** The named ILSVRC-2010 comparison method.
- **RGB:** The pixel representation normalized for A2F.
- **Global contrast normalization:** Subtracting each image/color channel’s mean pixel intensity in the stated preprocessing.
- **Local contrast normalization:** An A2F preprocessing alternative that gave no noticeable improvement.
- **ZCA whitening:** Mean-centering, rotating onto principal components, normalizing components and rotating back.
- **Log-filter banks:** Speech features generated with Kaldi.
- **Monophone system; forced alignment:** The system and procedure supplying speech-frame labels.
- **Kaldi:** The open-source toolkit used for speech preprocessing.
- **GPU; CUDA:** The implementation terms associated with the reported libraries.
- **cudamat; cuda-convnet:** The CUDA libraries used for GPU implementations.
- **OGS; NSERC; Early Researcher Award:** The support sources named in the Acknowledgments.
- **NA:** The entry used where Table 2 supplies no unit type or architecture.
- **K; M:** Size suffixes used in data-set and training-count entries.
- **dim; conv; Avg; NN:** Table abbreviations for dimensionality, convolutional, averaging and neural net.
- **% ; bits:** The units used for reported errors and C4Y scores, respectively.

### Symbols and mathematical notation

- **\(n\):** Network-unit count, or a layer’s hidden-unit count in width comparisons.
- **\(2^n\):** The number of possible thinned networks from an \(n\)-unit network.
- **\(O(n^2)\):** The stated order of the shared-parameter count.
- **\(L\):** Number of hidden layers.
- **\(l\):** Layer index.
- **\(i,j\):** Unit, input-dimension or data-point indices, as specified locally.
- **\(\mathbf x;\mathbf y^{(0)}\):** Network input.
- **\(\mathbf z^{(l)};z_i^{(l)}\):** Layer inputs, or a unit’s preactivation.
- **\(\mathbf y^{(l)};y_i^{(l)}\):** Layer outputs, or a unit’s output.
- **\(\mathbf W^{(l)};\mathbf w_i^{(l)}\):** Layer weights and a unit’s weight vector.
- **\(\mathbf b^{(l)};b_i^{(l)}\):** Layer biases and a unit’s bias.
- **\(f\):** Activation function.
- **\(\exp\):** The exponential operation appearing in activations and model distributions.
- **\(p\):** Probability of retaining a unit or input dimension.
- **\(\mathbf r^{(l)};r_j^{(l)}\):** Layer retention vector and its Bernoulli entries.
- **\(\widetilde{\mathbf y}^{(l)}\):** Thinned layer outputs.
- **\(*\):** Element-wise multiplication.
- **\(\mathbf W_{\mathrm{test}}^{(l)}\):** Test-time weights, equal to \(p\mathbf W^{(l)}\).
- **\(w;pw\):** Training and scaled test weights in Figure 2.
- **\(+1\):** The bias-input unit shown in Figure 3.
- **\(\mathbf w\):** An incoming-weight vector or the regression weight vector, depending on the section.
- **\(\|\cdot\|_2;\|\cdot\|^2\):** The weight-vector norm and squared-error or penalty norm used in the objectives.
- **\(c\):** The max-norm bound.
- **\(pn;n/p\):** Expected retained width and the practical width heuristic, respectively.
- **\(k\):** Number of independently sampled networks used for X9M.
- **\(\mathbf v;v_i;D\):** RBM visible vector, its entries and its dimension.
- **\(\mathbf h;h_j;F\):** RBM hidden vector, its entries and its dimension.
- **\(\theta\):** RBM parameters \(\{\mathbf W,\mathbf a,\mathbf b\}\).
- **\(\mathbf W;W_{ij}\):** RBM weight matrix and an individual connection weight.
- **\(\mathbf a;\mathbf b;a_i;b_j\):** Bias parameters appearing in the RBM equations as written.
- **\(P\):** A model probability or conditional distribution.
- **\(Z(\theta)\):** The ordinary RBM partition function.
- **\(\mathbf r;r_j\):** B8T hidden-unit retention vector and its entries.
- **\(Z'(\theta,\mathbf r)\):** The B8T conditional normalizer.
- **\(g(h_j,r_j)\):** The factor enforcing \(h_j=0\) when \(r_j=0\).
- **\(\mathbf1(\cdot)\):** The indicator used in the B8T equations.
- **\(\sigma(\cdot)\):** The activation appearing in the RBM conditional probabilities.
- **\(\top\):** Matrix or vector transpose in the stated equations.
- **\(\prod;\sum\):** Products and sums over the displayed indices.
- **\(\{0,1\}^D;\{0,1\}^F\):** The binary visible and hidden vector spaces.
- **\(\mathbf X;N;D\):** Regression data matrix, number of data points and number of input dimensions.
- **\(\mathbf y\):** Regression target vector.
- **\(\mathbb R\):** The real-valued spaces used for regression data, targets and weights.
- **\(\mathbf R;R_{ij}\):** The regression input-retention matrix and its Bernoulli entries.
- **\(\mathbb E\):** Expectation over the stated noise distribution.
- **\(\mathbf\Gamma\):** The penalty matrix \((\operatorname{diag}(\mathbf X^\top\mathbf X))^{1/2}\).
- **\(\operatorname{diag}\):** The diagonal operation used to define \(\mathbf\Gamma\).
- **\(\widetilde{\mathbf w}\):** Rescaled regression weights \(p\mathbf w\).
- **\((1-p)/p\):** The marginalized regression penalty coefficient and matched multiplier variance.
- **\(h_i\):** A hidden activation in the multiplicative-noise equations.
- **\(r;r'\):** Gaussian multipliers in the additive and multiplicative noise forms.
- **\(\mathcal N(0,1);\mathcal N(1,1);\mathcal N(1,\sigma^2)\):** The Gaussian distributions specified for the noise variables.
- **\(\sigma;\sigma^2\):** Gaussian-noise standard deviation and variance in §10.
- **\(r_b;r_g\):** The rescaled Bernoulli and Gaussian multipliers.
- **\(\operatorname{Var}\):** Multiplier variance.
- **\(\pm\):** The reported uncertainty notation in Table 10.
- **\(\epsilon_0,t,T\):** The quantities appearing in the N8Q decay schedule \(\epsilon_0(1+t/T)^{-1}\); the paper gives no further definitions there.
- **\(L_2\):** The norm used to order RBM features in Figure 12.
- **\(s;t;i\), in C4Y:** State, tissue-type and input indices.
- **\(p_{i,t}^{s}\):** Target probability for state \(s\), tissue type \(t\), input \(i\).
- **\(q_t^{s}(r_i)\):** Predicted probability for that state and tissue, given input \(r_i\).
- **\(r_i\), in C4Y:** A Z7P input.
- **\(\bar p^{s}\):** \(p_{i,t}^{s}\) averaged over \(i,t\).
- **\(\log\):** The logarithm in the C4Y score.
- **\(|\text{data points}|\):** Number of data points summed over in C4Y.
- **\(\to\infty\):** The increasing-sample limit used to describe X9M’s approach to the true model average.

---

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

Original-page map, including continuations: 1930—Figure 1 and Introduction; 1931—Figure 2 and Introduction; 1932—Introduction ending, §2 and §3 opening; 1933—§3 continuation and §4 opening; 1934—Figure 3, §4 ending and §5.1 opening; 1935—§5.1 continuation, §5.2 and §6 opening; 1936—§6 continuation, Table 1 and §6.1.1 opening with Table 2; 1937—§6.1.1 continuation, Figure 4 and §6.1.2 opening; 1938—Table 3, §6.1.2 continuation and §6.1.3; 1939—Figure 5, Table 4 and §6.1.4 opening; 1940—Figure 6, Tables 5–6, §6.1.4 continuation and §6.2 opening; 1941—§6.2 continuation, Table 7, §6.3 and §6.4 opening; 1942—Table 8, §6.4 continuation, §6.5 and §7 opening; 1943—Table 9, §7 continuation and §7.1 opening with Figure 7; 1944—§7.1 continuation, §7.2 opening with Figure 8; 1945—§7.2 ending and §7.3 with Figure 9; 1946—§§7.4–7.5 openings with Figures 10–11; 1947—§7.5 continuation, §8 and §8.1 opening; 1948—Figure 12, §8.1 continuation, §8.2 and §8.3 opening; 1949—Figure 13, §8.3 continuation, §8.4 and §9 opening; 1950—§9 continuation and §§9.1–9.2; 1951—Table 10, §10 and §11 opening; 1952—§11 continuation, Acknowledgments and Appendix A through A.1; 1953—A.2–A.4, Appendix B and B.1 opening; 1954—B.1 continuation, B.2 and B.3 opening; 1955—B.3 continuation and B.4–B.6.

## Abstract

*The paper presents random unit removal as a way to reduce overfitting and approximate model combination.*

Large, parameter-rich deep neural networks are powerful machine-learning systems. Their overfitting—learning training relationships that do not hold in test data—and slow use obstruct test-time combination of independently trained large models.

The technique **QX2, or dropout**, temporarily removes randomly selected units and their connections during training. It inhibits excessive co-adaptation, or dependence among units, and samples exponentially many thinned networks that share parameters. One unthinned network with reduced weights approximates their averaged predictions.

Compared with other regularizers, methods that limit overfitting, QX2 substantially reduces overfitting and produces major improvements. Supervised experiments in vision, speech recognition, document classification and computational biology attain state-of-the-art performance on many benchmarks.

**Keywords:** neural networks, regularization, model combination, deep learning

**Claims and evidence**
- QX2 improves results across the reported domains. Evidence is reported in Tables 2–8 and the unnumbered text-classification table in §6.3; the domain-specific comparisons are listed in §6.

## 1. Introduction

*QX2 trains many shared-parameter subnetworks and approximates their combined predictions with one test-time network.*

With unlimited computation, posterior-weighted averaging over every parameter setting is the best regularization of a fixed-sized model. Here, posterior weights incorporate prior information and data fit. Good approximations are sometimes available for simple or small models (Xiong et al., 2011; Salakhutdinov and Mnih, 2008).

The proposed, less expensive method approximates an equally weighted geometric mean of exponentially many learned models’ predictions. These models share parameters. A network with \(n\) units represents \(2^n\) possible thinned networks but retains only \(O(n^2)\), or fewer, parameters. A new member is sampled and trained for every presentation of every training case. Each particular member is therefore trained rarely, if ever.

Multilayer non-linear hidden representations express complicated input–output relationships. With limited data, sampling noise creates relationships that occur in training data but not in genuine test data, even when both come from the same distribution. Established responses include:

- **Early stopping:** stopping when validation performance deteriorates.
- **Soft weight sharing:** the regularization method of Nowlan and Hinton (1992).
- Weight penalties, including **O3U, or L1 regularization**, and **Y5F, or L2 regularization**.

These penalties constrain learned parameters rather than average every possible parameter setting.

Model combination, combining predictions from multiple models, nearly always improves machine-learning performance. But separately training many large nets makes output averaging prohibitively expensive. Useful diversity requires different architectures or training data. Architecture-specific optimal hyperparameters—settings selected for training—are difficult to find. Each large-net training run consumes substantial computation. Large data requirements may also prevent training different networks on different subsets. Even after successful training, joint inference can be infeasible for rapid-response applications.

The practical inference rule, **T6R, or weight scaling**, uses one unthinned network instead of evaluating exponentially many members. A more expensive alternative, **X9M, or Monte-Carlo model averaging**, averages predictions from sampled members. T6R gives significantly lower generalization error than other regularization methods across varied classification problems.

The same unit-removal method extends beyond feed-forward nets to graphical models, including Boltzmann Machines. **U2L, or Restricted Boltzmann Machines (RBM)**, admits the augmented model **B8T, or Dropout RBM**, while retaining shared parameters.

**Figure 1.** Standard neural net, left: two hidden layers. QX2-produced thinned net, right: crossed units removed, including their incoming and outgoing connections.

Paper map: §2 motivation; §3 previous work; §4 formal QX2 model; §5 training algorithm; §6 cross-domain experiments and comparisons with regularization and model combination; §7 network-property effects and hyperparameter interactions; §8 B8T; §9 QX2 marginalization; Appendix A practical training guidance, including detailed hyperparameter-choice considerations.

**Claims and evidence**
- T6R gives lower generalization error than the compared regularizers. Table 9, rows “Y5F,” “V9C,” “QX2 + Y5F” and “QX2 + V9C”: 1.62, 1.35, 1.25 and 1.05.
- X9M is more expensive than T6R. §7.5’s unnumbered comparison table reports around \(k=50\) samples before X9M becomes as good as T6R; Appendix B.1 describes sampling \(k\) networks per test case.

## 2. Motivation

*Sexual-reproduction and conspiracy analogies motivate reducing complex co-adaptations.*

Sexual reproduction combines half of one parent’s genes with half of another’s, plus a very small random mutation. Asexual reproduction makes a slightly mutated copy of one parent’s genes. Direct transmission preserves a successful co-adapted gene set. This makes asexual reproduction seem plausibly superior for optimizing individual fitness. Sexual recombination breaks such sets, especially large ones. It therefore seems disadvantageous for organisms with complicated co-adaptations. Nevertheless, most advanced organisms evolved through sexual reproduction.

Livnat et al. (2010) suggest that long-run selection may favor gene **mix-ability** rather than individual fitness. Mix-ability means functioning with another random gene set. Such sets are more robust. A gene cannot rely on receiving a large set of partners, so useful individual action or collaboration with few genes is favored. Reducing complex co-adaptations also helps useful new genes spread by lowering barriers to their improving individual fitness.

QX2-trained hidden units similarly face randomly selected partners. This favors independently useful, robust features instead of reliance on other hidden units to repair mistakes. Units in the same layer still learn different functions. Replicating each hidden unit many times is a poor solution, just as replica codes poorly address noisy channels.

Ten five-person conspiracies probably have an advantage in producing havoc over one conspiracy requiring fifty correct participants. Under unchanged conditions and ample rehearsal, the larger conspiracy can work. Under non-stationarity, smaller conspiracies are more likely to persist. Likewise, complex co-adaptations may succeed on training data but fail on novel test data more readily than several simpler co-adaptations that achieve the same result.

## 3. Related Work

*QX2 connects hidden-unit noise, model averaging and earlier work on corruption, adversarial deletion and marginalization.*

Noise in hidden-unit states gives a regularization interpretation of QX2. **S2H, or Denoising Autoencoders (DAEs)**, introduced by Vincent et al. (2008, 2010), corrupt inputs and reconstruct clean inputs. QX2 extends this approach with effective hidden-layer corruption and a model-averaging interpretation. It applies to supervised and unsupervised learning and to other neuron-based architectures, including Boltzmann Machines.

Test-time T6R permits much higher noise than usually works best for S2H.

| Setting | Noise level reported |
|---|---|
| S2H, typical best | 5% |
| QX2, often optimal input-unit removal | 20% |
| QX2, often optimal hidden-unit removal | 50% |

Stochastic minimization under a noise distribution minimizes expected loss. In contrast, Globerson and Roweis (2006) and Dekel et al. (2010) let an adversary choose removals, subject to a fixed maximum number rather than a probability distribution. Those models have no hidden units.

Analytically marginalizing QX2—taking the expectation over its noise—gives deterministic regularizers in simple cases. Van der Maaten et al. (2013) studied exponential-family input-noise regularizers, including QX2 under the name **blankout noise**. Their work was limited to models without hidden layers. Chen et al. (2012) studied marginalization for denoising autoencoders. Wang and Manning (2013) proposed marginalization to accelerate QX2.

**Claims and evidence**
- QX2 permits higher reported noise levels than typical S2H. The unnumbered noise-level table lists S2H at 5%, QX2 input-unit removal at 20% and QX2 hidden-unit removal at 50%.

## 4. Model Description

*The model independently gates unit outputs during training and scales weights by retention probability at inference.*

For every hidden unit, expected output under training-time deletion equals actual test-time output under the prescribed reduced weights. In the simplest construction, retention choices are mutually independent and use one fixed probability. Validation can select this probability. The usual default is close to optimal across many tasks and networks. Optimal input retention is generally nearer complete retention.

Let:

- \(L\) be the number of hidden layers.
- \(l\in\{1,\ldots,L\}\) index those layers.
- \(\mathbf z^{(l)}\) and \(\mathbf y^{(l)}\) be layer inputs and outputs.
- \(\mathbf y^{(0)}=\mathbf x\) be the network input.
- \(\mathbf W^{(l)}\) and \(\mathbf b^{(l)}\) be layer weights and biases.

For \(l\in\{0,\ldots,L-1\}\) and hidden unit \(i\), ordinary feed-forward computation is

\[
z_i^{(l+1)}
 =\mathbf w_i^{(l+1)}\mathbf y^l+b_i^{(l+1)},
\qquad
y_i^{(l+1)}=f\!\left(z_i^{(l+1)}\right).
\]

Here \(\mathbf w_i^{(l+1)}\) is the unit’s weight vector, \(b_i^{(l+1)}\) its bias, and \(f\) its activation function. Any activation is admissible, including the logistic function

\[
f(x)=\frac{1}{1+\exp(-x)}.
\]

QX2 instead computes

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

The symbol \(*\) means element-wise multiplication. Each entry of the retention vector \(\mathbf r^{(l)}\) is an independent Bernoulli variable: it equals one with probability \(p\), the retention probability. Multiplying this vector by layer outputs gives the thinned outputs \(\widetilde{\mathbf y}^{(l)}\) passed to the next layer.

Applying this operation at every layer samples a subnetwork. Loss derivatives backpropagate through that subnetwork. Both hidden and visible units can be removed.

At inference, no units are removed. Test weights are

\[
\mathbf W_{\mathrm{test}}^{(l)}=p\mathbf W^{(l)}.
\]

**Figure 2.** Training, left: a unit is present with probability \(p\), its connections to the next layer having weights \(w\). Testing, right: the unit is always present, its outgoing weights \(pw\).

**Figure 3.** Standard computation, left: \(y_1^{(l)},y_2^{(l)},y_3^{(l)}\), weights \(\mathbf w_i^{(l+1)}\), bias \(b_i^{(l+1)}\) from a \(+1\) unit, preactivation \(z_i^{(l+1)}\), activation \(f\), output \(y_i^{(l+1)}\). QX2 computation, right: corresponding gates \(r_1^{(l)},r_2^{(l)},r_3^{(l)}\) first produce \(\widetilde y_1^{(l)},\widetilde y_2^{(l)},\widetilde y_3^{(l)}\); the remaining operations retain their form.

## 5. Learning QX2 Nets

*QX2 uses backpropagation through sampled subnetworks and can finetune pretrained networks.*

### 5.1 Backpropagation

*Training samples a network per case and benefits from high momentum, decaying learning rates and a weight-norm bound.*

QX2 alone gives significant gains. Combining it with high momentum, large decaying learning rates and **V9C, or max-norm regularization**, gives a further significant boost. V9C constrains each hidden unit’s incoming-weight-vector norm. It usually improves deep-net stochastic-gradient training even without QX2 and was previously used for collaborative filtering (Srebro and Shraibman, 2005).

The proposed optimization account is as follows. A fixed-radius weight ball permits huge learning rates without weight explosion. QX2 noise explores otherwise hard-to-reach regions of weight space. Decaying learning rates then give progressively shorter, less exploratory steps that end at a minimum.

Stochastic gradient descent differs from standard-net training only through the sampled thinned network. Each mini-batch case receives forward and backward propagation through its own sampled network. Parameter gradients are averaged across cases. A case that does not use a parameter contributes zero to its gradient. Momentum, annealed learning rates and Y5F remain useful.

For an incoming hidden-unit weight vector \(\mathbf w\), optimization imposes

\[
\|\mathbf w\|_2\leq c.
\]

Here \(\|\mathbf w\|_2\) is its norm and \(c\) is the maximum allowed norm. If \(\mathbf w\) leaves the radius-\(c\) ball, projection returns it to the surface. Validation tunes \(c\).

**Claims and evidence**
- QX2 and V9C improve the regularizer comparison. Table 9, rows “V9C,” “QX2 + Y5F” and “QX2 + V9C”: 1.35, 1.25 and 1.05.
- High learning rates and momentum are reported to accelerate learning. Appendix A.2’s practical-values table gives typically 10–100 times the optimal standard-net learning rate and momentum around 0.95 to 0.99; no controlled numerical comparison is supplied here.
- The weight-space account is proposed, not established by a numerical result in this section.

### 5.2 Unsupervised Pretraining

*QX2 finetuning preserves pretrained information when learning rates are smaller and weights are first rescaled.*

Smaller finetuning learning rates preserve useful pretrained information. They improve final generalization compared with finetuning without QX2. In the initial experiments, rates comparable to the best rates for randomly initialized nets erased the pretrained information.

Pretraining itself is unchanged. Before QX2 finetuning, pretrained weights are multiplied by \(1/p\). This matches each unit’s expected output under random removal to its pretraining output.

Using unlabeled data through pretraining can significantly improve on random initialization in some cases. The QX2 finetuning procedure applies after pretraining with stacked U2L (Hinton and Salakhutdinov, 2006), autoencoders (Vincent et al., 2010), or **E5Q, or Deep Boltzmann Machines (DBM)** (Salakhutdinov and Hinton, 2009). Finetuning then uses backpropagation.

**Claims and evidence**
- QX2 improves pretrained H4Z nets. Table 2, G6W finetuning rows: 1.18 without QX2 and 0.92 with QX2; E5Q finetuning rows: 0.96 and 0.79.
- Smaller finetuning rates matter. Appendix B.4 reports about 0.01 as the best QX2 G6W-finetuning rate, compared with the listed high learning rate of 0.1; this section gives no numerical result for erasing pretrained information.

## 6. Experimental Results

*Experiments compare QX2 with other methods across vision, speech, text and genetics.*

QX2 improved generalization over neural nets without QX2 on every tested data set. The applications include object classification, digit recognition, speech, documents and computational biology.

The collections are:

- **H4Z, or MNIST:** standard handwritten-digit toy data.
- **N8Q, or TIMIT:** clean-speech benchmarking data.
- **J3V, or CIFAR-10**, and **W6R, or CIFAR-100:** tiny natural-image collections described by Krizhevsky (2009).
- **A2F, or Street View House Numbers (SVHN):** Google Street View house-number images supplied by Netzer et al. (2011).
- **K9D, or ImageNet:** natural images at large scale.
- **M5X, or Reuters-RCV1:** Reuters newswire documents.
- **Z7P, or the Alternative Splicing data set:** RNA-feature-based gene-splicing prediction data from Xiong et al. (2011).

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

Here K and M are the size suffixes used in the table; “dim” labels the frame dimensionality.

### 6.1 Image Data

*QX2 is used in models reported to attain state of the art on all five vision collections.*

The five vision collections differ in image type and training-set size. Models attaining state of the art on all five use QX2.

#### 6.1.1 H4Z

*QX2 improves permutation-invariant H4Z results, including large and pretrained networks.*

QX2 reduces error relative to the best permutation-invariant neural nets without QX2 or unsupervised pretraining. This comparison does not use two-dimensional spatial information. Replacing logistic units with **F3N, or rectified linear units (ReLUs)**, reduces error further (Jarrett et al., 2009). V9C adds another reduction. Larger networks improve results. **L8J, or maxout**, provides a further improvement (Goodfellow et al., 2013).

Standard regularization and early stopping make the largest reported network difficult to generalize. QX2 prevents its overfitting without requiring early stopping.

Pretrained nets also improve. **G6W, or DBN**, is the table designation for initialization using stacked U2L, distinct from E5Q initialization. The E5Q result with QX2 is the best reported permutation-invariant performance.

Better results remain possible using two-dimensional spatial information and distorted training-image augmentation. Those approaches are tested here on more interesting collections.

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

Here NN denotes neural net. SVM is the Gaussian-kernel comparison method. NA marks entries for which no unit type or architecture is supplied.

When every hyperparameter, including retention probability, is held fixed across architectures, test-error trajectories form two sharply separated clusters. They correspond to otherwise identical architectures trained with and without QX2. The improvement is large without architecture-specific hyperparameter tuning.

**Figure 4.** Test error during training for different architectures, with and without QX2; curves labeled “Without QX2” and “With QX2.”

| Axis | Label | Displayed ticks |
|---|---|---|
| Horizontal | Number of weight updates | 0, 200000, 400000, 600000, 800000, 1000000 |
| Vertical | Classification Error % | 1.0, 1.5, 2.0, 2.5 |

**Claims and evidence**
- QX2, F3N and V9C successively reduce the listed error. Table 2, standard logistic net: 1.60; QX2 logistic net: 1.35; QX2 F3N net: 1.25; QX2 F3N + V9C, 3 layers and 1024 units: 1.06.
- Larger QX2 + V9C networks improve results. Table 2, F3N rows: 1.06, 1.04, 1.01 and 0.95 for the listed 1024-, 2048-, 4096- and 8192-unit architectures.
- L8J improves the listed unpretrained result. Table 2, L8J row: 0.94.
- QX2 improves both pretrained comparisons. Table 2, G6W rows: 1.18 and 0.92; E5Q rows: 0.96 and 0.79. The E5Q + QX2 row is the reported best permutation-invariant result.
- Improvement persists without architecture-specific tuning, and the large net does not require early stopping. Figure 4 shows the separated trajectories through the displayed 1000000 updates; it supplies no numerical curve values.

#### 6.1.2 A2F

*QX2 improves convolutional networks first in fully connected layers and then in all layers.*

Convolutional nets (LeCun et al., 1989) outperform other methods on this task. Adding QX2 to fully connected layers improves performance. Adding it to convolutional layers improves performance further. L8J permits another gain.

The small number of convolutional parameters might suggest little overfitting and little benefit from QX2 in those layers. Nevertheless, lower-layer corruption supplies noisy inputs that prevent the higher fully connected layers from overfitting.

The comparison names WDCH, HOG and KMeans identify the listed A2F baselines. Pooling variants include average, L2, L4, max and stochastic pooling; the paper compares them as named below.

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

The used portion contains color images approximately centered on one house-number digit. The task is to identify that digit.

**Claims and evidence**
- The listed convolutional methods outperform the listed non-convolutional methods. Table 3: the latter range from 36.7 to 9.4; the convolutional rows range from 9.06 to 2.47.
- Extending QX2 improves the max-pooling net. Table 3, corresponding rows: 3.95 without QX2, 3.02 with QX2 in fully connected layers and 2.55 with QX2 in all layers.
- L8J gives a further gain. Table 3, “Conv Net + L8J”: 2.47.
- The noisy-input explanation for the convolutional-layer benefit is stated in the text; no separate numerical test of that explanation is supplied.

#### 6.1.3 J3V and W6R

*QX2 reduces natural-image error without augmentation beyond input dropout.*

Without data augmentation, Bayesian hyperparameter optimization gives the J3V result of Snoek et al. (2012). QX2 in fully connected layers improves it. Extending QX2 to every layer improves it further. L8J further reduces J3V error (Goodfellow et al., 2013). W6R also shows a large QX2 improvement.

Neither collection uses augmentation apart from input QX2.

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

**Claims and evidence**
- QX2 successively improves J3V. Table 4, Snoek baseline, fully connected QX2 and all-layer QX2 rows: 14.98, 14.32 and 12.61.
- L8J further improves J3V. Table 4, L8J row: 11.68.
- QX2 substantially improves W6R. Table 4, hand-tuned baseline, fully connected QX2 and all-layer QX2 rows: 43.48, 41.26 and 37.20.
- The no-augmentation condition, except input QX2, is stated in the text rather than measured in a table.

#### 6.1.4 K9D

*Convolutional networks with QX2 outperform the listed standard image-feature methods and win ILSVRC-2012.*

The convolutional-net/QX2 model won ILSVRC-2012. Its advantage over standard vision-feature methods is staggering. Its predictions remain very reasonable even when its best guess is wrong. On ILSVRC-2010, convolutional nets with QX2 outperform the alternatives by a large margin. Krizhevsky et al. (2012) describe the architecture and implementation.

The tables use the comparison feature names SIFT, Fisher Vectors (FVs), LBP, GIST and CSIFT. “Avg” denotes averaging classifiers. “val” denotes validation results.

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

K9D contains over 15 million labeled high-resolution images in roughly 22,000 categories. The annual ILSVRC competition began in 2010 as part of the Pascal Visual Object Challenge. It uses a subset with roughly 1000 images in each of 1000 categories.

Reporting includes top-1, or first-choice, error and top-5 error. Top-5 error counts test images whose correct label is absent from the five labels assigned the highest probabilities.

Only ILSVRC-2010 has available test labels, so it supplies most experiments. ILSVRC-2012 test labels are unavailable. Its reported test result is therefore restricted to the final submission; model variations are evaluated on validation data.

**Figure 6.** K9D test cases with the four most probable predicted labels; horizontal bar length is proportional to assigned probability, pink marks ground truth.

**Claims and evidence**
- QX2 convolutional nets outperform the listed ILSVRC-2010 alternatives. Table 5: QX2 gives 37.5 top-1 and 17.0 top-5, versus 47.1/28.2 for Sparse Coding and 45.7/25.7 for SIFT + Fisher Vectors.
- The reported ILSVRC-2012 submission has a large advantage. Table 6, averaged 5-net QX2 row: 16.4 top-5 test error, versus 27.3 and 26.2 in the two feature-method rows. The broad-comparison table states about 16% versus about 26%.
- Averaging improves the listed validation results. Table 6: one QX2 net gives 40.7/18.2; averaging 5 gives 38.1/16.4.
- Wrong-best-guess predictions can remain reasonable. Figure 6 shows four predicted labels per case; this is qualitative evidence.
- The competition win is stated in the text. The test-label limitation is also stated: only the final ILSVRC-2012 submission has a reported test result.

**Claims and evidence (§6.1)**
- The paper reports state-of-the-art QX2 models across all five vision collections. The reported results include Table 2’s 0.79 H4Z error; Table 3’s 2.47 A2F error; Table 4’s 11.68 J3V and 37.20 W6R errors; and Table 6’s 16.4 K9D top-5 test error.

### 6.2 N8Q

*QX2 improves unpretrained and pretrained speech networks, with both reported QX2-pretrained depths reaching 19.7%.*

QX2 improves the unpretrained network and both pretrained comparisons. The two QX2-pretrained depths attain the same phone error rate, the speech-recognition error measure used here.

The prose comparison for the deeper pretrained network uses a pre-QX2 baseline different from the corresponding table baseline. The designation mcRBM-G6W identifies another listed pretrained comparison.

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

**Claims and evidence**
- QX2 improves the unpretrained net. Table 7, 6-layer NN and QX2 NN rows: 23.4 and 21.8.
- QX2 improves both pretrained depths. Table 7, 4-layer rows: 22.7 and 19.7; 8-layer rows: 20.7 and 19.7.
- Both QX2-pretrained depths reach 19.7. Table 7, final two rows.
- The deeper baseline is inconsistent between prose and table. The unnumbered prose-comparison table gives 20.5%; Table 7’s G6W-pretrained 8-layer row gives 20.7.

### 6.3 Text Data

*QX2 improves bag-of-words document classification, but by less than in vision and speech.*

A bag-of-words document classifier on a subset of M5X improves with QX2. The gain is smaller than in vision and speech classification. Every tried architecture improved. A possible explanation is that the large collection makes overfitting less serious.

| Best neural-net document classifier | Error rate |
|---|---|
| Without QX2 | 31.05% |
| With QX2 | 29.62% |

**Claims and evidence**
- The best classifier improves with QX2. The unnumbered table’s “Without QX2” and “With QX2” rows give 31.05% and 29.62%.
- Every tried architecture improved, and the gain is described as smaller than for vision and speech. No architecture-by-architecture text results are supplied.
- Collection size is offered only as a possible account of the smaller gain; it is not separately tested here.

### 6.4 Bayesian Neural Networks

*On the small Z7P data set, QX2 beats the non-Bayesian comparisons but remains below Bayesian neural networks.*

Bayesian neural nets outperform QX2 on Z7P. QX2 nevertheless substantially improves standard neural nets and exceeds every other method.

Input reduction by **PCA**, the reduction method used in the comparisons, followed by SVMs or logistic regression offers one defense against small-training-set overfitting. QX2 prevents overfitting without reducing input dimensionality. It uses vastly more hidden units than the Bayesian network and exhibits strong regularization.

| Method | C4Y (bits) |
|---|---|
| Neural Network (early stopping) (Xiong et al., 2011) | 440 |
| Regression, PCA (Xiong et al., 2011) | 463 |
| SVM, PCA (Xiong et al., 2011) | 487 |
| Neural Network with QX2 | 567 |
| Bayesian Neural Network (Xiong et al., 2011) | 623 |

*Table 8. Z7P results.*

Bayesian neural networks use posterior weighting over network structures and parameters (Neal, 1996). This is the proper averaging procedure: it incorporates both prior information and data fit. QX2 instead gives equal weights to exponentially many models sharing weights.

Bayesian nets are extremely useful when data are scarce, including medical diagnosis, genetics, drug discovery and other computational-biology applications. But they train slowly, scale poorly to very large networks and require expensive predictions from many large nets. QX2 nets train and infer much faster.

Z7P is a small genetics collection on which Bayesian nets are known to perform well and attain state of the art (Xiong et al., 2011). **Alternative splicing** significantly contributes to cellular diversity in mammalian tissues. Predicting when it occurs in particular tissues and conditions matters for understanding many human diseases.

RNA features predict probabilities of three biologically important splicing-related events. The score is **C4Y, or Code Quality**, a distributional measure representing negative KL divergence between target and predicted distributions. Higher values are preferred. Table 8 reports the score in bits.

**Claims and evidence**
- Bayesian nets perform best; QX2 exceeds all other listed methods. Table 8: Bayesian Neural Network 623, Neural Network with QX2 567, SVM/PCA 487, Regression/PCA 463 and early-stopped Neural Network 440.
- QX2 uses many more hidden units without input reduction. Appendix B.6’s settings table gives QX2 scale “1000s,” Bayesian scale “a few tens,” and a trained QX2 network with two layers of 1024 units each.
- Bayesian training and prediction are described as slower and less scalable. No timing comparison is supplied.

### 6.5 Standard Regularizers

*QX2 combined with V9C gives the lowest error in the reported H4Z regularizer comparison.*

The compared regularizers include Y5F, the more general **Tikhonov regularization** (Tikhonov, 1943), **lasso**, the named L1 regularizer (Tibshirani, 1996), **KL-sparsity**, the compared sparsity regularizer, and V9C.

Validation selected decay constants, target sparsity, QX2 rate and the V9C bound. All methods used the same F3N architecture trained by stochastic gradient descent. QX2 with V9C gives the lowest generalization error.

| Method | Test Classification error % |
|---|---|
| Y5F | 1.62 |
| Y5F + O3U applied towards the end of training | 1.60 |
| Y5F + KL-sparsity | 1.55 |
| V9C | 1.35 |
| QX2 + Y5F | 1.25 |
| QX2 + V9C | 1.05 |

*Table 9. H4Z regularization comparison.*

**Claims and evidence**
- QX2 + V9C gives the lowest listed error. Table 9, “QX2 + V9C”: 1.05; the other rows give 1.62, 1.60, 1.55, 1.35 and 1.25.
- Adding QX2 improves both listed weight-regularization comparisons. Table 9: Y5F 1.62 versus QX2 + Y5F 1.25; V9C 1.35 versus QX2 + V9C 1.05.

**Claims and evidence (§6)**
- QX2 improves the reported neural-net comparisons across the tested domains. Evidence includes Table 2, 1.60 versus 1.35; Table 3, 3.95 versus 3.02 and 2.55; Table 4, 15.60/43.48 versus 12.61/37.20; Table 7, 23.4 versus 21.8; §6.3’s unnumbered table, 31.05% versus 29.62%; and Table 8, 440 versus 567.
- The experiments span vision, speech, text and genetics. Table 1 lists all collections, domains, dimensions and split sizes. Tables 5–6 report the large-scale image results.

## 7. Salient Features

*The following comparisons examine learned features, sparsity, retention, data size and prediction averaging.*

### 7.1 Features

*QX2 autoencoders learn localized visual features despite reconstruction errors similar to those without QX2.*

First-layer visual features differ markedly even when test reconstruction errors are similar. Without QX2, individual hidden units do not appear to detect meaningful features. With QX2, units detect edges, strokes and spots in different image regions. Breaking co-adaptations is proposed as the main source of lower generalization error.

An ordinary parameter derivative prescribes how to reduce loss conditional on every other unit’s behavior. This lets units repair one another’s mistakes and form complex co-adaptations that do not generalize. Under the QX2 hypothesis, particular partners are unreliable. Units cannot depend on specific corrective units and must work in the many contexts supplied by the remaining units.

**Figure 7.** H4Z features from identical single-hidden-layer autoencoders with 256 F3N units: (a) no QX2, (b) hidden-layer QX2 with \(p=0.5\). The former features are co-adapted for reconstruction; the latter have the localized structures described above.

**Claims and evidence**
- QX2 features are localized edges, strokes and spots, unlike those without QX2. Figure 7 compares identical 256-unit F3N autoencoders, with \(p=0.5\) for QX2.
- Reconstruction errors are described as similar; exact errors are not supplied.
- Reduced co-adaptation is proposed as the explanation, not established by a separate numerical test.

### 7.2 Sparsity

*QX2 produces sparse hidden activations without a sparsity-inducing regularizer.*

The autoencoder comparison uses a random test-set mini-batch. Good sparsity means few strongly active units per case and low average activity of each unit across cases.

For each model, the left histogram shows the distribution of hidden-unit mean activations over the mini-batch. The right histogram shows the activations themselves.

| Observation | No QX2 | QX2 |
|---|---|---|
| Most units’ mean activation; overall hidden-unit mean | about / close to 2.0 | about / around 0.7 |
| Activation histogram | Huge mode away from zero; substantial off-zero mass | Sharp zero peak |
| Highly active units | Large fraction | Very few |

**Figure 8.** Sparsity histograms for the F3N autoencoders: (a) without QX2, (b) QX2 with \(p=0.5\).

**Claims and evidence**
- QX2 produces lower mean activity and fewer highly active units. The unnumbered observation table gives about/close to 2.0 without QX2 and about/around 0.7 with QX2; its other rows contrast substantial off-zero mass with a sharp zero peak.
- The effect occurs without a sparsity-inducing regularizer. Figure 8 compares the F3N autoencoders with QX2 at \(p=0.5\); no additional numerical sparsity score is supplied.

### 7.3 Rate

*Low retention can underfit, while keeping expected retained width fixed reduces low-retention error.*

With architecture fixed, low retention gives few active units, high training error and high test error. As retention increases, error falls, flattens over an intermediate interval and rises again near complete retention.

Holding expected retained width fixed substantially reduces low-retention errors. Here \(n\) is a layer’s hidden-unit count and \(pn\) its expected retained count. The inference networks then differ in size: lower-retention networks are larger.

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

**Claims and evidence**
- Fixed-architecture error is flat over an intermediate interval. The unnumbered comparison table gives \(0.4\leq p\leq0.8\); Figure 9(a) shows the reported training/test pattern.
- Fixed expected width improves the low-retention comparison. The table gives 2.7% versus 1.7% at \(p=0.1\).
- Retention close to 0.6 appears best for the selected expected widths, and 0.5 is close to optimal. These are the final two rows of the comparison table.
- Lower-retention inference networks are larger under fixed \(pn\). This follows the comparison construction stated here and in Appendix B.1; no separate performance table is supplied.

### 7.4 Data-Set Size

*The QX2 gain is absent for the smallest sets, grows with more data and then declines.*

The two smallest tested training sets show no QX2 improvement. The network can memorize them despite removal noise. Increasing data first increases the gain and then decreases it.

For a fixed architecture and QX2 rate, the proposed “sweet spot” has enough data to resist noisy memorization but not enough to make overfitting unimportant. Standard training of huge nets on small collections massively overfits. Generalization from many parameters with little data therefore tests a regularizer.

**Figure 10.** H4Z error versus randomly selected training-set size, curves “With QX2” and “Without QX2.”

| Axis | Label | Displayed ticks |
|---|---|---|
| Horizontal | Dataset size | \(10^2,10^3,10^4,10^5\) |
| Vertical | Classification Error % | 0, 5, 10, 15, 20, 25, 30 |

**Claims and evidence**
- The two smallest sets show no improvement, followed by a gain that increases and then decreases. Figure 10 shows the curves. Appendix B.1’s data-size row lists 100, 500, 1K, 5K, 10K and 50K; no exact curve errors are supplied.
- The memorization and “sweet spot” account is proposed in the text; no separate numerical test is reported.

### 7.5 X9M vs. T6R

*Sampled averaging approaches weight scaling and becomes slightly better after enough samples.*

With enough samples, X9M matches T6R and then becomes slightly better, within one standard deviation. T6R is therefore a fairly good approximation to the true model average.

The H4Z comparison varies the sample count and uses a horizontal T6R reference. It measures how the finite-sample average approaches true-average error. Here \(k\) is the number of sampled networks.

| Quantity | Value |
|---|---|
| Sample count at which X9M becomes as good as T6R | around \(k=50\) |

**Figure 11.** X9M versus T6R.

| Axis | Label | Displayed ticks |
|---|---|---|
| Horizontal | Number of samples used for X9M \((k)\) | 0, 20, 40, 60, 80, 100, 120 |
| Vertical | Test Classification error % | 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35 |

Curves: “X9M” and “Approximate averaging by T6R.”

**Claims and evidence**
- X9M becomes as good as T6R at around \(k=50\). The unnumbered quantity table gives this count.
- Further improvement is slight and within one standard deviation. Figure 11 supplies the comparison; exact curve values and the standard deviation are not given.
- T6R is described as a fairly good approximation on this evidence.

## 8. B8T

*B8T extends dropout to shared-weight Restricted Boltzmann Machines.*

### 8.1 Model Description

*B8T mixes RBMs defined by different hidden-unit retention vectors.*

B8T is a mixture of exponentially many shared-weight U2L models. Each uses a different hidden-unit subset. Conditional on a retention vector, its visible–hidden distribution is an ordinary U2L distribution with the designated hidden units removed.

Let \(\mathbf v\in\{0,1\}^D\) be the visible vector with \(D\) entries, and \(\mathbf h\in\{0,1\}^F\) the hidden vector with \(F\) entries. Let \(\theta=\{\mathbf W,\mathbf a,\mathbf b\}\) be the weights and bias parameters, and \(Z\) the partition function. An ordinary U2L defines

\[
P(\mathbf h,\mathbf v;\theta)
=\frac{1}{Z(\theta)}
 \exp\!\left(
 \mathbf v^\top\mathbf W\mathbf h+
 \mathbf a^\top\mathbf h+
 \mathbf b^\top\mathbf v
 \right).
\]

Here \(P\) denotes probability, and \(\top\) denotes transpose.

B8T adds a retention vector \(\mathbf r\in\{0,1\}^F\). It independently retains \(h_j\) when \(r_j=1\), with probability \(p\), and otherwise drops it:

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

The normalizer is \(Z'(\theta,\mathbf r)\). The indicator \(\mathbf1(\cdot)\) is used in \(g\), which enforces \(h_j=0\) when \(r_j=0\). Products and sums below run over the displayed unit indices. The conditional hidden distribution factorizes:

\[
P(\mathbf h\mid\mathbf r,\mathbf v)
=\prod_{j=1}^{F}P(h_j\mid r_j,\mathbf v),
\]

\[
P(h_j=1\mid r_j,\mathbf v)
=\mathbf1(r_j=1)\,
 \sigma\!\left(b_j+\sum_iW_{ij}v_i\right).
\]

Here \(\sigma(\cdot)\) is the activation in the conditional probability, and \(W_{ij}\), \(b_j\) and \(a_i\) below are the weight and bias entries appearing in these equations.

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

*RBM learning algorithms train only the hidden units retained for each case.*

Contrastive Divergence and other U2L learning algorithms apply directly (Hinton et al., 2006). First sample \(\mathbf r\); then train only retained hidden units. Every training case in every mini-batch receives a different \(\mathbf r\). Experiments use **CD-1**, the Contrastive Divergence procedure named here.

### 8.3 Features

*B8T features are coarser and have fewer dead units than ordinary RBM features.*

This comparison tests whether the feature-quality effect of feed-forward QX2 transfers to B8T. B8T features appear coarser than U2L’s sharply defined, stroke-like features. B8T also has very few dead units relative to U2L.

**Figure 12.** H4Z features from 256-hidden-unit models, ordered by \(L_2\) norm: (a) ordinary U2L, (b) B8T with \(p=0.5\).

Here \(L_2\) is the norm used to order the features.

**Claims and evidence**
- B8T features are coarser and have fewer dead units. Figure 12 compares 256-hidden-unit models, with \(p=0.5\) for B8T. No numerical feature-quality or dead-unit count is supplied.

### 8.4 Sparsity

*B8T learns sparser representations without an extra sparsity regularizer.*

Without an additional sparsity-inducing regularizer, B8T learns much sparser representations than U2L. Histograms show hidden activations and their means over a test mini-batch.

**Figure 13.** (a) Without QX2: many activations away from zero. (b) QX2 with \(p=0.5\): many activations near zero and very few high activations.

**Claims and evidence**
- B8T is much sparser without an additional sparsity regularizer. Figure 13 contrasts many off-zero activations without QX2 against many near-zero and very few high activations with \(p=0.5\). No numerical sparsity score is supplied.

## 9. Marginalizing QX2

*Taking expectations over dropout noise can produce deterministic objectives and regularizers.*

Deterministic noise-marginalized models need no random bits. They permit gradients of marginalized losses, unlike standard Monte-Carlo QX2.

Globerson and Roweis (2006) studied deterministic robustness to test-time feature deletion. Srivastava (2013) discussed QX2 marginalization for linear regression. Wager et al. (2013) described QX2 as an adaptive regularizer. The earlier denoising-autoencoder, acceleration and input-noise regularizer studies retain the scopes stated in §3.

### 9.1 Linear Regression

*Expected dropout squared error equals a regression objective with a data-dependent quadratic penalty.*

In expectation, QX2 linear regression is ridge regression with a particular penalty matrix. A dimension’s standard deviation scales its weight cost. Weights on highly variable dimensions are squeezed more strongly. Absorbing retention into the weights reveals a regularization constant that is small near complete retention and grows with removal.

Let \(\mathbf X\in\mathbb R^{N\times D}\) contain \(N\) data points with \(D\) dimensions. Let \(\mathbf y\in\mathbb R^N\) contain targets and \(\mathbf w\in\mathbb R^D\) contain weights. These are real-valued arrays. Ordinary linear regression minimizes squared error:

\[
\|\mathbf y-\mathbf X\mathbf w\|^2.
\]

Retaining each input dimension with probability \(p\) gives \(\mathbf R*\mathbf X\). Here \(\mathbf R\in\{0,1\}^{N\times D}\) is the retention matrix, \(R_{ij}\sim\operatorname{Bernoulli}(p)\), and \(*\) is element-wise multiplication. The marginalized objective is

\[
\underset{\mathbf w}{\operatorname{minimize}}\quad
\mathbb E_{\mathbf R\sim\operatorname{Bernoulli}(p)}
\left[\|\mathbf y-(\mathbf R*\mathbf X)\mathbf w\|^2\right],
\]

where \(\mathbb E\) takes the expectation over retention noise. This reduces to

\[
\underset{\mathbf w}{\operatorname{minimize}}\quad
\|\mathbf y-p\mathbf X\mathbf w\|^2+
p(1-p)\|\mathbf\Gamma\mathbf w\|^2,
\]

\[
\mathbf\Gamma
=\left(\operatorname{diag}(\mathbf X^\top\mathbf X)\right)^{1/2}.
\]

The penalty matrix \(\mathbf\Gamma\) is defined by the displayed diagonal operation. With rescaled weights \(\widetilde{\mathbf w}=p\mathbf w\), an alternative form is

\[
\underset{\mathbf w}{\operatorname{minimize}}\quad
\|\mathbf y-\mathbf X\widetilde{\mathbf w}\|^2+
\frac{1-p}{p}\|\mathbf\Gamma\widetilde{\mathbf w}\|^2.
\]

**Claims and evidence**
- The expected dropout objective has the stated quadratic penalty. The displayed reduction gives \(p(1-p)\|\mathbf\Gamma\mathbf w\|^2\), with \(\mathbf\Gamma=(\operatorname{diag}(\mathbf X^\top\mathbf X))^{1/2}\).
- Rescaling exposes the retention-dependent coefficient. The final objective gives \((1-p)/p\). This is an analytical result; no numerical table is reported.

### 9.2 Logistic Regression and Deep Networks

*Approximate marginalization helps logistic regression, but the stated assumptions do not directly support deep networks.*

Wang and Manning (2013) report that approximate marginalization outperforms Monte-Carlo QX2 in both training time and generalization for logistic regression.

Closed-form marginalized models are difficult for logistic regression and deep nets. Under reasonable assumptions, logistic-unit input distributions and marginalized-model gradients are Gaussian, with efficiently computable means and variances. The assumptions become successively weaker as layers are added. The results therefore do not apply directly to deep networks.

**Claims and evidence**
- Approximate marginalization improves logistic-regression time and generalization. The section attributes this to Wang and Manning (2013), but supplies no numbers or table.
- The deep-network limitation is explicit: the assumptions weaken with additional layers, so direct applicability is not claimed.

## 10. Multiplicative Gaussian Noise

*Preliminary comparisons find Gaussian multiplicative noise as effective as, or slightly better than, binary dropout.*

Preliminary experiments suggest that Gaussian multiplicative corruption works as well as, or perhaps better than, binary corruption. The binary variant is **P7V, or Bernoulli dropout**. The Gaussian variant is **D4K, or Gaussian dropout**.

The two variants can match first and second moments through their means and variances, despite opposite entropy extremes. For the given moments, the Gaussian multiplier has the highest entropy and the Bernoulli multiplier the lowest. Both work well. Preliminary evidence slightly favors the high-entropy case.

| Data set | Architecture | P7V | D4K |
|---|---|---|---|
| H4Z | 2 layers, 1024 units each | \(1.08\pm0.04\) | \(0.95\pm0.04\) |
| J3V | 3 conv + 2 fully connected layers | \(12.6\pm0.1\) | \(12.5\pm0.1\) |

*Table 10. Classification error %.*

Here “conv” denotes convolutional layers; \(\pm\) preserves the reported uncertainty notation.

| Comparison setting | Value |
|---|---|
| H4Z P7V retention, hidden units | \(p=0.5\) |
| H4Z P7V retention, input units | \(p=0.8\) |
| J3V P7V retention, input to top | \(p=(0.9,0.75,0.75,0.5,0.5,0.5)\) |
| D4K standard deviation, each corresponding layer | \(\sigma=\sqrt{(1-p)/p}\) |
| Averaging | 10 different random seeds |

Bernoulli corruption multiplies hidden activations by variables equal to one with probability \(p\) and zero otherwise. Other multiplier distributions are possible.

For hidden activation \(h_i\), drawing a multiplier from \(\mathcal N(1,1)\) is equivalent to adding zero-mean Gaussian noise with standard deviation equal to that activation:

\[
h_i\longmapsto h_i+h_ir,\qquad r\sim\mathcal N(0,1),
\]

or

\[
h_i\longmapsto h_ir',\qquad r'\sim\mathcal N(1,1).
\]

Here \(r\) and \(r'\) are the Gaussian noise variables. Extending this to \(r'\sim\mathcal N(1,\sigma^2)\) makes \(\sigma\), the noise standard deviation, a tunable hyperparameter analogous to \(p\). Expected activations are unchanged, so test-time T6R is unnecessary.

There are two equivalent representations, with appropriately rescaled layerwise learning rates and weight initializations:

- Keep retained training activations unchanged and multiply outgoing test weights by \(p\).
- Multiply retained training activations by \(1/p\) and leave test weights unchanged.

In the second representation, the Bernoulli multiplier \(r_b\) equals \(1/p\) with probability \(p\) and zero otherwise. Its expectation and variance are

\[
\mathbb E[r_b]=1,\qquad
\operatorname{Var}[r_b]=\frac{1-p}{p}.
\]

Choosing \(\sigma^2=(1-p)/p\) for the Gaussian multiplier \(r_g\) gives

\[
\mathbb E[r_g]=1,\qquad
\operatorname{Var}[r_g]=\frac{1-p}{p}.
\]

**Claims and evidence**
- Gaussian corruption performs as well as, or perhaps better than, Bernoulli corruption in preliminary experiments. Table 10, H4Z row: \(1.08\pm0.04\) versus \(0.95\pm0.04\); J3V row: \(12.6\pm0.1\) versus \(12.5\pm0.1\).
- The reported comparison averages 10 random seeds. The unnumbered settings table gives this count and the exact retention settings.
- Moments match under the stated scaling. The equations give mean 1 and variance \((1-p)/p\) for both \(r_b\) and \(r_g\).
- The entropy comparison and removal of test-time scaling are stated analytically; no separate empirical table tests those properties.

## 11. Conclusion

*QX2 improves generalization by training sampled submodels, with a training-time cost that motivates acceleration.*

The method repeatedly samples and trains smaller submodels from a large model that easily overfits. U2L fits this construction, and B8T shows desirable empirical properties. QX2 methods attain state of the art on A2F, K9D, W6R and H4Z. They also considerably improve standard neural nets on other collections.

Training typically takes 2–3 times as long as training a standard network with the same architecture. Noisy parameter updates are a major cause. Each case trains a different random architecture, so computed gradients are not gradients of the final inference architecture.

The stochasticity likely prevents overfitting. This leaves a trade-off between overfitting and training time: more training time permits stronger QX2 and less overfitting.

Noise marginalization obtains some benefits without stochasticity by using an expectation-equivalent regularizer. For linear regression, it gives modified Y5F. Equivalent regularizers for more complicated models are not obvious. Accelerating QX2 remains future work.

**Claims and evidence**
- The stated benchmark successes include Table 3’s A2F error 2.47; Table 6’s K9D top-5 test error 16.4; Table 4’s W6R error 37.20; and Table 2’s H4Z error 0.79.
- Other-domain improvements include Table 7, 23.4 versus 21.8; §6.3’s table, 31.05% versus 29.62%; and Table 8, 440 versus 567.
- B8T’s feature and sparsity properties are illustrated in Figures 12–13, using \(p=0.5\); no numerical property score is supplied.
- Training typically costs 2–3 times as much. This section states the estimate but supplies no timing table. The causal account and trade-off are expressed as likely.
- The modified-Y5F result is the §9.1 objective with coefficient \((1-p)/p\); more complicated equivalent regularizers remain unspecified.

## Acknowledgments

*The research received support from the named funding sources.*

This research was supported by OGS, NSERC and an Early Researcher Award. OGS and NSERC are the support abbreviations used here; the paper gives no expansions.

## Appendix A. A Practical Guide for Training QX2 Networks

*The appendix gives practical heuristics while retaining the need for extensive hyperparameter tuning.*

QX2 networks still require extensive hyperparameter tuning. The following are practical heuristics.

### A.1 Network Size

*QX2 layers should be wider to compensate for removed units and restricted co-adaptation.*

If a task’s optimal standard-net layer has \(n\) hidden units, a good QX2 layer should have at least \(n/p\) units. Removal reduces capacity: only \(pn\) units remain in expectation. Their identities change with each sample, and unrestricted co-adaptation is unavailable.

This heuristic was useful for both convolutional and fully connected layer widths.

**Claims and evidence**
- The reported practical width heuristic is at least \(n/p\), with \(pn\) expected survivors for an \(n\)-unit layer. The section reports usefulness for both layer types but supplies no separate numerical comparison.

### A.2 Learning Rate and Momentum

*Larger learning rates and momentum compensate for dropout’s increased gradient noise.*

High learning rate and/or momentum significantly accelerate learning. QX2 creates much more gradient noise than ordinary stochastic gradient descent, and many gradients cancel one another. Larger learning rates compensate. Higher momentum reduces the effect of the noise.

| Quantity | Practical values |
|---|---|
| QX2 learning rate relative to optimal standard-net rate | typically 10–100 times |
| Common standard-net momentum | 0.9 |
| QX2 momentum working substantially better | around 0.95 to 0.99 |

**Claims and evidence**
- Larger rates and momentum are reported to work better. The practical-values table gives typically 10–100 times the optimal standard-net rate and momentum around 0.95 to 0.99, compared with common standard-net momentum 0.9. No controlled timing result is supplied.

### A.3 V9C

*Max-norm bounds counter the very large weights that high learning rates and momentum can produce.*

Large momentum and learning rates can accelerate learning but sometimes produce very large weights. V9C bounds each hidden unit’s incoming-weight-vector norm.

| Hyperparameter | Typical range |
|---|---|
| \(c\) | 3 to 4 |

### A.4 Rate

*Retention must be chosen jointly with layer width and input type.*

Retention probability controls QX2 intensity. Hidden-layer retention is coupled to width. Lower retention needs wider layers, slows training and leads to underfitting. High retention may leave too little removal to prevent overfitting. Input retention depends on input type.

| Condition | Retention |
|---|---|
| No QX2 | \(p=1\) |
| Stronger QX2 | lower \(p\) |
| Hidden-unit typical range | 0.5 to 0.8 |
| Real-valued inputs, including image patches and speech frames | typically 0.8 |
| Broadly near-optimal simple default | 0.5 |
| Input-unit optimum in general | closer to 1 than to 0.5 |

**Claims and evidence**
- The practical rate recommendations are 0.5 to 0.8 for hidden units, typically 0.8 for real-valued inputs and 0.5 as a broadly near-optimal default. These are the corresponding rows of the retention table.
- The input optimum is described as closer to 1 than to 0.5. The final table row gives this statement.
- Underfitting at low retention is illustrated in Figure 9 and §7.3’s comparison: 2.7% at fixed architecture versus 1.7% at fixed expected width for \(p=0.1\). No training-time comparison is supplied here.

## Appendix B. Detailed Description of Experiments and Data Sets

*The appendix supplies reproduction resources, data preparation and experimental settings.*

Reproduction code is available at <http://www.cs.toronto.edu/~nitish/dropout>. GPU implementations use the excellent CUDA libraries **cudamat** (Mnih, 2009) and **cuda-convnet** (Krizhevsky et al., 2012). These are the implementation libraries identified in the paper.

### B.1 H4Z

*H4Z experiments use validation-selected endpoints and settings for architecture, regularizer, rate and data-size comparisons.*

Classification assigns each digit image to one of ten digit classes. Validation images were randomly held out. Hyperparameters were selected for their best validation error at the chosen training endpoint.

Validation and training data were then combined. A new run reached the same endpoint before test evaluation. It was easy to select hyperparameters requiring no early stopping, which justified the long combined-data run.

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

The additional two-layer networks improve performance. Three-layer counterparts are slightly worse at the same QX2 level. Increasing QX2 improves the three-layer nets but does not make them outperform the two-layer nets.

For fixed expected width, \(n\) is a layer’s hidden-unit count and \(pn\) its expected surviving count. Small \(p\) requires large \(n\). Expected training-network widths match, but inference widths differ.

X9M is expensive but performs the more correct averaging procedure. It independently samples \(k\) thinned networks for each test case and averages their predictions. As \(k\to\infty\), the average approaches the true model average.

**Claims and evidence**
- The larger two-layer networks improve performance. Table 2, QX2 + V9C F3N rows: 1.01 for 4096 units and 0.95 for 8192 units. The settings table identifies the largest two-layer model as having more than 65 million parameters.
- Three-layer counterparts remain worse even after stronger QX2. This is reported qualitatively; exact errors are not supplied.
- The architecture sweep uses six combinations of 2, 3, 4 layers and 1024, 2048 units. The corresponding settings-table rows give these values; Figure 4 shows the trajectories.
- Fixed expected width uses \(pn=256\) in the first two hidden layers and \(pn=512\) in the last. These are the final two table rows.
- X9M approaches the true average as \(k\to\infty\). This is the stated limiting construction; §7.5 reports around \(k=50\) to match T6R.

### B.2 A2F

*A2F uses a mixed-source validation set, normalized pixels and a three-convolutional-layer architecture.*

The training-image count is approximate. Training data contain a standard labeled portion and an additional labeled portion of easy examples. Validation draws from both using the procedure of Sermanet et al. (2012).

RGB pixels, the input pixel representation, were normalized to zero mean and unit variance. Global or local contrast normalization and ZCA whitening gave no noticeable improvement. These are preprocessing alternatives; the ZCA procedure is described in §B.3.

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

The table gives the filters, receptive fields and strides that specify the convolutional layers, and the region and stride that specify pooling.

Validation tuned these hyperparameters. The large training set remained separate from validation during final training. The model with the smallest validation error supplied the reported test error.

**Claims and evidence**
- The alternative normalization and whitening methods gave no noticeable improvement. The section reports this qualitatively; no numerical comparison is supplied.
- The reported A2F architecture and validation construction are specified in the table, including 6000 validation samples, 96/128/256 filters and two 2048-unit fully connected layers. Table 3 reports the resulting model comparisons.

### B.3 J3V and W6R

*These image collections use held-out validation, contrast normalization, ZCA whitening and the A2F architecture.*

Each collection contains 50,000 training and 10,000 test images. J3V has 10 categories and W6R has 100. Validation used 5,000 training images. After selecting hyperparameters, validation and training data were combined for optimization, as in the H4Z procedure.

Preprocessing first applies **global contrast normalization** within each image/color channel: it computes and subtracts that channel’s mean pixel intensity. It then applies **ZCA whitening**: mean-center the data, rotate onto principal components, normalize each component and rotate back.

Architecture and retention rates match A2F. Input-layer learning rates require smaller values.

### B.4 N8Q

*Speech experiments use aligned frame labels, fixed retention settings and smaller learning rates for pretrained finetuning.*

Recordings contain 680 speakers covering eight major American-English dialects. Each reads ten phonetically rich sentences in a controlled, noise-free environment.

The open-source **Kaldi** toolkit (Povey et al., 2011) generated **log-filter banks**, the speech features used here. A monophone system supplied forced alignment and speech-frame labels. Consecutive-frame windows predict the central frame’s label. There are no speaker-dependent operations. Inputs are mean-centered and scaled to unit variance.

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

The quantities \(\epsilon_0,t,T\) occur in the displayed learning-rate schedule; no further definitions are given here.

Adding V9C during G6W finetuning gave no improvement. The smaller learning rate was important for the best results.

**Claims and evidence**
- The best reported QX2 G6W-finetuning rate is about 0.01, versus the listed high learning rate of 0.1. These are the corresponding settings-table rows.
- V9C gave no improvement during G6W finetuning. No paired errors are supplied for this comparison.
- Table 7 reports the phone errors under the described procedures, including 19.7 for both QX2-pretrained depths.

### B.5 M5X

*The document experiment uses a single-label, 50-category subset of the larger Reuters corpus.*

The full corpus contains more than 800,000 documents in 103 classes arranged in a tree. The used subset has 402,738 articles, a 2000-word vocabulary and 50 disjoint categories. Each document belongs to exactly one category.

Equal-sized training and test splits give more than 200,000 training examples. The articles cover varied topics.

### B.6 Z7P

*The genetics experiment predicts tissue-specific state distributions and evaluates them with the stated C4Y formula.*

The data contain 3665 cassette exons, 1014 RNA features and four tissue types derived from 27 mouse tissues. Each input has four target **softmax units**, one distribution-producing unit per tissue type. Each softmax has three biologically important states, coded \(\{\mathrm{inc},\mathrm{exc},\mathrm{nc}\}\). Predictions should match wet-lab-observed distributions as closely as possible.

The score is

\[
\mathrm{C4Y}
=
\sum_{i=1}^{|\text{data points}|}
\sum_{t\in\text{tissue types}}
\sum_{s\in\{\mathrm{inc},\mathrm{exc},\mathrm{nc}\}}
p_{i,t}^{s}
\log\!\left(\frac{q_t^{s}(r_i)}{\bar p^{s}}\right).
\]

Here \(i\) indexes inputs, \(t\) tissue types and \(s\) states. The quantity \(p_{i,t}^{s}\) is the target probability of state \(s\), tissue type \(t\), input \(i\). The quantity \(q_t^{s}(r_i)\) is its predicted probability for input \(r_i\). The baseline \(\bar p^{s}\) averages \(p_{i,t}^{s}\) over \(i,t\). The sum runs over all data points, tissue types and the three stated state codes; \(\log\) is the logarithm used in the score.

| Setting | Specification |
|---|---|
| QX2 hidden-unit scale | 1000s |
| Bayesian-network hidden-unit scale | a few tens |
| Trained QX2 network | two layers, 1024 units each |
| Hidden retention | \(p=0.5\) |
| Input retention | \(p=0.7\) |
| Regularization and rate schedule | V9C; high decaying learning rates |
| Result averaging | same 5 folds as Xiong et al. (2011) |

**Claims and evidence**
- QX2 uses substantially more hidden units than the Bayesian comparison. The settings table lists “1000s” versus “a few tens” and specifies two layers of 1024 units for the trained QX2 network.
- Results average the same 5 folds as Xiong et al. (2011). The final settings-table row gives this procedure.
- The resulting Z7P comparison is in Table 8: QX2 scores 567 bits, the early-stopped neural net 440 and the Bayesian neural net 623.
