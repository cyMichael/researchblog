# Latent variable modeling in Dialogue Generation

> Type: Tutorial. Can be worth Reading.

## Introduction

Seq2seq models have developed so much in recent years in text generation tasks such as NMT and dialogue systems. We have been sought to incorporate external factors(such as emotion, speaker identity, or task completion) and modeling language features(cohesion, relevance, diversity) into dialogue systems. 

## Modeling dialogue systems - A probablistic view
We may incorporate a variety of factors into dialogue generation. Let explicit factors be $S$ and latent factors be $Z$. For a basic example, consider incorporating "persona information" into dialogue models[1], where persona is labeled in the corpus. It can be modeled by:

$X,S\rightarrow Y$


\(imagine the probability graph in your mind)

The implementation of [1] is quite intuitive: The embedding vector of $S$ was concatenated into the input of the decoder. Actually, it's a popular and very effect method for incorporating external factors into language models.

[1] A Persona-Based Neural Conversation Model, Jiwei Li et al. ACL 2016

Next, consider a more complicated application, that models discourse diversity in dialogue generation. Here we deny that $Y$ is directly determined by $X$; instead, it is also decided by a stocastic latent factor $Z$. This modeling is appealing, since we all now human responses may differ even given the same context $X$.

$X \rightarrow Z \rightarrow Y$

$X \rightarrow Y$

However, we soon notice that $Z$ is not labeled. What's more, $Z$ is considered to be stocastic. Here we employ *neural variational inference* on $Z$, where $Z$ is treated as latent variables. We do not explain too much about how the learning objective of variational inference is deriviated. (Actually, *neural* version of variational inference comes from intuition more than from math formulation - as I believe)

$\mathcal{L} = -E_{q(Z|X,Y)}[p(Y|X,Z)] + KL[q(Z|X,Y)||p(Z|X)]$

The first term of the loss is generation loss. During training, we sample $Z$ from the output distribution of a DNN that takes $X$,$Y$ as input, then append it to the decoder to generate response for computing the sequence loss. The second term is the distance of prior(p) and posterior(q) distributions of latent variable $Z$, which are outputs of two networks respectively. During testing, we sample from $p$ instead of $q$ to feed into the generation network(note that we do not have input for q - Y is not given during testing). The distribution of $Z$ is often assumed to be gaussian to allow end-to-end supervised training. (see *reparameterization trick*), but it can also be multinomial, where we have to use reinforce algorithm for training ususally(sampling is discrete - we cannot compute graident for this action). However, multinomial distribution sometimes can be more interpretable and easy to sample from during testing than continuous guassian latent variables. 

## Example: Unsupervised Discrete Sentence Representation Learning for Interpretable Neural Dialog Generation， Tiancheng Zhao et al. ACL2018

https://github.com/snakeztc/NeuralDialog-LAED

The motivation of this paper is to discover interpretable meaning representaions(discrete latent actions) of utterances.
The key contribuitons are

- Unsupervised learning of discrete salient features
- Improve the learning objective and overcome the posterior collapsing issue

Latent variable $z$ is modeled independently with the context. The network consists of two parts - recognition network $q(z|x)$ and a generation network. When z is learned, the model introuduce and encoder-decoder network $p(x|z,c)$ and a policy network $p(z|c)$. They work together by first sampling z, then generate x.

For the baseline model, it incorportates VAE with discrete latent space to learn the sentece representation. When sampling from this distribution, they use the Gumble-Softmax trick to obtain low-variance gradients. The weighted embedding of latent variables are treated as the initial hidden state of the decoder.

The author then analyzed the problem of posterior collapse, showing that the KL term in ELBO is trying to reduce mutual information between latent variables and the input data by taking the expectation over the dataset, which is claimed to the cause for the posterior collapse issue

$\mathcal{L}_{VAE}=E_{q(z|x)p(x)}[logp(x|z)-I(Z,X)-KL(q(z)||p(z))]$

(refer to adversarial autoencoders, infoVAE for more details)

The author proposes to use the following learning objective

$\mathcal{L}_{VAE} + I(Z,X)=E_{q(z|x)p(x)}[logp(x|z)-KL(q(z)||p(z))]$

where q(z) is approximated calcluated by sampling x and averaging q(z|x) in a batch, called Batch Prior Regularization. The model is refered to as DI-VAE.

Inspired from that dialogue acts are closely related to adjacent context, the author proposes Skip Thought representation of sentences(just like skip-gram that predicts adjacent tokens) to capture semantic information. This training method is refered to as DI-VST.

The same as discuss at the beginning of this tutorial, when training the decoder, z is sampled from q(z|x); another network is trained to predict $E_{p(x|c)}[q(z|x)]$. In addition, the author employs techniques in the paper *controllable text generation framework* by penalizing the decoder if the responses do not reflect the attributes in z by

$L_{attr}=E_{q(z|x)p(c,x)}[logq(z|decoded(c,z)]$

The author points out the difference of the model with CVAE. In CVAE, I(x,z|c) is encouraged to learn z that is context dependent. Next, since z is trained to generate x via p(x|z,c), the z can only be interpreted along with a certain context. z has no indepenedent standalone.

Experiment has shown that the z is prone to cluster for similar semantics. Besides, AE and ST models forms different types of clusters.

## Example: Latent Intention Dialogue Model， TH Wen et al. ICML 2017

https://github.com/shawnwun/NNDIAL

## Example: Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders, Tiancheng Zhao et al. ACL 2017

https://github.com/snakeztc/NeuralDialog-CVAE


