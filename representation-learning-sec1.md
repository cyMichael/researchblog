# How representation learning helps

> Type: Personal Blog and Thoughts. Low Recommendation for learners.

## Introduction

Most of the machine learning mechanisms try to learn the underlying representations of data. One example is autoencoders. In Computer Vision tasks, we build models such as VAEs, GANs, seemingly just to "reconstruct the origin image from latent variables". Here comes a question: why do we do that?

Consider another technique that most NLP folks are familiar with: Pretrained embeddings. They are word vector representations trained on large corpora. Loading them as initial weights of models often turns out to be a good idea. But how does it help general supervised learning?

## Disentangling Causual Factors
Quite oftenly, data that we observe comes from a generative process from their canonical "causual factors". For example, the pictures of cats and dogs differs because they have distinct underlying objects. More generally, let $h$ be causual factors of $X$. The generation process of $X$ can be written as $P(X,h)=P(X|h)P(h)$ , and the distribution of $X$ has the marginal probability $P(X)=E_h[p(X|h)]$ . It indicates that knowledge of $h$ helps us to learn $P(X)$. 

However, in most scenarios, we would like to learn $P(y|X)$. Does knowledge of $P(X)$ helps learning $P(y|X)$ ? When is it helpful? Here is a concise conclusion: 

<b>Semi-supervised pretraining is helpful when $y$ is closely tied with a causual factor of $X$, and when it disentangle the underlying factors of variation.</b>

![](/assets/sec1-1.png)

Ideally, a perfect representation should uncover all these causual factors. In this scenario, if y is closely tied with one of causual factor, it will be nearly trivial to learn P(y|X). Consider an extreme example in <i>Figure 15.4 deep learning book</i>, where P(X) is a Gaussian Mixture which is determined by the value of y, which implies y is exactly a causual factor of X. In this example, the knowledge of the distribution shape of X enables almost perfect learning of Y. Another example is classifation tasks of cats and dogs. If an unsupervised clustering method disentanges the causal factor "cat" or "dog", it will be extremely easy to learn the mapping for an image to their label(which is also one of the causual factor)

However, usually it is impossible to learn all the causual factors of X. Thus it is crucial to determine what factors are the most salient. Different models have different calibrations for salience. Here are just a few examples:

- VAEs: In these models, Mean Squared Errors are computed - If the absence of a factor harshly increase the mean reconstruction error, the factor is considered to be salient.
- GANs: In these models, if the absence of a factor significantly increase the recognition of the generated outputs as "fake output", the factor is considered to be salient.

And these knowledge acquired for P(X) can help learning process of P(y|X). Now we could explain why clustering methods and glove embedding are helpful to models.

## Transfer learning
We discussed how models benefit from "pretraining" in the section above. It leads us to a more prospective conjecture: Do the models, that trained on a different dataset, help the current model?

Well, it is called <b>Transfer Learning</b>. Actually many of us may have practised on it - Fine tuning(from different models) is a basic instance of transfer learning. The method is supported by the assumption that different data may share common representations in layers. We may tune layers near inputs for different input datas, or tune layers near outputs for different tasks(multi-task learning).

An extreme case is <b>One-shot Learning</b> and <b>Zero-shot Learning</b>. In these cases, $P(y|X,T)$ is modeled, where y,X are traditional inputs and outputs. Though there are few label examples (y,X,T) for certain labels, the model can just learn the mapping from the unlabeled data (X,T). Here T can not be a one hot vector, instead, it should be a distributed representation that could somewhat generalize. Here are some examples for zero-shot learning:
- NMT with pretrained word embeddings: The model can map unseen words since unseen words may successfully map between two embeddng spaces via learned mapping function. Here, the embedding of the source word can be viewed as T.
- A VQA object detection problem for unseen entities with binary answers. The question asked can be viewed as T. For example, if the model manage to capture the causual factor that "cat have for legs" via unlabeled data pairs (X,T), it may produce valid answers  even there are no labeled data for cats.

All in all, zero-shot learning is possible only when the model learns the mapping between representations of inputs.

Zero-shot learning includes various implementations. Yet I have not tried on this. These methods may be introduced in a seperate section or in paper reviews, along with my study.