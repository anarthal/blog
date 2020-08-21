---
title: Deep dive into neural networks - multiclass classification
author: anarthal
date: 2020-08-18
categories: [Data Science, Machine Learning, Deep Learning]
tags: [machinelearning, deeplearning, classification]
math: true
description: How to perform multiclass classification using neural networks.
ogimage: neural-networks/layers.png
---

Some machine learning problems involve classifying an object into one of N classes. These are called [multiclass classification](https://en.wikipedia.org/wiki/Multiclass_classification) problems, as opposed to [binary classification](https://en.wikipedia.org/wiki/Binary_classification), where there is just a positive and a negative class. Handwritten digit recognition and image classification are two well-known instances of multiclass classification problems.

![Multiclass]({{ "/assets/img/neural-networks-multiclass/multiclass.jpeg" | relative_url }})

Image source: <https://towardsdatascience.com/multi-class-classification-one-vs-all-one-vs-one-94daed32a87b>

In this post we will explain how a neural network can be used to solve this problem. This is the second of a series of posts on neural networks I've been writing. If you are new to neural networks, you may want to have a look into [the first one]({{ "/posts/neural-networks/" | relative_url }}) before.

## Problem statement

Let's say we are working in an application for a factory processing fruits. The plant works with oranges, lemons and limes. For some reason, fruits are mixed together when they enter the factory. Guided by a camera, a robot separates the fruits. Our task is to develop a model that classifies a fruit into one of the three groups given an input image.

This is a multiclass classification problem with $$ K = 3 $$ classes. Each image can be translated into a set of input features $$ x_1, x_2, ..., x_n $$, with $$ x_i \in \mathbb{R} $$. A full discussion on the feature extraction process is beyond the scope of this post. As an option, we can understand the image as an array of numbers representing color intensities (one per pixel, for a grayscale image), and make each intensity an input feature. Other solutions may involve [autoencoders](https://en.wikipedia.org/wiki/Autoencoder) and [convolutional](https://en.wikipedia.org/wiki/Convolutional_neural_network) architectures. Our target is to predict a label $$ y \in \{0, 1, 2\} $$, with the following criteria:

$$ 

\begin{matrix}

y = 0 & \Rightarrow & Orange & \\
y = 1 & \Rightarrow & Lemon & \\
y = 2 & \Rightarrow & Lime & \\

\end{matrix}

$$

Our training set consists of a set of $$ m $$ labeled pairs $$ (x^{(i)}, y^{(i)}) $$, where $$ x^{(i)} \in \mathbb{R}^n $$ is the feature representation of the $$ i $$th image, and $$ y^{(i)}) \in \{0, 1, 2\} $$.

In the [previous post]({{ "/posts/neural-networks/" | relative_url }}) we presented the following neural network, suitable for binary classification:

![Neural network binary classification]({{ "/assets/img/neural-networks/layers.png" | relative_url }})

This network needs a couple changes before it can be used for multiclass classification. We will go over them in the next sections.

## Label representation: one-hot encoding

The label representation $$ y \in \{0, 1, 2\} $$ may seem natural to us, but doesn't work well for neural network training, as it implies than there is an ordering between categories. By using this representation we are telling our network that $$ Orange < Lemon < Lime $$, which does not make any sense.

Instead, we will use a [one-hot representation](https://en.wikipedia.org/wiki/One-hot). Our labels become vectors $$ y \in \mathbb{R}^3 $$:

$$ \begin{matrix}

y = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} & \Rightarrow & \text{Orange} \\
y = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} & \Rightarrow & \text{Lemon}  \\
y = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} & \Rightarrow & \text{Lime}   \\

\end{matrix} $$

In the general case, we will use vectors $$ y \in \mathbb{R}^K $$, with $$ K $$ being the number of classes. If an object belong to class $$ c $$, the vector's $$ c $$'th element will be a one and the rest will be zero.

This representation is adequate because the classes are mutually exclusive: a fruit may be an orange or a lemon, but not both at the same time!

## The output layer

In binary classification, the output layer produces a single value $$ y_{prob} $$, which we interpret as the probability of the input to belong to the positive class. This is coherent with labels, which are represented as a single value.

In multiclass classification, labels are vectors of $$ K $$ elements instead. That means we will have to update our output layer so it produces $$ K $$-dimensional predictions. The new output vector $$ $$ y_{prob} \in \mathbb{R}^K $$ will have as many numbers as possible classes. Each of these numbers will be between zero and one, and can be interpreted as the probability of the example to belong to each class.

For the sake of example, consider the following prediction:

$$ y_{prob} = \begin{bmatrix} 0.7 \\ 0.2 \\ 0.1 \end{bmatrix} $$

This means that our networks thinks there is a 70% chance that the input picture is an orange, a 20% chance that it's a lemon, and a 10% chance that it's a lime. The final prediction would be _orange_. Note that all the probabilities in the output vector sum 1.

With these changes, our network becomes:

![Layers multiclass]({{ "/assets/img/neural-networks/layers-multi-class.png" | relative_url }})

## The loss function

For binary classification we can use the log loss (also called the cross-entropy loss) with the following formulation:

$$ L(y, y_{prob}) = - y \ log(y_{prob}) - (1 - y) log(1 - y_{prob}) $$

Where $$ y_{prob} $$ is what our model predicted and $$ y $$ is the actual label.

The form shown above is a particularization of the cross-entropy loss for the case where we just have two classes. For our example, we can use the following loss:

$$ L(y, y_{prob}) = - y[0] \ log(y_{prob}[0]) - y[1] \ log(y_{prob}[1]) - y[2] \ log(y_{prob}[2]) $$

Where the square brackets mean array indexing (recall that both $$ y $$ and $$ y_{prob} $$ are 3-element arrays).

An example is worth a thousand words, so go through one. Let's say that we are given a picture of an orange. With our one-hot encoding strategy, the label would be:

$$ \text{The example was an orange} \Rightarrow y = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} $$

Our network works quite well, and was 70% sure that what is saw was an orange. Concretely, this were its predictions:

$$ y_{prob} = \begin{bmatrix} 0.7 \\ 0.2 \\ 0.1 \end{bmatrix} $$

$$ L(y, y_{prob}) = - 1 * log(0.7) - 0 * log(0.2) - 0 * log(0.1) = 0.3567 $$

The network could be unsure and think every fruit is equally likely. The loss would be higher in this case:

$$ y_{prob} = \begin{bmatrix} 0.33 \\ 0.33 \\ 0.33 \end{bmatrix} $$

$$ L(y, y_{prob}) = - 1 * log(0.33) - 0 * log(0.33) - 0 * log(0.33) = 1.1087 $$

What if the network had predicted a lime, instead of an orange?

$$ y_{prob} = \begin{bmatrix} 0.1 \\ 0.2 \\ 0.7 \end{bmatrix} $$

$$ L(y, y_{prob}) = - 1 * log(0.1) - 0 * log(0.2) - 0 * log(0.7) = 2.3026 $$

This seems right:
- If the network is confident about a prediction and it's right, the cost is low.
- If the network is uncertain, the cost is higher.
- If the network is confident but the prediction ends up being wrong, the cost is much higher.

Also notice that the loss function only pays attention to the predicted probability of the actual class. In this example, the actual class was the first one, so the loss function only cares about the predicted probabilities for the first class.

For the general case with $$ K $$ classes, the cross-entropy loss has the following formulation:

$$ L(y, y_{prob}) = \sum_{c=0}^{K-1} - y[c] \ log(y_{prob}[c]) $$

If you want to dig deeper, [this post](https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451) explores the cross-entropy loss with multiple classes in depth.


## The activation function

We could use the sigmoid function as the activation function for output layer, applying it independently to each unit. However, if we did this, we would have no guarantee that the output probabilities add up to one, thus breaking our interpretation. Note that this is required because the output labels are mutually exclusive: one picture can't be an orange and a lemon at the same time.

Instead, we will use the [softmax](https://en.wikipedia.org/wiki/Softmax_function) activation function for the output layer. Given a vector $$ \boldsymbol z \in \mathbb{R}^K $$, the softmax function computes another vector of the same dimension, with its $$ i $$'th element being:

$$ softmax(\boldsymbol z)[i] = \frac{e^{\boldsymbol z}[i]}{sum(e^{\boldsymbol z})} = \frac{e^{\boldsymbol z}[i]}{\sum_{c=0}^{K-1}e^{\boldsymbol z}[i]} $$

Where $$ e^{\boldsymbol z} $$ is the exponential function, applied element-wise to the vector $$ z $$. Note that we are dividing by the sum of the exponentials. This guarantees that, if we add all the elements produced by the softmax, the result will be 1.0.

With these considerations, the output layer will do the following:

$$ \boldsymbol z^{[3]} = \boldsymbol W^{[3]} \boldsymbol a^{[2]} + \boldsymbol b^{[3]} $$
 
$$ \boldsymbol a^{[3]} = y_{prob} = softmax(\boldsymbol z^{[3]}) $$

The matrix $$ \boldsymbol W^{[3]} $$ will be 3x2, thus guaranteeing that both $$ \boldsymbol z^{[3]} $$ and $$ \boldsymbol a^{[3]} $$ have as many elements as classes.

## Conclusion




## References


https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
https://towardsdatascience.com/choosing-the-right-encoding-method-label-vs-onehot-encoder-a4434493149b