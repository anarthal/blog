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

# Problem statement

Let's say we are working in an application for a factory processing fruits. The plant works with oranges, lemons and limes. For some reason, fruits are mixed together when they enter the factory. Guided by a camera, a robot separates the fruits. Our task is to develop a model that classifies a fruit into one of the three groups given an input image.

This is a multiclass classification problem with $$ K = 3 $$ classes. Each image can be translated into a set of input features $$ x_1, x_2, ..., x_n $$. A full discussion on the feature extraction process is beyond the scope of this post. As an option, we can understand the image as an array of numbers representing color intensities (one per pixel, for a grayscale image), and make each intensity an input feature. Other solutions may involve [autoencoders](https://en.wikipedia.org/wiki/Autoencoder) and [convolutional](https://en.wikipedia.org/wiki/Convolutional_neural_network) architectures. Our target is to predict a label $$ y \in \{0, 1, 2\} $$, with the following criteria:

\begin{align*}

- $$ y = 0 \Rightarrow Orange
- 

For the purpose of training, we are given a set of $$ m $$ training examples, each one consisting of $$ (x^{(i)}, y^{(i)}) $$ pairs. $$ x^{(i)} \in \mathbb{R}^n $$ is the feature representation of the image number $$ i $$, and $$ y^{(i)}) \in \{0, 1, 2\}

 The image can be translated into a se

Our task is to predict a output label $$ y \in \{0, 1, ..., K-1\} $$ given 

The approach presented above works as long as we deal with binary classification. What if the output label could have more than two values? Imagine that you are building an application that classifies fruit images, distinguishing apples, oranges and pears. Your output label could then take three possible values. This would be a [multiclass classification](https://en.wikipedia.org/wiki/Multiclass_classification) problem.

### One-hot encoding

First of all, we can no longer represent our labels as a single number $$ y \in {0, 1} $$. We could represent our training example labels as a number $$ y \in {0, 1, 2} $$, with $$ y = 0 $$ representing an apple, $$ y = 1 $$ being an orange, and $$ y = 2 $$, a pear. However, this method doesn't work well in neural networks. Instead, we will use a [one-hot representation](https://en.wikipedia.org/wiki/One-hot). We will assign a label $$ y \in \mathbb{R}^3 $$ to each example in the training set, with the following criteria:

$$ \text{Apple } \Rightarrow y = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} $$

$$ \text{Orange} \Rightarrow y = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} $$

$$ \text{Pear  } \Rightarrow y = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} $$

### The output layer

Next, we will modify the shape of our output layer. Instead of outputing a single number $$ y_{prob} \in \mathbb{R} $$, it will produce a vector with three elements, one for each class: $$ y_{prob} \in \mathbb{R}^3 $$. Each of these numbers will be between zero and one, and can be interpreted as the probability of the example to belong to each class. For example, let's say our network outputs the following prediction:

$$ y_{prob} = \begin{bmatrix} 0.7 \\ 0.2 \\ 0.1 \end{bmatrix} $$

This means that our networks thinks there is a 70% chance that the input picture is an apple, a 20% chance it's an orange, and a 10% chance of being a pear. The final prediction would be _apple_. Note that all the probabilities in the output vector sum 1.

With these changes, our network becomes:

![Layers multiclass]({{ "/assets/img/neural-networks/layers-multi-class.png" | relative_url }})

### The loss function

What about the loss function? We can no longer use the original formulation, as it is specific to binary classification. It turns out that the log loss can be generalized easily to more than two classes. For our example (subscripts indicate indexing into the vector):

$$ L(y^{(i)}, y_{prob}^{(i)}) = - y_1^{(i)}log(y_{prob1}^{(i)}) - y_2^{(i)}log(y_{prob2}^{(i)}) - y_3^{(i)}log(y_{prob3}^{(i)}) $$

Let's see it with an example where the network output a correct prediction:

$$ \text{The example was an apple} \Rightarrow y^{(i)} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} $$

$$ \text{Our network predicted   } y_{prob}^{(i)} = \begin{bmatrix} 0.7 \\ 0.2 \\ 0.1 \end{bmatrix} $$

$$ L(y^{(i)}, y_{prob}^{(i)}) = - 1 * log(0.7) - 0 * log(0.2) - 0 * log(0.1) = 0.3567 $$

Let's see what would happen if our network had predicted the wrong class (i.e. it was an apple but the network predicted a pear):

$$ \text{The example was an apple} \Rightarrow y^{(i)} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} $$

$$ \text{Our network predicted   } y_{prob}^{(i)} = \begin{bmatrix} 0.1 \\ 0.2 \\ 0.7 \end{bmatrix} $$

$$ L(y^{(i)}, y_{prob}^{(i)}) = - 1 * log(0.1) - 0 * log(0.2) - 0 * log(0.7) = 2.3026 $$

That seems right: the penalty is big for making wrong predictions.

Note that the log loss function is also called the *cross entropy* loss. [This post](https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451) may give you further insights on this topic.

### The activation function

We could use the sigmoid function as the activation function for output layer, applying it independently to each unit. However, if we did this, we would have no guarantee that the output probabilities sum one, thus breaking our interpretation. Note that this is required because the output labels are mutually exclusive: one picture can't be an orange and a pear at the same time.

Instead, we will use the [softmax](https://en.wikipedia.org/wiki/Softmax_function) activation function for the output layer. Given a vector $$ \boldsymbol z \in \mathbb{R}^K $$, the softmax function 


The output layer will perform the following computation, in our case:

$$ \boldsymbol z^{[3]} = \boldsymbol W^{[3]} \boldsymbol a^{[2]} + \boldsymbol b^{[3]} $$

$$ \boldsymbol a^{[3]} = y_{prob} = softmax(\boldsymbol z^{[3]}) = 
   \begin{bmatrix} \frac{e^{z_1^{[3]}}}{e^{z_1^{[3]}} + e^{z_2^{[3]}} + e^{z_3^{[3]}}} \end{bmatrix} $$


## Notes on optimization

There are a lot of _optimization methods_ to solve this optimization problem: [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) is the most basic one, while methods like the [ADAM optimization algorithm](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) refine the former to be more efficient. Keras allows you to select which method to use and lets customize some of their parameters.

Keras will take care of the optimization process for us. This is fortunate, as the minimization problem is not straightforward - we would need to compute the _derivatives_ of the function $$ J $$ with respect to _every single_ parameter in our network! This process is called _back propagation_, and is completely automatic in Keras. Explaining it in depth is beyond the scope of this post.

- Generalization for multiple outputs
- Notes on optimization: gradient descent and similar, mini batches, learning rate
- MNIST digit recognition: problem statement
- Intro to keras and the sequential API


https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451