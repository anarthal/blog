---
title: Deep dive into neural networks - Keras
author: anarthal
date: 2020-10-06
categories: [Data Science, Machine Learning, Deep Learning]
tags: [machinelearning, deeplearning, classification]
math: true
ogimage: neural-networks/layers.png
---

Have you heard about Keras but never known what is it about or where to start with it? In this post we will explain the basics of Keras and will use it to build a handwritten digit classifier with a high level of accuracy! This is the last of a series of posts on neural networks. If you don't know what terms like _layer_ or _hidden unit_ you may find it useful to read [the first post of the series]({{ "/posts/neural-networks/" | relative_url }}), on the basics of NNs. The problem presented here is a multiclass classification one, which the [second post]({{ "/posts/neural-networks-multiclass/" | relative_url }}) is all about. 

# Problem statement

Imagine you're working for the post service. Packets are routed using their postcode. You would like to come up with a model that knows how to read the handwritten numbers in the postcodes, so you can automate the entire process. This problem is called handwritten digit recognition.

We will be working with the [MNIST database of handwritten digits](https://en.wikipedia.org/wiki/MNIST_database), which is the de-facto "hello world" dataset for computer vision problems. It consists of a set of 28 by 28 grayscale images containing handwritten digits. Each image is labeled from 0 to 9, according to the digit it represents. This is thus a [multiclass classification](https://en.wikipedia.org/wiki/Multiclass_classification) problem.

Our task is to build a model that classifies as much images correctly as it can. We will be using accuracy to measure the classifier's performance. The MNIST dataset is available in Kaggle through [this competition](https://www.kaggle.com/c/digit-recognizer). I will show some code snippets throughout this post; you can find the entire code listing for it in [this Kaggle kernel](https://www.kaggle.com/anarthal/mnist-digit-recognition-plain-network).


# Tensorflow and Keras

# Keras workflow

# Optimization, mini-batches and epochs

# Fitting our model

# Evaluating our model


https://www.kaggle.com/allunia/how-to-attack-a-machine-learning-model
https://www.kaggle.com/c/digit-recognizer




## Notes on optimization

There are a lot of _optimization methods_ to solve this optimization problem: [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) is the most basic one, while methods like the [ADAM optimization algorithm](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) refine the former to be more efficient. Keras allows you to select which method to use and lets customize some of their parameters.

Keras will take care of the optimization process for us. This is fortunate, as the minimization problem is not straightforward - we would need to compute the _derivatives_ of the function $$ J $$ with respect to _every single_ parameter in our network! This process is called _back propagation_, and is completely automatic in Keras. Explaining it in depth is beyond the scope of this post.

- Generalization for multiple outputs
- Notes on optimization: gradient descent and similar, mini batches, learning rate
- MNIST digit recognition: problem statement
- Intro to keras and the sequential API

https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/