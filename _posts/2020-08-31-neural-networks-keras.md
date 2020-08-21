---
title: Deep dive into neural networks - Keras
author: anarthal
date: 2020-08-31
categories: [Data Science, Machine Learning, Deep Learning]
tags: [machinelearning, deeplearning, classification]
math: true
ogimage: neural-networks/layers.png
---


## Notes on optimization

There are a lot of _optimization methods_ to solve this optimization problem: [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) is the most basic one, while methods like the [ADAM optimization algorithm](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) refine the former to be more efficient. Keras allows you to select which method to use and lets customize some of their parameters.

Keras will take care of the optimization process for us. This is fortunate, as the minimization problem is not straightforward - we would need to compute the _derivatives_ of the function $$ J $$ with respect to _every single_ parameter in our network! This process is called _back propagation_, and is completely automatic in Keras. Explaining it in depth is beyond the scope of this post.

- Generalization for multiple outputs
- Notes on optimization: gradient descent and similar, mini batches, learning rate
- MNIST digit recognition: problem statement
- Intro to keras and the sequential API

https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/