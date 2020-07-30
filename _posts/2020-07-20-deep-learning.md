---
title: Deep dive into neural networks
author: anarthal
date: 2020-07-30
categories: [Data Science, Machine Learning, Deep Learning]
tags: [machinelearning, deeplearning, classification, keras, python]
math: true
---

They are on everyone's lips: every single post today seems to talk about deep neural networks and the bewildering variety of applications they are used for. [Speech recognition](https://en.wikipedia.org/wiki/Speech_recognition), [computer vision](https://en.wikipedia.org/wiki/Computer_vision), [natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing)... the possibilities seem endless. Neural networks are the workhorse of [deep learning](https://en.wikipedia.org/wiki/Deep_learning), which is a subset of machine learning. But what is a neural network? How does it work?

This post is intended as a crash course on neural networks and the math behind them. We will also go through a computer vision example: we will train a network that recognizes handwritten digits with 95% accuracy! We will use the popular [Keras](https://keras.io/) Python framework to build our networks.

To understand this post you should have basic notions of machine learning, linear algebra and calculus. In particular, you should already know what a classification problem is and how a logistic regression model can help. If you are not familiar with these concepts, [this blog post]({{ "/posts/logistic-regression/" | relative_url }}) may help. You should also know how to multiply matrices and what a derivative is.

Enough talk. Let's dive deep into deep learning (pun intended)!

## What is a neural network?

If you are super-hyped, expecting a definition that compares a neural network with human brain, you're out of luck. A neural network is a machine learning model, just like [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) or [XGBoost](https://xgboost.readthedocs.io/en/latest/). Like these models, neural networks can be used for tasks like classification and regression.

What is the difference between traditional models and neural networks then? The latter are much more complex models. They may have billions of parameters, which allows them to learn really complex functions. This makes them suitable for incredible applications, like detecting objects (e.g. cars) in an image or diagnosing lung cancer given a radiography.

There are many types of neural networks depending on the field of application. [Convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) and [recurrent neural networks](https://en.wikipedia.org/wiki/Recurrent_neural_network) are used in computer vision and natural language processing, respectively. In this post we will focus on [artificial neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network), often called _fully connected_ networks or just _neural networks_, for short. These are the basic building block of deep learning on which the others are based. It is important to understand these well before jumping into the others.

Most of the ideas presented here come from (TODO: insert link) Andrew Ng's Deep Learning specialization on Coursera.

## Logistic regression as a mini neural network

Neural networks can be thought of as a generalization of logistic regression. Let's review logistic regression from a different angle before jumping into full-blown neural networks.

Let's say we are trying to solve a binary classification problem using logistic regression (like [this one]({{ "/posts/logistic-regression/" | relative_url }})). We are given the input features $$ x_1, x_2... x_n $$ and we are asked to predict the target label $$ y \in {0, 1} $$. This is what logistic regression would do:

- Compute a linear function of the inputs.
- Apply the sigmoid non-linear function to generate the prediction.
  
$$ z = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b $$

$$ y_{prob} = \sigma(z) $$
  
Where $$ w_i $$ are the *weights* of the logistic regression model and $$ b $$ is the *bias term* (if you read [this]({{ "/posts/logistic-regression/" | relative_url }})), I called them $$ \theta_i $$ and $$ \theta_0 $$, respectively). These are the parameters the model would learn by minimizing the cost function.

The sigmoid function makes the output to have a non-linear relationship with the input. In the neural network context, these non-linear functions are called *activation functions*.

We can represent this logistic regression _unit_ as a computation graph like the following:

## Anatomy of a neural network

A neural network is composed of *units*. Each unit is a block taking a set of inputs and calculating a single output (like functions do). Units are stacked in *layers*. The following picture represents 

A neural network is composed of several *layers*. Each layer is composed of *units*. 

 There are three types of layers:

- *Input layer*. 

![Layers]({{ "/assets/img/neural-networks/layers.png" | relative_url }})
Click the hook will locate the footnote [^footnote].


- Logistic regression refresher
- MNIST digit recognition: problem statement
- Hidden units as logistic units
- Activation functions
- Matrix formulation
- Generalization for multiple layers
- Forward prop and backprop
- Notes on optimization: gradient descent and similar, mini batches, learning rate
- Intro to keras and the sequential API



## Conclusion

This finishes our study of logistic regression. I hope the example has helped to clarify some of the maths behind the model.

A couple final thoughts:

- For the sake of example, we have just considered two features. There are plenty of other variables to improve our model.
- The two employed features have very different scales. It may be beneficial for the model to normalize the variables, so they have a similar range and variance.
- Another term is usually added to the cost function presented here, called the regularization term, which helps prevent the overfitting problem. As this is not a concern in a model as simple as ours, we have omitted it here.
- We used accuracy for simplicity, but it may not be the best metric to choose, as the dataset is slightly imbalanced. Other metrics like ROC AUC or F1 score may be more adequate.

Hope you have liked the post! Feedback and suggestions are always welcome.

## References

* Kaggle dataset on board game data: <https://www.kaggle.com/mrpantherson/board-game-data>.
* Insights - Geek Board Game (Kaggle kernel): <https://www.kaggle.com/devisangeetha/insights-geek-board-game>.
* Board Game Geek: <https://boardgamegeek.com/>.
* Machine Learning, Coursera course by Andrew Ng: <https://www.coursera.org/learn/machine-learning/>.
* Sklearn documentation: <https://scikit-learn.org/stable/>
* <https://machinelearningmastery.com/types-of-classification-in-machine-learning/>
* <https://en.wikipedia.org/wiki/Supervised_learning>