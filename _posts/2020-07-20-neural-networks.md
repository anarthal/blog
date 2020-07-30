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

$$ y_{prob} = a = \sigma(z) $$
  
Where $$ w_i $$ are the *weights* of the logistic regression model and $$ b $$ is the *bias term* (if you read [this]({{ "/posts/logistic-regression/" | relative_url }}), I called them $$ \theta_i $$ and $$ \theta_0 $$, respectively). These are the parameters the model would learn by minimizing the cost function.

The sigmoid function makes the output to have a non-linear relationship with the input. In the neural network context, these non-linear functions are called *activation functions*.

This simple logistic regression model can be thought of as a computation graph:

![Logistic unit]({{ "/assets/img/neural-networks/logistic-unit.jpg" | relative_url }})

What does this have to do with neural networks? ANNs are composed of several *units* like the one shown above. A neural network may have thousands of these components connected together, which allows them to learn very complex non-linear functions. 

## Anatomy of a neural network: layers

The idea of a neural network is to connect several units like the one above together, such that the output of one unit is wired to the input of another. Units are grouped in *layers*, forming a structure like the following:

![Layers]({{ "/assets/img/neural-networks/layers.png" | relative_url }})

Don't get intimidated by the notation in this diagram! We will go through it in a minute.

As shown above, there are three types of layers:

- The *input layer* is always the first one, shown in blue. It does no computation: it is just a way to represent the input of our network (the features $$ x_i $$).
- The *hidden layers*, shown in green, perform the computation explained before: a linear function followed by a non-linear activation function. The first hidden layer gets the features as input. Deeper layers get the output of previous layer's units as inputs. These output values are called *activations*.
- The *output layer* is the last one, and also performs a similar computation as hidden units. For binary classification, the output layer has a single unit, and its activation is the prediction of the network.

As the input layer does not perform any computation, we don't count it as an actual layer. Thus, the above network has 3 layers. In general, a network may have $$ L $$ layers ( $$ L = 3 $$ in this case).

Note that each unit is connected to every single unit in the previous layer. This is why this network architecture is sometime called _fully connected_.

## Computing activations

We will denote by $$ a_i^{[l]} $$ the output (activation) of the $$ i $$th unit in layer number $$ l $$. For example, $$ a_3^{[1]} $$ is the output of the 3rd unit of the first layer. Look at the figure in the previous section to double-check that you understand this notation. The text inside each unit repesents its output.

How can we compute each activation? We will follow the same procedure as for logistic regression. For the sake of example, let's compute $$ a_1^{[2]} $$, the activation of the first unit in the second layer:

$$ z_1^{[2]} = w_{21}^{[2]} a_1^{[1]} + w_{22}^{[2]} a_2^{[1]} + w_{23}^{[2]} a_3^{[1]} + w_{24}^{[2]} a_4^{[1]} + b_1^{[2]} $$

$$ a_1^{[2]} = g^{[2]}(z_1^{[2]}) = \sigma(z_1^{[2]}) $$

Wow, that seems intimidating. Don't get fooled by this apparently complex expression: it is just computing the same expression as logistic regression! The only difference is that every single unit has different parameters: different values for the weights $$ w $$ and the bias term $$ b $$. We are using the following notation:

- $$ w_{ij}^{[l]} $$ is the weight that unit $$ j $$ in layer $$ l $$ gives to the activation coming from the $$ i $$th unit in the previous layer. In the diagram shown above, each arrow represents one of these weights. For example, the arrow going from $$ a_2^{[1]} $$ to $$ a_3^{[2]} $$ represents $$ w_{23}^{[2]} $$.
- $$ b_{j}^{[l]} $$ is the bias term for unit $$ j $$ in layer $$ l $$. They play the same role as $$ b $$ in logistic regression. They are not represented in the figure.
- $$ g^{[l]} $$ is the activation function for layer $$ l $$. For now, this is equivalent to the sigmoid function. We will see later that we can use other activation functions that work better than sigmoid for neural networks.

Both $$ w_{ij}^{[l]} $$ and $$ b_{j}^{[l]} $$ are the *learnable paremeters* of the neural network: they can be trained by minimizing a cost function, as in logistic regression.

### Matrix notation

Do you like subscript notation? Neither I do. It turns out that all the computations in a neural network can be expressed as matrix operations. This simplifies notation a lot and makes implementations much faster, as computers prefer matrix operations to loops.

We can stack the $$ z $$ values and the activations in a single column vector per layer. If we do this, we get:

$$ 
\boldsymbol z^{[l]} = \begin{bmatrix} z_1^{[l]} \\ z_2^{[l]} \\ ... \\ z_{n_l}^{[l]} \end{bmatrix} 
\text{            }
\boldsymbol a^{[l]} = \begin{bmatrix} a_1^{[l]} \\ a_2^{[l]} \\ ... \\ a_{n_l}^{[l]} \end{bmatrix}
$$

$$ n_l $$ represents the number of hidden units in layer $$ l $$. In the network above, $$ n_1 = 4 $$, $$ n_2 = 2 $$ and $$ n_3 = 1 $$. The vectors $$ \boldsymbol z^{[l]} $$ and $$ \boldsymbol a^{[l]} $$ have dimensions $$ (n_l, 1) $$.

We can also stack the biases into a vector and the weights into a matrix, defining:

$$ 
\boldsymbol b^{[l]} = \begin{bmatrix} b_1^{[l]} \\ b_2^{[l]} \\ ... \\ b_{n_l}^{[l]} \end{bmatrix} 
\text{            }
\boldsymbol W^{[l]} = \begin{bmatrix}
	w_{11}^{[l]} & w_{12}^{[l]} & ... & w_{1n_{l-1}}^{[l]} \\ 
	w_{21}^{[l]} & w_{22}^{[l]} & ... & w_{2n_{l-1}}^{[l]} \\ 
	... & ... & ... & ... \\
	w_{n_l1}^{[l]} & w_{n_l2}^{[l]} & ... & w_{n_ln_{l-1}}^{[l]}
\end{bmatrix}
$$

Where $$ \boldsymbol b^{[l]} $$ has dimensions $$ (n_l, 1) $$, and $$ \boldsymbol W^{[l]} $$ is $$ (n_l, n_{l-1}) $$.




- Matrix formulation
- Activation functions
- Generalization for multiple outputs
- Forward prop and backprop
- Notes on optimization: gradient descent and similar, mini batches, learning rate
- MNIST digit recognition: problem statement
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