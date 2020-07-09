---
title: Logistic regression
author: anarthal
date: 2020-07-08
categories: [Data Science, Machine Learning]
tags: [machinelearning, classification, supervised, sklearn, python]
math: true
---

In this post I will talk about one of the most basic models in Machine Learning: logistic regression. 

Logistic regression is a linear model employed for classification tasks in supervised learning. We will go through the basic maths behind the model and work through an example of binary classification.

## Problem statement

Recall that binary classification is the problem of predicting the class $$ y \in \{0, 1\} $$ of an object given a set of input features $$ x \in \mathbb{R}^n $$. To be able to train a model capable of making such predictions, we are given a set of correctly labeled examples: a set of points $$ (x^{(i)}, y^{(i)}) $$ with $$ i \in [1, m] $$.

As I am a big fan of board games, I will develop an example using [this Kaggle dataset](https://www.kaggle.com/mrpantherson/board-game-data) on board game data. The dataset comprises 5000 real boardgames, together with features as their number of players, average game duration, number of people that have bought the game, and so on. The games are ranked from best to worst popular using a rating defined by [this website](https://boardgamegeek.com/browse/boardgame).

We will try to predict whether a board game is "top" or not, where we define a "top board game" as one being among the best 1000. Thus, y = 1 if the game is one of the top 1000, and y = 0 otherwise.

Our task is to build a model to predict y. For the sake of example, we will just use the two most relevant input features (this will allow us to visualize the results better):

- $$ x_1 $$ will be the *number of buyers* (`owned` in our dataset). As it seems logical, there is a strong positive correlation between the rating and the number of buyers: popular games are generally bought by more people than other games.
- $$ x_2 $$ will be the game `weight`. Yes, apparently people like games with big boxes and a lot of stuff inside!

In this case, we have than the number of training examples m = 5000 and the number of features n = 2. Each example $$ x^{(i)} \in \mathbb{R}^2 $$.

## Making predictions

As every machine learning algorithm, logistic regression has a number of parameters that we will train to make the model fit the training set well. Knowing the parameters and the input features, the model is able to predict the output variable y. So what are the parameters of logistic regression and how can we use them to output predictions?

It turns out that logistic regression is a linear model. Similarly to what linear regression does, we will compute the following linear combination of the input features for each example:

$$ z = \theta_0 + \theta_1 x_1 + \theta_2 x_2 $$

Where $$ \theta_0 $$, $$ \theta_1 $$ and $$ \theta_2 $$ are the parameters of the logistic regression model.

However, z cannot be the output of our model, since we need to obtain a binary label, and $$ z \in \mathbb{R} $$. To obtain a prediction, we will apply the following function to z, known as the logistic or sigmoid function:

$$ y_{prob} = \sigma(z) = \frac{1}{1 + e^{-z}} $$

If we plot the function, we get the following shape:

![Sigmoid]({{ "/assets/img/logistic-regression/sigmoid.png" | relative_url }})

- As we can see, $$ y_{prob} \in (0, 1) $$.
- Big negative values of z output values close to zero.
- Big positive values of z output values close to one.
- Inputs close to zero output values close to 0.5.

We can interpret this number as the probability that the given example belongs to the class y = 1. So, if we perform this calculation for a certain example and obtain 0.1, that means that our model is convinced that this game is not top ranked. Conversely, if the had obtained 0.9, that would mean that our model is almost sure that the game is top ranked. We can take 0.5 as the boundary, such that we predict y = 0 if $$ y_{prob} < 0.5 $$, and y = 1 otherwise.

In the more general case, we have n input features. We can represent each example as a column vector of features $$ x \in \mathbb{R}^n $$. We can also define a column vector containing all the parameters in our model, $$ \theta \in \mathbb{R}^n $$. With this notation, we can write:

$$ z = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n = \theta_0 + \theta^T x $$

$$ y_{prob} = \sigma(z) $$

## Training the model

As with every machine learning model, training a logistic regressor is just optimizing a cost function that tells the model how well it's doing on the training set. For an individual training example (i), we will use the following function:

$$ L(y^{(i)}, y_{prob}^{(i)}) = - y^{(i)}log(y_{prob}) - (1 - y^{(i)})log(1 - y_{prob}) $$

Where $$ y^{(i)} $$ are the real labels of the examples (often called the ground truths) and $$ y_{prob} $$ are the probability output by our model. This function is called the [log loss function](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html). It models our problem well because of the following:

- If the actual label $$ y^{(i)} $$ is 0, the the first summand goes away, leaving the expression $$ L(0, y_{prob}^{(i)}) = - log(1 - y_{prob}) $$, which is zero for $$ y_{prob} = 0 $$ and tends to infinity for $$ y_{prob} = 1 $$. Thus, if the actual label is zero, we are rewarding the model for outputing probabilities close to zero, and we are penalizing it for the opposite.
- If the actual label $$ y^{(i)} $$ is 1, the the second summand goes away, leaving the expression $$ L(1, y_{prob}^{(i)}) = - log(y_{prob}) $$, which is zero for $$ y_{prob} = 1 $$ and tends to infinity for $$ y_{prob} = 0 $$ (the opposite to the previous bullet point).

This is for just one training example. The overall cost will be the average along all training examples:

$$ J(\theta_0, \theta) = \frac{1}{m} \sum_{i=1}^{m} L(y^{(i)}, y_{prob}^{(i)}(\theta_0, \theta)) $$

We can see that this function depends on the model parameters $$ \theta_0 $$ and $$ \theta $$, as the predictions $$ y_{prob} $$ are computed using these. Training the model just becomes an optimization problem: finding the parameters $$ \theta_0 $$ and $$ \theta $$ that makes the cost function J as small as possible. We can employ algorithms such as gradient descent to solve this problem.

### Example: the board game model

Let's use `sklearn` Python library to train a logistic regression model for our board game problem. [This Kaggle kernel](https://www.kaggle.com/anarthal/board-game-logistic-regression) has the complete code listing, together with some exploratory data analysis. We will show here the most relevant parts for our task. First of all, some imports:

```py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
```

We then load the data into a dataframe `df`. Our input features will be `owned` and `weight`, while the predicted variable will be `top`. I won't show the code the read the data here. Feel free to check the above kernel if you are curious.

As usual in ML, we split our data into a train and a test set. We will use the first one to train the model, and the second one to evaluate its performance. We then `fit` our `LogisticRegression` object, which will solve the optimization problem described above:

```py
X, y = df[['owned', 'weight']], df['top']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = LogisticRegression()
model.fit(X_train, y_train)
```

Alright, model trained. What are the value of the learned parameters? We can access them using:

```py
theta0 = model.intercept_[0]  # theta0 = -5.76
theta1 = model.coef_[0][0]    # theta1 = 0.000726
theta2 = model.coef_[0][1]    # theta2 = 0.891
```

This means that, to make predictions, our model is computing:

$$ y_{prob} = \sigma(0.000726 * owned + 0.891 * weight - 5.76) $$

## The decision boundary

As we are just using two features for predictions, it is easy to visualize them. The following figure shows the two features in the two axes. Points in green represent top board games (y=1), while points in red represent the other games (y=0):

![Features]({{ "/assets/img/logistic-regression/features.png" | relative_url }})

As you can see, points in the lower left corner are much more likely to have y=0 than points in the middle, for example. To make predictions, our model is going to split the feature space into two regions. Examples contained in the first region will be predicted as positive, with the other ones as negative. The decision boundary is the frontier between the two. As we are in $$ \mathbb{R}^2 $$ and the model is linear, the decision boundary will be a straight line:

![Decision boundary]({{ "/assets/img/logistic-regression/decision-boundary.png" | relative_url }})

The dashed black line is the decision boundary. The points in the red region are classified as negatives and the ones in the green region, as positives.



## Evaluating the model performance

How good is our is our model at making predictions? Let's use the test set to know it:

```py
model.score(X_test, y_test) # Yields 0.9088
```

By default, [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) uses accuracy as evaluation metric. Accuracy is defined as:

$$ accuracy = \frac{Correctly\ classified\ examples}{Total\ examples} $$

That means that our model predicted 90.88% of the examples in the test set correctly. Not too bad for just using two features!

## Conclusion

This finishes our study of logistic regression. We have gone thrhough basic 

## Predictions: a linear model

As every model in Machine Learning, 

- How to make predictions
  - Linear model
  - Sigmoid activation
  - Probability threshold
- Cost function
  - Why does it make sense
  - Maths behind: max likelihood estimator and GLM
- Training the model
- An example
- The decision boundary
- Going beyond a linear model
  - Polynomial features
  - Overfitting
  - Learning curves
  - Cost vs. num iterations curves
- The effect of changing the decision threshold

Further thoughts
 - f1 score
 - scaling
 - more features
 - Regularization

Note that the $$ theta_1 $$ (the weight assigned to `owned`) is very small because the scale of the feature is big in comparison to `weight`. When such a difference in scales is present, it's  

## Mathematics

The mathematics powered by [**MathJax**](https://www.mathjax.org/):

$$ \sum_{n=1}^\infty 1/n^2 = \frac{\pi^2}{6} $$

When \\(a \ne 0\\), there are two solutions to \\(ax^2 + bx + c = 0\\) and they are

$$ x = {-b \pm \sqrt{b^2-4ac} \over 2a} $$

## Titles

***
# H1

<h2 data-toc-skip>H2</h2>

<h3 data-toc-skip>H3</h3>

#### H4

***

## Paragraph

I wandered lonely as a cloud

That floats on high o'er vales and hills,

When all at once I saw a crowd,

A host, of golden daffodils;

Beside the lake, beneath the trees,

Fluttering and dancing in the breeze.

## Block Quote

> This line to shows the Block Quote.

## Tables

|Company|Contact|Country|
|:---|:--|---:|
|Alfreds Futterkiste | Maria Anders | Germany
|Island Trading | Helen Bennett | UK
|Magazzini Alimentari Riuniti | Giovanni Rovelli | Italy

## Link

<http://127.0.0.1:4000>


## Footnote

Click the hook will locate the footnote[^footnote].


## Image

![Desktop View]({{ "/assets/img/sample/mockup.png" | relative_url }})


## Inline code

This is an example of `Inline Code`.



## Code Snippet

### Common

```
This is a common code snippet, without syntax highlight and line number.
```

### Specific Languages

#### Console

```console
$ date
Sun Nov  3 15:11:12 CST 2019
```


#### Terminal

```terminal
$ env |grep SHELL
SHELL=/usr/local/bin/bash
PYENV_SHELL=bash
```

#### Ruby

```ruby
def sum_eq_n?(arr, n)
  return true if arr.empty? && n == 0
  arr.product(arr).reject { |a,b| a == b }.any? { |a,b| a + b == n }
end
```

#### Shell

```shell
if [ $? -ne 0 ]; then
    echo "The command was not successful.";
    #do the needful / exit
fi;
```

#### Liquid

{% raw %}
```liquid
{% if product.title contains 'Pack' %}
  This product's title contains the word Pack.
{% endif %}
```
{% endraw %}

#### HTML

```html
<div class="sidenav">
  <a href="#contact">Contact</a>
  <button class="dropdown-btn">Dropdown
    <i class="fa fa-caret-down"></i>
  </button>
  <div class="dropdown-container">
    <a href="#">Link 1</a>
    <a href="#">Link 2</a>
    <a href="#">Link 3</a>
  </div>
  <a href="#contact">Search</a>
</div>
```

**Horizontal Scrolling**

```html
<div class="panel-group">
  <div class="panel panel-default">
    <div class="panel-heading" id="{{ category_name }}">
      <i class="far fa-folder"></i>
      <p>This is a very long long long long long long long long long long long long long long long long long long long long long line.</p>
      </a>
    </div>
  </div>
</div>
```


## Reverse Footnote

[^footnote]: The footnote source.
