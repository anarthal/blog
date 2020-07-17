---
title: Underfitting, overfitting and model complexity
author: anarthal
date: 2020-07-16
categories: [Data Science, Machine Learning]
tags: [machinelearning, classification, supervised, sklearn, python]
math: true
---

In this post I will talk about model complexity and two phenomena that may arise when the former is inadequate: underfitting and overfitting.

This posts assumes that you know basic machine learning concepts like [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) or [binary classification](https://en.wikipedia.org/wiki/Binary_classification). We will use [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) as a model to demonstrate these concepts. If you are not familiar with it, you may check [my other post on logistic regression]({{ "/posts/logistic-regression/" | relative_url }}). A basic faimiliarity with Python and `sklearn` is also necessary.

We will demonstrate these concepts with a binary classification example, where we will try to predict whether a forest elf will survive until being an adult or not based on two numeric features. [This kernel](https://www.kaggle.com/anarthal/underfitting-overfitting-and-model-complexity) contains the full code listing for this post.

## Evaluation metric

The concepts explained in this post are closely related to model performance (how good is my model?). When working in any machine learning task, it is of vital importance to define an evaluation metric that allows us to assess the performance of any model.

As we will be dealing with binary classification, we will employ [classification accuracy](https://developers.google.com/machine-learning/crash-course/classification/accuracy) as a performance metric. Accuracy is defined as the ratio of correctly classified examples. It ranges between 0.0 and 1.0, and the higher, the better. 

I will also talk about errors during this post (as in "this model achieves a smaller error under these conditions"). I will consistently use the word _error_ to refer to the inverse of the performance metric: the lower the metric, the higher the error.

## Train and test sets

To understand what over and underfitting is, we first need to explain the concepts of train and test sets.

Recall that in binary classification, we train our model on a collection of correctly labeled examples. This dataset is called the **training set**. After the model has been trained on this set, we can calculate how many of these examples were correctly classified and how many wouldn't, and thus the accuracy of the model for the training set. This measures the **training error**.

However, doing well on the training set is not enough for a model to be good. We want our model to also do well on examples it hasn't seen previously. In other words, we want our model to be able to generalize. The **generalization error** measures this capability.

The training error is not a good estimate for the generalization error, as it is based on examples the model has already seen. The model could "memorize" the training examples, achieving low training error, but fail to generalize (we will see later that this is called overfitting).

Instead of feeding all of our labeled data to the model, we create two datasets from it:

- The actual training set, used to train our model.
- The test set, used to evaluate the model's performance.

After training the model, we ask it to predict the classification labels for all the examples in the test set. This allows us to compute the **test error**, which we will use as an estimate of the generalization error.

### An example

As promised, we will study how likely is a forest elf to surivive, given two physical features: `height` (which measures how tall the elf is) and `ear_length` (which should be obvious enough). We will try to predict whether the elf survives until being an adult or not.

As some readers may have already realized, this is a made up dataset. It's difficult to find real data that can show these concepts with just two features, which helps a lot with visualizations. The problematic presented in this post are much more likely to happen in higher dimensional datasets.

Let's visualize our dataset:

![Dataset]({{ "/assets/img/underfitting-overfitting/elf-dataset.png" | relative_url }})

Some observations on this dataset:

- It seems like there are heavy interactions between the two features. The variables are not independent.
- The features have been [standardized](https://en.wikipedia.org/wiki/Feature_scaling): they have zero mean and similar dispersion. We will talk about why this is important later.
- There is enough information to predict the class label using the two features. Negative examples are relatively grouped together, as positive examples are.

This dataset set has 2000 examples. Let's split it into train and test, with leaving 75% in the training set and 25% in the test set. We will use sklearn's [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html):

```py
from sklearn.model_selection import train_test_split

df, y = ... # df contains the features, y contains the ground-true labels
X_train, X_test, y_train, y_test = train_test_split(df, y, 
	test_size=0.25, random_state=0)
```

Note that sklearn will randomly shuffle the dataset before performing the split. Passing a constant as `random_state` ensures consistent results between runs.

The following figure shows the aforementioned split. Lighter points belong to the training set, while darker ones belong to the test set.

![Train-test split]({{ "/assets/img/underfitting-overfitting/train-test-split.png" | relative_url }})

## Underfitting

Let's first build a simple logistic regression model trained with out dataset (for an explanation about logistic regression, check out [this post]({{ "/posts/logistic-regression/" | relative_url }})):

```py
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))
```

This model has an 86.9% accurancy on the training set and an 85.4% accuracy on the test set. That's not optimal. Let's visualize the decision boundary to get a feel of what is going on here:

![Underfitting]({{ "/assets/img/underfitting-overfitting/underfitting.png" | relative_url }})

Our model is too simple to achieve a good performance on this dataset. Recall that logistic regression is a linear model. It is trying to separate the two classes using a straight line, which isn't quite right. Both the train and the test errors are high.

This situation is called **underfitting**. In our case, it happens because our model is too simple for the dataset. We will see later other causes that might also cause a model to underfit.

## Increasing model complexity: polynomial features

What can we do to make the situation better? We need to come up with a more complex model. One option would be to move away from logistic regression to a model that can learn non-linear decision boundaries, like [random forest](https://en.wikipedia.org/wiki/Random_forest) or [XGBoost](https://xgboost.readthedocs.io/en/latest/). The other option is to make a more complex logistic regression model by adding polynomial features. This will be the approach to follow here.

Recall that, to make predictions, logistic regression is computing the following:

$$ z = \theta_0 + \theta_1 x_1 + \theta_2 x_2 $$

$$ y_{prob} = \sigma(z) $$

Where $$ \theta_i $$ are the paremeters learnt by the model and $$ x_0 $$ and $$ x_1 $$ are our two input features. The output $$ y_{prob} $$ can be interpreted as a probability, thus predicting $$ y = 1 $$ if $$ y_{prob} $$ is above a certain threshold (usually 0.5). Under these circumstances, it can be shown that the decision boundary is a straight line with the following equation:

$$
\theta_0 + \theta_1 x_1 + \theta_2 x_2 = 0
$$

The problem here is that the decision boundary shouldn't be a linear function of the two features, but should also contain:

- Higher-degree polynomial terms, e.g. $$ x_0^2 $$, $$ x_0^3 $$ and so on.
- Interaction terms, like $$ x_0 x_1 $$ or $$ x_0^2 x_1^3 $$.

Resulting in a model like this:

$$ z = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_1^2 + \theta_4 x_1 x_2 + \theta_5 x_2^2 + ... $$

$$ y_{prob} = \sigma(z) $$

We can make logistic regression do this by manually creating polynomial features. We will create additional columns in the `X_train` and `X_test` matrices, containing the values of $$ x_1^2 $$, $$ x_2^2 $$, $$ x_0 x_1 $$ and so on until a certain degree. We will feed these features to the model as if they were additional variables.

Sklearn has a built-in [polynomial feature creator](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html):

```py
from sklearn.preprocessing import PolynomialFeatures

df, y = ... # df contains the features, y contains the ground-true labels
X = PolynomialFeatures(3).fit_transform(df)
X_train, X_test, y_train, y_test = train_test_split(df, y, 
	test_size=0.25, random_state=0)
model = LogisticRegression()
model.fit(X_train, y_train)
```

With this code, we get `X` to have the original features, as well as several additional columns: $$ x_1^2 $$, $$ x_1^3 $$, $$ x_2^2 $$, $$ x_2^3 $$, $$ x_1 x_2 $$, $$ x_1^2 x_2 $$ and $$ x_1 x_2^2 $$. The first argument to `PolynomialFeatures` is the maximum degree of the polynomial features to create.

Fitting this model yields 96.7% accuracy on the training set and 95.4% on the training set. That's much better! The decision boundary seems appropriate this time:

![Just right]({{ "/assets/img/underfitting-overfitting/just-right.png" | relative_url }})

## Overfitting

It seems like adding polynomial features helped the model performance. What happens if we use a very large degree polynomial? We will end up having an **overfitting** problem. Let's see what happens when using a 15 degree polynomial (I've also turned regularization off, which increases the overfitting effect - we will talk about this later):

![Overfitting]({{ "/assets/img/underfitting-overfitting/overfitting.png" | relative_url }})

This model achieves a 98.9% accuracy on the training set, but drops to 93% on the test set. The model has so much flexibility that is fitting an over-complicated decision boundary that does not generalize well. It is memorizing the training set, which proves useless when facing the test set.

## The bias-variance trade-off

Bias and variance are two properties of statistical estimators:

- [Bias](https://en.wikipedia.org/wiki/Bias_of_an_estimator) estimates how far is the expected value of the estimator from the real value.
- [Variance](https://en.wikipedia.org/wiki/Variance) measures the dispersion of the estimator around its expected value.

In machine learning, we can decompose a model's error using these properties: 

- Bias tells us if the model is able to approximate the real underlying problem well. Our first model had high bias, as it was too simple to represent the dataset we were trying to classify. Having a high bias is thus a synonym of **underfitting**.
- Variance tells us how much the predictions vary across different training sets coming from the same distribution. Our third model had high variance, as it was memorizing the training set. If we had trained the model on a slightly different training set, the model would have changed significantly. A model with high variance is thus **overfitting** the training set.

If you are interested in the mathematics underneath this decomposition, [this post](https://towardsdatascience.com/mse-and-bias-variance-decomposition-77449dd2ff55) provides an in-depth explanation. The derivation is usually performed using the MSE loss function, common in regression. However, the bias and variance concepts are also applicable to classification, as we have seen.

As you saw in previous sections, changing the model complexity affects both bias and variance. More complex models tend to have higher variance and lower bias. On the other hand, simpler models will have lower variance but more bias. There is thus a **trade-off** between the two sources of error. To make a good model, we must balance the two terms wisely.

### Factors affecting bias and variance

In the previous section, we saw how model complexity affects bias and variance. There are other aspects that may also affect these magnitudes:

- Adding more features tends to increase variance and decrease bias.
- Making the training set bigger usually decreases variance. It doesn't have much effect on bias.
- [Regularization](https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a) modifies the cost function to penalize complex models. Regularization makes variance smaller and bias higher. Sklearn's [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) uses regularization by default. It can be disabled by setting `penalty='none'`, and its magnitude is controlled by the `C` parameter (the smaller, the greater regularization effect).
- Any hyperparameter controlling model complexity is likely to have an effect on both bias and variance. For example, decision trees have a hyperparameter controlling the depth of the tree. The bigger the value, the bigger the tree and the more complex the model. Increasing this value will tend to decrease bias and increase variance.

## Diagnosing bias and variance problems

It is easy to diagnose if your model suffers from high bias or high variance when you have only two features so you can plot them. But what happens if your dataset has more than two dimensions? Then you should look at the training set and test set errors.

- If both the training set and test set errors are higher than what you would expect, it's likely that you have a bias problem.
- If the training set error is very low but the test set error is not, then your model is failing to generalize, thus having a variance problem.
- If both errors are similar and as low as you would expect, then congratulations! Your model is just right.

The below diagram shows the three logistic regression models we've come up with in this post, together with their train and test errors:

![Model comparison]({{ "/assets/img/underfitting-overfitting/model-comparison.png" | relative_url }})



## Other

- Model complexity and influence in this stuff
- How to detect if you're overfitting, underfitting, or okay without diagrams (maybe mention learning curves?)
- Bias/variance tradeoff and other ways to influence it: number of features, number of training examples, regularization
- Standarization
- CV

## Conclusion

Hope you have liked the post! Feedback and suggestions are always welcome.


## References

* Kaggle dataset on board game data: <https://www.kaggle.com/mrpantherson/board-game-data>.
* Insights - Geek Board Game (Kaggle kernel): <https://www.kaggle.com/devisangeetha/insights-geek-board-game>.
* Board Game Geek: <https://boardgamegeek.com/>.
* Machine Learning, Coursera course by Andrew Ng: <https://www.coursera.org/learn/machine-learning/>.
* Sklearn documentation: <https://scikit-learn.org/stable/>
* <https://machinelearningmastery.com/types-of-classification-in-machine-learning/>
* <https://en.wikipedia.org/wiki/Supervised_learning>



https://machinelearningmastery.com/polynomial-features-transforms-for-machine-learning/

https://towardsdatascience.com/mse-and-bias-variance-decomposition-77449dd2ff55

https://towardsdatascience.com/holy-grail-for-bias-variance-tradeoff-overfitting-underfitting-7fad64ab5d76