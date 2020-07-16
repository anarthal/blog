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

The following figure shows the aforementioned split. Lighter points belong to the training set, while darker ones belong to the test set. Note that the train and test set appear to be similarly distributed, which is good.

![Train-test split]({{ "/assets/img/underfitting-overfitting/train-test-split.png" | relative_url }})

## Other

- Generalization error. No free lunch theorem?
- Train/test split as a way to measure it (note: CV)
- Error decomposition in bias and variance
- Model complexity and influence in this stuff
- Examples with bias, variance, and just right
- How to detect if you're overfitting, underfitting, or okay without diagrams (maybe mention learning curves?)
- Bias/variance tradeoff and other ways to influence it: number of features, number of training examples, regularization
- Standarization

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