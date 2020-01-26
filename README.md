# NaiveBayesClassifier

This is the code for train model using Naive Bayes classifier.

# Overview

This is the code for how to train your model using Naive Bayes classier.

# Table of content

1.  [What is Naive Bayes?](#what-is-naive-bayes)
2.  [What is Bayes Theorem?](#what-is-bayes-theorem)
3.  [Type of Naive Bayes](#type-of-naive-bayes)
4.  [What are the application of Naive Bayes?](#what-are-the-application-of-naive-bayes)
5.  [How to train your model?](#how-to-train-your-model)
6.  [Dependencies](#dependencies)
7.  [Usage](#usage)
8.  [Conclusion](#conclusion)

# What is Naive Bayes?

Naive Bayes is a simple but powerful algorithm for Compelling research.
It is the set of supervised learning algorithms based on **Bayes
Theorem**. In simple terms, The Naive Bayes classifier assumes that the
presence of a particular feature in a class is unrelated to the presence
of any other feature even if these features depend on each other or upon
the existence of the other features.

# What is Bayes Theorem?

[Bayes Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) finds
the probability of an event occurring given the probability of another
event that has already occurred. Bayes theorem is stated mathematically
as the following equation:

![bayes theorem equation](images/bt.svg "Bayes Theorem Equation")

Where Events A and B are, and P(B) is not equals to 0

- P(A|B) is a conditional probability: the likelihood of event A
  occurring given that B is true.
- P(A) and P(B) are the probabilities of observing A and B respectively.
- P(A|B) is also a conditional probability.


# Type of Naive Bayes

- **Gaussian Naive Bayes:**

  The Gaussian naive bayes is the extension of the Naive bayes. It is
  the best way to work is to estimate the mean and standard deviation
  from your training data. It is used in classification and it assumes
  that the feature follow our normal distribution.

- **Multinomial Naive Bayes:**

  This algorithm implements data distributed in multinomial form. This
  is one of two classic naive variants of Bayes used in text
  classification (where the data is typically expressed as counts of
  word vectors).

- **Bernoulli Naive Bayes:**

This algorithm is useful if your feature vectors are binary (i.e. zeros
and ones). One application would be text classification with how often
word occurs in the document and word does not occur in the document
respectively.


# What are the application of Naive Bayes?

- News Grouping categorization
- Spam Filtering
- Face Recognition
- Medical Areas
- Weather forecast prediction

# How to train your model?


# Dependencies

- Pycharm - IDE is used to create the project.
- scikit-learn - Library is used to train model using Naive Bayes.
- pandas - Library is used to load data.
- matplotlib
- seaborn
- numpy

# Usage

- Just clone or download the repository.
- Open project in Pycharm IDE. If you don't have IDE then download from
  the [JetBrains PyCharm](https://www.jetbrains.com/pycharm/download/).
- Set your python environment and install library dependencies.

That's all,You're ready to fly.

# Conclusion

The classifiers of Naive Bayes are operating on the basis of the theorem
of the Bayes, which defines the probability of an occurrence based on
prior knowledge of the conditions of the event.
