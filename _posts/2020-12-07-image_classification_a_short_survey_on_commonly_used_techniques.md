---
title: "Image Classification with Machine Learning"
keywords: [Machine Learning, Computer Vision]
---

This is a conclusion of my final project for CS 200: Principles and Techniques of Data Science at UC Berkeley with [Cem Koç](http://cemkoc.me/). We received 100 out of 100 points for this project, while the average score is 91.59.
Our PDF Report can be accessed [here](/assets/DATA200_Report.pdf).

We applied supervised and unsupervised techniques for general multi-class classification to a dataset containing 20 labels. The images were processed and featurized using a variety of image processing techniques after doing an exhaustive exploratory data analysis. 

Afterwards, these feature vectors were used as inputs to the classifiers which are trained using 5-fold cross validation. We trained: logistic regression, k-nearest neighbors, decision tree classifier, SVM, random forest classifier, gradient boosting decision trees ([XGBoost](https://github.com/dmlc/xgboost)), and neural network ([VGG-16](https://arxiv.org/abs/1409.1556)).

We show that gradient boosting decision trees can achieve a higher accuracy in image classification task than many other traditional machine learning algorithms. However, even with careful feature engineering, we conclude that accuracies of traditional techniques are still significantly worse than the accuracy the neural net based classifier achieves on the training data.

{% include youtube-player.html id="m_QbgH7ZOAM" %}
