---
title: "Image Classification with Machine Learning"
keywords: [Machine Learning, Computer Vision]
---

This is a conclusion of my final project for CS 200: Principles and Techniques of Data Science at UC Berkeley with [Cem Koç](http://cemkoc.me/).
Our PDF Report can be accessed [here](/assets/DATA200_Report.pdf).

We applied supervised and unsupervised techniques for general multi-class classification to a dataset containing 20 labels. The images were processed and featurized using a variety of image processing techniques after doing an exhaustive exploratory data analysis. 

Afterwards, these feature vectors were used as inputs to the classifiers which are trained using 5-fold cross validation. We trained: logistic regression, k-nearest neighbors, decision tree classifier, SVM, random forest classifier, and gradient boosting decision trees ([XGBoost](https://github.com/dmlc/xgboost)).

Our preliminary results show that our non-neural-net based classifiers achieve at least 20% validation accuracy (SVM) across multiple trials with the best classifier: Gradient Boosting Trees (using XGBoost) achieving 86% validation accuracy.

{% include youtube-player.html id="m_QbgH7ZOAM" %}
