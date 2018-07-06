from __future__ import division
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def zeroTo(label):
    m = np.shape(label)[0]
    for j in range(m):
        if label[j] == 0:
            label[j] = -1
    return label

def toZero(label):
    m = np.shape(label)[0]
    for j in range(m):
        if label[j] == -1:
            label[j] = 0
    return label

def train(train_set, labels, T):
	weak = {}
	alpha = {}

	m = np.shape(train_set)[0]
	W = np.ones(m)/m

	for i in range(T):

		# using train weak learner and get this learner predict value
		weak[i] = tree.DecisionTreeClassifier(max_depth=1)
		weak[i].fit(train_set, labels, sample_weight=W)
		pred = weak[i].predict(train_set)

		# calculate error rate this iteration
		e = sum(np.multiply(pred != labels, W))
		if e > 0.5:
			break

		# calculate alpha this iteration
		alpha[i] = 0.5 * np.log((1.0 - e) / e)

		expn = -1 * alpha[i] * labels * pred
		W = W * np.exp(expn)
		W = W/W.sum()

	return weak, alpha

def test(test_set, classifierArr, alpha):
	m = np.shape(test_set)[0]
	final_predict = np.mat(np.zeros((m, 1)))
	for i in range(len(classifierArr)):
		predict_i = classifierArr[i].predict(test_set)
		final_predict += alpha[i] * np.mat(predict_i).T
	return final_predict

def cal_acc(result, test_labels):
	m = np.shape(result)[0]
	result = np.sign(result)
	e = 0
	for i in range(m):
		if result[i] != test_labels[i]:
			e += 1
	a = 1 - e/m
	a = a * 100
	print('准确率: ', '%.2f' % a)
	return a

def toInt(set):
	m = np.shape(set)[0]
	for i in range(m):
		set[i] = int(set[i])
	return set

def crossTest(train_set, train_labels, T):
	kf = KFold(n_splits=5)
	auc = []
	i = 0
	for train1, test1 in kf.split(train_set):
		classifierArr, alpha = train(train_set[train1], train_labels[train1], T)
		result = test(train_set[test1], classifierArr, alpha)
		test_auc = metrics.roc_auc_score(train_labels[test1], result)
		auc.append(test_auc)
		i += 1
	return sum(auc) / i

train_set = np.genfromtxt("adult_dataset/adult_train_feature.txt")
train_labels = np.genfromtxt("adult_dataset/adult_train_label.txt")
test_set = np.genfromtxt("adult_dataset/adult_test_feature.txt")
test_labels = np.genfromtxt("adult_dataset/adult_test_label.txt")
zeroTo(train_labels)
zeroTo(test_labels)
"""
X = []
Y = []
for i in range(210, 301, 10):
	auc = crossTest(train_set, train_labels, i)
	X.append(i)
	Y.append(auc)
	print('auc: ', '%.4f' % auc)
"""


classifierArr, alpha = train(train_set, train_labels, 300)
result = test(test_set, classifierArr, alpha)
auc = metrics.roc_auc_score(test_labels, result)
print('auc: ', '%.4f' % auc)
