from __future__ import division
import numpy as np
from sklearn import tree
import random
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

def cal_acc(result, test_labels):
	m = np.shape(result)[0]
	result = np.sign(result)
	e = 0
	for i in range(m):
		if result[i] != test_labels[i]:
			e += 1
	print('准确率: ', '%.4f' % (1- e/m))

def train(train_set, train_labels, T):
    weak = {}
    for i in range(T):
        sample = []
        sample_label = []
        m = np.shape(train_set)[0]
        for j in range(int(2/3 * m)):
            index = random.randint(0, m - 1)
            sample.append(train_set[index])
            sample_label.append(train_labels[index])
        weak[i] = tree.DecisionTreeClassifier(splitter='random')
        weak[i].fit(sample, sample_label)
    return weak

def test(test_set, classifierArr):
    m = np.shape(test_set)[0]
    final_predict = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        predict_i = classifierArr[i].predict(test_set)
        final_predict += np.mat(predict_i).T
    return final_predict

def crossTest(train_set, train_labels, T):
	kf = KFold(n_splits = 5)
	auc = []
	i = 0
	for train1, test1 in kf.split(train_set):
		classifierArr= train(train_set[train1], train_labels[train1], T)
		result = test(train_set[test1], classifierArr)
		test_auc = metrics.roc_auc_score(train_labels[test1], result)
		auc.append(test_auc)
		i += 1
	return sum(auc) / i

train_set = np.genfromtxt("adult_dataset/adult_train_feature.txt")
train_labels = np.genfromtxt("adult_dataset/adult_train_label.txt")
test_set = np.genfromtxt("adult_dataset/adult_test_feature.txt")
test_labels = np.genfromtxt("adult_dataset/adult_test_label.txt")
zeroTo(train_labels)

""""
X = []
Y = []
for i in range(0, 51, 10):
	auc = crossTest(train_set, train_labels, i)
	X.append(i)
	Y.append(auc)
	print('auc: ', '%.4f' % auc)

"""

classifierArr = train(train_set, train_labels, 280)
result = test(test_set, classifierArr)
toZero(result)
auc = metrics.roc_auc_score(test_labels, result)
print('auc: ', '%.4f' % auc)