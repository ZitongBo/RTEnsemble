from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn import linear_model
from reader import *
import numpy as np
import math
from queue import Queue
from copy import deepcopy
from classifier import Classifier
from collections import Counter

class EnsembleTree:

    def __init__(self, ensemble, val_set, rank=1, coefficient=0.7):
        self.label_map = ensemble.label_map
        self.coefficient = coefficient
        self.nodeNum = 1
        self.clf_types = []
        best_clf_index, _ = selectBestClf(val_set, ensemble, rank, self.coefficient)
        root_clf = ensemble.classifiers[best_clf_index]
        root_feature = ensemble.features[best_clf_index]
        children = []
        for i in range(len(self.label_map)):
            children.append(None)
        self.root = EnsembleTreeNode(root_clf, children, root_feature)

        curr = self.root
        ensemble_root = deepcopy(ensemble)
        ensemble_root.remove(best_clf_index)
        # splitData = np.zeros((len(self.label_map), 1, len(val_set.columns)))
        data_dict = dict()
        for d in range(0, len(val_set)):
            label = self.label_map[curr.clf.result(val_set.iloc[d:d+1, root_feature])[0]]
            if label not in data_dict:
                data_dict[label] = val_set.iloc[d:d+1, :]
            else:
                data_dict[label] = data_dict[label].append(val_set.iloc[d:d+1, :])
            # print(label, data_dict[label].shape)
        children = []

        for i in range(len(self.label_map)):
            if i in data_dict:
                best_clf = self.constructTreeNode(data_dict[i], ensemble_root)
                children.append(best_clf)

            else:
                children.append(None)

        curr.children = children

    def constructTreeNode(self, dataSet, ensemble, ):
        best_clf_index, obj = selectBestClf(dataSet, ensemble, 1, self.coefficient)
        if obj < 0:
            return None
        # print('obj',obj, 'index',best_clf_index)
        # print('ensemble size:', ensemble.size, ' data', len(dataSet))
        # print()
        root_clf = ensemble.classifiers[best_clf_index]
        root_feature = ensemble.features[best_clf_index]
        self.nodeNum += 1
        children = []
        for i in range(len(self.label_map)):
            children.append(None)
        treeNode = EnsembleTreeNode(root_clf, children, root_feature)

        if obj == self.coefficient:
            return treeNode
        ensemble_root = deepcopy(ensemble)
        ensemble_root.remove(best_clf_index)
        # splitData = np.zeros((len(self.label_map), 1, len(val_set.columns)))
        data_dict = dict()
        for d in range(0, len(dataSet)):
            label = self.label_map[root_clf.result(dataSet.iloc[d:d + 1, root_feature])[0]]
            if label not in data_dict:
                data_dict[label] = dataSet.iloc[d:d + 1, :]
            else:
                data_dict[label] = data_dict[label].append(dataSet.iloc[d:d + 1, :])
                # print(label, data_dict[label].shape)
        children = []


        for i in range(len(self.label_map)):
            if i in data_dict:
                best_clf = self.constructTreeNode(data_dict[i], ensemble_root)
                children.append(best_clf)
            else:
                children.append(None)
        treeNode.children = children

        # print(EnsembleTree.nodeNum)
        return treeNode

    def predict(self, sample, D):
        curr = self.root
        results = []

        while curr is not None:
            if curr.clf.C > D:
                break
            y = self.label_map[curr.predict(sample)]
            results.append(y)
            D -= curr.clf.C
            curr = curr.children[y]

        return results, D


class EnsembleForest:
    def __init__(self, trees, label_map):
        self.trees = trees
        self.number = len(trees)
        self.label_map = label_map

    def predict(self, sample, D):
        result = []
        for s in range(len(sample)):
            result_s = []
            rD = D
            for t in range(self.number):
                temp, rD = self.trees[t].predict(sample.iloc[s:s+1, :], rD)
                result_s += temp
                if D <= 0:
                    break
            result.append(result_s)
        vote = Counter()
        major_vote = []

        for s in result:
            for i in s:
                vote[i] += 1
            pred = vote.most_common()[0][0]
            major_vote.append(pred)
        return np.array(major_vote)

    def accuracy(self, data, D):
        test_X = data.iloc[:, :-1]
        test_y = data.iloc[:, -1].values
        result = self.predict(test_X, D)
        temp = 0
        for i in range(len(data)):
            if result[i] == self.label_map[test_y[i]]:
                temp += 1

        return temp / len(data)


class EnsembleTreeNode:
    def __init__(self, clf, children, features):
        self.clf = clf  # 分类器
        self.children = children
        self.features = features

    def predict(self, sample):
        X = sample.iloc[:, self.features]  # 选择对应的输入特征
        return self.clf.result(X)[0]

def calInfoGain(dataSet):
    numFeatures = len(dataSet[0]) - 1
    # 计数数据集的香农熵
    baseEntropy = calcShannonEnt(dataSet)
    # 信息增益
    bestInfoGain = 0.0
    # 最优特征的索引值
    bestFeature = -1
    infoGain = np.zeros(numFeatures)
    for i in range(numFeatures):
        # 获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        # 创建set集合{}，元素不可重复
        uniqueVals = set(featList)
        # 经验条件熵
        newEntropy = 0.0
        # 计算信息增益
        for value in uniqueVals:
            # subDataSet划分后的子集
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算子集的概率
            prob = len(subDataSet) / float(len(dataSet))
            # 根据公式计算经验条件熵
            newEntropy += prob * calcShannonEnt((subDataSet))
        # 信息增益
        # infoGain[i] = baseEntropy - newEntropy
        infoGain[i] = - newEntropy
        # 打印每个特征的信息增益
        # print("第%d个特征的增益为%.3f" % (i, infoGain[i]))
        # 计算信息增益
        # if (infoGain > bestInfoGain):
        #     # 更新信息增益，找到最大的信息增益
        #     bestInfoGain = infoGain
        #     # 记录信息增益最大的特征的索引值
        #     bestFeature = i
        #     # 返回信息增益最大特征的索引值
    return infoGain



def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def calcShannonEnt(dataSet):
    #返回数据集行数
    numEntries=len(dataSet)
    #保存每个标签（label）出现次数的字典
    labelCounts={}
    #对每组特征向量进行统计
    for featVec in dataSet:
        currentLabel=featVec[-1]                     #提取标签信息
        if currentLabel not in labelCounts.keys():   #如果标签没有放入统计次数的字典，添加进去
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1                 #label计数

    shannonEnt=0.0                                   #经验熵
    #计算经验熵
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries      #选择该标签的概率
        shannonEnt-=prob*math.log(prob,2)                 #利用公式计算
    return shannonEnt                                #返回经验熵


def selectBestClf(dataSet, ensemble,rank=1, coefficient=0.7):
    ret = []
    for i in range(ensemble.size):
        X = dataSet.iloc[:, ensemble.features[i]]
        res = ensemble.classifiers[i].result(X)
        ret.append(res)
    ret = np.transpose(ret)
    Y = dataSet.values[:, -1:]
    predict_result = np.hstack((ret, Y))
    infoGain = calInfoGain(predict_result.tolist())
    accuracy = ensemble.accuracy(dataSet)
    obj = infoGain + coefficient * np.array(accuracy)
    max_obj_index = np.argmax(obj)
    sorted_index = np.argsort(obj)
    return sorted_index[-rank], obj[sorted_index[-rank]]
