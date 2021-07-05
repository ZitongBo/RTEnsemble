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
from copy import deepcopy
from classifier import Classifier
from collections import Counter


class EnsembleTree:
    def __init__(self, ensemble, val_set, rank, coefficient):
        self.label_map = ensemble.label_map
        self.coefficient = coefficient
        self.nodeNum = 1
        self.clf_types = []
        best_clf_index, _ = ensemble.selectBestClf(val_set, rank, self.coefficient)
        root_clf = ensemble.classifiers[best_clf_index]
        root_feature = ensemble.features[best_clf_index]
        children = []
        for i in range(len(self.label_map)):
            children.append(None)
        self.root = EnsembleTreeNode(root_clf, children, root_feature)

        curr = self.root
        ensemble_root = deepcopy(ensemble)
        ensemble_root.remove(best_clf_index)
        print(best_clf_index, len(ensemble))
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
                best_clf = self.construct_tree_node(data_dict[i], ensemble_root)
                children.append(best_clf)
            else:
                children.append(None)

        curr.children = children

    def construct_tree_node(self, dataSet, ensemble, ):
        best_clf_index, obj = ensemble.selectBestClf(dataSet, 1, self.coefficient)
        if obj < 0.52:
            return None
        self.nodeNum += 1
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

        if obj > 0.9:
            return treeNode
        ensemble_root = deepcopy(ensemble)
        ensemble_root.remove(best_clf_index)
        # splitData = np.zeros((len(self.label_map), 1, len(val_set.columns)))
        data_dict = dict()
        for d in range(0, len(dataSet)):
            label = self.label_map[root_clf.result(dataSet.iloc[d, root_feature].values.reshape(1, -1))[0]]
            if label not in data_dict:
                data_dict[label] = dataSet.iloc[d:d + 1, :]
            else:
                data_dict[label] = data_dict[label].append(dataSet.iloc[d:d + 1, :])
                # print(label, data_dict[label].shape)

        for i in range(len(self.label_map)):
            if i in data_dict and len(data_dict[i]) > 6 and ensemble.size > 20:
                best_clf = self.construct_tree_node(data_dict[i], ensemble_root)
                treeNode.children[i] = best_clf
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
            # results = [y]
            D -= curr.clf.C
            curr = curr.children[y]
        return results, D

    def weightedVote(self, sample, D):
        vote = Counter()
        curr = self.root
        rD = D
        while curr is not None:
            if curr.clf.C > rD:
                break
            prob = curr.resProb(sample)
            for c, p in zip(curr.clf.classes(), prob):
                vote[c] += p
            rD -= curr.clf.C
            curr = curr.children[self.label_map[curr.predict(sample)]]
        return vote, rD

    def majorityVote(self, sample, D):
        vote = Counter()
        curr = self.root
        rD = D
        while curr is not None:
            if curr.clf.C > rD:
                break
            y = curr.predict(sample)
            vote[y] += 1
            rD -= curr.clf.C
            curr = curr.children[self.label_map[y]]
        return vote, rD


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
                if rD <= 0:
                    break
            result.append(result_s)
        major_vote = []
        for s in result:
            vote = Counter()
            for i in s:
                # print(i)
                vote[i] += 1
            pred = vote.most_common()[0][0]
            major_vote.append(pred)
        return np.array(major_vote)

    def majorityVote(self, data, D):
        conf_matrix = np.zeros((len(self.label_map), len(self.label_map)), dtype=np.int32)
        for i in range(data.shape[0]):
            real = data.iloc[i, -1]
            rD = D
            X = data.iloc[i, :]
            vote = Counter()
            for t in range(self.number):
                vote_t, rD = self.trees[t].majorityVote(X, rD)
                # feature = self.features[rank[j]]
                # X = data.iloc[i, feature].values.reshape(1, -1)
                # prob = self.classifiers[rank[j]].resProb(X)[0]
                for v in vote_t:
                    vote[v] += vote_t[v]
                if rD <= 0:
                    break
            pred = vote.most_common()[0][0]
            conf_matrix[self.label_map[real], self.label_map[pred]] += 1
        return conf_matrix

    def weightedVote(self, data, D):
        conf_matrix = np.zeros((len(self.label_map), len(self.label_map)), dtype=np.int32)
        for i in range(data.shape[0]):
            real = data.iloc[i, -1]
            rD = D
            X = data.iloc[i, :]
            vote = Counter()
            for t in range(self.number):
                vote_t, rD = self.trees[t].weightedVote(X, rD)
                # feature = self.features[rank[j]]
                # X = data.iloc[i, feature].values.reshape(1, -1)
                # prob = self.classifiers[rank[j]].resProb(X)[0]
                for v in vote_t:
                    vote[v] += vote_t[v]
                if rD <= 0:
                    break
            pred = vote.most_common()[0][0]
            conf_matrix[self.label_map[real], self.label_map[pred]] += 1
        return conf_matrix

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
        X = sample.iloc[self.features].values.reshape(1, -1)  # 选择对应的输入特征
        return self.clf.result(X)[0]

    def resProb(self, sample):
        X = sample.iloc[self.features].values.reshape(1, -1)
        return self.clf.resProb(X)[0]


# def selectBestClf(dataSet, ensemble, rank=1, coefficient=1.5):
#     ret = []
#     for i in range(ensemble.size):
#         X = dataSet.iloc[:, ensemble.features[i]]
#         res = ensemble.classifiers[i].result(X)
#         ret.append(res)
#     # ret M*N M:# of sample, N:# of classifier
#     ret = np.transpose(ret)
#     Y = dataSet.values[:, -1:]
#     predict_result = np.hstack((ret, Y))
#     infoGain = calInfoGain(predict_result)
#     # print(pd.DataFrame(infoGain))
#     accuracy = ensemble.accuracy(dataSet)
#     # print(pd.DataFrame(accuracy))
#     obj = 0.5 * infoGain + coefficient * np.array(accuracy)
#     # print(pd.DataFrame(obj))
#     # print(infoGain, accuracy)
#     max_obj_index = np.argmax(obj)
#     sorted_index = np.argsort(obj)
#     if accuracy[sorted_index[-rank]] < 0.5:
#         return None, 1
#     return sorted_index[-rank], obj[sorted_index[-rank]]
