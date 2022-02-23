from reader import *
import numpy as np
import math
import time
from copy import deepcopy
from classifier import Classifier
from collections import Counter
import util as U


class EnsembleTree:
    def __init__(self, ensemble, val_set, rank, coefficient, one_result=False, task=Classification):
        self.label_map = ensemble.label_map
        self.coefficient = coefficient
        self.nodeNum = 1
        self.task = task
        self.one_result = one_result
        best_learner_index, _ = ensemble.selectBestLearner(val_set, rank, self.coefficient)
        root_learner = ensemble.base_learners[best_learner_index]
        root_feature = ensemble.features[best_learner_index]
        children = []
        for i in range(len(self.label_map)):
            children.append(None)
        self.root = EnsembleTreeNode(root_learner, children, root_feature)

        curr = self.root
        ensemble_root = deepcopy(ensemble)
        ensemble_root.remove(best_learner_index)
        # splitData = np.zeros((len(self.label_map), 1, len(val_set.columns)))
        data_dict = dict()
        for d in range(0, len(val_set)):
            # print(curr.predictLabel(val_set.iloc[d, :]), d, self.label_map)
            label = self.label_map[curr.predictLabel(val_set.iloc[d, :])]
            if label not in data_dict:
                data_dict[label] = val_set.iloc[d:d+1, :]
            else:
                data_dict[label] = data_dict[label].append(val_set.iloc[d:d+1, :])
            # print(label, data_dict[label].shape)
        children = []

        for i in range(len(self.label_map)):
            if i in data_dict:
                best_learner = self.constructTreeNode(data_dict[i], ensemble_root)
                children.append(best_learner)
            else:
                children.append(None)
        curr.children = children

    def constructTreeNode(self, data_set, ensemble, ):
        best_learner_index, criterion = ensemble.selectBestLearner(data_set, 1, self.coefficient)
        if criterion < 0.52:
            return None
        self.nodeNum += 1
        # print('obj',obj, 'index',best_learner_index)
        # print('ensemble size:', ensemble.size, ' data', len(data_set))
        # print()
        root_learner = ensemble.base_learners[best_learner_index]
        root_feature = ensemble.features[best_learner_index]
        self.nodeNum += 1
        children = []
        for i in range(len(self.label_map)):
            children.append(None)
        treeNode = EnsembleTreeNode(root_learner, children, root_feature)

        if criterion > 0.9:
            return treeNode
        ensemble_root = deepcopy(ensemble)
        ensemble_root.remove(best_learner_index)
        # splitData = np.zeros((len(self.label_map), 1, len(val_set.columns)))
        data_dict = dict()
        for d in range(0, len(data_set)):
            label = self.label_map[treeNode.predictLabel(data_set.iloc[d, :])]
            if label not in data_dict:
                data_dict[label] = data_set.iloc[d:d + 1, :]
            else:
                data_dict[label] = data_dict[label].append(data_set.iloc[d:d + 1, :])
                import random
                for key in data_dict.keys():
                    if random.randint(0, 10) > 7:
                        data_dict[key] = data_dict[key].append(data_set.iloc[d:d + 1, :])

                # print(label, data_dict[label].shape)
        for i in range(len(self.label_map)):
            if i in data_dict and len(data_dict[i]) > 10 and ensemble.size > 20:
                best_learner = self.constructTreeNode(data_dict[i], ensemble_root)
                treeNode.children[i] = best_learner
        # print(EnsembleTree.nodeNum)
        return treeNode

    def predict(self, sample, D):
        curr = self.root
        results = []
        while curr is not None:
            if curr.learner.C > D:
                break
            result = curr.predict(sample)

            y = self.label_map[curr.predictLabel(sample)]
            if self.one_result:
                results = [y]
            else:
                results.append(result)
            #
            D -= curr.learner.C
            curr = curr.children[y]
        return results, D

    def weightedVote(self, sample, D):
        if self.task == Regression:
            raise Exception("Weighted vote is only available for classification task")
        vote = Counter()
        curr = self.root
        rD = D
        while curr is not None:
            if curr.learner.C > rD:
                break
            prob = curr.resProb(sample)
            for c, p in zip(curr.learner.classes(), prob):
                vote[c] += p
            rD -= curr.learner.C
            curr = curr.children[self.label_map[curr.predict(sample)]]
        return vote, rD

    def majorityVote(self, sample, D):
        vote = Counter()
        curr = self.root
        rD = D
        while curr is not None:
            if curr.learner.C > rD:
                break
            y = curr.predict(sample)
            if not self.one_result:
                vote[y] += 1
            rD -= curr.learner.C
            start = time.time()
            curr = curr.children[self.label_map[y]]
            # print('one model', time.time() - start, 's')
        if self.one_result:
            vote[y] += 1
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

    def MSE(self, data, D):
        # print(data)
        error = np.zeros((data.shape[0]))
        MAE = np.zeros((data.shape[0]))
        for i in range(data.shape[0]):
            real = data.iloc[i, -1]
            rD = D
            X = data.iloc[i, :]
            results = []
            for t in range(self.number):
                result, rD = self.trees[t].predict(X, rD)
                results.extend(result)
                if rD <= 0:
                    break
            # print('re',results)
            # print('real',real,'pred',np.mean(results))
            error[i] = (np.mean(results) - real) ** 2
            MAE[i] = abs(np.mean(results) - real)
        return np.mean(error), np.mean(MAE)

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
    def __init__(self, learner, children, features):
        self.learner = learner  # 学习器
        self.children = children
        self.features = features

    def predict(self, sample):
        X = sample.iloc[self.features].values.reshape(1, -1)  # 选择对应的输入特征
        return self.learner.result(X)[0]

    def predictLabel(self, sample):

        X = sample.iloc[self.features].values.reshape(1, -1)
        if self.learner.__class__ == Classifier:
            return self.learner.result(X)[0]
        else:
            return U.normal(self.learner.result(X)[0], 10, 2)

    def resProb(self, sample):
        X = sample.iloc[self.features].values.reshape(1, -1)
        return self.learner.resProb(X)[0]


# def selectBestClf(data_set, ensemble, rank=1, coefficient=1.5):
#     ret = []
#     for i in range(ensemble.size):
#         X = data_set.iloc[:, ensemble.features[i]]
#         res = ensemble.classifiers[i].result(X)
#         ret.append(res)
#     # ret M*N M:# of sample, N:# of classifier
#     ret = np.transpose(ret)
#     Y = data_set.values[:, -1:]
#     predict_result = np.hstack((ret, Y))
#     infoGain = calInfoGain(predict_result)
#     # print(pd.DataFrame(infoGain))
#     accuracy = ensemble.accuracy(data_set)
#     # print(pd.DataFrame(accuracy))
#     obj = 0.5 * infoGain + coefficient * np.array(accuracy)
#     # print(pd.DataFrame(obj))
#     # print(infoGain, accuracy)
#     max_obj_index = np.argmax(obj)
#     sorted_index = np.argsort(obj)
#     if accuracy[sorted_index[-rank]] < 0.5:
#         return None, 1
#     return sorted_index[-rank], obj[sorted_index[-rank]]
