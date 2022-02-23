from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from collections import Counter
from joblib import dump, load
import torch
import pandas as pd
import numpy as np
import os
from adaboost import AdaBoost
import fs
import reader as rdr
import util as U
import random as rd
import argparse
import time
from config import *
import sys
from collections import Counter

from importlib import import_module
from sklearn.model_selection import KFold


class Classifier:
    def __init__(self, classifier_type='', learner=None, C=1, **learner_kwarg):
        if learner is not None:
            self.learner = learner
        elif classifier_type == 'dt':
            self.learner = DecisionTreeClassifier(random_state=learner_kwarg['random_state'], 
                                              max_features='sqrt',
                                              max_depth=8)
        # elif classifier_type == 'rf':
        #     self.learner = RandomForestClassifier(random_state=learner_kwarg['random_state'])
        elif classifier_type == 'svm':
            self.learner = SVC(random_state=learner_kwarg['random_state'])
        elif classifier_type == 'mlp':
            self.learner = MLPClassifier(random_state=learner_kwarg['random_state'], 
                                     hidden_layer_sizes=(8,), max_iter=3000)
        elif classifier_type == 'knn':
            self.learner = KNeighborsClassifier(algorithm='brute')
        elif classifier_type == 'nb':
            self.learner = GaussianNB()
        else:
            raise ValueError('unrecognized classifier type')
        self.C = C

    def train(self, X, y):
        self.learner.fit(X, y)

    def accuracy(self, test_X, test_y):
        return self.learner.score(test_X, test_y)

    def result(self, X):
        return self.learner.predict(X)

    def label(self, X):
        return self.learner.predict(X)

    def resProb(self, X):
        return self.learner.predict_proba(X)

    def classes(self):
        return self.learner.classes_


class Regressor:
    def __init__(self, regressor_type='', rgs=None, C=1, **learner_kwarg):
        if rgs is not None:
            self.rgs = rgs
        elif regressor_type == 'dt':
            self.rgs = DecisionTreeRegressor(random_state=learner_kwarg['random_state'], max_features='sqrt',
                                                max_depth=8)
        # elif classifier_type == 'rf':
        #     self.learner = RandomForestClassifier(random_state=learner_kwarg['random_state'])
        elif regressor_type == 'svm':
            self.rgs = SVR()
        elif regressor_type == 'mlp':
            self.rgs = MLPRegressor(random_state=learner_kwarg['random_state'],
                                     hidden_layer_sizes=(42,32), max_iter=3000)
        elif regressor_type == 'nb':
            self.rgs = GaussianNB()
        elif regressor_type == 'lr':
            self.rgs = LinearRegression()
        else:
            raise ValueError('unrecognized classifier type')
        self.C = C

    def train(self, X, y):
        self.rgs.fit(X, y)

    def accuracy(self, test_X, test_y):
        # result = self.rgs(torch.from_numpy(test_X.values).to(torch.float32).to('cuda:0'))
        # acc = 0
        # for i in range(len(result)):
        #     if U.normal(test_y.values[i]) == U.normal(result[i]):
        #         acc += 1
        # return acc / len(test_y)
        return self.rgs.score(test_X, test_y)

    def result(self, X):
        # return self.rgs(torch.from_numpy(X).to(torch.float32).to('cuda:0')).cpu().detach().numpy()
        return self.rgs.predict(X)

    def label(self, X):
        result = self.rgs.predict(X)
        # result= self.rgs(torch.from_numpy(X.values).to(torch.float32).to('cuda:0'))
        for i in range(len(result)):
            result[i] = U.normal(result[i])
        return result


class Ensemble:
    def __init__(self, size, base_learner_types, features, label_map, task=Classification, persistence='', base_learners=None, **learner_kwarg):
        if len(features) != size:
            raise ValueError('length of feature does not match number of base_learners')
        self.size = size
        self.features = features
        self.label_map = label_map
        self.base_learner_types = []
        self.base_learners = []
        if os.path.isdir(persistence):
            self.loadLearner(persistence, base_learner_types)
        elif base_learners is not None:
            self.base_learners = base_learners
            self.base_learner_types = base_learner_types
        else:
            for i in range(size):
                self.base_learner_types.append(base_learner_types[i % len(base_learner_types)])
                learner_kwarg['random_state'] = rd.randint(1, 10000)
                if task == Regression:
                    self.base_learners.append(Regressor(
                        regressor_type=self.base_learner_types[i], **learner_kwarg))
                else:
                    self.base_learners.append(Classifier(
                        classifier_type=self.base_learner_types[i], **learner_kwarg))

    def train(self, data, bootstrap=True):
        for i in range(self.size):
            feature = self.features[i]
            if bootstrap:
                # this cannot be controlled by random_state need some other methods
                # indices = np.random.randint(0, data.shape[0], data.shape[0])
                indices = list()
                for _ in range(data.shape[0]):
                    indices.append(rd.randrange(0, data.shape[0]))
            else:
                indices = list(range(data.shape[0]))
            # use specified features to form X
            X = data.iloc[indices, feature]
            y = data.iloc[indices, -1]
            self.base_learners[i].train(X, y)

    def remove(self, index):
        self.base_learners.remove(self.base_learners[index])
        self.size -= 1
        self.features.remove(self.features[index])
        self.base_learner_types.remove(self.base_learner_types[index])

    def selectBestLearner(self, dataSet, rank=1, coefficient=1.5):
        ret = []
        for i in range(self.size):

            X = dataSet.iloc[:, self.features[i]]
            res = self.base_learners[i].label(X)
            # res = self.base_learners[i].label(X).cpu().detach().squeeze().numpy()
            # print(X.shape)
            ret.append(res)
            # print(res.shape)
        # ret M*N M:# of sample, N:# of classifier
        ret = np.transpose(ret)
        Y = dataSet.values[:, -1:]
        # print(ret.shape, Y.shape)
        predict_result = np.hstack((ret, Y))
        infoGain = U.calInfoGain(predict_result)
        accuracy = self.accuracy(dataSet)
        criterion = coefficient * infoGain + 1 * np.array(accuracy)
        sorted_index = np.argsort(criterion)
        return sorted_index[-rank], accuracy[sorted_index[-rank]]

    def accuracy(self, data):
        ret = []
        for i in range(self.size):
            feature = self.features[i]
            test_X = data.iloc[:, feature]
            test_y = data.iloc[:, -1]
            s = self.base_learners[i].accuracy(test_X, test_y)
            ret.append(s)
        return np.array(ret)

    def MSE(self, data):
        error = np.zeros((data.shape[0]))
        MAE = np.zeros((data.shape[0]))
        for i in range(data.shape[0]):
            real = data.iloc[i, -1]
            results = []
            for j in range(self.size):
                X = data.iloc[i, self.features[j]].values.reshape(1, -1)
                results.append(self.base_learners[j].result(X)[0])
            error[i] = (np.mean(results) - real) ** 2
            MAE[i] = abs(np.mean(results) - real)
        return np.mean(error), np.mean(MAE)



    def topKMajorityVote(self, train_data, test_data, k):
        acc = self.accuracy(train_data)
        rank = np.argsort(acc)[::-1]
        start = time.time()
        conf_matrix = np.zeros((len(self.label_map), len(self.label_map)), dtype=np.int32)
        for i in range(test_data.shape[0]):
            real = test_data.iloc[i, -1]
            vote = Counter()
            for j in range(k):
                feature = self.features[rank[j]]
                X = test_data.iloc[i, feature].values.reshape(1, -1)
                p = self.base_learners[rank[j]].result(X)[0]
                vote[p] += 1
            pred = vote.most_common()[0][0]
            conf_matrix[self.label_map[U.normal(real, 10)], self.label_map[U.normal(pred, 10)]] += 1
        # print('ta', time.time() - start)
        return conf_matrix

    def topKWeightedVote(self, data, k):
        acc = self.accuracy(data)
        rank = np.argsort(acc)[::-1]
        start = time.time()
        conf_matrix = np.zeros((len(self.label_map), len(self.label_map)), dtype=np.int32)
        for i in range(data.shape[0]):
            real = data.iloc[i, -1]
            vote = Counter()
            for j in range(k):
                feature = self.features[rank[j]]
                X = data.iloc[i, feature].values.reshape(1, -1)
                prob = self.base_learners[rank[j]].resProb(X)[0]
                for c, p in zip(self.base_learners[rank[j]].classes(), prob):
                    vote[c] += p
            pred = vote.most_common()[0][0]
            conf_matrix[self.label_map[real], self.label_map[pred]] += 1

        return conf_matrix

    def topKMSE(self, train_data, test_data, k):
        acc = self.accuracy(train_data)
        rank = np.argsort(acc)[::-1]
        error = np.zeros((test_data.shape[0]))
        MAE = np.zeros((test_data.shape[0]))
        for i in range(test_data.shape[0]):
            real = test_data.iloc[i, -1]
            results = []
            for j in range(k):
                feature = self.features[rank[j]]
                X = test_data.iloc[i, feature].values.reshape(1, -1)
                p = self.base_learners[rank[j]].result(X)[0]
                results.append(p)
            error[i] = (np.mean(results) - real) ** 2
            MAE[i] = abs(np.mean(results) - real)
        return np.mean(error), np.mean(MAE)

    def rsMSE(self, train_data, test_data, N):
        sample = np.random.choice(a=self.size, size=N, replace=False, p=None)
        error = np.zeros((test_data.shape[0]))
        MAE = np.zeros((test_data.shape[0]))
        for i in range(test_data.shape[0]):
            real = test_data.iloc[i, -1]
            results = []
            for j in range(N):
                feature = self.features[sample[j]]
                X = test_data.iloc[i, feature].values.reshape(1, -1)
                p = self.base_learners[sample[j]].result(X)[0]
                results.append(p)
            error[i] = (np.mean(results) - real) ** 2
            MAE[i] = abs(np.mean(results) - real)
        return np.mean(error), np.mean(MAE)

    def randomSelectMajorityVote(self, data, N):
        sample = np.random.choice(a=self.size, size=N, replace=False, p=None)
        start = time.time()
        conf_matrix = np.zeros((len(self.label_map), len(self.label_map)), dtype=np.int32)
        for i in range(data.shape[0]):
            real = data.iloc[i, -1]
            vote = Counter()
            for j in range(N):
                feature = self.features[sample[j]]
                X = data.iloc[i, feature].values.reshape(1, -1)
                p = self.base_learners[sample[j]].result(X)[0]
                vote[p] += 1
            pred = vote.most_common()[0][0]
            conf_matrix[self.label_map[U.normal(real, 10)], self.label_map[U.normal(pred, 10)]] += 1
        print('rs', time.time()-start)
        return conf_matrix

    def randomSelectWeightedVote(self, data, N):
        sample = np.random.choice(a=self.size, size=N, replace=False, p=None)
        conf_matrix = np.zeros((len(self.label_map), len(self.label_map)), dtype=np.int32)
        for i in range(data.shape[0]):
            real = data.iloc[i, -1]
            vote = Counter()
            for j in range(N):
                feature = self.features[sample[j]]
                X = data.iloc[i, feature].values.reshape(1, -1)
                prob = self.base_learners[sample[j]].resProb(X)[0]
                for c, p in zip(self.base_learners[sample[j]].classes(), prob):
                    vote[c] += p
            pred = vote.most_common()[0][0]
            conf_matrix[self.label_map[real], self.label_map[pred]] += 1
        return conf_matrix
    # return the matrix of result
    # each row for an instance
    # each column for a classifier

    def results(self, data):
        ret = []
        for i in range(self.size):
            X = data.iloc[:, self.features[i]]
            res = self.base_learners[i].result(X)
            ret.append(res)
        df = pd.DataFrame(ret)
        return df.T

    # return the matrix of probabilistic result
    # each element m_{i,j} is a possibility distribution
    # predicted by classifier j on instance i
    def resProb(self, data):
        ret = []
        for i in range(self.size):
            X = data.iloc[:, self.features[i]]
            prob = self.base_learners[i].resProb(X)
            # match the indices of classes in base_learners 
            # to the indices in the label map
            indices = [0] * len(self.label_map)
            for j, c in enumerate(self.base_learners[i].classes()):
                indices[self.label_map[c]] = j
            # reorder the probability distribution 
            # according to the indices in the label map
            prob = prob[:, indices]
            # reduce dimensionality
            tu_p = [tuple(p) for p in prob]
            ret.append(tuple(tu_p))
        df = pd.DataFrame(ret)
        return df.T

    def majorityVote(self, data):
        conf_matrix = np.zeros((len(self.label_map), len(self.label_map)), dtype=np.int32)
        for i in range(data.shape[0]):
            real = data.iloc[i, -1]
            vote = Counter()
            for j in range(self.size):
                feature = self.features[j]
                X = data.iloc[i, feature].values.reshape(1, -1)
                p = self.base_learners[j].result(X)[0]
                vote[p] += 1
            pred = vote.most_common()[0][0]
            conf_matrix[self.label_map[U.normal(real, 10)], self.label_map[U.normal(pred, 10)]] += 1
        return conf_matrix

    def weightedVote(self, data):
        conf_matrix = np.zeros((len(self.label_map), len(self.label_map)), dtype=np.int32)
        for i in range(data.shape[0]):
            real = data.iloc[i, -1]
            vote = Counter()
            for j in range(self.size):
                feature = self.features[j]
                X = data.iloc[i, feature].values.reshape(1, -1)
                prob = self.base_learners[j].resProb(X)[0]
                for c, p in zip(self.base_learners[j].classes(), prob):
                    vote[c] += p
            pred = vote.most_common()[0][0]
            conf_matrix[self.label_map[real], self.label_map[pred]] += 1
        return conf_matrix

    def loadLearner(self, persistence, types):
        if not os.path.isdir(persistence):
            raise IOError('No such directory')
        for i in range(self.size):
            learner = load('{}{:d}.learner'.format(persistence, i))
            self.base_learner_types.append(types[i % len(types)])
            self.base_learners.append(Classifier(learner=learner))

    def saveLearner(self, persistence):
        if not os.path.isdir(persistence):
            os.makedirs(persistence)
        for i in range(self.size):
            dump(self.base_learners[i].learner, '{}{:d}.learner'.format(persistence, i))

