from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from collections import Counter
from joblib import dump, load
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
import selection
from importlib import import_module
from sklearn.model_selection import KFold


class Classifier:

    def __init__(self, classifier_type='', clf=None, C=1, **clf_kwarg):
        if clf is not None:
            self.clf = clf
        elif classifier_type == 'dt':
            self.clf = DecisionTreeClassifier(random_state=clf_kwarg['random_state'], 
                                              max_features='sqrt',
                                              max_depth=8)
        # elif classifier_type == 'rf':
        #     self.clf = RandomForestClassifier(random_state=clf_kwarg['random_state'])
        # elif classifier_type == 'svm':
        #     self.clf = SVC(random_state=clf_kwarg['random_state'])
        elif classifier_type == 'mlp':
            self.clf = MLPClassifier(random_state=clf_kwarg['random_state'], 
                                     hidden_layer_sizes=(8,))
        elif classifier_type == 'knn':
            self.clf = KNeighborsClassifier(algorithm='brute')
        elif classifier_type == 'nb':
            self.clf = GaussianNB()
        else:
            raise ValueError('unrecognized classifier type')
        self.C = C

    def train(self, X, y):
        self.clf.fit(X, y)

    def accuracy(self, test_X, test_y):
        return self.clf.score(test_X, test_y)

    def result(self, X):
        return self.clf.predict(X)

    def resProb(self, X):
        return self.clf.predict_proba(X)

    def classes(self):
        return self.clf.classes_


class Ensemble:

    def __init__(self, size, types, features, label_map, persistence='', **clf_kwarg):
        if len(features) != size:
            raise ValueError('length of feature does not match number of classifiers')
        self.size = size
        self.features = features
        self.label_map = label_map
        self.clf_types = []
        self.classifiers = []
        if os.path.isdir(persistence):
            self.loadClf(persistence, types)
        else:
            for i in range(size):
                self.clf_types.append(types[i % len(types)])
                clf_kwarg['random_state'] = rd.randint(1, 10000)
                self.classifiers.append(Classifier(
                    classifier_type=self.clf_types[i], **clf_kwarg))

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
            self.classifiers[i].train(X, y)

    def remove(self, clf):
        self.classifiers.remove(self.classifiers[clf])
        self.size -= 1
        self.features.remove(self.features[clf])
        self.clf_types.remove(self.clf_types[clf])


    def accuracy(self, data):
        ret = []
        for i in range(self.size):
            feature = self.features[i]
            test_X = data.iloc[:, feature]
            test_y = data.iloc[:, -1]
            s = self.classifiers[i].accuracy(test_X, test_y)
            ret.append(s)
        return ret

    def randomSelect(self, data, N):
        sample = np.random.randint(0, self.size, N)
        test_X = data.iloc[:, :-1]
        test_y = data.iloc[:, -1].values
        result = []
        for s in range(len(data)):
            result_s = []
            for t in sample:
                temp = self.label_map[self.classifiers[t].result(test_X.iloc[s:s + 1, self.features[t]])[0]]
                result_s.append(temp)
            result.append(result_s)
        vote = Counter()
        major_vote = []
        # print(result)
        for s in result:
            for i in s:
                vote[i] += 1
            pred = vote.most_common()[0][0]
            major_vote.append(pred)
        temp = 0
        # print('mv',major_vote)
        # print('testy',test_y)
        # print( )
        for i in range(len(data)):
            # print('i',i, major_vote[i])
            # print(self.label_map[test_y[i]])
            if major_vote[i] == self.label_map[test_y[i]]:
                temp += 1
        return temp / len(data)

    # return the matrix of result
    # each row for an instance
    # each column for a classifier
    def results(self, data):
        ret = []
        for i in range(self.size):
            X = data.iloc[:, self.features[i]]
            res = self.classifiers[i].result(X)
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
            prob = self.classifiers[i].resProb(X)
            # match the indices of classes in classifiers 
            # to the indices in the label map
            indices = [0] * len(self.label_map)
            for j, c in enumerate(self.classifiers[i].classes()):
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
                p = self.classifiers[j].result(X)[0]
                vote[p] += 1
            pred = vote.most_common()[0][0]
            conf_matrix[self.label_map[real], self.label_map[pred]] += 1
        return conf_matrix

    def weightedVote(self, data):
        conf_matrix = np.zeros((len(self.label_map), len(self.label_map)), dtype=np.int32)
        for i in range(data.shape[0]):
            real = data.iloc[i, -1]
            vote = Counter()
            for j in range(self.size):
                feature = self.features[j]
                X = data.iloc[i, feature].values.reshape(1, -1)
                prob = self.classifiers[j].resProb(X)[0]
                for c, p in zip(self.classifiers[j].classes(), prob):
                    vote[c] += p
            pred = vote.most_common()[0][0]
            conf_matrix[self.label_map[real], self.label_map[pred]] += 1
        return conf_matrix

    def loadClf(self, persistence, types):
        if not os.path.isdir(persistence):
            raise IOError('No such directory')
        for i in range(self.size):
            clf = load('{}{:d}.clf'.format(persistence, i))
            self.clf_types.append(types[i % len(types)])
            self.classifiers.append(Classifier(clf=clf))

    def saveClf(self, persistence):
        if not os.path.isdir(persistence):
            os.makedirs(persistence)
        for i in range(self.size):
            dump(self.classifiers[i].clf, '{}{:d}.clf'.format(persistence, i))

