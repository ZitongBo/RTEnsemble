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
from importlib import import_module
from sklearn.model_selection import KFold


class Classifier:

    def __init__(self, classifier_type='', clf=None, **clf_kwarg):
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

    def accuracy(self, data):
        ret = []
        for i in range(self.size):
            feature = self.features[i]
            test_X = data.iloc[:, feature]
            test_y = data.iloc[:, -1]
            s = self.classifiers[i].accuracy(test_X, test_y)
            ret.append(s)
        return ret

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


def train(dataset,
          algorithm,
          random_state,
          num_clf=100,
          num_training=10000,
          learning_rate=0.1,
          discount_factor=1.0,
          epsilon=1.0,
          portion=0.56,
          sequential=True,
          **network_kwargs):
    rd.seed(random_state)
    np.random.seed(random_state)

    start_time = time.time()
    data = rdr.read(dataset)
    time_cost = time.time() - start_time
    print('reading data takes %.3f sec' % (time_cost))
    print('data shape:', data.shape)
    # shuffle dataset
    # data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    num_feature = data.shape[1] - 1
    label_map = dict()
    # label_map[None] = 'N'
    for l in data.iloc[:, -1]:
        if l not in label_map:
            label_map[l] = len(label_map)
    print('number of labels: %d' % (len(label_map)))

    clf_type = 1
    if clf_type == 1:
        clf_types = ['dt']
    elif clf_type == 2:
        clf_types = ['mlp']
    elif clf_type == 3:
        clf_types = ['knn']
    elif clf_type == 4:
        clf_types = ['nb']
    elif clf_type == 5:
        clf_types = ['dt', 'mlp', 'knn', 'nb']
    # print(clf_types)

    feature_type = 3
    features = list()
    for i in range(num_clf):
        if feature_type == 1:
            features.append(list(range(num_feature)))
        elif feature_type == 2:
            features.append(rd.choices(list(range(num_feature)),
                                       k=int(np.ceil(num_feature * 0.5))))
        elif feature_type == 3:
            # every 1/3 classifiers get 1/3 features
            first_cut = int(np.ceil(num_feature / 3))
            second_cut = int(np.ceil(num_feature / 3 * 2))
            index = int((num_clf - 1) / 3) + 1
            if i < index:
                features.append(list(range(first_cut)))
            elif i < 2 * index:
                features.append(list(range(first_cut, second_cut)))
            else:
                features.append(list(range(second_cut, num_feature)))
    # print(features)

    mv_stat = [0.0] * 4
    wv_stat = [0.0] * 4
    fs_stat = [0.0] * 4
    adab_stat = [0.0] * 4
    eprl_stat = [0.0] * 4
    ibrl_stat = [0.0] * 4
    time_costs = [0.0] * 7
    fs_size = 0.0
    eprl_size = 0.0
    ibrl_size = 0.0
    avg_full_test_accu = 0.0
    avg_part_test_accu = 0.0

    term = 10
    kf = KFold(n_splits=term)
    for i, (train_idx, test_idx) in enumerate(kf.split(data)):
        print('\nRunning iteration %d of 10 fold...' % (i + 1))
        out_model = []
        out_res = []
        out_time = []
        train = data.iloc[train_idx, :]
        test = data.iloc[test_idx, :]
        train_clf, train_ens = rdr.splitByPortion(train, portion, random_state)
        # print(train_clf.shape, train_ens.shape, test.shape)

        # train or load ensembles
        start_time = time.time()
        # full ensemble
        persistence = 'models/clfs/d{}n{:d}c{:d}f{:d}r{:d}/i{:d}full/'.format(
            dataset, num_clf, clf_type, feature_type, random_state, i)
        if LOAD_CLF:
            full_ensemble = Ensemble(num_clf, clf_types, features, label_map,
                                     persistence=persistence)
        else:
            full_ensemble = Ensemble(num_clf, clf_types, features, label_map,
                                     random_state=random_state)
            full_ensemble.train(train)
            full_ensemble.saveClf(persistence)
        # part ensemble
        persistence = 'models/clfs/d{}n{:d}c{:d}f{:d}r{:d}/i{:d}part/'.format(
            dataset, num_clf, clf_type, feature_type, random_state, i)
        if LOAD_CLF:
            part_ensemble = Ensemble(num_clf, clf_types, features, label_map,
                                     persistence=persistence)
        else:
            part_ensemble = Ensemble(num_clf, clf_types, features, label_map,
                                     random_state=random_state)
            part_ensemble.train(train_clf)
            part_ensemble.saveClf(persistence)
        time_cost = time.time() - start_time
        time_costs[0] += time_cost
        print('%s ensembles takes %.3f sec' %
              ('loading' if LOAD_CLF else 'training', time_cost))

        # creat environment
        # start_time = time.time()
        real_set = [train_ens.iloc[:, -1], test.iloc[:, -1]]
        res_set = [part_ensemble.results(train_ens), part_ensemble.results(test)]
        prob_set = [part_ensemble.resProb(train_ens), part_ensemble.resProb(test)]
        # # env = Environment(num_clf, real_set, res_set, prob_set, label_map)
        # time_cost = time.time() - start_time
        # time_costs[1] += time_cost
        # print('creating environment takes %.3f sec' % (time_cost))

        # get the performance of basic classifiers
        # full ensemble
        full_test_accu = full_ensemble.accuracy(test)
        avg_full_test_accu += np.mean(full_test_accu)
        # part ensemble
        part_test_accu = part_ensemble.accuracy(test)
        avg_part_test_accu += np.mean(part_test_accu)

        # voting techniques
        start_time = time.time()
        # majority vote
        mv_cmatrix = full_ensemble.majorityVote(test)
        # print(mv_cmatrix)
        mv_res = U.computeConfMatrix(mv_cmatrix)
        for s in range(4):
            mv_stat[s] += mv_res[s]
        out_model.append('mv')
        out_res.append(mv_res)
        # weighted vote
        wv_cmatrix = full_ensemble.weightedVote(test)
        # print(wv_cmatrix)
        wv_res = U.computeConfMatrix(wv_cmatrix)
        for s in range(4):
            wv_stat[s] += wv_res[s]
        out_model.append('wv')
        out_res.append(wv_res)
        time_cost = time.time() - start_time
        out_time.append(time_cost)
        time_costs[2] += time_cost
        print('voting takes %.3f sec' % (time_cost))

        # FS
        start_time = time.time()
        fs_model = fs.train(num_clf, real_set[0], res_set[0])
        fs_size += len(fs_model)
        fs_cmatrix = fs.evaluation(fs_model, real_set[1], res_set[1], label_map)
        # print(fs_cmatrix)
        fs_res = U.computeConfMatrix(fs_cmatrix)
        for s in range(4):
            fs_stat[s] += fs_res[s]
        out_model.append('fs')
        out_res.append(fs_res)
        time_cost = time.time() - start_time
        out_time.append(time_cost)
        time_costs[3] += time_cost
        print('FS takes %.3f sec' % (time_cost))

        # AdaBoost
        start_time = time.time()
        adab = AdaBoost(num_clf, random_state)
        adab.train(train)
        adab_cmatrix = adab.evaluation(test, label_map)
        adab_res = U.computeConfMatrix(adab_cmatrix)
        for s in range(4):
            adab_stat[s] += adab_res[s]
        out_model.append('adab')
        out_res.append(adab_res)
        time_cost = time.time() - start_time
        out_time.append(time_cost)
        time_costs[4] += time_cost
        print('AdaBoost takes %.3f sec' % (time_cost))

        '''
        # EPRL
        start_time = time.time()

        time_cost = time.time() - start_time
        out_time.append(time_cost)
        time_costs[5] += time_cost
        print('EPRL takes %.3f sec' % (time_cost))

        # IBRL
        start_time = time.time()
        model_folder = 'models/ibrls/d{}n{:d}c{:d}f{:d}r{:d}/'.format(
            dataset, num_clf, clf_type, feature_type, random_state)
        if not LOAD_IBRL and not os.path.isdir(model_folder):
            os.makedirs(model_folder)
        model_path = '{}/i{:d}.ibrl'.format(model_folder, i)
        # print(model_path)
        if LOAD_IBRL:
            model.load(model_path)
        else:
            learn = get_learn_function(algorithm)
            model = learn(env, 0, num_training, learning_rate, epsilon, 
                discount_factor, random_state, **network_kwargs)
            model.save(model_path)
        ibrl_cmatrix, avg_clf = env.evaluation(model, 1, verbose=False)
        ibrl_size += avg_clf
        # print(ibrl_cmatrix)
        ibrl_res = U.computeConfMatrix(ibrl_cmatrix)
        for s in range(4):
            ibrl_stat[s] += ibrl_res[s]
        out_model.append('rl')
        out_res.append(ibrl_res)
        time_cost = time.time() - start_time
        out_time.append(time_cost)
        time_costs[6] += time_cost
        print('IBRL takes %.3f sec' % (time_cost))
        '''
        U.outputs(out_model, out_res)
        print(np.mean(full_test_accu))
        print(U.formatFloats(full_test_accu, 2) + '\n')
        print(np.mean(part_test_accu))
        print(U.formatFloats(part_test_accu, 2) + '\n')

    mv_stat = [n / term for n in mv_stat]
    wv_stat = [n / term for n in wv_stat]
    fs_stat = [n / term for n in fs_stat]
    adab_stat = [n / term for n in adab_stat]
    eprl_stat = [n / term for n in eprl_stat]
    ibrl_stat = [n / term for n in ibrl_stat]
    time_costs = [n / term for n in time_costs]
    fs_size /= term
    eprl_size /= term
    ibrl_size /= term
    avg_full_test_accu /= term
    avg_part_test_accu /= term
    U.outputs(['mv', 'wv', 'fs', 'adab', 'eprl', 'ibrl'],
              [mv_stat, wv_stat, fs_stat, adab_stat, eprl_stat, ibrl_stat])
    print('time costs: C, E, V, FS, Ada, EPRL, IBRL\n       '
          + U.formatFloats(time_costs, 2))
    print('FS size: %.5f, EPRL size: %.5f, IBRL size: %.5f'
          % (fs_size, eprl_size, ibrl_size))
    print('full test avg accu: %.5f, part test avg accu: %.5f'
          % (avg_full_test_accu, avg_part_test_accu))