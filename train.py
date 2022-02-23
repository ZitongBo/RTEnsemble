from classifier import Ensemble, Classifier
from adaboost import AdaBoost
import fs
import reader as rdr
import util as U
import random as rd
import numpy as np
import argparse
import time
import sys
import os
from environment import Environment
from collections import Counter
from importlib import import_module
from sklearn.model_selection import KFold
import selection
from config import *


def train(dataset,
          classifier,
          algorithm,
          random_state,
          ddl,
          coefficient,
          one_result=False,
          num_clf=100,
          feature_type=3,
          num_training=10000,
          learning_rate=1e-2,
          discount_factor=0.95,
          epsilon=1.0,
          portion=0.56,
          sequential=True,
          **network_kwargs):
    rd.seed(random_state)
    np.random.seed(random_state)

    start_time = time.time()
    data, task = rdr.read(dataset)
    time_cost = time.time() - start_time
    print('reading', dataset, 'dataset takes %.3f sec' % (time_cost))
    print('data shape:', data.shape)
    # shuffle dataset
    # data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    num_feature = data.shape[1] - 1
    if task == Regression:
        num_children = 10
        label_map = dict()
        for i in range(num_children):
            label_map[i] = i
    else:
        label_map = dict()
        # label_map[None] = 'N'
        for l in data.iloc[:, -1]:
            if l not in label_map:
                label_map[l] = len(label_map)
    print('number of labels: %d' % (len(label_map)))
    print('classifier type', classifier)
    print(feature_type)
    clf_types = classifier
    if clf_types == ['dt']:
        clf_type = 1
    elif clf_types == ['mlp']:
        clf_type = 2
    elif clf_types == ['knn']:
        clf_type = 3
    elif clf_types == ['nb']:
        clf_type = 4
    elif clf_types == ["svm"]:
        clf_type = 5
    elif clf_types == ['dt', 'mlp', 'knn', 'nb']:
        clf_type = 6
    else:
        clf_type = 7
    # print(clf_types)

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

    time_costs = [0.0] * 7
    fs_size = 0.0
    eprl_size = 0.0
    ibrl_size = 0.0
    avg_full_test_accu = 0.0
    avg_part_test_accu = 0.0
    stat = np.zeros((len(algorithm), 4))
    mean_MSE = np.zeros((len(algorithm)))
    mean_MAE = np.zeros((len(algorithm)))
    term = 10
    kf = KFold(n_splits=term)
    for i, (train_idx, test_idx) in enumerate(kf.split(data)):
        print('\nRunning iteration %d of %d fold...' % (i + 1, term))
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
            full_ensemble = Ensemble(num_clf, clf_types, features, label_map, task,
                                     persistence=persistence)
        else:
            full_ensemble = Ensemble(num_clf, clf_types, features, label_map, task,
                                     random_state=random_state)
            full_ensemble.train(train)
            # full_ensemble.saveClf(persistence)

        # part ensemble
        persistence = 'models/clfs/d{}n{:d}c{:d}f{:d}r{:d}/i{:d}part/'.format(
            dataset, num_clf, clf_type, feature_type, random_state, i)
        if LOAD_CLF:
            part_ensemble = Ensemble(num_clf, clf_types, features, label_map,  task,
                                     persistence=persistence)
        else:
            part_ensemble = Ensemble(num_clf, clf_types, features, label_map,  task,
                                     random_state=random_state)
            part_ensemble.train(train_clf)
            # part_ensemble.saveClf(persistence)
            # # AdaBoost
            # start_time = time.time()
            # adab = AdaBoost(num_clf, random_state)
            # print(adab.clf.n_estimators, num_clf)
            # adab.train(train)
            # full_features = list()
            # adab_classifiers = []
            # for c in adab.clf.estimators_:
            #     adab_classifiers.append(Classifier(clf=c))
            #     full_features.append(list(range(num_feature)))
            # # print(adab.clf.estimators_,num_clf)
            # adab_ensemble = Ensemble(len(adab_classifiers), clf_types, full_features, label_map,
            #                          classifiers=adab_classifiers, random_state=random_state)
            # adab_cmatrix = adab.evaluation(test, label_map)
            # adab_res = U.computeConfMatrix(adab_cmatrix)
            # for s in range(4):
            #     adab_stat[s] += adab_res[s]
            # out_model.append('adab')
            # out_res.append(adab_res)
            time_cost = time.time() - start_time
            # out_time.append(time_cost)
            # time_costs[4] += time_cost
            # print('AdaBoost takes %.3f sec' % (time_cost))
        time_cost = time.time() - start_time
        ensemble_start_time = time.time()
        noTree = 20
        print('ddl', ddl)
        coef = coefficient
        trees = []
        for t in range(noTree):
            tree1 = selection.EnsembleTree(full_ensemble, train,  rank=t + 1, coefficient=coef, one_result=one_result,)
            trees.append(tree1)
        forest = selection.EnsembleForest(trees, full_ensemble.label_map)
        print(' full_ensemble, train, coef=', coef)
        # print('forest accuracy', forest.accuracy(test, 40))
        time_costs[0] += time_cost
        print('%s ensembles takes %.3f sec' %
              ('loading' if LOAD_CLF else 'training', time_cost))
        ensemble_time = time.time() - ensemble_start_time
        print('ensemble forest takes %.3f sec' % (ensemble_time))

        # creat environment
        start_time = time.time()
        real_set = [train_ens.iloc[:, -1], test.iloc[:, -1]]
        res_set = [part_ensemble.results(train_ens), part_ensemble.results(test)]


        time_cost = time.time() - start_time
        time_costs[1] += time_cost
        print('creating environment takes %.3f sec' % (time_cost))

        # voting techniques
        if task == Classification:
            for a in range(len(algorithm)):
                alg = algorithm[a]
                start_time = time.time()
                confidence_matrix = np.zeros((len(label_map), len(label_map)))

                # majority vote
                if alg == 'mv':
                    confidence_matrix = full_ensemble.majorityVote(test)
                elif alg == 'wv':
                    confidence_matrix = full_ensemble.weightedVote(test)
                elif alg == 'rs':
                    confidence_matrix = full_ensemble.randomSelectMajorityVote(test, ddl)
                elif alg == 'ta':
                    confidence_matrix = full_ensemble.topKMajorityVote(train, test, ddl)
                elif alg == 'efmv':
                    confidence_matrix = forest.majorityVote(test, ddl)
                elif alg == 'efwv':
                    confidence_matrix = forest.weightedVote(test, ddl)
                elif alg == "dqn":
                    prob_set = [part_ensemble.resProb(train_ens), part_ensemble.resProb(test)]
                    model_folder = 'models/ibrls/d{}n{:d}c{:d}f{:d}r{:d}/'.format(
                        dataset, num_clf, clf_type, feature_type, random_state)
                    if not LOAD_IBRL and not os.path.isdir(model_folder):
                        os.makedirs(model_folder)
                    model_path = '{}/i{:d}.ibrl'.format(model_folder, i)
                    # print(model_path)
                    if LOAD_IBRL:
                        model.load(model_path)
                    else:
                        learn = get_learn_function(alg)
                        env = Environment(num_clf, real_set, res_set, prob_set, label_map, ddl)
                        model = learn(env, 0, num_training, learning_rate, epsilon,
                                      discount_factor, random_state, **network_kwargs)
                        model.save(model_path)
                    confidence_matrix, avg_clf = env.evaluation(model, 1, ddl, verbose=False)
                    ibrl_size += avg_clf
                print(alg, 'take', time.time() - start_time, 's')
                res = U.computeConfMatrix(confidence_matrix)
                for s in range(4):
                    stat[a][s] += res[s] / term
                out_res.append(res)
                time_cost = time.time() - start_time
                out_time.append(time_cost)
                time_costs[2] += time_cost
                print(alg, 'takes %.3f sec' % (time_cost))
        else:
            for a in range(len(algorithm)):
                alg = algorithm[a]
                start_time = time.time()
                MSE = 0
                MAE = 0
                if alg == 'mv':
                    MSE,MAE = full_ensemble.MSE(test)
                elif alg == 'rs':
                    # confidence_matrix = full_ensemble.randomSelectMajorityVote(test, ddl)
                    MSE, MAE = full_ensemble.rsMSE(train, test, ddl)
                elif alg == 'ta':
                    MSE,MAE = full_ensemble.topKMSE(train, test, ddl)
                elif alg == 'efmv':
                    MSE, MAE = forest.MSE(test, ddl)
                print(alg, 'MSE=', MSE)
                mean_MSE[a] += MSE / term
                mean_MAE[a] += MAE /term
        # FS
        # start_time = time.time()
        # fs_model = fs.train(num_clf, real_set[0], res_set[0])
        # fs_size += len(fs_model)
        # fs_cmatrix = fs.evaluation(fs_model, real_set[1], res_set[1], label_map)
        # # print(fs_cmatrix)
        # fs_res = U.computeConfMatrix(fs_cmatrix)
        # for s in range(4):
        #     fs_stat[s] += fs_res[s]
        # out_model.append('fs')
        # out_res.append(fs_res)
        # time_cost = time.time() - start_time
        # out_time.append(time_cost)
        # time_costs[3] += time_cost
        # print('FS takes %.3f sec' % (time_cost))
        #


        '''
        # EPRL
        start_time = time.time()

        time_cost = time.time() - start_time
        out_time.append(time_cost)
        time_costs[5] += time_cost
        print('EPRL takes %.3f sec' % (time_cost))
        '''
        # IBRL
        # start_time = time.time()
        # model_folder = 'models/ibrls/d{}n{:d}c{:d}f{:d}r{:d}/'.format(
        #     dataset, num_clf, clf_type, feature_type, random_state)
        # if not LOAD_IBRL and not os.path.isdir(model_folder):
        #     os.makedirs(model_folder)
        # model_path = '{}/i{:d}.ibrl'.format(model_folder, i)
        # # print(model_path)
        # if LOAD_IBRL:
        #     model.load(model_path)
        # else:
        #     learn = get_learn_function(algorithm)
        #     model = learn(env, 0, num_training, learning_rate, epsilon,
        #         discount_factor, random_state, **network_kwargs)
        #     model.save(model_path)
        # ibrl_cmatrix, avg_clf = env.evaluation(model, 1, verbose=False)
        # ibrl_size += avg_clf
        # # print(ibrl_cmatrix)
        # ibrl_res = U.computeConfMatrix(ibrl_cmatrix)
        # for s in range(4):
        #     ibrl_stat[s] += ibrl_res[s]
        # out_model.append('rl')
        # out_res.append(ibrl_res)
        # time_cost = time.time() - start_time
        # out_time.append(time_cost)
        # time_costs[6] += time_cost
        # U.outputs(['rl'], out_res)
        # print('IBRL takes %.3f sec' % (time_cost))

        U.outputs(algorithm, out_res)
        # print(np.mean(full_test_accu))
        # print(U.formatFloats(full_test_accu, 2) + '\n')
        # print(np.mean(part_test_accu))
        # print(U.formatFloats(part_test_accu, 2) + '\n')

    # for a in
    # mv_stat = [n / term for n in mv_stat]
    # wv_stat = [n / term for n in wv_stat]
    # fs_stat = [n / term for n in fs_stat]
    # adab_stat = [n / term for n in adab_stat]
    # eprl_stat = [n / term for n in eprl_stat]
    # ibrl_stat = [n / term for n in ibrl_stat]
    # time_costs = [n / term for n in time_costs]
    fs_size /= term
    eprl_size /= term
    ibrl_size /= term
    avg_full_test_accu /= term
    avg_part_test_accu /= term
    U.outputs(algorithm,
              stat)
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    clf_str = ""
    for c in classifier:
        clf_str += c
    result_file = open(dataset + str(coefficient) + clf_str + '.txt', 'w+')
    result_file.write('     accuracy,  precision,  recall,  f_score\n')
    for a, m in zip(algorithm, stat):
        result_file.write(str(a) + '  ')
        result_file.write(str(m[0])[:5] + ' ')
        result_file.write(str(m[1])[:5] + ' ')
        result_file.write(str(m[2])[:5] + ' ')
        result_file.write(str(m[3])[:5] + ' ')
        result_file.write('\n')

    print('time costs: C, E, V, FS, Ada, EPRL, IBRL\n       '
          + U.formatFloats(time_costs, 2))
    print('FS size: %.5f, EPRL size: %.5f, IBRL size: %.5f'
          % (fs_size, eprl_size, ibrl_size))
    print('full test avg accu: %.5f, part test avg accu: %.5f'
          % (avg_full_test_accu, avg_part_test_accu))
    if task == Regression:
        print('algorithm   MSE ddl',ddl)
        for i in range(len(algorithm)):
            print(algorithm[i],mean_MSE[i], mean_MAE[i])

def get_learn_function(alg):
    # alg_module = import_module('.'.join(['src', alg]))
    alg_module = import_module(alg)
    return alg_module.learn
