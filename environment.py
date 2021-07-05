from collections import Counter
import random as rd
import numpy as np
import util as U


class State:

    def __init__(self, size, label_map, pred=None, oh_pred=None, visited=None):
        self.size = size
        self.label_map = label_map
        if pred is None:
            self.pred = np.full((size,), None, dtype=object)
        else:
            self.pred = pred
        if oh_pred is None:
            self.oh_pred = np.zeros(size * len(label_map), dtype=int)
        else:
            self.oh_pred = oh_pred
        if visited is None:
            self.visited = set()
        else:
            self.visited = visited

    def setPred(self, index, val):
        if index not in range(self.size):
            raise ValueError('index out of bound')
        self.visited.add(index)
        if val is not None:
            self.pred[index] = val
            oh_index = index * len(self.label_map) + self.label_map[val]
            self.oh_pred[oh_index] = 1
            # self.oh_pred[index] = 1  # thats the stupid mistake!!!

    def getPred(self, one_hot=False):
        if one_hot:
            return self.oh_pred
        else:
            return self.pred

    def usedClf(self):
        return self.visited

    def copy(self):
        ret = State(self.size, self.label_map, 
                    pred=np.copy(self.pred), 
                    oh_pred=np.copy(self.oh_pred), 
                    visited=self.visited.copy())
        return ret

    def eval_proba(self):
        c = Counter([p for p in self.pred if p is not None])
        if (len(c) == 0):
            return rd.choice(list(self.label_map.keys()))
        return (c.most_common()[0][0], c.most_common()[0][1] / len(self.visited))

    def evaluation(self):
        c = Counter([p for p in self.pred if p is not None])
        if (len(c) == 0):
            return rd.choice(list(self.label_map.keys()))
        return c.most_common()[0][0]

    def __str__(self):
        return ' '.join(str(self.label_map[e]) if e is not None else 'N' for e in self.pred)


class Action:

    def __init__(self, index, visit=None):
        self.index = index
        self.visit = visit

    def __str__(self):
        if self.index == -1:
            return 'E'
        elif self.visit is None:
            return 'V' + str(self.index)
        else:
            return ('V' if self.visit else 'S') + str(self.index)


class Environment():
    # read in training and test results, including probabilistic results
    # whether act sequentially, label map, and feature related info
    def __init__(self,
                 num_clf,
                 real_set,
                 res_set, 
                 prob_set, 
                 label_map,
                 ddl,
                 sequential=True,
                 features=None, 
                 feature_cost=None):
        self.num_clf = num_clf
        self.sequential = sequential
        self.real_set = real_set
        self.res_set = res_set
        self.prob_set = prob_set
        self.label_map = label_map
        self.ddl = ddl

    # return an initial blank state
    def initState(self):
        return State(self.num_clf, self.label_map)

    # return number of instance in a dataset
    def numInstance(self, index):
        return self.real_set[index].shape[0]

    # perform state transition s, a -> s', r
    def step(self, state, action, in_set, in_row,  deter=True):
        if deter:
            state_p = state.copy()
            reward = 0.0
            if action.index >= 0 and len(state_p.visited) < self.ddl:
                # get result of target classifier and perform transition
                val = None
                # check action.visit is None for non-sequential case
                if action.visit is None or action.visit:
                    val = self.res_set[in_set].iloc[in_row, action.index]
                state_p.setPred(action.index, val)
            else:
                # evaluation and get reward
                state_p = None
                pred, prob = state.eval_proba()
                if pred == self.real_set[in_set].iloc[in_row]:
                    reward = prob
                else:
                    reward = 0.0
                # pred = state.evaluation()
                # if pred == self.real_set[in_set].iloc[in_row]:
                #     reward = 1.0
                # else:
                #     reward = 0.0
        else:
            # TODO: nondeterministic state transition using probabilistic predictions
            pass
        return (state_p, reward)

    # get possible actions
    def legal_actions(self, state):
        if state is None:
            return []
        if self.sequential:
            used = sorted(state.usedClf())
            index = used[-1] + 1 if len(used) > 0 else 0
            if index == self.num_clf:
                return [Action(-1)]
            else:
                return [Action(index, visit=True), Action(index, visit=False)]
        else:
            unused = set(list(range(self.num_clf))) - state.usedClf()
            ret = [Action(index) for index in unused]
            if len(ret) < self.num_clf:
                ret.append(Action(-1))
            return ret

    # return the confusion matrix of a given model
    def evaluation(self, model, in_set, deter=True, verbose=False):
        conf_matrix = np.zeros((len(self.label_map), len(self.label_map)), dtype=np.int32)
        if deter:
            avg_clf = 0
            for in_row in range(len(self.res_set[in_set])):
                actions = []
                if verbose:
                    print('\ntest case %d' % (in_row))
                state = self.initState()
                while state is not None:
                    action = model.policy(state)
                    actions.append(str(action))

                    if verbose:
                        q_values = model.qValues(state)
                        print('\tstate: %s\n\t%s\n\taction: %s' % (str(state), U.formatFloats(q_values, 2), str(action)))
                    if action.index == -1 or len(state.visited) == self.ddl:
                        pred = state.evaluation()
                        real = self.real_set[in_set].iloc[in_row]
                        conf_matrix[self.label_map[real], self.label_map[pred]] += 1
                    state_p, reward = self.step(state, action, in_set, in_row)
                    state = state_p
                avg_clf += len(actions) - 1
                if verbose:
                    print('\t', actions, 
                          '\n\t# clfs:', len(actions) - 1, 
                          ', real:', self.label_map[real], 
                          ', pred:', self.label_map[pred])
            avg_clf /= len(self.res_set[in_set])
            # print('%.2f classifiers used' % (avg_clf))
        else:
            # TODO: nondeterministic state transition using probabilistic predictions
            pass
        return (conf_matrix, avg_clf)

