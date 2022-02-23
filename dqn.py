import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random as rd
from collections import Counter
from environment import Action
import util as U


debug_print = False
use_gpu = True


class Net(nn.Module):
    def __init__(self, input_size, hiddens, output_size):
        super(Net, self).__init__()
        # using dropout layers
        self.layers = nn.Sequential()
        # self.layers.add_module('in-drop', nn.Dropout(p=0.8))
        for i, hidden in enumerate(hiddens):
            self.layers.add_module('dense' + str(i), nn.Linear(input_size, hidden))
            self.layers.add_module('relu' + str(i), nn.ReLU())
            # self.layers.add_module('lrelu' + str(i), nn.LeakyReLU(negative_slope=0.01))
            # self.layers.add_module('dropout' + str(i), nn.Dropout(p=0.3))
            self.layers.add_module('dropout' + str(i), nn.Dropout(p=0.5))
            # self.layers.add_module('dropout' + str(i), nn.Dropout(p=0.8))
            input_size = hidden
        self.layers.add_module('out-dense', nn.Linear(input_size, output_size))
        # self.layers = nn.ModuleList()
        # for hidden in hiddens:
        #     self.layers.append(nn.Linear(input_size, hidden))
        #     input_size = hidden
        # self.layers.append(nn.Linear(input_size, output_size))

    def forward(self, x):
        return self.layers(x)
        # for layer in self.layers[:-1]:
        #     x = F.leaky_relu(layer(x), negative_slope=0.1)
        #     # x = F.relu(layer(x))
        # # return F.hardtanh(self.layers[-1](x), min_val=0)
        # return self.layers[-1](x)


class DQN:
    def __init__(self, env, hiddens, learning_rate, discount_factor):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        input_size = env.num_clf * len(env.label_map)
        self.num_act = env.num_clf + 1
        if torch.cuda.is_available() and use_gpu:
            print('using gpu')
            self.device = torch.device('cuda')
        else:
            print('using cpu')
            self.device = torch.device('cpu')
        # self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.net = Net(input_size, hiddens, self.num_act).to(self.device)
        # for param in self.net.parameters():
        #     print(param.data)
        self.criterion = nn.MSELoss()
        # self.criterion = nn.MSELoss(reduction='sum')
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001)

    def policy(self, state, randomness=0.0):
        actions = self.env.legal_actions(state)
        if len(actions) == 0:
            return None
        # exploration
        if rd.random() < randomness:
            return rd.choice(actions)
        # exploitation
        q_values = self.qValues(state)
        act_val = Counter()
        for a in actions:
            act_val[a] = q_values[a.index]
        return act_val.most_common()[0][0]

    def qValue(self, state):
        actions = self.env.legal_actions(state)
        if len(actions) == 0:
            return 0.0
        q_values = self.qValues(state)
        q_vs = list()
        for a in actions:
            q_vs.append(q_values[a.index])
        return max(q_vs)

    def qValues(self, state):
        self.net.eval()
        batch_in = torch.tensor(state.getPred(one_hot=True), dtype=torch.float32).to(self.device)
        q_values = self.net(batch_in).detach().cpu().numpy()
        return q_values

    def train(self, sample, verbose=False):
        self.net.train()
        if verbose:
            print('\tsample size: %d' % (len(sample)))
        # construct training X and y
        X = list()
        y = list()
        oh = list()
        for s in sample:
            # state: s[0], action: s[1], state_p: s[2], reward: s[3]
            if verbose:
                print('\n\ts :', str(s[0]), '-> action: ', str(s[1]))
                print('\ts\':', str(s[2]), '-> reward: ', s[3])
            # get target q value for action
            target_q = s[3] + self.discount_factor * self.qValue(s[2])
            # update with learning rate
            # next_q = self.qValue(s[2])
            # current_q = self.qValue(s[0])
            # target_q = current_q + self.learning_rate * (s[3] + self.discount_factor * next_q - current_q)
            X.append(s[0].getPred(one_hot=True))
            y.append(target_q)
            oh.append(s[1].index)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        # compute output tensor for observed action
        oh = np.eye(self.num_act)[oh]
        oh = torch.tensor(oh, dtype=torch.float32).to(self.device)
        y_ = torch.sum(self.net(X) * oh, dim=1)
        # start training
        self.optimizer.zero_grad()
        loss = self.criterion(y_, y)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))


def learn(env, in_set, num_training, learning_rate, epsilon, discount_factor, random_state, **network_kwargs):
    log_freq = 10000
    update_freq = 1000
    model = DQN(env, (256, 256), learning_rate, discount_factor)
    num_ins = env.numInstance(in_set)

    # shouldn't be here
    res_model = DQN(env, (256, 256), learning_rate, discount_factor)
    min_avg_clf = float('inf')
    max_accu = float('-inf')
    max_accu_episode = 0

    # compute loss for eaily termination
    min_loss = float('inf')
    min_loss_episode = 0
    losses = list()
    # static sampling
    sample_portion = 0.5
    # dynamic sampling
    # shrink_rate = 0.1 # logrithmic
    # shrink_rate = 0.0000005 # polynomial
    # max_point = 1000 # polynomial

    # dynamic exploration rate
    exploration = epsilon
    exploration = 0.5
    fading_rate = 0.999992

    sample = list()
    for i in range(1, num_training + 1):
        verbose = False
        # dynamic sampling
        # sample_portion = shrink_rate * np.log(i) # logrithmic
        # sample_portion = shrink_rate * (2 * max_point - min(max_point, i)) * min(max_point, i) # polynomial
        in_row = (i - 1) % num_ins
        state = env.initState()
        history = list()
        while state is not None:
            action = model.policy(state, randomness=exploration)
            state_p, reward = env.step(state, action, in_set, in_row)
            history.append((state, action, state_p, reward))
            state = state_p
        sample.append(history[-1])
        sample.extend(rd.choices(history[:-1], k=int(np.ceil(len(history) * sample_portion))))
        history.clear()

        if i % update_freq == 0:
            # update DQN
            loss = model.train(sample, verbose=(verbose and debug_print))
            losses.append(loss)
            if loss < min_loss:
                min_loss = loss
                min_loss_episode = i
            sample.clear()
        if i % log_freq == 0:
            # training log
            # print('\nfinished epoch: %d:\nexploration: %.3f, sampling: %.3f\nlosses: %s' 
            #       % (i, exploration, sample_portion, U.formatFloats(losses, 3)))
            rl_cmatrix, avg_clf = env.evaluation(model, 1, 20, verbose=False)
            rl_res = U.computeConfMatrix(rl_cmatrix)
            # U.outputs(['rl'], [rl_res])
            losses.clear()
            # shouldn't be here
            if rl_res[0] > max_accu or (rl_res[0] == max_accu and avg_clf < min_avg_clf):
                max_accu = rl_res[0]
                max_accu_episode = i
                min_avg_clf = avg_clf
                res_model.net.load_state_dict(model.net.state_dict())

        # dynamic exploration rate
        exploration *= fading_rate
        exploration = max(exploration, 0.1) # keep at least 0.1 exploration rate
    rl_cmatrix, _ = env.evaluation(res_model, 0, 20, verbose=False)
    rl_res = U.computeConfMatrix(rl_cmatrix)
    U.outputs(['max0'], [rl_res])
    #rl_cmatrix, _ = env.evaluation(res_model, 3, verbose=False)
    #rl_res = U.computeConfMatrix(rl_cmatrix)
    #U.outputs(['max3'], [rl_res])
    rl_cmatrix, _ = env.evaluation(res_model, 1, verbose=False)
    rl_res = U.computeConfMatrix(rl_cmatrix)
    U.outputs(['max1'], [rl_res])
    # print('\nmin loss: %.3f, episode: %d\nmax accu: %.3f, episode: %d\n'
    #       % (min_loss, min_loss_episode, max_accu, max_accu_episode))
    # return model
    # shouldn't be here
    return res_model






