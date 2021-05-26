import numpy as np
import numpy.random as rd
from config import *


class RTEnsemble:
    def __init__(self,):
        self.no_processor = 0
        self.learners = load_learners()
        self.results = []
        self.x = 0
        self.D = 0

    def reset(self, x):
        self.D = rd.randint(MIN_DEADLINE, MAX_DEADLINE)
        self.results = np.zeros(len(self.learners))
        self.x = x

    def step(self, action):
        if action != DONE:
            model = self.learners[action]
            C = rd.randn()*model.C_std + model.C
            result = model.get_result(self.x)
            self.D -= C
            if self.D >= 0:
                self.results[action] = result
        done = True if action == DONE or self.D <= 0 else False
        reward = self.get_result() if done else 0
        next_state = self.get_state()
        return next_state, reward, done

    def get_state(self):
        state = np.append(self.results, np.array([self.D]))
        return state

    def get_result(self):
        results = np.delete(self.results, np.where(self.results == 0), 0)
        return np.mean(results)


class Learner:
    def __init__(self, model, C, C_std, err):
        self.model = model
        self.C = C
        self.C_std = C_std
        self.err = err

    def get_result(self, x):
        result = self.model.p
        return result

def load_learners():
    learners = []
    return learners