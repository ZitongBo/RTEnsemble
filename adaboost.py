from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class AdaBoost:

    def __init__(self, num_clf, random_state):
        self.clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=8),
                                      n_estimators=num_clf, 
                                      random_state=random_state)

    def train(self, data):
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        self.clf.fit(X, y)

    def evaluation(self, data, label_map):
        conf_matrix = np.zeros((len(label_map), len(label_map)), dtype=np.int32)
        X = data.iloc[:, :-1]
        prediction = self.clf.predict(X)
        for i in range(data.shape[0]):
            real = data.iloc[i, -1]
            pred = prediction[i]
            conf_matrix[label_map[real], label_map[pred]] += 1
        return conf_matrix