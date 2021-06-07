from collections import Counter
import numpy as np


def train(num_clf, real_set, res_set):
	curr_accu = 0
	n = len(real_set)
	curr_ensemble = list()
	unused = set(range(num_clf))
	for _ in range(num_clf):
		best_clf = Counter()
		for c in unused:
			curr_ensemble.append(c)
			correct = 0
			for row in range(len(real_set)):
				pred = mjVote(res_set, row, curr_ensemble)
				real = real_set.iloc[row]
				if pred == real:
					correct += 1
			accu = correct / n
			if accu >= curr_accu:
				best_clf[c] = accu
			curr_ensemble.pop(-1)
		if len(best_clf) == 0:
			break
		c, a = best_clf.most_common()[0]
		unused.remove(c)
		curr_ensemble.append(c)
		curr_accu = a
	return curr_ensemble


def evaluation(model, real_set, res_set, label_map):
	conf_matrix = np.zeros((len(label_map), len(label_map)), dtype=np.int32)
	for row in range(len(real_set)):
		real = real_set.iloc[row]
		pred = mjVote(res_set, row, model)
		conf_matrix[label_map[real], label_map[pred]] += 1
	return conf_matrix


def mjVote(res_set, in_row, in_cols):
	preds = res_set.iloc[in_row, in_cols]
	vote = Counter()
	for p in preds:
		vote[p] += 1
	return vote.most_common()[0][0]
