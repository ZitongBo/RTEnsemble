from train import *

if __name__ == '__main__':
    # train('ha', 'dt', ['mv', 'wv', 'rs', 'ta', 'efmv', 'efwv'], 3, 0)
    #train('cmc', 'mlp', ['mv', 'wv', 'rs', 'ta', 'efmv', 'efwv'], 3)
    #train('ha', 'dt', ['mv', 'wv', 'rs', 'ta', 'efmv', 'efwv'], 3)
    train('ha', ['dt'], ['dqn','mv', 'wv', 'rs', 'ta', 'efmv', 'efwv'], 1, 1)

# 'abalone': abalone, 4177,11,28 y
# 'audio': audiology, 226,95,24 y
# 'avila': avila, 20867,11,12 y
# 'bc': breast_cancer, 286，16，2 n 准确率1
# 'bw': breast_w, 699,10,2 n 准确率1
# 'cmc': cmc, 1473,22,3 dt、mlp、knn 混合
# 'credit': credit_card, 30000,31,2
# 'demat': dematology,  366,35,6 dt:n mlp:y
# 'ecoli': ecoli, 336,8,8 dt:y mlp:y
# 'glass': glass, 216,10,6 n
# 'heart': heart, 270,21,2
# 'hepa': hepatitis, 155,20,2 n mlp:y
# 'ha': human_activity, 10299,562,6 y
# 'iris': iris, 300,5,3 n 准确率太高
# 'lymph': lymphography 148,39,4
