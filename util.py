import math
import numpy as np


def computeConfMatrix(conf_matrix):
    total_count = conf_matrix.sum()
    correct = 0
    precision = 0.0
    recall = 0.0
    f_score = 0.0
    r_count = 1e-5
    f_count = 1e-5
    for i in range(conf_matrix.shape[0]):
        tp = conf_matrix[i, i]
        tp_fp = conf_matrix.sum(axis=0)[i]
        tp_fn = conf_matrix.sum(axis=1)[i]
        if tp > 0:
            correct += tp
            # normalized by the portion of true label
            precision += float(tp) / float(tp_fp) * float(tp_fn) / float(total_count)
            recall += float(tp) / tp_fn
            f_score += float(2 * tp) / (tp_fp + tp_fn)
        if tp_fn > 0:
            r_count += 1
        if tp_fp + tp_fn > 0:
            f_count += 1
    accuracy = float(correct) / total_count
    # precision /= conf_matrix.shape[0]
    recall /= r_count
    f_score /= f_count
    return accuracy, precision, recall, f_score


def outputs(algs, measurements):
    if len(algs) == 0:
        return
    print('      accuracy, precision, recall, f_score')
    for a, m in zip(algs, measurements):
        print('{:>4}: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(a, m[0], m[1], m[2], m[3]))
    print()

def formatFloats(nums, digits):
    res = ''
    for n in nums:
        s = str(n)
        res += s[:s.find('.') + digits + 1] + ', '
    return res[:-2]


def calInfoGain(dataSet):
    numFeatures = len(dataSet[0]) - 1
    # 计数数据集的香农熵
    baseEntropy = calcShannonEnt(dataSet)
    # 信息增益
    bestInfoGain = 0.0
    # 最优特征的索引值
    bestFeature = -1
    infoGain = np.zeros(numFeatures)
    for i in range(numFeatures):
        # 获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        # 创建set集合{}，元素不可重复
        uniqueVals = set(featList)
        # 经验条件熵
        newEntropy = 0.0
        # 计算信息增益
        for value in uniqueVals:
            # subDataSet划分后的子集
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算子集的概率
            prob = len(subDataSet) / float(len(dataSet))
            # 根据公式计算经验条件熵
            newEntropy += prob * calcShannonEnt((subDataSet))
        # 信息增益
        # infoGain[i] = baseEntropy - newEntropy
        infoGain[i] = - newEntropy
        # 打印每个特征的信息增益
        # print("第%d个特征的增益为%.3f" % (i, infoGain[i]))
        # 计算信息增益
        # if (infoGain > bestInfoGain):
        #     # 更新信息增益，找到最大的信息增益
        #     bestInfoGain = infoGain
        #     # 记录信息增益最大的特征的索引值
        #     bestFeature = i
        #     # 返回信息增益最大特征的索引值
    return infoGain


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec = np.hstack((featVec[:axis], featVec[axis+1:]))
            retDataSet.append(reducedFeatVec)
    return retDataSet


def calcShannonEnt(dataSet):
    # 返回数据集行数
    numEntries = len(dataSet)
    # 保存每个标签（label）出现次数的字典
    labelCounts = {}
    # 对每组特征向量进行统计
    for featVec in dataSet:
        currentLabel = featVec[-1]                     # 提取标签信息
        if currentLabel not in labelCounts.keys():   # 如果标签没有放入统计次数的字典，添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1                 # label计数

    shannonEnt=0.0                                   # 经验熵
    # 计算经验熵
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries      # 选择该标签的概率
        shannonEnt -= prob * math.log(prob, 2)                 # 利用公式计算
    return shannonEnt                                # 返回经验熵


# 处理预测值
def normal(data, max_value=10, task=1):
    if task == 1:
        return data
    if data < 0:
        return 0
    elif data > max_value - 1:
        return max_value - 1
    else:
        return round(data)
