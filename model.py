"""
最大熵模型的定义
"""

import numpy as np


class MaxEnt(object):
    def __init__(self, data_set, iter_num=200, convergence_condition=0.001):
        self.data_set = data_set
        self.sample_sum = len(self.data_set)
        self.iter_num = iter_num
        self.convergence_condition = convergence_condition
        self.__init_paras()

    def __init_paras(self):
        """
        由数据集初始化部分参数
        """
        import collections
        self.feature_functions = collections.defaultdict(int)
        self.W = collections.defaultdict(int)
        label_set = set()
        for sample in self.data_set:
            label_set.add(sample[1])
            for feature in sample[0].keys():
                self.feature_functions[(feature, sample[0][feature], sample[1])] += 1
        self.labels = list(label_set)
        self.feature_functions_num = len(self.feature_functions)

    def __calc_ep_(self):
        """
        计算特征函数关于联合概率经验分布的期望
        """
        self.ep_ = {feat_fun: self.feature_functions[feat_fun] / self.sample_sum for feat_fun in
                    self.feature_functions.items()}

    def __calc_Z(self, unk_sample):
        """
        计算最大熵模型条件概率的规范化因子
        :param unk_sample:未标注的样本
        :return:当前样本的规范化因子值
        """
        return np.exp(np.sum(
            [self.W[(feature, unk_sample[feature], label)] for label in self.labels for feature in unk_sample.keys()]))

    def __calc_prob_lables_given_sample(self, unk_sample):
        """
        计算未标注样本的条件概率
        :param unk_sample:未标注的样本
        :return:未标注样本的属于各个label的条件概率
        """
        prob_y = np.array([])
        for label in self.labels:
            prob_y.append(np.exp(np.sum(
                [self.W[(feature, unk_sample[feature], label)] for feature in unk_sample.keys()])) / self.__calc_Z(
                unk_sample))
        return prob_y

    def build_model(self):
        return

    def predict(self, unk_sample):
        """
        预测样本的类别
        :param unk_sample:未标注样本
        :return:label值
        """
        return self.labels[np.argmax(self.__calc_prob_lables_given_sample(unk_sample))]


if __name__ == "__main__":
    import data_load

    data = data_load.get_data("data/zoo.train", data_load.format_sample)
    model = MaxEnt(data)
    import pprint

    pprint.pprint(model.feature_functions)
