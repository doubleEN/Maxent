"""
最大熵模型的定义
"""

import numpy as np


class MaxEnt(object):
    def __init__(self, data_set, iters_weight=200, iters_delta=50, convergence_condition=0.001):
        self.data_set = data_set
        self.sample_sum = len(self.data_set)
        self.iters_weight = iters_weight
        self.iters_delta = iters_delta
        self.convergence_condition = convergence_condition
        self.__init_paras()

    def __init_paras(self):
        """
        由数据集初始化部分参数
        """
        import collections
        self.feature_functions = collections.defaultdict(np.int8)
        self.W = collections.defaultdict(np.float32)
        label_set = set()
        for sample in self.data_set:
            label_set.add(sample[1])
            for feature in sample[0].keys():
                self.feature_functions[(feature, sample[0][feature], sample[1])] += 1
        self.labels = list(label_set)
        self.feature_functions_num = len(self.feature_functions)
        self.__calc_ep_()

    def __calc_ep_(self):
        """
        计算特征函数关于联合概率经验分布的期望
        """
        self.ep_ = {feat_fun: self.feature_functions[feat_fun] / self.sample_sum for feat_fun in
                    self.feature_functions.keys()}

    def __calc_Z(self, unk_sample):
        """
        计算最大熵模型条件概率的规范化因子
        :param unk_sample:未标注的样本
        :return:当前样本的规范化因子值
        """
        return np.exp(np.sum(
            [self.W[(feature, unk_sample[feature], label)] for label in self.labels for feature in unk_sample.keys()]))

    def __calc_probs_lables_given_sample(self, unk_sample):
        """
        计算未标注样本的条件概率
        :param unk_sample:未标注的样本
        :return:未标注样本的属于各个label的条件概率
        """
        return np.array([self.__calc_prob_lables_given_sample(unk_sample, label) for label in self.labels])

    def __calc_prob_lable_given_sample(self, unk_sample, label):
        return np.exp(
            np.sum([self.W[(feature, unk_sample[feature], label)] for feature in unk_sample.keys()])) / self.__calc_Z(
            unk_sample)

    def build_model(self):
        for iter in range(self.iters_weight):
            self.W = {feat_fun_key: self.W[feat_fun_key] + self.__iter_delta(feat_fun_key) for feat_fun_key in
                      self.W.keys()}

    def __iter_delta(self, feature_fun_key):
        delta = 0.0
        for iter in range(self.iters_delta):
            newton_g = 0.0
            newton_g_derivative = 0.0
            for sample in self.data_set:
                if sample[0].has_key(feature_fun_key[0]) and sample[0][feature_fun_key[0]] == feature_fun_key[1]:
                    pound_sign_count = self.__pound_sign_count(sample)
                    val = self.__calc_prob_lable_given_sample(sample[0], feature_fun_key[2]) * np.exp(
                        delta * pound_sign_count)
                    newton_g += val
                    newton_g_derivative += val * pound_sign_count
            newton_g = self.ep_[feature_fun_key] - newton_g / self.sample_sum
            if np.abs(newton_g) < 1e-7:
                return delta
            newton_g_derivative = -newton_g_derivative / self.sample_sum
            ratio = newton_g / newton_g_derivative
            delta -= ratio
            if ratio < self.convergence_condition:
                return delta

        raise Exception("未在迭代周期内收敛")

    def __pound_sign_count(self, sample):
        return np.sum(
            1 for feature in sample[0].keys() if self.feature_functions[feature, sample[0][feature], sample[1]] != 0)

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
    model.build_model()
    import pprint
    pprint.pprint(model.data_set)
    pprint.pprint(model.labels)
    pprint.pprint(model.sample_sum)
    pprint.pprint(model.feature_functions)
    pprint.pprint(model.feature_functions_num)
    pprint.pprint(model.iters_weight)
    pprint.pprint(model.iters_delta)
    pprint.pprint(model.ep_)
