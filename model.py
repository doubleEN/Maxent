"""
最大熵模型的定义
"""

import numpy as np
import collections
import pprint


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
        feat_fun_count = collections.defaultdict(np.int8)
        self.W = collections.defaultdict(np.float32)
        label_set = set()
        for sample in self.data_set:
            label_set.add(sample[1])
            for feature in sample[0].keys():
                feat_fun_count[(feature, sample[0][feature], sample[1])] += 1
        self.labels = list(label_set)
        self.feature_functions_num = len(feat_fun_count)
        self.feat_fun_id = collections.defaultdict(int)
        self.ep_ = []
        id = 0
        for feat_fun, count in feat_fun_count.items():
            self.feat_fun_id[feat_fun] = id
            id += 1
            self.ep_.append(count / self.sample_sum)
        self.W = [0.0] * self.feature_functions_num

    def build_model(self):
        """
        开始建模
        """
        for iter in range(self.iters_weight):
            for feat_fun_key, id in self.feat_fun_id.items():
                self.W[id] = self.W[id] + self.__iter_delta(feat_fun_key)
            print("ITERATIONS_" + str(iter), self.W)
        print("feature_fun_num", self.feature_functions_num)

    def __iter_delta(self, feature_fun_key):
        """
        使用牛顿法迭代得到权重改变量
        :param feature_fun_key:某权重的特征函数
        :return:权重的改变量
        """
        id = self.feat_fun_id[feature_fun_key]
        delta = 0.0
        for iter in range(self.iters_delta):
            newton_g = 0.0
            newton_g_derivative = 0.0
            for sample in self.data_set:
                if sample[0][feature_fun_key[0]] == feature_fun_key[1]:
                    pound_sign_count = self.__pound_sign_count(sample)
                    val = self.__calc_prob_lable_given_sample(sample[0], feature_fun_key[2]) * np.exp(
                        delta * pound_sign_count)
                    newton_g += val
                    newton_g_derivative += val * pound_sign_count
            newton_g = self.ep_[id] - newton_g / self.sample_sum
            if np.abs(newton_g) < 1e-7:
                return delta
            newton_g_derivative = -newton_g_derivative / self.sample_sum
            ratio = newton_g / newton_g_derivative
            delta -= ratio
            if ratio < self.convergence_condition:
                return delta

        raise Exception("未在迭代周期内收敛")

    def __pound_sign_count(self, sample):
        """
        计算样本的f#
        :param sample:样本
        :return:f#值
        """
        return np.sum(
            1 for feature in sample[0].keys() if
            self.feat_fun_id[(feature, sample[0][feature], sample[1])] >= 0)

    def __calc_prob_lable_given_sample(self, unk_sample, label):
        """
        当前模型下，给定未知样本，预测它为指定类别的概率
        :param unk_sample:未知样本
        :param label:指定类别
        :return:预测概率
        """
        return np.exp(
            np.sum([self.W[self.feat_fun_id[(feature, unk_sample[feature], label)]] for feature in unk_sample.keys() if
                    (feature, unk_sample[feature], label) in self.feat_fun_id.keys()])) / self.__calc_Z(
            unk_sample)

    def __calc_Z(self, unk_sample):
        """
        计算最大熵模型条件概率的规范化因子
        :param unk_sample:未标注的样本
        :return:当前样本的规范化因子值
        """
        return np.sum(np.exp(
            np.sum(self.W[self.feat_fun_id[(feature, unk_sample[feature], label)]] for feature in unk_sample.keys() if
                   (feature, unk_sample[feature], label) in self.feat_fun_id.keys()))
                      for label in self.labels)

    def __calc_probs_lables_given_sample(self, unk_sample):
        """
        计算未标注样本的条件概率
        :param unk_sample:未标注的样本
        :return:未标注样本的属于各个label的条件概率
        """
        return np.array([self.__calc_prob_lable_given_sample(unk_sample, label) for label in self.labels])

    def predict(self, unk_sample):
        """
        预测样本的类别
        :param unk_sample:未标注样本
        :return:label值
        """
        return self.labels[np.argmax(self.__calc_probs_lables_given_sample(unk_sample))]

if __name__ == "__main__":
    import data_load

    data = data_load.get_data("data/zoo2.train", data_load.format_sample)
    model = MaxEnt(data)
    model.build_model()

    test_data = data_load.get_data("data/zoo.test", data_load.format_sample)

    precies=[]
    for unk_sample,label in test_data:
        precies.append(model.predict(unk_sample)==label)
    print(np.sum(precies)/len(precies))