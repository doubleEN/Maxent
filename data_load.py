"""
数据加载的模块，将数据集放入列表中，每个样本的数据结构为：({"feature_1":val1,"feature_2":val...},label)
"""
import pprint

def format_sample(sample_str):
    """
    处理字符串形式的样本，将样本存储为 ({"feature_1":val1,"feature_2":val...},label)
    :param sample_str:字符串样本
    :return:({"feature_1":val1,"feature_2":val...},label)数据结构的样本
    """
    sample = sample_str.strip().split()
    feature_dict = {"feature_" + str(i): sample[i] for i in range(1, len(sample))}
    return (feature_dict, sample[0])


def get_data(data_dir, format_str):
    """
    按行加载字符文本数据
    :param data_dir:文本的路径
    :param format_str:每行文本处理的方式
    :return:返回加载完的数据集列表
    """
    return [format_str(line) for line in open(data_dir)]

if __name__ == "__main__":
    data = get_data("data/zoo.train",format_sample)
    pprint.pprint(data[0])
