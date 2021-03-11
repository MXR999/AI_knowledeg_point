import numpy as np

"""
预测一个人的性情
"""
person1 = [8.5, 34, 123, 54]
emotion = ['开心', '难过', '平静', '惊喜', '愤怒']
weights = np.random.randn(4, 4)
bias = np.random.random()
logits = np.dot(person1, weights) + bias

"""
softmax(x) 把输入x变成[0-1]之间的数字
"""


def softmax(x):
    x -= np.max(x)  # 防止exp(x)过大，导致结果溢出 nan
    return np.exp(x) / np.sum(np.exp(x))


# print(softmax(logits))


"""
cross_entropy

emotion = ['开心', '难过', '平静', '惊喜', '愤怒']
y_about_person = [0, 1, 0, 0]
"""
person2 = [85, 34, 34, 54]
y_about_person = [0, 1, 0, 0]
predicted_good = [0.1, 0.8, 0.005, 0.005]
predicted_bad = softmax(logits)  # [4.05929502e-107 3.05538703e-045 3.69645190e-028 1.00000000e+000]


# 通俗讲，就是需要用损失函数 cross_entropy() 来衡量 weights和bias的好坏
def cross_entropy(label, predicated):
    """

    :param label: 正确解的标签
    :param predicated: y=w*x+b 经 softmax() 计算过的预测结果
    :return: cross_entropy 交叉熵误差函数的结果
    """
    return -sum(label[i] * np.log(predicated[i]) for i in range(len(label)))


print(cross_entropy(y_about_person, predicted_good))
print(cross_entropy(y_about_person, predicted_bad))