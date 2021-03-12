import numpy as np
from sklearn.datasets import load_boston

# 数据加载
data = load_boston()
x_ = data['data']  # x_ (506, 13)
y = data['target']
y = y.reshape(y.shape[0], 1)  # y (506, 1)

# 数据规范化
x_ = (x_ - np.mean(x_, axis=0)) / np.std(x_, axis=0)

n_features = x_.shape[1]  # 13
n_hidden = 10
w1 = np.random.randn(n_features, n_hidden)  # w1(13, 10)
b1 = np.zeros(n_hidden)  # b1(10,)
w2 = np.random.randn(n_hidden, 1)  # w2(10, 1)
b2 = np.zeros(1)  # b2(1,)


# relu
def relu(x):
    result = np.where(x < 0, 0, x)
    return result


learning_rate = 1e-6


def MSE_loss(y, y_hat):
    return np.mean(np.square(y_hat - y))


def Linear(x, w1, b1):
    return np.dot(x, w1) + b1


# 迭代5000次
for i in range(5000):
    # 前向传播，计算预测值y
    l1 = Linear(x_, w1, b1)  # l1 (506, 10)
    s1 = relu(l1)  # s1 (506, 10)
    y_pred = Linear(s1, w2, b2)  # y_pred (506, 1)

    # 计算loss
    loss = MSE_loss(y, y_pred)

    # 反向传播
    grad_y_pred = 2.0 * (y_pred - y)  # grad_y_pred (506, 1)
    grad_w2 = s1.T.dot(grad_y_pred)  # grad_w2 (10, 1)
    grad_temp_relu = grad_y_pred.dot(w2.T)  # (506, 10)
    grad_temp = grad_temp_relu.copy()
    grad_temp_relu[l1 < 0] = 0
    grad_w1 = x_.T.dot(grad_temp)

    # 更新权重
    w1 -= grad_w1 * learning_rate
    w2 -= grad_w2 * learning_rate
print("w1={} \n w2={}".format(w1, w2))
