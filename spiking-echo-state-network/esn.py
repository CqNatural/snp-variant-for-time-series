# coding=utf-8

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


class EchoStateNetwork:
    def __init__(self, input_size,
                 output_size,
                 reservoir_size,
                 leaking_rate=0.3,
                 spectral_radius=0.9,
                 input_scaling=1):

        self.input_size = input_size
        self.output_size = output_size
        self.reservoir_size = reservoir_size
        # 输入向量的取值存疑，是不是有上下区间
        # 输入向量额外有1个偏移量basis=1（默认）
        self.W_in = (np.random.rand(reservoir_size,
                                    1 + input_size) - 0.5) * input_scaling  # shape=(r,i+1)

        # 设置储备池权重并稀疏化
        self.W_res = np.random.rand(reservoir_size, reservoir_size) - 0.5
        rhoW = max(abs(linalg.eig(self.W_res)[0]))
        self.W_res *= 1.25 / rhoW  # 设置谱半径，这里用的是原文的“直接缩放”，esnp用的是“归一化缩放”

        # 设置储备池状态
        self.current_x = np.zeros((reservoir_size))  # shape=(1000,20000)
        # 分配储备池状态的时序记录表，每列的长度有偏移值长度+输入长度u(t)+储备状态长度x(t)
        self.X = np.zeros((1 + input_size + reservoir_size,
                           train_len - init_len))
        # Target矩阵，和X矩阵的配套关系决定了预测关系，留空
        self.Yt = None

        # 输出矩阵留空
        self.W_out = None

    def train(self, input_data, target_data, init_len):

        # 记录表的格式（不包括空转部分），每一列是一个状态样本
        X = np.zeros((1 + self.input_size + self.reservoir_size, train_len - init_len))
        # 每一列是一个状态样本，列的数目和X是一样的
        # 这里要手动截掉空转的那部分数据
        Yt = target_data[:, init_len:]

        # x是个列向量，x.shape=(1000, 1)
        x = np.zeros((self.reservoir_size, 1))
        # print(f"x.shape={x.shape}")
        for t in range(input_data.shape[1]):
            u = input_data[:, t].reshape((self.input_size, 1))  # 重塑成列向量，否则默认是个1d的行向量，无法拼接
            # print("u:")
            # print(u)
            # print(f"x.shape={x.shape}")
            # print(f"u.shape={u.shape}")
            x = self.run_reservoir(u, x)
            # run_reservoir(u, x):
            # x_t = (1 - a) * x + a * np.tanh(np.dot(self.W_in, np.vstack((1, u))) +
            #                                 np.dot(self.W_res, x))
            if t >= init_len:
                X[:, t - init_len] = np.vstack((1, u, x))[:, 0]

            # print(f"np.vstack((1, u, x))[:, 0]={np.vstack((1, u, x))[:, 0]}")

        # print(f"init_len={init_len}")
        # print(X.shape)
        # print(X[:10, 0])
        # print(Yt.shape)
        # print(Yt[:, 0])
        reg = 1e-6
        reg = None
        X_T = X.T
        # 这段回归分析的代码还要再研究研究
        if reg is not None:
            # use ridge regression
            # 如果用linalg.solve()函数无法处理多维输出Y的情况
            Wout = np.dot(np.dot(Yt, X_T), linalg.inv(np.dot(X, X_T) + \
                                                      reg * np.eye(1 + self.input_size + self.reservoir_size)))
        else:
            # use pseudo inverse
            # 这个准确率也很高
            Wout = np.dot(Yt, linalg.pinv(X))
        # Wout.shape=(8, 60) 其中8就是单纯的输出向量有8个元素，60=resSize+inSize+basis
        self.current_x = x
        self.X = X
        self.Yt = Yt
        self.W_out = Wout
        print(f"W_out.shape={self.W_out.shape}")
        a = X[:, 150].reshape((self.reservoir_size + 1 + self.input_size, 1))
        b = Yt[:, 150].reshape(self.output_size, 1)
        print(f"a.shape={a.shape}")
        print(f"b.shape={b.shape}")
        print(np.dot(self.W_out, a) - b)

    def test(self, input_data, target_data, init_len):
        pass

    def predict(self, input_data):
        pass

    def generate(self, input_data):
        pass

    # 可以批量运行t行数据，然后返回一个储备池记录[x(0),...,x(t)]
    # x是启动时的储备池状态
    # 返回值是x的列向量
    def run_reservoir(self, u, x):
        # print(f"u.shape={u.shape}")
        x_t = (1 - a) * x + a * np.tanh(np.dot(self.W_in, np.vstack((1, u))) +
                                        np.dot(self.W_res, x))
        # print(f"x_t.shape={x_t.shape}")
        return x_t


def set_seed(seed=None):
    if seed is None:
        seed = int((time.time() * 10 ** 6) % 4294967295)  # 默认用日期做种子
    else:
        np.random.seed(seed)
    return seed


# 配置超参数
# 数据集参数
train_len = 200  # 训练集样本长度
test_len = 200  # 测试集样本长度
init_len = 10  # 空转步长
# dataset_name = "./datasets/MackeyGlass_t17.txt"  # 单变量数据集，10000个样本
dataset_name = "./datasets/Multivariate.txt"  # 多变量数据集，一行5个，90行样本
# 网络参数
np_seed = 42  # 用于Numpy的随机种子
input_size = 5  # 输入向量长度
output_size = 5  # 输出向量长度
reservoir_size = 1000  # 储备池神经元数目
spectral_radius = 0.9  # 谱半径
input_scaling = 1  # 输入尺度，默认为1，普通不用改
a = 0.3  # 学习率（储备池更新比重）

if __name__ == "__main__":
    # 加载数据
    # 加载完毕的数据集，一列代表一个样本，data.shape=(5, 200)代表有200个样本，每列样本5个分量（行项）。这么做是为了方便矩阵运算
    data = np.loadtxt(dataset_name)  # data.shape=(720, 5)，720行数据样本，每行5个分量
    if len(data.shape) == 1:
        # 把数据从(1000, )变成(1000,1)
        data = data.reshape(len(data), 1)  # 如何是单变量数据集，那么需要这一行
    # 截取训练数据
    train_data = data[:train_len].T  # data.shape=(5, 200)
    # 截取拟合目标样本
    train_target = data[1:train_len + 1, :].T  # data.shape=(5, 200)
    (in_size, out_size) = (train_data.shape[0], train_data.shape[0])

    # 展示一部分样例数据（前1000个样本）
    plt.figure(10).clear()
    plt.plot(data[:1000, :], label="data")
    plt.title('A sample of data')
    # plt.show()

    # 加载随机种子
    # seed = set_seed(np_seed)
    seed = set_seed(None)

    # 初始化网络
    esn = EchoStateNetwork(in_size, out_size, reservoir_size)
    esn.train(train_data, train_target, init_len)

    print("End.")
