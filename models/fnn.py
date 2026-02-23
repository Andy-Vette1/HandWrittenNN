import numpy as np
import pickle


class FNN:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化网络参数
        self.params = {}

        # 1. 初始化第一层权重 W1 和偏置 b1
        # 使用 He Initialization: np.random.randn(...) * np.sqrt(2 / input_size)
        self.params['W1'] = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.params['b1'] = np.zeros((1, hidden_size))

        # 2. 初始化第二层权重 W2 和偏置 b2
        # 继续使用 He Initialization
        self.params['W2'] = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.params['b2'] = np.zeros((1, output_size))

    def relu(self, Z):
        # ReLU 就是把小于 0 的变成 0，大于 0 的保持不变
        # 提示：用 np.maximum
        return np.maximum(0, Z)  # 帮你填了一个，感受一下

    def softmax(self, Z):
        # Softmax 公式: exp(Z) / sum(exp(Z))
        # 1. 减去最大值，防止 exp 爆炸 (数值稳定性技巧)
        shift_Z = Z - np.max(Z, axis=1, keepdims=True)

        # 2. 计算指数
        exp_Z = np.exp(shift_Z)

        # 3. 计算分母：每一行的和
        # axis=1 代表把这一行的 10 个数字加起来
        sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)

        # 4. 算出概率
        return exp_Z / sum_exp_Z

    def forward(self, X):
        # 1. Layer 1
        Z1 = np.dot(X, self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)

        # 2. Layer 2
        Z2 = np.dot(A1, self.params['W2']) + self.params['b2']
        A2 = self.softmax(Z2)

        # 关键点：我们需要把中间结果 Z1, A1 保存起来！
        # 因为反向传播求导时需要用到它们
        self.cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'X': X}

        return A2

    def backward(self, X, y, learning_rate=0.1):
        """
           反向传播算法
            :param X: 输入数据
            :param y: 真实标签 (One-Hot编码)
            :param learning_rate: 学习率
        """
        # 0. 准备工作
        m = X.shape[0]
        # 从 cache 中取出前向传播存好的中间变量
        # 如果你之前没有把 params 存入 cache，这里直接用 self.params
        W2 = self.params['W2']
        A1 = self.cache['A1']
        Z1 = self.cache['Z1']

        # 1. 第二层梯度 (Output Layer)
        A2 = self.cache['A2']
        dZ2 = A2 - y  # 交叉熵 + Softmax 的导数就是这么简洁

        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # 2. 第一层梯度 (Hidden Layer)
        # 2.1 误差回传：dZ2 -> dA1
        dA1 = np.dot(dZ2, W2.T)

        # 2.2 激活函数求导 (ReLU derivative)
        # 只有 Z1 > 0 的地方才有梯度，其他地方归零
        dZ1 = dA1 * (Z1 > 0)

        # 2.3 算出第一层权重的梯度
        # 这里才是你刚才想写的 X.T !
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # 3. 更新参数 (Gradient Descent)
        # W = W - learning_rate * dW
        self.params['W1'] -= learning_rate * dW1
        self.params['b1'] -= learning_rate * db1
        self.params['W2'] -= learning_rate * dW2
        self.params['b2'] -= learning_rate * db2

    def save(self, filename):
        """
        将模型参数保存到文件
        """
        print(f"正在保存模型到 {filename} ...")
        with open(filename, 'wb') as f:
            pickle.dump(self.params, f)
        print("保存成功！")

    def load(self, filename):
        """
        从文件加载模型参数
        """
        print(f"正在加载模型 {filename} ...")
        with open(filename, 'rb') as f:
            self.params = pickle.load(f)
        print("加载成功！")