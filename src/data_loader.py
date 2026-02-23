import numpy as np
import pandas
import os
from src.paths import MNIST_TRAIN_CSV


class DataLoader:
    def __init__(self, data_path):
        """
        初始化加载器
        :param data_path: 数据集路径 (支持 .csv 或 .csv.zip)
        """
        # 这里做一个简单的检查，防止路径写错找不到文件
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"找不到文件: {data_path}")

        self.data_path = data_path
        self.X = None
        self.y = None

    def load(self):
        print(f"正在加载数据: {self.data_path} ...")
        # pandas 会自动处理 zip 压缩格式
        data = pandas.read_csv(self.data_path)

        # 将 pandas dataframe 转换为 numpy 数组
        data_numpy = data.to_numpy()

        # 第一列是标签 (label)，剩下的 784 列是像素 (pixels)
        # 取所有行，第0列
        self.y = data_numpy[:, 0]
        # 取所有行，第1列到最后
        self.X = data_numpy[:, 1:]

        print(f"数据加载完毕! 形状: X={self.X.shape}, y={self.y.shape}")

    def normalize(self):
        """
        归一化：将像素值从 0-255 映射到 0-1
        """
        print("正在归一化数据...")
        # 转换数据类型为 float32 以便进行除法运算，防止溢出
        self.X = self.X.astype(np.float32)
        self.X /= 255.0

    def get_one_hot_y(self, num_classes=10):
        """
        将数字标签转换为 One-Hot 向量
        例如: 标签 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        这是手动实现反向传播的必要步骤
        """

        # 创建一个全 0 的矩阵 [样本数, 10]
        one_hot = np.zeros((self.y.shape[0], num_classes))

        # 这里的技巧：利用 numpy 的高级索引
        # 对于第 i 行，我们在 self.y[i] 指定的列位置填入 1
        for i in range(len(self.y)):
            one_hot[i, self.y[i]] = 1

        return one_hot


    def get_data(self):
        """
        返回处理好的 X 和 One-Hot 编码后的 y
        """
        y_one_hot = self.get_one_hot_y()
        return self.X, y_one_hot


if __name__ == '__main__':
    loader = DataLoader(str(MNIST_TRAIN_CSV))
    loader.load()
    loader.get_one_hot_y()
    X, y = loader.get_data()
    print("第一个样本的标签是:", np.argmax(y[0]))
    print("第一个样本的像素矩阵均值:", np.mean(X[0]))
