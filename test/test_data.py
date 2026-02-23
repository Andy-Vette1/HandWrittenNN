from models.fnn import FNN
from src.data_loader import DataLoader
from src.paths import MNIST_TRAIN_CSV
import numpy as np

loader = DataLoader(str(MNIST_TRAIN_CSV))
loader.load()
loader.normalize()
X, y = loader.get_data()

print("第一个样本的标签是:", np.argmax(y[0]))
print("第一个样本的像素矩阵均值:", np.mean(X[0]))

# 初始化模型
input_size = 784
hidden_size = 128
output_size = 10
model = FNN(input_size, hidden_size, output_size)

# 试着跑一下前向传播 (只取前 2 个样本试试)
fake_output = model.forward(X[:2])

print("前向传播输出形状:", fake_output.shape)
print("输出的概率和 (应该接近1):", np.sum(fake_output, axis=1))
