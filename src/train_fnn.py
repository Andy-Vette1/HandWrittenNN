from models.fnn import FNN
from src.data_loader import DataLoader
from src.paths import MNIST_TEST_CSV, MNIST_TRAIN_CSV, FNN_CHECKPOINT
import numpy as np

# 1. 加载数据
loader = DataLoader(str(MNIST_TRAIN_CSV))
loader.load()
loader.normalize()
X, y = loader.get_data()

# 2. 初始化模型
input_size = 784
hidden_size = 128
output_size = 10
learning_rate = 0.1
epochs = 100 # 先跑 100 轮试试
batch_size = 64


model = FNN(input_size, hidden_size, output_size)

# print("开始训练...")
# for i in range(epochs):
#     # 这里我们先偷个懒，用全量数据训练 (Batch Size = 60000)
#     # 虽然慢，但是代码简单，用来验证逻辑最好
#     # 真正的 Batch 训练我们可以之后加上
#
#     # 前向传播
#     output = model.forward(X)
#
#     # 计算 Loss (仅用于观察，不参与反向传播)
#     # Cross Entropy: -sum(y_true * log(y_pred))
#     # 为了防止 log(0)，通常加一个微小的值 1e-8
#     loss = -np.sum(y * np.log(output + 1e-8)) / X.shape[0]
#
#     # 反向传播并更新权重
#     model.backward(X, y, learning_rate)
#
#     if i % 10 == 0:
#         # 顺便算一下准确率
#         predictions = np.argmax(output, axis=1)
#         labels = np.argmax(y, axis=1)
#         accuracy = np.mean(predictions == labels)
#         print(f"Epoch {i}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
#
# print("训练结束！")
# 全量训练，最后的准确率在0.875左右

print(f"开始训练 | 批次大小: {batch_size} | 总轮数: {epochs}")
for epoch in range(epochs):

    # 打乱数据 (Shuffle) - 非常重要！
    # 就像洗牌一样，防止模型死记硬背数据的顺序
    permutation = np.random.permutation(X.shape[0])
    X_shuffled = X[permutation]
    y_shuffled = y[permutation]

    # 小批量遍历
    for i in range(0, X.shape[0], batch_size):
        # 截取一小撮数据
        X_batch = X_shuffled[i: i + batch_size]
        y_batch = y_shuffled[i: i + batch_size]

        # 1. 前向传播
        output = model.forward(X_batch)

        # 2. 反向传播 & 更新权重
        model.backward(X_batch, y_batch, learning_rate)

    # --- 每个 Epoch 结束后的“期末考试” ---
    # 我们用全量数据简单测一下当前的准确率
    full_output = model.forward(X)
    predictions = np.argmax(full_output, axis=1)
    labels = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == labels)

    # 计算全量 Loss 用于观察
    loss = -np.sum(y * np.log(full_output + 1e-8)) / X.shape[0]

    print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy * 100:.2f}%")

print("训练完成！")
# 小批量训练 准确率100

def evaluate(model, test_data_path):
    print(f"\n正在加载测试集: {test_data_path} ...")

    # 1. 加载测试数据
    test_loader = DataLoader(test_data_path)
    test_loader.load()
    test_loader.normalize()
    X_test, y_test = test_loader.get_data()

    # 2. 前向传播 (考试)
    print("正在进行测试...")
    output = model.forward(X_test)

    # 3. 计算准确率
    predictions = np.argmax(output, axis=1)
    labels = np.argmax(y_test, axis=1)

    accuracy = np.mean(predictions == labels)
    print("-" * 30)
    print(f"训练集准确率 (参考): 100.00% (可能过拟合)")
    print(f"测试集准确率 (真实): {accuracy * 100:.2f}%")
    print("-" * 30)

    if accuracy < 0.95:
        print("结论: 差距有点大，确实过拟合比较严重。")
    elif accuracy > 0.97:
        print("结论: 哇！虽然训练集过拟合，但泛化能力居然还不错！")

# 这里填你实际的测试集路径
# 如果你是在 train.py 里直接接着写，可以直接调用这个函数
evaluate(model, str(MNIST_TEST_CSV))

FNN_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
model.save(str(FNN_CHECKPOINT))
