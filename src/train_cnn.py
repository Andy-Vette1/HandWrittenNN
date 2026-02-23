import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.cnn import CNN
from src.paths import TORCHVISION_DATA_DIR, CNN_CHECKPOINT

# 1. 准备数据 (使用 PyTorch 自带的加载器，更方便)
# 这里的 transform 会自动把图片变成 Tensor 并归一化 (0-1)
# 修改 transform 定义
transform = transforms.Compose([
    transforms.RandomRotation(10), # 随机旋转 -10度 到 +10度
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # 标准化也是提分点，可以加上
])

# 直接下载/加载 MNIST
train_dataset = datasets.MNIST(root=str(TORCHVISION_DATA_DIR), train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 2. 初始化模型
# ✅ 修改后的代码：支持 CUDA (NVIDIA), MPS (Mac), 和 CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # <--- 这里就是 Mac 的 GPU 加速
    print("使用 Apple Metal (MPS) 加速中...")
else:
    device = torch.device("cpu")
    print("未检测到 GPU，正在使用 CPU 慢速训练...")

print(f"当前设备: {device}")

model = CNN().to(device)

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练循环
epochs = 20
print("开始训练 CNN ...")

for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向 + 反向 + 更新
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

print("训练完成！")

# 5. 保存模型
CNN_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), str(CNN_CHECKPOINT))
print(f"模型已保存到: {CNN_CHECKPOINT}")

print("\n正在测试集上评估 CNN 准确率...")

# 1. 加载测试集 (train=False)
test_dataset = datasets.MNIST(root=str(TORCHVISION_DATA_DIR), train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

correct = 0
total = 0

# 切换到评估模式 (非常重要！虽然这个简单CNN影响不大，但要养成好习惯)
model.eval()

with torch.no_grad():  # 测试时不需要算梯度，节省内存
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)

        # 获取预测结果 (最大值的索引)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'CNN 最终测试集准确率: {accuracy:.2f}%')

# 对比一下 FNN
print(f"相比 FNN 的 97.96%，提升了: {accuracy - 97.96:.2f}%")
