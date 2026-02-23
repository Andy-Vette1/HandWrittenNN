import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.cnn.model import CNN
from src.paths import TORCHVISION_DATA_DIR, CNN_CHECKPOINT


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("使用 Apple Metal (MPS) 加速中...")
        return torch.device("mps")

    print("未检测到 GPU，正在使用 CPU 慢速训练...")
    return torch.device("cpu")


def train_cnn(epochs=20, batch_size=64, learning_rate=0.001):
    # 1. Prepare data
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root=str(TORCHVISION_DATA_DIR),
        train=True,
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 2. Initialize model
    device = get_device()
    print(f"当前设备: {device}")
    model = CNN().to(device)

    # 3. Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Training loop
    print("开始训练 CNN ...")
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

    print("训练完成！")
    return model, device


def evaluate_cnn(model, device):
    print("\n正在测试集上评估 CNN 准确率...")
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_dataset = datasets.MNIST(
        root=str(TORCHVISION_DATA_DIR),
        train=False,
        transform=transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"CNN 最终测试集准确率: {accuracy:.2f}%")
    print(f"相比 FNN 的 97.96%，提升了: {accuracy - 97.96:.2f}%")
    return accuracy


def main():
    model, device = train_cnn()
    CNN_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(CNN_CHECKPOINT))
    print(f"模型已保存到: {CNN_CHECKPOINT}")
    evaluate_cnn(model, device)


if __name__ == "__main__":
    main()
