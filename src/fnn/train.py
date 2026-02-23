import numpy as np

from src.data_loader import DataLoader
from src.fnn.model import FNN
from src.paths import FNN_CHECKPOINT, MNIST_TEST_CSV, MNIST_TRAIN_CSV


def train_fnn(
    train_data_path,
    input_size=784,
    hidden_size=128,
    output_size=10,
    learning_rate=0.1,
    epochs=100,
    batch_size=64,
):
    # 1. Load data
    loader = DataLoader(train_data_path)
    loader.load()
    loader.normalize()
    X, y = loader.get_data()

    # 2. Initialize model
    model = FNN(input_size, hidden_size, output_size)

    print(f"开始训练 | 批次大小: {batch_size} | 总轮数: {epochs}")
    train_accuracy = 0.0
    for epoch in range(epochs):
        # Shuffle each epoch
        permutation = np.random.permutation(X.shape[0])
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]

        # Mini-batch training
        for i in range(0, X.shape[0], batch_size):
            X_batch = X_shuffled[i: i + batch_size]
            y_batch = y_shuffled[i: i + batch_size]

            model.forward(X_batch)
            model.backward(X_batch, y_batch, learning_rate)

        # Evaluate on full training set for logging
        full_output = model.forward(X)
        predictions = np.argmax(full_output, axis=1)
        labels = np.argmax(y, axis=1)
        train_accuracy = float(np.mean(predictions == labels))
        loss = -np.sum(y * np.log(full_output + 1e-8)) / X.shape[0]

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {train_accuracy * 100:.2f}%")

    print("训练完成！")
    return model, train_accuracy


def evaluate(model, test_data_path, train_accuracy=None):
    print(f"\n正在加载测试集: {test_data_path} ...")

    # 1. Load test data
    test_loader = DataLoader(test_data_path)
    test_loader.load()
    test_loader.normalize()
    X_test, y_test = test_loader.get_data()

    # 2. Forward pass
    print("正在进行测试...")
    output = model.forward(X_test)

    # 3. Accuracy
    predictions = np.argmax(output, axis=1)
    labels = np.argmax(y_test, axis=1)
    test_accuracy = float(np.mean(predictions == labels))

    print("-" * 30)
    if train_accuracy is not None:
        print(f"训练集准确率 (参考): {train_accuracy * 100:.2f}%")
    print(f"测试集准确率 (真实): {test_accuracy * 100:.2f}%")
    print("-" * 30)

    if train_accuracy is not None and train_accuracy - test_accuracy > 0.03:
        print("结论: 差距有点大，确实过拟合比较严重。")
    elif test_accuracy > 0.97:
        print("结论: 泛化能力表现不错。")

    return test_accuracy


def main():
    model, train_accuracy = train_fnn(
        train_data_path=str(MNIST_TRAIN_CSV),
        learning_rate=0.1,
        epochs=100,
        batch_size=64,
    )
    evaluate(model, str(MNIST_TEST_CSV), train_accuracy=train_accuracy)

    FNN_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(FNN_CHECKPOINT))


if __name__ == "__main__":
    main()
