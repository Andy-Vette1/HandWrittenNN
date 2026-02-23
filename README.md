# HandWrittenNN - MNIST Digit Recognition

基于 MNIST 的手写数字识别项目，包含：
- NumPy 手写前馈网络（FNN）
- PyTorch 卷积网络（CNN）
- Flask Web 推理页面

## Project Structure

- `src/`: 训练、导出、Web 入口与通用工具
- `models/`: 模型结构定义（代码）
- `data/`: 数据目录（CSV 数据集与 torchvision 下载数据）
- `artifacts/checkpoints/`: 训练输出权重（`.pkl`/`.pth`）
- `artifacts/exports/`: 导出产物（`.onnx`）
- `test/`: 测试代码

## Environment Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# NumPy FNN 训练
python -m src.train_fnn

# PyTorch CNN 训练
python -m src.train_cnn

# 导出 ONNX
python -m src.export_onnx

# 启动 Web
python -m src.web_app
```
