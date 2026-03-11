# HandWrittenNN - MNIST 手写数字识别

[English](./README.md)

基于 MNIST 的手写数字识别项目，包含：
- NumPy 手写前馈网络（FNN）
- PyTorch 卷积网络（CNN）
- Flask Web 推理页面

## 项目结构

- `src/`: 训练、导出、Web 入口与通用工具
- `models/`: 模型结构定义（代码）
- `data/`: 数据目录（CSV 数据集与 torchvision 下载数据）
- `artifacts/checkpoints/`: 训练输出权重（`.pkl`/`.pth`）
- `artifacts/exports/`: 导出产物（`.onnx`）
- `test/`: 测试代码

## 环境配置

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 使用方式

```bash
# NumPy FNN 训练
python -m src.fnn.train

# PyTorch CNN 训练
python -m src.cnn.train

# 导出 ONNX
python -m src.cnn.export_onnx

# 启动 Web
python -m src.web_app
```

## FNN 训练步骤

1. 明确任务并准备数据
   - 任务类型：分类/回归
   - 划分训练集、测试集、验证集
2. 数据预处理
   - 特征归一化或标准化
   - 标签编码（one-hot）
   - 可选：打乱、去噪、特征工程
3. 设计网络结构
   - 输入层维度：特征数
   - 隐藏层：合理设置宽度和深度
   - 输出层：
     - 分类：softmax（多分类）、sigmoid（二分类）
     - 回归：线性输出
   - 激活函数：ReLU 等
4. 初始化参数
   - 初始化权重
   - 偏置通常初始化为 0 或小常数
5. 定义损失函数和优化器
   - 分类常用交叉熵
   - 回归常用 MSE
   - 优化器：Adam、SGD、Momentum 等
   - 设置学习率、batch size、epoch 等
6. 循环训练
7. 验证与调参
   - 观察过拟合与欠拟合
8. 测试与部署

## 待解决问题

- 为什么使用 MNIST？
- 为什么要把原始像素值（0-255）压到 0-1？
- 如果不压缩，对梯度有什么影响？
- 为什么要做 one-hot？
- `w1`, `b1`, `w2`, `b2` 的形状分别是？
- 偏置初始化为什么通常设为 0？
- 什么是 He Initialization？
- 为什么要 `__init__.py`？没有它时为什么 `test` 文件夹下的文件无法 import？
