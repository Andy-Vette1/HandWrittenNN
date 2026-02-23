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


## FNN训练步骤
1. 明确任务并准备测试数据
    - 任务类型：分类/回归
    - 准备数据集：划分训练数据、测试数据、验证数据
2. 数据预处理
    - 特征归一化或标准化
    - 类别标签编码（one-hot）
    - 可选：打乱、去噪、特这工程
3. 设计网络结构
    - 输入层维度：特征数
    - 隐藏层：
      - 设计合适的宽度
      - 设计合适的深度
    - 输出层：
      - 分类：softmax（多分类）、sigmod（而分类）
      - 回归：线行输出
    - 激活函数：ReLU
4. 初始化参数 
    - 初始化权重
    - 偏置通常初始化为0或者小常数
5. 定义损失函数和优化器
    - 分类常用交叉熵
    - 回归常用MSE
    - 优化器：Adam、SGD、Momentum等
    - 设置学习率、batch size、epoch等
6. 循环训练
7. 验证与调参
    - 观察是否过拟合、欠拟合
8. 测试与部署