import torch.onnx
from models.cnn import CNN
from src.paths import CNN_CHECKPOINT, CNN_ONNX_EXPORT

# 1. 加载训练好的模型
model = CNN()
# 记得用 map_location='cpu'，导出时不需要 GPU
model.load_state_dict(torch.load(str(CNN_CHECKPOINT), map_location='cpu'))
model.eval()

# 2. 创建一个虚拟输入 (Dummy Input)
# 它的形状必须和模型输入一致: [Batch, Channel, Height, Width]
# ONNX 需要通过跑一遍这个虚拟输入来追踪网络结构
dummy_input = torch.randn(1, 1, 28, 28)

# 3. 导出为 ONNX
CNN_ONNX_EXPORT.parent.mkdir(parents=True, exist_ok=True)
output_path = str(CNN_ONNX_EXPORT)
print(f"正在导出模型到 {output_path} ...")

torch.onnx.export(model,               # 模型实例
                  dummy_input,         # 虚拟输入
                  output_path,         # 输出路径
                  export_params=True,  # 保存权重
                  opset_version=18,    # ONNX 版本 (10 或 11 兼容性最好)
                  do_constant_folding=True,  # 优化常量折叠
                  input_names = ['input'],   # 输入节点的名字 (JS里要用)
                  output_names = ['output'], # 输出节点的名字
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 允许动态 batch size
                                'output' : {0 : 'batch_size'}},
                  external_data=False)

print("导出成功！现在你有一个可以在浏览器里跑的模型文件了。")
