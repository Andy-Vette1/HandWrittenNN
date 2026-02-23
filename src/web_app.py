import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from src.paths import CNN_CHECKPOINT, FNN_CHECKPOINT

# 导入两个模型类
from models.fnn import FNN
from models.cnn import CNN

app = Flask(__name__, template_folder='web/templates')

# --- 1. 加载 NumPy FNN ---
fnn_model = FNN(784, 128, 10)
try:
    fnn_model.load(str(FNN_CHECKPOINT))
    print("FNN 模型加载成功")
except:
    print("FNN 模型未找到")

# --- 2. 加载 PyTorch CNN ---
cnn_model = CNN()
try:
    # map_location='cpu' 确保在没显卡的电脑上也能跑
    cnn_model.load_state_dict(torch.load(str(CNN_CHECKPOINT), map_location='cpu'))
    cnn_model.eval()  # 切换到评估模式 (这就好比要把"考试状态"开关打开)
    print("CNN 模型加载成功")
except:
    print("CNN 模型未找到 (请先运行 src/train_cnn.py)")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    pixels = data.get('pixels')
    is_black_on_white = data.get('is_black_on_white', False)
    # 获取前端传来的模型类型: 'fnn' 或 'cnn'
    model_type = data.get('model_type', 'fnn')

    if not pixels:
        return jsonify({'error': 'No data'}), 400

    # --- 数据预处理 (通用) ---
    x_input = np.array(pixels, dtype=np.float32)
    if is_black_on_white:
        x_input = 255.0 - x_input
    x_input = x_input / 255.0

    mean = 0.1307
    std = 0.3081
    x_input = (x_input - mean) / std

    # --- 分支逻辑 ---
    if model_type == 'cnn':
        # CNN 需要 (1, 1, 28, 28) 的 4D Tensor
        # reshape: [1, 28, 28] -> [1, 1, 28, 28]
        tensor_input = torch.tensor(x_input.reshape(1, 1, 28, 28))

        with torch.no_grad():  # 推理时不需要算梯度
            outputs = cnn_model(tensor_input)
            # 手动加 Softmax 算出概率
            probs = F.softmax(outputs, dim=1).numpy()[0]

    else:  # 默认为 FNN
        # FNN 需要 (1, 784) 的 2D Array
        fnn_input = x_input.reshape(1, -1)
        probs = fnn_model.forward(fnn_input)[0]

    prediction = int(np.argmax(probs))
    probs_list = [float(p) for p in probs]

    return jsonify({
        'prediction': prediction,
        'probabilities': probs_list,
        'confidence': f"{max(probs_list) * 100:.2f}%",
        'used_model': model_type.upper()
    })


if __name__ == '__main__':
    app.run(debug=True)
