import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from src.paths import CNN_CHECKPOINT, FNN_CHECKPOINT

# 导入两个模型类
from src.fnn import FNN
from src.cnn.model import CNN

app = Flask(__name__, template_folder='web/templates')

# --- 1. 加载 NumPy FNN ---
fnn_model = None
try:
    loaded_fnn = FNN(784, 128, 10)
    loaded_fnn.load(str(FNN_CHECKPOINT))
    fnn_model = loaded_fnn
    print("FNN 模型加载成功")
except FileNotFoundError:
    print(f"FNN 模型未找到: {FNN_CHECKPOINT}")
except Exception as err:
    print(f"FNN 模型加载失败: {err}")

# --- 2. 加载 PyTorch CNN ---
cnn_model = None
try:
    # map_location='cpu' 确保在没显卡的电脑上也能跑
    loaded_cnn = CNN()
    loaded_cnn.load_state_dict(torch.load(str(CNN_CHECKPOINT), map_location='cpu'))
    loaded_cnn.eval()  # 切换到评估模式 (这就好比要把"考试状态"开关打开)
    cnn_model = loaded_cnn
    print("CNN 模型加载成功")
except FileNotFoundError:
    print(f"CNN 模型未找到: {CNN_CHECKPOINT} (请先运行 src/train_cnn.py)")
except Exception as err:
    print(f"CNN 模型加载失败: {err}")


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
    x_input = x_input / 255.0  # Match FNN training normalization

    # --- 分支逻辑 ---
    if model_type == 'cnn':
        if cnn_model is None:
            return jsonify({'error': 'CNN model is not loaded'}), 503

        mean = 0.1307
        std = 0.3081
        cnn_input = (x_input - mean) / std

        # CNN 需要 (1, 1, 28, 28) 的 4D Tensor
        # reshape: [1, 28, 28] -> [1, 1, 28, 28]
        tensor_input = torch.tensor(cnn_input.reshape(1, 1, 28, 28))

        with torch.no_grad():  # 推理时不需要算梯度
            outputs = cnn_model(tensor_input)
            # 手动加 Softmax 算出概率
            probs = F.softmax(outputs, dim=1).numpy()[0]

    else:  # 默认为 FNN
        if fnn_model is None:
            return jsonify({'error': 'FNN model is not loaded'}), 503

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
