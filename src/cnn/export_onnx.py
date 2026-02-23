import torch
import torch.onnx

from src.cnn.model import CNN
from src.paths import CNN_CHECKPOINT, CNN_ONNX_EXPORT


def main():
    # 1. Load trained model
    model = CNN()
    model.load_state_dict(torch.load(str(CNN_CHECKPOINT), map_location='cpu'))
    model.eval()

    # 2. Dummy input for graph tracing
    dummy_input = torch.randn(1, 1, 28, 28)

    # 3. Export ONNX
    CNN_ONNX_EXPORT.parent.mkdir(parents=True, exist_ok=True)
    output_path = str(CNN_ONNX_EXPORT)
    print(f"正在导出模型到 {output_path} ...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
        external_data=False,
    )

    print("导出成功！现在你有一个可以在浏览器里跑的模型文件了。")


if __name__ == "__main__":
    main()
