# HandWrittenNN - MNIST Digit Recognition

[中文文档 (Chinese)](./README.zh-CN.md)

A handwritten digit recognition project based on MNIST, including:
- NumPy feedforward neural network (FNN) implementation
- PyTorch convolutional neural network (CNN)
- Flask web inference app

## Project Structure

- `src/`: Training, export, web entry, and shared utilities
- `models/`: Model architecture definitions
- `data/`: Data directory (CSV datasets and torchvision-downloaded data)
- `artifacts/checkpoints/`: Training checkpoints (`.pkl`/`.pth`)
- `artifacts/exports/`: Exported artifacts (`.onnx`)
- `test/`: Test code

## Environment Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Train NumPy FNN
python -m src.fnn.train

# Train PyTorch CNN
python -m src.cnn.train

# Export ONNX
python -m src.cnn.export_onnx

# Start Web app
python -m src.web_app
```

## FNN Training Workflow

1. Define the task and prepare data
   - Task type: classification/regression
   - Split train/test/validation sets
2. Data preprocessing
   - Normalize or standardize features
   - Encode labels (one-hot)
   - Optional: shuffling, denoising, feature engineering
3. Design network architecture
   - Input dimension: number of features
   - Hidden layers: choose appropriate width and depth
   - Output layer:
     - Classification: softmax (multi-class), sigmoid (binary)
     - Regression: linear output
   - Activation functions: ReLU, etc.
4. Initialize parameters
   - Initialize weights
   - Bias is often initialized to 0 or a small constant
5. Define loss and optimizer
   - Cross-entropy is common for classification
   - MSE is common for regression
   - Optimizers: Adam, SGD, Momentum, etc.
   - Set learning rate, batch size, epochs, etc.
6. Train iteratively
7. Validate and tune
   - Monitor overfitting and underfitting
8. Test and deploy

## Open Questions

- Why use MNIST?
- Why scale raw pixel values (0-255) to 0-1?
- What happens to gradients if we do not scale values?
- Why use one-hot encoding?
- What are the shapes of `w1`, `b1`, `w2`, and `b2`?
- Why is bias initialization often set to 0?
- What is He Initialization?
- Why is `__init__.py` needed, and why can files under `test` fail to import without it?
