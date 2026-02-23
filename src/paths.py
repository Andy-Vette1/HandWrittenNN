from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MNIST_TRAIN_CSV = DATA_DIR / "mnist_train.csv.zip"
MNIST_TEST_CSV = DATA_DIR / "mnist_test.csv.zip"
TORCHVISION_DATA_DIR = DATA_DIR / "torchvision"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
EXPORTS_DIR = ARTIFACTS_DIR / "exports"

FNN_CHECKPOINT = CHECKPOINTS_DIR / "mnist_fnn.pkl"
CNN_CHECKPOINT = CHECKPOINTS_DIR / "mnist_cnn.pth"
CNN_ONNX_EXPORT = EXPORTS_DIR / "mnist_cnn.onnx"
