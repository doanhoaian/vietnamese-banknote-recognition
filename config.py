from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "currency_model.pt"

CLASS_DIRS = [
    "000000", "000200", "000500", "001000", "002000", "005000",
    "010000", "020000", "050000", "100000", "200000", "500000",
]

LABELS_VI = {
    "000000": "Không có tiền",
    "000200": "200 đ",
    "000500": "500 đ",
    "001000": "1.000 đ",
    "002000": "2.000 đ",
    "005000": "5.000 đ",
    "010000": "10.000 đ",
    "020000": "20.000 đ",
    "050000": "50.000 đ",
    "100000": "100.000 đ",
    "200000": "200.000 đ",
    "500000": "500.000 đ",
}

IMAGE_SIZE = 224          # MobileNetV2 pretrained ~224x224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.15          # 15% Validation
TEST_RATIO = 0.15         # 15% Test  -> 70% Train
SEED = 42

# Chuẩn hoá theo ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
FONT_SIZE = 32
CONF_THRESHOLD = 0.60
NO_MONEY_CLASS = "000000"