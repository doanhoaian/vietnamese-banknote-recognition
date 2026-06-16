"""Cấu hình tập trung cho toàn bộ dự án nhận diện tiền Việt Nam."""
from pathlib import Path

# --- Đường dẫn ---
ROOT_DIR = Path(__file__).resolve().parent          # .../vietnam-currency
DATASET_DIR = ROOT_DIR.parent / "dataset"           # .../dataset (12 thư mục lớp)
OUTPUT_DIR = ROOT_DIR / "outputs"                    # nơi lưu model, biểu đồ
MODEL_PATH = OUTPUT_DIR / "currency_model.pt"        # checkpoint tốt nhất

# --- Danh sách lớp (tên thư mục) và nhãn hiển thị ---
# Tên thư mục là mệnh giá VND (đệm 0 ở đầu); 000000 = "Không có tiền".
CLASS_DIRS = [
    "000000", "000200", "000500", "001000", "002000", "005000",
    "010000", "020000", "050000", "100000", "200000", "500000",
]

# Nhãn thân thiện để in ra cho người dùng
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

# --- Tham số huấn luyện ---
IMAGE_SIZE = 224          # MobileNetV2 pretrained mong đợi ~224x224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3      # cho phần đầu phân loại (head)
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.15          # 15% validation
TEST_RATIO = 0.15         # 15% test  -> 70% train
SEED = 42

# Chuẩn hoá theo ImageNet (vì dùng backbone pretrained trên ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
