"""Cấu hình tập trung cho toàn bộ dự án nhận diện tiền Việt Nam."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR.parent / "dataset"
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

# --- Hiển thị (app.py) ---
# Danh sách font TrueType hỗ trợ tiếng Việt, thử lần lượt cho tới khi tìm thấy.
# OpenCV không vẽ được dấu tiếng Việt, nên app.py dùng PIL + font này.
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",        # Linux
    "C:/Windows/Fonts/arial.ttf",                              # Windows
]
FONT_SIZE = 32

# --- Định vị tờ tiền bằng Grad-CAM (app.py) ---
# Khung được suy ra từ vùng model chú ý nhất (lớp conv cuối MobileNetV2),
# nên bám đúng cái model dùng để đọc mệnh giá thay vì dựa vào màu sắc.
CONF_THRESHOLD = 0.60        # chỉ vẽ khung khi độ tin cậy >= ngưỡng này
NO_MONEY_CLASS = "000000"    # lớp "không có tiền" -> không vẽ khung
GRADCAM_THRESHOLD = 0.40     # giữ vùng có activation >= 40% đỉnh để tạo khung
BOX_SMOOTHING = 0.5          # làm mượt khung theo thời gian (0=không mượt, ->1 mượt hơn)
PROCESS_EVERY = 3            # chạy Grad-CAM mỗi N frame (giảm lag); 1 = mỗi frame
REFINE_BBOX = True           # tinh chỉnh cạnh khung bằng CV bên trong vùng Grad-CAM
REFINE_ROI_EXPAND = 0.15     # nới vùng Grad-CAM ra ngoài trước khi dò cạnh (15%)
