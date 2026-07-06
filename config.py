from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "currency_model.pt"

CLASS_DIRS = [
    "000000", "001000", "002000", "005000",
    "010000", "020000", "050000", "100000", "200000", "500000",
]

LABELS_VI = {
    "000000": "Không có tiền",
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

# --- Dữ liệu ---
IMAGE_SIZE = 224                 # MobileNetV2 pretrained ~224x224
VAL_RATIO = 0.15
TEST_RATIO = 0.15                # -> 70% train
NUM_WORKERS = 2
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Huấn luyện ---
BATCH_SIZE = 32
EPOCHS = 30
WEIGHT_DECAY = 1e-4
USE_CLASS_WEIGHTS = True

# Fine-tuning: đóng băng backbone nhưng mở N block cuối với LR thấp hơn head.
FREEZE_BACKBONE = True
UNFREEZE_LAST_N_BLOCKS = 3
HEAD_LR = 1e-3
BACKBONE_LR = 1e-4

# Scheduler (ReduceLROnPlateau theo val accuracy).
LR_FACTOR = 0.5
LR_PATIENCE = 2

# Early stopping theo val accuracy.
EARLY_STOPPING_PATIENCE = 6
EARLY_STOPPING_MIN_DELTA = 1e-4

# --- Suy luận / hiển thị ---
CONF_THRESHOLD = 0.60
NO_MONEY_CLASS = "000000"
UNCERTAIN_LABEL = "Không chắc chắn"

# Vùng quan tâm (ROI)
ROI_RATIO = 0.6
ROI_ASPECT = 2.2

# --- Tự động chụp khoảnh khắc tốt nhất ---
AUTO_STABLE_CONF = 0.75
AUTO_STABLE_FRAMES = 6
AUTO_REARM_FRAMES = 5

# --- Giọng nói ---
LABELS_SPEECH = {
    "000000": "Không có tiền",
    "001000": "Một nghìn đồng",
    "002000": "Hai nghìn đồng",
    "005000": "Năm nghìn đồng",
    "010000": "Mười nghìn đồng",
    "020000": "Hai mươi nghìn đồng",
    "050000": "Năm mươi nghìn đồng",
    "100000": "Một trăm nghìn đồng",
    "200000": "Hai trăm nghìn đồng",
    "500000": "Năm trăm nghìn đồng",
}
SPEECH_GREETING = "Sẵn sàng. Hãy đưa tờ tiền vào trước camera."
SPEECH_GUIDE_HOLD = "Giữ yên, đang nhận diện."
SPEECH_UNSURE = "Chưa rõ, bạn thử lại nhé."
SPEECH_NEXT = "Đưa tờ tiếp theo."
SPEECH_NO_CAMERA = "Không mở được camera."

GUIDE_COOLDOWN_S = 2.5       # khoảng cách tối thiểu giữa hai câu dẫn hướng
VOICE_NAME = "Linh"
VOICE_RATE = 180

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
FONT_SIZE = 32
