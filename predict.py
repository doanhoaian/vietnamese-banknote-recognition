import sys

from PIL import Image

import config
from inference import CurrencyClassifier


def main():
    if len(sys.argv) < 2:
        print("Dùng: python predict.py <đường_dẫn_ảnh>")
        sys.exit(1)
    if not config.MODEL_PATH.exists():
        print(f"Chưa có model tại {config.MODEL_PATH}. Hãy chạy: python train.py")
        sys.exit(1)

    clf = CurrencyClassifier()
    pred = clf.predict(Image.open(sys.argv[1]).convert("RGB"))

    print(f"\n=> Dự đoán: {pred.label}  (độ tin cậy {pred.confidence*100:.1f}%)\n")
    print("Top dự đoán:")
    for name, prob in pred.topk:
        print(f"  {name:>15s} : {prob*100:5.1f}%")


if __name__ == "__main__":
    main()
