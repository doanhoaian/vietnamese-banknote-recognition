# Nhận diện tiền Việt Nam 🇻🇳

Dự án huấn luyện mô hình nhận diện mệnh giá tiền Việt Nam từ ảnh, dùng
**PyTorch + Transfer Learning (MobileNetV2)**.

## Dataset

Nằm ở thư mục `../dataset` (cùng cấp với `vietnam-currency`). Mỗi ảnh là ảnh
webcam chụp một người cầm tờ tiền trước mặt. Có **12 lớp** (tên thư mục là mệnh
giá VND, đệm 0 ở đầu):

| Thư mục  | Ý nghĩa        |
|----------|----------------|
| `000000` | Không có tiền  |
| `000200` | 200 đ          |
| `000500` | 500 đ          |
| `001000` | 1.000 đ        |
| `002000` | 2.000 đ        |
| `005000` | 5.000 đ        |
| `010000` | 10.000 đ       |
| `020000` | 20.000 đ       |
| `050000` | 50.000 đ       |
| `100000` | 100.000 đ      |
| `200000` | 200.000 đ      |
| `500000` | 500.000 đ      |

Tổng ~2.712 ảnh PNG (192×144). File `ngtrdaiDataset.data` là phiên bản numpy
đóng gói sẵn của cùng dữ liệu — **dự án này không dùng tới nó** mà đọc trực tiếp
từ các thư mục ảnh.

## Cài đặt môi trường

> ⚠️ PyTorch chưa có wheel ổn định cho Python 3.14. Hãy dùng **Python 3.10–3.12**.

```bash
cd vietnam-currency
python3.12 -m venv .venv          # hoặc python3.11
source .venv/bin/activate
pip install -r requirements.txt
```

## Cấu trúc

| File              | Vai trò                                                        |
|-------------------|---------------------------------------------------------------|
| `config.py`       | Đường dẫn, danh sách lớp, siêu tham số                         |
| `data.py`         | Đọc ảnh, chia train/val/test (stratified), DataLoader         |
| `model.py`        | MobileNetV2 pretrained + head 12 lớp                          |
| `train.py`        | Huấn luyện, lưu model tốt nhất, vẽ biểu đồ loss/accuracy       |
| `predict.py`      | Dự đoán mệnh giá từ 1 ảnh                                      |
| `app.py`  | Nhận diện realtime qua webcam                                  |

## Huấn luyện

```bash
python train.py
```

Kết quả lưu ở `outputs/`:
- `currency_model.pt` — checkpoint tốt nhất (theo accuracy trên tập validation)
- `training_curves.png` — biểu đồ loss & accuracy

Sau khi train xong, script tự đánh giá trên **tập test** và in accuracy.

Điều chỉnh số epoch, batch size, learning rate… trong `config.py`.

## Dự đoán 1 ảnh

```bash
python predict.py ../dataset/100000/100000_0.png
```

In ra mệnh giá dự đoán + độ tin cậy + top-3.

## Demo webcam realtime

```bash
python webcam_demo.py
```

Cầm tờ tiền trước webcam; nhãn mệnh giá hiển thị trên khung hình. Nhấn `q` để thoát.

## Ghi chú kỹ thuật

- **Transfer learning**: đóng băng backbone MobileNetV2 (đã học đặc trưng từ
  ImageNet), chỉ huấn luyện lớp phân loại cuối → train nhanh, ít overfit với
  dataset nhỏ (~250 ảnh/lớp).
- **Chia dữ liệu stratified** 70/15/15 đảm bảo mọi tập đều có đủ 12 mệnh giá.
- **Augmentation** nhẹ (xoay ±8°, đổi sáng/tương phản, lật ngang) để tăng tính
  tổng quát mà không làm biến dạng mệnh giá.
- Tự động dùng GPU CUDA hoặc Apple Silicon (MPS) nếu có, ngược lại dùng CPU.
- Muốn tăng độ chính xác thêm: sau vài epoch, mở khoá backbone
  (`p.requires_grad = True`) và fine-tune với learning rate nhỏ hơn.
```
