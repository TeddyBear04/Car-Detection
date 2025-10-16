# Hướng Dẫn Training Model YOLO Cho Car Detection

## 📋 Tổng Quan

Project này sử dụng YOLOv8 để train model nhận diện xe ô tô từ Stanford Cars Dataset. Model sau khi train sẽ được sử dụng trong `Live_Car_Detector_YOLO.py` để detect xe trong video real-time.

## 🛠️ Cài Đặt

### 1. Cài đặt dependencies

```powershell
pip install -r requirements_yolo.txt
```

### 2. Cấu trúc thư mục

```
Computer vision/
├── Car data set/
│   └── car_dataset-master/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── valid/
│       │   ├── images/
│       │   └── labels/
│       └── test/
│           ├── images/
│           └── labels/
├── Train_Car_Model.py              # Script training
├── Live_Car_Detector_YOLO.py       # Script detection với YOLO
├── Live_Car_Detector.py            # Script cũ (Haar Cascade)
├── car_dataset.yaml                # Config dataset (tự động tạo)
├── car_detector_best.pt            # Model sau khi train
└── car_detection/                  # Thư mục kết quả training
    └── yolov8n_car/
        ├── weights/
        │   ├── best.pt
        │   └── last.pt
        └── results.png
```

## 🚀 Training Model

### Chạy training:

```powershell
python Train_Car_Model.py
```

### Các thông số training:

- **Model**: YOLOv8n (nano) - nhẹ, nhanh, phù hợp laptop
- **Epochs**: 50 (có thể điều chỉnh)
- **Batch size**: 16 (giảm nếu thiếu RAM)
- **Image size**: 640x640
- **Device**: Auto (GPU nếu có, CPU nếu không)

### Điều chỉnh thông số:

Mở `Train_Car_Model.py` và sửa các tham số trong `model.train()`:

```python
results = model.train(
    epochs=100,        # Tăng để train lâu hơn
    batch=8,           # Giảm nếu thiếu RAM
    imgsz=416,         # Giảm để train nhanh hơn
    patience=20,       # Tăng patience cho early stopping
)
```

## 📊 Theo Dõi Training

Training sẽ hiển thị:
- Loss (box, cls, dfl)
- mAP50, mAP50-95
- Precision, Recall
- Progress bar

Kết quả được lưu tại:
- `car_detection/yolov8n_car/weights/best.pt` - Model tốt nhất
- `car_detection/yolov8n_car/results.png` - Biểu đồ metrics
- `car_detection/yolov8n_car/confusion_matrix.png` - Confusion matrix
- `car_detection/yolov8n_car/val_batch*.jpg` - Ảnh validation

## 🎯 Sử Dụng Model Đã Train

### 1. Sau khi training xong:

Model tốt nhất sẽ được copy sang `car_detector_best.pt`

### 2. Chạy detection:

```powershell
python Live_Car_Detector_YOLO.py
```

### 3. Tùy chỉnh detection:

Mở `Live_Car_Detector_YOLO.py` và điều chỉnh:

```python
# Confidence threshold (0.0 - 1.0)
results = model(frame, conf=0.3)  # Tăng để detection chặt chẽ hơn

# Vùng nghi ngờ
SUSPICIOUS_ZONES = [
    (200, 150, 500, 400),  # (x1, y1, x2, y2)
]

# Kích thước hiển thị
DISPLAY_WIDTH = 1024
DISPLAY_HEIGHT = 768
```

## 🔍 So Sánh Với Haar Cascade

### Haar Cascade (`Live_Car_Detector.py`):
- ✅ Nhẹ, nhanh
- ❌ Độ chính xác thấp
- ❌ Nhiều false positives
- ❌ Không linh hoạt

### YOLOv8 (`Live_Car_Detector_YOLO.py`):
- ✅ Độ chính xác cao
- ✅ Ít false positives
- ✅ Có confidence score
- ✅ Linh hoạt, tùy chỉnh được
- ⚠️ Cần training trước
- ⚠️ Tốn tài nguyên hơn

## 💡 Tips & Tricks

### 1. Tối ưu tốc độ:

```python
# Sử dụng model nhẹ hơn
model = YOLO('yolov8n.pt')  # nano (nhanh nhất)
# model = YOLO('yolov8s.pt')  # small
# model = YOLO('yolov8m.pt')  # medium

# Giảm kích thước ảnh
results = model(frame, imgsz=416)

# Bỏ qua một số frame
if frame_count % 2 == 0:  # Xử lý mỗi 2 frames
    results = model(frame)
```

### 2. Tăng độ chính xác:

```python
# Train lâu hơn
epochs=100

# Tăng image size
imgsz=1280

# Sử dụng model lớn hơn
model = YOLO('yolov8m.pt')
```

### 3. Giảm RAM/VRAM usage:

```python
# Giảm batch size
batch=4

# Tắt cache
cache=False

# Giảm workers
workers=2
```

## 🐛 Xử Lý Lỗi

### Lỗi: Out of Memory
```python
# Giảm batch size
batch=4

# Giảm image size
imgsz=416

# Sử dụng model nhỏ hơn
model = YOLO('yolov8n.pt')
```

### Lỗi: CUDA Out of Memory
```python
# Sử dụng CPU
device='cpu'

# Hoặc giảm batch size
batch=2
```

### Model không detect được xe
```python
# Giảm confidence threshold
results = model(frame, conf=0.2)

# Hoặc train thêm
epochs=100
```

## 📈 Đánh Giá Model

### Metrics quan trọng:

- **mAP50**: mean Average Precision @ IoU 0.5
- **mAP50-95**: mean Average Precision @ IoU 0.5:0.95
- **Precision**: Tỷ lệ detection đúng / tất cả detection
- **Recall**: Tỷ lệ xe được phát hiện / tổng số xe

### Mục tiêu:

- mAP50 > 0.7: Tốt
- mAP50 > 0.8: Rất tốt
- mAP50 > 0.9: Xuất sắc

## 🎓 Học Thêm

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLO Training Guide](https://docs.ultralytics.com/modes/train/)
- [YOLO Prediction Guide](https://docs.ultralytics.com/modes/predict/)

## 📝 Notes

- Training có thể mất vài giờ tùy hardware
- GPU làm tăng tốc độ training rất nhiều (10-100x)
- Model size: ~6MB (YOLOv8n) đến ~100MB (YOLOv8x)
- Có thể resume training nếu bị gián đoạn: `resume=True`

## 🆘 Support

Nếu gặp vấn đề, kiểm tra:
1. Đã cài đặt đúng dependencies chưa
2. Dataset có đúng cấu trúc không
3. Đủ RAM/VRAM chưa (tối thiểu 4GB RAM cho CPU, 2GB VRAM cho GPU)
4. PyTorch có support CUDA không (nếu dùng GPU)

---

**Good luck with your training! 🚀**
