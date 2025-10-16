# Car Detection Project

Dự án nhận diện xe ô tô sử dụng YOLOv8 và Stanford Cars Dataset.

## 📁 Cấu Trúc Project

```
Computer vision/
├── config/                         # Configuration files
│   ├── __init__.py
│   └── config.py                   # Main configuration
│
├── core/                           # Core modules
│   ├── __init__.py
│   ├── detector.py                 # Car detector class
│   ├── trainer.py                  # Model trainer class
│   └── video_processor.py          # Video processing utilities
│
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── dataset.py                  # Dataset utilities
│   ├── logger.py                   # Logging utilities
│   └── visualization.py            # Visualization utilities
│
├── Car data set/                   # Dataset directory
│   └── car_dataset-master/
│       ├── train/
│       ├── valid/
│       └── test/
│
├── models/                         # Trained models (auto-created)
│   ├── car_detector_best.pt
│   └── car_detector_last.pt
│
├── results/                        # Training results (auto-created)
│   └── yolov8n_car/
│       ├── weights/
│       └── *.png (metrics plots)
│
├── logs/                           # Log files (auto-created)
│   ├── training.log
│   └── detection.log
│
├── train.py                        # Main training script
├── run_detection.py                # Main detection script
├── requirements_yolo.txt           # Dependencies
├── README.md                       # This file
└── vid.mp4                         # Test video

# Old files (for reference)
├── Train_Car_Model.py              # Old training script (monolithic)
├── Live_Car_Detector_YOLO.py       # Old detection script (monolithic)
└── README_TRAINING.md              # Old documentation
```

## 🚀 Quick Start

### 1. Cài đặt dependencies

```powershell
pip install -r requirements_yolo.txt
```

### 2. Kiểm tra cấu hình

```powershell
python -c "from config.config import print_config; print_config()"
```

### 3. Training model

```powershell
python train.py
```

### 4. Chạy detection

```powershell
python run_detection.py
```

## 📚 Modules Documentation

### Config Module (`config/`)

**`config.py`**: Chứa tất cả cấu hình project
- Đường dẫn (paths)
- Training hyperparameters
- Detection parameters
- Suspicious detection rules
- Device configuration

```python
from config.config import TrainingConfig, DetectionConfig

# Truy cập cấu hình
epochs = TrainingConfig.EPOCHS
conf_threshold = DetectionConfig.CONFIDENCE_THRESHOLD
```

### Core Module (`core/`)

**`detector.py`**: Car detection class
- Load YOLO model
- Detect cars trong ảnh
- Check suspicious vehicles
- Utility functions

```python
from core.detector import CarDetector

detector = CarDetector(model_path="models/car_detector_best.pt")
detections = detector.detect(frame)
```

**`trainer.py`**: Model training class
- Train YOLO model
- Validate model
- Export best model
- Manage training results

```python
from core.trainer import CarModelTrainer

trainer = CarModelTrainer()
trainer.train(data_yaml="dataset.yaml", epochs=50)
```

**`video_processor.py`**: Video processing utilities
- Read video frames
- Process với callback
- Save processed video
- FPS calculation

```python
from core.video_processor import VideoProcessor

processor = VideoProcessor("video.mp4")
processor.process(frame_callback=my_function)
```

### Utils Module (`utils/`)

**`dataset.py`**: Dataset utilities
- Verify dataset structure
- Count images
- Create YAML config
- Get statistics

```python
from utils.dataset import verify_dataset_structure, print_dataset_info

is_valid, msg = verify_dataset_structure(data_dir)
print_dataset_info(data_dir)
```

**`logger.py`**: Logging utilities
- Setup logger
- Training logger
- Detection logger
- System info logging

```python
from utils.logger import setup_logger, TrainingLogger

logger = setup_logger("MyLogger", "logs/my.log")
training_logger = TrainingLogger("logs/")
```

**`visualization.py`**: Visualization utilities
- Draw bounding boxes
- Draw info panels
- Draw FPS counter
- Create grid view
- Add watermark

```python
from utils.visualization import draw_box_with_label, draw_info_panel

frame = draw_box_with_label(frame, bbox, label, color)
frame = draw_info_panel(frame, info_dict)
```

## ⚙️ Configuration

### Training Configuration

Chỉnh sửa `config/config.py`:

```python
class TrainingConfig:
    EPOCHS = 50              # Số epochs
    BATCH_SIZE = 16          # Batch size
    IMAGE_SIZE = 640         # Kích thước ảnh
    LEARNING_RATE = 0.01     # Learning rate
    PATIENCE = 10            # Early stopping
    # ... thêm nhiều options
```

### Detection Configuration

```python
class DetectionConfig:
    CONFIDENCE_THRESHOLD = 0.3  # Ngưỡng confidence
    DISPLAY_WIDTH = 800         # Kích thước hiển thị
    DISPLAY_HEIGHT = 600
    COLOR_NORMAL = (0, 255, 0)  # Màu xe bình thường
    COLOR_SUSPICIOUS = (0, 0, 255)  # Màu xe suspicious
    # ... thêm options
```

### Suspicious Detection Rules

```python
class SuspiciousConfig:
    MAX_SIZE_RATIO_WIDTH = 0.4  # Xe quá lớn
    MAX_SIZE_RATIO_HEIGHT = 0.4
    
    SUSPICIOUS_ZONES = [
        (200, 150, 500, 400),  # (x1, y1, x2, y2)
        # Thêm vùng nghi ngờ
    ]
```

## 📊 Training

### Chạy training với cấu hình mặc định:

```powershell
python train.py
```

### Tùy chỉnh training:

Chỉnh sửa `config/config.py` hoặc sửa trực tiếp `train.py`

### Theo dõi training:

- Logs: `logs/training.log`
- Results: `results/yolov8n_car/`
- Plots: `results/yolov8n_car/*.png`
- Best model: `models/car_detector_best.pt`

## 🎯 Detection

### Chạy detection:

```powershell
python run_detection.py
```

### Tính năng:

- ✅ Real-time detection
- ✅ Suspicious vehicle marking
- ✅ Confidence scores
- ✅ FPS counter
- ✅ Detection logging

### Logs:

- Detection log: `logs/detection.log`
- Suspicious events được log riêng

## 🔧 Advanced Usage

### Custom Detection Pipeline

```python
from core.detector import CarDetector
from core.video_processor import VideoProcessor

detector = CarDetector("models/car_detector_best.pt")
processor = VideoProcessor("video.mp4")

def my_callback(frame, frame_num):
    detections = detector.detect(frame)
    # Custom processing here
    return frame

processor.process(my_callback)
```

### Save Processed Video

```python
processor.save_video(
    output_path="output.mp4",
    frame_callback=my_callback
)
```

### Batch Processing

```python
from pathlib import Path

video_dir = Path("videos/")
for video_path in video_dir.glob("*.mp4"):
    processor = VideoProcessor(video_path)
    processor.process(my_callback)
```

## 🐛 Troubleshooting

### Import errors

Đảm bảo chạy từ project root:

```powershell
cd "g:\language\Computer vision"
python train.py
```

### CUDA out of memory

Giảm batch size trong `config/config.py`:

```python
class TrainingConfig:
    BATCH_SIZE = 8  # hoặc 4
```

### Model not found

Chạy training trước:

```powershell
python train.py
```

## 📈 Performance Tips

### Tăng tốc độ detection:

1. Giảm confidence threshold
2. Skip frames: `video_processor.process(..., skip_frames=1)`
3. Giảm display size
4. Sử dụng GPU

### Tăng độ chính xác:

1. Train lâu hơn (tăng epochs)
2. Tăng image size khi training
3. Sử dụng model lớn hơn (yolov8s, yolov8m)
4. Augmentation mạnh hơn

## 📝 Notes

- Code được tổ chức theo module, dễ maintain và mở rộng
- Mỗi module có responsibility riêng
- Configuration tập trung tại một file
- Logging đầy đủ cho debugging
- Type hints cho code clarity

## 🆕 So với version cũ

**Ưu điểm:**
- ✅ Cấu trúc rõ ràng, professional
- ✅ Dễ maintain và extend
- ✅ Reusable components
- ✅ Better logging
- ✅ Centralized configuration
- ✅ Type hints
- ✅ Documentation đầy đủ

## 🤝 Contributing

Code được chia module rõ ràng. Muốn thêm tính năng:

1. Thêm config vào `config/config.py`
2. Implement ở module tương ứng
3. Update main scripts nếu cần

---

**Happy Coding! 🚀**

