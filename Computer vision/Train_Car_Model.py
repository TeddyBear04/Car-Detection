"""
Script training YOLO model để nhận diện xe ô tô
Dataset: Stanford Cars Dataset (YOLO format)
"""

from ultralytics import YOLO
import torch
import os

# Kiểm tra CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Đường dẫn dataset
DATASET_ROOT = r"G:\language\Computer vision\Car data set\car_dataset-master"

# Tạo file cấu hình dataset YAML
yaml_content = f"""# Car Dataset Configuration
path: {DATASET_ROOT}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: valid/images    # val images (relative to 'path')
test: test/images    # test images (optional)

# Classes
names:
  0: car
"""

# Lưu file YAML
yaml_path = os.path.join(os.path.dirname(__file__), "car_dataset.yaml")
with open(yaml_path, 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print(f"Dataset config saved to: {yaml_path}")

# Load pre-trained YOLOv8 model
# Sử dụng YOLOv8n (nano) - nhẹ và nhanh, phù hợp cho laptop
model = YOLO('yolov8n.pt')  # YOLOv8 nano model

print("\n" + "="*50)
print("BẮT ĐẦU TRAINING MODEL")
print("="*50 + "\n")

# Training
results = model.train(
    data=yaml_path,          # file cấu hình dataset
    epochs=50,               # số epoch training (có thể điều chỉnh)
    imgsz=640,               # kích thước ảnh input
    batch=16,                # batch size (giảm xuống nếu hết RAM)
    device=0 if torch.cuda.is_available() else 'cpu',  # GPU nếu có, không thì CPU
    workers=4,               # số workers để load data
    project='car_detection', # thư mục lưu kết quả
    name='yolov8n_car',      # tên experiment
    exist_ok=True,           # cho phép ghi đè
    patience=10,             # early stopping patience
    save=True,               # lưu checkpoint
    save_period=5,           # lưu mỗi 5 epochs
    cache=False,             # không cache (tốn RAM)
    pretrained=True,         # sử dụng pretrained weights
    optimizer='auto',        # tự động chọn optimizer
    verbose=True,            # hiển thị chi tiết
    seed=42,                 # random seed
    deterministic=True,      # training deterministic
    single_cls=True,         # chỉ 1 class (car)
    rect=False,              # rectangular training
    cos_lr=False,            # cosine learning rate
    close_mosaic=10,         # tắt mosaic augmentation 10 epochs cuối
    resume=False,            # tiếp tục từ checkpoint (nếu có)
    amp=True,                # Automatic Mixed Precision training
)

print("\n" + "="*50)
print("TRAINING HOÀN TẤT!")
print("="*50)

# Đánh giá model trên validation set
print("\n" + "="*50)
print("ĐÁNH GIÁ MODEL TRÊN VALIDATION SET")
print("="*50 + "\n")

metrics = model.val()

print(f"\nmAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p[0]:.4f}")
print(f"Recall: {metrics.box.r[0]:.4f}")

# Đánh giá trên test set
print("\n" + "="*50)
print("ĐÁNH GIÁ MODEL TRÊN TEST SET")
print("="*50 + "\n")

test_metrics = model.val(data=yaml_path, split='test')

print(f"\nmAP50: {test_metrics.box.map50:.4f}")
print(f"mAP50-95: {test_metrics.box.map:.4f}")

# Lưu model đã train
best_model_path = os.path.join('car_detection', 'yolov8n_car', 'weights', 'best.pt')
final_model_path = os.path.join(os.path.dirname(__file__), 'car_detector_best.pt')

if os.path.exists(best_model_path):
    import shutil
    shutil.copy(best_model_path, final_model_path)
    print(f"\n✓ Model tốt nhất đã được lưu tại: {final_model_path}")
    print(f"✓ Sử dụng model này trong Live_Car_Detector.py")
else:
    print(f"\n⚠ Không tìm thấy best model tại {best_model_path}")

print("\n" + "="*50)
print("TRAINING PIPELINE HOÀN TẤT!")
print("="*50)
print("\nKết quả training được lưu tại: car_detection/yolov8n_car/")
print("- Weights: car_detection/yolov8n_car/weights/")
print("- Logs: car_detection/yolov8n_car/")
print("- Visualizations: car_detection/yolov8n_car/")
