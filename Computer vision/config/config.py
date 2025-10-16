"""
Configuration file for Car Detection Project
Chứa tất cả các cấu hình, đường dẫn, hyperparameters
"""

import os
from pathlib import Path

# ===== PROJECT PATHS =====
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Car data set" / "car_dataset-master"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Tạo các thư mục nếu chưa có
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ===== DATA PATHS =====
TRAIN_IMAGES = DATA_DIR / "train" / "images"
TRAIN_LABELS = DATA_DIR / "train" / "labels"
VALID_IMAGES = DATA_DIR / "valid" / "images"
VALID_LABELS = DATA_DIR / "valid" / "labels"
TEST_IMAGES = DATA_DIR / "test" / "images"
TEST_LABELS = DATA_DIR / "test" / "labels"

# ===== VIDEO PATHS =====
VIDEO_PATH = PROJECT_ROOT / "vid.mp4"

# ===== MODEL PATHS =====
PRETRAINED_MODEL = "yolov8n.pt"  # Pretrained YOLO model
BEST_MODEL_PATH = MODELS_DIR / "car_detector_best.pt"
LAST_MODEL_PATH = MODELS_DIR / "car_detector_last.pt"

# ===== TRAINING CONFIG =====
class TrainingConfig:
    """Training hyperparameters"""
    EPOCHS = 50
    BATCH_SIZE = 16
    IMAGE_SIZE = 640
    LEARNING_RATE = 0.01
    PATIENCE = 10  # Early stopping patience
    SAVE_PERIOD = 5  # Save checkpoint every N epochs
    WORKERS = 4
    
    # Augmentation
    MOSAIC = 1.0
    MIXUP = 0.0
    DEGREES = 0.0  # Rotation
    TRANSLATE = 0.1
    SCALE = 0.5
    SHEAR = 0.0
    PERSPECTIVE = 0.0
    FLIPUD = 0.0  # Vertical flip
    FLIPLR = 0.5  # Horizontal flip
    
    # Advanced settings
    CACHE = False  # Cache images for faster training
    SINGLE_CLS = True  # Single class detection
    RECT = False  # Rectangular training
    COS_LR = False  # Cosine learning rate scheduler
    CLOSE_MOSAIC = 10  # Disable mosaic augmentation for last N epochs
    AMP = True  # Automatic Mixed Precision

# ===== DETECTION CONFIG =====
class DetectionConfig:
    """Detection parameters"""
    CONFIDENCE_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.45
    MAX_DETECTIONS = 100
    
    # Display settings
    DISPLAY_WIDTH = 800
    DISPLAY_HEIGHT = 600
    SHOW_CONFIDENCE = True
    SHOW_FPS = True
    
    # Colors (BGR format)
    COLOR_NORMAL = (0, 255, 0)  # Green
    COLOR_SUSPICIOUS = (0, 0, 255)  # Red
    COLOR_TEXT = (255, 255, 255)  # White
    COLOR_INFO = (255, 255, 0)  # Yellow
    
    # Thickness
    BOX_THICKNESS_NORMAL = 2
    BOX_THICKNESS_SUSPICIOUS = 3
    TEXT_THICKNESS = 2

# ===== SUSPICIOUS DETECTION CONFIG =====
class SuspiciousConfig:
    """Rules for suspicious vehicle detection"""
    # Size-based detection
    MAX_SIZE_RATIO_WIDTH = 0.4  # xe chiếm > 40% chiều rộng frame
    MAX_SIZE_RATIO_HEIGHT = 0.4  # xe chiếm > 40% chiều cao frame
    
    # Zone-based detection (x1, y1, x2, y2)
    SUSPICIOUS_ZONES = [
        # Ví dụ: (200, 150, 500, 400),
        # Thêm các vùng nghi ngờ tại đây
    ]
    
    # Speed-based detection (nếu có tracking)
    MIN_SUSPICIOUS_SPEED = 0  # km/h
    MAX_SUSPICIOUS_SPEED = 999  # km/h

# ===== DATASET CONFIG =====
class DatasetConfig:
    """Dataset information"""
    NUM_CLASSES = 1
    CLASS_NAMES = ['car']
    
    # Dataset statistics (optional)
    TRAIN_SIZE = None  # Sẽ được tính tự động
    VALID_SIZE = None
    TEST_SIZE = None

# ===== LOGGING CONFIG =====
class LoggingConfig:
    """Logging configuration"""
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = LOGS_DIR / "training.log"
    CONSOLE_OUTPUT = True

# ===== DEVICE CONFIG =====
class DeviceConfig:
    """Device configuration for training/inference"""
    # Auto-detect CUDA
    import torch
    USE_GPU = torch.cuda.is_available()
    DEVICE = 0 if USE_GPU else 'cpu'
    
    # Memory settings
    GPU_MEMORY_FRACTION = 0.9  # Use 90% of GPU memory

# ===== EXPORT SETTINGS =====
def get_dataset_yaml_content():
    """Generate YAML content for dataset configuration"""
    return f"""# Car Dataset Configuration
path: {str(DATA_DIR)}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: valid/images    # val images (relative to 'path')
test: test/images    # test images (optional)

# Classes
names:
  0: car

# Number of classes
nc: {DatasetConfig.NUM_CLASSES}
"""

# ===== PRINT CONFIG =====
def print_config():
    """Print current configuration"""
    print("=" * 60)
    print("CAR DETECTION PROJECT - CONFIGURATION")
    print("=" * 60)
    print(f"\n📂 Paths:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Dataset: {DATA_DIR}")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    
    print(f"\n🎓 Training:")
    print(f"  Epochs: {TrainingConfig.EPOCHS}")
    print(f"  Batch Size: {TrainingConfig.BATCH_SIZE}")
    print(f"  Image Size: {TrainingConfig.IMAGE_SIZE}")
    print(f"  Learning Rate: {TrainingConfig.LEARNING_RATE}")
    
    print(f"\n🔍 Detection:")
    print(f"  Confidence: {DetectionConfig.CONFIDENCE_THRESHOLD}")
    print(f"  IOU Threshold: {DetectionConfig.IOU_THRESHOLD}")
    print(f"  Display: {DetectionConfig.DISPLAY_WIDTH}x{DetectionConfig.DISPLAY_HEIGHT}")
    
    print(f"\n💻 Device:")
    print(f"  GPU Available: {DeviceConfig.USE_GPU}")
    print(f"  Device: {DeviceConfig.DEVICE}")
    
    print("=" * 60)

if __name__ == "__main__":
    print_config()
