"""
Live Car Detection using Custom Trained YOLO Model
Sử dụng model YOLO đã train với car dataset
"""

import cv2
import sys
from pathlib import Path
from ultralytics import YOLO

# Đường dẫn
VIDEO_PATH = "vid.mp4"
MODEL_PATH = "car_detector_best.pt"  # Model đã train

# Kiểm tra model có tồn tại không
if not Path(MODEL_PATH).exists():
    print(f"❌ Không tìm thấy model: {MODEL_PATH}")
    print("📝 Vui lòng chạy Train_Car_Model.py trước để training model!")
    sys.exit(1)

# Load YOLO model
print("🔄 Đang load model...")
model = YOLO(MODEL_PATH)
print(f"✓ Model loaded: {MODEL_PATH}")

# Mở video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    sys.exit("❌ Không mở được video")

# Lấy thông tin video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"📹 Video: {frame_width}x{frame_height} @ {fps} FPS")

# Kích thước hiển thị
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

# Định nghĩa vùng nghi ngờ (tùy chỉnh theo video của bạn)
SUSPICIOUS_ZONES = [
    # Ví dụ: (x1, y1, x2, y2)
    # (200, 150, 500, 400),
]

def is_suspicious(x, y, w, h, frame_w, frame_h):
    """
    Kiểm tra xe có đáng nghi không
    """
    # Xe quá lớn (chiếm > 40% khung hình)
    if w > frame_w * 0.4 or h > frame_h * 0.4:
        return True
    
    # Kiểm tra vùng nghi ngờ
    car_center_x = x + w // 2
    car_center_y = y + h // 2
    
    for (zone_x1, zone_y1, zone_x2, zone_y2) in SUSPICIOUS_ZONES:
        if zone_x1 <= car_center_x <= zone_x2 and zone_y1 <= car_center_y <= zone_y2:
            return True
    
    return False

print("\n🚀 Bắt đầu detection...")
print("Nhấn 'q' hoặc 'ESC' để thoát\n")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # YOLO inference
    results = model(frame, conf=0.3, verbose=False)  # confidence threshold = 0.3
    
    suspicious_count = 0
    car_id = 1
    
    # Xử lý kết quả detection
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # Lấy tọa độ bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            
            # Confidence score
            conf = float(box.conf[0])
            
            # Class (car)
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            # Kiểm tra suspicious
            is_susp = is_suspicious(x1, y1, w, h, frame.shape[1], frame.shape[0])
            
            if is_susp:
                suspicious_count += 1
                color = (0, 0, 255)  # Đỏ
                label = f"Car #{car_id} - SUSPICIOUS ({conf:.2f})"
                thickness = 3
            else:
                color = (0, 255, 0)  # Xanh lá
                label = f"Car #{car_id} ({conf:.2f})"
                thickness = 2
            
            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Vẽ nền cho text
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            
            # Vẽ label
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            car_id += 1
    
    # Hiển thị thông tin tổng quan
    total_cars = len(results[0].boxes) if len(results) > 0 else 0
    
    cv2.putText(frame, f"Total Cars: {total_cars}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if suspicious_count > 0:
        cv2.putText(frame, f"Suspicious: {suspicious_count}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Hiển thị FPS
    cv2.putText(frame, f"Frame: {frame_count}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Resize để hiển thị
    frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    
    cv2.imshow("YOLO Car Detection", frame_resized)
    
    # Nhấn 'q' hoặc ESC để thoát
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n✓ Đã xử lý {frame_count} frames")
print("👋 Thoát chương trình")
