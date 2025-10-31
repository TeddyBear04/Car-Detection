"""
Main detection script - Refactored version
Run car detection on video using trained YOLO model
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    VIDEO_PATH, BEST_MODEL_PATH,
    DetectionConfig, SuspiciousConfig
)
from core.detector import CarDetector
from core.video_processor import VideoProcessor
from utils.visualization import draw_box_with_label, draw_info_panel, draw_fps
from utils.logger import setup_logger, DetectionLogger


def main():
    """Main detection function"""
    
    # Setup logger
    from config.config import LOGS_DIR
    logger = setup_logger("Detection", LOGS_DIR / "detection.log")
    detection_logger = DetectionLogger(LOGS_DIR)
    
    logger.info("=" * 60)
    logger.info("CAR DETECTION - VIDEO PROCESSING")
    logger.info("=" * 60)
    
    # Check video exists
    if not VIDEO_PATH.exists():
        logger.error(f"❌ Video not found: {VIDEO_PATH}")
        sys.exit(1)
    
    # Check model exists
    if not BEST_MODEL_PATH.exists():
        logger.error(f"❌ Model not found: {BEST_MODEL_PATH}")
        logger.error("👉 Please run 'python train.py' first to train the model")
        sys.exit(1)
    
    # Initialize detector
    logger.info("\n🔄 Loading detector...")
    detector = CarDetector(
        model_path=BEST_MODEL_PATH,
        conf_threshold=DetectionConfig.CONFIDENCE_THRESHOLD,
        iou_threshold=DetectionConfig.IOU_THRESHOLD
    )
    
    # Initialize video processor
    logger.info(f"\n📹 Loading video: {VIDEO_PATH}")
    video_processor = VideoProcessor(VIDEO_PATH)
    
    # Detection callback
    def process_frame(frame, frame_num):
        """Process each frame"""
        
        # Detect cars
        detections = detector.detect(frame)
        
        # Count suspicious vehicles
        suspicious_count = 0
        
        # Draw detections
        for idx, detection in enumerate(detections, 1):
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            # Check if suspicious
            frame_size = (frame.shape[1], frame.shape[0])
            is_susp, reason = detector.is_suspicious(
                bbox,
                frame_size,
                suspicious_zones=SuspiciousConfig.SUSPICIOUS_ZONES,
                max_ratio_w=SuspiciousConfig.MAX_SIZE_RATIO_WIDTH,
                max_ratio_h=SuspiciousConfig.MAX_SIZE_RATIO_HEIGHT
            )
            
            if is_susp:
                suspicious_count += 1
                color = DetectionConfig.COLOR_SUSPICIOUS
                thickness = DetectionConfig.BOX_THICKNESS_SUSPICIOUS
                label = f"Car #{idx} - SUSPICIOUS ({confidence:.2f})"
                
                # Log suspicious event
                detection_logger.log_suspicious_event(frame_num, idx, reason)
            else:
                color = DetectionConfig.COLOR_NORMAL
                thickness = DetectionConfig.BOX_THICKNESS_NORMAL
                label = f"Car #{idx} ({confidence:.2f})" if DetectionConfig.SHOW_CONFIDENCE else f"Car #{idx}"
            
            # Draw bounding box with label
            frame = draw_box_with_label(
                frame, bbox, label, color, thickness,
                text_color=DetectionConfig.COLOR_TEXT
            )
        
        # Log detection
        detection_logger.log_detection(frame_num, len(detections), suspicious_count)
        
        # Draw info panel
        info = {
            'Total Cars': len(detections),
            'Frame': frame_num
        }
        
        if suspicious_count > 0:
            info['Suspicious'] = suspicious_count
        
        if DetectionConfig.SHOW_FPS:
            info['FPS'] = f"{video_processor.get_fps():.1f}"
        
        frame = draw_info_panel(frame, info, position='top-left')
        
        return frame
    
    # Process video
    logger.info("\n Starting detection...")
    logger.info("Press 'q' or 'ESC' to quit\n")
    
    try:
        video_processor.process(
            frame_callback=process_frame,
            display_size=(DetectionConfig.DISPLAY_WIDTH, DetectionConfig.DISPLAY_HEIGHT),
            window_name="Car Detection - YOLO"
        )
        
        logger.info("\n✓ Detection completed")
        
    except KeyboardInterrupt:
        logger.info("\n⚠ Detection interrupted by user")
        
    except Exception as e:
        logger.error(f"\n❌ Error during detection: {e}")
        sys.exit(1)
    
    logger.info("=" * 60)
    print("\n Detection finished")


if __name__ == "__main__":
    main()

