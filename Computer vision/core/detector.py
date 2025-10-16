"""
Car Detector class using YOLO
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from ultralytics import YOLO


class CarDetector:
    """
    Car Detection class using YOLOv8
    """
    
    def __init__(
        self,
        model_path: Path,
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.45
    ):
        """
        Initialize detector
        
        Args:
            model_path: Path to YOLO model
            conf_threshold: Confidence threshold
            iou_threshold: IOU threshold for NMS
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load model
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = YOLO(str(model_path))
        print(f"✓ Model loaded: {model_path}")
    
    def detect(
        self,
        image: np.ndarray,
        conf: float = None,
        iou: float = None
    ) -> List[Dict]:
        """
        Detect cars in image
        
        Args:
            image: Input image (BGR)
            conf: Confidence threshold (override default)
            iou: IOU threshold (override default)
        
        Returns:
            List of detections, each containing:
                - bbox: (x1, y1, x2, y2)
                - confidence: float
                - class_id: int
                - class_name: str
        """
        conf = conf or self.conf_threshold
        iou = iou or self.iou_threshold
        
        # Run inference
        results = self.model(image, conf=conf, iou=iou, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Extract box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Extract confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]
                
                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                }
                
                detections.append(detection)
        
        return detections
    
    def get_bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Get center point of bounding box
        
        Args:
            bbox: (x1, y1, x2, y2)
        
        Returns:
            (center_x, center_y)
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return center_x, center_y
    
    def get_bbox_size(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Get size of bounding box
        
        Args:
            bbox: (x1, y1, x2, y2)
        
        Returns:
            (width, height)
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return width, height
    
    def is_suspicious_size(
        self,
        bbox: Tuple[int, int, int, int],
        frame_size: Tuple[int, int],
        max_ratio_w: float = 0.4,
        max_ratio_h: float = 0.4
    ) -> bool:
        """
        Check if car size is suspicious
        
        Args:
            bbox: (x1, y1, x2, y2)
            frame_size: (frame_width, frame_height)
            max_ratio_w: Maximum width ratio
            max_ratio_h: Maximum height ratio
        
        Returns:
            True if suspicious
        """
        frame_w, frame_h = frame_size
        width, height = self.get_bbox_size(bbox)
        
        if width > frame_w * max_ratio_w or height > frame_h * max_ratio_h:
            return True
        
        return False
    
    def is_in_zone(
        self,
        bbox: Tuple[int, int, int, int],
        zone: Tuple[int, int, int, int]
    ) -> bool:
        """
        Check if car center is in zone
        
        Args:
            bbox: Car bounding box (x1, y1, x2, y2)
            zone: Zone coordinates (x1, y1, x2, y2)
        
        Returns:
            True if car is in zone
        """
        center_x, center_y = self.get_bbox_center(bbox)
        zone_x1, zone_y1, zone_x2, zone_y2 = zone
        
        if zone_x1 <= center_x <= zone_x2 and zone_y1 <= center_y <= zone_y2:
            return True
        
        return False
    
    def is_suspicious(
        self,
        bbox: Tuple[int, int, int, int],
        frame_size: Tuple[int, int],
        suspicious_zones: List[Tuple[int, int, int, int]] = None,
        max_ratio_w: float = 0.4,
        max_ratio_h: float = 0.4
    ) -> Tuple[bool, str]:
        """
        Check if detection is suspicious
        
        Args:
            bbox: (x1, y1, x2, y2)
            frame_size: (frame_width, frame_height)
            suspicious_zones: List of suspicious zones
            max_ratio_w: Maximum width ratio
            max_ratio_h: Maximum height ratio
        
        Returns:
            Tuple of (is_suspicious, reason)
        """
        # Check size
        if self.is_suspicious_size(bbox, frame_size, max_ratio_w, max_ratio_h):
            return True, "Abnormal size"
        
        # Check zones
        if suspicious_zones:
            for zone in suspicious_zones:
                if self.is_in_zone(bbox, zone):
                    return True, "Restricted zone"
        
        return False, ""
