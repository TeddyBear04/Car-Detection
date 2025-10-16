"""
Logging utilities for Car Detection Project
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str, log_file: Path = None, level: str = "INFO"):
    """
    Setup logger with console and file handlers
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        logging.Logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TrainingLogger:
    """Custom logger for training progress"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create training log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_{timestamp}.log"
        
        self.logger = setup_logger("Training", self.log_file)
    
    def log_start(self, config: dict):
        """Log training start"""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING STARTED")
        self.logger.info("=" * 60)
        
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
    
    def log_epoch(self, epoch: int, metrics: dict):
        """Log epoch results"""
        self.logger.info(f"Epoch {epoch}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_finish(self, best_metrics: dict):
        """Log training finish"""
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 60)
        
        self.logger.info("Best Metrics:")
        for key, value in best_metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")


class DetectionLogger:
    """Custom logger for detection"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create detection log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"detection_{timestamp}.log"
        
        self.logger = setup_logger("Detection", self.log_file)
    
    def log_detection(self, frame_num: int, num_cars: int, suspicious: int):
        """Log detection results"""
        self.logger.debug(
            f"Frame {frame_num}: {num_cars} cars detected, "
            f"{suspicious} suspicious"
        )
    
    def log_suspicious_event(self, frame_num: int, car_id: int, reason: str):
        """Log suspicious vehicle event"""
        self.logger.warning(
            f"Frame {frame_num}: Car #{car_id} marked as suspicious - {reason}"
        )


def log_system_info():
    """Log system information"""
    import torch
    import cv2
    from ultralytics import __version__ as yolo_version
    
    logger = logging.getLogger("SystemInfo")
    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"OpenCV: {cv2.__version__}")
    logger.info(f"YOLOv8: {yolo_version}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    logger.info("=" * 60)
