"""
Model training class
"""

import torch
from pathlib import Path
from typing import Dict
from ultralytics import YOLO
import shutil


class CarModelTrainer:
    """
    YOLO model trainer for car detection
    """
    
    def __init__(
        self,
        model_name: str = 'yolov8n.pt',
        project_dir: Path = None,
        experiment_name: str = 'car_detection'
    ):
        """
        Initialize trainer
        
        Args:
            model_name: Pre-trained model name
            project_dir: Directory to save results
            experiment_name: Experiment name
        """
        self.model_name = model_name
        self.project_dir = project_dir or Path('results')
        self.experiment_name = experiment_name
        
        # Load pre-trained model
        self.model = YOLO(model_name)
        print(f"✓ Pre-trained model loaded: {model_name}")
    
    def train(
        self,
        data_yaml: Path,
        epochs: int = 50,
        batch_size: int = 16,
        img_size: int = 640,
        device: str = None,
        **kwargs
    ) -> Dict:
        """
        Train model
        
        Args:
            data_yaml: Path to dataset YAML file
            epochs: Number of epochs
            batch_size: Batch size
            img_size: Image size
            device: Device to use ('cpu', 0, '0,1', etc.)
            **kwargs: Additional training arguments
        
        Returns:
            Training results
        """
        # Auto-detect device if not specified
        if device is None:
            device = 0 if torch.cuda.is_available() else 'cpu'
        
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        print(f"Dataset: {data_yaml}")
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Image Size: {img_size}")
        print(f"Device: {device}")
        print("=" * 60 + "\n")
        
        # Train
        results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            project=str(self.project_dir),
            name=self.experiment_name,
            exist_ok=True,
            verbose=True,
            **kwargs
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60 + "\n")
        
        return results
    
    def validate(self, data_yaml: Path = None, split: str = 'val') -> Dict:
        """
        Validate model
        
        Args:
            data_yaml: Path to dataset YAML file
            split: Dataset split to validate on ('val' or 'test')
        
        Returns:
            Validation results
        """
        print("\n" + "=" * 60)
        print(f"VALIDATING ON {split.upper()} SET")
        print("=" * 60 + "\n")
        
        if data_yaml:
            metrics = self.model.val(data=str(data_yaml), split=split)
        else:
            metrics = self.model.val(split=split)
        
        # Print metrics
        print(f"\nmAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.p[0]:.4f}")
        print(f"Recall: {metrics.box.r[0]:.4f}")
        
        return metrics
    
    def export_best_model(self, output_path: Path) -> Path:
        """
        Export best trained model
        
        Args:
            output_path: Path to save best model
        
        Returns:
            Path to exported model
        """
        # Find best model
        best_model_path = self.project_dir / self.experiment_name / 'weights' / 'best.pt'
        
        if not best_model_path.exists():
            raise FileNotFoundError(f"Best model not found at {best_model_path}")
        
        # Copy to output path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_model_path, output_path)
        
        print(f"\n✓ Best model exported to: {output_path}")
        
        return output_path
    
    def get_training_results_path(self) -> Path:
        """Get path to training results directory"""
        return self.project_dir / self.experiment_name
