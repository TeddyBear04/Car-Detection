"""
Main training script - Refactored version
Train YOLO model for car detection
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR,
    TrainingConfig, get_dataset_yaml_content
)
from core.trainer import CarModelTrainer
from utils.dataset import verify_dataset_structure, print_dataset_info, create_dataset_yaml
from utils.logger import setup_logger, log_system_info


def main():
    """Main training function"""
    
    # Setup logger
    logger = setup_logger("Training", LOGS_DIR / "training.log")
    
    logger.info("=" * 60)
    logger.info("CAR DETECTION - TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Log system info
    log_system_info()
    
    # Verify dataset
    logger.info("\n📂 Verifying dataset...")
    is_valid, message = verify_dataset_structure(DATA_DIR)
    
    if not is_valid:
        logger.error(f"❌ Dataset validation failed: {message}")
        sys.exit(1)
    
    logger.info(f"✓ {message}")
    print_dataset_info(DATA_DIR)
    
    # Create dataset YAML
    logger.info("\n📝 Creating dataset configuration...")
    yaml_path = PROJECT_ROOT / "car_dataset.yaml"
    create_dataset_yaml(DATA_DIR, yaml_path)
    logger.info(f"✓ Dataset config created: {yaml_path}")
    
    # Initialize trainer
    logger.info("\n🎓 Initializing trainer...")
    trainer = CarModelTrainer(
        model_name='yolov8n.pt',
        project_dir=RESULTS_DIR,
        experiment_name='yolov8n_car'
    )
    
    # Training configuration
    train_config = {
        'epochs': TrainingConfig.EPOCHS,
        'batch': TrainingConfig.BATCH_SIZE,
        'imgsz': TrainingConfig.IMAGE_SIZE,
        'lr0': TrainingConfig.LEARNING_RATE,
        'patience': TrainingConfig.PATIENCE,
        'save_period': TrainingConfig.SAVE_PERIOD,
        'workers': TrainingConfig.WORKERS,
        'cache': TrainingConfig.CACHE,
        'single_cls': TrainingConfig.SINGLE_CLS,
        'rect': TrainingConfig.RECT,
        'cos_lr': TrainingConfig.COS_LR,
        'close_mosaic': TrainingConfig.CLOSE_MOSAIC,
        'amp': TrainingConfig.AMP,
        'mosaic': TrainingConfig.MOSAIC,
        'mixup': TrainingConfig.MIXUP,
        'degrees': TrainingConfig.DEGREES,
        'translate': TrainingConfig.TRANSLATE,
        'scale': TrainingConfig.SCALE,
        'shear': TrainingConfig.SHEAR,
        'perspective': TrainingConfig.PERSPECTIVE,
        'flipud': TrainingConfig.FLIPUD,
        'fliplr': TrainingConfig.FLIPLR,
    }
    
    logger.info("\n🚀 Starting training...")
    logger.info(f"Configuration: {train_config}")
    
    # Train model
    try:
        results = trainer.train(
            data_yaml=yaml_path,
            **train_config
        )
        logger.info("✓ Training completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        sys.exit(1)
    
    # Validate on validation set
    logger.info("\n📊 Validating on validation set...")
    try:
        val_metrics = trainer.validate(data_yaml=yaml_path, split='val')
        logger.info("✓ Validation completed")
        
    except Exception as e:
        logger.error(f"⚠ Validation failed: {e}")
    
    # Validate on test set
    logger.info("\n📊 Validating on test set...")
    try:
        test_metrics = trainer.validate(data_yaml=yaml_path, split='test')
        logger.info("✓ Test evaluation completed")
        
    except Exception as e:
        logger.error(f"⚠ Test evaluation failed: {e}")
    
    # Export best model
    logger.info("\n💾 Exporting best model...")
    try:
        best_model_path = trainer.export_best_model(MODELS_DIR / "car_detector_best.pt")
        logger.info(f"✓ Best model exported to: {best_model_path}")
        
    except Exception as e:
        logger.error(f"⚠ Model export failed: {e}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Training results: {trainer.get_training_results_path()}")
    logger.info(f"Best model: {MODELS_DIR / 'car_detector_best.pt'}")
    logger.info(f"Logs: {LOGS_DIR}")
    logger.info("=" * 60)
    
    print("\n✅ Training pipeline completed successfully!")
    print(f"📁 Results saved to: {RESULTS_DIR}")
    print(f"🎯 Best model: {MODELS_DIR / 'car_detector_best.pt'}")
    print(f"\n👉 Next step: Run 'python run_detection.py' to use the model")


if __name__ == "__main__":
    main()
