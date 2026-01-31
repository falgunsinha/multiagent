"""
Train YOLOv8 on Augmented Dataset

This script trains YOLOv8 on the augmented dataset generated from new_dataset.
The augmented dataset contains 200-300 images with realistic variations.
"""

import os
from pathlib import Path
from ultralytics import YOLO

# Fix for Windows multiprocessing issues
os.environ['PYTHONWARNINGS'] = 'ignore'
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

# Configuration
DATASET_YAML = Path("C:/isaacsim/cobotproject/datasets/augmented_yolo/dataset.yaml")
MODEL_OUTPUT_DIR = Path("C:/isaacsim/cobotproject/models")
MODEL_NAME = "augmented_shapes"

# Training parameters
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
PATIENCE = 10  # Early stopping


def train_model():
    """Train YOLOv8 model on augmented dataset"""
    
    print("="*80)
    print("YOLOv8 Training on Augmented Dataset")
    print("="*80)
    
    # Check if dataset exists
    if not DATASET_YAML.exists():
        print(f"ERROR: Dataset YAML not found: {DATASET_YAML}")
        print("Please run generate_augmented_dataset.py first!")
        return False
    
    print(f"\nDataset: {DATASET_YAML}")
    print(f"Output: {MODEL_OUTPUT_DIR / MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Load base model
    print("\nLoading YOLOv8n base model...")
    model = YOLO('yolov8n.pt')
    
    print("\nStarting training...")
    print("This will take 2-4 hours depending on your GPU")
    print("Press Ctrl+C to stop (progress will be saved)")
    print("\nPress Enter to start or Ctrl+C to cancel...")
    input()
    
    # Train model
    results = model.train(
        data=str(DATASET_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name=MODEL_NAME,
        project=str(MODEL_OUTPUT_DIR),
        patience=PATIENCE,
        save=True,
        device=0,  # GPU 0
        workers=0,  # Single-process for Windows
        val=True,
        plots=True,
        verbose=True,
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    
    model_path = MODEL_OUTPUT_DIR / MODEL_NAME / "weights" / "best.pt"
    print(f"Best model: {model_path}")
    
    # Show final metrics
    print("\nFinal metrics:")
    print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    
    # Export to TensorRT for faster inference
    print("\nExporting to TensorRT for faster inference...")
    try:
        model_best = YOLO(str(model_path))
        model_best.export(format='engine', device=0, half=True, imgsz=IMG_SIZE)
        print("✓ TensorRT engine exported successfully!")
    except Exception as e:
        print(f"Warning: TensorRT export failed: {e}")
        print("You can still use the .pt model (just slower)")
    
    return True


def update_script_model_path():
    """Show instructions to update the Isaac Sim script"""
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Update your Isaac Sim script to use the new model:")
    print("\n   In franka_rrt_physXLidar_depth_camera_v1.7.py, change:")
    print("   OLD: model_path = 'C:/isaacsim/cobotproject/models/geometric_shapes3/weights/best.pt'")
    print("   NEW: model_path = 'C:/isaacsim/cobotproject/models/augmented_shapes/weights/best.pt'")
    print("\n2. Change confidence threshold back to 0.5:")
    print("   confidence_threshold=0.5  # Model trained on Isaac Sim-like images")
    print("\n3. Run your Isaac Sim script and test detection!")
    print("\n" + "="*80)


if __name__ == "__main__":
    print("YOLOv8 Training Script for Augmented Dataset")
    print("="*80)
    print("\nThis will train a NEW model on augmented Isaac Sim-like images.")
    print("The new model should work MUCH better than the line-drawing model!")
    print("\nDataset info:")
    print("  - Source: new_dataset/ (3D rendered images)")
    print("  - Augmented: 200-300 variations per image")
    print("  - Classes: cube, cylinder, container, cuboid")
    print("  - Image size: 640x640")
    
    success = train_model()
    
    if success:
        update_script_model_path()
    else:
        print("\nTraining failed. Please check the error messages above.")

