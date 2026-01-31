"""
Export YOLOv8 Model to TensorRT Engine (Offline)

Run this BEFORE using the model in Isaac Sim to avoid GPU conflicts.
This creates a TensorRT engine file for 2-3x faster inference.

Usage:
    C:\isaacsim\python.bat export_yolov8_tensorrt.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(r"C:\isaacsim\cobotproject")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def export_tensorrt(model_path: str, use_fp16: bool = False):
    """
    Export YOLOv8 model to TensorRT engine.
    
    Args:
        model_path: Path to YOLOv8 .pt model
        use_fp16: Use FP16 precision (faster but less stable)
    """
    try:
        from ultralytics import YOLO
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"ERROR: Model not found: {model_path}")
            return False
        
        print("="*80)
        print("YOLOv8 TensorRT Export")
        print("="*80)
        print(f"Model: {model_path}")
        print(f"Precision: {'FP16 (faster)' if use_fp16 else 'FP32 (more stable)'}")
        print()
        
        # Load model
        print("Loading YOLOv8 model...")
        model = YOLO(str(model_path))
        print("Model loaded successfully!")
        
        # Check if TensorRT engine already exists
        engine_path = model_path.parent / f"{model_path.stem}.engine"
        
        if engine_path.exists():
            print(f"\nWARNING: TensorRT engine already exists: {engine_path}")
            response = input("Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Export cancelled.")
                return False
            
            # Delete existing engine
            engine_path.unlink()
            print("Deleted existing engine.")
        
        # Export to TensorRT
        print("\nExporting to TensorRT...")
        print("This may take 2-5 minutes...")
        print("(Creating optimized GPU kernels for your RTX 3500)")
        
        try:
            model.export(
                format='engine',
                device=0,  # GPU 0
                half=use_fp16,  # FP16 or FP32
                imgsz=640,  # Input size
                batch=1,  # Batch size
                workspace=4,  # Workspace size in GB
                verbose=True
            )
            
            print("\n" + "="*80)
            print("SUCCESS! TensorRT engine created!")
            print("="*80)
            print(f"Engine saved to: {engine_path}")
            print(f"File size: {engine_path.stat().st_size / (1024*1024):.1f} MB")
            print()
            print("You can now use this engine in Isaac Sim for 2-3x faster inference!")
            print()
            
            return True
            
        except Exception as export_error:
            print(f"\nERROR: TensorRT export failed: {export_error}")
            print("\nPossible reasons:")
            print("1. TensorRT not installed (Isaac Sim may not include it)")
            print("2. CUDA version mismatch")
            print("3. Insufficient GPU memory")
            print("\nFalling back to PyTorch inference is recommended.")
            return False
        
    except ImportError:
        print("ERROR: Ultralytics not installed!")
        print("Install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\nYOLOv8 TensorRT Export Tool")
    print("="*80)
    
    # Model path
    model_path = "C:/isaacsim/cobotproject/models/geometric_shapes3/weights/best.pt"
    
    print(f"\nModel: {model_path}")
    print("\nChoose precision:")
    print("1. FP32 (more stable, recommended for Isaac Sim)")
    print("2. FP16 (faster, but may cause GPU conflicts)")
    
    choice = input("\nEnter choice (1 or 2, default=1): ").strip()
    
    use_fp16 = (choice == "2")
    
    print("\nStarting export...")
    success = export_tensorrt(model_path, use_fp16=use_fp16)
    
    if success:
        print("\nNext steps:")
        print("1. Run your Isaac Sim script")
        print("2. The TensorRT engine will be loaded automatically")
        print("3. Enjoy 2-3x faster object detection!")
    else:
        print("\nTensorRT export failed.")
        print("Your script will use standard PyTorch inference instead.")
        print("(Still fast enough for 10 Hz detection)")

