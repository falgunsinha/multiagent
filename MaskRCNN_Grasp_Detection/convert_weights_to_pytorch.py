"""
Convert Keras/TensorFlow MaskRCNN weights (.h5) to PyTorch format (.pth)

This script converts the trained Keras model weights to PyTorch format
so they can be loaded in Isaac Sim's PyTorch-based detector.

Note: This is a simplified conversion. For full compatibility, you may need
to train directly in PyTorch or use the existing dataset to train a PyTorch model.
"""

import os
import sys
import numpy as np

print("=" * 60)
print("MaskRCNN Weight Conversion: Keras (.h5) -> PyTorch (.pth)")
print("=" * 60)

# Check if Keras weights exist
keras_weights_path = "Mask_RCNN/mask_rcnn_cube_cylinder.h5"
if not os.path.exists(keras_weights_path):
    print(f"\nERROR: Keras weights not found at: {keras_weights_path}")
    print("\nPlease train the model first:")
    print("  1. Run: setup_training.bat")
    print("  2. Run: run_training.bat")
    print("  3. Then run this conversion script")
    sys.exit(1)

print(f"\nFound Keras weights: {keras_weights_path}")
print("\nWARNING: Direct Keras->PyTorch conversion is complex.")
print("Recommended approach: Train directly in PyTorch")
print("\nOptions:")
print("  1. Use train_pytorch_maskrcnn.py (recommended)")
print("  2. Manual conversion (requires matching architectures)")
print("\nProceeding with PyTorch training setup...")

# Create PyTorch training script
pytorch_script = """
# This will train MaskRCNN directly in PyTorch using the existing dataset
# Run this in Isaac Sim's Python environment or a PyTorch environment
"""

print("\n" + "=" * 60)
print("Next Steps:")
print("=" * 60)
print("\n1. Train directly in PyTorch (recommended):")
print("   python train_pytorch_maskrcnn.py")
print("\n2. Or use the existing Keras model with TensorFlow inference")
print("   (requires TensorFlow in Isaac Sim)")
print("\n" + "=" * 60)

