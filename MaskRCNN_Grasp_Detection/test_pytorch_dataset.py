"""
Quick test to verify dataset loading works
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATASET_DIR = "Mask_RCNN/datasets/project"

def test_dataset():
    """Test dataset loading"""
    print("=" * 60)
    print("Testing Dataset Loading")
    print("=" * 60)
    
    # Check train set
    train_dir = os.path.join(DATASET_DIR, 'train')
    val_dir = os.path.join(DATASET_DIR, 'val')
    
    print(f"\nTrain directory: {train_dir}")
    print(f"Val directory: {val_dir}")
    
    # Load val annotations
    val_annotations_path = os.path.join(val_dir, 'via_region_data.json')
    
    if not os.path.exists(val_annotations_path):
        print(f"\nERROR: Annotations not found at {val_annotations_path}")
        return
    
    with open(val_annotations_path) as f:
        annotations = json.load(f)
    
    print(f"\nTotal annotations: {len(annotations)}")
    
    # Count classes
    class_counts = {'cube': 0, 'cylinder': 0, 'cuboid': 0, 'gripper': 0, 'grasp': 0}
    
    for ann in annotations.values():
        if not ann['regions']:
            continue
        
        for region in ann['regions'].values():
            class_name = region['region_attributes'].get('name', '')
            if class_name in class_counts:
                class_counts[class_name] += 1
    
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    # Show first image
    first_ann = list(annotations.values())[0]
    img_path = os.path.join(val_dir, first_ann['filename'])
    
    if os.path.exists(img_path):
        print(f"\nFirst image: {first_ann['filename']}")
        print(f"  Regions: {len(first_ann['regions'])}")
        
        img = Image.open(img_path)
        print(f"  Size: {img.size}")
        
        # Show classes in first image
        for i, region in enumerate(first_ann['regions'].values()):
            class_name = region['region_attributes'].get('name', '')
            print(f"  Region {i}: {class_name}")
    
    print("\n" + "=" * 60)
    print("Dataset test complete!")
    print("=" * 60)
    print("\nDataset is ready for training.")
    print("\nNext step: Run train_pytorch.bat")
    print("=" * 60)


if __name__ == "__main__":
    test_dataset()

