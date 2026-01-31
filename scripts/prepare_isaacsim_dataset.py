"""
Prepare Isaac Sim Dataset for YOLOv8 Training

This script helps you create a training dataset from Isaac Sim camera captures.
Since the generic geometric shapes dataset (line drawings) doesn't match Isaac Sim's
photorealistic rendering, you need to train on actual Isaac Sim images.

QUICK START:
1. Run your Isaac Sim script
2. It will save camera images to: cobotproject/models/geometric_shapes3/images/
3. Run this script to organize them for training
4. Use Roboflow.com to label the images (free, easy)
5. Train a new model on the labeled data
"""

import cv2
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime

# Paths
ISAAC_SIM_IMAGES = Path("C:/isaacsim/cobotproject/models/geometric_shapes3/images")
OUTPUT_DATASET = Path("C:/isaacsim/cobotproject/datasets/isaacsim_yolo")
OUTPUT_IMAGES = OUTPUT_DATASET / "images"
OUTPUT_LABELS = OUTPUT_DATASET / "labels"

def check_dataset_mismatch():
    """
    Compare training dataset images with Isaac Sim camera images
    to show the mismatch problem.
    """
    print("="*80)
    print("DATASET MISMATCH ANALYSIS")
    print("="*80)
    
    # Check training dataset
    training_dataset = Path("C:/isaacsim/cobotproject/datasets/geometric_shapes_yolo/train/images")
    if training_dataset.exists():
        sample_files = list(training_dataset.glob("*.png"))[:3]
        if sample_files:
            print("\nTraining dataset sample images:")
            for f in sample_files:
                img = cv2.imread(str(f))
                if img is not None:
                    print(f"  {f.name}: {img.shape}, mean={img.mean():.1f}, std={img.std():.1f}")
    
    # Check Isaac Sim captures
    if ISAAC_SIM_IMAGES.exists():
        isaac_files = list(ISAAC_SIM_IMAGES.glob("*.png"))
        if isaac_files:
            print("\nIsaac Sim captured images:")
            for f in isaac_files:
                img = cv2.imread(str(f))
                if img is not None:
                    print(f"  {f.name}: {img.shape}, mean={img.mean():.1f}, std={img.std():.1f}")
    
    print("\n" + "="*80)
    print("PROBLEM IDENTIFIED:")
    print("="*80)
    print("Training dataset: Simple 2D line drawings (black & white)")
    print("Isaac Sim images: Photorealistic 3D renders (full color)")
    print("\nThese look COMPLETELY different!")
    print("The model learned to detect line drawings, NOT 3D objects.")
    print("\nSOLUTION: Train on Isaac Sim images instead!")
    print("="*80)


def prepare_dataset_structure():
    """Create dataset directory structure"""
    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)
    print(f"\nCreated dataset structure:")
    print(f"  Images: {OUTPUT_IMAGES}")
    print(f"  Labels: {OUTPUT_LABELS}")


def copy_isaac_sim_images():
    """Copy Isaac Sim camera captures to dataset"""
    if not ISAAC_SIM_IMAGES.exists():
        print(f"ERROR: Isaac Sim images directory not found: {ISAAC_SIM_IMAGES}")
        print("Please run your Isaac Sim script first to capture images!")
        return 0
    
    # Find all PNG images
    image_files = list(ISAAC_SIM_IMAGES.glob("*.png"))
    
    if len(image_files) == 0:
        print(f"No images found in {ISAAC_SIM_IMAGES}")
        return 0
    
    print(f"\nFound {len(image_files)} images in Isaac Sim directory")
    print("Copying to dataset...")
    
    copied = 0
    for img_file in image_files:
        # Copy image
        dest_img = OUTPUT_IMAGES / f"isaacsim_{copied:04d}.png"
        shutil.copy(img_file, dest_img)
        
        # Create empty label file (to be filled by labeling tool)
        dest_label = OUTPUT_LABELS / f"isaacsim_{copied:04d}.txt"
        dest_label.touch()
        
        copied += 1
    
    print(f"Copied {copied} images successfully!")
    return copied


def create_labeling_instructions():
    """Create instructions for labeling"""
    readme = OUTPUT_DATASET / "README.md"
    
    with open(readme, 'w') as f:
        f.write("# Isaac Sim YOLOv8 Training Dataset\n\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Classes\n\n")
        f.write("0: cube (small cubes to pick)\n")
        f.write("1: cylinder (small cylinders to pick)\n")
        f.write("2: container (large container to place in)\n")
        f.write("3: obstacle (robot, large objects, etc.)\n\n")
        f.write("## Next Steps\n\n")
        f.write("### Option 1: Use Roboflow (Recommended - Easiest)\n\n")
        f.write("1. Go to https://roboflow.com/ and create free account\n")
        f.write("2. Create new project: 'Isaac Sim Pick and Place'\n")
        f.write("3. Upload images from: images/\n")
        f.write("4. Label each image:\n")
        f.write("   - Draw boxes around small cubes → label as 'cube'\n")
        f.write("   - Draw boxes around cylinders → label as 'cylinder'\n")
        f.write("   - Draw boxes around container → label as 'container'\n")
        f.write("   - Draw boxes around robot/obstacles → label as 'obstacle'\n")
        f.write("5. Export as 'YOLOv8' format\n")
        f.write("6. Download and use for training\n\n")
        f.write("### Option 2: Use LabelImg (Local tool)\n\n")
        f.write("1. Install: pip install labelImg\n")
        f.write("2. Run: labelImg\n")
        f.write("3. Open directory: images/\n")
        f.write("4. Label images and save in YOLO format\n\n")
        f.write("## Training\n\n")
        f.write("After labeling, train YOLOv8 on this dataset:\n")
        f.write("```python\n")
        f.write("from ultralytics import YOLO\n")
        f.write("model = YOLO('yolov8n.pt')\n")
        f.write("model.train(data='dataset.yaml', epochs=50)\n")
        f.write("```\n")
    
    print(f"\nCreated instructions: {readme}")


if __name__ == "__main__":
    print("Isaac Sim Dataset Preparation Tool")
    print("="*80)
    
    # Step 1: Show the mismatch problem
    check_dataset_mismatch()
    
    print("\n\nDo you want to prepare Isaac Sim images for training? (y/n): ", end="")
    response = input().strip().lower()
    
    if response == 'y':
        # Step 2: Create structure
        prepare_dataset_structure()
        
        # Step 3: Copy images
        copied = copy_isaac_sim_images()
        
        if copied > 0:
            # Step 4: Create instructions
            create_labeling_instructions()
            
            print("\n" + "="*80)
            print("DATASET PREPARED!")
            print("="*80)
            print(f"Images: {OUTPUT_IMAGES}")
            print(f"Labels: {OUTPUT_LABELS} (empty - need labeling)")
            print(f"\nNext step: Label the images using Roboflow.com")
            print("See README.md for detailed instructions")
        else:
            print("\nNo images to prepare. Run Isaac Sim script first!")
    else:
        print("\nCancelled.")

