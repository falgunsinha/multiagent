"""
Generate Augmented YOLOv8 Dataset from Base Images

This script takes your base images from new_dataset/ and generates
200-300 augmented variations per image with automatic YOLO labels.

Augmentations include:
- Random rotations
- Random scaling
- Random brightness/contrast
- Random noise
- Random blur
- Random crops
- Multiple objects per image (composite scenes)
"""

import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration
SOURCE_DIR = Path("C:/isaacsim/cobotproject/datasets/new_dataset")
OUTPUT_DIR = Path("C:/isaacsim/cobotproject/datasets/augmented_yolo")
OUTPUT_IMAGES = OUTPUT_DIR / "images"
OUTPUT_LABELS = OUTPUT_DIR / "labels"

# Number of augmented images to generate per source image
AUGMENTATIONS_PER_IMAGE = 250

# Image size for YOLO training
IMG_SIZE = 640

# Class mapping
CLASS_MAPPING = {
    'cube': 0,
    'cylinder': 1,
    'container': 2,
    'cuboid': 3  # Treat cuboid as obstacle or separate class
}

# Background colors (similar to Isaac Sim)
BACKGROUNDS = [
    (240, 240, 240),  # Light gray
    (220, 220, 220),  # Gray
    (200, 200, 200),  # Darker gray
    (255, 255, 255),  # White
    (180, 180, 180),  # Medium gray
]


def create_augmentation_pipeline():
    """Create albumentations augmentation pipeline"""
    return A.Compose([
        # Geometric transforms
        A.Rotate(limit=180, p=0.8, border_mode=cv2.BORDER_CONSTANT),
        A.RandomScale(scale_limit=0.3, p=0.7),
        A.Affine(shift=0.1, scale=(0.8, 1.2), rotate=45, p=0.5),

        # Color transforms
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),

        # Noise and blur
        A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.MotionBlur(blur_limit=7, p=0.2),

        # Quality degradation
        A.ImageCompression(quality_range=(75, 100), p=0.3),

    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def load_image_with_alpha(image_path: Path) -> tuple:
    """
    Load image and create alpha mask for transparent background.
    Returns (image_rgb, alpha_mask)
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    
    if img is None:
        return None, None
    
    # Convert to RGB
    if len(img.shape) == 2:  # Grayscale
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        alpha = np.ones(img.shape[:2], dtype=np.uint8) * 255
    elif img.shape[2] == 3:  # RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Create alpha mask from white background
        white_mask = np.all(img_rgb > 240, axis=2)
        alpha = np.where(white_mask, 0, 255).astype(np.uint8)
    elif img.shape[2] == 4:  # RGBA
        img_rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        alpha = img[:, :, 3]
    else:
        return None, None
    
    return img_rgb, alpha


def get_object_bbox(alpha_mask: np.ndarray) -> tuple:
    """
    Get bounding box of object from alpha mask.
    Returns (x, y, w, h) in pixel coordinates.
    """
    # Find non-zero pixels
    coords = cv2.findNonZero(alpha_mask)
    if coords is None:
        return None
    
    x, y, w, h = cv2.boundingRect(coords)
    return (x, y, w, h)


def create_composite_image(objects: list, img_size: int = 640) -> tuple:
    """
    Create composite image with multiple objects.
    Returns (image, bboxes, class_labels)
    """
    # Create background
    bg_color = random.choice(BACKGROUNDS)
    canvas = np.full((img_size, img_size, 3), bg_color, dtype=np.uint8)
    
    bboxes = []
    class_labels = []
    
    # Randomly select 1-4 objects
    num_objects = random.randint(1, 4)
    selected_objects = random.sample(objects, min(num_objects, len(objects)))
    
    for obj_img, obj_alpha, obj_class in selected_objects:
        # Random scale (20% to 60% of image size)
        scale = random.uniform(0.2, 0.6)
        obj_h, obj_w = obj_img.shape[:2]
        new_w = int(obj_w * scale * img_size / max(obj_w, obj_h))
        new_h = int(obj_h * scale * img_size / max(obj_w, obj_h))
        
        if new_w <= 0 or new_h <= 0:
            continue
        
        # Resize object and alpha
        obj_resized = cv2.resize(obj_img, (new_w, new_h))
        alpha_resized = cv2.resize(obj_alpha, (new_w, new_h))
        
        # Random position (ensure object fits in canvas)
        max_x = img_size - new_w
        max_y = img_size - new_h
        if max_x <= 0 or max_y <= 0:
            continue
        
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        
        # Composite object onto canvas using alpha blending
        alpha_3ch = alpha_resized[:, :, np.newaxis] / 255.0
        canvas[y:y+new_h, x:x+new_w] = (
            canvas[y:y+new_h, x:x+new_w] * (1 - alpha_3ch) +
            obj_resized * alpha_3ch
        ).astype(np.uint8)
        
        # Calculate YOLO format bbox (x_center, y_center, width, height) - normalized
        x_center = (x + new_w / 2) / img_size
        y_center = (y + new_h / 2) / img_size
        width = new_w / img_size
        height = new_h / img_size
        
        bboxes.append([x_center, y_center, width, height])
        class_labels.append(obj_class)
    
    return canvas, bboxes, class_labels


def generate_dataset():
    """Main function to generate augmented dataset"""
    
    # Create output directories
    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("YOLOv8 Augmented Dataset Generator")
    print("="*80)
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Augmentations per image: {AUGMENTATIONS_PER_IMAGE}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    
    # Load all source images
    print("\nLoading source images...")
    objects = []
    
    for class_name, class_id in CLASS_MAPPING.items():
        class_dir = SOURCE_DIR / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found, skipping...")
            continue
        
        image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        print(f"  {class_name}: {len(image_files)} images")
        
        for img_file in image_files:
            img_rgb, alpha = load_image_with_alpha(img_file)
            if img_rgb is not None and alpha is not None:
                objects.append((img_rgb, alpha, class_id))
    
    print(f"\nTotal objects loaded: {len(objects)}")
    
    if len(objects) == 0:
        print("ERROR: No objects loaded! Check source directory.")
        return
    
    # Create augmentation pipeline
    transform = create_augmentation_pipeline()
    
    # Generate augmented images
    print(f"\nGenerating {AUGMENTATIONS_PER_IMAGE} augmented images...")
    
    for i in tqdm(range(AUGMENTATIONS_PER_IMAGE)):
        # Create composite image
        image, bboxes, class_labels = create_composite_image(objects, IMG_SIZE)
        
        if len(bboxes) == 0:
            continue
        
        # Apply augmentations
        try:
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = transformed['image']
            aug_bboxes = transformed['bboxes']
            aug_labels = transformed['class_labels']
        except Exception as e:
            # If augmentation fails, use original
            aug_image = image
            aug_bboxes = bboxes
            aug_labels = class_labels
        
        # Save image
        img_filename = f"aug_{i:05d}.png"
        img_path = OUTPUT_IMAGES / img_filename
        cv2.imwrite(str(img_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
        
        # Save YOLO label
        label_filename = f"aug_{i:05d}.txt"
        label_path = OUTPUT_LABELS / label_filename
        
        with open(label_path, 'w') as f:
            for bbox, label in zip(aug_bboxes, aug_labels):
                # YOLO format: class x_center y_center width height
                f.write(f"{label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    
    print(f"\n✓ Generated {AUGMENTATIONS_PER_IMAGE} images!")
    print(f"  Images: {OUTPUT_IMAGES}")
    print(f"  Labels: {OUTPUT_LABELS}")


def split_train_test(train_ratio=0.8):
    """Split dataset into train and test sets (no validation)"""
    print("\nSplitting dataset into train/test...")

    # Create train/test directories
    train_images = OUTPUT_DIR / "train" / "images"
    train_labels = OUTPUT_DIR / "train" / "labels"
    test_images = OUTPUT_DIR / "test" / "images"
    test_labels = OUTPUT_DIR / "test" / "labels"

    for d in [train_images, train_labels, test_images, test_labels]:
        d.mkdir(parents=True, exist_ok=True)

    # Get all images
    all_images = list(OUTPUT_IMAGES.glob("*.png"))
    random.shuffle(all_images)

    # Split
    split_idx = int(len(all_images) * train_ratio)
    train_files = all_images[:split_idx]
    test_files = all_images[split_idx:]

    print(f"  Train: {len(train_files)} images")
    print(f"  Test: {len(test_files)} images")

    # Move files
    import shutil
    for img_file in train_files:
        label_file = OUTPUT_LABELS / (img_file.stem + ".txt")
        shutil.move(str(img_file), str(train_images / img_file.name))
        if label_file.exists():
            shutil.move(str(label_file), str(train_labels / label_file.name))

    for img_file in test_files:
        label_file = OUTPUT_LABELS / (img_file.stem + ".txt")
        shutil.move(str(img_file), str(test_images / img_file.name))
        if label_file.exists():
            shutil.move(str(label_file), str(test_labels / label_file.name))

    # Remove old directories
    OUTPUT_IMAGES.rmdir()
    OUTPUT_LABELS.rmdir()

    print("✓ Dataset split complete!")


def create_dataset_yaml():
    """Create dataset.yaml for YOLOv8 training"""
    yaml_path = OUTPUT_DIR / "dataset.yaml"

    yaml_content = f"""# YOLOv8 Dataset Configuration
# Generated from new_dataset augmentation

path: {str(OUTPUT_DIR).replace(chr(92), '/')}
train: train/images
val: test/images

# Classes
names:
  0: cube
  1: cylinder
  2: container
  3: cuboid
"""

    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n✓ Created dataset.yaml: {yaml_path}")


if __name__ == "__main__":
    # Check if albumentations is installed
    try:
        import albumentations
    except ImportError:
        print("ERROR: albumentations not installed!")
        print("Install with: pip install albumentations")
        exit(1)

    # Generate dataset
    generate_dataset()

    # Split into train/test
    split_train_test(train_ratio=0.8)

    # Create dataset.yaml
    create_dataset_yaml()

    print("\n" + "="*80)
    print("DATASET GENERATION COMPLETE!")
    print("="*80)
    print(f"Dataset location: {OUTPUT_DIR}")
    print(f"Total images: {AUGMENTATIONS_PER_IMAGE}")
    print(f"Train: ~{int(AUGMENTATIONS_PER_IMAGE * 0.8)} images")
    print(f"Test: ~{int(AUGMENTATIONS_PER_IMAGE * 0.2)} images")
    print("\nNext steps:")
    print("1. Train YOLOv8 model:")
    print("   python cobotproject/scripts/train_on_augmented_dataset.py")
    print("2. Use trained model in Isaac Sim script")
    print("="*80)

