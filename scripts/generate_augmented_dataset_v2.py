"""
YOLOv8 Augmented Dataset Generator v2.0
Generates 500 augmented images per class (NO mixing between classes)
"""

import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import albumentations as A
import yaml

# ============================================================================
# CONFIGURATION
# ============================================================================

SOURCE_DIR = Path("C:/isaacsim/cobotproject/datasets/new_dataset")
OUTPUT_DIR = Path("C:/isaacsim/cobotproject/datasets/augmented_yolo_v2")
IMAGES_PER_CLASS = 500  # Generate 500 images per class
IMG_SIZE = 640
TRAIN_RATIO = 0.8  # 80% train, 20% test

CLASS_MAPPING = {
    'cube': 0,
    'cylinder': 1,
    'container': 2,
    'cuboid': 3
}

# Background colors (RGB)
BACKGROUNDS = [
    (240, 240, 240),  # Light gray
    (255, 255, 255),  # White
    (220, 220, 220),  # Gray
    (200, 200, 200),  # Darker gray
    (210, 210, 210),  # Medium gray
    (230, 230, 230),  # Light gray 2
]


def create_augmentation_pipeline():
    """Create albumentations augmentation pipeline"""
    return A.Compose([
        # Geometric transforms
        A.Rotate(limit=180, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        A.RandomScale(scale_limit=0.4, p=0.8),
        A.Affine(shift=0.15, scale=(0.7, 1.3), rotate=45, p=0.6),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # Color transforms
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
        A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=40, val_shift_limit=25, p=0.6),
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.6),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),

        # Noise and blur
        A.GaussNoise(var_limit=(10.0, 60.0), mean=0, p=0.4),
        A.GaussianBlur(blur_limit=(3, 9), p=0.4),
        A.MotionBlur(blur_limit=9, p=0.3),
        A.MedianBlur(blur_limit=5, p=0.2),

        # Quality degradation
        A.ImageCompression(quality_range=(70, 100), p=0.4),
        A.Downscale(scale_min=0.7, scale_max=0.9, p=0.3),

    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))


def load_image_with_alpha(image_path: Path) -> tuple:
    """Load image and create alpha mask. Returns (image_rgb, alpha_mask)"""
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
    """Get bounding box of object from alpha mask. Returns (x, y, w, h) in pixels"""
    coords = np.column_stack(np.where(alpha_mask > 128))

    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return (x_min, y_min, x_max - x_min, y_max - y_min)


def create_single_object_image(img_rgb: np.ndarray, alpha: np.ndarray, class_id: int, img_size: int = 640) -> tuple:
    """
    Create image with single object on random background.
    Returns (image, bboxes, class_labels) in YOLO format
    """
    # Create background
    bg_color = random.choice(BACKGROUNDS)
    canvas = np.full((img_size, img_size, 3), bg_color, dtype=np.uint8)

    # Get object bbox
    bbox_pixels = get_object_bbox(alpha)
    if bbox_pixels is None:
        return None, None, None

    x, y, w, h = bbox_pixels
    obj_img = img_rgb[y:y+h, x:x+w]
    obj_alpha = alpha[y:y+h, x:x+w]

    # Random scale (50% to 90% of canvas)
    scale_factor = random.uniform(0.5, 0.9)
    max_dim = max(w, h)
    target_size = int(img_size * scale_factor)
    scale = target_size / max_dim

    new_w = int(w * scale)
    new_h = int(h * scale)

    obj_img_resized = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    obj_alpha_resized = cv2.resize(obj_alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Random position
    max_x = img_size - new_w
    max_y = img_size - new_h
    pos_x = random.randint(0, max(0, max_x))
    pos_y = random.randint(0, max(0, max_y))

    # Composite object onto canvas
    for c in range(3):
        canvas[pos_y:pos_y+new_h, pos_x:pos_x+new_w, c] = \
            canvas[pos_y:pos_y+new_h, pos_x:pos_x+new_w, c] * (1 - obj_alpha_resized / 255.0) + \
            obj_img_resized[:, :, c] * (obj_alpha_resized / 255.0)

    # Calculate YOLO format bbox (normalized)
    x_center = (pos_x + new_w / 2) / img_size
    y_center = (pos_y + new_h / 2) / img_size
    width = new_w / img_size
    height = new_h / img_size

    bboxes = [[x_center, y_center, width, height]]
    class_labels = [class_id]

    return canvas.astype(np.uint8), bboxes, class_labels


def load_class_images(class_name: str) -> list:
    """Load all images for a specific class"""
    class_dir = SOURCE_DIR / class_name
    if not class_dir.exists():
        print(f"Warning: {class_dir} does not exist!")
        return []

    images = []
    for img_path in class_dir.glob("*.png"):
        img_rgb, alpha = load_image_with_alpha(img_path)
        if img_rgb is not None:
            images.append((img_rgb, alpha, img_path.name))

    print(f"  {class_name}: {len(images)} images")
    return images


def generate_augmented_images_for_class(class_name: str, class_id: int, num_images: int):
    """Generate augmented images for a single class"""
    print(f"\nGenerating {num_images} images for class '{class_name}'...")

    # Load base images
    base_images = load_class_images(class_name)
    if len(base_images) == 0:
        print(f"  No images found for {class_name}, skipping!")
        return []

    augmentation_pipeline = create_augmentation_pipeline()
    generated_images = []

    for i in tqdm(range(num_images), desc=f"  {class_name}"):
        # Randomly select a base image
        img_rgb, alpha, img_name = random.choice(base_images)

        # Create single object image
        canvas, bboxes, class_labels = create_single_object_image(img_rgb, alpha, class_id, IMG_SIZE)

        if canvas is None:
            continue

        # Apply augmentations
        try:
            augmented = augmentation_pipeline(image=canvas, bboxes=bboxes, class_labels=class_labels)
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_labels = augmented['class_labels']

            if len(aug_bboxes) > 0:
                generated_images.append({
                    'image': aug_image,
                    'bboxes': aug_bboxes,
                    'class_labels': aug_labels,
                    'class_name': class_name
                })
        except Exception as e:
            # Skip if augmentation fails
            continue

    print(f"  Successfully generated {len(generated_images)} images for {class_name}")
    return generated_images


def save_dataset(all_images: list, train_ratio: float = 0.8):
    """Save images and labels, split into train/test"""
    print(f"\nSaving dataset to {OUTPUT_DIR}...")

    # Create directories
    train_images_dir = OUTPUT_DIR / "train" / "images"
    train_labels_dir = OUTPUT_DIR / "train" / "labels"
    test_images_dir = OUTPUT_DIR / "test" / "images"
    test_labels_dir = OUTPUT_DIR / "test" / "labels"

    for dir_path in [train_images_dir, train_labels_dir, test_images_dir, test_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Shuffle all images
    random.shuffle(all_images)

    # Split train/test
    split_idx = int(len(all_images) * train_ratio)
    train_data = all_images[:split_idx]
    test_data = all_images[split_idx:]

    # Save train set
    for idx, data in enumerate(tqdm(train_data, desc="  Saving train")):
        img_path = train_images_dir / f"aug_{idx:05d}.jpg"
        label_path = train_labels_dir / f"aug_{idx:05d}.txt"

        # Save image
        img_bgr = cv2.cvtColor(data['image'], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(img_path), img_bgr)

        # Save label
        with open(label_path, 'w') as f:
            for bbox, class_id in zip(data['bboxes'], data['class_labels']):
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

    # Save test set
    for idx, data in enumerate(tqdm(test_data, desc="  Saving test")):
        img_path = test_images_dir / f"aug_{idx:05d}.jpg"
        label_path = test_labels_dir / f"aug_{idx:05d}.txt"

        # Save image
        img_bgr = cv2.cvtColor(data['image'], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(img_path), img_bgr)

        # Save label
        with open(label_path, 'w') as f:
            for bbox, class_id in zip(data['bboxes'], data['class_labels']):
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

    print(f"  Train: {len(train_data)} images")
    print(f"  Test: {len(test_data)} images")

    return len(train_data), len(test_data)



def create_dataset_yaml():
    """Create dataset.yaml for YOLOv8"""
    yaml_content = {
        'path': str(OUTPUT_DIR.absolute()),
        'train': 'train/images',
        'val': 'test/images',
        'names': {
            0: 'cube',
            1: 'cylinder',
            2: 'container',
            3: 'cuboid'
        }
    }

    yaml_path = OUTPUT_DIR / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"\nDataset YAML created: {yaml_path}")


def main():
    """Main execution"""
    print("=" * 80)
    print("YOLOv8 Augmented Dataset Generator v2.0")
    print("=" * 80)
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Images per class: {IMAGES_PER_CLASS}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Train/Test split: {int(TRAIN_RATIO*100)}% / {int((1-TRAIN_RATIO)*100)}%")
    print()

    # Load source images
    print("Loading source images...")

    # Generate images for each class
    all_generated_images = []

    for class_name, class_id in CLASS_MAPPING.items():
        class_images = generate_augmented_images_for_class(class_name, class_id, IMAGES_PER_CLASS)
        all_generated_images.extend(class_images)

    print(f"\nTotal images generated: {len(all_generated_images)}")

    # Save dataset
    num_train, num_test = save_dataset(all_generated_images, TRAIN_RATIO)

    # Create dataset.yaml
    create_dataset_yaml()

    # Print summary
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 80)
    print(f"Dataset location: {OUTPUT_DIR}")
    print(f"Total images: {len(all_generated_images)}")
    print(f"Train: {num_train} images ({int(TRAIN_RATIO*100)}%)")
    print(f"Test: {num_test} images ({int((1-TRAIN_RATIO)*100)}%)")
    print()
    print("Per-class breakdown:")
    for class_name in CLASS_MAPPING.keys():
        class_count = sum(1 for img in all_generated_images if img['class_name'] == class_name)
        print(f"  {class_name}: {class_count} images")
    print()
    print("Next steps:")
    print("1. Train YOLOv8 model:")
    print("   python cobotproject/scripts/train_on_augmented_dataset_v2.py")
    print("2. Use trained model in Isaac Sim script")
    print("=" * 80)


if __name__ == "__main__":
    main()


