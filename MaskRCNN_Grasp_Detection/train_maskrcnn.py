"""
Train MaskRCNN on custom cube/cylinder dataset
Run this in native Python 3.7 environment (NOT Isaac Sim Python)
"""

import os
import sys

# Add Mask_RCNN to path
MASKRCNN_ROOT = os.path.join(os.path.dirname(__file__), "Mask_RCNN")
sys.path.append(MASKRCNN_ROOT)

from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils

# Path to COCO pre-trained weights
COCO_WEIGHTS_PATH = os.path.join(MASKRCNN_ROOT, "mask_rcnn_coco.h5")

# Directory to save logs and trained model
DEFAULT_LOGS_DIR = os.path.join(MASKRCNN_ROOT, "logs")

# Dataset directory
DATASET_DIR = os.path.join(MASKRCNN_ROOT, "datasets", "project")


class CubeConfig(Config):
    """Configuration for training on cube/cylinder dataset"""
    NAME = "cube_cylinder"
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + cube + cylinder + cuboid
    
    # Training steps
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 20
    
    # Detection confidence threshold
    DETECTION_MIN_CONFIDENCE = 0.7
    
    # Image dimensions
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    
    # Learning rate
    LEARNING_RATE = 0.001


class CubeDataset(utils.Dataset):
    """Dataset class for cube/cylinder detection"""
    
    def load_cube_dataset(self, dataset_dir, subset):
        """Load cube/cylinder dataset
        dataset_dir: Root directory of dataset
        subset: 'train' or 'val'
        """
        # Add classes
        self.add_class("cube", 1, "cube")
        self.add_class("cube", 2, "cylinder")
        self.add_class("cube", 3, "cuboid")
        
        # Load existing project dataset
        import json
        assert subset in ["train", "val"]
        dataset_path = os.path.join(dataset_dir, subset)
        
        # Load annotations
        annotations_path = os.path.join(dataset_path, "via_region_data.json")
        if not os.path.exists(annotations_path):
            print(f"Warning: No annotations found at {annotations_path}")
            return
            
        annotations = json.load(open(annotations_path))
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]
        
        # Add images
        for a in annotations:
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            faces = [t['region_attributes'] for t in a['regions'].values()]
            
            # Map class names to IDs
            class_ids = []
            for face in faces:
                name = face.get('name', '')
                if name == "cube":
                    class_ids.append(1)
                elif name == "cylinder":
                    class_ids.append(2)
                elif name == "cuboid":
                    class_ids.append(3)
                else:
                    # Skip gripper and grasp classes
                    continue
            
            if not class_ids:
                continue
                
            # Load image to get dimensions
            import skimage.io
            image_path = os.path.join(dataset_path, a['filename'])
            if not os.path.exists(image_path):
                continue
                
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            
            self.add_image(
                "cube",
                image_id=a['filename'],
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
                class_ids=class_ids
            )
    
    def load_mask(self, image_id):
        """Generate instance masks for image"""
        image_info = self.image_info[image_id]
        if image_info["source"] != "cube":
            return super(self.__class__, self).load_mask(image_id)
        
        # Convert polygons to masks
        import numpy as np
        import skimage.draw
        
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        
        class_ids = np.array(info["class_ids"], dtype=np.int32)
        return mask.astype(np.bool), class_ids
    
    def image_reference(self, image_id):
        """Return image path"""
        info = self.image_info[image_id]
        if info["source"] == "cube":
            return info["path"]
        else:
            return super(self.__class__, self).image_reference(image_id)


def train_model():
    """Train MaskRCNN model"""
    import numpy as np

    # Configuration
    config = CubeConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)

    # Download COCO weights if not exists
    if not os.path.exists(COCO_WEIGHTS_PATH):
        print("Downloading COCO weights...")
        utils.download_trained_weights(COCO_WEIGHTS_PATH)

    # Load COCO weights (exclude final layers)
    print(f"Loading COCO weights from {COCO_WEIGHTS_PATH}")
    model.load_weights(
        COCO_WEIGHTS_PATH,
        by_name=True,
        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
    )

    # Load training dataset
    print("Loading training dataset...")
    dataset_train = CubeDataset()
    dataset_train.load_cube_dataset(DATASET_DIR, "train")
    dataset_train.prepare()
    print(f"Training images: {len(dataset_train.image_ids)}")

    # Load validation dataset
    print("Loading validation dataset...")
    dataset_val = CubeDataset()
    dataset_val.load_cube_dataset(DATASET_DIR, "val")
    dataset_val.prepare()
    print(f"Validation images: {len(dataset_val.image_ids)}")

    # Train
    print("\nTraining network heads (30 epochs)...")
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=30,
        layers='heads'
    )

    print("\nTraining all layers (10 epochs)...")
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE / 10,
        epochs=40,
        layers='all'
    )

    # Save final weights
    model_path = os.path.join(MASKRCNN_ROOT, "mask_rcnn_cube_cylinder.h5")
    model.keras_model.save_weights(model_path)
    print(f"\nTraining complete! Weights saved to: {model_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("MaskRCNN Training - Cube/Cylinder Detection")
    print("=" * 60)
    print(f"Dataset: {DATASET_DIR}")
    print(f"Logs: {DEFAULT_LOGS_DIR}")
    print("=" * 60)

    train_model()

