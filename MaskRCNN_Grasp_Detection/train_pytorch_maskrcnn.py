"""
Train MaskRCNN in PyTorch for cube/cylinder detection
Uses existing annotated dataset from Mask_RCNN/datasets/project
"""

import os
import sys
import json
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import skimage.draw

# Paths
DATASET_DIR = "Mask_RCNN/datasets/project"
OUTPUT_WEIGHTS = "mask_rcnn_cube_cylinder_pytorch.pth"

# Training config
NUM_CLASSES = 4  # background + cube + cylinder + cuboid
BATCH_SIZE = 2
NUM_EPOCHS = 30
LEARNING_RATE = 0.005
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class CubeDataset(Dataset):
    """PyTorch dataset for cube/cylinder detection"""
    
    def __init__(self, root_dir, subset='train'):
        self.root_dir = os.path.join(root_dir, subset)
        self.subset = subset
        
        # Load annotations
        annotations_path = os.path.join(self.root_dir, 'via_region_data.json')
        with open(annotations_path) as f:
            annotations = json.load(f)
        
        self.annotations = list(annotations.values())
        self.annotations = [a for a in self.annotations if a['regions']]
        
        # Class mapping
        self.class_map = {
            'cube': 1,
            'cylinder': 2,
            'cuboid': 3
        }
        
        print(f"Loaded {len(self.annotations)} images from {subset} set")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Load image
        img_path = os.path.join(self.root_dir, ann['filename'])
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)
        
        # Get regions
        regions = ann['regions'].values()
        
        boxes = []
        labels = []
        masks = []
        
        for region in regions:
            # Get class
            class_name = region['region_attributes'].get('name', '')
            if class_name not in self.class_map:
                continue  # Skip gripper, grasp, etc.
            
            label = self.class_map[class_name]
            
            # Get polygon
            shape = region['shape_attributes']
            all_points_x = shape['all_points_x']
            all_points_y = shape['all_points_y']
            
            # Create mask
            mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
            rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
            mask[rr, cc] = 1
            
            # Get bounding box from mask
            pos = np.where(mask)
            if len(pos[0]) == 0:
                continue
            
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
            masks.append(mask)
        
        if len(boxes) == 0:
            # No valid objects, create dummy
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, img_array.shape[0], img_array.shape[1]), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        
        # Convert image to tensor
        img_tensor = torch.as_tensor(img_array, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx])
        }
        
        return img_tensor, target


def get_model(num_classes):
    """Build MaskRCNN model"""
    model = maskrcnn_resnet50_fpn(pretrained=True)
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        if (i + 1) % 10 == 0:
            print(f"  Batch [{i+1}/{len(data_loader)}], Loss: {losses.item():.4f}")
    
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch+1}] Average Loss: {avg_loss:.4f}")
    return avg_loss


def main():
    """Main training function"""
    print("=" * 60)
    print("PyTorch MaskRCNN Training - Cube/Cylinder Detection")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Num classes: {NUM_CLASSES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print("=" * 60)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = CubeDataset(DATASET_DIR, 'train')
    val_dataset = CubeDataset(DATASET_DIR, 'val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Build model
    print("\nBuilding model...")
    model = get_model(NUM_CLASSES)
    model.to(DEVICE)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    print("\nStarting training...")
    print("=" * 60)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 60)

        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)

        # Update learning rate
        lr_scheduler.step()

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    torch.save(model.state_dict(), OUTPUT_WEIGHTS)
    print(f"\nFinal weights saved to: {OUTPUT_WEIGHTS}")
    print(f"\nCopy this file to: cobotproject/models/{OUTPUT_WEIGHTS}")
    print("\nTo use in Isaac Sim:")
    print("  1. Copy weights to cobotproject/models/")
    print("  2. Update detector initialization to use CustomMaskRCNNDetector")
    print("  3. Point to the weights file")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

