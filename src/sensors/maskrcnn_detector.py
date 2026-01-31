"""
MaskRCNN Object Detector for Isaac Sim

Provides object detection and segmentation using Mask R-CNN model.
Replaces OwlV2 detector with MaskRCNN for better segmentation and detection.
"""

import numpy as np
import carb
from typing import List, Dict, Tuple, Optional

try:
    import torch
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.transforms import functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    carb.log_warn("[MASKRCNN] PyTorch/torchvision not available. MaskRCNN detector will not work.")


class MaskRCNNDetector:
    """
    MaskRCNN-based object detector for pick-and-place operations
    
    Uses pre-trained MaskRCNN model for object detection and instance segmentation.
    Supports 3D position estimation when depth data is provided.
    """
    
    def __init__(self, 
                 confidence_threshold=0.5,
                 device="auto",
                 camera_params=None,
                 target_classes=None):
        """
        Initialize MaskRCNN detector
        
        Args:
            confidence_threshold: Minimum confidence score for detections (0.0-1.0)
            device: Device to run model on ("cpu", "cuda", or "auto")
            camera_params: Camera intrinsic parameters dict with fx, fy, cx, cy
            target_classes: List of target class names to detect (e.g., ["cube", "cylinder"])
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and torchvision are required for MaskRCNN detector")
        
        self.confidence_threshold = confidence_threshold
        self.camera_params = camera_params or {}
        self.target_classes = target_classes or ["cube", "cylinder", "cuboid"]
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load pre-trained MaskRCNN model
        print(f"[MASKRCNN] Loading MaskRCNN model on {self.device}...")
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # COCO class names (MaskRCNN is trained on COCO dataset)
        self.coco_classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        print(f"[MASKRCNN] Model loaded successfully")
        print(f"[MASKRCNN] Confidence threshold: {self.confidence_threshold}")
        print(f"[MASKRCNN] Target classes: {self.target_classes}")
        print(f"[MASKRCNN] Device: {self.device}")
    
    def detect(self, rgb_image: np.ndarray, verbose=False) -> List[Dict]:
        """
        Detect objects in RGB image
        
        Args:
            rgb_image: RGB image as numpy array (H, W, 3) or (H, W, 4)
            verbose: Whether to print detection details
            
        Returns:
            List of detection dictionaries with keys:
                - label: Object class name
                - confidence: Detection confidence score
                - bbox: Bounding box [x1, y1, x2, y2]
                - mask: Segmentation mask (H, W) boolean array
                - center: Center point [x, y] in image coordinates
        """
        if rgb_image is None:
            return []
        
        # Convert to RGB if RGBA
        if rgb_image.shape[2] == 4:
            rgb_image = rgb_image[:, :, :3]
        
        # Convert to tensor and normalize
        image_tensor = F.to_tensor(rgb_image).to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model([image_tensor])[0]
        
        # Process detections
        detections = []
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        masks = predictions['masks'].cpu().numpy()
        
        for i, (box, label, score, mask) in enumerate(zip(boxes, labels, scores, masks)):
            if score < self.confidence_threshold:
                continue
            
            class_name = self.coco_classes[label] if label < len(self.coco_classes) else "unknown"
            
            # Map COCO classes to our target classes (simple heuristic)
            # This is a placeholder - you may need custom training for specific objects
            mapped_class = self._map_coco_to_target(class_name)
            if mapped_class is None:
                continue
            
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            detection = {
                'label': mapped_class,
                'confidence': float(score),
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'mask': mask[0] > 0.5,  # Binary mask
                'center': [int(center_x), int(center_y)]
            }
            
            detections.append(detection)
        
        if verbose:
            print(f"[MASKRCNN] Detected {len(detections)} objects")
            for det in detections:
                print(f"  - {det['label']}: {det['confidence']:.2%} at {det['center']}")

        return detections

    def detect_3d(self, rgb_image: np.ndarray, depth_image: np.ndarray, verbose=False) -> List[Dict]:
        """
        Detect objects with 3D position estimation

        Args:
            rgb_image: RGB image as numpy array (H, W, 3) or (H, W, 4)
            depth_image: Depth image as numpy array (H, W) in meters
            verbose: Whether to print detection details

        Returns:
            List of detection dictionaries with additional 3D information:
                - pos_3d: 3D position in camera frame [x, y, z]
                - size_3d: Estimated 3D size [width, height, depth]
        """
        # Get 2D detections first
        detections = self.detect(rgb_image, verbose=False)

        if depth_image is None or len(detections) == 0:
            return detections

        # Add 3D information to each detection
        for detection in detections:
            center_x, center_y = detection['center']

            # Get depth at center point
            if 0 <= center_y < depth_image.shape[0] and 0 <= center_x < depth_image.shape[1]:
                depth = depth_image[center_y, center_x]

                # Unproject to 3D using camera intrinsics
                pos_3d = self._unproject_point([center_x, center_y], depth)
                detection['pos_3d'] = pos_3d

                # Estimate 3D size from bounding box and depth
                x1, y1, x2, y2 = detection['bbox']
                width_px = x2 - x1
                height_px = y2 - y1

                # Simple size estimation (can be improved with mask analysis)
                fx = self.camera_params.get('fx', 320.0)
                fy = self.camera_params.get('fy', 320.0)

                width_3d = (width_px * depth) / fx
                height_3d = (height_px * depth) / fy
                depth_3d = min(width_3d, height_3d)  # Assume cube-like objects

                detection['size_3d'] = np.array([width_3d, height_3d, depth_3d])

        if verbose:
            print(f"[MASKRCNN 3D] Detected {len(detections)} objects with 3D positions")
            for i, det in enumerate(detections, 1):
                if 'pos_3d' in det:
                    print(f"  [{i}] {det['label']}: pos={det['pos_3d']}, size={det['size_3d']}")

        return detections

    def _unproject_point(self, image_point: Tuple[float, float], depth: float) -> np.ndarray:
        """
        Unproject 2D image point to 3D camera frame

        Args:
            image_point: (x, y) in image coordinates
            depth: Depth value in meters (radial distance from camera)

        Returns:
            3D position in camera frame [x, y, z]
        """
        fx = self.camera_params.get('fx', 320.0)
        fy = self.camera_params.get('fy', 320.0)
        cx = self.camera_params.get('cx', 320.0)
        cy = self.camera_params.get('cy', 320.0)

        u, v = image_point

        # Direction ratios
        rx = (u - cx) / fx
        ry = (v - cy) / fy

        # Scale factor for radial distance
        scale = np.sqrt(1 + rx**2 + ry**2)

        # 3D position in camera frame
        x = (rx * depth) / scale
        y = (ry * depth) / scale
        z = depth / scale

        return np.array([x, y, z])

    def _map_coco_to_target(self, coco_class: str) -> Optional[str]:
        """
        Map COCO class names to target class names

        This is a simple heuristic mapping. For production use, you should:
        1. Train a custom MaskRCNN model on your specific objects
        2. Or use a more sophisticated mapping based on object properties

        Args:
            coco_class: COCO dataset class name

        Returns:
            Mapped target class name or None if not a target
        """
        # Simple mapping based on shape similarity
        # Note: This is a placeholder - MaskRCNN trained on COCO won't detect
        # custom objects like "cube" or "cylinder" directly

        # For demonstration, map some COCO classes to our targets
        mapping = {
            'bottle': 'cylinder',
            'cup': 'cylinder',
            'vase': 'cylinder',
            'bowl': 'cube',
            'book': 'cuboid',
            'laptop': 'cuboid',
            'cell phone': 'cuboid',
            'remote': 'cuboid',
            'mouse': 'cube',
        }

        return mapping.get(coco_class, None)


