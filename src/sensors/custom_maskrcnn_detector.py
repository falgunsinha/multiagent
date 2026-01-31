"""
Custom MaskRCNN Detector - Loads custom trained weights for cube/cylinder detection
"""

import numpy as np
import carb
from typing import List, Dict, Optional
import os

try:
    import torch
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.transforms import functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    carb.log_warn("[CUSTOM MASKRCNN] PyTorch/torchvision not available")


class CustomMaskRCNNDetector:
    """
    Custom MaskRCNN detector for cube/cylinder/cuboid detection
    Loads custom trained weights instead of COCO pre-trained weights
    """
    
    def __init__(self,
                 weights_path,
                 confidence_threshold=0.7,
                 device="auto",
                 camera_params=None,
                 num_classes=4,  # background + cube + cylinder + cuboid
                 save_detections=True,
                 output_dir="detection_results",
                 femto_camera=None):
        """
        Initialize custom MaskRCNN detector

        Args:
            weights_path: Path to custom trained .pth weights file
            confidence_threshold: Minimum confidence for detections
            device: Device to run on ("cpu", "cuda", or "auto")
            camera_params: Camera intrinsics dict
            num_classes: Number of classes including background
            save_detections: Whether to save detection images
            output_dir: Directory to save detection images
            femto_camera: FemtoCamera instance for coordinate transformations
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for MaskRCNN")

        self.confidence_threshold = confidence_threshold
        self.camera_params = camera_params or {}
        self.num_classes = num_classes
        self.save_detections = save_detections
        self.output_dir = output_dir
        self.detection_counter = 0
        self.femto_camera = femto_camera

        # Create output directory if saving detections
        if self.save_detections:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"[CUSTOM MASKRCNN] Saving detections to: {os.path.abspath(self.output_dir)}")
        
        # Class names
        self.class_names = ['__background__', 'cube', 'cylinder', 'cuboid']
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model
        print(f"[CUSTOM MASKRCNN] Loading custom model on {self.device}...")
        self.model = self._build_model(num_classes)
        
        # Load custom weights
        if os.path.exists(weights_path):
            print(f"[CUSTOM MASKRCNN] Loading weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("[CUSTOM MASKRCNN] Custom weights loaded successfully")
        else:
            print(f"[CUSTOM MASKRCNN] WARNING: Weights file not found: {weights_path}")
            print("[CUSTOM MASKRCNN] Using randomly initialized weights (will not detect properly)")
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[CUSTOM MASKRCNN] Detector ready")
        print(f"[CUSTOM MASKRCNN] Classes: {self.class_names}")
        print(f"[CUSTOM MASKRCNN] Confidence threshold: {self.confidence_threshold}")
    
    def _build_model(self, num_classes):
        """Build MaskRCNN model with custom number of classes"""
        # Load pre-trained model
        model = maskrcnn_resnet50_fpn(pretrained=False)
        
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
    
    def detect(self, rgb_image, verbose=False):
        """
        Detect objects in RGB image
        
        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            verbose: Print detection info
            
        Returns:
            List of detection dicts with keys: label, confidence, bbox, mask, center
        """
        # Convert to tensor
        image_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model([image_tensor])[0]
        
        # Process detections
        detections = []
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        masks = predictions['masks'].cpu().numpy()
        
        for i in range(len(boxes)):
            score = scores[i]
            if score < self.confidence_threshold:
                continue
            
            label_id = labels[i]
            if label_id >= len(self.class_names):
                continue
                
            label = self.class_names[label_id]
            if label == '__background__':
                continue
            
            bbox = boxes[i]
            mask = masks[i, 0] > 0.5
            
            # Calculate center
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            detection = {
                'label': label,
                'confidence': float(score),
                'bbox': bbox.tolist(),
                'mask': mask,
                'center': (int(center_x), int(center_y))
            }
            
            detections.append(detection)
            
            if verbose:
                print(f"[CUSTOM MASKRCNN] Detected {label} (conf: {score:.2f})")

        # Save detection visualization if enabled
        if self.save_detections and len(detections) > 0:
            self._save_detection_image(rgb_image, detections)

        return detections

    def detect_3d(self, rgb_image, depth_image, verbose=False):
        """
        Detect objects with 3D position estimation

        Args:
            rgb_image: RGB image (H, W, 3)
            depth_image: Depth image (H, W) in meters
            verbose: Print detection info

        Returns:
            List of detection dicts with 3D position
        """
        # Get 2D detections
        detections = self.detect(rgb_image, verbose=verbose)

        # Add 3D positions
        intrinsics_printed = False
        for det in detections:
            center_x, center_y = det['center']

            # Get depth at center
            if 0 <= center_y < depth_image.shape[0] and 0 <= center_x < depth_image.shape[1]:
                depth = depth_image[center_y, center_x]
                print(f"[DEPTH DEBUG] Object '{det['label']}' at pixel ({center_x}, {center_y}): depth={depth:.3f}m")

                # Debug: print camera intrinsics for first detection
                if not intrinsics_printed:
                    print(f"[INTRINSICS DEBUG] fx={self.camera_params.get('fx'):.2f}, fy={self.camera_params.get('fy'):.2f}, cx={self.camera_params.get('cx'):.2f}, cy={self.camera_params.get('cy'):.2f}")
                    intrinsics_printed = True

                # Check if Camera wrapper is available
                if not intrinsics_printed:
                    if self.femto_camera:
                        print(f"[TRANSFORM DEBUG] femto_camera exists: True")
                        print(f"[TRANSFORM DEBUG] camera_wrapper exists: {self.femto_camera.camera_wrapper is not None}")
                    else:
                        print(f"[TRANSFORM DEBUG] femto_camera exists: False")

                # Use Isaac Sim Camera's built-in transformation if available
                if self.femto_camera and self.femto_camera.camera_wrapper:
                    try:
                        # Use Isaac Sim Camera's get_world_points_from_image_coords
                        points_2d = np.array([[center_x, center_y]], dtype=np.float32)
                        depths = np.array([depth], dtype=np.float32)

                        if not intrinsics_printed:
                            print(f"[TRANSFORM] Calling Camera.get_world_points_from_image_coords()")
                            print(f"  Input: pixel=({center_x}, {center_y}), depth={depth:.3f}m")

                        world_points = self.femto_camera.camera_wrapper.get_world_points_from_image_coords(
                            points_2d, depths
                        )
                        pos_3d = world_points[0].tolist()

                        if not intrinsics_printed:
                            print(f"  Output: world_pos={pos_3d}")
                    except Exception as e:
                        print(f"[TRANSFORM ERROR] Camera transformation failed: {e}")
                        import traceback
                        traceback.print_exc()
                        # Fallback to manual unprojection
                        pos_3d = self._unproject_point((center_x, center_y), depth)
                        if not intrinsics_printed:
                            print(f"[TRANSFORM] Using manual unprojection (fallback after error)")
                else:
                    # Fallback to manual unprojection
                    pos_3d = self._unproject_point((center_x, center_y), depth)
                    if not intrinsics_printed:
                        print(f"[TRANSFORM] Using manual unprojection (no camera wrapper)")

                det['pos_3d'] = pos_3d

                # Estimate size from mask
                mask = det['mask']
                bbox = det['bbox']
                width_px = bbox[2] - bbox[0]
                height_px = bbox[3] - bbox[1]

                # Rough size estimation
                fx = self.camera_params.get('fx', 320.0)
                size_x = (width_px * depth) / fx
                size_y = (height_px * depth) / fx
                size_z = size_x  # Assume cube-like

                det['size_3d'] = [size_x, size_y, size_z]

                if verbose:
                    print(f"[CUSTOM MASKRCNN] 3D position: {pos_3d}")

        return detections

    def _unproject_point(self, image_point, depth):
        """
        Unproject 2D image point to 3D camera frame using USD camera convention.

        Isaac Sim USD Camera Convention (from official docs):
        - +X: right
        - +Y: up
        - -Z: forward (optical axis points in -Z direction)

        The depth from distance_to_camera is the Euclidean (radial) distance
        from the camera origin to the point.

        For radial distance r and pixel (u, v):
        - Direction vector in USD camera frame: d = [(u-cx)/fx, -(v-cy)/fy, -1]
          Note: -Z is forward, so dz = -1
          Note: +Y is up, but image Y increases downward, so dy = -(v-cy)/fy
        - Normalize: d_norm = d / ||d||
        - Point in camera frame: p = r * d_norm
        """
        fx = self.camera_params.get('fx', 320.0)
        fy = self.camera_params.get('fy', 320.0)
        cx = self.camera_params.get('cx', 320.0)
        cy = self.camera_params.get('cy', 320.0)

        u, v = image_point

        # Direction ratios in USD camera convention
        # +X right: (u - cx) / fx
        # +Y up: -(v - cy) / fy  (image Y increases downward, camera Y is up)
        # -Z forward: -1
        dx = (u - cx) / fx
        dy = -(v - cy) / fy  # Negate because image Y is down, camera Y is up
        dz = -1.0  # Forward is -Z in USD camera convention

        # Normalize the direction vector
        norm = np.sqrt(dx**2 + dy**2 + dz**2)

        # Scale by radial distance to get 3D point in USD camera frame
        x = (dx / norm) * depth
        y = (dy / norm) * depth
        z = (dz / norm) * depth

        # Debug output for first unprojection
        if not hasattr(self, '_unproject_debug_printed'):
            print(f"[UNPROJECT DEBUG] pixel=({u}, {v}), depth={depth:.3f}m")
            print(f"  USD camera direction=({dx:.3f}, {dy:.3f}, {dz:.3f}), norm={norm:.3f}")
            print(f"  USD camera_frame=({x:.3f}, {y:.3f}, {z:.3f})")
            self._unproject_debug_printed = True

        return [x, y, z]

    def _save_detection_image(self, rgb_image, detections):
        """Save detection visualization with bounding boxes and labels"""
        try:
            import cv2

            # Create copy for visualization
            vis_image = rgb_image.copy()

            # Draw each detection
            for det in detections:
                bbox = det['bbox']
                label = det['label']
                conf = det['confidence']

                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox)
                color = (0, 255, 0)  # Green
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

                # Draw label
                text = f"{label} {conf:.2f}"
                cv2.putText(vis_image, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw center point
                center = det['center']
                cv2.circle(vis_image, center, 5, (0, 0, 255), -1)

            # Save image
            self.detection_counter += 1
            filename = f"detection_{self.detection_counter:04d}.png"
            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

            print(f"[CUSTOM MASKRCNN] Saved detection image: {filepath}")

        except Exception as e:
            carb.log_warn(f"[CUSTOM MASKRCNN] Error saving detection image: {e}")

