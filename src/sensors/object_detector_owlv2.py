"""
OwlV2 Object Detector Module with 3D Detection
Open-vocabulary object detection using OwlV2 (Transformers) + Depth Camera.
Detects cubes, cylinders, cuboids, and any custom objects.

Classification:
- Cubes and cylinders are classified as 'target' (objects to pick)
- Cuboids are classified as 'obstacle' (objects to avoid)

Features:
- Non-Maximum Suppression (NMS) to remove duplicate detections
- 3D position estimation using depth camera
- Automatic size estimation from bounding boxes + depth
- Configurable confidence threshold
- GPU acceleration support

Based on Robot Environment framework with Isaac Sim compatibility.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import torch
from PIL import Image


class OwlV2ObjectDetector:
    """OwlV2-based open-vocabulary object detector"""

    def __init__(
        self,
        object_labels: List[str] = None,
        confidence_threshold: float = 0.20,
        device: str = "auto",
        camera_params: Dict = None
    ):
        """
        Initialize OwlV2 object detector with 3D detection support.

        Args:
            object_labels: List of object labels to detect (e.g., ["cube", "cylinder", "cuboid"])
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
            device: Device to use ("cuda", "cpu", or "auto")
            camera_params: Camera intrinsic parameters for 3D unprojection
                          {'fx': focal_x, 'fy': focal_y, 'cx': center_x, 'cy': center_y}
        """
        self.object_labels = object_labels or ["cube", "cylinder", "cuboid"]
        self.confidence_threshold = confidence_threshold

        # Class-specific thresholds (cuboid needs higher threshold to avoid false positives)
        self.class_thresholds = {
            'cube': confidence_threshold,
            'cylinder': confidence_threshold,
            'cuboid': max(0.30, confidence_threshold)  # Higher threshold for cuboids
        }

        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Camera parameters for 3D unprojection
        self.camera_params = camera_params

        # Initialize model (lazy loading)
        self.model = None
        self.processor = None
        self._model_loaded = False

        print(f"[OWLV2 DETECTOR] Initialized")
        print(f"[OWLV2 DETECTOR] Object labels: {self.object_labels}")
        print(f"[OWLV2 DETECTOR] Confidence threshold: {confidence_threshold}")
        print(f"[OWLV2 DETECTOR] Device: {self.device}")
        if camera_params:
            print(f"[OWLV2 DETECTOR] 3D detection enabled with camera params")

    def _load_model(self):
        """Load OwlV2 model (lazy loading)"""
        if self._model_loaded:
            return

        try:
            # Clear GPU cache before loading model to free memory
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("[OWLV2 DETECTOR] GPU cache cleared before model loading")

            print("[OWLV2 DETECTOR] Loading OwlV2 model...")
            from transformers import Owlv2Processor, Owlv2ForObjectDetection

            # Disable SSL verification for Hugging Face downloads (workaround for corporate SSL)
            import ssl
            import os
            import warnings

            # Disable SSL verification globally
            ssl._create_default_https_context = ssl._create_unverified_context

            # Set environment variables to disable SSL verification for requests library
            os.environ['CURL_CA_BUNDLE'] = ''
            os.environ['REQUESTS_CA_BUNDLE'] = ''
            os.environ['SSL_CERT_FILE'] = ''

            # Suppress SSL warnings
            warnings.filterwarnings('ignore', message='Unverified HTTPS request')
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

            # Monkey-patch requests to disable SSL verification
            import requests
            from functools import partialmethod
            requests.Session.request = partialmethod(requests.Session.request, verify=False)

            # Load model and processor
            # Use smaller base model instead of ensemble to reduce memory usage
            model_name = "google/owlv2-base-patch16"  # Smaller than ensemble version

            # Try to use local cache first, disable SSL verification if downloading
            try:
                # Try loading from local cache only (no network)
                self.processor = Owlv2Processor.from_pretrained(
                    model_name,
                    use_fast=True,
                    local_files_only=True
                )
                self.model = Owlv2ForObjectDetection.from_pretrained(
                    model_name,
                    local_files_only=True
                )
                print("[OWLV2 DETECTOR] Loaded from local cache")
            except Exception as cache_error:
                # If local cache fails, download with SSL verification disabled
                print(f"[OWLV2 DETECTOR] Local cache not found, downloading...")
                self.processor = Owlv2Processor.from_pretrained(model_name, use_fast=True)
                self.model = Owlv2ForObjectDetection.from_pretrained(model_name)

            self.model.to(self.device)
            self.model.eval()

            self._model_loaded = True
            print(f"[OWLV2 DETECTOR] Model loaded successfully on {self.device}")

            # Print GPU memory usage after loading
            if self.device == "cuda" and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                print(f"[OWLV2 DETECTOR] GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        except Exception as e:
            print(f"[OWLV2 DETECTOR ERROR] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def detect(self, rgb_image: np.ndarray, verbose: bool = False) -> List[Dict]:
        """
        Detect objects in RGB image using OwlV2.

        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            verbose: Print detection details

        Returns:
            List of detected objects with classification
        """
        try:
            # Clear GPU cache before detection to free memory
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Load model if not already loaded
            self._load_model()

            if verbose:
                print(f"[OWLV2 DETECTOR] Input image shape: {rgb_image.shape}")

            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(rgb_image)

            # Downscale image to reduce memory usage (OwlV2 will resize anyway)
            # This significantly reduces GPU memory requirements
            original_size = pil_image.size
            max_size = 640  # Reduce from default 768 to save memory
            if max(original_size) > max_size:
                scale = max_size / max(original_size)
                new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
                pil_image = pil_image.resize(new_size, Image.BILINEAR)
                if verbose:
                    print(f"[OWLV2 DETECTOR] Downscaled image: {original_size} -> {new_size}")

            # Prepare text queries with better descriptions to distinguish cube vs cuboid
            # Use more descriptive prompts to help OwlV2 distinguish between similar objects
            # NOTE: Do NOT include color in prompts - obstacles can be any color
            text_queries = []
            for label in self.object_labels:
                if label.lower() == "cube":
                    # Small cube: emphasize small size and equal dimensions
                    text_queries.append("a small cube block")
                elif label.lower() == "cuboid":
                    # Tall cuboid: emphasize vertical orientation and rectangular shape
                    # Do NOT mention color - obstacles can be any color
                    text_queries.append("a tall rectangular block standing upright")
                elif label.lower() == "cylinder":
                    text_queries.append("a cylinder")
                else:
                    text_queries.append(f"a {label}")
            text_queries = [text_queries]
            
            # Process inputs
            inputs = self.processor(
                text=text_queries, 
                images=pil_image, 
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process results
            target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs=outputs,
                threshold=self.confidence_threshold,
                target_sizes=target_sizes
            )[0]
            
            detected_objects = []
            
            # Parse detections
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            labels = results["labels"].cpu().numpy()

            # Calculate scale factor if image was downscaled
            scale_x = original_size[0] / pil_image.size[0]
            scale_y = original_size[1] / pil_image.size[1]

            if verbose:
                print(f"[OWLV2 DETECTOR] Found {len(boxes)} detections")
                if scale_x != 1.0 or scale_y != 1.0:
                    print(f"[OWLV2 DETECTOR] Scaling boxes back: scale_x={scale_x:.2f}, scale_y={scale_y:.2f}")

            for box, score, label_idx in zip(boxes, scores, labels):
                class_name = self.object_labels[label_idx]

                # Apply class-specific threshold
                class_threshold = self.class_thresholds.get(class_name, self.confidence_threshold)
                if score < class_threshold:
                    continue

                # Scale bounding box back to original image size
                x1, y1, x2, y2 = box.astype(int)
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                width = int(x2 - x1)
                height = int(y2 - y1)

                object_type = self._classify_object_type(class_name)

                detection = {
                    'class': class_name,
                    'type': object_type,
                    'confidence': float(score),
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'width': width,
                    'height': height,
                }

                detected_objects.append(detection)

                if verbose:
                    print(f"[OWLV2 DETECTOR] Detected {class_name} -> {object_type} (conf={score:.2%})")

            # Apply Non-Maximum Suppression (NMS) to remove duplicate detections
            detected_objects = self._apply_nms(detected_objects, iou_threshold=0.5)

            # Sort by confidence (highest first)
            detected_objects.sort(key=lambda x: x['confidence'], reverse=True)

            return detected_objects
            
        except Exception as e:
            print(f"[OWLV2 DETECTOR ERROR] Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove duplicate/overlapping detections.
        Also removes cross-class duplicates (same object detected as cube AND cuboid).
        """
        if len(detections) == 0:
            return []

        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        keep = []

        while len(detections) > 0:
            best = detections.pop(0)
            keep.append(best)

            # Remove overlapping detections (same class or different class)
            filtered = []
            for det in detections:
                iou = self._calculate_iou(best['bbox'], det['bbox'])
                if iou < iou_threshold:
                    filtered.append(det)

            detections = filtered

        return keep

    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1: (x1, y1, x2, y2)
            box2: (x1, y1, x2, y2)

        Returns:
            IoU value (0.0-1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            # No intersection
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def _classify_object_type(self, class_name: str) -> str:
        """
        Classify object as target or obstacle.

        Args:
            class_name: Object class name

        Returns:
            Object type: 'target' or 'obstacle'
        """
        class_lower = class_name.lower()

        # Target classification (objects to pick)
        # Cubes and cylinders are targets
        if 'cube' in class_lower and 'cuboid' not in class_lower:
            return 'target'

        if 'cylinder' in class_lower:
            return 'target'

        # Obstacle classification (objects to avoid)
        # Cuboids are obstacles
        if 'cuboid' in class_lower:
            return 'obstacle'

        # Everything else is an obstacle
        return 'obstacle'

    def get_targets(self, detections: List[Dict]) -> List[Dict]:
        """Get only target objects (cubes/cuboids to pick)"""
        return [d for d in detections if d['type'] == 'target']

    def get_obstacles(self, detections: List[Dict]) -> List[Dict]:
        """Get only obstacle objects (to avoid)"""
        return [d for d in detections if d['type'] == 'obstacle']

    def get_containers(self, detections: List[Dict]) -> List[Dict]:
        """Get only container objects"""
        return [d for d in detections if d['type'] == 'container']

    def get_objects_by_label(self, detections: List[Dict], label: str) -> List[Dict]:
        """
        Get objects by specific label.

        Args:
            detections: List of detections
            label: Label to filter by (e.g., "cube", "cylinder")

        Returns:
            List of detections matching the label
        """
        return [d for d in detections if d['class'].lower() == label.lower()]

    def clear_cache(self):
        """Clear GPU cache to free memory"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            print("[OWLV2 DETECTOR] GPU cache cleared")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage.

        Returns:
            Dictionary with memory usage info
        """
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()

        result = {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        }

        if self.device == "cuda" and torch.cuda.is_available():
            result['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            result['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024

        return result

    def detect_3d(self, rgb_image: np.ndarray, depth_image: np.ndarray, verbose: bool = False) -> List[Dict]:
        """
        Detect objects in 3D using RGB + Depth images.

        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            depth_image: Depth image as numpy array (H, W) in meters (radial distance from distance_to_camera)
            verbose: Print detection details

        Returns:
            List of detected objects with 3D positions and sizes
        """
        # First, run 2D detection
        detections_2d = self.detect(rgb_image, verbose=verbose)

        if len(detections_2d) == 0:
            return []

        # Debug: Reset unproject counter for each detection cycle
        if hasattr(self, '_unproject_debug_count'):
            self._unproject_debug_count = 0

        # Add 3D information to each detection
        detections_3d = []
        for i, det in enumerate(detections_2d):
            # Get 3D position from depth
            position_3d = self._unproject_to_3d(det['center'], depth_image)

            # Estimate 3D size from bounding box + depth
            size_3d = self._estimate_3d_size(det['bbox'], depth_image)

            # Add 3D information
            det['position_3d'] = position_3d
            det['size_3d'] = size_3d

            detections_3d.append(det)

            if verbose or i < 5:  # Print first 5 detections
                print(f"[3D DETECTION {i+1}] {det['class']}: bbox={det['bbox']}, center={det['center']}, pos_3d={position_3d}, size_3d={size_3d}")

        return detections_3d

    def _unproject_to_3d(self, image_point: Tuple[int, int], depth_image: np.ndarray) -> np.ndarray:
        """
        Unproject 2D image point to 3D camera frame using depth.

        SIMPLIFIED APPROACH:
        - depth_image contains radial distance from camera to object (distance_to_camera annotator)
        - We convert pixel (u,v) + radial_distance to 3D direction vector
        - Result is in camera frame (NOT world frame - transform happens later)

        Args:
            image_point: (x, y) in image coordinates
            depth_image: Depth image (H, W) in meters (radial distance from camera)

        Returns:
            3D position as np.array([x, y, z]) in camera frame
        """
        if self.camera_params is None:
            return None

        u, v = image_point

        # Get depth at this pixel (with bounds checking)
        h, w = depth_image.shape
        v = max(0, min(v, h - 1))
        u = max(0, min(u, w - 1))
        radial_distance = depth_image[v, u]

        # Debug: Print depth at detection point
        if not hasattr(self, '_unproject_debug_count'):
            self._unproject_debug_count = 0

        if self._unproject_debug_count < 5:
            print(f"\n[UNPROJECT DEBUG {self._unproject_debug_count+1}] Pixel ({u}, {v}): radial_distance={radial_distance:.3f}m")
            self._unproject_debug_count += 1

        # Check for invalid depth
        if radial_distance <= 0 or np.isnan(radial_distance) or np.isinf(radial_distance):
            if self._unproject_debug_count <= 5:
                print(f"  ❌ Invalid depth!")
            return None

        # Unproject using camera intrinsics
        fx = self.camera_params['fx']
        fy = self.camera_params['fy']
        cx = self.camera_params['cx']
        cy = self.camera_params['cy']

        # SIMPLIFIED: Convert pixel + radial distance to 3D direction
        # The radial distance is the distance from camera origin to the point
        # We need to find the 3D point (x, y, z) such that:
        # 1. It projects to pixel (u, v): u = fx*x/z + cx, v = fy*y/z + cy
        # 2. Its distance from origin is radial_distance: √(x² + y² + z²) = radial_distance

        # From projection equations:
        # x/z = (u - cx)/fx
        # y/z = (v - cy)/fy
        # Let's call these ratios: rx = (u-cx)/fx, ry = (v-cy)/fy

        rx = (u - cx) / fx
        ry = (v - cy) / fy

        # Now: x = rx*z, y = ry*z
        # Substitute into distance equation:
        # √((rx*z)² + (ry*z)² + z²) = radial_distance
        # √(z² * (rx² + ry² + 1)) = radial_distance
        # z * √(rx² + ry² + 1) = radial_distance
        # z = radial_distance / √(rx² + ry² + 1)

        scale = np.sqrt(rx**2 + ry**2 + 1.0)
        z_cam = radial_distance / scale
        x_cam = rx * z_cam
        y_cam = ry * z_cam

        if self._unproject_debug_count <= 5:
            print(f"  Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
            print(f"  Direction ratios: rx={rx:.3f}, ry={ry:.3f}, scale={scale:.3f}")
            print(f"  → Camera frame: ({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f})")
            print(f"  Verification: distance = {np.linalg.norm([x_cam, y_cam, z_cam]):.3f}m (should be {radial_distance:.3f}m)")

        return np.array([x_cam, y_cam, z_cam])

    def _estimate_3d_size(self, bbox: Tuple[int, int, int, int], depth_image: np.ndarray) -> np.ndarray:
        """
        Estimate 3D size of object from 2D bounding box + depth.

        SIMPLIFIED APPROACH:
        - Get median depth in bounding box
        - Convert bbox pixel dimensions to 3D dimensions using depth
        - Assume object depth ≈ min(width, height) for cubes

        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            depth_image: Depth image (H, W) in meters (radial distance)

        Returns:
            3D size as np.array([width, height, depth]) in meters
        """
        if self.camera_params is None:
            return None

        u1, v1, u2, v2 = bbox

        # Get median depth in bounding box region (more robust than mean)
        h, w = depth_image.shape
        v1 = max(0, min(v1, h - 1))
        v2 = max(0, min(v2, h - 1))
        u1 = max(0, min(u1, w - 1))
        u2 = max(0, min(u2, w - 1))

        depth_roi = depth_image[v1:v2, u1:u2]
        valid_depths = depth_roi[(depth_roi > 0) & ~np.isnan(depth_roi) & ~np.isinf(depth_roi)]

        if len(valid_depths) == 0:
            return None

        median_depth = np.median(valid_depths)

        # Get camera intrinsics
        fx = self.camera_params['fx']
        fy = self.camera_params['fy']
        cx = self.camera_params['cx']
        cy = self.camera_params['cy']

        # Unproject bbox corners to 3D to get actual size
        # Top-left corner
        rx1 = (u1 - cx) / fx
        ry1 = (v1 - cy) / fy
        scale1 = np.sqrt(rx1**2 + ry1**2 + 1.0)
        z1 = median_depth / scale1
        x1_3d = rx1 * z1
        y1_3d = ry1 * z1

        # Bottom-right corner
        rx2 = (u2 - cx) / fx
        ry2 = (v2 - cy) / fy
        scale2 = np.sqrt(rx2**2 + ry2**2 + 1.0)
        z2 = median_depth / scale2
        x2_3d = rx2 * z2
        y2_3d = ry2 * z2

        # Calculate 3D dimensions
        width_3d = abs(x2_3d - x1_3d)
        height_3d = abs(y2_3d - y1_3d)

        # Depth estimation: assume object is roughly cubic
        # For cubes: depth ≈ min(width, height)
        # For cuboids: depth might be different, but this is a reasonable estimate
        depth_3d = min(width_3d, height_3d)

        return np.array([width_3d, height_3d, depth_3d])

