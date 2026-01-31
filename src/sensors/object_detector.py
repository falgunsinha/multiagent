"""
Object Detector Module using YOLOv8
Detects and classifies objects for pick-and-place operations.

Automatically classifies:
- Target cubes: Small cubes to pick
- Containers: Larger cuboids/containers to place cubes in
- Obstacles: Robot, large objects, spheres, cones, etc.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class ObjectDetector:
    """YOLOv8-based object detector with automatic classification and TensorRT acceleration"""

    def __init__(self, model_path: str, confidence_threshold: float = 0.5, use_tensorrt: bool = True):
        """
        Initialize object detector.

        Args:
            model_path: Path to trained YOLOv8 model (.pt or .engine file)
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
            use_tensorrt: Use TensorRT for faster inference (2-3x speedup)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.use_tensorrt = use_tensorrt
        self.model = None
        self.tensorrt_model_path = None

        # Class names from training (updated for augmented_shapes_v2 dataset)
        self.class_names = ['cube', 'cylinder', 'container', 'cuboid']

        # Classification rules
        self.size_threshold_small = 0.15  # Objects < 15% of image are "small" (targets)
        self.size_threshold_large = 0.35  # Objects > 35% of image are "large" (obstacles)

        print(f"[OBJECT DETECTOR] Initializing with model: {model_path}")
        print(f"[OBJECT DETECTOR] Confidence threshold: {confidence_threshold}")
        print(f"[OBJECT DETECTOR] TensorRT acceleration: {use_tensorrt}")
    
    def load_model(self):
        """Load YOLOv8 model with optional TensorRT acceleration"""
        try:
            from ultralytics import YOLO

            if not self.model_path.exists():
                print(f"[OBJECT DETECTOR ERROR] Model not found: {self.model_path}")
                print("[OBJECT DETECTOR] Please train the model first using train_on_augmented_dataset_v2.py")
                return False

            # Check for pre-exported TensorRT engine
            engine_path = self.model_path.parent / f"{self.model_path.stem}.engine"

            if self.use_tensorrt and engine_path.exists():
                # Use pre-exported TensorRT engine (safest option)
                try:
                    print(f"[OBJECT DETECTOR] Loading pre-exported TensorRT engine: {engine_path}")
                    self.model = YOLO(str(engine_path))
                    self.tensorrt_model_path = engine_path
                    print("[OBJECT DETECTOR] TensorRT acceleration ENABLED (2-3x faster)")
                    print("[OBJECT DETECTOR] Using pre-exported engine (no GPU conflicts)")
                    return True
                except Exception as tensorrt_error:
                    print(f"[OBJECT DETECTOR WARNING] Failed to load TensorRT engine: {tensorrt_error}")
                    print("[OBJECT DETECTOR] Falling back to PyTorch model...")
                    # Fall through to load PyTorch model

            # Load PyTorch model (standard inference)
            print(f"[OBJECT DETECTOR] Loading PyTorch model: {self.model_path}")
            self.model = YOLO(str(self.model_path))
            print(f"[OBJECT DETECTOR] Model loaded successfully")

            # Warn if TensorRT was requested but not available
            if self.use_tensorrt and not engine_path.exists():
                print("[OBJECT DETECTOR] TensorRT engine not found")
                print(f"[OBJECT DETECTOR] To enable TensorRT (2-3x faster), run:")
                print(f"[OBJECT DETECTOR]   C:\\isaacsim\\python.bat scripts/export_yolov8_tensorrt.py")
                print("[OBJECT DETECTOR] Using standard PyTorch inference (still fast enough)")
                self.use_tensorrt = False

            return True

        except ImportError:
            print("[OBJECT DETECTOR ERROR] Ultralytics not installed!")
            print("Install with: pip install ultralytics")
            return False
        except Exception as e:
            print(f"[OBJECT DETECTOR ERROR] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def detect(self, rgb_image: np.ndarray, verbose: bool = False) -> List[Dict]:
        """
        Detect objects in RGB image.

        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            verbose: Print detection details

        Returns:
            List of detected objects with classification
        """
        if self.model is None:
            print("[OBJECT DETECTOR] Model not loaded, attempting to load...")
            if not self.load_model():
                print("[OBJECT DETECTOR ERROR] Failed to load model!")
                return []

        try:
            if verbose:
                print(f"[OBJECT DETECTOR] Input image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")
                print(f"[OBJECT DETECTOR] Input range: min={rgb_image.min()}, max={rgb_image.max()}")
                print(f"[OBJECT DETECTOR] Model expects classes: {self.class_names}")
                print(f"[OBJECT DETECTOR] Confidence threshold: {self.confidence_threshold}")

            # Run YOLOv8 inference
            if verbose:
                print("[OBJECT DETECTOR] Running YOLO model inference...")
            results = self.model(rgb_image, verbose=False, conf=0.1)  # Lower threshold to see all detections

            if verbose:
                print(f"[OBJECT DETECTOR] Got {len(results)} result objects from YOLO")

            detected_objects = []
            all_raw_detections = []  # Track all detections including low confidence

            # Get image dimensions
            img_height, img_width = rgb_image.shape[:2]
            img_area = img_height * img_width

            # Process results
            for i, result in enumerate(results):
                boxes = result.boxes

                if verbose:
                    print(f"[OBJECT DETECTOR] Result {i}: boxes={boxes}, len={len(boxes) if boxes is not None else 0}")

                if boxes is None or len(boxes) == 0:
                    if verbose:
                        print(f"[OBJECT DETECTOR] Result {i}: No boxes detected")
                        print(f"[DATASET DEBUG] Model detected NOTHING - possible reasons:")
                        print(f"  1. Camera image is blank/black (check camera_raw_rgb.png)")
                        print(f"  2. Objects look very different from training data")
                        print(f"  3. Lighting/colors don't match training images")
                        print(f"  4. Camera angle/perspective is wrong")
                        print(f"  5. Objects are too small/large compared to training")
                    continue
                
                for j, box in enumerate(boxes):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Get class and confidence
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())

                    # Track all raw detections for debugging
                    all_raw_detections.append({
                        'class': self.class_names[cls],
                        'confidence': conf,
                        'bbox': (int(x1), int(y1), int(x2), int(y2))
                    })

                    if verbose:
                        print(f"[OBJECT DETECTOR] Box {j}: class={self.class_names[cls]}, conf={conf:.3f}, bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)})")

                    # Skip low confidence detections
                    if conf < self.confidence_threshold:
                        if verbose:
                            print(f"  -> SKIPPED (conf {conf:.3f} < threshold {self.confidence_threshold})")
                        continue
                    
                    # Calculate bbox properties
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    bbox_area = bbox_width * bbox_height
                    bbox_size_ratio = bbox_area / img_area
                    
                    # Get detected class
                    detected_class = self.class_names[cls]
                    
                    # Classify object based on size and class
                    object_type = self._classify_object(
                        detected_class, 
                        bbox_size_ratio,
                        bbox_width,
                        bbox_height
                    )
                    
                    # Create detection object
                    detection = {
                        'class': detected_class,
                        'type': object_type,  # 'target', 'container', or 'obstacle'
                        'confidence': conf,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                        'size_ratio': bbox_size_ratio,
                        'width': int(bbox_width),
                        'height': int(bbox_height),
                    }
                    
                    detected_objects.append(detection)
                    
                    if verbose:
                        print(f"[OBJECT DETECTOR] Detected {detected_class} -> {object_type}")
                        print(f"                  Confidence: {conf:.2f}, Size: {bbox_size_ratio:.2%}")
            
            # Sort by confidence (highest first)
            detected_objects.sort(key=lambda x: x['confidence'], reverse=True)

            # Dataset debugging: analyze why detection failed
            if verbose and len(detected_objects) == 0:
                print(f"\n[DATASET DEBUG] ========== DETECTION FAILURE ANALYSIS ==========")
                print(f"[DATASET DEBUG] Total raw detections (conf > 0.1): {len(all_raw_detections)}")

                # Check training dataset
                from pathlib import Path
                import cv2
                training_images_dir = self.model_path.parent.parent / 'images'
                print(f"[DATASET DEBUG] Training dataset location: {training_images_dir}")

                if training_images_dir.exists():
                    train_images = list(training_images_dir.glob('*.jpg')) + list(training_images_dir.glob('*.png'))
                    print(f"[DATASET DEBUG] Found {len(train_images)} training images")

                    if len(train_images) > 0:
                        # Load first training image for comparison
                        sample_train_img = cv2.imread(str(train_images[0]))
                        if sample_train_img is not None:
                            # Compare image statistics
                            train_mean = sample_train_img.mean()
                            train_std = sample_train_img.std()
                            current_mean = rgb_image.mean()
                            current_std = rgb_image.std()

                            print(f"[DATASET DEBUG] Image statistics comparison:")
                            print(f"  Training image: mean={train_mean:.1f}, std={train_std:.1f}")
                            print(f"  Current image:  mean={current_mean:.1f}, std={current_std:.1f}")
                            print(f"  Difference:     mean_diff={abs(train_mean-current_mean):.1f}, std_diff={abs(train_std-current_std):.1f}")

                            if abs(train_mean - current_mean) > 50:
                                print(f"[DATASET DEBUG] WARNING: Large brightness difference! Images look very different.")
                            if abs(train_std - current_std) > 30:
                                print(f"[DATASET DEBUG] WARNING: Large contrast difference! Images look very different.")

                if len(all_raw_detections) == 0:
                    print(f"[DATASET DEBUG] PROBLEM: Model detected NOTHING at all!")
                    print(f"[DATASET DEBUG] This means:")
                    print(f"  - Training images look VERY different from current camera view")
                    print(f"  - Check training dataset images vs camera_raw_rgb.png")
                    print(f"  - Compare: lighting, colors, object appearance, background")
                else:
                    print(f"[DATASET DEBUG] Model detected {len(all_raw_detections)} objects but all below confidence threshold!")
                    print(f"[DATASET DEBUG] Raw detections (conf > 0.1):")
                    for det in all_raw_detections:
                        print(f"  - {det['class']}: conf={det['confidence']:.3f}, bbox={det['bbox']}")
                    print(f"[DATASET DEBUG] PROBLEM: Detections exist but confidence too low (< {self.confidence_threshold})")
                    print(f"[DATASET DEBUG] This means:")
                    print(f"  - Model recognizes objects but is uncertain")
                    print(f"  - Objects look SIMILAR but not identical to training data")
                    print(f"  - Try lowering confidence threshold or retrain with similar images")
                print(f"[DATASET DEBUG] ================================================\n")

            return detected_objects
            
        except Exception as e:
            print(f"[OBJECT DETECTOR ERROR] Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _classify_object(self, detected_class: str, size_ratio: float,
                        width: float, height: float) -> str:
        """
        Classify detected object as target, container, or obstacle.

        Args:
            detected_class: Class from YOLOv8 ('cube', 'cylinder', 'container', 'cuboid')
            size_ratio: Bbox area / image area
            width: Bbox width in pixels
            height: Bbox height in pixels

        Returns:
            'target', 'container', or 'obstacle'
        """
        # Rule 1: Very large objects are obstacles (likely robot or background)
        if size_ratio > self.size_threshold_large:
            return 'obstacle'

        # Rule 2: Containers are always containers
        if detected_class == 'container':
            return 'container'

        # Rule 3: Cuboids are typically obstacles (larger rectangular objects)
        if detected_class == 'cuboid':
            # Large cuboids are obstacles
            if size_ratio > self.size_threshold_small:
                return 'obstacle'
            else:
                # Small cuboids could be targets
                return 'target'

        # Rule 4: Cubes and cylinders are targets (unless very large)
        if detected_class in ['cube', 'cylinder']:
            if size_ratio < self.size_threshold_large:
                # Small and medium-sized cubes/cylinders are targets
                return 'target'
            else:
                # Very large cubes/cylinders are obstacles
                return 'obstacle'

        # Default: treat as obstacle
        return 'obstacle'
    
    def get_targets(self, detections: List[Dict]) -> List[Dict]:
        """Get only target objects (cubes/cylinders to pick)"""
        return [d for d in detections if d['type'] == 'target']
    
    def get_containers(self, detections: List[Dict]) -> List[Dict]:
        """Get only container objects (where to place)"""
        return [d for d in detections if d['type'] == 'container']
    
    def get_obstacles(self, detections: List[Dict]) -> List[Dict]:
        """Get only obstacle objects (to avoid)"""
        return [d for d in detections if d['type'] == 'obstacle']
    
    def visualize_detections(self, rgb_image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on image.
        
        Args:
            rgb_image: RGB image
            detections: List of detections from detect()
        
        Returns:
            Image with visualizations
        """
        img_vis = rgb_image.copy()
        
        # Color mapping for object types
        colors = {
            'target': (0, 255, 0),      # Green for targets
            'container': (0, 0, 255),   # Blue for containers
            'obstacle': (255, 0, 0),    # Red for obstacles
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = colors.get(det['type'], (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det['type']}: {det['class']} {det['confidence']:.2f}"
            cv2.putText(img_vis, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img_vis

