"""
Roboflow Object Detector Module
Detects and classifies objects using Roboflow Hosted API with inference_sdk.

Automatically classifies:
- Target cubes: Small cubes to pick
- Obstacles: Other shapes (cylinder, sphere, cone, etc.)
- Container: Hardcoded, not detected
"""

import numpy as np
from typing import List, Dict, Optional
from inference_sdk import InferenceHTTPClient


class RoboflowObjectDetector:
    """Roboflow-based object detector using inference_sdk"""

    def __init__(
        self, 
        api_key: str,
        model_id: str = "finalshapesegment/1",
        api_url: str = "https://detect.roboflow.com",
        confidence_threshold: float = 0.20,
        overlap_threshold: float = 0.50
    ):
        """
        Initialize Roboflow object detector.

        Args:
            api_key: Roboflow API key
            model_id: Model ID in format "workspace/project/version"
            api_url: Roboflow API URL (hosted or local server)
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
            overlap_threshold: IoU threshold for NMS (0.0-1.0)
        """
        self.api_key = api_key
        self.model_id = model_id
        self.api_url = api_url
        self.confidence_threshold = confidence_threshold
        self.overlap_threshold = overlap_threshold
        
        # Initialize inference client
        self.client = InferenceHTTPClient(
            api_url=self.api_url,
            api_key=self.api_key
        )
        
        print(f"[ROBOFLOW DETECTOR] Initialized")
        print(f"[ROBOFLOW DETECTOR] Model: {model_id}")
        print(f"[ROBOFLOW DETECTOR] API URL: {api_url}")
        print(f"[ROBOFLOW DETECTOR] Confidence threshold: {confidence_threshold}")

    def detect(self, rgb_image: np.ndarray, verbose: bool = False) -> List[Dict]:
        """
        Detect objects in RGB image using Roboflow API.

        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            verbose: Print detection details

        Returns:
            List of detected objects with classification
        """
        try:
            if verbose:
                print(f"[ROBOFLOW DETECTOR] Input image shape: {rgb_image.shape}")
            
            # Run inference using inference_sdk
            result = self.client.infer(
                rgb_image, 
                model_id=self.model_id
            )
            
            # Parse predictions
            predictions = result.get('predictions', [])
            
            if verbose:
                print(f"[ROBOFLOW DETECTOR] API returned {len(predictions)} predictions")
            
            detected_objects = []
            
            for pred in predictions:
                # Get prediction data
                class_name = pred.get('class', 'unknown')
                confidence = pred.get('confidence', 0.0)
                x = pred.get('x', 0)
                y = pred.get('y', 0)
                width = pred.get('width', 0)
                height = pred.get('height', 0)
                
                # Skip low confidence detections
                if confidence < self.confidence_threshold:
                    if verbose:
                        print(f"[ROBOFLOW DETECTOR] Skipped {class_name} (conf={confidence:.2f} < {self.confidence_threshold})")
                    continue
                
                # Convert center coordinates to bbox
                x1 = int(x - width / 2)
                y1 = int(y - height / 2)
                x2 = int(x + width / 2)
                y2 = int(y + height / 2)
                
                # Classify object type (cube = target, others = obstacle)
                object_type = 'target' if class_name == 'cube' else 'obstacle'
                
                # Create detection object
                detection = {
                    'class': class_name,
                    'type': object_type,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'center': (int(x), int(y)),
                    'width': int(width),
                    'height': int(height),
                }
                
                # Add segmentation points if available
                if 'points' in pred:
                    detection['segmentation'] = pred['points']
                
                detected_objects.append(detection)
                
                if verbose:
                    print(f"[ROBOFLOW DETECTOR] Detected {class_name} -> {object_type} (conf={confidence:.2%})")
            
            # Sort by confidence (highest first)
            detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
            
            return detected_objects
            
        except Exception as e:
            print(f"[ROBOFLOW DETECTOR ERROR] Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_targets(self, detections: List[Dict]) -> List[Dict]:
        """Get only target objects (cubes to pick)"""
        return [d for d in detections if d['type'] == 'target']
    
    def get_obstacles(self, detections: List[Dict]) -> List[Dict]:
        """Get only obstacle objects (to avoid)"""
        return [d for d in detections if d['type'] == 'obstacle']

