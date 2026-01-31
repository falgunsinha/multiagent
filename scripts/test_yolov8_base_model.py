"""
Test YOLOv8 Base Model (COCO Pretrained) in Isaac Sim
Simple script to test if YOLOv8 base model can detect objects in the current viewport.
Uses the pretrained YOLOv8n model trained on COCO dataset (80 classes).
"""

import asyncio
import numpy as np
import cv2
from pathlib import Path
import omni.ui as ui
from omni.kit.async_engine import run_coroutine
import omni.timeline
import omni.usd

# Isaac Sim imports
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera


class YOLOv8BaseModelTest:
    """Test YOLOv8 base model (COCO pretrained) on current Isaac Sim scene"""

    def __init__(self):
        self.window = None
        self.world = None
        self.main_camera = None
        self.yolo_model = None
        self.timeline = omni.timeline.get_timeline_interface()
        
        # UI elements
        self.detect_btn = None
        self.status_label = None
        self.result_label = None
        
        self.build_ui()
    
    def build_ui(self):
        """Build simple UI"""
        self.window = ui.Window("YOLOv8 Base Model Test", width=450, height=350)
        
        with self.window.frame:
            with ui.VStack(spacing=10):
                ui.Label("YOLOv8 Base Model Detection Test",
                        alignment=ui.Alignment.CENTER,
                        style={"font_size": 18})
                
                ui.Spacer(height=10)
                
                ui.Label("Using: YOLOv8n pretrained on COCO (80 classes)",
                        alignment=ui.Alignment.CENTER,
                        style={"font_size": 12, "color": 0xFF888888})
                
                ui.Spacer(height=10)
                
                ui.Label("Instructions:",
                        style={"font_size": 14})
                ui.Label("1. Make sure you have objects in the scene",
                        style={"font_size": 12})
                ui.Label("2. Make sure Main_Camera is at /World/Main_Camera",
                        style={"font_size": 12})
                ui.Label("3. Click 'Run Detection Test'",
                        style={"font_size": 12})
                
                ui.Spacer(height=10)
                
                self.detect_btn = ui.Button("Run Detection Test", 
                                            height=40, 
                                            clicked_fn=self._on_detect)
                
                ui.Spacer(height=10)
                
                self.status_label = ui.Label("Ready - Click button to test",
                                            alignment=ui.Alignment.CENTER,
                                            style={"font_size": 14})
                
                ui.Spacer(height=10)
                
                self.result_label = ui.Label("",
                                            alignment=ui.Alignment.LEFT,
                                            word_wrap=True,
                                            style={"font_size": 12})
    
    def _update_status(self, message):
        """Update status label"""
        if self.status_label:
            self.status_label.text = message
    
    def _update_results(self, message):
        """Update results label"""
        if self.result_label:
            self.result_label.text = message
    
    def _on_detect(self):
        """Run detection test"""
        self._update_status("Running detection...")
        self._update_results("")
        run_coroutine(self._run_detection_test())
    
    async def _run_detection_test(self):
        """Run YOLOv8 base model detection test"""
        try:
            # Initialize world if not already done
            if self.world is None:
                self.world = World.instance()
                if self.world is None:
                    self.world = World(stage_units_in_meters=1.0)
            
            # Check if Main_Camera exists
            stage = omni.usd.get_context().get_stage()
            camera_prim = stage.GetPrimAtPath("/World/Main_Camera")
            
            if not camera_prim or not camera_prim.IsValid():
                self._update_status("ERROR: Main_Camera not found!")
                self._update_results("Please add Main_Camera to /World/Main_Camera")
                return
            
            # Initialize camera
            if self.main_camera is None:
                print("[CAMERA] Initializing Main_Camera...")
                self.main_camera = Camera(
                    prim_path="/World/Main_Camera",
                    resolution=(640, 640),
                    frequency=10
                )
                self.main_camera.initialize()

                # Wait a few frames for camera to fully initialize
                print("[CAMERA] Waiting for camera initialization...")
                for _ in range(10):
                    await omni.kit.app.get_app().next_update_async()
            
            # Load YOLOv8 base model
            if self.yolo_model is None:
                print("[YOLO] Loading YOLOv8n base model (COCO pretrained)...")
                self._update_status("Loading YOLOv8n base model...")
                
                try:
                    from ultralytics import YOLO
                    
                    # Load pretrained YOLOv8n model (will auto-download if not in cache)
                    # This uses the model from C:\Users\Simulation\.cache\ultralytics\yolov8n.pt
                    self.yolo_model = YOLO('yolov8n.pt')
                    print("[YOLO] YOLOv8n base model loaded successfully")
                    print("[YOLO] Model trained on COCO dataset (80 classes)")
                    
                except Exception as e:
                    self._update_status("ERROR: Failed to load YOLOv8!")
                    self._update_results(f"Error: {str(e)}\n\nMake sure ultralytics is installed:\npip install ultralytics")
                    return
            
            # Start timeline if not running
            if not self.timeline.is_playing():
                print("[TIMELINE] Starting timeline...")
                self.timeline.play()
            
            # Wait for camera to warm up
            print("[CAMERA] Waiting for camera to warm up...")
            self._update_status("Warming up camera...")
            max_warmup_attempts = 100
            camera_ready = False

            for attempt in range(max_warmup_attempts):
                await omni.kit.app.get_app().next_update_async()

                try:
                    rgba = self.main_camera.get_rgba()
                    if rgba is not None and len(rgba.shape) == 3:
                        # Check if it's a valid 3D array (height, width, channels)
                        if rgba.shape[2] >= 3:
                            rgb = rgba[:, :, :3]
                            if rgb.max() > 0:
                                print(f"[CAMERA] Camera ready after {attempt+1} frames")
                                print(f"[CAMERA] Image shape: {rgba.shape}, RGB range: {rgb.min()}-{rgb.max()}")
                                camera_ready = True
                                break
                except Exception as e:
                    # Skip this frame if there's an indexing error
                    if attempt % 10 == 0:
                        print(f"[CAMERA] Warming up... attempt {attempt+1}/{max_warmup_attempts}")
                    continue

            if not camera_ready:
                self._update_status("ERROR: Camera failed to initialize!")
                self._update_results("Camera did not produce valid images after 100 frames.\nTry:\n1. Check camera position\n2. Restart timeline\n3. Reload scene")
                return

            # Capture image
            print("[CAMERA] Capturing image...")
            self._update_status("Capturing image...")

            rgba = self.main_camera.get_rgba()

            if rgba is None or len(rgba.shape) != 3:
                self._update_status("ERROR: Failed to capture valid image!")
                self._update_results(f"Camera returned invalid data.\nShape: {rgba.shape if rgba is not None else 'None'}")
                return

            rgb_frame = rgba[:, :, :3]
            print(f"[CAMERA] Image captured: {rgb_frame.shape}, range: {rgb_frame.min()}-{rgb_frame.max()}")
            
            # Save raw image for debugging
            debug_dir = Path("C:/isaacsim/cobotproject/models/yolov8_base_test")
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / "test_input.png"), 
                       cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
            print(f"[DEBUG] Raw image saved to: {debug_dir / 'test_input.png'}")
            
            # Run detection with YOLOv8 base model
            print("[YOLO] Running detection with base model...")
            self._update_status("Running YOLOv8 detection...")

            # Run inference with VERY LOW confidence threshold to see everything
            print("[YOLO] Using confidence threshold: 0.05 (very low to see all detections)")
            results = self.yolo_model(rgb_frame, conf=0.05, verbose=True)
            
            # Parse results
            detections = []
            all_detections = []  # Store ALL detections including low confidence

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = result.names[cls]

                    det = {
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'center_x': int((x1 + x2) / 2),
                        'center_y': int((y1 + y2) / 2)
                    }

                    all_detections.append(det)

                    # Only add to main detections if confidence > 0.15
                    if conf > 0.15:
                        detections.append(det)

            # Print ALL detections (including very low confidence)
            print(f"\n[DEBUG] Total detections (conf > 0.05): {len(all_detections)}")
            for i, det in enumerate(all_detections, 1):
                print(f"  {i}. {det['class']} - confidence: {det['confidence']:.3f}")
            
            # Display results
            print(f"\n[RESULTS] High confidence detections (>0.15): {len(detections)}")
            results_text = f"High Confidence (>0.15): {len(detections)} objects\n"
            results_text += f"All Detections (>0.05): {len(all_detections)} objects\n\n"

            if len(detections) == 0:
                results_text += "No high-confidence detections!\n\n"

                if len(all_detections) > 0:
                    results_text += f"Low confidence detections ({len(all_detections)}):\n"
                    for i, det in enumerate(all_detections[:5], 1):  # Show first 5
                        results_text += f"{i}. {det['class']} (conf: {det['confidence']:.3f})\n"
                    results_text += "\n"

                results_text += "Possible reasons:\n"
                results_text += "- 3D renders look different from\n"
                results_text += "  real photos in COCO dataset\n"
                results_text += "- Objects are stylized/simplified\n"
                results_text += "- Need custom trained model\n"
            else:
                for i, det in enumerate(detections, 1):
                    print(f"  {i}. {det['class']} - confidence: {det['confidence']:.2f}")
                    results_text += f"{i}. {det['class']}\n"
                    results_text += f"   Confidence: {det['confidence']:.2f}\n"
                    results_text += f"   Position: ({det['center_x']}, {det['center_y']})\n\n"
            
            # Draw bounding boxes on image (show ALL detections with color coding)
            vis_image = rgb_frame.copy()
            for det in all_detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']

                # Color code by confidence:
                # Green (0, 255, 0) for high confidence (>0.5)
                # Yellow (255, 255, 0) for medium confidence (0.15-0.5)
                # Red (255, 0, 0) for low confidence (<0.15)
                if conf > 0.5:
                    color = (0, 255, 0)  # Green
                    thickness = 2
                elif conf > 0.15:
                    color = (255, 255, 0)  # Yellow
                    thickness = 2
                else:
                    color = (255, 0, 0)  # Red
                    thickness = 1

                # Draw box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
                # Draw label
                label = f"{det['class']} {conf:.3f}"
                cv2.putText(vis_image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            
            # Save visualization
            cv2.imwrite(str(debug_dir / "test_output.png"),
                       cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print(f"[DEBUG] Detection visualization saved to: {debug_dir / 'test_output.png'}")
            
            results_text += f"\nImages saved to:\n{debug_dir}"
            
            self._update_status(f"Detection complete! Found {len(detections)} objects")
            self._update_results(results_text)
            
        except Exception as e:
            self._update_status(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()


# Create and show the test window
test_window = YOLOv8BaseModelTest()

