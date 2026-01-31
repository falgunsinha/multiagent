"""
Test Roboflow YOLOv8 Model (FinalShapeSegment) in Isaac Sim 5.0.0
Downloads and uses the instance segmentation model from Roboflow Universe.
Model: https://universe.roboflow.com/shapesegmentation/finalshapesegment
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


class RoboflowModelTest:
    """Test Roboflow YOLOv8 instance segmentation model on Isaac Sim scene"""

    def __init__(self):
        self.window = None
        self.world = None
        self.main_camera = None
        self.yolo_model = None
        self.timeline = omni.timeline.get_timeline_interface()
        
        # UI elements
        self.download_btn = None
        self.detect_btn = None
        self.status_label = None
        self.result_label = None
        
        self.build_ui()
    
    def build_ui(self):
        """Build simple UI"""
        self.window = ui.Window("Roboflow Model Test", width=500, height=400)

        with self.window.frame:
            with ui.VStack(spacing=10):
                ui.Label("Roboflow YOLOv8 Instance Segmentation",
                        alignment=ui.Alignment.CENTER,
                        style={"font_size": 18})

                ui.Spacer(height=10)

                ui.Label("Model: FinalShapeSegment (API Mode)",
                        alignment=ui.Alignment.CENTER,
                        style={"font_size": 12, "color": 0xFF00FF00})

                ui.Label("Classes: circle, cone, cube, cylinder, heart,",
                        alignment=ui.Alignment.CENTER,
                        style={"font_size": 11, "color": 0xFF888888})

                ui.Label("pyramid, rectangle, sphere, square, triangle",
                        alignment=ui.Alignment.CENTER,
                        style={"font_size": 11, "color": 0xFF888888})

                ui.Spacer(height=10)

                ui.Label("Uses Roboflow Hosted API (No Installation Required)",
                        alignment=ui.Alignment.CENTER,
                        style={"font_size": 11, "color": 0xFFFFAA00})

                ui.Label("Make sure Main_Camera is at /World/Main_Camera",
                        alignment=ui.Alignment.CENTER,
                        style={"font_size": 11, "color": 0xFF888888})

                ui.Spacer(height=10)

                self.detect_btn = ui.Button("Run Detection Test",
                                            height=50,
                                            clicked_fn=self._on_detect)

                ui.Spacer(height=10)

                self.status_label = ui.Label("Ready - Click button to detect objects",
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
        """Run detection using Roboflow model via direct API"""
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

                # Wait for camera initialization
                print("[CAMERA] Waiting for camera initialization...")
                for _ in range(10):
                    await omni.kit.app.get_app().next_update_async()

            # Get API key if not already set
            if self.yolo_model is None:
                print("[ROBOFLOW] Enter API key in console...")
                self._update_status("Enter API key in console...")
                self._update_results("Get your free API key from:\nhttps://app.roboflow.com/settings/api\n\nPaste it in the Script Editor console below.")

                print("\n" + "="*60)
                print("ROBOFLOW API KEY REQUIRED")
                print("="*60)
                print("Get your free API key from: https://app.roboflow.com/settings/api")
                api_key = input("Enter your Roboflow API key: ").strip()

                if not api_key:
                    self._update_status("ERROR: No API key provided")
                    return

                # Store API key for future use
                self.yolo_model = api_key  # We'll use this as API key storage
                print("[ROBOFLOW] API key saved")

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
                        if rgba.shape[2] >= 3:
                            rgb = rgba[:, :, :3]
                            if rgb.max() > 0:
                                print(f"[CAMERA] Camera ready after {attempt+1} frames")
                                camera_ready = True
                                break
                except Exception as e:
                    if attempt % 10 == 0:
                        print(f"[CAMERA] Warming up... attempt {attempt+1}/{max_warmup_attempts}")
                    continue

            if not camera_ready:
                self._update_status("ERROR: Camera failed to initialize!")
                return

            # Capture image
            print("[CAMERA] Capturing image...")
            self._update_status("Capturing image...")

            rgba = self.main_camera.get_rgba()
            if rgba is None or len(rgba.shape) != 3:
                self._update_status("ERROR: Failed to capture valid image!")
                return

            rgb_frame = rgba[:, :, :3]

            # Save raw image
            debug_dir = Path("C:/isaacsim/cobotproject/models/roboflow_finalshapesegment/test_results")
            debug_dir.mkdir(parents=True, exist_ok=True)

            input_path = str(debug_dir / "test_input.png")
            cv2.imwrite(input_path, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
            print(f"[DEBUG] Raw image saved to: {input_path}")

            # Run detection with Roboflow API (direct HTTP request)
            print("[ROBOFLOW] Running inference via API...")
            self._update_status("Running Roboflow inference...")

            # Use Roboflow Hosted Inference API
            import requests
            import base64
            import urllib3

            # Disable SSL warnings (due to corporate firewall)
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

            # Convert image to base64
            with open(input_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')

            # Roboflow API endpoint
            api_key = self.yolo_model  # We stored the API key here
            api_url = f"https://detect.roboflow.com/finalshapesegment/1"

            params = {
                "api_key": api_key,
                "confidence": 20,  # Minimum confidence threshold (20%)
                "overlap": 50  # Higher overlap threshold to remove duplicates
            }

            # Send request (SSL verification disabled for corporate firewall)
            response = requests.post(
                api_url,
                params=params,
                data=img_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                verify=False  # Bypass SSL certificate verification
            )

            if response.status_code != 200:
                self._update_status(f"ERROR: API request failed ({response.status_code})")
                self._update_results(f"API Error: {response.text}\n\nCheck your API key!")
                return

            result = response.json()

            # Parse results
            detections = []
            for pred in result.get('predictions', []):
                det = {
                    'class': pred['class'],
                    'confidence': pred['confidence'],
                    'bbox': [int(pred['x'] - pred['width']/2),
                            int(pred['y'] - pred['height']/2),
                            int(pred['x'] + pred['width']/2),
                            int(pred['y'] + pred['height']/2)],
                    'center_x': int(pred['x']),
                    'center_y': int(pred['y'])
                }
                detections.append(det)

            # Print detections
            print(f"\n[RESULTS] Total detections: {len(detections)}")
            for i, det in enumerate(detections, 1):
                print(f"  {i}. {det['class']} - confidence: {det['confidence']:.3f}")

            # Display results
            results_text = f"Detected {len(detections)} objects:\n\n"

            if len(detections) == 0:
                results_text += "No objects detected!\n"
            else:
                for i, det in enumerate(detections, 1):
                    results_text += f"{i}. {det['class']}\n"
                    results_text += f"   Confidence: {det['confidence']:.2f}\n"
                    results_text += f"   Position: ({det['center_x']}, {det['center_y']})\n\n"

            # Draw bounding boxes
            vis_image = rgb_frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                color = (0, 255, 0) if conf > 0.8 else (255, 255, 0)

                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                label = f"{det['class']} {conf:.2f}"
                cv2.putText(vis_image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Save visualization
            cv2.imwrite(str(debug_dir / "test_output.png"),
                       cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

            results_text += f"\nImages saved to:\n{debug_dir}"

            self._update_status(f"Detection complete! Found {len(detections)} objects")
            self._update_results(results_text)

        except Exception as e:
            self._update_status(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()


# Create and show the test window
test_window = RoboflowModelTest()

