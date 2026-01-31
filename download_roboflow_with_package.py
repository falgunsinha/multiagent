"""
Download Roboflow FinalShapeSegment Model using roboflow package
This script uses the roboflow package which handles SSL certificates better
Run with SYSTEM Python (not Isaac Sim's Python): py download_roboflow_with_package.py
"""

from roboflow import Roboflow
from pathlib import Path

print("="*70)
print("ROBOFLOW MODEL DOWNLOADER (Using Roboflow Package)")
print("="*70)
print("\nModel: FinalShapeSegment (Instance Segmentation)")
print("URL: https://universe.roboflow.com/shapesegmentation/finalshapesegment")
print("\n" + "="*70)

# API Keys
print("\nYour API Keys:")
print("  Private API Key: WF1HIzXyqs1Ioxdsldgc")
print("  Publishable API Key: rf_PX6Tp16U8Pgb3O8rRFpWfQdXgOf2")
print("  Folder API Key: 08JHY7n4nteutqdUtVzR")

print("\n" + "="*70)
print("STEP 1: Select API Key")
print("="*70)
print("\n1. Private API Key (recommended)")
print("2. Publishable API Key")
print("3. Folder API Key")
print("4. Enter custom API key")

choice = input("\nSelect option (1-4) [default: 1]: ").strip() or "1"

if choice == "1":
    api_key = "WF1HIzXyqs1Ioxdsldgc"
    print("Using Private API Key")
elif choice == "2":
    api_key = "rf_PX6Tp16U8Pgb3O8rRFpWfQdXgOf2"
    print("Using Publishable API Key")
elif choice == "3":
    api_key = "08JHY7n4nteutqdUtVzR"
    print("Using Folder API Key")
else:
    api_key = input("Enter your API key: ").strip()

if not api_key:
    print("\nERROR: No API key provided!")
    exit(1)

print("\n" + "="*70)
print("STEP 2: Downloading Dataset from Roboflow")
print("="*70)

try:
    # Initialize Roboflow
    print("\n[1/4] Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)
    
    # Access the project
    print("[2/4] Accessing FinalShapeSegment project...")
    project = rf.workspace("shapesegmentation").project("finalshapesegment")
    
    # Get version 1
    print("[3/4] Getting model version 1...")
    version = project.version(1)
    
    # Download in YOLOv8 format
    print("[4/4] Downloading dataset in YOLOv8 format...")
    print("\nThis may take several minutes (4,153 images)...")
    
    # Download to current directory
    dataset = version.download("yolov8")
    
    print("\n" + "="*70)
    print("SUCCESS! Dataset Downloaded")
    print("="*70)
    print(f"\nDownloaded to: {dataset.location}")
    print("\nFolder structure:")
    print("  FinalShapeSegment-1/")
    print("  ├── train/          (training images)")
    print("  ├── valid/          (validation images)")
    print("  ├── test/           (test images)")
    print("  ├── data.yaml       (dataset configuration)")
    print("  └── README.roboflow.txt")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\nOption A: Train your own model")
    print("-" * 70)
    print("py -m pip install ultralytics")
    print("py")
    print(">>> from ultralytics import YOLO")
    print(">>> model = YOLO('yolov8n-seg.pt')")
    print(f">>> model.train(data=r'{dataset.location}\\data.yaml', epochs=100)")
    print(">>> # Trained weights: runs/segment/train/weights/best.pt")
    
    print("\n\nOption B: Use Roboflow API in Isaac Sim (No training needed)")
    print("-" * 70)
    print("1. Open Isaac Sim 5.0.0")
    print("2. Load: cobotproject/scripts/test_roboflow_model.py")
    print("3. Click Run")
    print("4. Enter API key when prompted")
    print("5. Detect objects in real-time!")
    
    print("\n" + "="*70)
    print("DATASET INFO")
    print("="*70)
    print("\nClasses (10):")
    print("  - circle, cone, cube, cylinder, heart")
    print("  - pyramid, rectangle, sphere, square, triangle")
    print(f"\nDataset location: {dataset.location}")
    print("\n" + "="*70)

except Exception as e:
    print("\n" + "="*70)
    print("ERROR!")
    print("="*70)
    print(f"\n{str(e)}")
    print("\nPossible issues:")
    print("1. Invalid API key - check https://app.roboflow.com/settings/api")
    print("2. Network/firewall blocking connection")
    print("3. SSL certificate issue")
    print("\nIf SSL error persists, contact your IT department about")
    print("the self-signed certificate in the certificate chain.")
    print("\n" + "="*70)
    raise

