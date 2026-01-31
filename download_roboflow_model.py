"""
Download Roboflow FinalShapeSegment Model
This script downloads the YOLOv8 model from Roboflow Universe
Uses direct API calls (no roboflow package needed)
"""

import requests
import zipfile
from pathlib import Path
import os

print("="*70)
print("ROBOFLOW MODEL DOWNLOADER")
print("="*70)
print("\nModel: FinalShapeSegment (Instance Segmentation)")
print("URL: https://universe.roboflow.com/shapesegmentation/finalshapesegment")
print("\n" + "="*70)

# Step 1: Get API Key
print("\nSTEP 1: Get your Roboflow API Key")
print("-" * 70)
print("1. Go to: https://app.roboflow.com/")
print("2. Sign up for FREE account (if you don't have one)")
print("3. Go to: https://app.roboflow.com/settings/api")
print("4. Copy your API key")
print("-" * 70)

api_key = input("\nPaste your API key here: ").strip()

if not api_key:
    print("\nERROR: No API key provided!")
    print("Get your free API key from: https://app.roboflow.com/settings/api")
    exit(1)

print("\n" + "="*70)
print("STEP 2: Downloading Dataset from Roboflow")
print("="*70)

try:
    # Download URL for YOLOv8 format
    download_url = f"https://universe.roboflow.com/ds/shapesegmentation/finalshapesegment/1?key={api_key}&format=yolov8"

    print("\n[1/3] Requesting download from Roboflow...")
    print(f"URL: {download_url}")

    # Download the ZIP file
    print("[2/3] Downloading dataset (this may take several minutes)...")
    response = requests.get(download_url, stream=True)

    if response.status_code != 200:
        print(f"\nERROR: Download failed with status code {response.status_code}")
        print("Response:", response.text)
        print("\nPossible issues:")
        print("1. Invalid API key")
        print("2. No access to this dataset")
        print("3. Network issue")
        exit(1)

    # Save ZIP file
    zip_path = Path("FinalShapeSegment-1.zip")
    total_size = int(response.headers.get('content-length', 0))

    with open(zip_path, 'wb') as f:
        if total_size == 0:
            f.write(response.content)
        else:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')

    print("\n[3/3] Extracting dataset...")

    # Extract ZIP
    extract_dir = Path("FinalShapeSegment-1")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # Remove ZIP file
    zip_path.unlink()

    dataset_location = extract_dir.absolute()
    
    print("\n" + "="*70)
    print("SUCCESS! Dataset Downloaded")
    print("="*70)
    print(f"\nDownloaded to: {dataset_location}")
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
    print("from ultralytics import YOLO")
    print("model = YOLO('yolov8n-seg.pt')  # Load base segmentation model")
    print(f"model.train(data=r'{dataset_location}\\data.yaml', epochs=100)")
    print("# Trained weights will be in: runs/segment/train/weights/best.pt")

    print("\n\nOption B: Use pre-trained weights (if available)")
    print("-" * 70)
    print("Check if the download includes pre-trained weights:")

    # Check for weights
    weights_path = dataset_location / "weights" / "best.pt"
    if weights_path.exists():
        print(f"✓ Found pre-trained weights at: {weights_path}")
        print("\nYou can use them directly in your script:")
        print(f"model = YOLO(r'{weights_path}')")
    else:
        print("✗ No pre-trained weights found")
        print("You need to train the model first (see Option A above)")

    print("\n" + "="*70)
    print("DATASET INFO")
    print("="*70)
    print("\nClasses (10):")
    print("  - circle, cone, cube, cylinder, heart")
    print("  - pyramid, rectangle, sphere, square, triangle")
    print(f"\nDataset location: {dataset_location}")
    print("\n" + "="*70)
    
except Exception as e:
    print("\n" + "="*70)
    print("ERROR!")
    print("="*70)
    print(f"\n{str(e)}")
    print("\nPossible issues:")
    print("1. Invalid API key - check https://app.roboflow.com/settings/api")
    print("2. No internet connection")
    print("3. Roboflow package not installed: pip install roboflow")
    print("\n" + "="*70)
    import traceback
    traceback.print_exc()

