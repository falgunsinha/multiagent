@echo off
echo ============================================================
echo MaskRCNN Training Setup
echo ============================================================

echo.
echo Step 1: Creating Python 3.7 virtual environment...
py -3.7 -m venv venv_maskrcnn

echo.
echo Step 2: Activating virtual environment...
call venv_maskrcnn\Scripts\activate.bat

echo.
echo Step 3: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 4: Installing TensorFlow 1.15 (CPU version)...
pip install tensorflow==1.15.0

echo.
echo Step 5: Installing Keras 2.1.6...
pip install keras==2.1.6

echo.
echo Step 6: Installing other dependencies...
pip install numpy==1.16.4
pip install scipy==1.2.1
pip install Pillow==6.1.0
pip install cython==0.29.13
pip install matplotlib==3.0.3
pip install scikit-image==0.15.0
pip install opencv-python==4.1.0.25
pip install h5py==2.9.0
pip install imgaug==0.2.9
pip install IPython[all]==7.6.1

echo.
echo Step 7: Downloading COCO pre-trained weights...
cd Mask_RCNN
if not exist mask_rcnn_coco.h5 (
    echo Downloading mask_rcnn_coco.h5...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5' -OutFile 'mask_rcnn_coco.h5'"
) else (
    echo COCO weights already exist.
)
cd ..

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo To train the model:
echo   1. Activate environment: venv_maskrcnn\Scripts\activate.bat
echo   2. Run training: python train_maskrcnn.py
echo.
echo ============================================================

pause

