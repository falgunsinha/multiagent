@echo off
echo ============================================================
echo PyTorch MaskRCNN Training
echo ============================================================

echo.
echo This will train MaskRCNN in PyTorch using the existing dataset
echo.

echo Installing required packages...
C:\isaacsim\python.bat -m pip install scikit-image

echo.
echo Starting training...
C:\isaacsim\python.bat train_pytorch_maskrcnn.py

echo.
echo ============================================================
echo Training Complete!
echo ============================================================

pause

