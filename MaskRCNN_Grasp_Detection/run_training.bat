@echo off
echo ============================================================
echo Starting MaskRCNN Training
echo ============================================================

call venv_maskrcnn\Scripts\activate.bat

echo.
echo Training MaskRCNN on cube/cylinder dataset...
echo.

python train_maskrcnn.py

echo.
echo ============================================================
echo Training Complete!
echo ============================================================
echo.
echo Trained weights saved to: Mask_RCNN\mask_rcnn_cube_cylinder.h5
echo.
echo Next steps:
echo   1. Copy weights to: cobotproject\models\mask_rcnn_cube_cylinder.h5
echo   2. Update detector to load custom weights
echo.

pause

