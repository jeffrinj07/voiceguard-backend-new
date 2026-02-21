@echo off
echo ===============================================
echo Installing Local Dependencies for Python 3.11
echo ===============================================
echo Targeting your Python 3.11.9 installation specifically.
py -3.11 -m pip install --upgrade pip
py -3.11 -m pip install tensorflow-cpu librosa opencv-python-headless numpy
echo.
echo ===============================================
echo Verifying Installation on 3.11...
py -3.11 -c "import tensorflow as tf; print('✅ TensorFlow Version (3.11):', tf.__version__)"
echo ===============================================
pause
