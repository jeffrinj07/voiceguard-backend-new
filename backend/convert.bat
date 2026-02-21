@echo off
echo Starting TFLite conversion (Python 3.11)...
py -3.11 "c:\Users\Lenovo\Desktop\voiceguard_project\backend\convert_to_tflite.py"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ FAILED! Please run install_local_deps.bat first.
) else (
    echo.
    echo ✅ SUCCESS! Check your models folder for the .tflite file.
)
pause
