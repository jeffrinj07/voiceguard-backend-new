# Quick OpenCV Installation & Verification Script
# Run this to install OpenCV and test the COVID model loading

import subprocess
import sys

print("=" * 70)
print("🔧 INSTALLING OPENCV (cv2)")
print("=" * 70)

# Install OpenCV
print("\n📦 Installing opencv-python...")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "opencv-python"],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print("✅ OpenCV installed successfully!")
else:
    print(f"❌ OpenCV installation failed: {result.stderr}")
    sys.exit(1)

print("\n" + "=" * 70)
print("🧪 TESTING COVID MODEL LOADING")
print("=" * 70)

# Test imports
print("\n1️⃣ Testing imports...")
try:
    import cv2
    print("   ✅ cv2 (OpenCV) imported successfully")
except ImportError as e:
    print(f"   ❌ cv2 import failed: {e}")
    sys.exit(1)

try:
    import tensorflow as tf
    print("   ✅ TensorFlow imported successfully")
except ImportError as e:
    print(f"   ❌ TensorFlow import failed: {e}")
    sys.exit(1)

# Test model loading
print("\n2️⃣ Testing COVID model loading...")
try:
    model_path = "models/voiceguard_audio_model_final.keras"
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"   ✅ Model loaded successfully!")
    print(f"   📊 Input shape: {model.input_shape}")
    print(f"   📊 Output shape: {model.output_shape}")
except Exception as e:
    print(f"   ❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\n🚀 You can now restart your server with:")
print("   python app.py")
print("\n💡 The COVID Audio Model should now show as ✅ LOADED")
print("=" * 70)
