"""
Quick test script to verify:
1. Firebase Admin is importable
2. COVID Model can load
"""
import sys

print("=" * 70)
print("🧪 VOICEGUARD QUICK TEST")
print("=" * 70)

# Test 1: Firebase Admin
print("\n1️⃣ Testing Firebase Admin...")
try:
    import firebase_admin
    print(f"   ✅ Firebase Admin imported successfully (v{firebase_admin.__version__})")
except ImportError as e:
    print(f"   ❌ Firebase Admin import failed: {e}")
    print("   💡 Fix: pip install firebase-admin")

# Test 2: TensorFlow
print("\n2️⃣ Testing TensorFlow...")
try:
    import tensorflow as tf
    print(f"   ✅ TensorFlow imported successfully (v{tf.__version__})")
except ImportError as e:
    print(f"   ❌ TensorFlow import failed: {e}")
    sys.exit(1)

# Test 3: COVID Model
print("\n3️⃣ Testing COVID Model Loading...")
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "voiceguard_audio_model_final.keras")

if not os.path.exists(MODEL_PATH):
    print(f"   ❌ Model file not found at: {MODEL_PATH}")
else:
    print(f"   ✅ Model file exists ({os.path.getsize(MODEL_PATH):,} bytes)")
    try:
        print("   📥 Loading model with compile=False...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("   ✅ Model loaded successfully!")
        print(f"   📊 Input shape: {model.input_shape}")
        print(f"   📊 Output shape: {model.output_shape}")
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        print("\n   Stack trace:")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("✅ Test complete!")
print("=" * 70)
