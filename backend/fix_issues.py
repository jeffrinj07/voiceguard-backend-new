"""
Fix script for VoiceGuard issues:
1. Check and install Firebase Admin
2. Diagnose COVID model loading issue
"""
import subprocess
import sys
import os

print("=" * 70)
print("🔧 VOICEGUARD FIX SCRIPT")
print("=" * 70)

# Issue 1: Install Firebase Admin
print("\n📦 Checking Firebase Admin...")
try:
    import firebase_admin
    print("✅ Firebase Admin is already installed")
    print(f"   Version: {firebase_admin.__version__}")
except ImportError:
    print("❌ Firebase Admin not found")
    print("📥 Installing firebase-admin...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "firebase-admin==6.2.0"])
        print("✅ Firebase Admin installed successfully!")
    except Exception as e:
        print(f"❌ Failed to install Firebase Admin: {e}")

# Issue 2: Check COVID model
print("\n🎤 Checking COVID Audio Model...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "voiceguard_audio_model_final.keras")

print(f"Model path: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    file_size = os.path.getsize(MODEL_PATH)
    print(f"✅ Model file exists ({file_size:,} bytes)")
    
    # Try to load with TensorFlow
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        print("\n📥 Attempting to load model...")
        
        # Force CPU to avoid GPU issues
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully!")
        print(f"\nModel Info:")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        model.summary()
        
    except Exception as e:
        print(f"❌ Failed to load model")
        print(f"\nError details:")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        print("=" * 70)
        
        # Common fixes
        print("\n💡 Possible solutions:")
        print("1. Model was saved with a different TensorFlow version")
        print("2. Try loading with compile=False")
        print("3. Check for custom layers or objects")
        
else:
    print(f"❌ Model file NOT found at: {MODEL_PATH}")

print("\n" + "=" * 70)
print("🏁 Fix script complete!")
print("=" * 70)
