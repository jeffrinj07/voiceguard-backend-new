"""
ALL-IN-ONE Fix Script for VoiceGuard
- Installs Firebase Admin
- Tests COVID Model Loading
- Shows exact errors
"""
import subprocess
import sys
import os

print("=" * 70)
print("🔧 VOICEGUARD ALL-IN-ONE FIX")
print("=" * 70)

# ============================================
# 1. INSTALL FIREBASE ADMIN
# ============================================
print("\n1️⃣ Installing Firebase Admin...")
try:
    import firebase_admin
    print(f"   ✅ Already installed: v{firebase_admin.__version__}")
except ImportError:
    print("   Installing firebase-admin...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "firebase-admin", "-q"])
        import firebase_admin
        print(f"   ✅ Installed successfully: v{firebase_admin.__version__}")
    except Exception as e:
        print(f"   ❌ Installation failed: {e}")

# ============================================
# 2. TEST COVID MODEL
# ============================================
print("\n2️⃣ Testing COVID Model...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging

try:
    import tensorflow as tf
    print(f"   TensorFlow version: {tf.__version__}")
except ImportError:
    print("   ❌ TensorFlow not installed!")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "voiceguard_audio_model_final.keras")

print(f"   Model path: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print(f"   ❌ File does not exist!")
else:
    file_size = os.path.getsize(MODEL_PATH)
    print(f"   ✅ File exists: {file_size:,} bytes")
    
    print(f"   Loading with compile=False...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"   ✅ MODEL LOADED SUCCESSFULLY!")
        print(f"   📊 Input shape: {model.input_shape}")
        print(f"   📊 Output shape: {model.output_shape}")
    except Exception as e:
        print(f"   ❌ LOADING FAILED!")
        print(f"\n   Error: {e}\n")
        print("   Full traceback:")
        print("   " + "-" * 60)
        import traceback
        traceback.print_exc()
        print("   " + "-" * 60)

# ============================================
# 3. CHECK FIREBASE CREDENTIALS
# ============================================
print("\n3️⃣ Checking Firebase Credentials...")
cred_file = os.path.join(BASE_DIR, "firebase-credentials.json")
cred_file_wrong = os.path.join(BASE_DIR, "firebase-credentials.json.json")

if os.path.exists(cred_file):
    print(f"   ✅ Found: firebase-credentials.json")
elif os.path.exists(cred_file_wrong):
    print(f"   ⚠️  Found: firebase-credentials.json.json (WRONG)")
    print(f"   💡 Rename it to: firebase-credentials.json")
else:
    print(f"   ❌ No credentials file found")
    print(f"   💡 App will use dummy credentials")

print("\n" + "=" * 70)
print("✅ DIAGNOSTIC COMPLETE!")
print("=" * 70)
print("\nNEXT STEPS:")
print("1. If Firebase Admin was just installed, you're good!")
print("2. If COVID model failed, see the error above")
print("3. Restart your server: python app.py")
print("=" * 70)
