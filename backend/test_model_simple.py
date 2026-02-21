"""
Simple test to load COVID model outside of Flask
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

print("Testing COVID model loading...")
print("=" * 60)

# Import TensorFlow
try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__}")
except ImportError as e:
    print(f"❌ TensorFlow not available: {e}")
    exit(1)

# Find model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "voiceguard_audio_model_final.keras")
print(f"\nModel path: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print(f"❌ File not found!")
    exit(1)

print(f"✅ File exists: {os.path.getsize(MODEL_PATH):,} bytes")

# Try to load
print("\nAttempting to load with compile=False...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ SUCCESS! Model loaded!")
    print(f"   Input: {model.input_shape}")
    print(f"   Output: {model.output_shape}")
    print("\n✅ The model CAN be loaded!")
    print("   The issue must be elsewhere in app.py")
except Exception as e:
    print(f"❌ FAILED: {e}")
    print("\nFull error:")
    import traceback
    traceback.print_exc()
    
print("=" * 60)
