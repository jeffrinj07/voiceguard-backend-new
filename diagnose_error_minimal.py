
import os
import sys

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print(f"Python: {sys.version}")

try:
    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__}")
except ImportError:
    print("TensorFlow: NOT FOUND")

try:
    import keras
    print(f"Keras: {keras.__version__}")
except ImportError:
    print("Keras: NOT FOUND")

MODEL_PATH = "backend/models/voiceguard_audio_model_final.keras"

def try_load(path):
    print(f"\n--- Testing Load for {path} ---")
    
    # Try tf.keras
    try:
        print("1. Attempting tf.keras.models.load_model(compile=False)...")
        model = tf.keras.models.load_model(path, compile=False)
        print("✅ SUCCESS via tf.keras")
        return
    except Exception as e:
        print(f"❌ FAILED via tf.keras: {type(e).__name__}")
        # Print only the first 200 chars of the error to avoid the JSON dump
        print(f"   Error snippet: {str(e)[:500]}")

    # Try standalone keras
    try:
        import keras
        print("\n2. Attempting keras.models.load_model(compile=False)...")
        model = keras.models.load_model(path, compile=False)
        print("✅ SUCCESS via direct keras")
        return
    except Exception as e:
        print(f"❌ FAILED via direct keras: {type(e).__name__}")
        print(f"   Error snippet: {str(e)[:500]}")

    # Try keras.saving (Keras 3)
    try:
        import keras
        if hasattr(keras, 'saving'):
            print("\n3. Attempting keras.saving.load_model(path)...")
            model = keras.saving.load_model(path)
            print("✅ SUCCESS via keras.saving")
            return
    except Exception as e:
        print(f"❌ FAILED via keras.saving: {type(e).__name__}")
        print(f"   Error snippet: {str(e)[:500]}")

if os.path.exists(MODEL_PATH):
    try_load(MODEL_PATH)
else:
    print(f"ERROR: File not found: {MODEL_PATH}")
