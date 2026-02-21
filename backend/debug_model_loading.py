
import os
import tensorflow as tf
import sys

# Force CPU to avoid CUDA errors hiding the real issue
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "voiceguard_audio_model_final.keras")

print(f"Python Version: {sys.version}")
print(f"TensorFlow Version: {tf.__version__}")
print(f"Model Path: {MODEL_PATH}")

if os.path.exists(MODEL_PATH):
    print(f"File size: {os.path.getsize(MODEL_PATH)} bytes")
    try:
        print("Attempting to load model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
        model.summary()
    except Exception as e:
        print("\n❌ FAILED TO LOAD MODEL")
        print("="*50)
        import traceback
        traceback.print_exc()
        print("="*50)
else:
    print("❌ Model file does not exist!")
