
import os
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_model")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
covid_model_path = os.path.join(MODEL_DIR, "voiceguard_audio_model_final.keras")

print(f"Checking path: {covid_model_path}")
print(f"Exists: {os.path.exists(covid_model_path)}")

if os.path.exists(covid_model_path):
    try:
        print("Attempting to load via tf.keras...")
        model = tf.keras.models.load_model(covid_model_path, compile=False)
        print("✅ Success via tf.keras!")
    except Exception as keras_err:
        print(f"❌ tf.keras failed: {str(keras_err)}")
        if "deserialization" in str(keras_err).lower() or "functional" in str(keras_err).lower():
            print("Trying direct keras import fallback...")
            try:
                import keras
                print(f"Keras version: {keras.__version__}")
                model = keras.models.load_model(covid_model_path, compile=False)
                print("✅ Success via direct keras!")
            except Exception as e2:
                print(f"❌ Both failed. Second error: {str(e2)}")
        else:
            print("Error doesn't match fallback criteria.")
else:
    print("Model file not found!")
