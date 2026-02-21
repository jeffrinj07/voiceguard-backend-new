import tensorflow as tf
import os
import sys

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "voiceguard_audio_model_final.keras")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "voiceguard_audio_model_final.tflite")

def convert_model():
    print(f"🚀 Starting conversion: {KERAS_MODEL_PATH}")
    
    if not os.path.exists(KERAS_MODEL_PATH):
        print(f"❌ Error: Keras model not found at {KERAS_MODEL_PATH}")
        return False

    try:
        # 1. Load the Keras model
        print("📥 Loading Keras model...")
        model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)
        print("✅ Keras model loaded successfully.")

        # 2. Convert to TFLite
        print("🔄 Converting to TFLite format...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optional: Optimization (can further reduce size but might slightly affect accuracy)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        print("✅ Conversion successful.")

        # 3. Save the TFLite model
        print(f"💾 Saving TFLite model to: {TFLITE_MODEL_PATH}")
        with open(TFLITE_MODEL_PATH, 'wb') as f:
            f.write(tflite_model)
        
        keras_size = os.path.getsize(KERAS_MODEL_PATH) / (1024 * 1024)
        tflite_size = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024)
        
        print(f"📊 Summary:")
        print(f"   • Keras size: {keras_size:.2f} MB")
        print(f"   • TFLite size: {tflite_size:.2f} MB")
        print(f"   • Reduction: {((keras_size - tflite_size) / keras_size) * 100:.1f}%")
        
        return True

    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = convert_model()
    if not success:
        sys.exit(1)
