
import os
import sys
import traceback

# Force CPU to avoid GPU issues hanging the process
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    MODEL_PATH = "backend/models/voiceguard_audio_model_final.keras"
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)
        
    print(f"Attempting to load model from {MODEL_PATH}...")
    # Load without compiling to avoid optimizer/version issues
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
except Exception as e:
    print("❌ Model loading failed!")
    print(f"Error: {e}")
    traceback.print_exc()
