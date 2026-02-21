
import os
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def test_models():
    print("="*50)
    print("🧪 TESTING MODELS")
    print("="*50)
    
    # 1. Test Disease Model
    disease_model_path = os.path.join(MODEL_DIR, "disease_classification_model.pkl")
    print(f"\nChecking Disease Model at: {disease_model_path}")
    
    if os.path.exists(disease_model_path):
        try:
            model = joblib.load(disease_model_path)
            print("✅ Disease model loaded successfully")
            print(f"   Type: {type(model)}")
            
            # Create dummy features (23 features as per app.py)
            dummy_features = np.zeros((1, 23))
            print(f"   Testing prediction with shape {dummy_features.shape}...")
            
            try:
                pred = model.predict(dummy_features)
                print(f"   ✅ Prediction successful: {pred}")
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(dummy_features)
                    print(f"   ✅ Probabilities: {proba}")
            except Exception as e:
                print(f"   ❌ Prediction failed: {e}")
                
        except Exception as e:
            print(f"❌ Failed to load disease model: {e}")
    else:
        print("❌ Disease model file not found")

    # 2. Test Scaler
    scaler_path = os.path.join(MODEL_DIR, "disease_scaler.pkl")
    # backup name check from app.py
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(MODEL_DIR, "disease_scaler (1).pkl")
        
    print(f"\nChecking Scaler at: {scaler_path}")
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print("✅ Scaler loaded successfully")
            print(f"   Type: {type(scaler)}")
        except Exception as e:
            print(f"❌ Failed to load scaler: {e}")
    else:
        print("❌ Scaler file not found")

    # 3. Test Audio Model
    try:
        import tensorflow as tf
        audio_model_path = os.path.join(MODEL_DIR, "voiceguard_audio_model_final.keras")
        print(f"\nChecking Audio Model at: {audio_model_path}")
        
        if os.path.exists(audio_model_path):
            try:
                model = tf.keras.models.load_model(audio_model_path)
                print("✅ Audio model loaded successfully")
                print(f"   Input Shape: {model.input_shape}")
                print(f"   Output Shape: {model.output_shape}")
                model.summary()
            except Exception as e:
                print(f"❌ Failed to load audio model: {e}")
        else:
            print("❌ Audio model file not found")
    except ImportError:
        print("\n⚠️ TensorFlow not available, skipping audio model test")

if __name__ == "__main__":
    test_models()
